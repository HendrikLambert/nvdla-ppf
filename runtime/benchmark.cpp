#include "benchmark.hpp"
#include <iostream>
#include <filesystem>
#include <chrono>
#include <thread>
#include "cuda_utils.hpp"

using namespace std;

Benchmark::Benchmark(const string powerSensor, const std::string csvFileName)
    : ps3(powerSensor) {
    cout << "Initializing Benchmark with PowerSensor: " << powerSensor << endl;
    cout << "CSV file: " << csvFileName << endl;
    csvFile.open(csvFileName, std::ios::out);
}

Benchmark::~Benchmark() noexcept {
    // Runtimes are cleaned up automatically.
    // Loadables also.
}

bool Benchmark::init() {
    // Write header to csv file.
    csvFile << "loadable,dla,buffers,runs,samples,host_time,device_time,total_energy,base_energy" << endl;

    if (!CUDLARuntime::initializeCuda(0)) {
        cerr << "Failed to initialize CUDLARuntime." << endl;
        return false;
    }

    // Create the runtimes.
    for (int i = 0; i < RUNTIMES; i++) {
        auto runtime = CUDLARuntime::create();
        if (!runtime) {
            return false;
        }
        runtimes.push_back(runtime);
    }

    return true;
}

void Benchmark::load_files(const string& dir) {
    cout << "Loading files from directory: " << dir << endl;

    try {
        for (const auto& entry : filesystem::directory_iterator(dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".nvdla") {
                string filePath = entry.path().string();
                files.push_back(filePath);
            }
        }
    }
    catch (const filesystem::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }
}

tuple<size_t, size_t, size_t> Benchmark::calculateBuffersAndRuns(std::tuple<int, int, int, int> inputShape, int dlaCount) const noexcept {
    // We want to ensure that the cache/sram does not cache the data, so we can only reuse the buffers after a certain number of runs. 
    auto [n, c, h, w] = inputShape;
    size_t buffers = CACHE_SIZE / (n * c * h * w * 2); // Number of buffers we need to allocate
    // Ensure we have at least MIN_BUFFERS buffers
    if (buffers < MIN_BUFFERS) {
        buffers = MIN_BUFFERS;
    }
    
    // We want to compute BATCHES_TO_RUN samples.
    size_t batchesPerBuffer = buffers * n;
    size_t runs = BATCHES_TO_RUN / batchesPerBuffer / dlaCount; // Number of runs we can do with the allocated buffers
    // Ensure we have at least 1 run
    if (runs == 0) {
        runs = 1;
    }

    // Calculate the number of samples per run
    size_t samplesPerRun = buffers * n;

    return make_tuple(buffers, runs, samplesPerRun);
}

void Benchmark::run() {
    cout << "Running benchmark..." << endl;

    // For each iteration
    for (size_t i = 0; i < ITERATIONS; i++) {
        // For each iteration, we will run the files on both DLA engines

        for (const auto& file : files) {

            // On all runtimes
            for (int r = 0; r < RUNTIMES; r++) {
                cout << "Running benchmark for file: " << file << " on dla: " << r << " iteration " << (i+1) << "/" << ITERATIONS << endl;
                run_single_dla(file, r);
            }
        }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    // Stop the dump
}

void Benchmark::run_single_dla(const string& file, const int dla) {
    cudaError_t cudaStatus;
    auto runtime = runtimes[dla];

    // Create the loadable.
    auto loadable = Loadable::create(file, dla);
    if (!loadable) {
        cerr << "Failed to create Loadable from file: " << file << endl;
        return;
    }

    // Calculate the amount of buffers required and the number of runs we can do with the buffers
    auto [buffers, runs, samplesPerRun] = calculateBuffersAndRuns(loadable->getInputTensorShape(0), 1);

    // Allocate the buffers
    loadable->allocateBuffers(buffers);

    // Delay to ensure the buffer allocation isnt considered in the base power draw.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    // Start base power measurement
    base_start = ps3.read();
    // Wait for a short time to gather more samples.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    base_stop = ps3.read();

    cudaStream_t stream = runtime->getStream();
    cudaEvent_t startEvent, stopEvent;

    CHECK_CUDA_ERR(cudaEventCreate(&startEvent), "Failed to record start event");

    CHECK_CUDA_ERR(cudaEventCreate(&stopEvent), "Failed to create stop event");


    // Warm up the device
    for (size_t b = 0; b < buffers; b++) {
        if (!loadable->runTask(stream, b)) {
            cout << "Failed to run task for buffer index: " << b << endl;
            return;
        }
    }

    // Synchronize the stream to ensure all warm-up tasks are completed
    // CHECK_CUDA_ERR(cudaStreamSynchronize(stream), "Failed to synchronize stream after warm-up");

    // Measure start power
    CHECK_CUDA_ERR(cudaLaunchHostFunc(stream, &Benchmark::hostCallbackStart, this), "Failed to launch host function for start callback");

    // Insert start timer
    CHECK_CUDA_ERR(cudaEventRecord(startEvent, stream), "Failed to record start event");

    // Loop runs and then buffers
    for (size_t r = 0; r < runs; r++) {
        // For each run, we need to run the task for each buffer
        for (size_t b = 0; b < buffers; b++) {
            if (!loadable->runTask(stream, b)) {
                cerr << "Failed to run task for buffer index: " << b << endl;
                return;
            }
        }
    }

    // Insert stop timer
    CHECK_CUDA_ERR(cudaEventRecord(stopEvent, stream), "Failed to record stop event");


    // Measure stop power
    CHECK_CUDA_ERR(cudaLaunchHostFunc(stream, &Benchmark::hostCallbackStop, this), "Failed to launch host function for stop callback");

    // Wait for the stream to finish
    CHECK_CUDA_ERR(cudaStreamSynchronize(stream), "Failed to synchronize stream after runs");

    // Calculate statistics
    size_t samples = runs * samplesPerRun * 256; // 256 complex samples per buffer
    float deviceTime;
    CHECK_CUDA_ERR(cudaEventElapsedTime(&deviceTime, startEvent, stopEvent), "Failed to calculate elapsed time");
    auto totalEnergy = PowerSensor3::Joules(start, stop);
    auto hostTime = PowerSensor3::seconds(start, stop) * 1000.0; // Convert to milliseconds

    // Calculate base energy
    auto baseDraw = PowerSensor3::Watt(base_start, base_stop);
    auto baseEnergy = baseDraw * hostTime / 1000.0; // Convert to Joules

    // Write results to CSV file
    // "loadable,dla,buffers,runs,samples,host_time,device_time,total_energy,base_energy" << endl;
    csvFile << file << "," << dla << "," << buffers << "," << runs << ","
        << samples << "," << hostTime << "," << deviceTime << ","
        << totalEnergy << "," << baseEnergy << endl;

    // Clean up events
    cudaStatus = cudaEventDestroy(startEvent);
    if (cudaStatus != cudaSuccess) {
        const char* errPtr = cudaGetErrorName(cudaStatus);
        cout << "Error: destroying start event = " << errPtr << endl;
        return;
    }

    cudaStatus = cudaEventDestroy(stopEvent);
    if (cudaStatus != cudaSuccess) {
        const char* errPtr = cudaGetErrorName(cudaStatus);
        cout << "Error: destroying stop event = " << errPtr << endl;
        return;
    }

}


void CUDART_CB Benchmark::hostCallbackStart(void* instance) {
    Benchmark* benchmark = static_cast<Benchmark*>(instance);
    benchmark->start = benchmark->ps3.read();
}

void CUDART_CB Benchmark::hostCallbackStop(void* instance) {
    Benchmark* benchmark = static_cast<Benchmark*>(instance);
    benchmark->stop = benchmark->ps3.read();
}