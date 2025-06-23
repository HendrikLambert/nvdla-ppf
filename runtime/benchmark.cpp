#include "benchmark.hpp"
#include <iostream>
#include <filesystem>
#include <chrono>
#include <thread>
#include "cuda_utils.hpp"

using namespace std;

Benchmark::Benchmark(const string powerSensor, const std::string csvFileName)
    : ps3(powerSensor) {
    csvFile.open(csvFileName, std::ios::out);
}

Benchmark::~Benchmark() noexcept {
    // Runtimes are cleaned up automatically.
    // Loadables also.
}

bool Benchmark::init() {
    // Write header to csv file.
    csvFile << "loadable,dla,buffers,runs,samples,energy,host_time,device_time" << endl;

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
            if (entry.is_regular_file()) {
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

    // Calculate the number of samples per run
    size_t samplesPerRun = buffers * n;

    return make_tuple(buffers, runs, samplesPerRun);
}

void Benchmark::run() {
    cout << "Running benchmark..." << endl;

    // For each run
    for (size_t r = 0; r < RUNS; r++) {
        // For each run, we will run the files on both DLA engines

        for (const auto& file : files) {

            // On all runtimes
            for (int i = 0; i < RUNTIMES; i++) {
                cout << "Running benchmark for file: " << file << " on dla: " << i << " run " << (r+1) << "/" << RUNS << endl;
                run_single_dla(file, i);
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

    // Delay to ensure the buffer allocation isnt considered in the power draw.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

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
    float devicetime;
    CHECK_CUDA_ERR(cudaEventElapsedTime(&devicetime, startEvent, stopEvent), "Failed to calculate elapsed time");
    auto energyUsed = PowerSensor3::Joules(start, stop);
    auto hosttime = PowerSensor3::seconds(start, stop) * 1000.0; // Convert to milliseconds

    // Write results to CSV file
    // "loadable,dla,buffers,runs,samples,energy,host_time,device_time"
    csvFile << file << "," << dla << "," << buffers << "," << runs << ","
        << samples << "," << energyUsed << "," << hosttime << "," << devicetime << endl;

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