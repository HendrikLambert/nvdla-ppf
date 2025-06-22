#include "benchmark.hpp"
#include <iostream>
#include <filesystem>

using namespace std;

Benchmark::Benchmark(const string powerSensor) 
    : ps3(powerSensor) {
}

Benchmark::~Benchmark() noexcept {

}

bool Benchmark::init() {
    ps3.dump("benchmark_dump.txt");

    if (!CUDLARuntime::initializeCuda(0)) {
        cerr << "Failed to initialize CUDLARuntime." << endl;
        return false;
    }

    // Create the runtimes.
    for (int i = 0; i < RUNTIMES; i++) {
        auto runtime = CUDLARuntime::create(i);
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

void Benchmark::run() {
    for (const auto& file : files) {
        run_single(file);
        return; // For now, run only the first file
    }

    // Stop the dump
    ps3.dump("");
}

void Benchmark::run_single(const string& file) {
    cout << "Running benchmark for file: " << file << endl;

    auto runtime = runtimes[0]; // Use the first runtime for now

    auto loadable = Loadable::create(file, runtime->getDevice());
    if (!loadable) {
        cerr << "Failed to create Loadable from file: " << file << endl;
        return;
    }

    loadable->allocateBuffers(1);

    cudaStream_t stream = runtime->getStream();
    
    // Insert blocking event to ensure no wait occurs in the stream

    // Insert the tasks
    ps3.mark('A');
    for (int i = 0; i < 1; i++) {
        // if (!loadable->runTask(stream, i)) {
        //     cerr << "Failed to run task for buffer index: " << i << endl;
        //     return;
        // }
    }
    cudaError_t cudaStatus = cudaStreamSynchronize(stream);
    if (cudaStatus != cudaSuccess) {
        const char* errPtr = cudaGetErrorName(cudaStatus);
        cout << "Error: synchronizing stream = " << errPtr << endl;
        return;
    }
    ps3.mark('B');




    // loadable->printTensorDesc();
}


