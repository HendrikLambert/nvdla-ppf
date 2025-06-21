#include "benchmark.h"
#include <iostream>
#include <filesystem>

using namespace std;

Benchmark::Benchmark() noexcept {

}

Benchmark::~Benchmark() noexcept {

}

bool Benchmark::init() noexcept {
    if (!runtime.initialize(0)) {
        cerr << "Failed to initialize CUDLARuntime." << endl;
        return false;
    }
    return true;
}

void Benchmark::load_files(const string& dir) noexcept {
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

    for (const auto& file : files) {
        cout << "Loaded file: " << file << endl;
    }

}

