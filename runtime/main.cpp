#include "benchmark.hpp"
#include <iostream>

using namespace std;

int main() {
    // Initialize the benchmark
    Benchmark* benchmark = new Benchmark("/dev/ttyACM0", "benchmark_results.csv");
    if (!benchmark->init()) {
        cerr << "Failed to initialize benchmark." << endl;
        return -1;
    }

    // Load files from the specified directory
    string dir = "/var/scratch/dsl2511/loadables";
    benchmark->load_files(dir);

    cout << "Benchmark initialized and files loaded successfully." << endl;

    benchmark->run();

    delete benchmark;
    return 0;
}