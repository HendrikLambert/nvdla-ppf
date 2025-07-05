#include "benchmark.hpp"
#include <iostream>

using namespace std;

void printUsage(string programName) {
    cout << "Usage:" << endl;
    cout << "  " << programName << " benchmark <directory> <results.csv>" << endl;
}

int main(int argc, char* argv[]) {
    // Initialize the benchmark
    if (argc != 4) {
        printUsage(argv[0]);
        return -1;
    }

    // Init the benchmark

    Benchmark* benchmark = new Benchmark("/dev/ttyACM0", string(argv[3]));
    if (!benchmark->init()) {
        cerr << "Failed to initialize benchmark." << endl;
        delete benchmark;
        return -1;
    }

    // Check if first arg equals benchmark
    if (string(argv[1]) == "benchmark") {
        cout << "Running benchmark... with " << argv[2] << endl;
        // Load files from the specified directory
        // string dir = "/var/scratch/dsl2511/loadables";
        benchmark->load_files(string(argv[2]));
        cout << "Benchmark initialized and files loaded successfully." << endl;
        benchmark->run();
    } else {
        printUsage(argv[0]);
    }

    delete benchmark;
    return 0;
}