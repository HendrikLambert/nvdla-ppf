#pragma once

#include <string>
#include <vector>
#include <memory>
#include <tuple>
#include <fstream>
#include <iostream>

#include "PowerSensor.hpp"
#include "loadable.hpp"
#include "cudla_runtime.hpp"
#include "cudla.h"

#define RUNTIMES 2
#define MIN_BUFFERS 8
// Every batch contains 256 complex samples (and the 15 other historical taps)
#define BATCHES_TO_RUN 10'000
// Cache size in bytes. Consider every loction where the samples can be cached.
#define CACHE_SIZE 4'000'000
// How often we loop through all the files and execute them
#define RUNS 20


class Benchmark {

public:
    Benchmark(const std::string powerSensor, const std::string csvFileName);
    ~Benchmark() noexcept;

    bool init();

    void load_files(std::string const& dir);

    void run();

    void run_single_dla(const std::string& file, const int dla = 0);

private:
    std::ofstream csvFile;
    PowerSensor3::PowerSensor ps3;
    PowerSensor3::State start, stop;

    std::vector<std::string> files;
    std::vector<std::shared_ptr<CUDLARuntime>> runtimes;

    std::tuple<size_t, size_t, size_t> calculateBuffersAndRuns(std::tuple<int, int, int, int> inputShape, int dlaCount) const noexcept;

    static void CUDART_CB hostCallbackStart(void* instance);
    static void CUDART_CB hostCallbackStop(void* instance);
};
