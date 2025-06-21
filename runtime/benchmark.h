#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <string>
#include <vector>

#include "loadable.h"
#include "cudla_dev.h"
#include "cudla_runtime.h"

class Benchmark {

public:

    Benchmark() noexcept;
    ~Benchmark() noexcept;

    bool init() noexcept;

    void load_files(std::string const& dir) noexcept;


private:
    std::vector<std::string> files;
    CUDLARuntime runtime;
};

#endif