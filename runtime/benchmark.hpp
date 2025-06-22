#pragma once

#include <string>
#include <vector>
#include <memory>

#include "PowerSensor.hpp"
#include "loadable.hpp"
#include "cudla_runtime.hpp"
#include "cudla.h"

#define RUNTIMES 1

class Benchmark {

public:

    Benchmark(const std::string powerSensor);
    ~Benchmark() noexcept;

    bool init();

    void load_files(std::string const& dir);

    void run();

    void run_single(const std::string& file);


private:
    PowerSensor3::PowerSensor ps3;
    std::vector<std::string> files;
    std::vector<std::shared_ptr<CUDLARuntime>> runtimes;
};
