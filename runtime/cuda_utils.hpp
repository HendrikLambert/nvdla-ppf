#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

// Modified from: https://github.com/NVIDIA-AI-IOT/cuDLA-samples/blob/main/src/cudla_context_standalone.cpp
#define CHECK_CUDA_ERR(err, msg)                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        if (err != cudaSuccess)                                                                                        \
        {                                                                                                              \
            std::cerr << msg << " FAILED in " << __FILE__ << ":" << __LINE__                                           \
                      << ", CUDA ERR: " << static_cast<int>(err) << " (" << cudaGetErrorName(err) << ")\n";            \
        }                                                                                                              \
    } while (0)
