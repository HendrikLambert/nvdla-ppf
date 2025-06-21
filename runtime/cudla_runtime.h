#ifndef CUDLA_MANAGER_H
#define CUDLA_MANAGER_H

#include "cuda_runtime.h"
#include "cuda.h"

class CUDLARuntime {

public:
    /**
     * Constructor for the CUDLARuntime class.
     */
    CUDLARuntime() noexcept;

    ~CUDLARuntime() noexcept;

    bool initialize(const int deviceId = 0) noexcept;
    
    
    // Disable copy semantics
    CUDLARuntime(const CUDLARuntime&) = delete;
    CUDLARuntime& operator=(const CUDLARuntime&) = delete;

    // Disable move semantics
    CUDLARuntime(CUDLARuntime&&) = delete;
    CUDLARuntime& operator=(CUDLARuntime&&) = delete;

private:
    cudaStream_t stream = nullptr;
};

#endif