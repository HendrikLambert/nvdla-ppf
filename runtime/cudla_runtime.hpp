#pragma once

#include "cuda_runtime.h"
#include "cuda.h"
#include "cudla.h"
#include <vector>
#include <memory>

class CUDLARuntime {

public:
    
    ~CUDLARuntime() noexcept;

    /**
     * Initialize CUDA for the specified device.
     * Needs to be called before creating a CUDLARuntime instance.
     * 
     * @param deviceId The ID of the CUDA device to initialize (default is 0).
     */
    static bool initializeCuda(const int deviceId = 0) noexcept;

    /**
     * Create a CUDLARuntime instance for the specified DLA device.
     * 
     * @param dlaId The ID of the DLA device to create the runtime for.
     * @return A shared pointer to the created CUDLARuntime instance, or nullptr on failure.
     */
    static std::shared_ptr<CUDLARuntime> create(const int dlaId = 0) noexcept;
    
    /**
     * Get the NVDLA device handle for the current runtime.
     * 
     * @return cudlaDevHandle The handle to the CUDLA device.
     */
    cudlaDevHandle getDevice() const noexcept;

    /**
     * Get the CUDA stream associated with this runtime.
     * 
     * @return cudaStream_t The CUDA stream.
     */
    cudaStream_t getStream() const noexcept;
    
    
    // Disable copy semantics
    CUDLARuntime(const CUDLARuntime&) = delete;
    CUDLARuntime& operator=(const CUDLARuntime&) = delete;

    // Disable move semantics
    CUDLARuntime(CUDLARuntime&&) = delete;
    CUDLARuntime& operator=(CUDLARuntime&&) = delete;

private:
    cudaStream_t stream = nullptr;
    cudlaDevHandle devHandle = nullptr;

     /**
     * Private constructor for the CUDLARuntime class.
     */
    explicit CUDLARuntime() noexcept;
};
