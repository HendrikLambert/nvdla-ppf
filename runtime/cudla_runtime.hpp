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
     * Create a CUDLARuntime instance
     *
     * @return A shared pointer to the created CUDLARuntime instance, or nullptr on failure.
     */
    static std::shared_ptr<CUDLARuntime> create() noexcept;

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

    /**
    * Private constructor for the CUDLARuntime class.
    */
    explicit CUDLARuntime() noexcept;
};
