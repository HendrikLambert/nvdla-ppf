#include "cudla_runtime.hpp"

#include <iostream>
#include "cuda_runtime.h"
#include "cudla.h"

using namespace std;

std::shared_ptr<CUDLARuntime> CUDLARuntime::create() noexcept {
    auto obj = std::shared_ptr<CUDLARuntime>(new CUDLARuntime());

    cudaError_t cudaStatus = cudaStreamCreateWithFlags(&obj->stream, cudaStreamNonBlocking);
    if (cudaStatus != cudaSuccess) {
        const char* errPtr = cudaGetErrorName(cudaStatus);
        cout << "Error in creating cuda stream: " << errPtr << endl;
        return nullptr;
    }

    return obj;
}

CUDLARuntime::CUDLARuntime() noexcept {

}

CUDLARuntime::~CUDLARuntime() noexcept {
    // Destroy CUDA stream
    if (stream) {
        cudaError_t cudaStatus = cudaStreamDestroy(stream);
        if (cudaStatus != cudaSuccess) {
            const char* errPtr = cudaGetErrorName(cudaStatus);
            cout << "Error: destroying stream: " << errPtr << endl;
        }
    }

}

bool CUDLARuntime::initializeCuda(const int deviceId) noexcept {
    // Initialize CUDA

    // Set the CUDA device
    cudaError_t cudaStatus = cudaSetDevice(deviceId);
    if (cudaStatus != cudaSuccess) {
        const char* errPtr = cudaGetErrorName(cudaStatus);
        cout << "Error: creating cudaSetDevice: " << errPtr << endl;
        return false;
    }

    cudaStatus = cudaFree(0);
    if (cudaStatus != cudaSuccess) {
        const char* errPtr = cudaGetErrorName(cudaStatus);
        cout << "Error: creating cudaFree: " << errPtr << endl;
        return false;
    }

    return true;
}

cudaStream_t CUDLARuntime::getStream() const noexcept {
    return stream;
}