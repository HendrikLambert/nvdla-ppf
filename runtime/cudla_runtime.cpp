#include "cudla_runtime.hpp"

#include <iostream>
#include "cuda_runtime.h"
#include "cudla.h"

using namespace std;

std::shared_ptr<CUDLARuntime> CUDLARuntime::create(const int deviceId) noexcept {
    auto obj = std::shared_ptr<CUDLARuntime>(new CUDLARuntime());

    cudaError_t cudaStatus = cudaStreamCreateWithFlags(&obj->stream, cudaStreamNonBlocking);
    if (cudaStatus != cudaSuccess) {
        const char* errPtr = cudaGetErrorName(cudaStatus);
        cout << "Error in creating cuda stream: " << errPtr << endl;
        return nullptr;
    }
    cout << "devHandle: " << obj->devHandle << endl;
    // Initialize CUDLA device
    cudlaStatus status = cudlaCreateDevice(deviceId, &obj->devHandle, CUDLA_CUDA_DLA);
    if (status != cudlaSuccess) {
        cout << "cudlaCreateDevice failed for device: " << deviceId << " with error: " << status << endl;
        return nullptr;
    }
    cout << "devHandle: " << obj->devHandle << endl;

    return obj;
}

CUDLARuntime::CUDLARuntime() noexcept {

}


CUDLARuntime::~CUDLARuntime() noexcept {
    cout << "CUDLARuntime destructor called." << endl;
    cout << "runtime devHandle: " << devHandle << endl;
    // Destroy CUDA stream
    if (stream) {
        cudaError_t cudaStatus = cudaStreamDestroy(stream);
        if (cudaStatus != cudaSuccess) {
            const char* errPtr = cudaGetErrorName(cudaStatus);
            cout << "Error: destroying stream: " << errPtr << endl;
        }
    }

    // Destroy CUDLA device
    if (devHandle) {
        cudlaStatus status = cudlaDestroyDevice(devHandle);
        if (status != cudlaSuccess) {
            cout << "cudlaDestroyDevice failed: " << status << endl;
        }
    }

}

bool CUDLARuntime::initializeCuda(const int deviceId) noexcept {
    cout << "Initializing CUDA for device ID: " << deviceId << endl;
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

cudlaDevHandle CUDLARuntime::getDevice() const noexcept {
    return devHandle;
}

cudaStream_t CUDLARuntime::getStream() const noexcept {
    return stream;
}