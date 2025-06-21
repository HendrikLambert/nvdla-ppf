#include "cudla_runtime.h"

#include <iostream>
#include "cuda_runtime.h"
#include "cudla.h"

using namespace std;

CUDLARuntime::CUDLARuntime() noexcept {
    
}


CUDLARuntime::~CUDLARuntime() noexcept {
    if (stream) {
        cudaError_t cudaStatus = cudaStreamDestroy(stream);
        if (cudaStatus != cudaSuccess) {
            const char* errPtr = cudaGetErrorName(cudaStatus);
            cout << "Error: destroying stream: " << errPtr << endl;
        }
    }
}

bool CUDLARuntime::initialize(const int deviceId) noexcept {
    // Initialize CUDA
    cudaError_t cudaStatus = cudaFree(0);
    if (cudaStatus != cudaSuccess) {
        const char* errPtr = cudaGetErrorName(cudaStatus);
        cout << "Error: creating cudaFree: " << errPtr << endl;
        return false;
    }

    // Set the CUDA device
    cudaStatus = cudaSetDevice(deviceId);
    if (cudaStatus != cudaSuccess) {
        const char* errPtr = cudaGetErrorName(cudaStatus);
        cout << "Error: creating cudaSetDevice: " << errPtr << endl;
        return false;
    }

    cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (cudaStatus != cudaSuccess) {
        const char* errPtr = cudaGetErrorName(cudaStatus);
        cout << "Error in creating cuda stream: " << errPtr << endl;
        return false;
    }

    return true;
}
