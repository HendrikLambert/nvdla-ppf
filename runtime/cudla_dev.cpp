#include "cudla_dev.h"
#include <iostream>

using namespace std;

bool CUDLADev::initialize(const int deviceId) noexcept {
    cudlaStatus status = cudlaCreateDevice(deviceId, &devHandle, CUDLA_CUDA_DLA);
    if (status != cudlaSuccess) {
        cout << "Error initializing CUDLA device: " << status << std::endl;
        return false;
    }
    return true;
}

CUDLADev::~CUDLADev() noexcept {
    if (devHandle) {
        cudlaStatus status = cudlaDestroyDevice(devHandle);
        if (status != cudlaSuccess) {
            cout << "Error destroying CUDLA device: " << status << std::endl;
        }
    }
}