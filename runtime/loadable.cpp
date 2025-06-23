#include "loadable.hpp"
#include <fstream>
#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"

using namespace std;


std::unique_ptr<Loadable> Loadable::create(const std::string& filename, const uint64_t dlaId) {
    auto obj = std::unique_ptr<Loadable>(new Loadable(filename));

    if (!obj->loadBinary()) {
        return nullptr;
    }

    if (!obj->createDevHandle(dlaId)) {
        return nullptr;
    }

    if (!obj->loadModule()) {
        return nullptr;
    }

    if (!obj->loadModuleAttributes()) {
        return nullptr;
    }

    return obj;
}

Loadable::Loadable(const string& filename) noexcept
    : filename(filename) {
}

Loadable::~Loadable() noexcept {
    // First, free the buffers
    freeBuffers();

    // Unload the module if it was loaded
    if (moduleHandle) {
        cudlaStatus status = cudlaModuleUnload(moduleHandle, 0);
        if (status != cudlaSuccess) {
            cout << "cudlaModuleUnload failed: " << status << endl;
        }
    }

    // Free the device handle if it was created
    if (devHandle) {
        cudlaStatus status = cudlaDestroyDevice(devHandle);
        if (status != cudlaSuccess) {
            cout << "cudlaDestroyDevice failed: " << status << endl;
        }
    }

}

bool Loadable::loadBinary() {
    ifstream file(filename, ios::binary | ios::ate);
    if (!file) {
        cout << "Unable to open file " << filename << endl;
        return false;
    }

    fileSize = file.tellg();
    file.seekg(0, ios::beg);
    loadableData = make_unique<char[]>(fileSize);

    if (!file.read(loadableData.get(), fileSize)) {
        cout << "Unable to read file " << filename << endl;
        return false;
    }

    return true;
}

bool Loadable::loadModule() noexcept {
    cudlaStatus status = cudlaModuleLoadFromMemory(
        devHandle,
        reinterpret_cast<const uint8_t*>(loadableData.get()),
        fileSize,
        &moduleHandle,
        0);

    if (status != cudlaSuccess) {
        cout << "cudlaModuleLoadFromMemory failed: " << status << endl;
        return false;
    }

    return true;
}

bool Loadable::loadModuleAttributes() noexcept {
    // Get module attributes
    cudlaModuleAttribute attribute;

    // Get number of input tensors
    cudlaStatus status = cudlaModuleGetAttributes(moduleHandle, CUDLA_NUM_INPUT_TENSORS, &attribute);
    if (status != cudlaSuccess) {
        cout << "cudlaModuleGetAttributes for input tensors failed: " << status << endl;
        return false;
    }
    if (attribute.numInputTensors != 1) {
        cout << "Currently only support 1 input tensor, got: " << attribute.numInputTensors << " might not work as expected" << endl;
    }
    inputTensorDesc.resize(attribute.numInputTensors);

    // Get number of output tensors
    status = cudlaModuleGetAttributes(moduleHandle, CUDLA_NUM_OUTPUT_TENSORS, &attribute);
    if (status != cudlaSuccess) {
        cout << "cudlaModuleGetAttributes for output tensors failed: " << status << endl;
        return false;
    }
    if (attribute.numOutputTensors != 1) {
        cout << "Currently only support 1 output tensor, got: " << attribute.numOutputTensors << " might not work as expected" << endl;
    }
    outputTensorDesc.resize(attribute.numOutputTensors);

    // Get input tensor descriptors
    attribute.inputTensorDesc = inputTensorDesc.data();
    status = cudlaModuleGetAttributes(moduleHandle, CUDLA_INPUT_TENSOR_DESCRIPTORS, &attribute);
    if (status != cudlaSuccess) {
        cout << "cudlaModuleGetAttributes for input tensor descriptors failed: " << status << endl;
        return false;
    }

    // Get output tensor descriptors
    attribute.outputTensorDesc = outputTensorDesc.data();
    status = cudlaModuleGetAttributes(moduleHandle, CUDLA_OUTPUT_TENSOR_DESCRIPTORS, &attribute);
    if (status != cudlaSuccess) {
        cout << "cudlaModuleGetAttributes for output tensor descriptors failed: " << status << endl;
        return false;
    }

    return true;
}

bool Loadable::createDevHandle(const uint64_t devId) noexcept {
    // Create a device handle for the DLA
    cudlaStatus status = cudlaCreateDevice(devId, &devHandle, CUDLA_CUDA_DLA);
    if (status != cudlaSuccess) {
        cout << "cudlaCreateDevice failed: " << status << endl;
        return false;
    }
    return true;
}

bool Loadable::allocateBuffers(int count) noexcept {
    // Free any previously allocated buffers
    freeBuffers();

    // Allocate the buffers 
    for (int n = 0; n < count; n++) {
        // For input tensors
        for (size_t i = 0; i < inputTensorDesc.size(); i++) {
            uint64_t size = inputTensorDesc[i].size;
            if (!allocateSingleBuffer(size, bufferGPUInput, bufferDLAInput)) {
                cout << "Failed to allocate input buffer for tensor " << inputTensorDesc[i].name << endl;
                return false;
            }
        }
        // For output tensors
        for (size_t i = 0; i < outputTensorDesc.size(); i++) {
            uint64_t size = outputTensorDesc[i].size;
            if (!allocateSingleBuffer(size, bufferGPUOutput, bufferDLAOutput)) {
                cout << "Failed to allocate output buffer for tensor " << outputTensorDesc[i].name << endl;
                return false;
            }
        }
    }
    return true;
}

bool Loadable::allocateSingleBuffer(uint64_t size, std::vector<void*>& gpuBuffer, std::vector<uint64_t*>& dlaBuffer) const noexcept {
    void* gpuPtr = nullptr;
    uint64_t* dlaPtr = nullptr;

    // Allocate GPU buffer
    cudaError_t cudaStatus = cudaMalloc(&gpuPtr, size);
    if (cudaStatus != cudaSuccess) {
        const char* errPtr = cudaGetErrorName(cudaStatus);
        cout << "Error allocating GPU buffer: " << errPtr << endl;
        return false;
    }

    // Register DLA buffer
    cudlaStatus cudlaStatus = cudlaMemRegister(devHandle, reinterpret_cast<uint64_t*>(gpuPtr), size, &dlaPtr, 0);
    if (cudlaStatus != cudlaSuccess) {
        cout << "cudlaMemRegister failed: " << cudlaStatus << endl;
        cudaFree(gpuPtr);
        return false;
    }
    gpuBuffer.push_back(gpuPtr);
    dlaBuffer.push_back(dlaPtr);

    return true;
}

void Loadable::printTensorDesc() const noexcept {
    for (const auto& desc : inputTensorDesc) {
        printTensorDescHelper(desc);
    }
    for (const auto& desc : outputTensorDesc) {
        printTensorDescHelper(desc);
    }
}

bool Loadable::runTask(const cudaStream_t stream, const int buffIndex) const noexcept {
    // Craete the task
    cudlaTask task;
    task.moduleHandle = moduleHandle;
    task.numInputTensors = 1;
    task.inputTensor = &bufferDLAInput[buffIndex];
    task.numOutputTensors = 1;
    task.outputTensor = &bufferDLAOutput[buffIndex];
    task.waitEvents = NULL;
    task.signalEvents = NULL;

    cudlaStatus status = cudlaSubmitTask(devHandle, &task, 1, stream, 0);
    if (status != cudlaSuccess) {
        cout << "cudlaSubmitTask failed: " << status << endl;
        return false;
    }

    return true;
}

std::tuple<int, int, int, int> Loadable::getInputTensorShape(int index) const noexcept {
    if (index < 0 || index >= static_cast<int>(inputTensorDesc.size())) {
        cout << "Index out of bounds for input tensor descriptors." << endl;
        return std::make_tuple(-1, -1, -1, -1);
    }
    const auto& desc = inputTensorDesc[index];
    return std::make_tuple(desc.n, desc.c, desc.h, desc.w);
}

void Loadable::printTensorDescHelper(const cudlaModuleTensorDescriptor& tensorDesc) const noexcept {
    cout << "Tensor desc: \n";
    cout << "Name: " << tensorDesc.name << "\n";
    cout << "Size: " << tensorDesc.size << "\n";
    cout << "N: " << tensorDesc.n << " C: " << tensorDesc.c << " H: " << tensorDesc.h << " W: " << tensorDesc.w << "\n";
    cout << "Data format: " << (int)tensorDesc.dataFormat << "\n";
    cout << "Data type: " << (int)tensorDesc.dataType << "\n";
    cout << "Data category: " << (int)tensorDesc.dataCategory << "\n";
    cout << "Pixel format: " << (int)tensorDesc.pixelFormat << "\n";
    cout << "Pixel mapping: " << (int)tensorDesc.pixelMapping << "\n";
    cout << "Stride: ";
    for (unsigned int i = 0; i < CUDLA_LOADABLE_TENSOR_DESC_NUM_STRIDES; i++) {
        cout << tensorDesc.stride[i] << " ";
    }
    cout << "\n\n";
}

void Loadable::freeBuffers() noexcept {
    for (auto& buffer : bufferDLAInput) {
        if (buffer) {
            cudlaMemUnregister(devHandle, buffer);
            buffer = nullptr;
        }
    }
    for (auto& buffer : bufferDLAOutput) {
        if (buffer) {
            cudlaMemUnregister(devHandle, buffer);
            buffer = nullptr;
        }
    }

    for (auto& buffer : bufferGPUInput) {
        if (buffer) {
            cudaFree(buffer);
            buffer = nullptr;
        }
    }
    for (auto& buffer : bufferGPUOutput) {
        if (buffer) {
            cudaFree(buffer);
            buffer = nullptr;
        }
    }

    bufferGPUInput.clear();
    bufferGPUOutput.clear();
    bufferDLAInput.clear();
    bufferDLAOutput.clear();
}
