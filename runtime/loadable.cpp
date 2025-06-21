#include "loadable.h"
#include <fstream>
#include <iostream>

using namespace std;

Loadable::Loadable(const string& filename) noexcept {
    this->filename = filename;
}

Loadable::~Loadable() noexcept {
    // Unload the module if it was loaded
    if (moduleHandle) {
        cudlaStatus status = cudlaModuleUnload(moduleHandle, 0);
        if (status != cudlaSuccess) {
            cout << "cudlaModuleUnload failed: " << status << endl;
        }
    }
}

bool Loadable::loadBinary() {
    // Load the file into memory
    ifstream file(filename, ios::binary | ios::ate);
    if (!file) {
        cout << "Unable to open file " << filename << endl;
        return false;
    }

    // Get the size of the file
    fileSize = file.tellg();
    file.seekg(0, ios::beg);
    loadableData = make_unique<char[]>(fileSize);

    // Read the file into loadableData. Return if failed
    if (!file.read(loadableData.get(), fileSize)) {
        cout << "Unable to read file " << filename << endl;
        return false;
    }
    
    return true;
}

bool Loadable::loadModule(cudlaDevHandle const devHandle) noexcept {
    // Load the module from memory
    cudlaStatus status = cudlaModuleLoadFromMemory(devHandle,
        reinterpret_cast<const uint8_t*>(loadableData.get()),
        fileSize,
        &moduleHandle,
        0);

    if (status != cudlaSuccess) {
        cout << "cudlaModuleLoadFromMemory failed: " << status << endl;
        return false;
    }

    if (!loadModuleAttributes()) {
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
    inputTensorDesc.resize(attribute.numInputTensors);

    // Get number of output tensors
    status = cudlaModuleGetAttributes(moduleHandle, CUDLA_NUM_OUTPUT_TENSORS, &attribute);
    if (status != cudlaSuccess) {
        cout << "cudlaModuleGetAttributes for output tensors failed: " << status << endl;
        return false;
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

void Loadable::printTensorDesc() const noexcept {
    for (const auto& desc : inputTensorDesc) {
        printTensorDescHelper(desc);
    }
    for (const auto& desc : outputTensorDesc) {
        printTensorDescHelper(desc);
    }
}

void Loadable::printTensorDescHelper(cudlaModuleTensorDescriptor const& tensorDesc) const noexcept {
    cout << "Tensor desc: " << endl;
    cout << "Name: " << tensorDesc.name << endl;
    cout << "Size: " << tensorDesc.size << endl;
    cout << "N: " << tensorDesc.n << " C: " << tensorDesc.c << " H: " << tensorDesc.h << " W: " << tensorDesc.w << endl;
    cout << "Data format: " << (int)tensorDesc.dataFormat << endl;
    cout << "Data type: " << (int)tensorDesc.dataType << endl;
    cout << "Data category: " << (int)tensorDesc.dataCategory << endl;
    cout << "Pixel format: " << (int)tensorDesc.pixelFormat << endl;
    cout << "Pixel mapping: " << (int)tensorDesc.pixelMapping << endl;
    cout << "Stride: ";
    for (unsigned int i = 0; i < CUDLA_LOADABLE_TENSOR_DESC_NUM_STRIDES; i++) {
        cout << tensorDesc.stride[i] << " ";
    }
    cout << endl << endl;
}

