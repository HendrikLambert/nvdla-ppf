#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "cuda_runtime.h"
#include "cudla.h"
#include "cuda_fp16.h"

void printTensorDesc(cudlaModuleTensorDescriptor* tensorDesc);

using namespace std;

int main() {
    // Load loadable
    const char *loadableFile = "pfb_model_dft-c256-t16-b1.nvdla";
    unsigned char *loadableData = NULL;

    FILE *fp = NULL;
    struct stat st;
    size_t file_size;
    size_t actually_read = 0;

    fp = fopen(loadableFile, "rb");
    if (fp == NULL) {
        std::cerr << "Error: Unable to open file " << loadableFile << std::endl;
        return -1;
    }

    if (stat(loadableFile, &st) != 0) {
        std::cerr << "Error: Unable to open file " << loadableFile << std::endl;
        return -11;
    }

    file_size = st.st_size;

    loadableData = (unsigned char *)malloc(file_size);
    if (loadableData == NULL) {
        //   DPRINTF("Cannot Allocate memory for loadable\n");
        std::cerr << "Error: Unable to malloc" << std::endl;
        return 1;
    }

    actually_read = fread(loadableData, 1, file_size, fp);
    if (actually_read != file_size) {
        free(loadableData);
        std::cerr << "Error: Unable to open file " << loadableFile << std::endl;
        return 1;
    }

    fclose(fp);
    

    // CUDA
    cudaStream_t stream;
    cudaError_t cudaStatus;
    const char *errPtr = NULL;

    // cudla
    cudlaDevHandle devHandle;
    cudlaModule moduleHandle;
    cudlaStatus dlaStatus;
    cudlaModuleAttribute attribute;
    uint32_t numInputTensors = 0;
    uint32_t numOutputTensors = 0;

    // Init CUDA
    cudaStatus = cudaFree(0);
    if (cudaStatus != cudaSuccess) {
        errPtr = cudaGetErrorName(cudaStatus);
        cout << "Error: creating cudaFree = " << errPtr << endl;
        return -1;
    }

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        errPtr = cudaGetErrorName(cudaStatus);
        cout << "Error: creating cudaSetDevice = " << errPtr << endl;
        return -1;
    }

    // Create cuda stream
    cudaStatus = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (cudaStatus != cudaSuccess) {
        errPtr = cudaGetErrorName(cudaStatus);
        cout << "Error in creating cuda stream = " << errPtr << endl;
        return -1;
    }

    // Init cuDLA
    dlaStatus = cudlaCreateDevice(0, &devHandle, CUDLA_CUDA_DLA);
    if (dlaStatus != cudlaSuccess) {
        cout << "cudlaCreateDevice: " << dlaStatus << endl;
        return -1;
    }

    // Load module
    dlaStatus = cudlaModuleLoadFromMemory(devHandle, loadableData, file_size, &moduleHandle, 0);
    if (dlaStatus != cudlaSuccess) {
        cout << "cudlaModuleLoadFromMemory: " << dlaStatus << endl;
        return -1;
    }

    // Get model attributes (input tensors)
    dlaStatus = cudlaModuleGetAttributes(moduleHandle, CUDLA_NUM_INPUT_TENSORS, &attribute);
    if (dlaStatus != cudlaSuccess) {
        cout << "cudlaModuleGetAttributes: " << dlaStatus << endl;
        return -1;
    }
    numInputTensors = attribute.numInputTensors;
    cout << "numInputTensors = " << numInputTensors << endl;

    // Get model attributes (output tensors)
    dlaStatus = cudlaModuleGetAttributes(moduleHandle, CUDLA_NUM_OUTPUT_TENSORS, &attribute);
    if (dlaStatus != cudlaSuccess) {
        cout << "cudlaModuleGetAttributes: " << dlaStatus << endl;
        return -1;
    }
    numOutputTensors = attribute.numOutputTensors;
    cout << "numOutputTensors = " << numOutputTensors << endl;

    // Get model shapes
    cudlaModuleTensorDescriptor* inputTensorDesc = (cudlaModuleTensorDescriptor*)malloc(sizeof(cudlaModuleTensorDescriptor) * numInputTensors);
    cudlaModuleTensorDescriptor* outputTensorDesc = (cudlaModuleTensorDescriptor*)malloc(sizeof(cudlaModuleTensorDescriptor) * numOutputTensors);

    if (inputTensorDesc == NULL || outputTensorDesc == NULL) {
        cout << "Error: Unable to malloc" << endl;
        return -1;
    }

    // Get input tensor descriptors
    attribute.inputTensorDesc = inputTensorDesc;
    dlaStatus = cudlaModuleGetAttributes(moduleHandle, CUDLA_INPUT_TENSOR_DESCRIPTORS, &attribute);
    if (dlaStatus != cudlaSuccess) {
        cout << "cudlaModuleGetAttributes input tensor desc: " << dlaStatus << endl;
        return -1;
    }
    printTensorDesc(inputTensorDesc);

    // Get output tensor descriptors
    attribute.outputTensorDesc = outputTensorDesc;
    dlaStatus = cudlaModuleGetAttributes(moduleHandle, CUDLA_OUTPUT_TENSOR_DESCRIPTORS, &attribute);
    if (dlaStatus != cudlaSuccess) {
        cout << "cudlaModuleGetAttributes output tensor desc: " << dlaStatus << endl;
        return -1;
    }
    printTensorDesc(outputTensorDesc);

    // Setup host buffers
    __half* hostInputBuffer = (__half*) malloc(inputTensorDesc[0].size);
    __half* hostOutputBuffer = (__half*) malloc(outputTensorDesc[0].size);
    if (hostInputBuffer == NULL || hostOutputBuffer == NULL) {
        cout << "Error: Unable to malloc" << endl;
        return -1;
    }

    // memset(hostInputBuffer, 0, inputTensorDesc[0].size);
    // Fill input buffer with data
    for (unsigned int i = 0; i < inputTensorDesc[0].size/2; i++) {
        __half val = __float2half(0.0f);
        hostInputBuffer[i] = val;
    }
    
    
    // for (unsigned int i = 0; i < (16*17); i++) {
    //     hostInputBuffer[i] = __float2half(1.0f);
    // }

    // Input 2 * 256 * 1 * 16
    // How is this strided?
    // It looks like it is grouped by 16*16?

    // Batch x channels x 1 x taps
    // array[0] waarde 0, tap 0
    // aaray[1] waarde 0, tap 1
    // array[16] waarde 1, tap 0


    hostInputBuffer[0] = __float2half(1.0f);
    hostInputBuffer[1] = __float2half(2.0f);
    // hostInputBuffer[3] = __float2half(1.0f);
    // hostInputBuffer[16] = __float2half(2.0f);
    // hostInputBuffer[32] = __float2half(2.0f);

    // time 2
    // hostInputBuffer[256*16] = __float2half(1.0f);
    // hostInputBuffer[256*16 + 16] = __float2half(1.0f);
    // hostInputBuffer[256*16 + 32] = __float2half(2.0f);


    memset(hostOutputBuffer, 0, outputTensorDesc[0].size);

    // Setup GPU buffers
    void* inputBufferGPU = NULL;
    void* outputBufferGPU = NULL;

    cudaStatus = cudaMalloc(&inputBufferGPU, inputTensorDesc[0].size);
    if (cudaStatus != cudaSuccess) {
        errPtr = cudaGetErrorName(cudaStatus);
        cout << "Error: creating inputBufferGPU = " << errPtr << endl;
        return -1;
    }
    cudaStatus = cudaMalloc(&outputBufferGPU, outputTensorDesc[0].size);
    if (cudaStatus != cudaSuccess) {
        errPtr = cudaGetErrorName(cudaStatus);
        cout << "Error: creating outputBufferGPU = " << errPtr << endl;
        return -1;
    }

    // DLA buffers
    uint64_t* inputBufferDLA = 0;
    uint64_t* outputBufferDLA = 0;
    // Register CUDA buffer for DLA
    dlaStatus = cudlaMemRegister(devHandle, (uint64_t*) inputBufferGPU, inputTensorDesc[0].size, &inputBufferDLA, 0);
    if (dlaStatus != cudlaSuccess) {
        cout << "cudlaMemRegister input buffer: " << dlaStatus << endl;
        return -1;
    }
    dlaStatus = cudlaMemRegister(devHandle, (uint64_t*) outputBufferGPU, outputTensorDesc[0].size, &outputBufferDLA, 0);
    if (dlaStatus != cudlaSuccess) {
        cout << "cudlaMemRegister output buffer: " << dlaStatus << endl;
        return -1;
    }

    // Create task
    cudlaTask task;
    task.moduleHandle = moduleHandle;
    task.inputTensor = &inputBufferDLA;
    task.numInputTensors = 1;
    task.outputTensor = &outputBufferDLA;
    task.numOutputTensors = 1;
    task.waitEvents = NULL;
    task.signalEvents = NULL;

  
    // Copy input data to GPU
    cudaStatus = cudaMemcpyAsync(inputBufferGPU, hostInputBuffer, inputTensorDesc[0].size, cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) {
        errPtr = cudaGetErrorName(cudaStatus);
        cout << "Error: copying inputBufferGPU = " << errPtr << endl;
        return -1;
    }

    // Submit task
    dlaStatus = cudlaSubmitTask(devHandle, &task, 1, stream, 0);
    if (dlaStatus != cudlaSuccess) {
        cout << "cudlaSubmitTask: " << dlaStatus << endl;
        return -1;
    }

    // Copy output data from GPU
    cudaStatus = cudaMemcpyAsync(hostOutputBuffer, outputBufferGPU, outputTensorDesc[0].size, cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) {
        errPtr = cudaGetErrorName(cudaStatus);
        cout << "Error: copying outputBufferGPU = " << errPtr << endl;
        return -1;
    }

    // Synchronize stream
    cudaStatus = cudaStreamSynchronize(stream);
    if (cudaStatus != cudaSuccess) {
        errPtr = cudaGetErrorName(cudaStatus);
        cout << "Error: synchronizing stream = " << errPtr << endl;
        return -1;
    }
    

    // Check output data
    for (unsigned int i = 0; i < outputTensorDesc[0].size/2; i++) {
    // for (unsigned int i = 0; i < 1024; i++) {
        __half val = hostOutputBuffer[i];
        float floatVal = __half2float(val);
        // if (floatVal != 0.0f) {
            printf("%.10f", floatVal);
        // }
    }
    cout << endl;

    // Unregister DLA buffers
    dlaStatus = cudlaMemUnregister(devHandle, inputBufferDLA);
    if (dlaStatus != cudlaSuccess) {
        cout << "cudlaMemUnregister input buffer: " << dlaStatus << endl;
        return -1;
    }
    dlaStatus = cudlaMemUnregister(devHandle, outputBufferDLA);
    if (dlaStatus != cudlaSuccess) {
        cout << "cudlaMemUnregister output buffer: " << dlaStatus << endl;
        return -1;
    }
    // Free GPU buffers
    cudaStatus = cudaFree(inputBufferGPU);
    if (cudaStatus != cudaSuccess) {
        errPtr = cudaGetErrorName(cudaStatus);
        cout << "Error: freeing inputBufferGPU = " << errPtr << endl;
        return -1;
    }
    cudaStatus = cudaFree(outputBufferGPU);
    if (cudaStatus != cudaSuccess) {
        errPtr = cudaGetErrorName(cudaStatus);
        cout << "Error: freeing outputBufferGPU = " << errPtr << endl;
        return -1;
    }
    // Free host buffers
    free(hostInputBuffer);
    free(hostOutputBuffer);
    // Free DLA buffers
    free(inputTensorDesc);
    free(outputTensorDesc);
    // Unload module
    dlaStatus = cudlaModuleUnload(moduleHandle, 0);
    if (dlaStatus != cudlaSuccess) {
        cout << "cudlaModuleUnload: " << dlaStatus << endl;
        return -1;
    }
    // Destroy stream
    cudaStatus = cudaStreamDestroy(stream);
    if (cudaStatus != cudaSuccess) {
        errPtr = cudaGetErrorName(cudaStatus);
        cout << "Error: destroying stream = " << errPtr << endl;
        return -1;
    }
    // Destroy device
    dlaStatus = cudlaDestroyDevice(devHandle);
    if (dlaStatus != cudlaSuccess) {
        cout << "cudlaDestroyDevice: " << dlaStatus << endl;
        return -1;
    }
    // Free loadable data
    free(loadableData);
    cout << "Success" << endl;
    return 0;
}

void printTensorDesc(cudlaModuleTensorDescriptor* tensorDesc) {
    cout << "Tensor desc: " << endl;
    cout << "Name: " << tensorDesc[0].name << endl;
    cout << "Size: " << tensorDesc[0].size << endl;
    cout << "N: " << tensorDesc[0].n << " C: " << tensorDesc[0].c << " H: " << tensorDesc[0].h << " W: " << tensorDesc[0].w << endl;
    cout << "Data format: " << (int)tensorDesc[0].dataFormat << endl;
    cout << "Data type: " << (int)tensorDesc[0].dataType << endl;
    cout << "Data category: " << (int)tensorDesc[0].dataCategory << endl;
    cout << "Pixel format: " << (int)tensorDesc[0].pixelFormat << endl;
    cout << "Pixel mapping: " << (int)tensorDesc[0].pixelMapping << endl;
    cout << "Stride: ";
    for (unsigned int i = 0; i < CUDLA_LOADABLE_TENSOR_DESC_NUM_STRIDES; i++) {
        cout << tensorDesc[0].stride[i] << " ";
    }
    cout << endl << endl;
}