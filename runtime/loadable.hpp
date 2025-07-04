#pragma once

#include <string>
#include <vector>
#include <memory>
#include <tuple>

#include "cudla.h"
#include "cuda_runtime.h"

class Loadable {
public:
    // Factory method to create a Loadable instance
    static std::unique_ptr<Loadable> create(const std::string& filename, const uint64_t dlaId = 0);

    ~Loadable() noexcept;

    /**
     * Generate input and output buffers based on the loaded module.
     * This method allocates GPU and DLA buffers for the specified count of tensors.
     * 
     * @param count The number of input/output tensors to allocate buffers for.
     * @return true if buffers were successfully allocated, false otherwise.
     */
    bool allocateBuffers(int count) noexcept;

    /**
     * Print the input and output tensor descriptors to the console.
     */
    void printTensorDesc() const noexcept;

    /**
     * Run the task on the DLA using the specified stream and buffer index.
     */
    bool runTask(const cudaStream_t stream, const int buffIndex) const noexcept;

    /**
     * Get the input tensor shape for the specified index.
     * 
     * @param index The index of the input tensor descriptor.
     * @return A tuple containing the shape of the input tensor (n, c, h, w).
     */
    std::tuple<int, int, int, int> getInputTensorShape(int index) const noexcept;

private:
    std::string filename;
    size_t fileSize = 0;

    // CUDLA device handle and module handle (owner)
    cudlaDevHandle devHandle = nullptr;
    cudlaModule moduleHandle = nullptr;

    std::unique_ptr<char[]> loadableData;

    std::vector<cudlaModuleTensorDescriptor> inputTensorDesc;
    std::vector<cudlaModuleTensorDescriptor> outputTensorDesc;

    std::vector<void*> bufferGPUInput;
    std::vector<void*> bufferGPUOutput;
    std::vector<uint64_t*> bufferDLAInput;
    std::vector<uint64_t*> bufferDLAOutput;

    
    explicit Loadable(const std::string& filename) noexcept;

    /**
     * Load the binary data from the specified file.
     * 
     * @return true if the binary data was successfully loaded, false otherwise.
     */
    bool loadBinary();

    /**
     * Load the CUDLA module from the binary data.
     * 
     * @return true if the module was successfully loaded, false otherwise.
     */
    bool loadModule() noexcept;

    /**
     * Load the module attributes such as input and output tensor descriptors.
     * 
     * @return true if the module attributes were successfully loaded, false otherwise.
     */
    bool loadModuleAttributes() noexcept;

    /**
     * Create a device handle for the specified DLA ID.
     * 
     * @param dlaId The ID of the DLA to create a handle for.
     * @return true if the device handle was successfully created, false otherwise.
     */
    bool createDevHandle(const uint64_t dlaId) noexcept;

    /**
     * Print the tensor descriptor to the console.
     * 
     * @param tensorDesc The tensor descriptor to print.
     */
    void printTensorDescHelper(const cudlaModuleTensorDescriptor& tensorDesc) const noexcept;

    /**
     * Free the allocated buffers.
     * Unregisters the buffers from the DLA first, then frees the CUDA memory.
     */
    void freeBuffers() noexcept;

    /**
     * Allocate a single buffer of the specified size.
     * 
     * @param size The size of the buffer to allocate.
     * @param gpuBuffer The vector to store the allocated GPU buffer pointer.
     * @param dlaBuffer The vector to store the allocated DLA buffer pointer.
     * @return true if the buffer was successfully allocated, false otherwise.
     */
    bool allocateSingleBuffer(uint64_t size, std::vector<void*>& gpuBuffer, std::vector<uint64_t*>& dlaBuffer) const noexcept;
    
};
