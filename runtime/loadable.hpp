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

    bool runTask(const cudaStream_t stream, const int buffIndex) const noexcept;

    std::tuple<int, int, int, int> getInputTensorShape(int index) const noexcept;

private:
    std::string filename;
    size_t fileSize = 0;

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

    bool loadBinary();

    bool loadModule() noexcept;

    bool loadModuleAttributes() noexcept;

    bool createDevHandle(const uint64_t dlaId) noexcept;

    void printTensorDescHelper(const cudlaModuleTensorDescriptor& tensorDesc) const noexcept;

    void freeBuffers() noexcept;

    bool allocateSingleBuffer(uint64_t size, std::vector<void*>& gpuBuffer, std::vector<uint64_t*>& dlaBuffer) const noexcept;
    
};
