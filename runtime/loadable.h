#ifndef LOADABLE_H
#define LOADABLE_H

#include <string>
#include <memory>
#include <vector>
#include "cudla.h"

class Loadable {

public:
    /**
     * Constructor that initializes the Loadable object with a filename.
     *
     * @param filename The name of the file to load.
     */
    Loadable(const std::string& filename) noexcept;

    ~Loadable() noexcept;

    /**
     * Loads the data from the file specified in the constructor.
     *
     * @return true if the data was loaded successfully, false otherwise.
     */
    bool loadBinary();

    /**
     * Loads the module from the loaded data.
     *
     * @return true if the module was loaded successfully, false otherwise.
     */
    bool loadModule(cudlaDevHandle const devHandle) noexcept;

    /**
     * Print the input and output tensor descriptors of the module.
     */
    void printTensorDesc() const noexcept;

    // Disable copy semantics
    Loadable(const Loadable&) = delete;
    Loadable& operator=(const Loadable&) = delete;

    // Disable move semantics
    Loadable(Loadable&&) = delete;
    Loadable& operator=(Loadable&&) = delete;


private:
    std::string filename;
    std::unique_ptr<char[]> loadableData;
    size_t fileSize;
    cudlaModule moduleHandle = nullptr;

    std::vector<cudlaModuleTensorDescriptor> inputTensorDesc;
    std::vector<cudlaModuleTensorDescriptor> outputTensorDesc;

    bool loadModuleAttributes() noexcept;

    void printTensorDescHelper(cudlaModuleTensorDescriptor const &tensorDesc) const noexcept;
};

#endif