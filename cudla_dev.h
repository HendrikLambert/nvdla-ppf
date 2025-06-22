#ifndef cudla_dev_h
#define cudla_dev_h

#include "cudla.h"

class CUDLADev {
public:
    /**
     * Destructor for the CUDLADev class.
     */
    ~CUDLADev() noexcept;

    /**
     * Initializes the CUDLA device.
     *
     * @param deviceId The ID of the CUDA device to use.
     * @return true if the device was initialized successfully, false otherwise.
     */
    bool initialize(const int deviceId = 0) noexcept;

     // Disable copy semantics
    CUDLADev(const CUDLADev&) = delete;
    CUDLADev& operator=(const CUDLADev&) = delete;

    // Disable move semantics
    CUDLADev(CUDLADev&&) = delete;
    CUDLADev& operator=(CUDLADev&&) = delete;

protected:
    cudlaDevHandle devHandle = nullptr;
};

#endif // cudla_dev_h