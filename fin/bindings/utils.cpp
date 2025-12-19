#include <pybind11/pybind11.h>
#include <cuda_runtime.h>

namespace py = pybind11;

extern "C" {
    int has_cuda_gpu() {
        int deviceCount = 0;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        return (error == cudaSuccess && deviceCount > 0) ? deviceCount : 0;
    }
}

PYBIND11_MODULE(_utils, m) {
    m.def("has_cuda_gpu", &has_cuda_gpu, "Check if CUDA GPU is available");
    m.doc() = "GPU detection utilities for f5c integration";
}