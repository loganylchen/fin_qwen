#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <memory>

namespace py = pybind11;

// Forward declarations for f5c functions
extern "C" {
    typedef struct {
        char* read_name;
        char* contig;
        int position;
        float current;
        float stdv;
        float duration;
        char* model_kmer;
        int model_mean;
        int model_stdv;
    } EventAlignmentResult;

    // CPU implementation
    EventAlignmentResult* f5c_eventalign_cpu(
        const char* fast5_dir,
        const char* bam_file,
        const char* genome_file,
        int batch_size,
        int threads
    );

    // GPU implementation (if available)
    EventAlignmentResult* f5c_eventalign_gpu(
        const char* fast5_dir,
        const char* bam_file,
        const char* genome_file,
        int batch_size,
        int gpu_device
    );

    // Cleanup function
    void free_eventalign_results(EventAlignmentResult* results);

    // GPU detection
    int has_cuda_gpu();
}

class EventAlignModule {
private:
    bool use_gpu;
    int gpu_device;

public:
    EventAlignModule(bool force_cpu = false) {
        if (force_cpu) {
            use_gpu = false;
        } else {
            use_gpu = (has_cuda_gpu() > 0);
        }
        gpu_device = 0; // Default GPU device
    }

    py::list run_eventalign(
        const std::string& fast5_dir,
        const std::string& bam_file,
        const std::string& genome_file,
        int batch_size = 1000,
        int threads = 4
    ) {
        EventAlignmentResult* results = nullptr;

        try {
            if (use_gpu) {
                results = f5c_eventalign_gpu(
                    fast5_dir.c_str(),
                    bam_file.c_str(),
                    genome_file.c_str(),
                    batch_size,
                    gpu_device
                );
            } else {
                results = f5c_eventalign_cpu(
                    fast5_dir.c_str(),
                    bam_file.c_str(),
                    genome_file.c_str(),
                    batch_size,
                    threads
                );
            }

            // Convert results to Python list of dictionaries
            py::list py_results;
            // Implementation to convert C results to Python objects
            // This would need to be filled in based on actual f5c API

            return py_results;
        } catch (const std::exception& e) {
            if (results) free_eventalign_results(results);
            throw std::runtime_error(std::string("EventAlign error: ") + e.what());
        }
    }

    bool is_using_gpu() const {
        return use_gpu;
    }
};

PYBIND11_MODULE(_eventalign, m) {
    py::class_<EventAlignModule>(m, "EventAlignModule")
        .def(py::init<bool>(), py::arg("force_cpu") = false)
        .def("run_eventalign", &EventAlignModule::run_eventalign,
            py::arg("fast5_dir"),
            py::arg("bam_file"),
            py::arg("genome_file"),
            py::arg("batch_size") = 1000,
            py::arg("threads") = 4)
        .def("is_using_gpu", &EventAlignModule::is_using_gpu)
        .def_property_readonly("gpu_available", []() { return has_cuda_gpu() > 0; });

    m.doc() = "f5c eventalign bindings for Python with automatic CPU/GPU detection";
}