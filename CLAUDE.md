# Integration Instructions for f5c into Python Package "fin"

## Overview
This document provides comprehensive instructions for integrating the f5c library (https://github.com/hasindu2008/f5c) into the Python package "fin". The goal is to create Python bindings for f5c's eventalign functionality with automatic CPU/GPU detection and seamless pip installation.

## Project Structure
```
fin/
├── __init__.py
├── _f5c/
│   ├── src/               # f5c source code
│   ├── include/           # f5c header files
│   ├── config.h           # Generated config file
│   └── ...                # Other f5c build files
├── bindings/
│   ├── __init__.py
│   ├── _eventalign.cpp    # Pybind11 bindings for eventalign
│   └── utils.cpp          # GPU detection and utility functions
├── core/
│   ├── __init__.py
│   └── eventalign.py      # Python wrapper API
├── tests/
│   ├── test_eventalign_cpu.py
│   ├── test_eventalign_gpu.py
│   └── test_gpu_detection.py
├── setup.py
├── setup.cfg
├── pyproject.toml
└── requirements.txt
```

## Step 1: Repository Setup and f5c Integration

### 1.1 Initialize Project Structure
```bash
mkdir -p fin/{_f5c,bindings,core,tests}
touch fin/__init__.py
touch fin/bindings/__init__.py
touch fin/core/__init__.py
```

### 1.2 Clone and Integrate f5c
```bash
# Clone f5c repository into the _f5c directory
git clone https://github.com/hasindu2008/f5c.git fin/_f5c

# Remove git history to keep the package clean
rm -rf fin/_f5c/.git

# Backup original Makefile for reference
cp fin/_f5c/Makefile fin/_f5c/Makefile.original
```

### 1.3 Modify f5c Build System for Python Integration
Create `fin/_f5c/Makefile.python` with the following content:
```makefile
# Custom Makefile for Python integration
CC = gcc
CXX = g++
NVCC = nvcc
CFLAGS = -O3 -Wall -fPIC -Iinclude -Isrc
CXXFLAGS = $(CFLAGS) -std=c++11
LDFLAGS = -shared

# Detect CUDA availability
HAS_CUDA := $(shell which nvcc >/dev/null 2>&1 && echo 1 || echo 0)

ifeq ($(HAS_CUDA),1)
    CUDA_FLAGS = -DUSE_CUDA -I/usr/local/cuda/include
    CUDA_LIBS = -lcudart -L/usr/local/cuda/lib64
    NVCC_FLAGS = -arch=sm_50 -Xcompiler -fPIC
endif

# Source files for eventalign functionality
EVENTALIGN_SRCS = src/eventalign.c src/align.c src/index.c src/common.c src/squiggle.c
EVENTALIGN_OBJS = $(EVENTALIGN_SRCS:.c=.o)

# GPU versions if available
ifeq ($(HAS_CUDA),1)
    GPU_SRCS = src/eventalign_gpu.cu src/align_gpu.cu
    GPU_OBJS = $(GPU_SRCS:.cu=.o)
endif

all: libf5c_eventalign.so

libf5c_eventalign.so: $(EVENTALIGN_OBJS) $(GPU_OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(CUDA_LIBS) -lm -lz -lhdf5 -lpthread

%.o: %.c
	$(CC) $(CFLAGS) $(CUDA_FLAGS) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(CUDA_FLAGS) -c -o $@ $<

clean:
	rm -f $(EVENTALIGN_OBJS) $(GPU_OBJS) libf5c_eventalign.so
```

## Step 2: Pybind11 Bindings Implementation

### 2.1 Create Pybind11 Bindings
Create `fin/bindings/_eventalign.cpp`:
```cpp
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
```

### 2.2 GPU Detection Utility
Create `fin/bindings/utils.cpp`:
```cpp
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
```

## Step 3: Python Wrapper API

Create `fin/core/eventalign.py`:
```python
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union

from fin.bindings._eventalign import EventAlignModule
from fin.bindings._utils import has_cuda_gpu

class EventAlign:
    """
    Python wrapper for f5c eventalign functionality with automatic CPU/GPU detection.
    """
    
    def __init__(self, force_cpu: bool = False):
        """
        Initialize EventAlign module.
        
        Args:
            force_cpu: If True, force CPU mode even if GPU is available
        """
        self.module = EventAlignModule(force_cpu=force_cpu)
        self.gpu_available = has_cuda_gpu() > 0
        self.is_using_gpu = self.module.is_using_gpu()
        
        if self.is_using_gpu:
            print(f"Using GPU acceleration (device 0)")
        else:
            print("Using CPU mode")
    
    def run(
        self,
        fast5_dir: str,
        bam_file: str,
        genome_file: str,
        batch_size: int = 1000,
        threads: int = 4,
        output_format: str = 'dataframe'
    ) -> Union[pd.DataFrame, List[Dict]]:
        """
        Run event alignment on nanopore sequencing data.
        
        Args:
            fast5_dir: Directory containing FAST5 files
            bam_file: BAM file with alignments
            genome_file: Reference genome FASTA file
            batch_size: Number of reads to process in each batch
            threads: Number of CPU threads to use (CPU mode only)
            output_format: 'dataframe' or 'dict' for output format
            
        Returns:
            Event alignment results in specified format
        """
        # Validate input files
        if not os.path.exists(fast5_dir):
            raise FileNotFoundError(f"FAST5 directory not found: {fast5_dir}")
        if not os.path.exists(bam_file):
            raise FileNotFoundError(f"BAM file not found: {bam_file}")
        if not os.path.exists(genome_file):
            raise FileNotFoundError(f"Genome file not found: {genome_file}")
        
        # Run eventalign
        results = self.module.run_eventalign(
            fast5_dir=fast5_dir,
            bam_file=bam_file,
            genome_file=genome_file,
            batch_size=batch_size,
            threads=threads
        )
        
        # Convert to appropriate format
        if output_format == 'dataframe':
            return self._to_dataframe(results)
        elif output_format == 'dict':
            return results
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _to_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        return pd.DataFrame(results)
    
    def get_device_info(self) -> Dict[str, Union[bool, int]]:
        """Get information about the computing device being used."""
        return {
            'gpu_available': self.gpu_available,
            'using_gpu': self.is_using_gpu,
            'device_type': 'GPU' if self.is_using_gpu else 'CPU'
        }
```

## Step 4: Build System and Package Configuration

### 4.1 Setup Script
Create `setup.py`:
```python
import os
import sys
import subprocess
import platform
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class BuildExt(build_ext):
    def run(self):
        try:
            subprocess.check_output(['make', '--version'])
        except OSError:
            raise RuntimeError("Make must be installed to build the f5c extension")
        
        for ext in self.extensions:
            self.build_extension(ext)
    
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable
        ]
        
        build_args = ['--']
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        # Build f5c library
        f5c_dir = os.path.join('fin', '_f5c')
        build_dir = os.path.join(f5c_dir, 'build')
        
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)
        
        # Check for CUDA
        has_cuda = False
        try:
            subprocess.check_output(['nvcc', '--version'])
            has_cuda = True
        except (OSError, subprocess.CalledProcessError):
            pass
        
        # Build the f5c library
        make_cmd = ['make', '-f', 'Makefile.python']
        if not has_cuda:
            make_cmd.append('HAS_CUDA=0')
        
        subprocess.check_call(make_cmd, cwd=f5c_dir)
        
        # Copy the built library to the extension directory
        lib_name = 'libf5c_eventalign.so'
        if platform.system() == 'Darwin':
            lib_name = 'libf5c_eventalign.dylib'
        elif platform.system() == 'Windows':
            lib_name = 'f5c_eventalign.dll'
        
        src_lib = os.path.join(f5c_dir, lib_name)
        dst_lib = os.path.join(extdir, lib_name)
        
        import shutil
        shutil.copy2(src_lib, dst_lib)

setup(
    name='fin',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=[CMakeExtension('fin._f5c')],
    cmdclass=dict(build_ext=BuildExt),
    install_requires=[
        'pybind11>=2.6.0',
        'numpy>=1.19.0',
        'pandas>=1.0.0',
        'h5py>=2.10.0',
    ],
    extras_require={
        'gpu': ['cupy>=8.0.0'],
    },
    python_requires='>=3.7',
    author='Your Name',
    author_email='your.email@example.com',
    description='Python bindings for f5c nanopore event alignment with CPU/GPU acceleration',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/fin',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)
```

### 4.2 PyProject Configuration
Create `pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=42", "wheel", "pybind11>=2.6.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["fin", "fin.core", "fin.bindings", "fin.tests"]

[tool.setuptools.package-data]
fin = ["_f5c/*.so", "_f5c/*.dylib", "_f5c/*.dll"]
```

## Step 5: Test Scripts

### 5.1 CPU Test Script
Create `fin/tests/test_eventalign_cpu.py`:
```python
import os
import pytest
import numpy as np
import pandas as pd
from fin.core.eventalign import EventAlign

class TestEventAlignCPU:
    @pytest.fixture
    def test_data_dir(self):
        return os.path.join(os.path.dirname(__file__), 'data')
    
    @pytest.fixture
    def eventalign_cpu(self):
        return EventAlign(force_cpu=True)
    
    def test_initialization(self, eventalign_cpu):
        assert not eventalign_cpu.is_using_gpu
        assert eventalign_cpu.gpu_available is not None
    
    @pytest.mark.skipif(not os.path.exists('test_data'), reason="Test data not available")
    def test_run_eventalign_cpu(self, eventalign_cpu, test_data_dir):
        fast5_dir = os.path.join(test_data_dir, 'fast5')
        bam_file = os.path.join(test_data_dir, 'alignments.bam')
        genome_file = os.path.join(test_data_dir, 'genome.fa')
        
        # Check if test files exist
        assert os.path.exists(fast5_dir)
        assert os.path.exists(bam_file)
        assert os.path.exists(genome_file)
        
        # Run eventalign
        results = eventalign_cpu.run(
            fast5_dir=fast5_dir,
            bam_file=bam_file,
            genome_file=genome_file,
            batch_size=100,
            threads=2,
            output_format='dataframe'
        )
        
        # Validate results
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        
        # Check expected columns
        expected_columns = ['read_name', 'contig', 'position', 'current', 'stdv', 'duration']
        for col in expected_columns:
            assert col in results.columns
        
        # Check data types
        assert results['current'].dtype == np.float32
        assert results['position'].dtype == np.int32
    
    def test_device_info(self, eventalign_cpu):
        device_info = eventalign_cpu.get_device_info()
        assert device_info['using_gpu'] is False
        assert isinstance(device_info['gpu_available'], bool)
```

### 5.2 GPU Test Script
Create `fin/tests/test_eventalign_gpu.py`:
```python
import os
import pytest
import numpy as np
import pandas as pd
from fin.core.eventalign import EventAlign
from fin.bindings._utils import has_cuda_gpu

@pytest.mark.skipif(has_cuda_gpu() == 0, reason="No CUDA GPU available")
class TestEventAlignGPU:
    @pytest.fixture
    def test_data_dir(self):
        return os.path.join(os.path.dirname(__file__), 'data')
    
    @pytest.fixture
    def eventalign_gpu(self):
        return EventAlign(force_cpu=False)
    
    def test_initialization(self, eventalign_gpu):
        assert eventalign_gpu.is_using_gpu
        assert eventalign_gpu.gpu_available is True
    
    @pytest.mark.skipif(not os.path.exists('test_data'), reason="Test data not available")
    def test_run_eventalign_gpu(self, eventalign_gpu, test_data_dir):
        fast5_dir = os.path.join(test_data_dir, 'fast5')
        bam_file = os.path.join(test_data_dir, 'alignments.bam')
        genome_file = os.path.join(test_data_dir, 'genome.fa')
        
        # Check if test files exist
        assert os.path.exists(fast5_dir)
        assert os.path.exists(bam_file)
        assert os.path.exists(genome_file)
        
        # Run eventalign with smaller batch size for testing
        results = eventalign_gpu.run(
            fast5_dir=fast5_dir,
            bam_file=bam_file,
            genome_file=genome_file,
            batch_size=50,
            threads=1,  # Not used in GPU mode
            output_format='dataframe'
        )
        
        # Validate results
        assert isinstance(results, pd.DataFrame)
        assert len(results) > 0
        
        # Check expected columns
        expected_columns = ['read_name', 'contig', 'position', 'current', 'stdv', 'duration']
        for col in expected_columns:
            assert col in results.columns
        
        # Performance check - GPU should be faster than CPU for large batches
        # (This would need benchmarking setup)
    
    def test_device_info(self, eventalign_gpu):
        device_info = eventalign_gpu.get_device_info()
        assert device_info['using_gpu'] is True
        assert device_info['gpu_available'] is True
```

### 5.3 GPU Detection Test Script
Create `fin/tests/test_gpu_detection.py`:
```python
import pytest
from fin.bindings._utils import has_cuda_gpu
from fin.core.eventalign import EventAlign

class TestGPUDetection:
    def test_cuda_detection_function(self):
        gpu_count = has_cuda_gpu()
        assert isinstance(gpu_count, int)
        assert gpu_count >= 0
    
    def test_eventalign_gpu_detection(self):
        # Test CPU mode
        ea_cpu = EventAlign(force_cpu=True)
        assert not ea_cpu.is_using_gpu
        
        # Test auto detection mode
        ea_auto = EventAlign(force_cpu=False)
        # This could be either True or False depending on system
        assert isinstance(ea_auto.is_using_gpu, bool)
        
        # Test device info
        device_info = ea_auto.get_device_info()
        assert 'gpu_available' in device_info
        assert 'using_gpu' in device_info
        assert 'device_type' in device_info
    
    @pytest.mark.skipif(has_cuda_gpu() == 0, reason="No CUDA GPU available")
    def test_gpu_mode_when_available(self):
        ea = EventAlign(force_cpu=False)
        assert ea.is_using_gpu
        assert ea.get_device_info()['device_type'] == 'GPU'
```

## Step 6: Build and Installation Instructions

### 6.1 Building from Source
```bash
# Install build dependencies
pip install pybind11 numpy pandas h5py setuptools wheel

# Clone your repository
git clone https://github.com/yourusername/fin.git
cd fin

# Install the package in editable mode
pip install -e .

# For GPU support (if available)
pip install -e .[gpu]
```

### 6.2 Testing the Package
```bash
# Run all tests
pytest fin/tests/

# Run CPU-specific tests only
pytest fin/tests/test_eventalign_cpu.py

# Run GPU-specific tests only (if GPU available)
pytest fin/tests/test_eventalign_gpu.py

# Run GPU detection tests
pytest fin/tests/test_gpu_detection.py
```

### 6.3 Using the Package
```python
import fin

# Initialize with automatic GPU detection
ea = fin.EventAlign()

# Get device information
device_info = ea.get_device_info()
print(f"Using {device_info['device_type']} mode")

# Run event alignment
results = ea.run(
    fast5_dir='/path/to/fast5/files',
    bam_file='/path/to/alignments.bam',
    genome_file='/path/to/reference.fa',
    batch_size=1000,
    threads=8
)

# Results as pandas DataFrame
print(results.head())
```

## Step 7: Important Notes for Claude Code

1. **f5c API Integration**: The actual C/C++ function signatures in `eventalign.cpp` and `utils.cpp` need to be adapted based on the real f5c source code structure. You'll need to:
   - Examine the f5c source code in `fin/_f5c/src/` to identify the correct function signatures
   - Create proper wrapper functions that expose the eventalign functionality
   - Handle memory management correctly (allocation/freeing of results)

2. **CUDA Integration**: For GPU support:
   - Ensure CUDA toolkit is properly installed on the build system
   - The Makefile needs to correctly detect CUDA availability and set appropriate flags
   - GPU memory management must be handled carefully to avoid leaks

3. **Cross-Platform Compatibility**:
   - Handle different library extensions (.so, .dylib, .dll)
   - Test on Linux, macOS, and Windows if needed
   - Consider different CUDA versions and GPU architectures

4. **Error Handling**:
   - Add comprehensive error checking in the C++ bindings
   - Handle file I/O errors gracefully
   - Add timeouts for long-running operations
   - Validate input parameters before calling f5c functions

5. **Performance Optimization**:
   - Use appropriate batch sizes for GPU vs CPU
   - Implement proper memory pooling for GPU operations
   - Consider asynchronous operations for better performance
   - Add progress reporting for long-running alignments

6. **Documentation**:
   - Add docstrings to all Python functions and classes
   - Create API documentation using Sphinx
   - Include examples in the README.md
   - Document system requirements and dependencies

7. **Testing Data**:
   - Create small test datasets for CI/CD testing
   - Include example FAST5 files, BAM files, and reference genomes
   - Add checksums for test data validation
   - Consider using synthetic data for testing to avoid large file dependencies

This integration provides a robust foundation for incorporating f5c's powerful event alignment capabilities into your Python package with seamless CPU/GPU acceleration. The modular design allows for easy extension and maintenance while providing a clean Python API for end users.
