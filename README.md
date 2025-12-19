# fin - Python bindings for f5c

Python package providing seamless integration with [f5c](https://github.com/hasindu2008/f5c) for nanopore event alignment with automatic CPU/GPU detection and acceleration.

## Features

- **Automatic GPU Detection**: Seamlessly switches between CPU and GPU modes based on system capabilities
- **High Performance**: Leverages CUDA acceleration when available for faster processing
- **Easy Installation**: Standard pip installation with automatic dependency management
- **Pythonic API**: Clean, intuitive Python interface with pandas DataFrame output
- **Cross-Platform**: Works on Linux, macOS, and Windows (where CUDA is available)

## Installation

### From Source

```bash
# Install build dependencies
pip install pybind11 numpy pandas h5py setuptools wheel

# Clone the repository
git clone https://github.com/yourusername/fin.git
cd fin

# Install in editable mode
pip install -e .

# For GPU support (if CUDA is available)
pip install -e .[gpu]
```

### Requirements

- Python 3.7+
- CUDA Toolkit 8.0+ (optional, for GPU acceleration)
- HDF5 library
- C++ compiler with C++11 support

## Quick Start

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

## CPU/GPU Modes

### Force CPU Mode
```python
ea = fin.EventAlign(force_cpu=True)
```

### Automatic Detection (Default)
```python
ea = fin.EventAlign()  # Automatically uses GPU if available
```

## API Reference

### EventAlign Class

#### `__init__(force_cpu=False)`
Initialize the EventAlign module.

- `force_cpu` (bool): If True, force CPU mode even if GPU is available

#### `run(fast5_dir, bam_file, genome_file, batch_size=1000, threads=4, output_format='dataframe')`
Run event alignment on nanopore sequencing data.

**Parameters:**
- `fast5_dir` (str): Directory containing FAST5 files
- `bam_file` (str): BAM file with alignments
- `genome_file` (str): Reference genome FASTA file
- `batch_size` (int): Number of reads to process in each batch (default: 1000)
- `threads` (int): Number of CPU threads to use (CPU mode only, default: 4)
- `output_format` (str): 'dataframe' or 'dict' for output format (default: 'dataframe')

**Returns:**
- pandas.DataFrame or List[Dict]: Event alignment results

#### `get_device_info()`
Get information about the computing device being used.

**Returns:**
- Dict[str, Union[bool, int]]: Device information including GPU availability and usage

## Testing

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

## Performance Notes

- **GPU Mode**: Recommended for large datasets (>10,000 reads)
- **CPU Mode**: Suitable for smaller datasets or when CUDA is not available
- **Batch Size**: Larger batch sizes generally improve GPU performance
- **Memory Usage**: GPU mode requires sufficient GPU memory for batch processing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [f5c](https://github.com/hasindu2008/f5c) - Original C/C++ implementation
- [Pybind11](https://github.com/pybind/pybind11) - Python C++ binding library