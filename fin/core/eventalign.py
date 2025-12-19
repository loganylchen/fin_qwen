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