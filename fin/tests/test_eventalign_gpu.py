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