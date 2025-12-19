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