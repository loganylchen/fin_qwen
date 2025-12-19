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