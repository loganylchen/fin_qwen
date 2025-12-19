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
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
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