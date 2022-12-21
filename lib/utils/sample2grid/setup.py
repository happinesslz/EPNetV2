"""
Created by silver at 2019/10/11 21:23
Email: xiwuchencn[at]gmail[dot]com
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
        name = 'gridvoxel',
        ext_modules = [
            CUDAExtension('gridvoxel_cuda', [
                'Voxel_gpu.cu'
            ],
                          extra_compile_args = { 'cxx': ['-g'], 'nvcc': ['-O2']
                                                 })
        ],
        cmdclass = { 'build_ext': BuildExtension }
        , include_dirs = ['./'],
)
setup(
        name = 'gaussiangridvoxel',
        ext_modules = [
            CUDAExtension('gaussian_gridvoxel_cuda', [
                'Gaussian_voxel_gpu.cu'
            ],
                          extra_compile_args = { 'cxx': ['-g'], 'nvcc': ['-O2']
                                                 })
        ],
        cmdclass = { 'build_ext': BuildExtension }
        , include_dirs = ['./'],
)

setup(
        name = 'bilineargridvoxel',
        ext_modules = [
            CUDAExtension('bilinear_gridvoxel_cuda', [
                'Bilinear_voxel_gpu.cu'
            ],
                          extra_compile_args = { 'cxx': ['-g'], 'nvcc': ['-O2']
                                                 })
        ],
        cmdclass = { 'build_ext': BuildExtension }
        , include_dirs = ['./'],
)