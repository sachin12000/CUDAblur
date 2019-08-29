# CUDAblur
A program that implements box blur with GPU acceleration using CUDA. Currently only supports the PPM image format.

This program can only be compiled with NVDIA's CUDA compiler. Also a CUDA compatible GPU with the correct drivers is needed to run this program.

The command is

CUDAblur [blur box radius] "input file directory" "output file directory"

example:
CUDAblur 40 "./fox.ppm" "./fox_blurred.ppm"