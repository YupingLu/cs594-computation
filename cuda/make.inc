
CC      = cc
NVCC    = /pkgs/cuda-5.5/bin/nvcc
CFLAGS  = -O3
NVFLAGS = -O3

CUDA    = /pkgs/cuda-5.5
LAPACK  =  /home/ylu20/cuda/lib
LAPACKE = /home/ylu20/cuda/lapack-3.5.0/lapacke

INC = -I$(CUDA)/include -I$(LAPACKE)/include
LIB = -L$(CUDA)/lib64 -L$(LAPACK)
LIBS = -llapacke -llapack -lblas -lcudart -lgfortran
