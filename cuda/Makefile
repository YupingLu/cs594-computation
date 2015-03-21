
include make.inc

all:
	$(NVCC) $(NVFLAGS) $(NVINC) -c chol_kernel.cu
	$(NVCC) $(CFLAGS) $(INC) -c chol_driver.c
	$(CC) chol_driver.o chol_kernel.o $(LIB) -o chol_exec $(LIBS)

clean:
	rm -f *.o chol_exec
