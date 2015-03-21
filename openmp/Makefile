all: omp_dgemm seq_dgemm

omp_dgemm : omp_dgemm.cpp
	g++ omp_dgemm.cpp -o omp_dgemm -fopenmp

seq_dgemm : seq_dgemm.cpp
	g++ seq_dgemm.cpp -o seq_dgemm
