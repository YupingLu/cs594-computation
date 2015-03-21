all: laplace laplace_shmem

laplace : laplace.c
	gcc laplace.c -o laplace -lm

laplace_shmem : laplace_shmem.c
	shmemcc laplace_shmem.c -o laplace_shmem -lm
