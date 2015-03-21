all: diagonal transpose ring grids pdgemm

diagonal : diagonal.c
	mpicc diagonal.c -o diagonal

transpose : transpose.c
	mpicc transpose.c -o transpose

ring : ring.c
	mpicc ring.c -o ring

grids : grids.c
	mpicc grids.c -o grids

pdgemm : pdgemm.c
	mpicc pdgemm.c -o pdgemm
