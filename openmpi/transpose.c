/*
* hw 3 Yuping Lu
* MPI datatype: Using only MPI features provide a simple solution for a matrix transpose
*/
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "mpi.h"

int main (int argc, char *argv[]) {
	int ierr, rank, numtasks;
	int i, j;
	MPI_Datatype sendtype;
	MPI_Datatype recvtype;
	MPI_Status stat;

	/* input 2d matrix a*/
	int a[8][8] = {
		1, 2, 3, 4, 5, 6, 7, 8, 
		9, 10, 11, 12, 13, 14, 15, 16, 
		17, 18, 19, 20, 21, 22, 23, 24,
		25, 26, 27, 28, 29, 30, 31, 32,
		33, 34, 35, 36, 37, 38, 39, 40,
		41, 42, 43, 44, 45, 46, 47, 48,
		49, 50, 51, 52, 53, 54, 55, 56,
		57, 58, 59, 60, 61, 62, 63, 64
	};

	/* store the transposed matrix b*/
	int b[8][8];

	ierr = MPI_Init ( &argc, &argv );
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
	ierr = MPI_Comm_size(MPI_COMM_WORLD, &numtasks); 

	/* new datatype for matrix transpose */
	MPI_Type_vector(8, 1, 8, MPI_INT, &sendtype);
	MPI_Type_hvector(8, 1, sizeof(int), sendtype, &recvtype);

	MPI_Type_commit(&sendtype);
	MPI_Type_commit(&recvtype);

	/* please choose 2 processes */
	if(numtasks != 2) {
		if (rank == 0) { 
	 		printf("please choose 2 processes. \n");
		}
	}else {
		if (rank == 0) { 
	 		MPI_Send(a, 64, MPI_INT, 1, 1, MPI_COMM_WORLD); 
		} else {
			MPI_Recv(b, 1, recvtype, 0, 1, MPI_COMM_WORLD, &stat); 
			/*
			* rank= 1 
			* 1 9 17 25 33 41 49 57 
			* 2 10 18 26 34 42 50 58 
			* 3 11 19 27 35 43 51 59 
			* 4 12 20 28 36 44 52 60 
			* 5 13 21 29 37 45 53 61 
			* 6 14 22 30 38 46 54 62 
			* 7 15 23 31 39 47 55 63 
			* 8 16 24 32 40 48 56 64
			*/
			printf("rank= %d \n", rank); 

			for(i=0; i<8; i++) {
				for(j=0; j<8; j++)
					printf("%d ", b[i][j]); 
				printf("\n"); 
			}
			
		}
	}

	MPI_Type_free(&sendtype);
	MPI_Type_free(&recvtype);
	ierr = MPI_Finalize();
}