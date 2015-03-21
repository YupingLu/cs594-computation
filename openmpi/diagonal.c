/*
* hw 3 Yuping Lu
* MPI datatype: a datatype describing a matrix diagonal
*/
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "mpi.h"

int main (int argc, char *argv[]) {
	int ierr, rank, numtasks;
	int i;
	MPI_Datatype newtype;
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

	/* store the diagonal */
	int b[8];

	typedef struct
	{
		int v[8];
	} one_by_cacheline;

	ierr = MPI_Init ( &argc, &argv );
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
	ierr = MPI_Comm_size(MPI_COMM_WORLD, &numtasks); 

	/*new datatype for matrix diagonal */
	MPI_Datatype array_of_types[] = { MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_UB }; 
	MPI_Aint start, array_of_displs[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	int array_of_lengths[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	one_by_cacheline c[9];

	MPI_Get_address( &c[0], &(start) ); 
	MPI_Get_address( &c[0].v[0], &(array_of_displs[0]) ); 
	MPI_Get_address( &c[1].v[1], &(array_of_displs[1]) ); 
	MPI_Get_address( &c[2].v[2], &(array_of_displs[2]) );
	MPI_Get_address( &c[3].v[3], &(array_of_displs[3]) );
	MPI_Get_address( &c[4].v[4], &(array_of_displs[4]) );
	MPI_Get_address( &c[5].v[5], &(array_of_displs[5]) );
	MPI_Get_address( &c[6].v[6], &(array_of_displs[6]) );
	MPI_Get_address( &c[7].v[7], &(array_of_displs[7]) );
	MPI_Get_address( &c[8], &(array_of_displs[8]) );

	for( i = 0; i < 9; i++ ) 
		array_of_displs[i] -= start;

	MPI_Type_create_struct(9, array_of_lengths, array_of_displs, array_of_types, &newtype);

	MPI_Type_commit(&newtype);

	/* please choose 2 processes */
	if(numtasks != 2) {
		if (rank == 0) { 
	 		printf("please choose 2 processes. \n");
		}
	}else {
		if (rank == 0) { 
	 		MPI_Send(a, 1, newtype, 1, 1, MPI_COMM_WORLD); 
		} else {
			MPI_Recv(b, 8, MPI_INT, 0, 1, MPI_COMM_WORLD, &stat); 
			/*
			* output
			* b= 1 10 19 28 37 46 55 64
			*/
			printf("rank= %d  b= %d %d %d %d %d %d %d %d\n", 
			rank,b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7]); 
		}
	}

	MPI_Type_free(&newtype);
	ierr = MPI_Finalize ();
}



