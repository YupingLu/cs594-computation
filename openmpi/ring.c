/*
* hw 3 Yuping Lu
* Ring Application: a simple application that passes a token between all the processes in one communicator.
*/
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "mpi.h"

int main (int argc, char *argv[]) {
	int ierr, rank, numtasks;
	int a;

	MPI_Status stat;

	ierr = MPI_Init (&argc, &argv);
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
	ierr = MPI_Comm_size(MPI_COMM_WORLD, &numtasks); 

 	if (rank == 0) { 
 		MPI_Send(&rank, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD); 
	} else {
		MPI_Recv(&a, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD, &stat); 
		printf("rank= %d, received token %d. \n", rank, a);
		if(rank < (numtasks-1))
			MPI_Send(&rank, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD); 
	}

	ierr = MPI_Finalize();
}