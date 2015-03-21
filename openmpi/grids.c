/*
* hw 3 Yuping Lu
* Process Grids
*/
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"

int main (int argc, char *argv[]) {
	int rank, size, P=3, Q=3, p, q, mb, nb, i, j, t, row_id, col_id; 
	MPI_Status stat;
	MPI_Comm row_comm, col_comm; 

	/* input 2d matrix a*/
	int a[6][6] = {
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12,
		13, 14, 15, 16, 17, 18,

		19, 20, 21, 22, 23, 24,
		25, 26, 27, 28, 29, 30, 
		31, 32, 33, 34, 35, 36
	};

	MPI_Init (&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
	MPI_Comm_size(MPI_COMM_WORLD, &size); 

	p = rank / Q; 
	q = rank % Q; 
	mb = 6 / P;
	nb = 6 / Q;
	int b[mb][nb], token[mb][nb];

	for(i=0; i<mb; i++)
		for(j=0; j<nb; j++)
			b[i][j] = a[P*i+p][Q*j+q];

	/* row and column communicators*/	
	MPI_Comm_split(MPI_COMM_WORLD, p, q, &row_comm); 
	MPI_Comm_split(MPI_COMM_WORLD, q, p, &col_comm); 
	MPI_Comm_rank(row_comm, &row_id);
	MPI_Comm_rank(col_comm, &col_id);

	/*row communication*/
	for(i=0; i<Q; i++) {
		if(row_id == 2) {
			MPI_Send(b, mb*nb, MPI_INT, 0, p, row_comm);
		}else {
			MPI_Send(b, mb*nb, MPI_INT, row_id+1, p, row_comm);
		}
		if(row_id == 0) {
			MPI_Recv(token, mb*nb, MPI_INT, Q-1, p, row_comm, &stat);
			for(t=0; t<mb; t++)
				for(j=0; j<nb; j++)
					b[t][j] = token[t][j];
		}else {
			MPI_Recv(token, mb*nb, MPI_INT, row_id-1, p, row_comm, &stat); 
			for(t=0; t<mb; t++)
				for(j=0; j<nb; j++)
					b[t][j] = token[t][j];
		}
	}

	/* print the first and second token after the row iteration */
	if(rank == 0) {
		printf("Row iteration finishes. \n"); 
		printf("rank[%d], 1st token: %d, 2nd token: %d \n",rank,b[0][0], b[0][1]); 
	}

	/*column communication*/
	for(i=0; i<P; i++) {
		if(col_id == 2) {
			MPI_Send(b, mb*nb, MPI_INT, 0, q, col_comm);
		}else {
			MPI_Send(b, mb*nb, MPI_INT, col_id+1, q, col_comm);
		}
		if(col_id == 0) {
			MPI_Recv(token, mb*nb, MPI_INT, P-1, q, col_comm, &stat);
			for(t=0; t<mb; t++)
				for(j=0; j<nb; j++)
					b[t][j] = token[t][j];
		}else {
			MPI_Recv(token, mb*nb, MPI_INT, col_id-1, q, col_comm, &stat); 
			for(t=0; t<mb; t++)
				for(j=0; j<nb; j++)
					b[t][j] = token[t][j];
		}
	}

	/* print the first and second token after the column iteration */
	if(rank == 0) {
		printf("Column iteration finishes. \n"); 
		printf("rank[%d], 1st token: %d, 2nd token: %d \n",rank,b[0][0], b[0][1]); 
	}

	MPI_Finalize();
}