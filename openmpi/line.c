# include <stdlib.h>
# include <stdio.h>
# include <time.h>

# include "mpi.h"

int main (int argc, char *argv[]) {
	int ierr, rank, numtasks;
	int i;
	MPI_Datatype newtype;
	MPI_Status stat;

	int a[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	int b[3];

	typedef struct
	{
		int v[4];
	} one_by_cacheline;

	ierr = MPI_Init ( &argc, &argv );
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
	ierr = MPI_Comm_size(MPI_COMM_WORLD, &numtasks); 

	MPI_Datatype array_of_types[] = { MPI_INT, MPI_INT, MPI_INT, MPI_UB }; 
	MPI_Aint start, array_of_displs[] = { 0, 0, 0, 0 };
	int array_of_lengths[] = { 1, 1, 1, 1 };
	one_by_cacheline c[4];

	MPI_Get_address( &c[0], &(start) ); 
	MPI_Get_address( &c[0].v[1], &(array_of_displs[0]) ); 
	MPI_Get_address( &c[1].v[1], &(array_of_displs[1]) ); 
	MPI_Get_address( &c[2].v[1], &(array_of_displs[2]) ); 
	MPI_Get_address( &c[3], &(array_of_displs[3]) );

	for( i = 0; i < 4; i++ ) 
		array_of_displs[i] -= start;

	MPI_Type_create_struct(4, array_of_lengths, array_of_displs, array_of_types, &newtype);

	MPI_Type_commit(&newtype);

 	if (rank == 0) { 
 		MPI_Send(a, 1, newtype, 1, 1, MPI_COMM_WORLD); 
	} else {
		MPI_Recv(b, 3, MPI_INT, 0, 1, MPI_COMM_WORLD, &stat); 
		printf("rank= %d  b= %d %d %d\n", 
		rank,b[0],b[1],b[2]); 
	}

	MPI_Type_free(&newtype);
	ierr = MPI_Finalize ();
}



