/*
* hw 3 Yuping Lu
* PDGEMM: a parallel version of the matrix matrix multiplication using MPI and SUMMA
*/
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include "mpi.h"

/*PDGEMM("No transpose", "No transpose", N, N, N, α = 1, A, B, β = 0, C)*/

double** a;  //store the first matrix file, A
double** b;  //store the second matrix file, B
double** c;  //store the third matrix file, C
double** d;   //store the result of matrix multiplication 
int n;  //the dimension of the matrix

double btime, etime;

#define LINESIZE 100000

/* read matrix from file */
double** readMatrix(char* file) {
	char buf[LINESIZE];
	char * pch;
	int i, j;
	FILE *fp;
	double **matrix;

	if ((fp=fopen(file,"r"))==NULL) {
		printf ("Error opening input file %s\n",file);
		exit(1);
	}

	//read the number of data
	rewind(fp);

	fscanf(fp,"%d",&n);

	//allocate the memory for Matrix
	if ((matrix=(double**) malloc(n*sizeof(double*)))==NULL) {
		printf ("Error allocating memory for matrix\n");
		exit(1);
	}
	
	for (i=0; i<n; i++) {
		if ((matrix[i]=(double*) calloc(n, sizeof(double)))==NULL) {
			printf ("Error allocating memory for matrix matrix[%d]\n",i);
			exit(1);
		}
	}

	i = 0;
	rewind(fp);
	fgets(buf, LINESIZE, fp);
	while(fgets(buf, LINESIZE, fp) != NULL)
	{

	 	pch = strtok(buf, " ");
	 	j = 0;
	 	while (pch != NULL)
	 	{
	 		matrix[i][j++] = atof(pch);
	 		//printf("%lf\n", atof(pch));
	 		pch = strtok(NULL, " ");
		}
		i++;
	}
	
	fclose(fp);

	return matrix;
}

//time function, provided by Dr. Dongarra
double get_cur_time() {
  struct timeval   tv;
  struct timezone  tz;
  double cur_time;
  
  gettimeofday(&tv, &tz);
  cur_time = tv.tv_sec + tv.tv_usec / 1000000.0;
  
  return cur_time;
}

int main (int argc, char *argv[]) {
	/*assum cyclic value k is 1 now*/
	int rank, size, i, j, t; 
	int count = 1;
	MPI_Status stat;
	//MPI_Comm row_comm, col_comm;

	MPI_Init (&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
	MPI_Comm_size(MPI_COMM_WORLD, &size); 

	a = readMatrix(argv[1]);
	b = readMatrix(argv[2]);
	//c = readMatrix(argv[3]);

	//allocate the memory for Matrix C
	if ((c=(double**) malloc(n*sizeof(double*)))==NULL) {
		printf ("Error allocating memory for c\n");
		exit(1);
	}
	
	for (i=0; i<n; i++) {
		if ((c[i]=(double*) calloc(n, sizeof(double)))==NULL) {
			printf ("Error allocating memory for matrix c[%d]\n",i);
			exit(1);
		}
	}

	//allocate the memory for Matrix
	if ((d=(double**) malloc(n*sizeof(double*)))==NULL) {
		printf ("Error allocating memory for d\n");
		exit(1);
	}
	
	for (i=0; i<n; i++) {
		if ((d[i]=(double*) calloc(n, sizeof(double)))==NULL) {
			printf ("Error allocating memory for matrix d[%d]\n",i);
			exit(1);
		}
	}

	btime = get_cur_time();

	/* matrix multiplication */
	for(t=rank; t<n; t+=size) {
		for(i=0; i<n; i++) {
			for(j=0; j<n; j++) {
				c[i][j] += a[i][t]*b[t][j];
			}
		}
	}

	/* get the multiplication result from each process and sum them */
	for(i=0; i<n; i++)
		MPI_Reduce(&(c[i][0]), &(d[i][0]), n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	etime = get_cur_time();

	/* Output the matrix C */
	if(rank == 0) {
		fprintf(stderr, "Elapsed time: %lf seconds.\n", etime-btime);

		printf("The Matrix C. \n"); 
		for(i=0; i<n; i++) {
			for(j=0; j<n; j++) {
				printf("%lf ",d[i][j]); 
			}
			printf("\n"); 
		}
	}

	MPI_Finalize();
}