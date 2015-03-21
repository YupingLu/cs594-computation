/*
* hw 4 Yuping Lu
* OpenSHMEM: Stencil code with Ghost cell
*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <shmem.h>

#ifndef MAX
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#endif

// nreduce
static const int nred = 1;
double diff, convergence;

double btime, etime;

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
	int n, m, i, j, it, t, gap, block;
	double** local_a;
	double** local_b;
	int nprocs, me;
	double* left;
	double* right;
	double* src;
	double* dst;

	long *pSync;
	
 	double *pWrk;
	int pWrk_size;

	/*
	* Configure input matrix a.
	* If a is s*s, set b (s-1)*(s-1) and n = s-2.
	*/
	double a[10][10] = {

		0.81472,0.15761,0.65574,0.70605,0.43874,0.27603,0.75127,0.84072,0.35166,0.075854,
		0.90579,0.97059,0.035712,0.031833,0.38156,0.6797,0.2551,0.25428,0.83083,0.05395,
		0.12699,0.95717,0.84913,0.27692,0.76552,0.6551,0.50596,0.81428,0.58526,0.5308,
		0.91338,0.48538,0.93399,0.046171,0.7952,0.16261,0.69908,0.24352,0.54972,0.77917,
		0.63236,0.80028,0.67874,0.097132,0.18687,0.119,0.8909,0.92926,0.91719,0.93401,
		0.09754,0.14189,0.75774,0.82346,0.48976,0.49836,0.95929,0.34998,0.28584,0.12991,
		0.2785,0.42176,0.74313,0.69483,0.44559,0.95974,0.54722,0.1966,0.7572,0.56882,
		0.54688,0.91574,0.39223,0.3171,0.64631,0.34039,0.13862,0.25108,0.75373,0.46939,
		0.95751,0.79221,0.65548,0.95022,0.70936,0.58527,0.14929,0.61604,0.38045,0.011902,
		0.96489,0.95949,0.17119,0.034446,0.75469,0.22381,0.25751,0.47329,0.56782,0.33712
	};

	double b[9][9];

	n = 8;

	start_pes (0);

	nprocs = shmem_n_pes (); 
	me = shmem_my_pe ();

	if(nprocs > n) {
		if(me == 0)
			printf ("The process number should not be larger than n.\n");
		exit(1);
	}

	pWrk_size = MAX (nred/2 + 1, _SHMEM_REDUCE_MIN_WRKDATA_SIZE);
	pWrk = (double *) shmalloc (pWrk_size);
	assert (pWrk != NULL);

	pSync = (long *) shmalloc (SHMEM_REDUCE_SYNC_SIZE);
	assert (pSync != NULL);
	for (i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i += 1)
	{
		pSync[i] = _SHMEM_SYNC_VALUE;
	}

	

	block = n/nprocs;
	m = block;
	gap = n-nprocs*m;
	if(me < gap)
		m++;

	left = (double *) shmalloc (n * sizeof(*left));
	right = (double *) shmalloc (n * sizeof(*right));
	dst = (double *) shmalloc (n * sizeof(*dst));

	if ((src=(double*) malloc((n)*sizeof(double)))==NULL) {
		printf ("Error allocating memory for src\n");
		exit(1);
	}

	//allocate the memory for local A
	if ((local_a=(double**) malloc((n+2)*sizeof(double*)))==NULL) {
		printf ("Error allocating memory for local_a\n");
		exit(1);
	}
	for (i=0; i<(n+2); i++) {
		if ((local_a[i]=(double*) calloc(m+2, sizeof(double)))==NULL) {
			printf ("Error allocating memory for local_a[%d]\n",i);
			exit(1);
		}
	}

	//initialize local A
	if(block != m) {
		for(t=0,j=me*m; j<(me*m+m+2); j++, t++)
			for(i=0; i<(n+2); i++)
				local_a[i][t] = a[i][j];
	}else {
		for(t=0,j=(me*m+gap); j<(me*m+gap+m+2); j++, t++)
			for(i=0; i<(n+2); i++)
				local_a[i][t] = a[i][j];
	}

	//allocate the memory for local B
	if ((local_b=(double**) malloc((n+1)*sizeof(double*)))==NULL) {
		printf ("Error allocating memory for local_b\n");
		exit(1);
	}
	for (i=0; i<(n+1); i++) {
		if ((local_b[i]=(double*) calloc(m+1, sizeof(double)))==NULL) {
			printf ("Error allocating memory for local_b[%d]\n",i);
			exit(1);
		}
	}

	btime = get_cur_time();

	it = 0;
	convergence = 1.0;
	//start the computation here
	while(convergence > 1.0e-2 && it < 100) {

		diff = 0.0;
		for(j=1; j<(m+1); j++)
			for(i=1; i<(n+1); i++) {
				local_b[i][j] = 0.25*(local_a[i-1][j] + local_a[i+1][j] + local_a[i][j-1] + local_a[i][j+1]);
				diff += (local_b[i][j]-local_a[i][j])*(local_b[i][j]-local_a[i][j]);
			}

		//copy results from local_b back to local_a
		for(j=1; j<(m+1); j++)
			for(i=1; i<(n+1); i++)
				local_a[i][j] = local_b[i][j];

		//send results to other pes
		if(me > 0) {
			for(i=1; i<(n+1); i++) {
				src[i-1] = local_b[i][1];
			}
			shmem_double_put(right, src, n, me-1);
		}

		shmem_barrier_all ();

		if(me < (nprocs-1)) {
			for(i=1; i<(n+1); i++) {
				src[i-1] = local_b[i][m];
			}
			shmem_double_put(left, src, n, me+1);
		}

		shmem_barrier_all ();

		//receive results to other pes
		if(me > 0) {
			for(i=1; i<(n+1); i++) {
				local_a[i][0] = left[i-1];
			}
		}
		if(me < (nprocs-1)) {
			for(i=1; i<(n+1); i++) {
				local_a[i][m+1] = right[i-1];
			}
		}

		shmem_barrier_all ();

		//add all diff to convergence
		shmem_double_sum_to_all (&convergence, &diff, nred, 0, 0, nprocs, pWrk, pSync);

		convergence = sqrt(convergence);
		it++;

		shmem_barrier_all ();
	}

	//Gather all results to process 0
	for(i=0; i<n; i++) {
		if(block != m) {
			shmem_double_put(&(dst[me*m]), &(local_a[i+1][1]), m, 0);
		}else {
			shmem_double_put(&(dst[me*m+gap]), &(local_a[i+1][1]), m, 0);
		}
		shmem_barrier_all ();

		if(me == 0) {
			for(j=0; j<n; j++)
				a[i+1][j+1] = dst[j];
		}
	}
	
	etime = get_cur_time();

	//print the matrix
	if(me == 0) {
		fprintf(stderr, "Elapsed time: %lf seconds.\n", etime-btime);
		printf("iterations: %d \n", it);
		for(i=0; i<n+2; i++) {
			for(j=0; j<n+2; j++) {
				printf("%lf ", a[i][j]);
			}
			printf("\n");
		}
	}
	
	shfree (pSync);
	shfree (pWrk);
	shfree (left);
	shfree (right);
	free(dst);
	free(src);
	free(local_a);
	free(local_b);

	return 0;
}



