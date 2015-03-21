#include <iostream>
#include <sys/time.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>

using namespace std;

/*xGEMM("No transpose", "No transpose", M, N, K, α = −1, A, B, β = 1, C)*/

double btime, etime;
int u, v, y; //store the dimension of matrix, matrix A is u*v, matrix B is v*y

double** a;  //store the first matrix file
double** b;  //store the second matrix file
double** c;  //store the third matrix file

#define LINESIZE 100000


void readMatrixa(char* file) {
	char buf[LINESIZE];
	char * pch;
	int i, j;
	FILE *fp;

	if ((fp=fopen(file,"r"))==NULL) {
		printf ("Error opening input file %s\n",file);
		exit(1);
	}

	//read the number of data
	rewind(fp);

	fscanf(fp,"%d %d",&u, &v);

	//allocate the memory for Matrix A
	if ((a=(double**) malloc(u*sizeof(double*)))==NULL) {
		printf ("Error allocating memory for matrix a\n");
		exit(1);
	}
	
	for (i=0; i<u; i++) {
		if ((a[i]=(double*) malloc(v*sizeof(double)))==NULL) {
			printf ("Error allocating memory for matrix a[%d]\n",i);
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
	 		a[i][j++] = atof(pch);
	 		//printf("%lf\n", atof(pch));
	 		pch = strtok(NULL, " ");
		}
		i++;
	}
	
	fclose(fp);
	//free(buf);
}

void readMatrixb(char* file) {
	char buf[LINESIZE];
	char * pch;
	int i, j;
	FILE *fp;

	if ((fp=fopen(file,"r"))==NULL) {
		printf ("Error opening input file %s\n",file);
		exit(1);
	}

	//read the number of data
	rewind(fp);

	fscanf(fp,"%d %d",&v, &y);

	//allocate the memory for Matrix B
	if ((b=(double**) malloc(v*sizeof(double*)))==NULL) {
		printf ("Error allocating memory for matrix b\n");
		exit(1);
	}
	
	for (i=0; i<v; i++) {
		if ((b[i]=(double*) malloc(y*sizeof(double)))==NULL) {
			printf ("Error allocating memory for matrix b[%d]\n",i);
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
	 		b[i][j++] = atof(pch);
	 		//printf("%lf\n", atof(pch));
	 		pch = strtok(NULL, " ");
		}
		i++;
	}
	
	fclose(fp);
	//free(buf);
}

void readMatrixc(char* file) {
	char buf[LINESIZE];
	char * pch;
	int i, j;
	FILE *fp;

	if ((fp=fopen(file,"r"))==NULL) {
		printf ("Error opening input file %s\n",file);
		exit(1);
	}

	//read the number of data
	rewind(fp);

	fscanf(fp,"%d %d",&u, &y);

	//allocate the memory for Matrix C
	if ((c=(double**) malloc(u*sizeof(double*)))==NULL) {
		printf ("Error allocating memory for matrix c\n");
		exit(1);
	}
	
	for (i=0; i<u; i++) {
		if ((c[i]=(double*) malloc(y*sizeof(double)))==NULL) {
			printf ("Error allocating memory for matrix c[%d]\n",i);
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
	 		c[i][j++] = atof(pch);
	 		//printf("%lf\n", atof(pch));
	 		pch = strtok(NULL, " ");
		}
		i++;
	}
	
	fclose(fp);
	//free(buf);
}


//a simple dgemm
void xgemm(int m, int n, int k, double **a, double **b, double **c) {
	int i, j, t;

	double **mm = new double*[m];
	for(i=0; i<m; i++) {
		mm[i] = new double[n];
		for(int j=0; j<n; j++) {
			mm[i][j] = 0.0;
		}
	}

	for(i=0; i<m; i++) {
		for(t=0; t<n; t++) {
			for(j=0; j<k; j++) {
				mm[i][t] += (-1)*a[i][j]*b[j][t];
			}
		}
	}

	for(i=0; i<m; i++) {
		for(t=0; t<n; t++) {
			c[i][t] = mm[i][t] + c[i][t];
		}
	}

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

int main (int argc, char* argv[])
{
	readMatrixa(argv[1]);
	readMatrixb(argv[2]);
	readMatrixc(argv[3]);
	

	btime = get_cur_time();
	xgemm(u, y, v, a, b, c);
	etime = get_cur_time();
	cerr << "Elapsed time: " << etime-btime << " seconds." <<endl;

	for(int i=0; i<u; i++) {
		for(int t=0; t<y; t++) {
			cout << c[i][t] << " ";
		}
		cout << endl;
	}

	return 0;
}