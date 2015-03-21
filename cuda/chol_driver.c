
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <math.h>
#include <lapacke.h>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
//  Make a CUDA call and check the return status.
#define CUDA_CALL(status) cuda_call_line_file(status,__LINE__,__FILE__)
void cuda_call_line_file(cudaError_t status, const int line, const char *file)
{
    if (status != cudaSuccess) {
        printf("%s, line: %d, file: %s\n", cudaGetErrorString(status), line, file);
        exit(EXIT_FAILURE);
    }
}

////////////////////////////////////////////////////////////////////////////////
//  Make a LAPACK call and check the return status.
#define LAPACK_CALL(status) lapack_call_line_file(status,__LINE__,__FILE__)
void lapack_call_line_file(lapack_int status, const int line, const char *file)
{
    if (status != 0) {
        printf("LAPACK ERROR %ld, line: %d, file: %s\n", (long)status, line, file);
        exit(EXIT_FAILURE);
    }
}

////////////////////////////////////////////////////////////////////////////////
//  Print a matrix located in the host (CPU) memory.
void print_host_matrix(int M, int N, float *A)
{
    int m;
    int n;
    printf("\n");
    for (m = 0; m < M; m++) {
        for (n = 0; n < N; n++)
            printf("%8.2lf", A[n*M+m]);
        printf("\n");
    }
}

////////////////////////////////////////////////////////////////////////////////
//  Print a matrix located in the device (GPU) memory.
void print_device_matrix(int M, int N, float *A)
{
    float *B = (float*)malloc(M*N*sizeof(float));
    assert(B != NULL);
    CUDA_CALL(cudaMemcpy(B, A, M*N*sizeof(float), cudaMemcpyDeviceToHost));
    int m;
    int n;
    printf("\n");
    for (m = 0; m < M; m++) {
        for (n = 0; n < N; n++)
            printf("%8.2lf", B[n*M+m]);
        printf("\n");
    }
    free(B);
}

////////////////////////////////////////////////////////////////////////////////
//  Compare two matrices located in the host (CPU) memory.
//  Print a "." for a match (within some tolerance).
//  Print a "#" for a mismatch.
void diff_host_matrix(int M, int N, float *A, float *B)
{
    int m;
    int n;
    printf("\n");
    for (m = 0; m < M; m++) {
        printf("\n");
        for (n = 0; n < N; n++) {
            float a = A[n*M+m];
            float b = B[n*M+m];
            if (fabs(a-b) < 0.000001f)
                printf(".");
            else
                printf("#");                
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
void sposv_batched(int n, int batch, float *dA, cudaStream_t stream);

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
    // Create a CUDA stream.
    cudaStream_t stream;
    CUDA_CALL(cudaStreamCreate(&stream));

    /**
     *  Set up the input.
     */
    assert(argc == 3);
    int N = atoi(argv[1]);
    int batch = atoi(argv[2]);

    float *hA = (float*)malloc(N*N*batch*sizeof(float));
    assert(hA != NULL);
    lapack_int seed[] = {0, 1, 2, 3};
    LAPACK_CALL(LAPACKE_slarnv(2, seed, N*N*batch, hA));
    int i;
    int b;
    for (b = 0; b < batch; b++)
        for (i = 0; i < N; i++)
            hA[b*N*N+i*N+i] += (float)N;

    print_host_matrix(N, N, hA);
    /**
     *  Solve on device.
     */
    float *dA;
    CUDA_CALL(cudaMalloc((void**)&dA, N*N*batch*sizeof(float)));
    CUDA_CALL(cudaMemcpy(dA, hA, N*N*batch*sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start;
    cudaEvent_t stop;  
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    CUDA_CALL(cudaEventRecord(start, stream));

    sposv_batched(N, batch, dA, stream);

    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaEventRecord(stop, stream));

    /**
     *  Solve on host and check results.
     */
    for (b = 0; b < batch; b++)
        LAPACK_CALL(LAPACKE_spotrf(LAPACK_COL_MAJOR, 'L', N, hA+b*N*N, N));
    
    float *hC = (float*)malloc(N*N*batch*sizeof(float));
    assert(hC != NULL);
    CUDA_CALL(cudaMemcpy(hC, dA, N*N*batch*sizeof(float), cudaMemcpyDeviceToHost));

    print_host_matrix(N, N, hA);
    print_host_matrix(N, N, hC);

    int n;
    int m;
    float max_error = 0.0f;
    for (b = 0; b < batch; b++)
        for (n = 0; n < N; n++)
            for (m = n; m < N; m++) {
                float a = hA[b*N*N+n*N+m];
                float c = hC[b*N*N+n*N+m];
                if (fabs(a-c) > max_error)
                    max_error = fabs(a-c);
            }
    printf("\nERROR:\t%f", max_error);

    /**
     *  Report performance.
     */
    printf("\n");
    float elapsed;
    double flops = 1.0f/3.0f*N*N*N*batch;
    CUDA_CALL(cudaEventElapsedTime(&elapsed, start, stop));
    printf("\nGLOPS:\t%lf\n\n", flops/elapsed/1000000.0);
}
