
__global__
void sposv_batched_kernel(int n, int threads_per_block, float *dA);

////////////////////////////////////////////////////////////////////////////////
extern "C"
void sposv_batched(int n, int batch, float *dA, cudaStream_t stream)
{
    /*
    * Each block has 1024 threads. The default block size is 50000. 
    * If the batch is larger than block size, call sposv_batched_kernel several times accordingly.
    */
    int threads_per_block = 1024;
    int block = 50000;
    int num_blocks;
    int size = 0;
    int count = ((batch%block) == 0) ? (batch/block) : (batch/block + 1);

    for (int i = 0; i < count; i++)
    {
        num_blocks = (batch>block) ? block : batch;
        batch -= block;
        dim3 dimGrid(num_blocks, 1, 1);
        dim3 dimBlock(threads_per_block, 1, 1);
        sposv_batched_kernel<<<dimGrid, dimBlock, 0, stream>>>(n, threads_per_block, &dA[size*n*n]);
        size += num_blocks;
    }
}

////////////////////////////////////////////////////////////////////////////////
__global__
void sposv_batched_kernel(int N, int threads_per_block, float *dA)
{   
    int k;
    int i;
    int n;
    int s;
    int blockid = blockIdx.x;
    // get the address for one matrix
    float *pA = &dA[blockid*N*N];
    int threadid = threadIdx.x;

    int matrix_size = N*N;

    int repeat = ((matrix_size % threads_per_block) == 0) ? (matrix_size / threads_per_block) : (matrix_size / threads_per_block + 1);

    // Single Cholesky factorization.
    for (k = 0; k < N; k++) {

        // Panel factorization.
        // Only one thread in the block do the sqrtf function
        if(threadid == 0) {
            pA[k*N+k] = sqrtf(pA[k*N+k]);
        }
        
        __syncthreads();

        // If N is larger than thread size, each thread may compute more than onece accordingly.
        if(N > threads_per_block) {
            for(i=0; i<repeat; i++) {
                if((threadid > k) && (threadid < N)){
                    pA[k*N+threadid] /= pA[k*N+k];
                }
                threadid += threads_per_block;
            }
            //reset threadid
            threadid = threadIdx.x;
        }else {
            if((threadid > k) && (threadid < N)){
                pA[k*N+threadid] /= pA[k*N+k];
            }
        }
       
        __syncthreads();

        // Update of the trailing submatrix.
        // If matrix size is larger than thread size, one loop is not parallelled.
        if (matrix_size > threads_per_block )
        {
            if(N > threads_per_block) {
                for(i=0; i<repeat; i++) {
                    if((threadid > k) && (threadid < N)) {
                        for (s = threadid; s < N; s++)
                            pA[threadid*N+s] -= (pA[k*N+threadid]*pA[k*N+s]);
                    }
                    threadid += threads_per_block;
                }
                //reset threadid
                threadid = threadIdx.x;
            }else {
                if((threadid > k) && (threadid < N)) {
                    for (s = threadid; s < N; s++)
                        pA[threadid*N+s] -= (pA[k*N+threadid]*pA[k*N+s]);
                }
            } 
        }else {
            n = threadid / N;
            s = threadid % N;

            if((n > k) && (n < N)) {
                if((s >= n) && (s < N))
                    pA[n*N+s] -= (pA[k*N+s]*pA[k*N+n]);
            }
        }
        __syncthreads();
    }
}
