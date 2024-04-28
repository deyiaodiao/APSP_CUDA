#include "utils.h"
/*
recursive call cuda kernel
-----
|A B|
|C D|
-----
*/
__global__ void min_plus(float* A, float* B, float* C, int a_height, int a_width, int b_width, int total_width,
    int a_startx, int a_starty, int b_startx, int b_starty, int c_startx, int c_starty, bool add);

__global__ void single_floyd(float*A, int m_size, int start_x, int start_y, int total_width);

__global__ void gpu_floyd_d(float* A, int m_size, int k);

void gpu_floyd(float* A, int m_size);

void recursive_apsp(float* in_dist_d, int m_size, int start_x, int start_y, int total_width);

void cuda_apsp(float* in_dist_d, int m_size)
{
    
    // gpu_floyd(in_dist_d, m_size);
    recursive_apsp(in_dist_d, m_size, 0, 0, m_size);
}

void gpu_floyd(float* A, int m_size)
{
    int grid_size = ceil(float(m_size)/float(BLOCKSIZE));
    dim3 blocks(grid_size, grid_size);
    dim3 threads(BLOCKSIZE, BLOCKSIZE);
    for (int k=0; k<m_size; k++)
        gpu_floyd_d<<<blocks, threads>>> (A, m_size, k);
    return;
}

__global__ void gpu_floyd_d(float* A, int m_size, int k)
{
    int dx = blockIdx.x;    int dy = blockIdx.y;
    int tx = threadIdx.x;   int ty = threadIdx.y;
    
    int Row = dy*BLOCKSIZE + ty;
    int Col = dx*BLOCKSIZE + tx;

    if (Row<m_size && Col<m_size){
        A[Row*m_size + Col] = fminf(A[Row*m_size + k] + A[k*m_size+ Col], A[Row*m_size + Col]);
    }
}
/*
do in-place operation on in_dist_d
*/
void recursive_apsp(float* in_dist_d, int m_size, int start_x, int start_y, int total_width)
{
    if (m_size<=BLOCKSIZE){
        // for matrix small than blocksize we use floyd-warshall using one block
        // fast for small graph
        printf("small graph\n");
        dim3 blocks(1, 1);
        dim3 threads(BLOCKSIZE, BLOCKSIZE);
        single_floyd <<<blocks, threads>>> (in_dist_d, m_size, start_x, start_y, total_width);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
                printf("CUDA error: %s\n", cudaGetErrorString(error));
    // 进行适当的错误处理
}

        return;
    }
    else{
        int new_size = ceil(m_size/2);  // split matrix into A,B,C,D
        int a_startx = start_x, a_starty = start_y;
        int b_startx = start_x, b_starty = start_y+new_size;
        int c_startx = start_x+new_size, c_starty = start_y;
        int d_startx = start_x+new_size, d_starty = start_y+new_size;
        int a_height = new_size, a_width = new_size;
        int b_height = new_size;
        int b_width = m_size - new_size;
        int c_height = m_size - new_size;
        int c_width = new_size;
        int d_height = m_size - new_size, d_width = m_size - new_size;

        int grid_size = ceil(float(new_size)/float(BLOCKSIZE));
        dim3 blocks(grid_size, grid_size);
        dim3 threads(BLOCKSIZE, BLOCKSIZE);
        // A = A*
        //minplus(B,A,B,add=0); // B = AB
        //minplus(C,C,A,add=0); // C = CA
        //minplus(D,C,B,add=1); // D = D+CB
        recursive_apsp(in_dist_d, new_size, start_x, start_y, total_width);
        min_plus <<<blocks, threads>>> (in_dist_d,in_dist_d,in_dist_d, b_height, b_width, a_width, 
            total_width, b_startx, b_starty, a_startx, a_starty, b_startx, b_starty,false);
        min_plus <<<blocks, threads>>> (in_dist_d,in_dist_d,in_dist_d, c_height, c_width, c_width, 
            total_width, c_startx, c_starty, c_startx, c_starty, a_startx, a_starty,false);
        min_plus <<<blocks, threads>>> (in_dist_d,in_dist_d,in_dist_d, d_height, d_width, c_width, 
            total_width, d_startx, d_starty, c_startx, c_starty, b_startx, b_starty,true);
        
        // D = D*
        //minplus(B,B,D,add=0); //B = BD
        //minplus(C,D,C,add=0); //C = DC
        //minplus(A,B,C,add=1); //A = A+BC
        recursive_apsp(in_dist_d, m_size-new_size, start_x+new_size, start_y+new_size, total_width);
        min_plus <<<blocks, threads>>> (in_dist_d,in_dist_d,in_dist_d, b_height, b_width, b_width, 
            total_width, b_startx, b_starty, b_startx, b_starty, d_startx, d_starty,false);
        min_plus <<<blocks, threads>>> (in_dist_d,in_dist_d,in_dist_d, c_height, c_width, d_width, 
            total_width, c_startx, c_starty, d_startx, d_starty, c_startx, c_starty,false);
        min_plus <<<blocks, threads>>> (in_dist_d,in_dist_d,in_dist_d, a_height, a_width, b_width, 
            total_width, a_startx, a_starty, b_startx, b_starty, c_startx, c_starty,true);
        return;
    }
}

__global__ void single_floyd(float*A, int m_size, int start_x, int start_y, int total_width)
{
    
    __shared__ float As[BLOCKSIZE][BLOCKSIZE];
    // only one block will be launched
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = ty + start_x;
    int Col = tx + start_y;
    // load A to shared memory
    if (tx<m_size && ty<m_size){
        As[ty][tx] = A[Row*total_width + Col];
    }
    else As[ty][tx] = finf;
    
    __syncthreads();
    //floyd
    for (int k=0; k<m_size; k++)
    {
        
        As[ty][tx] = fminf(As[ty][k]+As[k][tx], As[ty][tx]);
        
        __syncthreads();
    }
    printf("calc  As\t%f\n",As[ty][tx]);
    //save to the original A
    if (tx<m_size && ty<m_size){
        A[Row*total_width + Col] = As[ty][tx];
    }
}


__global__ void min_plus_global(float* A, float* B, float* C, int a_height, int a_width, int b_width, int total_width,
        int a_startx, int a_starty, int b_startx, int b_starty, int c_startx, int c_starty, bool add)
{
    // do we need __syncthreads() ?
    int x = blockIdx.x * blockDim.y + threadIdx.x;
    int y = blockIdx.y * blockDim.x + threadIdx.y;
    if (x<a_height && y<a_width){
        int a_index = (a_startx+x)*total_width + a_starty+y;
        float min_value = finf;  ////////////////////////
        int b_index, c_index;
        for(int i=0; i<b_width; i++)
        {
            b_index = (x+b_startx)*total_width + b_starty + i;
            c_index = (i+c_startx)*total_width + c_starty + y;
            min_value = fminf(min_value, B[b_index] + C[c_index]);
        }
        if(add){
            min_value = fminf(min_value, A[a_index]);
        }
#ifdef debug
        printf("%f ", min_value);
#endif
        A[a_index] = min_value;
    }
}

__global__ void min_plus(float* A, float* B, float* C, int a_height, int a_width, int b_width, int total_width,
    int a_startx, int a_starty, int b_startx, int b_starty, int c_startx, int c_starty, bool add)
{   
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];
    __shared__ float Cs[BLOCKSIZE][BLOCKSIZE];
    
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by*BLOCKSIZE + ty;     // Row first to make it more cache friendly
    int Col = bx*BLOCKSIZE + tx;
    
    float min_value  = finf;    //init with finf (not zero...)

    int out_loop = ceilf( float(b_width)/float(BLOCKSIZE) );
    
    for (int m=0; m<out_loop; m++){
        int Bs_index = (Row+b_startx)*total_width + m*BLOCKSIZE+tx+b_starty;
        int Cs_index = (ty+m*BLOCKSIZE+c_startx)*total_width + Col+c_starty;

        if (Row<a_height && (m*BLOCKSIZE+tx)<b_width )
            Bs[ty][tx] = B[Bs_index];
        else
            Bs[ty][tx] = finf;
        
        if ( (ty+m*BLOCKSIZE)<b_width && Col<a_width )
            Cs[ty][tx] = C[Cs_index];
        else
            Cs[ty][tx] = finf;
        __syncthreads();

        for (int k=0; k<BLOCKSIZE; k++)
            min_value = fminf(Bs[ty][k]+Cs[k][tx], min_value);
        __syncthreads();
    }
    if (Row<a_height && Col<a_width){
        int A_index = (a_startx+Row)*total_width + a_starty+Col;
        if (add)
            min_value = fminf(A[A_index], min_value);
        A[A_index] = min_value;
    }
}