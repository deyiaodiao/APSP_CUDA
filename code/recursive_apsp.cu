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

void recursive_apsp(float* in_dist_d, int m_size, int start_x, int start_y, int total_width);

void cuda_apsp(float* in_dist_d, int m_size)
{
    recursive_apsp(in_dist_d, m_size, 0, 0, m_size);
}

/*
do in-place operation on in_dist_d
*/
void recursive_apsp(float* in_dist_d, int m_size, int start_x, int start_y, int total_width)
{
    if (m_size<=2)
        return;
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
        // A = A*
        //minplus(B,A,B,add=0); // B = AB
        //minplus(C,C,A,add=0); // C = CA
        //minplus(D,C,B,add=1); // D = D+CB
        recursive_apsp(in_dist_d, new_size, start_x, start_y, total_width);
        int grid_size = ceil(float(new_size)/float(BLOCKSIZE));
        dim3 blocks(grid_size, grid_size);
        dim3 threads(BLOCKSIZE, BLOCKSIZE);
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
    }
}

__global__ void min_plus(float* A, float* B, float* C, int a_height, int a_width, int b_width, int total_width,
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
            c_index = (i+a_startx)*total_width + a_starty + y;
            min_value = fminf(min_value, B[b_index] + C[c_index]);
        }
        if(add){
            min_value = fminf(min_value, A[a_index]);
        }
        A[a_index] = min_value;
    }
}