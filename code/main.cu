#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

int main(int argc, char* argv[])
{   
    char* graphfile = argv[1];
    int m_size = atoi(argv[2]);

    // load graph adjacent matrix and modify zero to float max
    float* in_dist;
    float* out_dist;
    in_dist = loadGraph(graphfile, m_size);
    for(int i=0; i<m_size*m_size; i++){
        if (in_dist[i] < 1e-10)
            in_dist[i] = finf;
    }

    float* in_dist_d;
    float* out_dist_d;
    out_dist_d = (float*) malloc(sizeof(float)*m_size*m_size);
    cudaMalloc((void**)&in_dist_d, sizeof(float)*m_size*m_size);
    cudaMemcpy(in_dist_d, in_dist, sizeof(float)*m_size*m_size,
                cudaMemcpyHostToDevice);
    
    cudaEvent_t start_GPU, stop_GPU;
    float gpu_time;
    cudaEventCreate(&start_GPU);
    cudaEventCreate(&stop_GPU);
    cudaEventRecord(start_GPU, 0);

    // cuda compute
    cuda_apsp(in_dist_d, m_size);
    
    cudaEventRecord(stop_GPU, 0);
    cudaEventSynchronize(stop_GPU);
    cudaEventElapsedTime(&gpu_time, start_GPU, stop_GPU);
    cudaEventDestroy(start_GPU);
    cudaEventDestroy(stop_GPU);
    
    // run apsp in cpu and record time
    out_dist = (float*) malloc(sizeof(float)*m_size*m_size);
    auto start = chrono::steady_clock::now();
    computeGold(out_dist, in_dist, m_size);
    auto end = chrono::steady_clock::now();
    auto elapsed = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    // copy result to host
    cudaMemcpy(out_dist_d, in_dist_d, sizeof(float)*m_size*m_size, 
                cudaMemcpyDeviceToHost);
    // verity the correctness of gpu result
    bool res = correct(out_dist, out_dist_d, m_size, 0.0001);

    if (res) cout<<"test pass!"<<endl;
    else cout<<"test fail!!"<<endl;

    cout << "CPU time " << elapsed.count() << " milliseconds." << endl;
    cout << "GPU time " << gpu_time << " milliseconds." << endl;
    // free memory
    free(in_dist); in_dist=NULL;
    free(out_dist); out_dist=NULL;
    free(out_dist_d); out_dist_d=NULL;
    cudaFree(in_dist_d); in_dist_d=NULL;
    return 0;
}