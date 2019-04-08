#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

void test();
void single_test(char* graphfile, int m_size, int*);

int main(int argc, char* argv[])
{   
    test();
}

void test()
{
    int n_vertex[10] = {2,10,100,200,500,1000,2000,5000,10000,20000};
    //int n_vertex[3] = {2,10,100};
    char buffer[200];
    int times[2] = {0,0};
    ofstream results;
    results.open("results.txt");
    for (int n=0; n<sizeof(n_vertex)/sizeof(n_vertex[0]); n++){
        sprintf(buffer,"../data/graph_0_%d.txt", n_vertex[n]);
        single_test(buffer, n_vertex[n], times);
        results<< n_vertex[n] <<" " <<times[0]<<" "<<times[1]<<endl;
    }
}

void single_test(char* graphfile, int m_size, int* times)
{
    cout << "load: " << graphfile << endl;
    // load graph adjacent matrix and modify zero to float max
    float* in_dist;
    float* out_dist;
    in_dist = loadGraph(graphfile, m_size);

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
    
    // copy result to host
    cudaMemcpy(out_dist_d, in_dist_d, sizeof(float)*m_size*m_size, 
                cudaMemcpyDeviceToHost);

    out_dist = (float*) malloc(sizeof(float)*m_size*m_size);
    
    // run apsp in cpu and record time
    auto start = chrono::steady_clock::now();
    computeGold(out_dist, in_dist, m_size);
    auto end = chrono::steady_clock::now();
    auto elapsed = chrono::duration_cast<chrono::milliseconds>(end - start);
    
    // verity the correctness of gpu result
    bool res = correct(out_dist, out_dist_d, m_size, 0.0001);

    if (res) cout<<"test pass!"<<endl;
    else cout<<"test fail!!"<<endl;
    elapsed.count();
    cout << "CPU time " << elapsed.count() << " milliseconds." << endl;
    cout << "GPU time " << gpu_time << " milliseconds." << endl;
    // free memory
    free(in_dist);          in_dist=NULL;
    free(out_dist);         out_dist=NULL;
    free(out_dist_d);       out_dist_d=NULL;
    cudaFree(in_dist_d);    in_dist_d=NULL;

    times[0] = elapsed.count();
    times[1] = gpu_time;
    return;
}