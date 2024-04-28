#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib> // 包含随机数生成函数
#include <ctime>   // 包含时间函数，用于设置随机数种子
using namespace std;

void test();
void single_test(char* graphfile, int m_size, int*, bool);
void printMatrix(float* matrix, int rows, int cols);

float* initializeMatrix( int N) {
    // 设置随机数种子
    srand(time(NULL));
    
    float* in_dist;
    in_dist = (float*) malloc(sizeof(float)*N*N);
    // 遍历矩阵每个元素
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                // 对角线元素为0
                in_dist[i * N + j] = 0.0f;
            } else {
                // 其余元素为随机值，这里简单地生成[0, 1)之间的随机值
                in_dist[i * N + j] = static_cast<float>(rand()) / RAND_MAX;
            }
        }
    }
    return in_dist;
}


int main(int argc, char* argv[])
{   
    test();
}

void test()
{
    // int n_vertex[10] = {2,10,100,200,500,1000,2000,5000,10000,20000};
    int n_vertex[3] = {2,3,4};
    char buffer[200];
    int times[2] = {0,0};
    ofstream results;
    results.open("unroll.txt");
    for (int n=0; n<sizeof(n_vertex)/sizeof(n_vertex[0]); n++){
        sprintf(buffer,"../data/graph_5_%d.txt", n_vertex[n]);
        single_test(buffer, n_vertex[n], times, false);
        results<< n_vertex[n] <<" " <<times[0]<<" "<<times[1]<<endl;
    }
}

void single_test(char* graphfile, int m_size, int* times, bool gpu_only)
{
    cout << "load: " << graphfile << endl;
    // load graph adjacent matrix and modify zero to float max
    //cpu matrix
    float* in_dist;
    // int* in_path;
    float* out_dist;
    // int* out_path;
    auto start_load = chrono::steady_clock::now();
    // in_dist = loadGraph(graphfile, m_size);
    in_dist=initializeMatrix(m_size);
    printMatrix(in_dist,m_size,m_size);
    auto end_load = chrono::steady_clock::now();
    auto elapsed_load = chrono::duration_cast<chrono::milliseconds>(end_load - start_load);
    cout<<"graph load time: "<<elapsed_load.count()<<endl;

    
    
    cudaEvent_t start_GPU, stop_GPU;
    float gpu_time;
    cudaEventCreate(&start_GPU);
    cudaEventCreate(&stop_GPU);
    cudaEventRecord(start_GPU, 0);

    float* in_dist_d;
    float* out_dist_d;
    out_dist_d = (float*) malloc(sizeof(float)*m_size*m_size);





    cudaMalloc((void**)&in_dist_d, sizeof(float)*m_size*m_size);
    cudaMemcpy(in_dist_d, in_dist, sizeof(float)*m_size*m_size,
                cudaMemcpyHostToDevice);

    // float* print_dist_d;
    // print_dist_d = (float*) malloc(sizeof(float)*m_size*m_size);
    // cudaMemcpy(print_dist_d, in_dist_d, sizeof(float)*m_size*m_size, 
    //     cudaMemcpyDeviceToHost);
    // cout << "GPU input -------------------------------------------" << endl;
    //     printMatrix(print_dist_d,m_size,m_size);


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
    
    if (!gpu_only){
        // run apsp in cpu and record time
        auto start = chrono::steady_clock::now();
        computeGold(out_dist, in_dist,  m_size);

        auto end = chrono::steady_clock::now();
        auto elapsed = chrono::duration_cast<chrono::milliseconds>(end - start);
        
        // verity the correctness of gpu result
        bool res = correct(out_dist, out_dist_d, m_size, 0.00001);
        
        if (res) cout<<"test pass!"<<endl;
        else cout<<"test fail!!"<<endl;
        //elapsed.count();
        cout << "CPU time " << elapsed.count() << " milliseconds." << endl;
        times[0] = elapsed.count();
    }
    else
        times[0] = 0.0;
    cout << "GPU time " << gpu_time << " milliseconds." << endl;
    cout << "GPU input -------------------------------------------" << endl;
    printMatrix(in_dist,m_size,m_size);
    cout << "GPU result -------------------------------------------" << endl;
    printMatrix(out_dist_d,m_size,m_size);
    // free memory

    free(in_dist);          in_dist=NULL;
    free(out_dist);         out_dist=NULL;
    free(out_dist_d);       out_dist_d=NULL;
    cudaFree(in_dist_d);    in_dist_d=NULL;

    times[1] = gpu_time;
    return;
}