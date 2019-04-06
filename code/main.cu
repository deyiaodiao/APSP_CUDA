#include "utils.h"

/*
cudaEvent_t start, stop;
float time;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(&start, 0)

call kernel;

cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
cudaEventElapseTime(&time, start, stop)
cudaEventDestroy(start);
cudaEventDestroy(end);
*/
using namespace std;

int main(int argc, char* argv[])
{   
    char* graphfile = argv[1];
    int m_size = atoi(argv[2]);

    float* in_dist;
    float* out_dist;
    in_dist = loadGraph(graphfile, m_size);
    
    for(int i=0; i<m_size*m_size; i++){
        if (in_dist[i] < 1e-10)
            in_dist[i] = fmax;
    }
    out_dist = (float*) malloc(sizeof(float)*m_size*m_size);
    
    auto start = chrono::steady_clock::now();
    computeGold(out_dist, in_dist, m_size);
    auto end = chrono::steady_clock::now();
    auto elapsed = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "CPU time " << elapsed.count() << " milliseconds." << endl;
    /*
    for(int i=0; i<m_size;i++){
        for(int j=0; j<m_size; j++)
            cout << out_dist[i*m_size+j]<<" ";
        cout << endl;
    }
    */
    free(in_dist); in_dist=NULL;
    free(out_dist); out_dist=NULL;
    return 0;
}