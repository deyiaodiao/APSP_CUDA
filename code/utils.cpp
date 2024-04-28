#include "utils.h"
#include<cmath>

using namespace std;

float min(float a, float b)
{   
    return (a < b? a: b);
}

float add(float a,float b)
{
	return a+b;
}

float* 
loadGraph(char* filename, int size)
{
    float* in_dist;
    in_dist = (float*) malloc(sizeof(float)*size*size);
    cout<<"in_dist size: "<<size*size<<endl;
    ifstream graphfile;
    graphfile.open(filename);
    float weight;
    int i=0;
    while (graphfile >> weight){
        in_dist[i++] = weight;
    }
    int edge = 0;
    for(int i=0; i<size; i++){
        for(int j=0; j<size; j++){
            if ( (in_dist[i*size+j] < 1e-10) && (i!=j) )
                in_dist[i*size+j] = finf;
            else edge+=1;
        }
    }
#ifdef debug
    for (int i=0; i<size; i++){
        for (int j=0; j<size; j++)
            cout<<in_dist[i*size+j] << ' ';
        cout<<endl;
    }
#endif
    cout<<"average degree: "<<edge/float(size)<<endl;
    return in_dist;
}


void 
printMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << "\t"; // 打印矩阵元素，\t 为制表符，用于对齐
        }
        std::cout << std::endl; // 打印完一行后换行
    }
}


bool 
correct(float* out_dist, float* out_dist_d, int size, float eps)
{
    bool res = true;
    for (int i=0; i<size; i++)
        for (int j=0; j<size; j++)
        {
            if ( abs(out_dist[i*size+j] - out_dist_d[i*size+j]) > eps )
            {
#ifdef debug
                cout<< "row " << i << " col"<< j<< " doesn't match: ";
                cout<< out_dist[i*size+j] << " vs "<< out_dist_d[i*size+j] <<endl;
#endif
                res = false;
            }
        }
#ifdef debug
    for (int i=0; i<size; i++){
        for (int j=0; j<size; j++)
            cout<<out_dist[i*size+j] << ' ';
        cout<<endl;
    }
    cout<<"------------------"<<endl;
    for (int i=0; i<size; i++){
        for (int j=0; j<size; j++)
            cout<<out_dist_d[i*size+j] << ' ';
        cout<<endl;
    }
#endif
    return res;
}

// inplace floyd Warshall all pair shortest path algorighm as gold     
void
computeGold(float *out_dist, float *in_dist, int m_size)
{
    for (int j = 0; j < m_size; ++j)
        for (int i = 0; i < m_size; ++i)
            out_dist[j*m_size + i] = in_dist[j*m_size + i];

    for (int k = 0; k < m_size; ++k)
        for (int j = 0; j < m_size; ++j)
            for (int i = 0; i < m_size; ++i)
            {
                // cout<<out_dist[j*m_size + i]<<'\t'<<add(out_dist[k*m_size + i],out_dist[j*m_size + k])<<endl;
                out_dist[j*m_size + i] = min (out_dist[j*m_size + i], add(out_dist[k*m_size + i],out_dist[j*m_size + k]));
                // out_dist[j*m_size + i] = min (out_dist[j*m_size + i], out_dist[k*m_size + i]+out_dist[j*m_size + k]);
                // cout<<out_dist[j*m_size + i]<<endl;
            }
               
}