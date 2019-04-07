#include "utils.h"

#define debug
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
    ifstream graphfile;
    graphfile.open(filename);
    float weight;
    int i=0;
    while (graphfile >> weight){
        in_dist[i++] = weight;
    }
    return in_dist;
}

bool 
correct(float* out_dist, float* out_dist_d, int size, float eps)
{
    bool res = true;
    for (int i=0; i<size; i++)
        for (int j=0; j<size; j++)
        {
            if (out_dist[i*size+j] != out_dist_d[i*size+j])
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
computeGold(float * out_dist, float * in_dist, int M_size)
{
    for (int j = 0; j < M_size; ++j)
        for (int i = 0; i < M_size; ++i)
            out_dist[j*M_size + i] = in_dist[j*M_size + i];

    for (int k = 0; k < M_size; ++k)
        for (int j = 0; j < M_size; ++j)
            for (int i = 0; i < M_size; ++i)
                out_dist[j*M_size + i] = min (out_dist[j*M_size + i], add(out_dist[k*M_size + i],out_dist[j*M_size + k]));
}