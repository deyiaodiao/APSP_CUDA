#pragma once
#include <limits>
#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>

using namespace std;

#define BLOCKSIZE 16

//#define debug

//#define finf numeric_limits<float>::infinity()

#define finf 1e10

void cuda_apsp(float* in_dist_d, int m_size);

bool correct(float* out_dist, float* out_dist_d, int size, float);

//extern "C"
void computeGold(float * out_dist, float * in_dist, int size);

//extern "C"
float* loadGraph(char* filename, int size);