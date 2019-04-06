#pragma once
#include <limits>
#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

using namespace std;

#define fmax std::numeric_limits<double>::max()

//extern "C"
void computeGold(float * out_dist, float * in_dist, int size);

//extern "C"
float* loadGraph(char* filename, int size);