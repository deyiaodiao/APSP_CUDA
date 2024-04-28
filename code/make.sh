nvcc -Xptxas -O1 -std=c++17 main.cu utils.cpp recursive_apsp.cu -o main
./main