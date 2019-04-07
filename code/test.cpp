#include <limits>
#include <iostream>
#include <math.h>

using namespace std;

int main()
{
    float a = numeric_limits<float>::infinity();
    cout<< a <<endl;
    cout<< a+a <<endl;
    cout<< a+a+a <<endl;
    bool res = 10<a;
    cout << res << endl;
}