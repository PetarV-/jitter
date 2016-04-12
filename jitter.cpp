#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <list>
#include <string>
#include <algorithm>
#include <queue>
#include <stack>
#include <set>
#include <map>
#include <complex>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <random>

#include "mixmodel.h"

using namespace std;
typedef long long lld;

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        printf("Usage: ./jitter <model_file> <mean [N/num]> <preselect_dev? [Y/N]> <#samples>\n");
        return -1;
    }
    
    ifstream f(argv[1]);
    bool no_mean = (argv[2][0] == 'N');
    double mean = no_mean ? -1 : atof(argv[2]);
    bool preselect = (argv[3][0] == 'Y');
    int samples = atoi(argv[4]);
    
    MixModel jit(1, 1);
    f >> jit;
    
    int dev = preselect ? jit.sample_dev() : -1;
    
    int total_samples = 0;
    
    while (total_samples < samples)
    {
        vector<double> cur_samples = no_mean ? jit.sample(dev) : jit.sample(mean, dev);
        int bound = min(samples - total_samples, (int)cur_samples.size());
        for (int i=0;i<bound;i++)
        {
            printf("%.10lf ", cur_samples[i]);
        }
        total_samples += bound;
    }
    printf("\n");
    
    f.close();
    
    return 0;
}
