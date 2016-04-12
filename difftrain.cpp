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
        printf("Usage: ./difftrain <diff_file> <dev_cnt> <act_cnt> <base>\n");
        return -1;
    }
    
    ifstream f(argv[1]);
    int dev_cnt = atoi(argv[2]);
    int act_cnt = atoi(argv[3]);
    string base = argv[4];
    
    MixModel jit(dev_cnt, act_cnt);
    
    printf("Starting the training of the jitter models...\n");
    
    string dev, act;
    int stint_len;
    
    while (f >> dev >> act >> stint_len)
    {
        for (int i=0;i<stint_len;i++)
        {
            jit.push_line(dev, act);
            double cur_diff;
            f >> cur_diff;
            jit.push_delta(dev, act, cur_diff);
        }
        jit.push_line(dev, act);
        jit.push_len(dev, act, stint_len + 1);
    }
    
    f.close();
    
    printf("Done! Writing the models to files...\n");
    
    ofstream g_main(base + "_main.mx");
    ofstream g_best(base + "_best.mx");
    ofstream g_worst(base + "_worst.mx");
    
    g_main << jit;
    g_best << jit.best_case();
    g_worst << jit.worst_case();
    
    g_main.close();
    g_best.close();
    g_worst.close();
    
    printf("Done. Models written to %s_main.mx, %s_best.mx, and %s_worst.mx.\n", base.c_str(), base.c_str(), base.c_str());
    printf("N.B. DO NOT TRAIN %s_best.mx and %s_worst.mx further!\n", base.c_str(), base.c_str());
    
    return 0;
}