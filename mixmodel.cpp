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

static random_device dev;
static mt19937 gen(dev());

MixModel::MixModel(int dev_cnt, int act_cnt) : dev_cnt(dev_cnt), act_cnt(act_cnt)
{
    device.resize(dev_cnt);
    activity.resize(dev_cnt);
    lambda.resize(dev_cnt);
    sigma.resize(dev_cnt);
    
    d_id = a_id = 0;
    amt_P.resize(dev_cnt);
    amt_N.resize(dev_cnt);
    M.resize(dev_cnt);
    S.resize(dev_cnt);
    
    for (int i=0;i<dev_cnt;i++)
    {
        device[i] = 0;
        
        amt_P[i].resize(act_cnt);
        amt_N[i].resize(act_cnt);
        M[i].resize(act_cnt);
        S[i].resize(act_cnt);
        
        activity[i].resize(act_cnt);
        lambda[i].resize(act_cnt);
        sigma[i].resize(act_cnt);
        for (int j=0;j<act_cnt;j++)
        {
            activity[i][j] = amt_P[i][j] = amt_N[i][j] = 0;
            lambda[i][j] = M[i][j] = S[i][j] = sigma[i][j] = 0.0;
        }
    }
}

int MixModel::get_dev_id(string dev)
{
    if (dev_map.count(dev)) return dev_map[dev];
    return (dev_map[dev] = d_id++);
}

int MixModel::get_act_id(string act)
{
    if (act_map.count(act)) return act_map[act];
    return (act_map[act] = a_id++);
}

// push a new device + activity observation
void MixModel::push_line(string dev, string act)
{
    int d = get_dev_id(dev);
    int a = get_act_id(act);
    device[d]++;
    activity[d][a]++;
}

// push a new length value
void MixModel::push_len(string dev, string act, int len)
{
    int d = get_dev_id(dev);
    int a = get_act_id(act);
    
    amt_P[d][a]++;
    
    lambda[d][a] += (len * 1.0 - lambda[d][a]) / amt_P[d][a];
}

// push a new delta value
void MixModel::push_delta(string dev, string act, double delta)
{
    int d = get_dev_id(dev);
    int a = get_act_id(act);
    
    amt_N[d][a]++;
    
    double old_m = M[d][a];
    M[d][a] += (delta - M[d][a]) / amt_N[d][a];
    S[d][a] += (delta - old_m) * (delta - M[d][a]);
    
    sigma[d][a] = sqrt(S[d][a] / (amt_N[d][a] - 1));
}

int MixModel::sample_dev()
{
    discrete_distribution<int> d_dev(device.begin(), device.end());
    return d_dev(gen);
}

// sample the mixture model using the computed means
vector<double> MixModel::sample(int dev)
{
    // first choose the device
    int d = (dev == -1) ? sample_dev() : dev;
    
    // then choose the activity
    discrete_distribution<int> d_act(activity[d].begin(), activity[d].end());
    int a = d_act(gen);
    
    // then choose the output sequence length
    poisson_distribution<int> Pois(lambda[d][a]);
    int l = Pois(gen);
    
    // finally, choose the values
    normal_distribution<double> N(M[d][a], sigma[d][a]);
    vector<double> ret(l);
    for (int i=0;i<l;i++)
    {
        ret[i] = N(gen);
    }
    
    return ret;
}

// sample the mixture model assuming a given mean value
vector<double> MixModel::sample(double mean, int dev)
{
    // first choose the device
    int d = (dev == -1) ? sample_dev() : dev;
    
    // then choose the activity
    discrete_distribution<int> d_act(activity[d].begin(), activity[d].end());
    int a = d_act(gen);
    
    // then choose the output sequence length
    poisson_distribution<int> Pois(lambda[d][a]);
    int l = Pois(gen);
    
    // finally, choose the values
    normal_distribution<double> N(mean, sigma[d][a]);
    vector<double> ret(l);
    for (int i=0;i<l;i++)
    {
        ret[i] = N(gen);
    }
    
    return ret;
}

MixModel MixModel::best_case()
{
    MixModel ret(dev_cnt, act_cnt);
    
    int outer_best = -1;
    double outer_val = -1.0;
    
    // always choose the device-activity pair with the smallest variance
    for (int i=0;i<dev_cnt;i++)
    {
        int best = -1;
        double val = -1.0;
        for (int j=0;j<act_cnt;j++)
        {
            ret.sigma[i][j] = sigma[i][j];
            if (amt_N[i][j] < 2) continue;
            if (best == -1 || sigma[i][j] < val)
            {
                val = sigma[i][j];
                best = j;
                if (outer_best == -1 || sigma[i][j] < outer_val)
                {
                    outer_val = sigma[i][j];
                    outer_best = i;
                }
            }
        }
        ret.lambda[i][best] = 100.0; // irrelevant, just needs to be +ve
        ret.activity[i][best] = 1;
    }
    
    ret.device[outer_best] = 1;
    
    return ret;
}

MixModel MixModel::worst_case()
{
    MixModel ret(dev_cnt, act_cnt);
    
    int outer_best = -1;
    double outer_val = -1.0;
    
    // always choose the device-activity pair with the largest variance
    for (int i=0;i<dev_cnt;i++)
    {
        int best = -1;
        double val = -1.0;
        for (int j=0;j<act_cnt;j++)
        {
            ret.sigma[i][j] = sigma[i][j];
            if (amt_N[i][j] < 2) continue;
            if (best == -1 || sigma[i][j] > val)
            {
                val = sigma[i][j];
                best = j;
                if (outer_best == -1 || sigma[i][j] > outer_val)
                {
                    outer_val = sigma[i][j];
                    outer_best = i;
                }
            }
        }
        ret.lambda[i][best] = 100.0; // irrelevant, just needs to be +ve
        ret.activity[i][best] = 1;
    }
    
    ret.device[outer_best] = 1;
    
    return ret;
}

istream& operator>>(istream &in, MixModel &M)
{
    int dev_cnt, act_cnt;
    in >> dev_cnt >> act_cnt;
    
    M = MixModel(dev_cnt, act_cnt);
    
    in >> M.d_id >> M.a_id;
    
    for (int i=0;i<M.d_id;i++)
    {
        string key;
        int val;
        in >> key >> val;
        M.dev_map[key] = val;
    }
    for (int i=0;i<M.a_id;i++)
    {
        string key;
        int val;
        in >> key >> val;
        M.dev_map[key] = val;
    }
    
    for (int i=0;i<M.dev_cnt;i++)
    {
        for (int j=0;j<M.act_cnt;j++)
        {
            in >> M.amt_P[i][j];
        }
    }
    
    for (int i=0;i<M.dev_cnt;i++)
    {
        for (int j=0;j<M.act_cnt;j++)
        {
            in >> M.amt_N[i][j];
        }
    }
    
    for (int i=0;i<M.dev_cnt;i++)
    {
        for (int j=0;j<M.act_cnt;j++)
        {
            in >> M.M[i][j];
        }
    }
    
    for (int i=0;i<M.dev_cnt;i++)
    {
        for (int j=0;j<M.act_cnt;j++)
        {
            in >> M.S[i][j];
        }
    }
    
    for (int i=0;i<M.dev_cnt;i++)
    {
        in >> M.device[i];
    }
    
    for (int i=0;i<M.dev_cnt;i++)
    {
        for (int j=0;j<M.act_cnt;j++)
        {
            in >> M.activity[i][j];
        }
    }
    
    for (int i=0;i<M.dev_cnt;i++)
    {
        for (int j=0;j<M.act_cnt;j++)
        {
            in >> M.lambda[i][j];
        }
    }
    
    for (int i=0;i<M.dev_cnt;i++)
    {
        for (int j=0;j<M.act_cnt;j++)
        {
            in >> M.sigma[i][j];
        }
    }
    
    return in;
}

ostream& operator<<(ostream &out, const MixModel &M)
{
    out << M.dev_cnt << " " << M.act_cnt << endl;
    out << M.d_id << " " << M.a_id << endl;
    
    for (auto kv : M.dev_map)
    {
        out << kv.first << " " << kv.second << endl;
    }
    for (auto kv : M.act_map)
    {
        out << kv.first << " " << kv.second << endl;
    }
    
    for (int i=0;i<M.dev_cnt;i++)
    {
        for (int j=0;j<M.act_cnt;j++)
        {
            out << M.amt_P[i][j] << " ";
        }
        out << endl;
    }
    
    for (int i=0;i<M.dev_cnt;i++)
    {
        for (int j=0;j<M.act_cnt;j++)
        {
            out << M.amt_N[i][j] << " ";
        }
        out << endl;
    }
    
    for (int i=0;i<M.dev_cnt;i++)
    {
        for (int j=0;j<M.act_cnt;j++)
        {
            out << M.M[i][j] << " ";
        }
        out << endl;
    }
    
    for (int i=0;i<M.dev_cnt;i++)
    {
        for (int j=0;j<M.act_cnt;j++)
        {
            out << M.S[i][j] << " ";
        }
        out << endl;
    }
    
    for (int i=0;i<M.dev_cnt;i++)
    {
        out << M.device[i] << " ";
    }
    out << endl;
    
    for (int i=0;i<M.dev_cnt;i++)
    {
        for (int j=0;j<M.act_cnt;j++)
        {
            out << M.activity[i][j] << " ";
        }
        out << endl;
    }
    
    for (int i=0;i<M.dev_cnt;i++)
    {
        for (int j=0;j<M.act_cnt;j++)
        {
            out << M.lambda[i][j] << " ";
        }
        out << endl;
    }
    
    for (int i=0;i<M.dev_cnt;i++)
    {
        for (int j=0;j<M.act_cnt;j++)
        {
            out << M.sigma[i][j] << " ";
        }
        out << endl;
    }
    
    return out;
}