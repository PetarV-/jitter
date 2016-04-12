/*
 Petar 'PetarV' Velickovic
 Data Structure: Jitter Mixture Model
*/

#ifndef MIXMODEL
#define MIXMODEL

#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;
typedef long long lld;

class MixModel
{
private:
    int dev_cnt;
    int act_cnt;
    
    int d_id;
    int a_id;
    
    unordered_map<string, int> dev_map; // device name -> id
    unordered_map<string, int> act_map; // activity name -> id
    
    vector<vector<int> > amt_P, amt_N; // intermediate values
    vector<vector<double> > M, S; // intermediate values
    
    vector<int> device; // P(device)
    vector<vector<int> > activity; // P(activity | device)
    vector<vector<double> > lambda; // P(#samples | device, activity) ~ Pois(lambda)
    vector<vector<double> > sigma; // P(offset | device, activity) ~ N(mu, sigma)
    // N.B. sigma estimated from data, mu will be provided when computing
    
public:
    MixModel(int dev_cnt, int act_cnt);
    
    int get_dev_id(string dev);
    int get_act_id(string act);
    
    // push a new device + activity observation
    void push_line(string dev, string act);
    // push a new length value
    void push_len(string dev, string act, int len);
    // push a new delta value
    void push_delta(string dev, string act, double delta);
    
    // sample the device distirbution (helper method)
    int sample_dev();
    
    // sample the mixture model using the computed means
    vector<double> sample(int dev = -1);
    
    // sample the mixture model assuming a given mean value
    vector<double> sample(double mean, int dev = -1);
    
    MixModel best_case();
    MixModel worst_case();
    
    // I/O operator overloads
    friend istream& operator>>(istream &in, MixModel &M);
    friend ostream& operator<<(ostream &out, const MixModel &M);
};

#endif