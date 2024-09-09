#pragma once
#include <vector>
#include <cstdint>
//dont use using namespace std here because it will cause name conflict

struct Features {
    double entropy;
    double mean;
    double stdDev;
    double autocorrelation;
    std::vector<double> blockFrequencies;
    double longRepetitions;
    double keyLengthEstimate;
};

Features extractFeatures(const std::vector<uint8_t>& data);
double calculateEntropy(const std::vector<uint8_t>& data);
double calculateMean(const std::vector<uint8_t>& data);
double calculateStandardDeviation(const std::vector<uint8_t>& data, double mean);
double calculateAutocorrelation(const std::vector<uint8_t>& data, int lag);
