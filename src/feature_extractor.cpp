#include "../include/feature_extractor.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_map>

double calculateEntropy(const std::vector<uint8_t>& data) {
    std::vector<int> frequency(256, 0);
    for (uint8_t byte : data) {
        frequency[byte]++;
    }

    double entropy = 0.0;
    double dataSize = static_cast<double>(data.size());
    for (int freq : frequency) {
        if (freq > 0) {
            double probability = freq / dataSize;
            entropy -= probability * std::log2(probability);
        }
    }

    return entropy;
}

double calculateMean(const std::vector<uint8_t>& data) {
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

double calculateStandardDeviation(const std::vector<uint8_t>& data, double mean) {
    double squareSum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
    double variance = squareSum / data.size() - mean * mean;
    return std::sqrt(variance);
}

double calculateAutocorrelation(const std::vector<uint8_t>& data, int lag) {
    double mean = calculateMean(data);
    double variance = calculateStandardDeviation(data, mean);
    variance *= variance;

    double autocorr = 0.0;
    for (size_t i = 0; i < data.size() - lag; ++i) {
        autocorr += (data[i] - mean) * (data[i + lag] - mean);
    }
    autocorr /= (data.size() - lag) * variance;

    return autocorr;
}

std::vector<double> calculateBlockFrequencies(const std::vector<uint8_t>& data, int blockSize) {
    std::unordered_map<uint64_t, int> blockCounts;
    for (size_t i = 0; i <= data.size() - blockSize; i += blockSize) {
        uint64_t block = 0;
        for (int j = 0; j < blockSize; ++j) {
            block = (block << 8) | data[i + j];
        }
        blockCounts[block]++;
    }

    std::vector<double> frequencies;
    double totalBlocks = data.size() / blockSize;
    for (const auto& pair : blockCounts) {
        frequencies.push_back(pair.second / totalBlocks);
    }
    std::sort(frequencies.begin(), frequencies.end(), std::greater<double>());
    
    // return top 10 frequencies or pad with zeros because i dont want to sort it
    frequencies.resize(10, 0.0);
    return frequencies;
}

double calculateLongRepetitions(const std::vector<uint8_t>& data) {
    int maxRepetition = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = i + 1; j < data.size(); ++j) {
            int repetition = 0;
            while (j + repetition < data.size() && data[i + repetition] == data[j + repetition]) {
                repetition++;
            }
            maxRepetition = std::max(maxRepetition, repetition);
        }
    }
    return static_cast<double>(maxRepetition) / data.size();
}

double estimateKeyLength(const std::vector<uint8_t>& data) {
    std::vector<double> autocorrelations;
    for (int lag = 1; lag <= 32; ++lag) {
        autocorrelations.push_back(std::abs(calculateAutocorrelation(data, lag)));
    }
    auto maxIt = std::max_element(autocorrelations.begin(), autocorrelations.end());
    return static_cast<double>(std::distance(autocorrelations.begin(), maxIt) + 1);
}
// add more features else our prediction percentage will be too low add features like hamming distance, entropy of the data, etc.
Features extractFeatures(const std::vector<uint8_t>& data) {
    Features features;
    
    features.entropy = calculateEntropy(data);
    features.mean = calculateMean(data);
    features.stdDev = calculateStandardDeviation(data, features.mean);
    features.autocorrelation = calculateAutocorrelation(data, 1);
    features.blockFrequencies = calculateBlockFrequencies(data, 4);
    features.longRepetitions = calculateLongRepetitions(data);
    features.keyLengthEstimate = estimateKeyLength(data);
    
    return features;
}

