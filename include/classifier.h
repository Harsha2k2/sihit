#pragma once
#include <vector>
#include <string>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/cross_entropy_error.hpp>
#include "feature_extractor.h"
// honestly i know only these algorithm names so goodluck if you want to add more also change feature extractor in src/main.cpp if you want to add more features
enum class Algorithm {
    AES, DES, RSA, MD5, SHA1, SHA256, BLOWFISH, TWOFISH, UNKNOWN
};

class Classifier {
public:
    Classifier();
    void train(const std::vector<Features>& trainingData, const std::vector<Algorithm>& labels);
    Algorithm classify(const Features& features);
    std::string algorithmToString(Algorithm alg);

private:
    mlpack::FFN<mlpack::CrossEntropyError<>> model;
    bool isTrained;
    size_t inputSize;
    size_t numClasses;
};
