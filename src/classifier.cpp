#include "../include/classifier.h"
#include <mlpack/methods/ann/init_rules/he_init.hpp>

using namespace mlpack;

Classifier::Classifier() : isTrained(false), inputSize(0), numClasses(9) {}

void Classifier::train(const std::vector<Features>& trainingData, const std::vector<Algorithm>& labels) {
    inputSize = 5 + trainingData[0].blockFrequencies.size();
    arma::mat data(inputSize, trainingData.size());
    arma::mat labelsOneHot(numClasses, labels.size(), arma::fill::zeros);

    for (size_t i = 0; i < trainingData.size(); ++i) {
        const auto& features = trainingData[i];
        data(0, i) = features.entropy;
        data(1, i) = features.mean;
        data(2, i) = features.stdDev;
        data(3, i) = features.autocorrelation;
        data(4, i) = features.longRepetitions;
        for (size_t j = 0; j < features.blockFrequencies.size(); ++j) {
            data(5 + j, i) = features.blockFrequencies[j];
        }
        labelsOneHot(static_cast<size_t>(labels[i]), i) = 1;
    }

    // Define the network architecture
    model = FFN<CrossEntropyError<>>();
    model.Add<Linear<>>(inputSize, 64);
    model.Add<ReLULayer<>>();
    model.Add<Linear<>>(64, 32);
    model.Add<ReLULayer<>>();
    model.Add<Linear<>>(32, numClasses);
    model.Add<LogSoftMax<>>();

    // Set up the optimizer
    ens::Adam optimizer(0.001, 64, 0.9, 0.999, 1e-8, 100000, 1e-5, true);

    // setup the model 
    model.Train(data, labelsOneHot, optimizer);

    isTrained = true;
}

Algorithm Classifier::classify(const Features& features) {
    if (!isTrained) {
        return Algorithm::UNKNOWN;
    }

    arma::vec sample(inputSize);
    sample[0] = features.entropy;
    sample[1] = features.mean;
    sample[2] = features.stdDev;
    sample[3] = features.autocorrelation;
    sample[4] = features.longRepetitions;
    for (size_t i = 0; i < features.blockFrequencies.size(); ++i) {
        sample[5 + i] = features.blockFrequencies[i];
    }

    arma::mat prediction;
    model.Predict(sample, prediction);

    arma::uword maxIndex;
    prediction.max(maxIndex);

    return static_cast<Algorithm>(maxIndex);
}

std::string Classifier::algorithmToString(Algorithm alg) {
    switch (alg) {
        case Algorithm::AES: return "AES";
        case Algorithm::DES: return "DES";
        case Algorithm::RSA: return "RSA";
        case Algorithm::MD5: return "MD5";
        case Algorithm::SHA1: return "SHA1";
        case Algorithm::SHA256: return "SHA256";
        case Algorithm::BLOWFISH: return "Blowfish";
        case Algorithm::TWOFISH: return "Twofish";
        default: return "Unknown";
    }
}
