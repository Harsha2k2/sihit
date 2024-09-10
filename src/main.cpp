#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include "../include/feature_extractor.h"
#include "../include/classifier.h"

using namespace std;

vector<uint8_t> loadData(const string& filename) {
    ifstream file(filename, ios::binary);
    vector<uint8_t> data;
    
    if (!file) {
        cerr << "Unable to open file: " << filename << endl;
        return data;
    }
    
    data = vector<uint8_t>((istreambuf_iterator<char>(file)),
                            istreambuf_iterator<char>());
    
    return data;
}

int main() {
    vector<string> filenames = {"cipher1.bin", "cipher2.bin", "cipher3.bin", "cipher4.bin",
                                "cipher5.bin", "cipher6.bin", "cipher7.bin", "cipher8.bin"};
    
    Classifier classifier;
    
    //we need a pre trained model here with humongous dataset please find it
    vector<Features> trainingData;
    vector<Algorithm> trainingLabels = {Algorithm::AES, Algorithm::DES, Algorithm::RSA, Algorithm::MD5,
                                        Algorithm::SHA1, Algorithm::SHA256, Algorithm::BLOWFISH};
    
    // Shuffle the filenames to randomize training and test sets
    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(filenames), std::end(filenames), rng);
    
    for (size_t i = 0; i < filenames.size() - 1; ++i) {
        string filepath = "../data/" + filenames[i];
        vector<uint8_t> data = loadData(filepath);
        
        if (data.empty()) {
            cerr << "Failed to load data from " << filenames[i] << endl;
            continue;
        }

        Features features = extractFeatures(data);
        trainingData.push_back(features);
    }
    
    classifier.train(trainingData, trainingLabels);
    
    // our test file is the last file in the list because we dont have any other choice
    string testFilepath = "../data/" + filenames.back();
    vector<uint8_t> testData = loadData(testFilepath);
    
    if (!testData.empty()) {
        Features testFeatures = extractFeatures(testData);
        Algorithm result = classifier.classify(testFeatures);
        
        cout << "File: " << filenames.back() << endl;
        cout << "Identified algorithm: " << classifier.algorithmToString(result) << endl;
    }

    return 0;
}
