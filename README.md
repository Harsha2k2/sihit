# Cryptographic Algorithm Identifier

we use only C++ (because i am noob and dont understand python) to identify cryptographic algorithms from given datasets using machine learning techniques and if u get a doubt how we use c++ to identify crypto algorithms then we are using mlpack library for machine learning techniques and arma::mat for matrix operations. and an api call from nn.transformers.py to get embeddings of the data and then we are doing some feature engineering and then training a neural network to identify the cryptographic algorithm (good luck mate).

## Dependencies

This project requires the following libraries:

1. mlpack (version 3.4.2 or later)
2. Armadillo (version 10.1.0 or later)
3. Boost (version 1.71.0 or later)
4. OpenBLAS or Intel MKL
*5. hopefully someone from our team will create a docker image for this project so u dont have to install these dependencies manually (please do it before 19th sep 2024)

## Installation
will update this later at the end of the project because we are not allowed to use any other compiler than gcc and linker linker linker. hehe :3
### Ubuntu/Debian
i honestly dont know how to install these dependencies in ubuntu/debian, if you are reading this and you are a sysadmin, please help us.

## Usage

To run the program, use the following command:
(it wont work now because we didnt create a docker image or create a package for this project)

```bash
./main
```
## Thank you
ss, claude, chatgpt, bing, and all the other ai models for helping us to complete this project in last 3 days of the hackathon.
