# Multilayer Perceptron for Ecoli Dataset Classification

This project implements a **Multilayer Perceptron (MLP)** to classify two specific classes (`cp` and `im`) in the Ecoli dataset. The assignment involves implementing the MLP from scratch and using a high-level library such as PyTorch to compare the performance and ease of implementation.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [From Scratch](#from-scratch)
  - [Using PyTorch](#using-pytorch)
- [Results](#results)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The task is to implement a **Multilayer Perceptron** to classify two classes (`cp` and `im`) from the **Ecoli dataset**. The dataset is preprocessed to keep only these two classes, and the MLP is implemented in two ways:
1. From scratch using only Python.
2. Using a high-level library (PyTorch) to streamline the process.

## Dataset

The dataset used is the **Ecoli dataset** from the UCI Machine Learning Repository:
- [Download the dataset](https://archive.ics.uci.edu/ml/datasets/Ecoli)

We focus on predicting two classes: `cp` and `im`. All other classes are removed from the dataset during preprocessing.

## Methodology

### From Scratch

- **Dataset Preprocessing**: 
  - We load the dataset and filter out rows that do not belong to the `cp` or `im` classes.
  - The dataset is split into **training** and **testing** sets with an 80/20 split.

- **Network Architecture**:
  - **Input Layer**: 2 neurons (since the dataset has been reduced to two features).
  - **Hidden Layer**: 1 neuron.
  - **Activation Function**: Sigmoid function applied to both layers.
  
- **Training**:
  - The weights and biases are updated using a custom backpropagation method.
  - The model is trained over multiple epochs, and accuracy is evaluated using a custom prediction function.
  
### Using PyTorch

- **Dataset Preprocessing**:
  - Similar to the scratch implementation, we filter out the `cp` and `im` rows and convert the labels into binary values (`0` for `cp` and `1` for `im`).
  - The dataset is split into **training** and **validation** sets with an 80/20 split.

- **Network Architecture**:
  - **Input Layer**: 7 neurons (corresponding to the 7 features of the Ecoli dataset).
  - **Hidden Layer**: 2 neurons with a Sigmoid activation function.
  - **Output Layer**: 1 neuron with a Sigmoid activation function.

- **Training**:
  - **Loss Function**: Binary Cross-Entropy (improved from initial use of Mean Squared Error).
  - **Optimizer**: Adam optimizer (after trial and error with SGD).
  - **Learning Rate**: 0.1 (tested with 0.01 and 0.001, but 0.1 yielded the best result).
  - **Epochs**: 100 (the model converged after about 20 epochs).

### Performance

The PyTorch implementation showed much better performance and faster implementation time compared to the from-scratch version. The dataset was small, making it easy to overfit, but proper tuning led to reasonable accuracy.

## Results

- **From Scratch**: The model was functional but less accurate and slower to converge.
- **PyTorch**: Achieved higher accuracy and consistency, with validation accuracy fluctuating between 60% and 95% depending on the random initialization and learning rate.

## Conclusion

The **PyTorch** implementation was much more efficient, both in terms of accuracy and ease of use, compared to the custom-built neural network from scratch. With the small dataset, it was easy to overfit the data using the high-level library. However, the flexibility and power of libraries like PyTorch provide a significant advantage in terms of both accuracy and development speed.

