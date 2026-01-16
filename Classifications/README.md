# Circle Classification with MLP in PyTorch

A simple neural network implementation for classifying 2D points as inside or outside a circle using PyTorch. This project demonstrates basic concepts of neural networks, binary classification, and model evaluation.

## Project Overview

This project creates a synthetic dataset of 2D points and trains a Multi-Layer Perceptron (MLP) to classify whether points are inside or outside a circle with center (0.5, 0.5) and radius 0.3. The implementation covers the complete machine learning workflow including data generation, model definition, training, and evaluation.

## Features

- **Synthetic Data Generation**: Creates 5000 random 2D points with binary labels based on circular region  
- **MLP Model**: Two-layer neural network with ReLU activation  
- **Training Pipeline**: Complete training loop with Adam optimizer and CrossEntropy loss  
- **Visualization**: Multiple plots showing data distribution and learning curves  
- **Evaluation**: Comprehensive error tracking on both training and test sets  

## Model Architecture

```python
MLP(
    (mlp1): Linear(in_features=2, out_features=15, bias=True)
    (func): ReLU()
    (mlp2): Linear(in_features=15, out_features=2, bias=True)
)
