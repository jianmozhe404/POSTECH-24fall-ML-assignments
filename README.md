# Machine Learning Assignments Overview

This repository contains the solutions and implementations for three assignments in a machine learning course. The assignments cover a variety of topics, ranging from fundamental statistical learning methods to advanced deep learning models. Below is a summary of the three assignments.

---

## Assignment 1: Statistical Learning Basics

### **Overview**
This assignment focuses on fundamental statistical learning concepts, including maximum likelihood estimation, Naïve Bayes classification, ridge regression, and optimal predictors.

### **Key Topics**
1. **Maximum Likelihood Estimation (MLE)**:
   - Estimating the parameter \( \lambda \) of a Poisson distribution using MLE.
   
2. **Naïve Bayes Classifier**:
   - Derivation of model parameters (\(p, \mu_0, \mu_1, \sigma_0, \sigma_1\)) under the univariate Gaussian likelihood assumption.

3. **Ridge Regression**:
   - Introducing regularization to linear regression to prevent overfitting.
   - Reformulating the problem in matrix form and solving for the optimal parameters.

4. **Optimal Predictors**:
   - Exploring optimal solutions for minimizing squared loss and absolute loss based on conditional expectations.

---

## Assignment 2: Machine Learning Algorithms

### **Overview**
This assignment focuses on implementing machine learning algorithms for regression, classification, and clustering, as well as exploring optimization techniques.

### **Key Topics**
1. **Linear Regression**:
   - Implementation of univariate and multivariate linear regression.
   - Feature normalization, cost function calculation, and gradient descent optimization.

2. **Logistic Regression**:
   - Implementation of the sigmoid function, cost function, and gradient descent for binary classification.
   - Visualization of decision boundaries.

3. **Clustering**:
   - Implementation of **K-Means** and **Gaussian Mixture Models (GMM)** using the Expectation-Maximization (EM) algorithm.

4. **Optimization Algorithms**:
   - Implementation of **SGD with Momentum**, **AdaGrad**, and **Adam** optimizers to explore their impact on model training.

5. **Neural Networks**:
   - Building a two-layer fully connected neural network with ReLU and Softmax activations.
   - Training and validating the model on the **Fashion-MNIST** dataset.

6. **Convolutional Neural Networks (CNN)**:
   - Designing and training a simple CNN for image classification using convolutional, pooling, and fully connected layers.

---

## Assignment 3: Final Project - Radar Image Classification

### **Overview**
The final project focuses on developing a multi-class classification model to identify specific weather patterns (e.g., heavy snow or rain) from radar images of South Korea. It involves tackling challenges like data imbalance, model design, and training optimization.

### **Key Topics**
1. **Data Preprocessing**:
   - Addressing data imbalance by balancing positive and negative samples.
   - Resizing radar images and standardizing pixel values for faster training.

2. **Model Architecture**:
   - Using **ResNet** as the base model, with independent fully connected layers for each class.
   - Adding dropout and batch normalization for improved generalization and faster convergence.

3. **Training Strategy**:
   - Employing **curriculum learning** to mitigate overfitting by progressively training on subsets of the dataset.
   - Optimizing decision thresholds for multi-label classification.

4. **Performance Optimization**:
   - Leveraging local GPU resources with limited memory.
   - Balancing batch size and computation efficiency to train the model effectively.

5. **Future Directions**:
   - Exploring ensemble learning, label-specific models, and incorporating temporal continuity in radar images.

---

## Repository Structure
- **Assignment1/**: Contains code and report for statistical learning methods.
- **Assignment2/**: Contains implementations of regression, clustering, and neural networks.
- **Assignment3/**: Contains the final project code and report for radar image classification.

---
