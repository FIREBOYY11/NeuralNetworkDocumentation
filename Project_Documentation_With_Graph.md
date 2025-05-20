# Project Overview and Documentation

---

## What are we working on?

We are working on building a simple neural network to understand the basics of Machine Learning, specifically using a sigmoid-activated, single-layer model. The goal is to train the model using a dataset generated through a linear regression-style function to predict outputs and minimize error through training.

---

## What are we building?

We are constructing a basic feedforward neural network with:

- One input layer  
- One output layer  
- No hidden layers (single-layer perceptron)  
- Sigmoid activation function

The model is trained using manually coded gradient descent via weight updates from calculated loss.

### Code Snippet

```python
import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Randomized Input dataset
np.random.seed(1)
num_samples = 100
num_features = 10
inputs = np.random.rand(num_samples, num_features)

target_weights = np.random.uniform(-3, 3, (num_features, 1))
noise = np.random.normal(0, 0.1, (num_samples, 1))
outputs = np.dot(inputs, target_weights) + noise

# Initialize weights randomly with mean 0
weights = 2 * np.random.random((num_features, 1)) - 1

losses = []

# Number of iterations
epochs = 10000

for epoch in range(epochs):
    # Forward propagation
    input_layer = inputs
    weighted_sum = np.dot(input_layer, weights)
    predictions = sigmoid(weighted_sum)

    # Error
    error = outputs - predictions

    # Backpropagation
    adjustments = error * sigmoid_derivative(predictions)

    # Update weights
    weights += np.dot(input_layer.T, adjustments) / num_samples

    # Loss storage
    loss = np.mean(np.abs(error))
    losses.append(loss)
```

---

## What knowledge repository are we referring to?

We are using the tutorial from [Real Python: Python AI Neural Network](https://realpython.com/python-ai-neural-network/) — particularly Chapter 3, which outlines building a simple neural network from scratch using NumPy.

We also refer to standard machine learning practices including:

- Sigmoid function usage  
- Error calculation using Mean Absolute Error (MAE)  
- Weight updates using backpropagation

---

## What aspects of Machine Learning are we using?

- **Supervised Learning** — We provide input-output pairs and train the model to minimize the error.  
- **Activation Functions** — Use of sigmoid to simulate neuron firing.  
- **Gradient Descent** — Manual implementation of gradient descent for training.  
- **Loss Function** — MAE is used to measure performance.

---

## What are the components of our model?

- **Inputs** — 100 samples with 10 features each, randomly generated.  
- **Weights** — Randomly initialized, adjusted during training.  
- **Activation** — Sigmoid function for non-linearity.  
- **Outputs** — Predicted probabilities (scaled between 0 and 1).  
- **Training** — Gradient descent over 10,000 epochs.

### Graph

![Training Loss Graph](WhatsApp Image 2025-04-30 at 10.52.52 AM (1).jpeg
)

---

## What problem will this model solve?

This model is primarily educational. It demonstrates how neural networks learn by adjusting weights to minimize prediction error. It can be expanded to classification or regression problems in real-world scenarios.

### Use Case Example

- Predicting likelihood of binary outcomes based on numeric input features (e.g., customer churn, email spam detection).

---

## Timelines of Activities

| Activity                               | Timeline |
|----------------------------------------|----------|
| Model Planning                         | Week 1   |
| Dataset Generation                     | Week 2   |
| Initial Code Implementation            | Week 2   |
| Increasing dataset size using regression | Week 4 |
| Training & Testing                     | Week 4   |
| Documentation                          | Week 4   |
