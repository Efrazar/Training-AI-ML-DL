# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 16:15:12 2025

@author: ezarazua


      LOSS FUNCTION AND GRADIENT DESCENT EXAMPLE

"""

import numpy as np
import matplotlib.pyplot as plt
import GRADIENT_DESCENT_FUNCTIONS

# # Let's create some sample data that looks like a line with some noise
# # X = Transcription Factor Concentration
# # y = Gene Expression Level
np.random.seed(42) # for reproducibility
X = 2 * np.random.rand(100, 1)
y = 5 + 2 * X + np.random.randn(100, 1)

# Let's visualize our data
plt.figure(figsize=(10, 6))
plt.scatter(X, y)
plt.title("Simulated Gene Expression Data")
plt.xlabel("Transcription Factor Concentration")
plt.ylabel("Gene Expression Level")

# Initialize random values for our model's parameters: m (slope) and b (intercept)
# These are our starting points on the "loss landscape"
np.random.seed(0)
m = np.random.randn()
b = np.random.randn()

print(f"Starting with: m = {m:.4f}, b = {b:.4f}")

               ##########  ##########
               
# Hyperparameters for gradient descent
alpha = 0.1
n_epochs = 1000
n = float(len(y))

GRADIENT_DESCENT_FUNCTIONS.Gradient_Descent(X, n_epochs, alpha, n, m ,b, y)
