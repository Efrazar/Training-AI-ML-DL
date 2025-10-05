# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 16:59:44 2025

@author: ezarazua
"""

# --- THE GRADIENT DESCENT LOOP ---
import numpy as np
import matplotlib.pyplot as plt

# Prepare functions:

    # Prediction values calculation
def Gradient_Descent(X, n_epochs, alpha, n, m ,b, y):
    
    loss_history = []
    m_history = [m]
    b_history = [b]
    
    for i in range(n_epochs):
        # 1. Make predictions with the current m and b
        y_predicted = m * X + b
        
        # 2. Calculate the loss (MSE)
        loss = (1/n) * np.sum((y - y_predicted)**2)
        loss_history.append(loss)
        
        # 3. Calculate the gradients
        # The derivative of the loss function with respect to m and b
        grad_m = (-2/n) * np.sum(X * (y - y_predicted))
        grad_b = (-2/n) * np.sum(y - y_predicted)
        
        # 4. Update the parameters (take a step downhill)
        m = m - alpha * grad_m
        b = b - alpha * grad_b
        
        # Store history for visualization
        m_history.append(m)
        b_history.append(b)
    
        # Print progress every 100 epochs
        if (i+1) % 100 == 0:
            print(f"Epoch {i+1:4d}: Loss = {loss:.4f}, m = {m:.4f}, b = {b:.4f}")
    
    print("\n--- Training Complete ---")
    print(f"Final parameters: m = {m:.4f}, b = {b:.4f}")
    
    # Plotting the final regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, label='Data Points')
    plt.plot(X, m * X + b, color='red', linewidth=3, label='Final Regression Line')
    plt.title("Linear Regression with Gradient Descent")
    plt.xlabel("Transcription Factor Concentration")
    plt.ylabel("Gene Expression Level")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting the loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(n_epochs), loss_history)
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error (Loss)")
    plt.grid(True)
    plt.show()