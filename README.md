# K-means from Scratch — Theory & Implementation

## Overview

This repository contains a **from-scratch implementation of the K-means clustering algorithm**, together with a **rigorous mathematical analysis** of its objective function.

The project is structured in two parts:

1. **Theoretical foundations**
   - Definition of the K-means objective
   - Proof of the optimal centroid for fixed clusters
   - Proof of the non-convexity of the global objective

2. **Practical implementation**
   - Pure NumPy implementation
   - Manual cluster assignment
   - Iterative centroid updates

The goal is to understand both **why K-means works** and **how it is implemented**, without relying on black-box libraries.

---

## Mathematical Background

We consider a dataset:

x₁, …, xₙ ∈ ℝᵈ

partitioned into k clusters C₁, …, Cₖ.

### K-means Objective Function

The objective minimized by K-means is:

J(C, μ) = Σⱼ Σᵢ∈Cⱼ ||xᵢ − μⱼ||²

where:
- Cⱼ is the j-th cluster
- μⱼ is the centroid of cluster Cⱼ

---

## Key Result: Optimal Centroid

For fixed cluster assignments, the function

Σᵢ∈Cⱼ ||xᵢ − μ||²

is minimized by the empirical mean:

μⱼ* = (1 / |Cⱼ|) Σᵢ∈Cⱼ xᵢ

This result justifies the centroid update step of the K-means algorithm.

✔ Convex with respect to μ for fixed clusters  
✘ Not convex globally

---

## Non-Convexity of K-means

The global objective can be written as:

F(μ₁, …, μₖ) = Σᵢ minⱼ ||xᵢ − μⱼ||²

A simple counterexample shows that this function violates the convexity inequality, proving that **K-means is non-convex**.

This explains:
- Sensitivity to initialization
- Convergence to local minima
- Different solutions across runs

---

## Implementation

### Main Class

```python
class KMEANS:
    def __init__(self, k, max_iter=10)
    def fit(self, X)
    def predict(self, X)
```

### Core Functions

- `Euclid_Distance(x, mu)`
- `assignation(x, centroids)`
- `cluster_assign(X, centroids)`
- `centroids_update(X, labels, k)`

---

## Example Usage

```python
import numpy as np
from kmeans import KMEANS

X = np.random.rand(100, 2)

model = KMEANS(k=3, max_iter=20)
model.fit(X)

labels = model.predict(X)
centroids = model.centroids
```

---

## Requirements

- Python ≥ 3.10
- NumPy
- Matplotlib (optional)

---

## Notes

- Random initialization
- No K-means++ yet
- Intended for educational purposes

---

## Future Work

- K-means++ initialization
- Inertia computation
- Vectorized distance computation
- Unit tests

---

## Author : Lefki Meidi 

Project developed to connect **optimization theory** with **algorithmic implementation**.
