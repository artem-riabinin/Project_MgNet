# Final Programming Assignment: Implementation of Multigrid and MgNet

## Overview
This project focuses on implementing and analyzing the Multigrid method and MgNet for solving linear systems and image classification. The assignment is divided into two main tasks:
1. **Multigrid Method**: Solve a given linear system using both the gradient descent method and the multigrid method, compare their performance, and analyze the error convergence.
2. **MgNet Implementation**: Modify the Multigrid implementation to implement MgNet and apply it to the CIFAR-10 dataset to achieve high test accuracy under parameter constraints.

This project is implemented in a Jupyter Notebook.

## Task 1: Multigrid Method

### Problem Statement
We consider the linear system:
\[ A u = f \]
or equivalently,
\[ u = \arg\min_{v \in \mathbb{R}^{n \times n}} \frac{1}{2} (A v, v)_F - (f, v)_F \]
where \( u, v, f \in \mathbb{R}^{n \times n} \) and the Frobenius inner product is defined as:
\[ (f, v)_F = \sum_{i,j=1}^{n} f_{ij} v_{ij} \]

The convolution kernel \( A \) is given by:
\[ A = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix} \]
and the right-hand side term is:
\[ f_{ij} = \frac{1}{(n+1)^2}. \]

### Implementation
- Use **gradient descent** and **multigrid method** to solve the problem with a random initial guess \( u_0 \).
- Compute solutions \( u_{GD} \) and \( u_{MG} \) obtained from gradient descent and multigrid, respectively.
- Perform experiments with parameters:
  - \( J = 4 \), \( n = 2^J - 1 \)
  - Number of iterations \( M = 100 \)

### Evaluation
1. **Visualize the solutions**: Plot the surface of \( u_{GD} \) and \( u_{MG} \).
2. **Analyze error convergence**:
   - Compute error metrics:
     \[ e_m^{GD} = \| A u_m^{GD} - f \|_F \]
     \[ e_m^{MG} = \| A u_m^{MG} - f \|_F \]
   - Plot both error functions as a function of iterations.
3. **Compute iteration efficiency**:
   - Determine the minimal \( m_1 \) such that \( e_{m_1}^{GD} < 10^{-5} \).
   - Determine the minimal \( m_2 \) such that \( e_{m_2}^{MG} < 10^{-5} \).
   - Measure and report computational time for both methods.

## Task 2: MgNet Implementation

### Modifications to Multigrid for MgNet
- Implement MgNet as shown in Algorithm 2.
- Address key questions:
  1. Where should **BatchNormalization** be placed?
  2. Should activation functions or BatchNormalization be used for Interpolation and Restriction in MgNet?

### Application to CIFAR-10/CIFAR-100
- Train MgNet on the CIFAR-10 dataset to achieve **94%.
- Experiment with different configurations to improve accuracy while ensuring **fewer than 40M free parameters**.