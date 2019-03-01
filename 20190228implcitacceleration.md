# Read note for "[On the Optimization of Deep Networks: implicit Acceleration by Overparameterization](https://arxiv.org/pdf/1802.06509.pdf)"

## Abstract
Increasing depth can speed up optimization:
1. Additional layer leads to over-parameterization- linear neutal networks.
2. Theory and experiments show that depth acts as a preconditioner which **may** accelerate convergence.
3. For linear regression with l_p loss, p > 2, gradient descent benifit from transitioning to a nonconvex over-parameterized objective.


##Introduction
Difficulties in optimizing deeper networks are long clear: 
1. Gradient gets buried as it propagates through many layers. 
2. Batch normalization & residual connections have somewaht alleviated these difficulties in practice.
This paper present a surprising result: deeper, better!

