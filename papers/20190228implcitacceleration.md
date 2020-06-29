# Read note for "[On the Optimization of Deep Networks: implicit Acceleration by Overparameterization](https://arxiv.org/pdf/1802.06509.pdf)"

## Abstract
Increasing depth can speed up optimization:
1. Additional layer leads to over-parameterization- linear neutal networks.
2. Theory and experiments show that depth acts as a preconditioner which **may** accelerate convergence.
3. For linear regression with l_p loss, p > 2, gradient descent benifit from transitioning to a nonconvex over-parameterized objective.


## Introduction
Difficulties in optimizing deeper networks are long clear: 
1. Gradient gets buried as it propagates through many layers. 
2. Batch normalization & residual connections have somewaht alleviated these difficulties in practice.

This paper present a surprising result: deeper, better!
1. For simple linear regression  with l_p loss, p>2, overparameterization via depth can significantly speed up training.
2. It was also faster than **AdaGrad** and **Adadelta**.
3. The implicit acceleration of overparameterization is different from standard regularization.

## L_p Regression
**Observation1**: Gradient descent over L(W1,W2), with fixed small learning rate and near-zero initialization, is equivalent to gradient descent over l(w) with particular adaptive learning rate and momentum terms.

## Linear Neural Networks
L^N is a depth-N network loss function. The gradient descent on L^N, a complicated and seemingly pointless overparameterization, can be directly rewritten as a particular preconditioner scheme over gradient descent on L^1.'

## Comments
1. Why the acceleration does not work for L_2 loss? I guess this is due to the manifold of loss function.
2. I guess that the acceleration is due to more global optimizers. We know that, under overparameterization, there exist multiple global minimizers. More parameters, more global minimizers. The L_p loss makes the loss function shaper and the more global minimizers make the optimization easier.
3. My question is that does over-parameterization benifit generalization, too? 
