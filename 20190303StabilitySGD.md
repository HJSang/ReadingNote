# Read Note for [Data-Dependent Stability of Stochastic Gradient Descent](https://arxiv.org/pdf/1703.01678.pdf)

## Summary
The autors employ the algorithm stability of SGD to develop generalization bound. 
1. In the non-convex case, they prove that the expected curvature of the objective function aroound the initializationpoint has crucial influence on the generalization error.
2. In the convex case, they show that the bound depends on the risk at the initilization point.

## Stability of SGD
1. ![Notation](https://github.com/HJSang/ReadingNote/blob/master/Screen%20Shot%202019-03-03%20at%2009.25.28.png)
2. ![Stability](https://github.com/HJSang/ReadingNote/blob/master/Screen%20Shot%202019-03-03%20at%2009.28.01.png)
3. From this theorem, we can see that the generalization error bound is determinated by uniform stability bound. From the definition of uniform stability, we can see that: stability means less depends on data.

## Main Result
![Theorem](https://github.com/HJSang/ReadingNote/blob/master/Screen%20Shot%202019-03-03%20at%2009.28.52.png)
The assumptions are commonly used in most of context. 

![convex](https://github.com/HJSang/ReadingNote/blob/master/Screen%20Shot%202019-03-03%20at%2009.29.02.png)
![Nonconvex](https://github.com/HJSang/ReadingNote/blob/master/Screen%20Shot%202019-03-03%20at%2009.29.19.png)

