# Read Note for [Stability and Generalization of Learning Algorithms that Converge to Global Optima](https://arxiv.org/pdf/1710.08402.pdf)
## Summary
In this paper, the autors establish generalization bounds for learning algorithms that converge to global minima.
1. The results are established for nonconvex functions satisfying the Polyak-Lojasieewicz and the quadratic growth condition.
2. They use the black-box results to establish the stability of optimization algorithms.
3. They show comparable stability for SGD and GD in PL setting.
4. There exist simple neural networs with multiple local minima where SGD is table but GD is not.

## Introduction
1. A useful proxy for analyzing generalization is stability.
2. A traning algorithm is atble if a small change in the traning set results in small difference in the output.

## Priliminary
1.  An algorithm A is uniformly epsilon stable, if for all data sets S, S' differing in at most one example, we have the sup_z of expectation of loss(A(S),z) - loss(A(S'),z) <= epsion.
2. Suppose A is uniformly epsion stable, then | expectation w.r.t. A and S [ R_S(A(S)) - R(A(S))]| <= epsion.
3. Pointwise Hypothesis Stablility: instead of sup_z, the stability inequality holds for z_i, i =1,2,3,..n.
4. Suppose a learning algorithm A is pointwise hypothesis stability with bound beta and loss function is bounded from M, then the generalization error bound <= sqrt{(M^2 + 12 Mnbeta )/(2ndelta)} with probability at least 1- delta.


