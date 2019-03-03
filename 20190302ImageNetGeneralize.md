# Read Note for [Do ImageNet Classifiers Generalize to ImageNet?](http://people.csail.mit.edu/ludwigs/papers/imagenet.pdf)

## Abstract
 This paper uses experiments to study the generalization of ImageNet classifiers on the new test data set. The authors find that:
 1. They evaluate a broad range of models and find accuracy drops 3% -- 15% on CIFAR-10 and 11% -- 14% on ImageNet.
 2. However, accuracy gaisn on the original test sets translate to larger gains on the new test sets. 
 3. The accuracy drops are not caused by adaptivity, but the models inability to generalize to slightly "harder" images than those found in the original test sets.
 
 ## Introduction 
 1. The generalization in this paper is defined as the performance of the model on a held-out test set.
 2. The conventional guess is that such drops arise because the mdeos have been adapted to the specific images in the original test sets.  However, the experiments show that the relative order of models is almost exactly preserved on our new test sets: the model with highest accuracy on the original test sets are still the models with highest accuracy on the new test sets.
 3. Adaptivity is therefore an unlikely explanation for the accuracy drops.
 
 ## Error Decomposition
 1. True data distribution D and true sample realizations S
 2. Proposed data distribution D' and corresponding sample realizations S'
 3. The miss classification error on S is L_S(\hat f).
 4. L_S - L_S' = (L_S - L_D) + (L_D- L_D') + (L_D' + L_S'): Adapativity gap + Distribution Gap + Generalization Gap
 5. By construction, the new test data set S' is independent of the existing classifier \hat f. 
 
 
