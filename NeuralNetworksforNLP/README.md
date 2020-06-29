# Neural Networks for NLP
* [Course site](http://www.phontron.com/class/nn4nlp2019/schedule.html)

## Predicting the Next Word
* [Lecture 1](https://github.com/HJSang/ReadingNote/blob/master/NeuralNetworksforNLP/nn4nlp-02-lm.pdf)
* The simple idea is to catch the co-occurrence of individual words using the join probability distribution. The intuition is that the more often the pattern happens, the more likelihood the sentence is OK. In mathematical language, assume the context is <img src="https://render.githubusercontent.com/render/math?math=(x_1,x_2,\cdots,x_{i-1})"> and the next word is <img src="https://render.githubusercontent.com/render/math?math=x_i">. Then, we can calculate the probability of a Sentence <img src="https://render.githubusercontent.com/render/math?math=X"> as <img src="https://render.githubusercontent.com/render/math?math=Pr(X)=\prod_{i=1}^I Pr(x_i|x_1,\cdots,x_{i-1})">. The big problem is how to preict <img src="https://render.githubusercontent.com/render/math?math=Pr(x_i|x_1,\cdots,x_{i-1})">?

* Count-based Language Models:
  * Count up the frequency and divide: <img src="https://render.githubusercontent.com/render/math?math=Pr_{ML}(x_i|x_{i - n + 1},\cdots,x_{i-1}):=\frac{c(x_{i - n + 1},\cdots,x_i)}{c(x_{i - n + 1},\cdots,x_{i-1})}">.
  * Add smoothing, to deal with zero counts: <img src="https://render.githubusercontent.com/render/math?math=Pr(x_i|x_{i - n + 1},\cdots, x_{i-1})=\lambda Pr_{ML}(x_i|x_{ i - n + 1},\cdots,x_{i-1}) + (1-\lambda)Pr(x_i|x_{i - n + 2},\cdots,x_{i-1})">.
  
* A refresher on Evaluation:
  * Log-likelihood: <img src="https://render.githubusercontent.com/render/math?math=LL(\epsilon_{tes})=\sum_{E\in \epsilon_{test}}\logP(E)">.
  * Per-word Log-Likelihood: <img src="https://render.githubusercontent.com/render/math?math=WLL(\epsilon_{test})=\frac{1}{\sum_{E\in\epsilon_{test}}|E|}\sum_{E\in\epsilon_{test}}\logP(E)">
  * Per-word (Cross) Entropy: <img src="https://render.githubusercontent.com/render/math?math=H(\epsilon_{test})=\frac{1}{\sum_{E\in\epsilon_{test}}}\sum_{E\in\epsilon_{test}}-\log_2P(E)">.
  * Perplexity: <img src="https://render.githubusercontent.com/render/math?math=ppl(\epsilon_{test})=2^{H(\epsilon_{test})}">.

* What can we do w/ LMs?
  * Score sentences: <img src="https://render.githubusercontent.com/render/math?math=Pr(X)">
  * Generate sentences
* Problems and Solutions?
  * Cannot share strength among similar words. Solution: class based language models.
  * Cannot condition on context with intervening words: Dr. Jane Smith VS Dr. Gertrude Smith. Solution: skip-gram language models
  * Cannot handle long-distance dependencies:
  ```
  for tennis class he wanted to buy his own racquet
  for programming class he wanted to buy his own computer
  ```
  Solution: cache, trigger, topic, syntatic models, etc.

* An alternative: Featurized Log-Linear Models:
  * Calculate features of the context
  * Based on the features, calculate probabilities
  * Optimize feature weights using gradient descent, etc.
  * Example: Previous words: "giving a", predict the next word in {a, the, talk, gift, hat,...}. $b=(3.0,2.5,-0.2,0.1,1.2,\cdots)^T$ represents how likely they are? $w_{1,a}=(-6.0,-5.1,0.2,0.1,0.5,\cdots)^T$: how likely are they given the prev word is "a"? $w_{2,giving}=(-0.2,-0.3,1.0,2.0,-1.2,\cdots)^T$: How likely are they given 2nd prev word is "giving"? $s=(-3.2,-2.9,1.0,2.2,0.6,\cdots)^T$: Total socre.
