# Neural Networks for NLP
* [Course site](http://www.phontron.com/class/nn4nlp2019/schedule.html)

## Predicting the Next Word
* [Lecture 1](https://github.com/HJSang/ReadingNote/blob/master/NeuralNetworksforNLP/nn4nlp-02-lm.pdf)
* The simple idea is to catch the co-occurrence of individual words using the join probability distribution. The intuition is that the more often the pattern happens, the more likelihood the sentence is OK. In mathematical language, assume the context is <img src="https://render.githubusercontent.com/render/math?math=\(x_1,x_2,\cdots,x_{i-1}\)"> and the next word is <img src="https://render.githubusercontent.com/render/math?math=x_i">. Then, we can calculate the probability of a Sentence <img src="https://render.githubusercontent.com/render/math?math=X"> as <img src="https://render.githubusercontent.com/render/math?math=Pr(X)=\prod_{i=1}^I Pr(x_i|x_1,\cdots,x_{i-1})">. The big problem is how to preict <img src="https://render.githubusercontent.com/render/math?math=Pr(x_i\|x_1,\cdots,x_{i-1})">?

* Count-based Language Models:
  * Count up the frequency and divide: <img src="https://render.githubusercontent.com/render/math?math=Pr_{ML}(x_i\|x_{i-n+1},\cdots,x_{i-1}):=\frac{c(x_{i-n+1},\cdots,x_i)}{c(x_{i-n+1},\cdots,x_{i-1})}">.
  * Add smoothing, to deal with zero counts: <img src="https://render.githubusercontent.com/render/math?math=Pr(x_i|x_{i-n+1},\cdots, x_{i-1})=\lambdaPr_{ML}(x_i\|x_{i-n+1},\cdots,x_{i-1})+(1-\lambda)Pr(x_i\|x_{i-n+2},\cdots,x_{i-1})">.
  
* A refresher on Evaluation:
  * Log-likelihood: <img src="https://render.githubusercontent.com/render/math?math=LL(\epsilon_{tes})=\sum_{E\in \epsilon_{test}}\logP(E)">.
  * Per-word Log-Likelihood: <img src="https://render.githubusercontent.com/render/math?math=WLL(\epsilon_{test})=\frac{1}{\sum_{E\in\epsilon_{test}}|E|}\sum_{E\in\epsilon_{test}}\logP(E)">
  * Per-word (Cross) Entropy: <img src="https://render.githubusercontent.com/render/math?math=H(\epsilon_{test})=\frac{1}{\sum_{E\in\epsilon_{test}}}\sum_{E\in\epsilon_{test}}-\log_2P(E)">.
  * Perplexity: <img src="https://render.githubusercontent.com/render/math?math=ppl(\epsilon_{test})=2^{H(\epsilon_{test})}">.
