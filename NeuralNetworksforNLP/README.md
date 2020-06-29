# Neural Networks for NLP
* [Course site](http://www.phontron.com/class/nn4nlp2019/schedule.html)

## Predicting the Next Word
* [Lecture 1](https://github.com/HJSang/ReadingNote/blob/master/NeuralNetworksforNLP/nn4nlp-02-lm.pdf)
* The simple idea is to catch the co-occurrence of individual words using the join probability distribution. The intuition is that the more often the pattern happens, the more likelihood the sentence is OK. In mathematical language, assume the context is <img src="https://render.githubusercontent.com/render/math?math=\(x_1,x_2,\cdots,x_{i-1}\) = -1"> and the next word is <img src="https://render.githubusercontent.com/render/math?math=x_i = -1">. Then, we can calculate the probability of a Sentence <img src="https://render.githubusercontent.com/render/math?math=X = -1"> as <img src="https://render.githubusercontent.com/render/math?math=Pr(X)=\prod_{i=1}^I Pr(x_i|x_1,\cdots,x_{i-1}) = -1">.
