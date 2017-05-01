import numpy as np
import random
from w2v.word2vec.functions import softmax, sigmoid


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """
    Implements the Softmax cost function and gradients for word2vec

    :param predicted: ndarray, the predicted (center) word vector (v_c)
    :param target: integer, the index of the target word
    :param outputVectors: 2D ndarray, output word vectors (as rows) for all tokens
    :param dataset: an interface to the dataset
    :return: 
        cost:     cross entropy cost for softmax prediction
        gradPred: gradient with respect to predicted (center) word vector
        grad:     gradient with respect to output word vectors
    """

    predicted = predicted.T
    outputVectors = outputVectors.T
    assert (predicted.shape == (3, 1))
    assert (outputVectors.shape == (3, 5))

    yhat = softmax(outputVectors.T.dot(predicted))
    cost = -1.0 * np.log(yhat[target])

    delta = yhat
    delta[target] -= 1

    gradPred = outputVectors.dot(delta).flatten()

    grad = predicted.dot(delta.T)
    grad = grad.T

    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K=10):
    """
    Implements the negative sampling cost function and gradients for word2vec

    :param predicted: ndarray, the predicted (center) word vector(v_c)
    :param target: integer, the index of the target word
    :param outputVectors: 2D ndarray, output word vectors (as rows)
    :param dataset: an interface into the dataset
    :param K: integer, no of negative samples
    :return: 
        cost:     cost function for negative sampling
        gradPred: gradient with respect to predicted (input / center) word vector
        grad:     gradient with respect to output word vectors
    """

    grad = np.zeros(outputVectors.shape)
    gradPred = np.zeros(predicted.shape)

    indices = [target]
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices += [newidx]

    labels = np.array([1] + [-1 for k in xrange(K)]).reshape(-1, 1)
    vecs = outputVectors[indices, :]

    t = sigmoid(vecs.dot(predicted.T) * labels)
    cost = -np.sum(np.log(t))

    delta = labels * (t - 1)

    gradPred = delta.reshape((1, K + 1)).dot(vecs).flatten()

    gradtemp = delta.dot(predicted)

    for k in xrange(K + 1):
        grad[indices[k]] += gradtemp[k, :]

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """
    Implements the skipgram model in word2vec
    
    :param currentWord: string, current word
    :param C: integer, context size
    :param contextWords: list: a list (of size upto 2*C) of strings containing context words
    :param tokens: dict, a dict mapping words to their indices in the word vector list
    :param inputVectors: input word vectors (as rows)
    :param outputVectors: output word vectors (as rows)
    :param dataset: an interface into the dataset
    :param word2vecCostAndGradient: function computing costs and gradients (can be softmax (default) or neg sampling)
    :return: 
        cost: cost for the skip gram model
        grad: gradient with respect to word vectors
    """

    cur_word_token = tokens[currentWord]
    predicted = inputVectors[cur_word_token, np.newaxis, :]
    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    for i in range(len(contextWords)):
        target_i = tokens[contextWords[i]]
        _cost, _gradPred, _grad = word2vecCostAndGradient(predicted=predicted,
                                                          target=target_i,
                                                          outputVectors=outputVectors,
                                                          dataset=dataset)
        cost += _cost
        gradIn[cur_word_token, :] += _gradPred
        gradOut += _grad

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """
    Implements the continous-bag-of-words model for word2vec
    
    :param currentWord: string, current word
    :param C: integer, context size
    :param contextWords: list: a list (of size upto 2*C) of strings containing context words
    :param tokens: dict, a dict mapping words to their indices in the word vector list
    :param inputVectors: input word vectors (as rows)
    :param outputVectors: output word vectors (as rows)
    :param dataset: an interface into the dataset
    :param word2vecCostAndGradient: function computing costs and gradients (can be softmax (default) or neg sampling)
    :return: 
        cost: cost for the skip gram model
        grad: gradient with respect to word vectors
    """

    v_hat = np.zeros(inputVectors[0, np.newaxis, :].shape)

    for i in range(len(contextWords)):
        cur_word_token = tokens[contextWords[i]]
        v_hat += inputVectors[cur_word_token, np.newaxis, :]

    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    cost, _gradPred, gradOut = word2vecCostAndGradient(predicted=v_hat,
                                                       target=tokens[currentWord],
                                                       outputVectors=outputVectors,
                                                       dataset=dataset)

    for i in range(len(contextWords)):
        cur_word_token = tokens[contextWords[i]]
        gradIn[cur_word_token, :] += _gradPred

    return cost, gradIn, gradOut


def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    """
    Runs batchsize no of gradient descent steps on the word2vec model of choice (skipgram / cbow)
    :param word2vecModel: skipgram or cbow
    :param tokens: list, a list of tokens
    :param wordVectors: ndarray, word vectors stacked row-wise. The first N/2 vectors are input vectors,
     and the rest are output vectors
    :param dataset: an object that allows us to sample from the dataset
    :param C: integer, max context size
    :param word2vecCostAndGradient: function, cost function for word2vec: can be softmaxCostAndGradient (default) or
    negSamplingCostAndGradient
    :return: cost and gradient with respect to input / output word vectors
    """
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N / 2, :]
    outputVectors = wordVectors[N / 2:, :]
    for i in xrange(batchsize):
        C1 = random.randint(1, C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset,
                                     word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N / 2, :] += gin / batchsize / denom
        grad[N / 2:, :] += gout / batchsize / denom

    return cost, grad