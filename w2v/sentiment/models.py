import numpy as np
import random
from w2v.utils.utils import StanfordSentiment
from w2v.word2vec.functions import softmax
from w2v.word2vec.gradcheck import gradcheck_naive
from w2v.word2vec.sgd import sgd, load_saved_params


def getSentenceFeature(tokens, wordVectors, sentence):
    """
    Obtains the sentence feature for sentiment analysis by averaging its word vectors
    :param tokens: a dictionary that maps words to their indices in    
                    the word vector list   
    :param wordVectors: word vectors (each row) for all tokens   
    :param sentence: a list of words in the sentence of interest 
    :return: 
        sentVector: feature vector for the sentence
    """

    sentVector = np.zeros((wordVectors.shape[1],))

    N = len(sentence)
    indices = [tokens[word] for word in sentence]
    word_vecs = wordVectors[indices, :]
    sentVector = np.sum(word_vecs, axis=0) / N

    return sentVector


def softmaxRegression(features, labels, weights, regularization=0.0, nopredictions=False):
    """
    Implements Softmax regresion
    :param features: feature vectors, each row is a feature vector
    :param labels: labels corresponding to the feature vectors
    :param weights: weights of the regressor
    :param regularization: L2 regularization constant
    :param nopredictions: 
    :return: 
        cost: cost of the regressor
        grad: gradient of the regressor cost with respect to its weights
        pred: label predictions of the regressor 
    """

    prob = softmax(features.dot(weights))
    if len(features.shape) > 1:
        N = features.shape[0]
    else:
        N = 1

    # A vectorized implementation of    1/N * sum(cross_entropy(x_i, y_i)) + 1/2*|w|^2
    cost = np.sum(-np.log(prob[range(N), labels])) / N
    cost += 0.5 * regularization * np.sum(weights ** 2)


    y_hat = prob
    y = np.zeros(y_hat.shape)
    y[range(N), labels] = 1.0
    grad = features.T.dot((y_hat - y)) / N
    grad += (2.0 * 0.5 * regularization * weights)
    pred = np.argmax(prob, axis=1)


    if nopredictions:
        return cost, grad
    else:
        return cost, grad, pred

def accuracy(y, yhat):
    """ Precision for classifier """
    #assert(y.shape == yhat.shape)
    return np.sum(y == yhat) * 100.0 / y.size

def softmax_wrapper(features, labels, weights, regularization = 0.0):
    """
    wraps models.softmaxRegression. Inputs are the same as softmaxRegression
    
    :param features: 
    :param labels: 
    :param weights: 
    :param regularization: 
    :return: 
    """
    cost, grad, _ = softmaxRegression(features, labels, weights,
        regularization)
    return cost, grad

