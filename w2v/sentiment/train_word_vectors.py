import random
import numpy as np
from w2v.utils.utils import StanfordSentiment
from w2v.word2vec.sgd import sgd
from w2v.word2vec.word2vec import word2vec_sgd_wrapper, skipgram, negSamplingCostAndGradient
from w2v import config

def train_word_vectors():
    random.seed(314)
    dataset = StanfordSentiment()
    tokens = dataset.tokens()
    nWords = len(tokens)
    dimVectors = config.dimVectors
    C = config.C
    random.seed(31415)
    np.random.seed(9265)
    wordVectors = np.concatenate(((np.random.rand(nWords, dimVectors) - .5) / \
                                   dimVectors, np.zeros((nWords, dimVectors))), axis=0)
    wordVectors0 = sgd(lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C,
                                    negSamplingCostAndGradient),
                                    x0 = wordVectors, step = config.step,
                                    iterations = config.iterations, postprocessing=None,
                                    useSaved=True, PRINT_EVERY=500)

    print "check: cost at convergence should be around or below 10"
    # sum the input and output word vectors
    wordVectors = (wordVectors0[:nWords, :] + wordVectors0[nWords:, :])
    print "finished training word vectors "

    return wordVectors


if __name__=='__main__':
    train_word_vectors()

