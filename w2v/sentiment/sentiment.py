from w2v.utils.utils import StanfordSentiment
from w2v.word2vec.sgd import sgd, load_saved_params
from w2v.sentiment.models import softmaxRegression, getSentenceFeature, accuracy, softmax_wrapper
import w2v.config
import numpy as np
import random
import matplotlib.pyplot as plt
import os

REGULARIZATION = w2v.config.REGULARIZATION

class Sentiment(object):

    def load_dataset(self):
        # Load the dataset
        self.dataset = StanfordSentiment()
        self.tokens = self.dataset.tokens()
        self.nWords = len(self.tokens)

    def load_wordvectors(self):
        # Load the word vectors we trained earlier
        _, self.wordVectors0, _ = load_saved_params()
        self.wordVectors = (self.wordVectors0[:self.nWords, :] + self.wordVectors0[self.nWords:, :])
        self.dimVectors = self.wordVectors.shape[1]

    def load_trainset(self):
        # Load the train set
        self.trainset = self.dataset.getTrainSentences()
        self.nTrain = len(self.trainset)
        self.trainFeatures = np.zeros((self.nTrain, self.dimVectors))
        self.trainLabels = np.zeros((self.nTrain,), dtype=np.int32)
        for i in xrange(self.nTrain):
            words, self.trainLabels[i] = self.trainset[i]
            self.trainFeatures[i, :] = getSentenceFeature(self.tokens, self.wordVectors, words)

    def load_devset(self):
        # Prepare dev set features
        self.devset = self.dataset.getDevSentences()
        self.nDev = len(self.devset)
        self.devFeatures = np.zeros((self.nDev, self.dimVectors))
        self.devLabels = np.zeros((self.nDev,), dtype=np.int32)
        for i in xrange(self.nDev):
            words, self.devLabels[i] = self.devset[i]
            self.devFeatures[i, :] = getSentenceFeature(self.tokens, self.wordVectors, words)

    def _train(self, verbose = True):
        """
        Implements the core part of the training procedure
        :param verbose: 
        :return: 
        """

        # Try various regularization parameters
        self.results = []
        for regularization in REGULARIZATION:
            random.seed(3141)
            np.random.seed(59265)
            weights = np.random.randn(self.dimVectors, 5)
            print "Training for reg=%f" % regularization

            # We will do batch optimization
            weights = sgd(lambda weights: softmax_wrapper(self.trainFeatures, self.trainLabels,
                                          weights, regularization), weights, 3.0, 10000, PRINT_EVERY=500)

            # Test on train set
            _, _, pred = softmaxRegression(self.trainFeatures, self.trainLabels, weights)
            trainAccuracy = accuracy(self.trainLabels, pred)
            print "Train accuracy (%%): %f" % trainAccuracy

            # Test on dev set
            _, _, pred = softmaxRegression(self.devFeatures, self.devLabels, weights)
            devAccuracy = accuracy(self.devLabels, pred)
            print "Dev accuracy (%%): %f" % devAccuracy

            # Save the results and weights
            self.results.append({
                "reg": regularization,
                "weights": weights,
                "train": trainAccuracy,
                "dev": devAccuracy})

        if verbose:
            # Print the accuracies
            print ""
            print "=== Recap ==="
            print "Reg\t\tTrain\t\tDev"
            for result in self.results:
                print "%E\t%f\t%f" % (
                    result["reg"],
                    result["train"],
                    result["dev"])
            print ""

        # Pick the best regularization parameters
        self.BEST_REGULARIZATION = None
        self.BEST_WEIGHTS = None


        _, best_result = max(enumerate(self.results), key=lambda x: x[1]["dev"])
        self.BEST_REGULARIZATION = best_result["reg"]
        self.BEST_WEIGHTS = best_result["weights"]

    def test_accuracy(self):
        # Test your findings on the test set
        testset = self.dataset.getTestSentences()
        nTest = len(testset)
        testFeatures = np.zeros((nTest, self.dimVectors))
        testLabels = np.zeros((nTest,), dtype=np.int32)
        for i in xrange(nTest):
            words, testLabels[i] = testset[i]
            testFeatures[i, :] = getSentenceFeature(self.tokens, self.wordVectors, words)

        _, _, pred = softmaxRegression(testFeatures, testLabels, self.BEST_WEIGHTS)
        print "Best regularization value: %E" % self.BEST_REGULARIZATION
        print "Test accuracy (%%): %f" % accuracy(testLabels, pred)
        return accuracy(testLabels, pred)

    def visualize(self):
        """
        visualize training results: plot accuracy vs regularization for train and dev and save plot to docs
        :return: 
        """
        regplot_filepath = os.path.join(w2v.config.DOCS_PATH, "reg_v_acc.png")
        plt.plot(REGULARIZATION, [x["train"] for x in self.results])
        plt.plot(REGULARIZATION, [x["dev"] for x in self.results])
        plt.xscale('log')
        plt.xlabel("regularization")
        plt.ylabel("accuracy")
        plt.legend(['train', 'dev'], loc='upper left')
        plt.savefig(regplot_filepath)
        print "saved plot in {}".format(regplot_filepath)

    def train(self):
        self.load_dataset()
        self.load_wordvectors()
        self.load_trainset()
        self.load_devset()
        self._train()


if __name__=='__main__':
    sentiment = Sentiment()
    sentiment.train()
    sentiment.test_accuracy()
    sentiment.visualize()









