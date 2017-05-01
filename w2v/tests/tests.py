import numpy as np
from w2v.word2vec.functions import softmax, sigmoid, sigmoid_grad, normalizeRows
from w2v.word2vec.gradcheck import gradcheck_naive
from w2v.word2vec.word2vec import word2vec_sgd_wrapper, skipgram, cbow, negSamplingCostAndGradient
import unittest
import random


class FunctionsTests(unittest.TestCase):
    """
    Tests function definitions for softmax, sigmoid and normalize_rows in functions.py
    """
    def test_softmax(self):
        """
        Testing the softmax implementation
        """
        test1 = softmax(np.array([1, 2]))
        self.assertLessEqual(np.amax(np.fabs(test1 - np.array([0.26894142, 0.73105858]))), 1e-6,
                             msg="q1: softmax(np.array([1, 2])) \n "
                                 "test1: {}, answer = np.array([0.26894142, 0.73105858])".format(test1))

        test2 = softmax(np.array([[1001, 1002], [3, 4]]))
        answer2 = np.array([[0.26894142, 0.73105858], [0.26894142, 0.73105858]])
        self.assertLessEqual(np.amax(np.fabs(test2 - answer2)), 1e-6,
                             msg="q2: softmax(np.array([[1001, 1002], [3, 4]])) \n" 
                                 " test2 : {}, answer = {} ".format(test2, answer2))

        test3 = softmax(np.array([[-1001, -1002]]))
        answer3 = np.array([0.73105858, 0.26894142])
        self.assertLessEqual(np.amax(np.fabs(test3 - answer3)),  1e-6,
                             msg="q3: softmax(np.array([[-1001, -1002]])) \n" 
                                 " test3 = {}, answer3 = {}".format(test3, answer3))

    def test_sigmoid(self):
        """
        testing sigmoid implementation
        """
        x = np.array([[1, 2], [-1, -2]])
        f = sigmoid(x)
        g = sigmoid_grad(f)
        ans_f = np.array([[0.73105858, 0.88079708], [0.26894142, 0.11920292]])
        ans_g = np.array([[0.19661193, 0.10499359], [0.19661193, 0.10499359]])

        self.assertLessEqual(np.amax(f - ans_f), 1e-6,
                             msg="x: {}, f: {}, ans_f: {}".format(x, f, ans_f))

        self.assertLessEqual(np.amax(g - ans_g), 1e-6, msg="x: {}, g: {}, ans_g: {}".format(x, g, ans_g))

    def test_normalize_rows(self):
        x = normalizeRows(np.array([[3.0, 4.0], [1, 2]]))
        # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
        self.assertEquals(x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all(), True)


class GradCheck(unittest.TestCase):
    """
    tests gradcheck_naive in gradcheck.py
    """
    def setUp(self):
        self.quad  = lambda x: (np.sum(x ** 2), x * 2)

    def test_gradcheck_1d(self):
        gradcheck_naive(self.quad, np.array(123.456))  # scalar test
        gradcheck_naive(self.quad, np.random.randn(3, ))  # 1-D test
        gradcheck_naive(self.quad, np.random.randn(4, 5))  # 2-D test


class Word2Vec(unittest.TestCase):

    def setUp(self):
        self.dataset = type('dummy', (), {})()

        def dummySampleTokenIdx():
            return random.randint(0, 4)

        def getRandomContext(C):
            tokens = ["a", "b", "c", "d", "e"]
            return tokens[random.randint(0, 4)], [tokens[random.randint(0, 4)] \
                                                  for i in xrange(2 * C)]

        self.dataset.sampleTokenIdx = dummySampleTokenIdx
        self.dataset.getRandomContext = getRandomContext

        random.seed(31415)
        np.random.seed(9265)
        self.dummy_vectors = normalizeRows(np.random.randn(10, 3))
        self.dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])

    def test_word2vec_skipgram(self):
        print "==== Gradient check for skip-gram ===="
        self.assertTrue(gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram,
                                        self.dummy_tokens, vec, self.dataset, 5),
                                        self.dummy_vectors))
        g_check = gradcheck_naive( lambda vec: word2vec_sgd_wrapper(skipgram, self.dummy_tokens, vec,
                  self.dataset, 5, negSamplingCostAndGradient), self.dummy_vectors)
        self.assertTrue(g_check)


    def test_word2vec_cbow(self):
        print "\n==== Gradient check for CBOW      ===="
        g_check = gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, self.dummy_tokens, vec, self.dataset, 5),
                        self.dummy_vectors)
        self.assertTrue(g_check)

        g_check = gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, self.dummy_tokens, vec,
                        self.dataset, 5, negSamplingCostAndGradient),
                        self.dummy_vectors)
        self.assertTrue(g_check)



if __name__=='__main__':
    unittest.main()