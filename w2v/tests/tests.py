import numpy as np
from w2v.word2vec.functions import softmax, sigmoid, sigmoid_grad
from w2v.word2vec.gradcheck import gradcheck_naive
import unittest


class FunctionsTests(unittest.TestCase):

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


class GradCheck(unittest.TestCase):

    def setUp(self):
        self.quad  = lambda x: (np.sum(x ** 2), x * 2)

    def test_gradcheck_1d(self):
        gradcheck_naive(self.quad, np.array(123.456))  # scalar test
        gradcheck_naive(self.quad, np.random.randn(3, ))  # 1-D test
        gradcheck_naive(self.quad, np.random.randn(4, 5))  # 2-D test

if __name__=='__main__':
    unittest.main()