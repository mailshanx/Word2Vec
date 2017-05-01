import glob
import random
import numpy as np
import os
import os.path as op
import cPickle as pickle
import w2v.config

def load_saved_params():
    """ A helper function that loads previously saved parameters and resets iteration start """
    st = 0
    glob_pat = "saved_params_*.npy"
    for f in glob.glob(os.path.join(w2v.config.PARAMS_PATH, glob_pat)):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter

    if st > 0:
        saved_param_fname = "saved_params_{}.npy".format(st)
        with open(os.path.join(w2v.config.PARAMS_PATH, saved_param_fname), "r") as f:
            params = pickle.load(f)
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None


def save_params(iter, params):
    saved_param_fname = "saved_params_{}.npy".format(iter)

    with open(os.path.join(w2v.config.PARAMS_PATH, saved_param_fname), "w") as f:
        pickle.dump(params, f)
        pickle.dump(random.getstate(), f)


def sgd(f, x0, step, iterations, postprocessing=None, useSaved=False, PRINT_EVERY=10):
    """
    Implements stochastic gradient descent
    
    :param f: funtion to optimize - it must take a single argument and yield two outputs, a cost and the 
              gradient with respect to the arguments
    :param x0: initial guess for input
    :param step: step size SGD
    :param iterations: total number of iterations
    :param postprocessing: postprocessing for input parameters (pass the normalization function here for word2vec)
    :param useSaved: uses saved params if True. Default value is False
    :param PRINT_EVERY: how often should we print progress
    :return: 
        x: parameter value after SGD terminates
    """

    ANNEAL_EVERY = w2v.config.ANNEAL_EVERY

    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx;
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)

        if state:
            random.setstate(state)
    else:
        start_iter = 0

    x = x0

    if not postprocessing:
        postprocessing = lambda x: x

    expcost = None

    for iter in xrange(start_iter + 1, iterations + 1):

        cost = None

        #core of the SGD process
        cost, grad = f(x)
        x = x - step * grad
        x = postprocessing(x)
        ###
        
        if iter % PRINT_EVERY == 0:
            if not expcost:
                expcost = cost
            else:
                expcost = .95 * expcost + .05 * cost
            print "iter %d: %f" % (iter, expcost)

        if useSaved and iter % w2v.config.SAVE_PARAMS_EVERY == 0:
            save_params(iter, x)

        if iter % ANNEAL_EVERY == 0:
            step *= 0.5

    return x
