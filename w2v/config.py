import os

# Save parameters every a few SGD iterations as fail-safe
SAVE_PARAMS_EVERY = 1000

# annealing for SGD
ANNEAL_EVERY = 20000

#word2vec dimension size
dimVectors = 10

# Context size
C = 5

#no of iterations for SGD
iterations = 40000

#step size for SGD
step = 0.3

#Regularization choices for sentiment analysis
REGULARIZATION = [0.000001, 0.00001, 0.0001, 0.001]

#paths
CONFIG_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CONFIG_PATH, "..")
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")
PARAMS_PATH = os.path.join(MODELS_PATH, "params")
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
SST_PATH = os.path.join(DATA_PATH, "raw", "stanfordSentimentTreebank")
DOCS_PATH = os.path.join(PROJECT_ROOT, "docs")


