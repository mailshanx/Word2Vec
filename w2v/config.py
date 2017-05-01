import os

# Save parameters every a few SGD iterations as fail-safe
SAVE_PARAMS_EVERY = 1000
# annealing for SGD
ANNEAL_EVERY = 20000

#paths
CONFIG_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CONFIG_PATH, "..")
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")
PARAMS_PATH = os.path.join(PROJECT_ROOT, "models", "params")
DATA_PATH = os.path.join(PROJECT_ROOT, "data")
