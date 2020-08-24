import os
import torch

MODEL_DIR = "/home/goncalo/Documents/rnndissect-master/model"
ASSETS_DIR = "/home/goncalo/Documents/rnndissect-master/assets"

VOCAB_PATH = os.path.join(ASSETS_DIR, "imdb_vocab.pickle")

ACTS_DIR = "/home/.rnndissect/activations"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
