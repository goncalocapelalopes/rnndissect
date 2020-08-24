import torch
import pickle
from .nlp_utils import sentence_to_tensor
from ..settings import *
import spacy

nlp = spacy.load('en')

def predict_sentiment(model, sentence, vocab):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [vocab[t.lower()].index for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(DEVICE)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction