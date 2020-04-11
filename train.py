import torch
import os
from torch import nn
from torch.nn import init
from tqdm import tqdm
import flair
from flair.data import Sentence 
from flair.embeddings import ELMoEmbeddings
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pretrained import PretrainedEmbeddings
from finetune import FineTune
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from itertools import permutations
import random
from collections import Counter
import spacy
nlp = spacy.load('en_core_web_lg')
from tqdm import tqdm, tqdm_notebook, tnrange
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from model import model_handler
from main import external_use, external_use_verbose


def fetch_data():
	quote_sentences = []
	plot_sentences = []
	pickled = True

	if pickled:
		plot_sentences = pickle.load(open('plot.txt', 'rb'))
		quote_sentences = pickle.load(open('quote.txt', 'rb'))
	else:
		with open('plot.tok.gt9.5000', mode = 'r', encoding='utf-8', errors='ignore') as f:
			for row in f:
				plot_sentences.append(row)
		print(len(plot_sentences))
		pickle.dump(plot_sentences, open('plot.txt', 'wb'))
		with open('quote.tok.gt9.5000', mode = 'r', encoding='utf-8', errors='ignore') as f:
			for row in f:
				quote_sentences.append(row)
		print(len(quote_sentences))
		pickle.dump(quote_sentences, open('quote.txt', 'wb'))
		exit()
	return plot_sentences, quote_sentences


def get_all_permutations(sentences, file_name, pickled):
	return external_use_verbose(sentences, file_name, pickled = pickled)


def get_permutations(sentences, order):
	pickled = False
	all_permutations = get_all_permutations(sentences, 'subj_permutations_no_minla', pickled)
	exit()
	print(all_permutations[-1])
	exit()
	permutations = [perm[order]['permutation'] for perm in all_permutations] 
	return permutations


def main():
	order = 'minLA' # One of standard, random, bandwidth, or minLA
	print("Fetching data")
	plots, quotes = fetch_data()
	embeds = PretrainedEmbeddings()
	print("Fetching Embeddings")
	plot_embeddings, quote_embeddings = embeds(plots, tag='plot'), embeds(quotes, tag='quote')
	inputs = plot_embeddings + quote_embeddings
	print("Computing permutations")
	permutations = get_permutations(plots + quotes, order)
	outputs = [0] * 5000 + [1] * 5000
	print("Running model")
	train_accuracy, valid_accuracy = model_handler(inputs, permutations, outputs)
	print(train_accuracy, valid_accuracy)
	file_reader = open(order + ".results.txt", 'w+')
	file_reader.write(str([train_accuracy, valid_accuracy]))
	file_reader.close()
	exit()

if __name__ == '__main__':
	main()