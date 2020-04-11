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
from tqdm import tqdm, tqdm_notebook, tnrange
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import numpy as np

class FineTune(nn.Module):
	def __init__(self):
		super(FineTune, self).__init__()
		self.lstm = nn.LSTM(2048, 256, num_layers=1, bidirectional=True)
		self.loss = nn.CrossEntropyLoss(reduction='mean')
		self.out = nn.Linear(512, 2)

	def compute_loss(self, pred, target):
		return self.loss(pred, target)

	def permute_seq(self, seq, perm):
		seq = seq[perm,]
		return seq

	def forward(self, seq, permutation):
		lstm_in = self.permute_seq(seq, permutation)
		lstm_in = lstm_in.unsqueeze(1)
		lstm_out, _ = self.lstm(lstm_in)
		max_pool = F.adaptive_max_pool1d(lstm_out.permute(1,2,0),1).view(lstm_in.size(1),-1)
		outp = self.out(max_pool)
		return outp


def validate(m, inputs, perms, outputs):
	m.eval()
	correct, total = 0, 0 
	m_zero, m_one = 0, 0
	g_zero, g_one = 0, 0
	for i, p, o in (zip(inputs, perms, outputs)):
		pred = m(i, p)
		if np.argmax(pred.data).item() == o:
			correct += 1
		if np.argmax(pred.data).item() == 0:
			m_zero += 1
		else:
			m_one += 1
		if o == 0:
			g_zero += 1
		else:
			g_one += 1
		total += 1
	return correct, total, m_zero, m_one, g_zero, g_one


def model_handler(inputs, permutations, outputs):
	n_train = 8000
	x = list(zip(inputs, permutations, outputs))
	random.shuffle(x)
	inputs, permutations, outputs = zip(*x)
	inputs, permutations, outputs = list(inputs), list(permutations), list(outputs)
	train_inputs, train_permutations, train_outputs = inputs[:n_train], permutations[:n_train], outputs[:n_train]
	dev_inputs, dev_permutations, dev_outputs = inputs[n_train:9000], permutations[n_train:9000], outputs[n_train:9000]
	test_inputs, test_permutations, test_outputs = inputs[9000:], permutations[9000:], outputs[9000:]
	print([len(y) for y in [train_inputs, train_outputs, dev_inputs, dev_outputs, test_inputs, test_outputs]])
	finetuner = FineTune()
	optimizer = optim.Adam(finetuner.parameters())
	batch_size = 16
	print("Begin Training")
	train_accuracy = {}
	valid_accuracy = {}
	for epoch in range(21):
		print("Starting epoch: {}".format(epoch))
		minibatches = n_train // batch_size
		data_permutation = torch.randperm(n_train)
		finetuner.train()
		for group in tqdm(range(minibatches)):
			finetuner.zero_grad()
			total_loss = None
			start = group * batch_size
			end = min((group + 1) * batch_size, n_train)
			for i in range(start, end):
				j = data_permutation[i]
				sent, permutation, output = train_inputs[j], train_permutations[j], train_outputs[j]
				pred = finetuner(sent, permutation)
				loss_ = finetuner.compute_loss(pred, torch.tensor([output]))
				if not total_loss:
					total_loss = loss_
				else:
					total_loss += loss_
			if random.randint(0, 100) > 98:
				print('Training Loss for Group {} in Epoch {}: {}'.format(group, epoch, total_loss))
			total_loss.backward()
			optimizer.step()
		print("Validation for epoch {}".format(epoch))
		correct, total, m_zero, m_one, g_zero, g_one = validate(finetuner, dev_inputs, dev_permutations, dev_outputs)
		print("Validation accuracy for epoch {}: {}; {} / {}".format(epoch, correct / total, correct, total))
		print("Model Zeros: {}, Model Ones: {}, Gold Zeros: {}, Gold Ones {}".format(m_zero, m_one, g_zero, g_one))
		valid_accuracy[epoch] = round(correct / total, 3)
		print(valid_accuracy)
		print("Validation(Training) for epoch {}".format(epoch))
		correct, total, m_zero, m_one, g_zero, g_one = validate(finetuner, train_inputs[:500], train_permutations[:500], train_outputs[:500])
		print("Training accuracy for epoch {}: {}; {} / {}".format(epoch, correct / total, correct, total))
		print("Model Zeros: {}, Model Ones: {}, Gold Zeros: {}, Gold Ones {}".format(m_zero, m_one, g_zero, g_one))
		train_accuracy[epoch] = round(correct / total, 3)
		print(train_accuracy)
	return train_accuracy, valid_accuracy


if __name__ == '__main__':
	main()

