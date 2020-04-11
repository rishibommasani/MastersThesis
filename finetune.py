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

class FineTune(nn.Module):
	def __init__(self):
		super(FineTune, self).__init__()
		self.lstm = nn.LSTM(768, 256, num_layers=1, bidirectional=bidirectional)
		self.loss_entity = nn.CrossEntropyLoss(reduction='mean')
		self.out = nn.Linear(256, 2)

	def permute_seq(seq, perm):
		return seq

	def forward(self, seq, permutation):
		lstm_in = self.permute_seq(seq, permutation)
		lstm_out, _ = self.lstm(lstm_in)
		max_pool = F.adaptive_max_pool1d(lstm_out.permute(1,2,0),1).view(seq.size(1),-1)
		outp = self.out(max_pool)
		return F.log_softmax(outp)