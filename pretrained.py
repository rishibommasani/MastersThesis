import torch
import os
from torch import nn
from torch.nn import init
from tqdm import tqdm
import flair
from flair.data import Sentence 
from flair.embeddings import ELMoEmbeddings
import pickle
from pytorch_pretrained_bert import BertTokenizer, BertModel
import spacy
nlp = spacy.load('en_core_web_lg')
import logging
logging.basicConfig(level=logging.INFO)
from allennlp.modules.elmo import Elmo, batch_to_ids
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


class PretrainedEmbeddings(nn.Module):
    def __init__(self):
        super(PretrainedEmbeddings, self).__init__()
        self.elmo = Elmo(options_file, weight_file, 2, dropout=0)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.eval()

        
    def forward(self, sentences, tag=''):
        # bert_embeddings = []
        elmo_embeddings = []
        pickled = True
        if pickled:
            embeddings = pickle.load(open('elmo_' + tag + '.txt', 'rb'))
            return embeddings
        for sentence in tqdm(sentences):
            sent = nlp(sentence)
            sent = [token.text for token in sent]
            character_ids = batch_to_ids([sent])
            first, second = self.elmo(character_ids)['elmo_representations']
            first, second = first.squeeze(0), second.squeeze(0)
            embeddings = torch.cat([first, second], 1)
            embeddings = embeddings.detach()
            embeddings.requires_grad = False
            elmo_embeddings.append(embeddings)
        pickle.dump(elmo_embeddings, open('elmo_' + tag + '.txt', 'wb'))
        return elmo_embeddings