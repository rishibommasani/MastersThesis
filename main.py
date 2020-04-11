import spacy
from tqdm import tqdm
import numpy as np
from itertools import permutations
nlp = spacy.load('en_core_web_lg')
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from copy import deepcopy
import pickle
from collections import deque
import operator
import random
from tree import make_tree, print_tree
import copy
from spacy import displacy


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

# The nodes in the graph correspond to the indices of the sequence
# Hence, in data structures, something documented as a 'node' can also be taken as a sequence index


# Input: Expects a sequence of spacy tokens [s]
# Output: Returns tokens2index, index2token dictionaries 
def make_dictionaries(s):
	token2index, index2token = {}, {}
	for i, token in enumerate(s):
		token2index[token] = i 
		index2token[i] = token 
	return token2index, index2token


# Input: Expect a sequence of spacy tokens [s] and token2index dictionary [t2i]
# Output: Reformatted dependency information for each position in [s]
# Output: Formatted as {Index: {'Position': p, 'neighbors': {n_1, ..., n_k}}}
def make_graph(s, t2i):
	output = {}
	for token in s:
		index = t2i[token]
		position = index # Initialize value of node to be its position in original seq.
		neighbors = [t2i[child] for child in token.children]
		# output[token] = {'position' : position, 'neighbors' : list(token.children)}
		output[index] = {'position' : position, 'neighbors' : neighbors}
	return output


# Input: The graph [g] and the pooling method [method]
# Optional: A permutation supplying the position assignments instead of those in [g]
# Output: The score associated with the graph
def score(graph, method, positions=None):
	g = deepcopy(graph)
	pooling_fs = {'max': max, 'sum': sum}
	f = pooling_fs[method]
	out = 0
	if positions:
		n = len(positions)
		for i in range(n):
			g[i]['position'] = positions[i]
		for i in g:
			g[i]['neighbors'] = [g[n]['position'] for n in g[i]['neighbors']]
	# print(g)
	for node in g:
		position = g[node]['position']
		neighbors = g[node]['neighbors']
		if neighbors:
			out = f([out, f([abs(position - n) for n in neighbors])])
	return out


def get_f(tree, i2p):
	root = tree.data
	v = i2p[root] 
	children = tree.children
	return sum([abs(v - i2p[child]) for child in children]) + sum([get_f(child, i2p) for child in children])

def f_and_g(tree, i2p):
	root = tree.data 
	pi_r = i2p[root]
	f = get_f(tree, i2p)
	return f, f + pi_r - 1



def get_size(node):
	return sum([get_size(child) for child in node.children]) + 1


def rooted_MLA(tree):
	root = tree.data
	children = tree.children
	if not children:
		return {root : 0} 

	# Start of step 5
	children_sizes = {child : get_size(child) for child in children}
	ordered_children = [x for x, _ in sorted(children_sizes.items(), key=operator.itemgetter(1))]
	ordered_children.reverse()
	t0 = ordered_children[0]
	q = -1
	num_children = len(ordered_children)
	max_p = (num_children - 2) // 2
	for p in range(max_p, -1, -1):
		y = n - sum(ordered_children[0 : min(2 * p + 2, num_children)])
		if ordered_children[2 * p + 1] >= (t0 // 2 + 1) + (y // 2 + 1):
			q = p 
			break
	p = q
	# End of step 5

	# Start of step 6
	if p == -1:
		t0 = copy.deepcopy(t0)
		i2p_t0 = rooted_MLA(t0)
		f_t0, g_t0 = f_and_g(t0, i2p_t0)
		tree2 = copy.deepcopy(tree)
		tree2.remove_child(t0)
		i2p_remainder = rooted_MLA(tree2)
		f_rem, g_rem = f_and_g(tree2, i2p_remainder)
	# End of step 6

	# Start of step 7
	else:	
		pass
	# End of step 7


def get_edge_list(graph):
	edge_list = []
	for u in graph:
		for v in graph[u]['neighbors']:
			edge_list.append((u,v))
	return edge_list


def bandwidth_with_edge_list(edge_list, i2p):
	if not edge_list:
		return 0
	return max((abs(i2p[u] - i2p[v]) for u,v in edge_list))


def mla_with_edge_list(edge_list, i2p):
	return sum((abs(i2p[u] - i2p[v]) for u,v in edge_list))


def heuristic_MLA(graph):
	n = len(graph)
	edge_list = get_edge_list(graph)
	i2p = {i : i for i in range(n)}
	curr = mla_with_edge_list(edge_list, i2p)
	for step in (range(10000)):
		u = random.randint(0, n - 1)
		v = random.randint(0, n - 1)
		targ = copy.copy(i2p)
		targ[u], targ[v] = targ[v], targ[u]
		new = mla_with_edge_list(edge_list, targ)
		if new < curr:
			i2p = targ 
			curr = new 
	return i2p

# Input: Rooted tree [graph]
# Output: Maximum linear arrangement
def MLA(graph, n):
	i2p = heuristic_MLA(graph)
	p2i = {i2p[i] : i for i in i2p}
	permutation = [p2i[p] for p in range(len(p2i))]
	return i2p, permutation

	# roots = set(graph.keys())
	# for key in graph:
	# 	for neighbor in graph[key]['neighbors']:
	# 		roots.remove(neighbor)
	# assert len(roots) == 1
	# root = list(roots)[0]
	# tree = make_tree(root, graph)
	
	# children_nodes = graph[root]['neighbors']
	# children_sizes = {node : get_size(graph, node) for node in children_nodes}
	# ordered_children = [x for x, _ in sorted(children_sizes.items(), key=operator.itemgetter(1))]
	# ordered_children.reverse() # Want t0 >= t1 ...
	# t0 = ordered_children[0]
	# q = -1	
	# for p in range((n - 1) // 2, -1, -1):
	# 	y = n - sum(ordered_children[0 : min(2 * p + 2, len(ordered_children))])
	# 	if 2 * p + 1 < len(ordered_children) and ordered_children[2 * p + 1] >= (t0 // 2 + 1) + (y // 2 + 1):
	# 		q = p 
	# 		break
	# p = q
	# if p == -1:
	# else:



# Input: The depenendency parse [graph] generated by make_graph(s, t2i)
# Input: The pooling method being minimized [method] - either 'sum' or 'max'
# Note: Sum - Corresponds to maximum linear arrangement problem (NP- Complete)
# Note: Max - Corresponds to bandwith problem (NP-Complete)
# Outputs: node2position dictionary
def find_permutation(graph, method, verbose=True):
	n = len(graph)
	if verbose:
		identity_i2p = {i : i for i in range(n)}
		bandwidth_initial = score(graph, 'max', identity_i2p)
		MLA_initial = score(graph, 'sum', identity_i2p)
		x = list(range(n))
		random.shuffle(x)
		random_i2p = {i : v for i, v in enumerate(x)}
		bandwidth_random = score(graph, 'max', random_i2p)
		MLA_random = score(graph, 'sum', random_i2p)
	if method == 'max':
		x = np.zeros((n,n))
		for i in range(n):
			for j in graph[i]['neighbors']:
				x[i][j] = 1
				x[j][i] = 1
		csr = csr_matrix(x)
		permutation = reverse_cuthill_mckee(csr, symmetric_mode=True)
		I, J = np.ix_(permutation, permutation)
		i2p = {v : i for i,v in enumerate(permutation)}
		x2 = x[I,J]
		permutation = list(permutation)
	else:
		i2p, permutation = MLA(graph, n)
	if verbose:
		bandwidth_final = score(graph, 'max', i2p)
		MLA_final = score(graph, 'sum', i2p)
		return i2p, permutation, {'Initial bandwidth' : bandwidth_initial, 'Random bandwidth' : bandwidth_random, 'Final bandwidth' : bandwidth_final, 
		'Initial MLA' : MLA_initial, 'Random MLA' : MLA_random, 'Final MLA' : MLA_final}
	else:
		return i2p, permutation


# Input: Index2position dictionary [i2p] and index2token dictionary [i2t]
# Output: String corresponding to permuted sentence specified by [i2p]
def reform_sentence(i2p, i2t, p2i=None):
	if p2i is None:
		p2i = {i2p[i] : i for i in i2p}
	sentence = [i2t[p2i[p]] for p in range(len(p2i))]
	return '\\& '.join(sentence)


def load_data():
	print("Loading data")
	data = pickle.load(open('dump.txt', 'rb'))
	print("There are {} sentences, of which the 100th is: {}".format(len(data), data[99]))
	print("Loaded data")
	return data

def compute_outputs(edge_list, i2p, i2t, p2i):
	bandwidth = bandwidth_with_edge_list(edge_list, i2p)
	minLA = mla_with_edge_list(edge_list, i2p)
	reconstruction = reform_sentence(i2p, i2t, p2i)
	return bandwidth, minLA, reconstruction

def template_print(template):

	edge_list = template['general']['edge_list']
	for key in template:
		print(key + ':')
		if key == 'general':
			print(template[key])
		else:
			i2p = template[key]['i2p']
			print(template[key]['bandwidth'], template[key]['minLA'])
			print([(i2p[u] + 1, i2p[v] + 1) for u,v in edge_list])
			print(template[key]['reconstruction'])


def external_use_verbose(sentences, file_name, pickled=False):
	output = []
	if pickled:
		return pickle.load(open(file_name + '.pickle', 'rb'))
	for sent in tqdm(sentences):
		template = {}
		s = nlp(sent)
		t2i, i2t = make_dictionaries(s)
		i2t = {key : i2t[key].text for key in i2t}
		graph = make_graph(s, t2i)
		edge_list = get_edge_list(graph)
		l, n = len(sent), len(s)
		# Cannot store s, t2i, or graph using pickle; i.e. spacy components :(
		template['general'] = {'sentence' : sent, 'sentence_length' : l, 'spacy_length' : n, 'i2t' : i2t, 'edge_list' : edge_list}
		permutation, i2p, p2i = list(range(n)), {i : i for i in range(n)}, {i : i for i in range(n)}
		bandwidth, minLA, reconstruction = compute_outputs(edge_list, i2p, i2t, p2i)
		template['standard'] = {'permutation' : permutation, 'i2p' : i2p, 'p2i' : p2i, 'bandwidth' : bandwidth, 'minLA' : minLA, 'reconstruction' : reconstruction}
		permutation = list(range(n))
		random.shuffle(permutation)
		i2p, p2i = {i : v for i,v in enumerate(permutation)}, {v : i for i,v in enumerate(permutation)}
		bandwidth, minLA, reconstruction = compute_outputs(edge_list, i2p, i2t, p2i)
		template['random'] = {'permutation' : permutation, 'i2p' : i2p, 'p2i' : p2i, 'bandwidth' : bandwidth, 'minLA' : minLA, 'reconstruction' : reconstruction}
		i2p, permutation = find_permutation(graph, 'max', verbose=False)
		p2i = {i2p[i] : i for i in i2p} 
		bandwidth, minLA, reconstruction = compute_outputs(edge_list, i2p, i2t, p2i)
		template['bandwidth'] = {'permutation' : permutation, 'i2p' : i2p, 'p2i' : p2i, 'bandwidth' : bandwidth, 'minLA' : minLA, 'reconstruction' : reconstruction}
		i2p, permutation = find_permutation(graph, 'sum', verbose=False)
		p2i = {i2p[i] : i for i in i2p} 
		bandwidth, minLA, reconstruction = compute_outputs(edge_list, i2p, i2t, p2i)
		template['minLA'] = {'permutation' : permutation, 'i2p' : i2p, 'p2i' : p2i, 'bandwidth' : bandwidth, 'minLA' : minLA, 'reconstruction' : reconstruction}
		output.append(template)
	if pickled is not None:
		pickle.dump(output, open(file_name + '.pickle', 'wb'))
	return output


def external_use(sentences, order):
	if order == 'minLA':
		method = 'sum'
	else:
		method = 'max'
	permutations = []
	for sentence in tqdm(sentences):
		s = nlp(sentence)
		t2i, i2t = make_dictionaries(s)
		graph = make_graph(s, t2i)
		i2p, perm = find_permutation(graph, method, verbose=False) 
		permutations.append(perm)
	return permutations


def ordered_edge_list(edge_list):
	return list(sorted(edge_list, key = lambda t: abs(t[0] - t[1])))

def main():
	small = ["The reject, unlike the highly celebrated actor, won"]
	small_metadata = external_use_verbose(small, 'subj_short', pickled=None)
	for template in small_metadata:
		doc = nlp(template['general']['sentence'])
		displacy.render(doc, style='dep')
		template_print(template)
	exit()

if __name__ == '__main__':
	main()