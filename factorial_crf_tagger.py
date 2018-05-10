from __future__ import division, print_function
from message import Factor, Var, FactorGraph, Messages
import utils
import bp

import numpy as np
import string
import math
import sys
import time
import pdb
import itertools
import copy

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DynamicCRF(nn.Module):
	def __init__(self, args, word_freq, langs, char_vocab_size, word_vocab_size, uniqueTags):

		super(DynamicCRF, self).__init__()

		self.model_type = args.model_type
		self.embedding_dim = args.emb_dim
		self.mlp_size = args.mlp_dim
		self.hidden_dim = args.hidden_dim
		self.char_vocab_size = char_vocab_size
		self.word_vocab_size = word_vocab_size
		self.uniqueTags = uniqueTags
		self.no_transitions = args.no_transitions
		self.no_pairwise = args.no_pairwise

		self.n_layers = args.n_layers
		self.sum_word_char = args.sum_word_char
		self.sent_attn = args.sent_attn
		self.word_freq = word_freq
		self.langs = langs
		self.gpu= args.gpu
		self.dropout = args.dropout

		# CharLSTM
		self.char_embeddings = nn.Embedding(self.char_vocab_size, self.embedding_dim)
		self.char_lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.n_layers, dropout=self.dropout, bidirectional=True)
		# The linear layer that maps from hidden state space to 
		# self.proj1 = nn.Linear(2 * self.hidden_dim, self.mlp_size)
		# self.proj2 = nn.Linear(self.mlp_size, self.char_vocab_size)
		self.char_hidden = self.init_hidden()

		if self.sum_word_char:
			self.word_embeddings = nn.Embedding(self.word_vocab_size, 2 * self.hidden_dim)

		if self.sent_attn:
			self.proj1 = nn.Linear(2 * self.hidden_dim, self.mlp_size, bias=False)
			self.proj2 = nn.Linear(self.mlp_size, 1, bias=False)

		self.lstm = nn.LSTM(self.hidden_dim*2, self.hidden_dim, num_layers=self.n_layers, dropout=self.dropout, bidirectional=True)
		
		# The linear layer that maps from hidden state space to tag space
		if self.model_type=="mono" or self.model_type=="baseline":
			self.hidden2tag = nn.Linear(2 * self.hidden_dim, self.uniqueTags.tagSetSize())

		elif self.model_type=="specific" or self.model_type=="joint":
			self.hidden2tag_1 = nn.Linear(2 * self.hidden_dim, self.uniqueTags.tagSetSize())
			self.hidden2tag_2 = nn.Linear(2 * self.hidden_dim, self.uniqueTags.tagSetSize())

		# Lang ID model
		if self.model_type=="joint":
			self.joint_tf1 = nn.Linear(2 * self.hidden_dim, self.uniqueTags.tagSetSize(), bias=False)
			self.joint_tf2 = nn.Linear(self.uniqueTags.tagSetSize(), len(self.langs), bias=False)



		
		self.pairs = list(itertools.combinations(range(self.uniqueTags.size()), 2))
		self.lstm_weights = nn.ParameterList([nn.Parameter(torch.randn(t.size())) for t in self.uniqueTags])
		
		# Tensor of pairwise parameters.  Entry i,j in a matrix is the cooccurrence score of
		# label i of tag1 *with* label j of tag2.
		if not self.no_pairwise:
			self.pairwise_weights = nn.ParameterList([nn.Parameter(torch.zeros(self.uniqueTags.getTagbyIdx(i).size(), self.uniqueTags.getTagbyIdx(j).size())) \
												for i, j in self.pairs])

		# Tensor of transition parameters.  Entry i,j in a matrix is the score of
		# transitioning *from* i *to* j.
		if not self.no_transitions:
			self.transition_weights = nn.ParameterList([nn.Parameter(torch.zeros(t.size(), t.size())) for t in self.uniqueTags])
		
		if self.model_type=="specific":
			self.lang_pairwise_weights = nn.ParameterList([nn.Parameter(torch.zeros(len(self.langs), self.uniqueTags.getTagbyIdx(i).size(), \
														self.uniqueTags.getTagbyIdx(j).size())) for i, j in self.pairs])
			self.lang_transition_weights = nn.ParameterList([nn.Parameter(torch.zeros(len(self.langs), t.size(), t.size())) \
																	for t in self.uniqueTags])

		
		# calculate tag offsets for lstm features
		tag_count = 0
		self.tag_offsets = {}
		for tag in self.uniqueTags:
			self.tag_offsets[tag.name] = tag_count
			tag_count += tag.size()

		self.hidden = self.init_hidden()


	def init_hidden(self):
		# Before we've done anything, we dont have any hidden state.
		# The axes semantics are (num_layers, minibatch_size, hidden_dim)
		return (utils.get_var(torch.zeros(self.n_layers * 2, 1, self.hidden_dim), gpu=self.gpu),
				utils.get_var(torch.zeros(self.n_layers * 2, 1, self.hidden_dim), gpu=self.gpu))
		

	def get_lstm_features(self, words, word_idxs=None, lang=None, test=False):
		"""
		words: list of tensors with idxs for each character
		"""

		embeds = [self.char_embeddings(word) for word in words]

		if not self.sent_attn:
			char_embs = [self.char_lstm(embeds[i].view(len(word), 1, -1), self.char_hidden)[0][-1] for i, word in enumerate(words)]
		else:
			word_attns = [self.char_lstm(embeds[i].view(len(word), 1, -1), self.char_hidden)[0].view(len(word), -1) for i, word in enumerate(words)]
			attn_probs = [F.tanh(self.proj1(w_attn)) for w_attn in word_attns]
			attn_probs = [F.softmax(self.proj2(w_attn).view(w_attn.size(0))) for w_attn in attn_probs]
			char_embs = [torch.sum(a.unsqueeze(1).repeat(1, w_attn.size(1))* w_attn, 0) for a, w_attn in zip(attn_probs, word_attns)]

		char_embs = torch.stack(char_embs).view(len(words), 1, -1)
		if self.sum_word_char:
			mask = torch.FloatTensor([1 if self.word_freq[int(w.cpu().data.numpy())]>=5 else 0 for w in word_idxs])
			if self.gpu:
				mask = mask.cuda()
			word_embs = self.word_embeddings(word_idxs) * Variable(mask.unsqueeze(1).repeat(1, 2*self.hidden_dim))
			char_embs = char_embs.view(len(words), -1) + word_embs
			char_embs = char_embs.unsqueeze(1)

		lstm_out, self.hidden = self.lstm(char_embs, self.hidden)

		if self.model_type=="specific" and lang:
			# tag_space = self.hidden2tag[lang](lstm_out.view(char_embs.size(0), -1))
			if self.langs.index(lang)==0:
				tag_space = self.hidden2tag_1(lstm_out.view(char_embs.size(0), -1))
			else:
				tag_space = self.hidden2tag_2(lstm_out.view(char_embs.size(0), -1))

		elif self.model_type=="joint" and lang:
			tf1 = F.tanh(self.joint_tf1(lstm_out.view(char_embs.size(0), -1)))
			tf2 = F.log_softmax(self.joint_tf2(tf1))
			if self.langs.index(lang)==0:
				tag_space = self.hidden2tag_1(lstm_out.view(char_embs.size(0), -1))
			else:
				tag_space = self.hidden2tag_2(lstm_out.view(char_embs.size(0), -1))

		elif self.model_type=="joint" and test:
			tf1 = F.tanh(self.joint_tf1(lstm_out.view(char_embs.size(0), -1)))
			tf2 = F.log_softmax(self.joint_tf2(tf1))
			pred_lang_scores, idxs = torch.max(tf2, 1)
			tag_space_1 = self.hidden2tag_1(lstm_out.view(char_embs.size(0), -1))
			tag_space_2 = self.hidden2tag_2(lstm_out.view(char_embs.size(0), -1))
			tag_space = [tag_space_1[i] if idx==0 else tag_space_2[i] for i, idx in enumerate(idxs.cpu().data.numpy().flatten())]
			tag_space = torch.stack(tag_space)

		else:
			tag_space = self.hidden2tag(lstm_out.view(char_embs.size(0), -1))

		# tag_scores = F.log_softmax(tag_space)
		if self.model_type=="joint" and not test:
			tag_scores = tag_scores + tf2[: , self.langs.index(lang)].repeat(1, self.uniqueTags.tagSetSize())
		elif self.model_type=="joint" and test:
			tag_scores = tag_scores + pred_lang_scores.repeat(1, self.uniqueTags.tagSetSize())

		return tag_space

	# @profile
	def belief_propogation_log(self, gold_tags, lang, sentLen, batch_lstm_feats, test=False):

		# fwd messages, then bwd messages for each tag => ! O(n^2)

		# start_time = time.time()

		threshold = 0.05
		maxIters = 50
		batch_size = len(batch_lstm_feats)
		if self.model_type=="specific":
			langIdx = self.langs.index(lang)

		# Initialize factor graph, add vars and factors
		print("Creating Factor Graph...")
		graph = FactorGraph(sentLen, batch_size, self.gpu)

		# Add variables to graph
		for tag in self.uniqueTags:
			for t in range(sentLen):
				label=None
				graph.addVariable(tag, label, t)

		if not self.no_pairwise:
			# Add pairwise factors to graph
			kind = "pair"
			for tag1 in self.uniqueTags:
				for tag2 in self.uniqueTags:
					if tag1!=tag2 and tag1.idx<tag2.idx:
						for t in range(sentLen):
							var1 = graph.getVarByTimestepnTag(t, tag1.idx)
							var2 = graph.getVarByTimestepnTag(t, tag2.idx)
							graph.addFactor(kind, var1, var2)

			# Retrieve pairwise weights
			pairwise_weights_np = []
			for i in range(len(self.pairs)):
				pairwise_weights_np.append(self.pairwise_weights[i].cpu().data.numpy())

			if self.model_type=="specific":
				for i in range(len(self.pairs)):
					pairwise_weights_np[i] = utils.logSumExp(pairwise_weights_np[i], self.lang_pairwise_weights[i][langIdx].cpu().data.numpy())



		if not self.no_transitions:
			# Add transition factors to graph
			kind = "trans"
			for tag in self.uniqueTags:
				for t in range(sentLen-1):
					var1 = graph.getVarByTimestepnTag(t, tag.idx)
					var2 = graph.getVarByTimestepnTag(t+1, tag.idx)
					graph.addFactor(kind, var1, var2)


			transition_weights_np = {}
			for tag in self.uniqueTags:
				transition_weights_np[tag.idx] = self.transition_weights[tag.idx].cpu().data.numpy() 

			if self.model_type=="specific":
				for tag in self.uniqueTags:
					transition_weights_np[tag.idx] = utils.logSumExp(transition_weights_np[tag.idx], self.lang_transition_weights[tag.idx][langIdx].cpu().data.numpy())


		kind = "lstm"
		for tag in self.uniqueTags:
			for t in range(sentLen):
				var = graph.getVarByTimestepnTag(t, tag.idx)
				graph.addFactor(kind, var, "LSTMVar")
		
		# Initialize messages
		messages = Messages(graph, batch_size)

		# Add LSTM unary factor message to each variable
		for tag in self.uniqueTags:
			for t in range(sentLen):
				lstm_vecs = []
				var = graph.getVarByTimestepnTag(t, tag.idx)
				lstm_factor = graph.getFactorByVars(var, "LSTMVar")
				cur_tag_lstm_weights = self.lstm_weights[tag.idx]

				for batchIdx in range(batch_size):
					lstm_feats = batch_lstm_feats[batchIdx]
					cur_lstm_feats = lstm_feats[t]
					cur_tag_lstm_feats = cur_lstm_feats[self.tag_offsets[tag.name]: self.tag_offsets[tag.name]+tag.size()]
					lstm_vec = torch.unsqueeze(cur_tag_lstm_weights + cur_tag_lstm_feats, 0)
					lstm_vec = utils.logNormalizeTensor(lstm_vec).squeeze(dim=0)
					lstm_vecs.append(lstm_vec.cpu().data.numpy())

				messages.updateMessage(lstm_factor, var, np.array(lstm_vecs))
				
		
		iter = 0
		while iter<maxIters:
			print("[BP iteration %d]" %iter, end=" ")
			maxVal = [-float("inf")]*batch_size
			for tag in self.uniqueTags:
				var_list = graph.getVarsByTag(tag.idx)
				
				# FORWARD

				for t in range(sentLen):

					var = var_list[t]

					# Get pairwise potentials
					
					factor_list = graph.getFactorByVars(var)
					factor_sum = np.zeros((batch_size, var.tag.size()))

					# Maintaining factor sum improves efficiency
					for factor_mult in factor_list:
						factor_sum += messages.getMessage(factor_mult, var).value

					for factor in factor_list:
						if factor.kind=="pair":
							var2 = factor.getOtherVar(var)

							# variable2factor

							message = np.zeros((batch_size, var.tag.size()))
 							message = factor_sum - messages.getMessage(factor, var).value
							message = utils.logNormalize(message)
							curVal = messages.getMessage(var, factor).value

							# From (Sutton, 2012)
							maxVal = np.maximum(maxVal, np.amax(np.abs(curVal - message), 1))
							messages.updateMessage(var, factor, message)

							# factor2variable
							if var2.tag.idx < var.tag.idx:
								pairwise_idx = self.pairs.index((var2.tag.idx, var.tag.idx))
								transpose = False
							else:
								pairwise_idx = self.pairs.index((var.tag.idx, var2.tag.idx))
								transpose = True


							cur_pairwise_weights = pairwise_weights_np[pairwise_idx]

							if transpose:
								
								pairwise_pot = utils.logDot(cur_pairwise_weights, messages.getMessage(var2, factor).value, redAxis=1)
							else:
								pairwise_pot = utils.logDot(messages.getMessage(var2, factor).value, cur_pairwise_weights, redAxis=0)

							pairwise_pot = utils.logNormalize(pairwise_pot)
							curVal = messages.getMessage(factor, var).value
							maxVal = np.maximum(maxVal, np.amax(np.abs(curVal - pairwise_pot), 1))
							messages.updateMessage(factor, var, pairwise_pot)
							factor_sum += pairwise_pot - curVal

					if not self.no_transitions:

						cur_tag_weights = transition_weights_np[tag.idx]

						# Get transition potential
						if t!=sentLen-1:

							var2 = graph.getVarByTimestepnTag(t+1, tag.idx)
							trans_factor = graph.getFactorByVars(var, var2)

							# Variable2Factor Message

							message = np.zeros((batch_size, var.tag.size()))
							message = factor_sum - messages.getMessage(trans_factor, var).value

							# for factor_mult in factor_list:
							# 	if factor_mult!=trans_factor:
							# 		message += messages.getMessage(factor_mult, var).value

							message = utils.logNormalize(message)
							curVal = messages.getMessage(var, trans_factor).value
							maxVal = np.maximum(maxVal, np.amax(np.abs(curVal - message), 1))
							messages.updateMessage(var, trans_factor, message)
							# Factor2Variable Message

							transition_pot = utils.logDot(messages.getMessage(var, trans_factor).value, cur_tag_weights, redAxis=0)

							transition_pot = utils.logNormalize(transition_pot)
							curVal = messages.getMessage(trans_factor, var2).value
							maxVal = np.maximum(maxVal, np.amax(np.abs(curVal - transition_pot), 1))
							messages.updateMessage(trans_factor, var2, transition_pot)

				# BACKWARD
				if not self.no_transitions:		

					for t in range(sentLen-1, 0, -1):

						var = var_list[t]						
						factor_list = graph.getFactorByVars(var)

						# Variable2Factor Message

						var2 = graph.getVarByTimestepnTag(t-1, tag.idx)
						trans_factor = graph.getFactorByVars(var, var2)

						message = np.zeros((batch_size, var.tag.size()))
						
						for i, factor_mult in enumerate(factor_list):
							if factor_mult!=trans_factor:
								message += messages.getMessage(factor_mult, var).value

						message = utils.logNormalize(message)
						curVal = messages.getMessage(var, trans_factor).value
						maxVal = np.maximum(maxVal, np.amax(np.abs(curVal - message), 1))
						messages.updateMessage(var, trans_factor, message)
						transition_pot = utils.logDot(cur_tag_weights, messages.getMessage(var, trans_factor).value, redAxis=1)

						transition_pot = utils.logNormalize(transition_pot)
						curVal = messages.getMessage(trans_factor, var2).value
						maxVal = np.maximum(maxVal, np.amax(np.abs(curVal - transition_pot), 1))
						messages.updateMessage(trans_factor, var2, transition_pot)

			iter += 1
			print("Max Res Value: %f" % max(maxVal))
			if max(maxVal) <= threshold:
				print("Converged in %d iterations" %(iter))
				break
			if iter==1000:
				print("Diverging :( Finished 1000 iterations.")
				return None

		# Calculate belief values and marginals

		# Variable beliefs
		for tag in self.uniqueTags:
			for t in range(sentLen):

				var = graph.getVarByTimestepnTag(t, tag.idx)

				factor_list = graph.getFactorByVars(var)
				for factor in factor_list:
					factorMsg = Variable(torch.FloatTensor(messages.getMessage(factor, var).value))
					if self.gpu:
						factorMsg = factorMsg.cuda()
					var.belief = var.belief + factorMsg
					
				# Normalize
				var.belief = utils.logNormalizeTensor(var.belief)

		# Factor beliefs
		for factor in graph.iterFactors():
			var1, var2 = graph.getVarsByFactor(factor)
			if factor.kind=="trans":
				factor.belief = self.transition_weights[var1.tag.idx]
				if self.model_type=="specific":
					factor.belief = utils.logSumExpTensors(factor.belief, self.lang_transition_weights[var1.tag.idx][langIdx])
			elif factor.kind=="pair":
				pairwise_idx = self.pairs.index((var1.tag.idx, var2.tag.idx))
				factor.belief = self.pairwise_weights[pairwise_idx]
				if self.model_type=="specific":
					factor.belief = utils.logSumExpTensors(factor.belief, self.lang_pairwise_weights[pairwise_idx][langIdx])
			else:
				continue

			factor.belief = factor.belief.view(1, factor.belief.size(0), -1).expand(batch_size, -1, -1)

			msg1 = torch.FloatTensor(messages.getMessage(var1, factor).value)
			msg2 = torch.FloatTensor(messages.getMessage(var2, factor).value)
			if self.gpu:
				msg1 = msg1.cuda()
				msg2 = msg2.cuda()

			factor.belief = Variable(msg1.view(batch_size, -1, 1).expand(-1, -1, var2.tag.size())) + factor.belief
			factor.belief = Variable(msg2.view(batch_size, 1, -1).expand(-1, var1.tag.size(), -1)) + factor.belief
			factor.belief = utils.logNormalizeTensor(factor.belief)

		# Calculate likelihood
		# likelihood = self.calc_likelihood(graph, gold_tags)
		# print("--- %s seconds ---" % (time.time() - start_time))
		return graph, max(maxVal)


	def calc_likelihood(self, graph, gold_tags):
		"""
		Calculates sentence likelihood according to Eq (3) from (Sutton, 2007)
		"""

		num = 0.0
		denom = 0.0
		t = 0

		for tags, next_tags in zip(gold_tags, gold_tags[1:]):
			tags_dict = utils.unfreeze_dict(tags)
			next_tags_dict = utils.unfreeze_dict(next_tags)
			cur_tags = tags_dict.keys()

			for i, tag in enumerate(cur_tags):
				# transition factors
				tag_idx = self.uniqueTags.tag2idx[tag]
				cur_label_idx = self.uniqueTags.getTagbyIdx(tag_idx).label2idx[tags_dict[tag]]
				var1 = graph.getVarByTimestepnTag(t, tag_idx)

				# avoid overcounting by exponentiating with degree-1
				# where degree is number of factors that depend on var1
				if t==0:
					degree = self.uniqueTags.size()
				else:
					degree = self.uniqueTags.size() + 1

				denom += var1.belief[cur_label_idx] * (degree - 1)

				if tag in next_tags_dict:
					var2 = graph.getVarByTimestepnTag(t+1, tag_idx)
					next_label_idx = self.uniqueTags.getTagbyIdx(tag_idx).label2idx[next_tags_dict[tag]]
					num += graph.getFactorByVars(var1, var2).belief[cur_label_idx][next_label_idx]
					
				# pairwise factors
				for coocc_tag in cur_tags[i:]:
					if tag!=coocc_tag:
						coocc_tag_idx = self.uniqueTags.tag2idx[coocc_tag]
						var2 = graph.getVarByTimestepnTag(t, coocc_tag_idx)
						coocc_label_idx = self.uniqueTags.getTagbyIdx(coocc_tag_idx).label2idx[tags_dict[coocc_tag]]
						factor = graph.getFactorByVars(var1, var2)
						if factor.var1.tag.name==tag:
							num += factor.belief[cur_label_idx][coocc_label_idx]
						else:
							num += factor.belief[coocc_label_idx][cur_label_idx]
			t += 1


		last_tags_dict = utils.unfreeze_dict(gold_tags[-1])
		last_tags = last_tags_dict.keys()
		for i, tag in enumerate(last_tags):
			tag_idx = self.uniqueTags.tag2idx[tag]
			cur_label_idx = self.uniqueTags.getTagbyIdx(tag_idx).label2idx[last_tags_dict[tag]]
			var1 = graph.getVarByTimestepnTag(t, tag_idx)
			degree = self.uniqueTags.size()
			denom += var1.belief[cur_label_idx] * (degree - 1)
			for coocc_tag in last_tags[i:]:
				if tag!=coocc_tag:
					coocc_tag_idx = self.uniqueTags.tag2idx[coocc_tag]
					var2 = graph.getVarByTimestepnTag(t, coocc_tag_idx)
					coocc_label_idx = self.uniqueTags.getTagbyIdx(coocc_tag_idx).label2idx[last_tags_dict[coocc_tag]]
					factor = graph.getFactorByVars(var1, var2)
					if factor.var1.tag.name==tag:
						num += factor.belief[cur_label_idx][coocc_label_idx]
					else: 
						num += factor.belief[coocc_label_idx][cur_label_idx]

		return (num-denom)

	# @profile
	def get_scores(self, graph, gold_tags, lstm_feats, batchIdx):


		# Add labels to variables in the graph
		for tag in self.uniqueTags:
			for t in range(graph.T):
				label="NULL"
				gold_dict = utils.unfreeze_dict(gold_tags[t])
				var = graph.getVarByTimestepnTag(t, tag.idx)
				if tag.name in gold_dict:
					label = gold_dict[tag.name]
				var.label = label

		targets = {}
		tag_scores = {}
		lstm_scores = {}

		# get variable scores
		
		for tag in self.uniqueTags:
			var_list = graph.getVarsByTag(tag.idx)
			targets[tag.name] = []
			tag_scores[tag.name] = []
			lstm_scores[tag.name] = []
			tagExists = False
			for t in range(graph.T):
				cur_lstm_feats = lstm_feats[t]
				cur_tag_lstm_feats = cur_lstm_feats[self.tag_offsets[tag.name]: self.tag_offsets[tag.name]+tag.size()]
				cur_tag_lstm_weights = self.lstm_weights[tag.idx]
				lstm_score = cur_tag_lstm_weights + cur_tag_lstm_feats
				# Normalize LSTM features
				lstm_score = torch.unsqueeze(lstm_score, 0)
				lstm_score = utils.logNormalizeTensor(lstm_score).squeeze(dim=0)
				var = var_list[t]

				if var.label!=None:
					labelIdx = tag.label2idx[var.label]
					targets[tag.name].append(labelIdx)
					tag_scores[tag.name].append(var.belief[batchIdx])
					lstm_scores[tag.name].append(lstm_score)
					tagExists = True
			if tagExists:
				tag_scores[tag.name] = torch.stack(tag_scores[tag.name])
				lstm_scores[tag.name] = torch.stack(lstm_scores[tag.name])
				# targets[tag.name] = torch.stack(targets[tag.name]).squeeze(1)
			else:
				del tag_scores[tag.name]
				del targets[tag.name]
				del lstm_scores[tag.name]

		# get factor scores

		trans_factor_scores = {}
		trans_factor_beliefs = {}

		pairwise_factor_scores = {}
		pairwise_factor_beliefs = {}

		for t, tags in enumerate(gold_tags, start=0):
			
			tags_dict = utils.unfreeze_dict(tags)
			cur_tags = tags_dict.keys()

			if t!=len(gold_tags)-1:
				next_tags_dict = utils.unfreeze_dict(gold_tags[t+1])

			for i, tag in enumerate(cur_tags):
				tag_idx = self.uniqueTags.tag2idx[tag]
				cur_label_idx = self.uniqueTags.getTagbyIdx(tag_idx).label2idx[tags_dict[tag]]
				var1 = graph.getVarByTimestepnTag(t, tag_idx)


				if not self.no_pairwise:

					# Collect pairwise scores

					for coocc_tag in cur_tags[i:]:
						if tag!=coocc_tag:
							coocc_tag_idx = self.uniqueTags.tag2idx[coocc_tag]
							var2 = graph.getVarByTimestepnTag(t, coocc_tag_idx)
							coocc_label_idx = self.uniqueTags.getTagbyIdx(coocc_tag_idx).label2idx[tags_dict[coocc_tag]]
							
							if var2.tag.idx < var1.tag.idx:
								pairwise_idx = self.pairs.index((var2.tag.idx, var1.tag.idx))
								gold_belief = coocc_label_idx * var1.tag.size() + cur_label_idx
							else:
								pairwise_idx = self.pairs.index((var1.tag.idx, var2.tag.idx))
								gold_belief = cur_label_idx * var2.tag.size() + coocc_label_idx

							target_belief = graph.getFactorByVars(var1, var2).belief[batchIdx]

							if pairwise_idx not in pairwise_factor_scores:
								pairwise_factor_scores[pairwise_idx] = []
								pairwise_factor_beliefs[pairwise_idx] = []
							
							pairwise_factor_scores[pairwise_idx].append(gold_belief)
							pairwise_factor_beliefs[pairwise_idx].append(target_belief)


				if not self.no_transitions:

					# Collect transition scores
					
					if t!=len(gold_tags)-1:

						# tagExists = False

						if tag in next_tags_dict:
							var2 = graph.getVarByTimestepnTag(t+1, tag_idx)
							next_label_idx = self.uniqueTags.getTagbyIdx(tag_idx).label2idx[next_tags_dict[tag]]
							gold_belief = cur_label_idx * var1.tag.size() + next_label_idx
							target_belief = graph.getFactorByVars(var1, var2).belief[batchIdx]
							if tag not in trans_factor_scores:
								trans_factor_scores[tag] = []
								trans_factor_beliefs[tag] = []
							trans_factor_scores[tag].append(gold_belief)
							trans_factor_beliefs[tag].append(target_belief)
							# tagExists = True
							
						# if not tagExists:
						# 	del trans_factor_scores[tag]
						# 	del trans_factor_beliefs[tag]


		all_factors = []
		
		if not self.no_transitions:
			all_factors.append((trans_factor_beliefs, trans_factor_scores))
		
		if not self.no_pairwise:
			all_factors.append((pairwise_factor_beliefs, pairwise_factor_scores))

		all_factors.append((lstm_scores, targets))

		return all_factors



	def getBestSequence(self, graph, batchIdx):

		bestSequence = []

		for t in range(graph.T):
			sequence = {}
			for tag in self.uniqueTags:
				var = graph.getVarByTimestepnTag(t, tag.idx)
				score = var.belief[batchIdx]
				val, ind = torch.max(score, 0)
				sequence[tag.name] = tag.idx2label[ind.cpu().data[0]]
			bestSequence.append(sequence)

		return bestSequence

	# @profile
	def compute_loss(self, all_factors_batch, loss_function):

		loss = utils.get_var(torch.FloatTensor([0.0]), self.gpu)
		# factor_kinds = [transition_factors, pairwise_factors, lstm_factors]

		# for k in targets.keys():
		# 	loss += loss_function(tag_scores[k], targets[k])

		for all_factors in all_factors_batch:
			for factor in all_factors:
				beliefs, scores = factor
				for k in beliefs.keys():
					if k in scores and k in beliefs:
						belief = torch.stack(beliefs[k]).view(len(scores[k]),-1)
						score = Variable(torch.LongTensor(scores[k]), requires_grad=False)
						if self.gpu:
							score = score.cuda()
						loss += loss_function(belief, score)

		return loss

	def forward(self, sents, gold_tags, word_idxs=None, langs=None, test=False):

		lstm_feat_sents = []
		for i, sent in enumerate(sents):
			lang = langs[i] if langs!=None else None
			widx = word_idxs[i] if word_idxs!=None else None
			lstm_feat_sents.append(self.get_lstm_features(sent, widx, lang, test))

		graph, maxVal = self.belief_propogation_log(gold_tags, lang, len(lstm_feat_sents[0]), lstm_feat_sents, test=test)
		#graph, maxVal = bp.belief_propogation_log(self, lang, len(lstm_feat_sents[0]), lstm_feat_sents, test=test)
		return lstm_feat_sents, graph, maxVal

	def gradient_check(self, all_factors):

		# Check gradient of transition parameters

		trans_factors = all_factors[0]
		trans_grads = {}

		for i, p in enumerate(self.transition_weights):
			trans_grads[self.uniqueTags.idx2tag[i]] = p.grad

		trans_weights = [p.data for p in self.transition_weights]

		beliefs, scores = trans_factors

		num_grads = {}
		for tag in self.uniqueTags:
			num_grads[tag.name] = torch.zeros(tag.size(), tag.size())
			if self.gpu:
				num_grads[tag.name] = num_grads[tag.name].cuda()

		for k in beliefs.keys():
			if k in scores and k in beliefs:
				tag = self.uniqueTags.getTagbyName(k)
				for i in range(len(scores[k])):
					feat_score = torch.zeros(tag.size(), tag.size())
					if self.gpu:
						feat_score = feat_score.cuda()
					idx1 = int(scores[k][i]/tag.size())
					idx2 = int(scores[k][i]%tag.size())
					feat_score[idx1, idx2] = 1
					num_grads[k] += feat_score
					num_grads[k] -= torch.exp(beliefs[k][i]).data
				num_grads[k] /= len(scores[k])
			else:
				num_grads[k] = None


		pdb.set_trace()
		# Assert numerical gradients and computed gradients are equal

		return None	


