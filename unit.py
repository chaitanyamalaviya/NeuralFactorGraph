from __future__ import division, print_function
import utils
import argparse
import torch
import itertools
import numpy as np
import math
import sys
import unittest,utils
import pdb
import factorial_crf_tagger

def create_sample_data(tagSize, labelSize, wordCount):
	words = ['sample' + str(i) for i in range(wordCount)]
	tags = []
	labelPtr = [-1] * tagSize
	k = -1

	for t in range(wordCount):
		tag_dict = {}
		for i in range(tagSize):
			# for labelIdx in range(labelSize[i]):
			if labelPtr[i] < labelSize[i]-1:
				labelPtr[i] += 1
			elif labelPtr[i] == labelSize[i]-1:
				labelPtr[i] -= 1
			tag_dict['tag'+str(i)] = 'label'+ str(i)  + "_" + str(labelPtr[i])
		tags.append(utils.freeze_dict(tag_dict))

	training_data = (words, tags)
	train_tgt_labels = set(tags)

	return training_data, train_tgt_labels


class TestBP:

	def __init__(self):
		self.model = None

	def setUp(self, tagger_model, gold_tags, sentLen, lstm_feats):
		print("Setting up..")
		self.model = tagger_model
		self.gold_tags = gold_tags
		self.sentLen = sentLen
		_, graph, maxVal = tagger_model.belief_propogation_log(gold_tags, sentLen, lstm_feats)
		all_sequences, sent_likelihood = self.bruteForce(graph, lstm_feats)
		self.assertEqualMarginals(graph, all_sequences, sent_likelihood)

	def assertEqualMarginals(self, graph, all_sequences, sent_likelihood):
		"""
		Check factor/variable marginals are approximately equal 
		to marginals obtained from brute force inference
		"""

		# Check variable marginals
		threshold = 0.01
		eq = True

		denom = -float('inf')
		maxDiff = -float('inf')

		for s, sequence in enumerate(all_sequences):
			denom = utils.logSumExp(sent_likelihood[s], denom)

		# Iterate over all timesteps
		for t in range(graph.T):
			for tag in self.model.uniqueTags:
				tagBeliefs = graph.getVarByTimestepnTag(t, tag.idx).belief.cpu().data.numpy()
				for labelIdx in range(tag.size()):
					num = -float('inf')
					for s, sequence in enumerate(all_sequences):
						if sequence[t][tag.idx]==labelIdx:
							num = utils.logSumExp(sent_likelihood[s], num)

					# Check difference
					# maxDiff = max(maxDiff, np.max(np.abs(tagBeliefs[labelIdx]- np.exp(num-denom))))
					tagLogProb = np.exp(num-denom)
					maxDiff = max(maxDiff, np.max(np.abs(np.exp(tagBeliefs[labelIdx]) - tagLogProb)))
					if maxDiff > threshold:
						eq = False

		if not eq:
			print("Marginals not equal. Max difference of %f" %maxDiff)
		else:
			print("Passed unit test!")

		sys.exit(0)

	def bruteForce(self, graph, lstm_feats):
		tagRanges = [range(tag.size()) for tag in self.model.uniqueTags]
		tag_combinations = list(itertools.product(*tagRanges))
		all_timesteps = [tag_combinations] * self.sentLen
		all_sequences = list(itertools.product(*all_timesteps))
		# sent_likelihood = [-float('inf')] * len(all_sequences)
		sent_likelihood = [0] * len(all_sequences)

		# calculate tag offsets for lstm features
		tag_count = 0
		tag_offsets = {}
		for tag in self.model.uniqueTags:
			tag_offsets[tag.idx] = tag_count
			tag_count += tag.size()

		# Iterate over all possible sequences
		for s, sequence in enumerate(all_sequences):
			for t, tags in enumerate(sequence):
				for i, tag1 in enumerate(tags):

					# LSTM Potential
					cur_lstm_feats = lstm_feats[t]
					cur_tag_lstm_weights = self.model.lstm_weights[i].cpu().data.numpy()
					cur_tag_lstm_feats = cur_lstm_feats[tag_offsets[i]: \
											tag_offsets[i]+self.model.uniqueTags.getTagbyIdx(i).size()].cpu().data.numpy()

					lstm_vec = utils.logNormalize(cur_tag_lstm_weights + cur_tag_lstm_feats)
					sent_likelihood[s] += lstm_vec[tag1]

					# Pairwise Potential
					for j, tag2 in enumerate(tags):

						if i<j:

							if (i, j) in self.model.pairs:
								pairwise_idx = self.model.pairs.index((i, j))
								cur_pairwise_weights = self.model.pairwise_weights[pairwise_idx].cpu().data.numpy()
								cur_weight_val = cur_pairwise_weights[tag1][tag2]

							sent_likelihood[s] += cur_weight_val

					# Transition potential
					if t+1!=len(sequence):
						next_label = sequence[t+1][i]
						trans_weights = self.model.transition_weights[i].cpu().data.numpy()
						transition_pot = trans_weights[tag1][next_label]
						sent_likelihood[s] += transition_pot

			print("Seq Likelihood: %f" %sent_likelihood[s])

		return all_sequences, sent_likelihood


if __name__ == '__main__':
    unittest.main(gpu=True)
