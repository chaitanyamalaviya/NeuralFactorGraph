from __future__ import division, print_function
import numpy as np
cimport numpy as np

import cyutils
import utils
from cymessage import Messages
from cygraph import FactorGraph, Factor, Var
from libcpp cimport bool
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def belief_propogation_log(model, char *lang, int sentLen, list batch_lstm_feats, test=False):

	cdef float threshold = 0.05
	cdef int maxIters = 50
	cdef int batch_size = len(batch_lstm_feats)

	if model.model_type=="specific":
		langIdx = model.langs.index(lang)

	# Initialize factor graph, add vars and factors
	print("Creating Factor Graph...")
	graph = FactorGraph(sentLen, batch_size, model.gpu)

	# Add variables to graph
	for tag in model.uniqueTags:
		for t in range(sentLen):
			label=None
			graph.addVariable(tag, label, t)

	if not model.no_pairwise:
		# Add pairwise factors to graph
		kind = "pair"
		for tag1 in model.uniqueTags:
			for tag2 in model.uniqueTags:
				if tag1!=tag2 and tag1.idx<tag2.idx:
					for t in range(sentLen):
						var1 = graph.getVarByTimestepnTag(t, tag1.idx)
						var2 = graph.getVarByTimestepnTag(t, tag2.idx)
						graph.addFactor(kind, var1, var2)

		# Retrieve pairwise weights
		pairwise_weights_np = []
		for i in range(len(model.pairs)):
			pairwise_weights_np.append(model.pairwise_weights[i].cpu().data.numpy())

		if model.model_type=="specific":
			for i in range(len(model.pairs)):
				pairwise_weights_np[i] = cyutils.logSumExp(pairwise_weights_np[i], model.lang_pairwise_weights[i][langIdx].cpu().data.numpy())



	if not model.no_transitions:
		# Add transition factors to graph
		kind = "trans"
		for tag in model.uniqueTags:
			for t in range(sentLen-1):
				var1 = graph.getVarByTimestepnTag(t, tag.idx)
				var2 = graph.getVarByTimestepnTag(t+1, tag.idx)
				graph.addFactor(kind, var1, var2)


		transition_weights_np = {}
		for tag in model.uniqueTags:
			transition_weights_np[tag.idx] = model.transition_weights[tag.idx].cpu().data.numpy() 

		if model.model_type=="specific":
			for tag in model.uniqueTags:
				transition_weights_np[tag.idx] = cyutils.logSumExp(transition_weights_np[tag.idx], model.lang_transition_weights[tag.idx][langIdx].cpu().data.numpy())


	kind = "lstm"
	for tag in model.uniqueTags:
		for t in range(sentLen):
			var = graph.getVarByTimestepnTag(t, tag.idx)
			graph.addFactor(kind, var, "LSTMVar")
	
	# Initialize messages
	messages = Messages(graph, batch_size)

	# Add LSTM unary factor message to each variable
	for tag in model.uniqueTags:
		for t in range(sentLen):
			lstm_vecs = []
			var = graph.getVarByTimestepnTag(t, tag.idx)
			lstm_factor = graph.getFactorByVars(var, "LSTMVar")
			cur_tag_lstm_weights = model.lstm_weights[tag.idx]

			for batchIdx in range(batch_size):
				lstm_feats = batch_lstm_feats[batchIdx]
				cur_lstm_feats = lstm_feats[t]
				cur_tag_lstm_feats = cur_lstm_feats[model.tag_offsets[tag.name]: model.tag_offsets[tag.name]+tag.size()]
				lstm_vec = torch.unsqueeze(cur_tag_lstm_weights + cur_tag_lstm_feats, 0)
				lstm_vec = utils.logNormalizeTensor(lstm_vec).squeeze(dim=0)
				lstm_vecs.append(lstm_vec.cpu().data.numpy())

			messages.updateMessage(lstm_factor, var, np.array(lstm_vecs))
			
	
	cdef int iter = 0
	while iter<maxIters:
		print("[BP iteration %d]" %iter, end=" ")
		maxVal = [-float("inf")]*batch_size
		for tag in model.uniqueTags:
			var_list = graph.getVarsByTag(tag.idx)
			
			# FORWARD

			for t in range(sentLen):

				var = var_list[t]

				# Get pairwise potentials
				
				factor_list = graph.getFactorByVars(var)
				factor_sum = np.zeros([batch_size, var.tag.size()], dtype=DTYPE)

				# Maintaining factor sum improves efficiency
				for factor_mult in factor_list:
					factor_sum += messages.getMessage(factor_mult, var).value

				for factor in factor_list:
					if factor.kind=="pair":
						var2 = factor.getOtherVar(var)

						# variable2factor

						message = np.zeros([batch_size, var.tag.size()], dtype=DTYPE)
						message = factor_sum - messages.getMessage(factor, var).value
						message = cyutils.logNormalize(message)
						curVal = messages.getMessage(var, factor).value

						# From (Sutton, 2012)
						maxVal = np.maximum(maxVal, np.amax(np.abs(curVal - message), 1))
						messages.updateMessage(var, factor, message)

						# factor2variable
						if var2.tag.idx < var.tag.idx:
							pairwise_idx = model.pairs.index((var2.tag.idx, var.tag.idx))
							transpose = False
						else:
							pairwise_idx = model.pairs.index((var.tag.idx, var2.tag.idx))
							transpose = True


						cur_pairwise_weights = pairwise_weights_np[pairwise_idx]

						if transpose:
							if test:
								pairwise_pot = cyutils.logMax(cur_pairwise_weights, messages.getMessage(var2, factor).value, redAxis=1)
							else:
								pairwise_pot = cyutils.logDot(cur_pairwise_weights, messages.getMessage(var2, factor).value, redAxis=1)
						else:
							if test:
								pairwise_pot = cyutils.logMax(messages.getMessage(var2, factor).value, cur_pairwise_weights, redAxis=0)
							else:
								pairwise_pot = cyutils.logDot(messages.getMessage(var2, factor).value, cur_pairwise_weights, redAxis=0)

						pairwise_pot = cyutils.logNormalize(pairwise_pot)
						curVal = messages.getMessage(factor, var).value
						maxVal = np.maximum(maxVal, np.amax(np.abs(curVal - pairwise_pot), 1))
						messages.updateMessage(factor, var, pairwise_pot)
						factor_sum += pairwise_pot - curVal

				if not model.no_transitions:

					cur_tag_weights = transition_weights_np[tag.idx]

					# Get transition potential
					if t!=sentLen-1:

						var2 = graph.getVarByTimestepnTag(t+1, tag.idx)
						trans_factor = graph.getFactorByVars(var, var2)

						# Variable2Factor Message

						message = np.zeros([batch_size, var.tag.size()], dtype=DTYPE)
						message = factor_sum - messages.getMessage(trans_factor, var).value

						# for factor_mult in factor_list:
						# 	if factor_mult!=trans_factor:
						# 		message += messages.getMessage(factor_mult, var).value

						message = cyutils.logNormalize(message)
						curVal = messages.getMessage(var, trans_factor).value
						maxVal = np.maximum(maxVal, np.amax(np.abs(curVal - message), 1))
						messages.updateMessage(var, trans_factor, message)
						# Factor2Variable Message

						if test:
							transition_pot = cyutils.logMax(messages.getMessage(var, trans_factor).value, cur_tag_weights, redAxis=0)
						else:
							transition_pot = cyutils.logDot(messages.getMessage(var, trans_factor).value, cur_tag_weights, redAxis=0)

						transition_pot = cyutils.logNormalize(transition_pot)
						curVal = messages.getMessage(trans_factor, var2).value
						maxVal = np.maximum(maxVal, np.amax(np.abs(curVal - transition_pot), 1))
						messages.updateMessage(trans_factor, var2, transition_pot)

			# BACKWARD
			if not model.no_transitions:		

				for t in range(sentLen-1, 0, -1):

					var = var_list[t]						
					factor_list = graph.getFactorByVars(var)

					# Variable2Factor Message

					var2 = graph.getVarByTimestepnTag(t-1, tag.idx)
					trans_factor = graph.getFactorByVars(var, var2)

					message = np.zeros([batch_size, var.tag.size()], dtype=DTYPE)
					
					for i, factor_mult in enumerate(factor_list):
						if factor_mult!=trans_factor:
							message += messages.getMessage(factor_mult, var).value

					message = cyutils.logNormalize(message)
					curVal = messages.getMessage(var, trans_factor).value
					maxVal = np.maximum(maxVal, np.amax(np.abs(curVal - message), 1))
					messages.updateMessage(var, trans_factor, message)
					if test:
						transition_pot = cyutils.logMax(cur_tag_weights, messages.getMessage(var, trans_factor).value, redAxis=1)
					else:
						transition_pot = cyutils.logDot(cur_tag_weights, messages.getMessage(var, trans_factor).value, redAxis=1)

					transition_pot = cyutils.logNormalize(transition_pot)
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
	for tag in model.uniqueTags:
		for t in range(sentLen):

			var = graph.getVarByTimestepnTag(t, tag.idx)

			factor_list = graph.getFactorByVars(var)
			for factor in factor_list:
				factorMsg = Variable(torch.FloatTensor(messages.getMessage(factor, var).value))
				if model.gpu:
					factorMsg = factorMsg.cuda()
				var.belief = var.belief + factorMsg
				
			# Normalize
			var.belief = utils.logNormalizeTensor(var.belief)

	# Factor beliefs
	for factor in graph.iterFactors():
		var1, var2 = graph.getVarsByFactor(factor)
		if factor.kind=="trans":
			factor.belief = model.transition_weights[var1.tag.idx]
			if model.model_type=="specific":
				factor.belief = utils.logSumExpTensors(factor.belief, model.lang_transition_weights[var1.tag.idx][langIdx])
		elif factor.kind=="pair":
			pairwise_idx = model.pairs.index((var1.tag.idx, var2.tag.idx))
			factor.belief = model.pairwise_weights[pairwise_idx]
			if model.model_type=="specific":
				factor.belief = utils.logSumExpTensors(factor.belief, model.lang_pairwise_weights[pairwise_idx][langIdx])
		else:
			continue

		factor.belief = factor.belief.view(1, factor.belief.size(0), -1).expand(batch_size, -1, -1)

		msg1 = torch.FloatTensor(messages.getMessage(var1, factor).value)
		msg2 = torch.FloatTensor(messages.getMessage(var2, factor).value)
		if model.gpu:
			msg1 = msg1.cuda()
			msg2 = msg2.cuda()

		factor.belief = Variable(msg1.view(batch_size, -1, 1).expand(-1, -1, var2.tag.size())) + factor.belief
		factor.belief = Variable(msg2.view(batch_size, 1, -1).expand(-1, var1.tag.size(), -1)) + factor.belief
		factor.belief = utils.logNormalizeTensor(factor.belief)

	# Calculate likelihood
	# likelihood = model.calc_likelihood(graph, gold_tags)
	# print("--- %s seconds ---" % (time.time() - start_time))
	return graph, max(maxVal)
