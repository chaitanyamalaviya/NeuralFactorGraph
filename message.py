from __future__ import division, print_function
import torch
from torch.autograd import Variable
import numpy as np
import pdb
import sys
np.set_printoptions(threshold=np.nan)


class FactorGraph:

	"""
	Class defining the Factor Graph
	"""

	def __init__(self, T, batch_size, gpu=True):
		self.vars = []
		self.factors = []
		self.var2idx = {}
		self.var2factor = {}
		self.T = T
		self.batch_size = batch_size
		self.gpu = gpu

	def __iter__():
		for var in self.vars:
			yield(var)

	def iterFactors(self):
		for factor in self.factors:
			yield(factor)

	def addFactor(self, kind, var1, var2):

		# var1 and var2 are Var objects

		self.factors.append(Factor(kind, var1, var2, self.batch_size, self.gpu))
		if var1 not in self.var2factor:
			self.var2factor[var1] = []
		if var2 not in self.var2factor:
			self.var2factor[var2] = []

		self.var2factor[var1].append(len(self.factors)-1)
		self.var2factor[var2].append(len(self.factors)-1)
		# self.factors.append(Factor(kind, var1, var2))

	def addVariable(self, tag, label, timestep):
		newVar = Var(tag, label, timestep, self.batch_size, self.gpu)
		self.vars.append(newVar)

	def getVarsByTimestep(self, t, tagSize):

		var_list = []
		for idx in range(tagSize):
			var_list.append(self.getVarByTimestepnTag(t, idx))

		return var_list


	def getVarsByFactor(self, factor):
		return factor.var1, factor.var2

	def getVarsByTag(self, tagIdx):

		var_list = []
		for t in range(self.T):
			var_list.append(self.getVarByTimestepnTag(t, tagIdx))
		return var_list

	def getVarByTimestepnTag(self, t, tagIdx):
		idx = tagIdx * self.T + t
		return self.vars[idx]


	def getFactorByVars(self, var1, var2=None):

		if var1==var2:
			sys.exit("Error => var1 and var2 are equal!")

		if var2!=None:
			idx = list(set(self.var2factor[var1]).intersection(self.var2factor[var2]))[0]
			return self.factors[idx]

		else:

			neighbor_factor_idxs = self.var2factor[var1]
			neighbor_factors = []
			for idx in neighbor_factor_idxs:
				neighbor_factors.append(self.factors[idx])
			return neighbor_factors


class Factor:
	"""
	Class for Factors
	"""

	def __init__(self, kind, var1, var2, batch_size=1, gpu=True):
		# Kind -> pair or trans or lstm
		self.kind = kind
		self.var1 = var1
		self.var2 = var2

		if self.kind=="lstm":
			self.belief = None
		else:
			self.belief = Variable(torch.zeros((batch_size, var1.tag.size(), var2.tag.size())), requires_grad=True)
			if gpu:
				self.belief = self.belief.cuda()

	def __hash__(self):
		return hash((self.var1, self.var2, self.kind))

	def updateBelief(self, value):
		self.belief = value

	def getOtherVar(self, var):
		if self.var1==var:
			return self.var2
		else:
			return self.var1


class Var:
	"""
	Class for Variables
	"""

	def __init__(self, tag, label, timestep, batch_size=1, gpu=True):
		self.tag = tag
		self.label = label
		self.timestep = timestep
		self.belief = Variable(torch.zeros(batch_size, tag.size()), requires_grad=True)
		if gpu:
			self.belief = self.belief.cuda()

	def __hash__(self):
		return hash((self.tag.name, self.timestep))

	def updateValue(self, value):
		self.belief = value


class Messages:
	def __init__(self, graph, batch_size=1, test=False):

		self.messages = {}
		count = 0
		self.batch_size = batch_size

		for var in graph.vars:
			self.messages[var] = {}

		for factor in graph.factors:
			self.messages[factor] = {}

		# Initialize messages to the uniform distribution
		for var in graph.vars:
			for factor in graph.getFactorByVars(var):
				self.messages[factor][var] = {}
				self.addMessage(factor, var, np.full((batch_size, var.tag.size()), np.log(1./var.tag.size())))
				# self.addMessage(factor, var, np.repeat(1./var.tag.size(), var.tag.size()))
				count += 1

		for factor in graph.factors:
			# Don't add messages from variable to LSTM unary factor
			if factor.kind!="lstm":
				for var in graph.getVarsByFactor(factor):
					self.messages[var][factor] = {}
					self.addMessage(var, factor, np.full((batch_size, var.tag.size()), np.log(1./var.tag.size())))
					# self.addMessage(var, factor, np.repeat(1./var.tag.size(), var.tag.size()))
					count += 1


	def __iter__(self):

		for var in self.messages.keys():
			for msg in self.messages[var].values():
				yield(msg)

	def __copy__(self, graph):

		msgsCopy = Messages(graph)

		for var in graph.vars:
			for factor in graph.getFactorByVars(var):
				msgsCopy.updateMessage(factor, var, self.getMessage(factor, var).value)

		for factor in graph.factors:
			for var in graph.getVarsByFactor(factor):
				msgsCopy.updateMessage(var, factor, self.getMessage(var, factor).values)

		return msgsCopy


	def addMessage(self, frm, to, value):
		self.messages[frm][to] = Message(frm, to, value)

	def updateMessage(self, frm, to, value):
		existing_msg = self.getMessage(frm, to)
		existing_msg.updateValue(value)

	def getMessage(self, frm, to):
		return self.messages[frm][to]

class Message:
	def __init__(self, frm, to, value):
		"""
		Message from variable to factor
		or factor to variable

		"""
		self.frm = frm
		self.to = to
		self.value = value

	def updateValue(self, value):
		self.value = value
