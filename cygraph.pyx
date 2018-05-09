from __future__ import division, print_function
import torch
from torch.autograd import Variable
from libcpp cimport bool
import numpy as np
cimport numpy as np
import sys
np.set_printoptions(threshold=np.nan)

cdef class FactorGraph:

	"""
	Class defining the Factor Graph
	"""
	
	cdef public list cvars
	cdef public list factors
	cdef public dict var2idx
	cdef public dict var2factor
	cdef public int T
	cdef public int batch_size
	cdef public int gpu

	def __init__(self, int T, int batch_size, gpu=True):
		self.cvars = []
		self.factors = []
		self.var2idx = {}
		self.var2factor = {}
		self.T = T
		self.batch_size = batch_size
		self.gpu = 1 if gpu else 0

	def __iter__(self):
		for var in self.cvars:
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
		self.cvars.append(newVar)

	def getVarsByTimestep(self, t, tagSize):

		cdef list var_list = []
		for idx in range(tagSize):
			var_list.append(self.getVarByTimestepnTag(t, idx))

		return var_list


	def getVarsByFactor(self, factor):
		return factor.var1, factor.var2

	def getVarsByTag(self, tagIdx):

		cdef list var_list = []
		for t in range(self.T):
			var_list.append(self.getVarByTimestepnTag(t, tagIdx))
		return var_list

	def getVarByTimestepnTag(self, t, tagIdx):
		cdef int idx = tagIdx * self.T + t
		return self.cvars[idx]


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