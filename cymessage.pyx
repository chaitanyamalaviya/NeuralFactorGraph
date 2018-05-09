from __future__ import division, print_function
import torch
from torch.autograd import Variable
import numpy as np
cimport numpy as np
import sys
np.set_printoptions(threshold=np.nan)


cdef class Messages:

	cdef public dict messages 
	cdef public int count
	cdef public int batch_size

	def __init__(self, graph, batch_size=1, test=False):

		self.messages = {}
		count = 0
		self.batch_size = batch_size

		for var in graph.cvars:
			self.messages[var] = {}

		for factor in graph.factors:
			self.messages[factor] = {}

		# Initialize messages to the uniform distribution
		for var in graph.cvars:
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

		for var in graph.cvars:
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
