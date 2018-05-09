from __future__ import print_function

class Tags:

	def __init__(self):
		self.tags = []
		self.tag2idx = {}
		self.idx2tag = {}

	def __iter__(self):
		for tag in self.tags:
			yield(tag)

	def addTag(self, name):
		idx = len(self.tags)
		self.tags.append(Tag(name, idx))
		self.tag2idx[name] = idx
		self.idx2tag[idx] = name

	def size(self):
		return len(self.tags)

	def getTagbyIdx(self, idx):
		return self.tags[idx]

	def getTagbyName(self, name):
		idx = self.tag2idx[name]
		return self.tags[idx]

	def tagExists(self, tag):
		return any(t.name == tag for t in self.tags)

	def printTags(self):
		for idx, tagName in self.idx2tag.items():
			print("%d : %s" %(idx, tagName))

	def tagSetSize(self):
		return sum([tag.size() for tag in self.tags])

class Tag:

	def __init__(self, name, idx):

		self.name = name
		self.idx = idx
		self.labels = []
		self.label2idx = {}
		self.idx2label = {}

	def __iter__(self):
		for label in self.labels:
			yield(label)

	def __hash__(self):
		return hash(self.name)

	def addLabel(self, name):
		idx = len(self.labels)
		self.labels.append(Label(self, name, idx))
		self.label2idx[name] = idx
		self.idx2label[idx] = name

	def getLabels(self):
		return self.labels

	def getTagIdx(self):
		return self.idx

	def getLabelbyIdx(self, idx):
		return self.labels[idx]

	def getLabelbyName(self, name):
		idx = self.label2idx[name]
		return self.labels[idx]

	def labelExists(self, label):
		return any(l.name == label for l in self.labels)

	def size(self):
		return len(self.labels)

	def printLabels(self):
		print("Labels for tag %s" %self.name)
		for idx, labelName in self.idx2label.items():
			print("%d : %s" % (idx, labelName))

class Label:

	def __init__(self, parentTag, name, idx):
		self.parentTag = parentTag
		self.name = name
		self.idx = idx

	def getParentTag(self):
		return self.parentTag

	def getName(self):
		return self.name

	def getLabelIdx(self):
		return self.idx
