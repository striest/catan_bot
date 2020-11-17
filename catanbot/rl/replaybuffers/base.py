import abc

class ReplayBuffer(object, metaclass=abc.ABCMeta):
	"""
	General interface for replay buffers
	"""

	@abc.abstractmethod
	def insert(self, samples):
		pass

	@abc.abstractmethod
	def sample(self, nsamples):
		pass
