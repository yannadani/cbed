import numpy as np

class D():
	def __init__(self, func, *args, **keywords):
		def newfunc(*fargs, **fkeywords):
			newkeywords = {**keywords, **fkeywords}
			return func(*args, *fargs, **newkeywords)
		newfunc.func = func
		newfunc.args = args
		newfunc.keywords = keywords
		self.partial = newfunc

	def sample(self, size):
		return self.partial(size=size)

class Constant():
    def __init__(self, value):
        self.value = value

    def sample(self, size):
        return np.array([self.value]*size)