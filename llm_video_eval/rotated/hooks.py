import torch

class BaseHook(object):
	def __init__(self):
		self.sign = {}
		self.n_bits = {}
	
	def __call__(self, name):
		self.sign[name] = True
		self.n_bits[name] = 8
  
class MinMaxHook(BaseHook):
	def __init__(self):
		super().__init__()
		self.min_vals = {}
		self.max_vals = {}

	def __call__(self, module, input, output, name):
		super().__call__(name)
		if isinstance(output, tuple) or name.endswith(".mlp"):
			pass
		else:
		# Extract the minimum and maximum values from the output tensor
			if name not in self.min_vals:
				self.min_vals[name] = output.min().item()
				self.max_vals[name] = output.max().item()
			else:
				self.min_vals[name] = min(self.min_vals[name], output.min().item())
				self.max_vals[name] = max(self.max_vals[name], output.max().item())


class AbsMinMaxHook(BaseHook):
	def __init__(self):
		super().__init__()
		self.min_vals = {}
		self.max_vals = {}
		self.abs_min_vals = {}
		self.abs_max_vals = {}

	def __call__(self, module, input, output, name):
		super().__call__(name)
		tensor = output#input[0]
		tensor[tensor == -float("Inf")] = 0
		if name not in self.abs_min_vals:
			self.abs_min_vals[name] = torch.abs(tensor).min().item()
			self.abs_max_vals[name] = torch.abs(tensor).max().item()
		else:
			self.abs_min_vals[name] = min(self.abs_min_vals[name], torch.abs(tensor).min().item())
			self.abs_max_vals[name] = max(self.abs_max_vals[name], torch.abs(tensor).max().item())
		if name not in self.min_vals:
			self.min_vals[name] = tensor.min().item()
			self.max_vals[name] = tensor.max().item()
		else:
			self.min_vals[name] = min(self.min_vals[name], tensor.min().item())
			self.max_vals[name] = max(self.max_vals[name], tensor.max().item())

class MovingAverageMinMaxHook(BaseHook):
	def __init__(self, averaging_constant = 0.1):
		super().__init__()
		self.averaging_constant = averaging_constant
		self.min_vals = {}
		self.max_vals = {}
	
	def __call__(self, module, input, output, name):
		super().__call__(name)
		if name not in self.min_vals:
			self.min_vals[name] = output.min().item()
			self.max_vals[name] = output.max().item()
		else:
			new_min = output.min().item()
			new_max = output.max().item()
			self.min_vals[name] = (1 - self.averaging_constant) * self.min_vals[name] + self.averaging_constant * new_min
			self.max_vals[name] = (1 - self.averaging_constant) * self.max_vals[name] + self.averaging_constant * new_max

