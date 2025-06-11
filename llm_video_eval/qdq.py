# import pandas as pd
import torch
import functools

class Tee:
	def __init__(self, *files):
		self.files = files

	def write(self, obj):
		for f in self.files:
			f.write(obj)
			f.flush()  # Ensure the output is written immediately

	def flush(self):
		for f in self.files:
			f.flush()

# # Redirect stdout to a file
# # with open('results.txt', 'w') as f:
# curr_time = int(strftime('%Y%m%d%H%M%S'))
# f = open(f"output/checkpoint_commercial_results_{curr_time}.txt", 'w')
# tee = Tee(sys.stdout, f)
# sys.stdout = tee
# cumulative_results_df = pd.DataFrame()

def get_dtype_info( n_bits = 8, sign = True):
	dtype = torch.int8
	if sign :
		if n_bits == 8:
			dtype = torch.int8
		elif n_bits == 16:
			dtype = torch.int16
		elif n_bits == 32:
			dtype = torch.int32
		else:
			dtype = torch.int8
	else:
		if n_bits == 8:
			dtype = torch.uint8
		elif n_bits == 16:
			dtype = torch.int16
		elif n_bits == 32:
			dtype = torch.int32
		else:
			dtype = torch.uint8
	return dtype, torch.iinfo(dtype)


def qdq(tensor, scale, n_bits = 8, sign = True):
	dtype, dtype_info = get_dtype_info(n_bits, sign)
	quantized_tensor = torch.zeros_like(tensor, dtype=dtype)
	quantized_tensor = torch.clamp((tensor * scale).round() , dtype_info.min , dtype_info.max)
	qdq_tensor = quantized_tensor / scale
	return qdq_tensor

def round_ste(x: torch.Tensor):
	return (x.round() - x).detach() + x

def qdq_group(tensor: torch.Tensor, bits = 8, group_size = None, offset_enabled = False, is_two_power = False, axis = -1, is_unsigned = False):
	tensor_shape = tensor.shape
	if group_size is not None:
		shape = tensor.shape
		
		if axis == -1: 
			tensor = tensor.reshape(-1, group_size) 
		else:
			# axis += len(shape)
			shape = shape[axis + 1:]
			tensor = tensor.reshape(-1, group_size,*shape)
		
	reduce_shape = [axis]
	xmin = tensor.amin(reduce_shape, keepdim=True)
	xmax =  tensor.amax(reduce_shape, keepdim=True)
	if is_two_power:
		if offset_enabled:
			offset = xmax - 15	
			xmax = torch.clamp(xmax.sub(offset),0,15)
			xmin = torch.clamp(xmin.sub(offset),0,15)
			tensor = torch.clamp(tensor.sub(offset),0,15)
			abs_max = torch.max(xmax.abs(),xmin.abs())
			abs_max = abs_max + 1.22443e-15
			intPart = torch.floor(torch.log2(abs_max)) + torch.ones_like(abs_max)
			fracPart = (bits)*torch.ones_like(intPart) - intPart
			scale = (2**fracPart)
			scale = scale.pow(-1)
			tensor_int = round_ste(tensor / scale) 
			tensor_int = torch.clamp(tensor_int,  0, (2**(bits))-1)
			tensor_dequant = tensor_int.mul(scale)
		else:
			if is_unsigned:
				abs_max = torch.max(xmax.abs(),xmin.abs())
				abs_max = abs_max + 1.22443e-15
				intPart = torch.floor(torch.log2(abs_max)) + torch.ones_like(abs_max)
				fracPart = (bits)*torch.ones_like(intPart) - intPart
				scale = (2**fracPart)
				scale = scale.pow(-1)
				scale = scale.clamp(min=1e-6, max=1e6)
				tensor_int = torch.clamp(round_ste(tensor / scale) ,0, (2**(bits))-1)
				tensor_dequant = tensor_int.mul(scale)
			else:
				abs_max = torch.max(xmax.abs(),xmin.abs())
				abs_max = abs_max + 1.22443e-15
				intPart = torch.floor(torch.log2(abs_max)) + torch.ones_like(abs_max)
				fracPart = (bits-1)*torch.ones_like(intPart) - intPart
				scale = (2**fracPart)
				scale = scale.pow(-1)
				scale = scale.clamp(min=1e-6, max=1e6)
				tensor_int = torch.clamp(round_ste(tensor / scale) , -2**(bits-1), (2**(bits-1))-1)
				tensor_dequant = tensor_int.mul(scale)
	else:
		if offset_enabled:
			diff = xmax - xmin
			scale = diff/(2**(bits)-1)
			scale = scale.clamp(min=1e-6, max=1e6)
			offset = round_ste(-xmin/scale)
			tensor_int = round_ste(tensor / scale)
			tensor_int = tensor_int.add(offset)
			tensor_int = torch.clamp(tensor_int, 0, (2**(bits))-1)
			tensor_int = tensor_int.sub(offset)
			tensor_dequant = tensor_int.mul(scale)
		else:
			abs_max = torch.max(xmax.abs(),xmin.abs())
			scale = abs_max / (2**(bits-1) - 1)
			scale = scale.clamp(min=1e-6, max=1e6)
			tensor_int = torch.clamp(round_ste(tensor / scale) , -2**(bits-1), (2**(bits-1))-1)
			tensor_dequant = tensor_int.mul(scale)
	if group_size is not None:
		tensor_dequant = tensor_dequant.reshape(tensor_shape)
	return tensor_dequant

class OutputQDQ32_8_Hook_language(object):
	def __init__(self):
		self.q_bits = 32

	def __call__(self, module, input, output, dic, name):
		int32_scale = 2**14
		flag = False
		group_size = 64
		axis = -1

		qdq_int = qdq(output, int32_scale, 32, True)
		if "v_out" in name:
			shape = qdq_int.shape
			qdq_int = qdq_int.reshape(4,-1, 128)
			t = qdq_int.shape[1]
			x = 64 - (t%64)
			if x == 64:
				x = 0
			pad = (0,0,0,x)
			qdq_int = torch.nn.functional.pad(qdq_int, pad, "constant", 0)
			qdq_int = qdq_group(qdq_int, bits = 8, group_size = 64, offset_enabled = False, is_two_power = True, axis = -2)
			qdq_int = qdq_int[:,:t,:]
			qdq_int = qdq_int.reshape(*shape)
			output = qdq_int
		elif "sfmx" in name:
			t = qdq_int.shape[-1]
			x = 64 - (t%64)
			if x == 64:
				x = 0
			pad = (0,x)
			qdq_int = torch.nn.functional.pad(qdq_int, pad, "constant", 0)
			qdq_int = qdq_group(qdq_int, bits = 8, group_size = 64, offset_enabled = False, is_two_power = True, axis = -1, is_unsigned = True)
			qdq_int = qdq_int[:,:,:,:t]
			output = qdq_int
		else:
			qdq_int = qdq_group(qdq_int, bits = 8, group_size = 64, offset_enabled = False, is_two_power = True, axis = -1)
			# print(f"for module {name}, max diff in qdq is", (output - qdq_int).abs().max())
			output = qdq_int
		return output

def add_output_hooks_language(model, scales):
	output_qdq_hooks = []
	output_qdq_hook = OutputQDQ32_8_Hook_language()
	for name, module in model.named_modules():
		if (isinstance(module, torch.nn.Linear) and ("q_proj" not in name and "k_proj" not in name and "v_proj" not in name and "gate_proj" not in name)) or "q_out" in name or "k_out" in name or "v_out" in name or "sfmx" in name or "attn_output" in name or "act_fn" in name or "eltmul" in name or "norm" in name or "residue" in name:
			output_qdq_hooks.append(
				module.register_forward_hook(
					functools.partial(output_qdq_hook, dic = None, name = name)))	
			print("Added output qdq for: ", name)
	return model, output_qdq_hooks, output_qdq_hook

class OutputQDQ32_8_Hook_visual(object):
	def __init__(self):
		self.q_bits = 32

	def __call__(self, module, input, output, dic, name):
		int32_scale = 2**14
		flag = False
		group_size = 64
		axis = -1

		# if "q_proj" in name:
		# 	output = output / factor
		# if "k_proj" in name:
		# 	output = output / factor

		qdq_int = qdq(output, int32_scale, 32, True)
		# if "qkt_proj" in name:# or (("q_proj" in name or "k_proj" in name) and (".0." in name)):
		# 	qdq_int = qdq_group(qdq_int, bits = 8, group_size = None, offset_enabled = True, is_two_power = True)
		# else:
		if "qkv" in name: 
			qkv_shape = qdq_int.shape
			qkv = qdq_int.reshape(-1, 1280 * 3)
			q = qdq_group(qkv[:, :1280], bits = 8, group_size = 80, offset_enabled = False, is_two_power = True, axis = -1)
			k = qdq_group(qkv[:, 1280:2*1280], bits = 8, group_size = 80, offset_enabled = False, is_two_power = True, axis = -1)
			# q = qkv[:, :1280]
			# k = qkv[:, 1280:2*1280]
			v = qdq_group(qkv[:, 2*1280:3*1280], bits = 8, group_size = 64, offset_enabled = False, is_two_power = True, axis = -2)
			# v = qkv[:, 1280:2*1280]
			new_qkv = torch.cat((q, k, v), dim = -1)
			new_qkv = new_qkv.reshape(qkv_shape)
			output = new_qkv
		elif "sfmx" in name:
			qdq_int = qdq_group(qdq_int, bits = 8, group_size = 64, offset_enabled = False, is_two_power = True, axis = -1, is_unsigned=True)
			output = qdq_int
		else:
			qdq_int = qdq_group(qdq_int, bits = 8, group_size = 64, offset_enabled = False, is_two_power = True, axis = -1)
			output = qdq_int
		return output

def add_output_hooks_visual(model, scales):
	output_qdq_hooks = []
	output_qdq_hook = OutputQDQ32_8_Hook_visual()
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Linear) or "qkt_proj" in name or "sfmx" in name or "attn_output" in name or "act_fn" in name or "eltmul" in name or "norm" in name or "residue" in name:
			output_qdq_hooks.append(
				module.register_forward_hook(
					functools.partial(output_qdq_hook, dic = None, name = name)))	
			# print("Added output qdq for: ", name)
	return model, output_qdq_hooks, output_qdq_hook

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
		if isinstance(output, tuple):
			pass
		else:
		# Extract the minimum and maximum values from the output tensor
			if name not in self.min_vals:
				self.min_vals[name] = output.min().item()
				self.max_vals[name] = output.max().item()
			else:
				self.min_vals[name] = min(self.min_vals[name], output.min().item())
				self.max_vals[name] = max(self.max_vals[name], output.max().item())
	
def add_stat_hooks(model, observer):
	hooks = []
	hook = MinMaxHook()
	for name, module in model.named_modules():
		if isinstance(module, torch.nn.Linear) or "qkt_proj" in name or "q_out" in name or "k_out" in name or "sfmx" in name or "attn_output" in name or "act_fn" in name or "eltmul" in name or "norm" in name or "residue" in name:
			hooks.append(
				module.register_forward_hook(
					functools.partial(hook, name = name)
				)
			)

	return model, hooks, hook

def quantize_model_weights_language(model):
	for name,module in model.named_modules():
		if isinstance(module, torch.nn.Linear) and "lm_head" in name:
			print(f"Quantizing the weight of the Layer: {name}, bits: 4, group_size: 64", module.weight.shape)
			module.weight = torch.nn.Parameter(qdq_group(module.weight,4, 64, True))
		elif isinstance(module, torch.nn.Linear) and "visual" not in name:
			print(f"Quantizing the weight of the Layer: {name}, bits: 4, group_size: 64", module.weight.shape)
			module.weight = torch.nn.Parameter(qdq_group(module.weight,8, None, False))
			# qdq_group(tensor: torch.Tensor, bits = 8, group_size = None, offset_enabled = False, is_two_power = False, axis = -1):

def quantize_model_weights_vision(model):
	for name,module in model.named_modules():
		if isinstance(module, torch.nn.Linear):
			print(f"Quantizing the weight of the vision Layer: {name}, bits: 8, group_size: none", module.weight.shape)
			module.weight = torch.nn.Parameter(qdq_group(module.weight,8, None, False))
			



# print("*"*100)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#             MODEL_ID_GPT4_DEQ_PATH, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=CACHE_DIR, attn_implementation = "eager"
#         )

# rotate_visual_model(model)
# rotate_llm_model(model)

# quantize_model_weights_vision(model.visual)
# quantize_model_weights_language(model)

# model.visual, output_hooks, hook_ref = add_output_hooks_visual(model.visual, {})
# model.language_model, output_hooks, hook_ref = add_output_hooks_language(model.language_model, {})

# # model.language_model, stat_hooks, stat_hook_ref = add_stat_hooks(model.language_model, {})

# from inference import Qwen2_5_VL_HF_Inference
# from constants import videos, prompts

# inference = Qwen2_5_VL_HF_Inference(model, MODEL_ID, fps = 2, frame_height = 12, frame_width = 12, rotate = False)
# out = inference.infer(video = videos[-2], prompt = prompts[0])
# print(out[0])
