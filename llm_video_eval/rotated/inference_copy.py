from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
import torch, functools
from pathlib import Path
import numpy as np
import random
from hooks import MinMaxHook
import csv  # Added for CSV export

# Path to the pretrained model
# Cache directory for model files
CACHE_DIR = Path("/auto/work/sw/dgundimeda")

# Layers to collect statistics from (attention projection and MLP down projection)
stat_layer_names = ["attn.proj", "mlp.down_proj"]
# Hidden size for visual 
visual_hidden_size = 1280
# Size parameter for spatial merging blocks
spatial_merge_size = 2
# Hidden size for Language model
model_hidden_size = 3584
# Random seed for reproducibility
seed = 1

def set_seed(seed):
    """
    Set random seeds for reproducibility across all random number generators.
    This ensures consistent results across multiple runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(seed)

def generate_random_orthogonal_matrix(dim, dtype=torch.float64, device='cpu'):
    """
    Generates a random orthogonal matrix of shape (dim, dim).
    
    An orthogonal matrix Q has the property Q^T * Q = I, which preserves vector norms
    during transformation. This is useful for maintaining stability during model weight modifications.
    
    Args:
        dim: Dimension of the square matrix
        dtype: Data type of the matrix
        device: Device to create the matrix on
    
    Returns:
        A random orthogonal matrix of shape (dim, dim)
    """
    # Create a random matrix
    random_matrix = torch.randn((dim, dim), dtype=dtype, device=device)
    # Perform QR decomposition to get an orthogonal matrix Q
    q, r = torch.linalg.qr(random_matrix)
    # Ensure a uniform distribution over orthogonal matrices by adjusting signs
    d = torch.diag(r)
    ph = d.sign()
    q *= ph
    return q

def add_stat_hooks(model, observer):
    """
    Adds hooks to collect statistics from specific layers during forward pass.
    
    Args:
        model: The model to add hooks to
        observer: The hook object that will collect statistics
    
    Returns:
        Tuple of (model, hooks_list, hook_object)
    """
    hooks = []
    hook = MinMaxHook()
    for name, module in model.named_modules():
        # Add hooks only to the specified layers
        #if any(name.endswith(layer_name) for layer_name in stat_layer_names):
        if isinstance(module, torch.nn.Linear) and "visual" in name:
            hooks.append(
                module.register_forward_hook(
                    functools.partial(hook, name = name)
                )
            )

    return model, hooks, hook

# Load the pretrained model with bfloat16 precision
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    CACHE_DIR / "Qwen2.5-VL-3B-Instruct-GPTQ-Decoder",
    torch_dtype=torch.bfloat16,
    device_map="auto",  # Automatically place model across available GPUs
)
model.eval()  # Set model to evaluation mode

# Alternative loading with flash attention for better performance
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# Set image resolution parameters
my_height = 16*28  # 448 pixels
my_width = 16*28   # 448 pixels
min_pixels = my_height * my_width  # Minimum number of pixels in processed image
max_pixels = my_height * my_width  # Maximum number of pixels in processed image

# Initialize the processor with specific image size constraints
processor = Qwen2_5_VLProcessor.from_pretrained(
    CACHE_DIR / "Qwen2.5-VL-3B-Instruct-GPTQ-Decoder",
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    use_fast=True,
    padding_side = "left",  # Padding on the left side for better attention
)

# Alternative processor configuration with different pixel ranges
# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# Video processing parameters
fps = 1  # Frames per second to extract from video

# Define the input message with a video and a prompt
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/auto/work/sw/kapil/qwen2_vl/demos/MicrosoftTeams-video_cropped.mp4",
                # Alternative video options:
                # "video": "/auto/worka/kapil/qwen2_vl/demos/Live accident caught on camera.mp4",
                # "video": "/auto/worka/kapil/qwen2_vl/demos/Falling_people.mp4",
                "fps": fps,
                "resized_height": my_height,
                "resized_width": my_width,
                # "max_pixels": 256 * 28 * 28
            },
            {"type": "text", "text": "Could you describe the video?"},
        ],
    }
]

# Prepare the input text by applying the chat template
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# Generate a random orthogonal matrix for the basis transformation
# This matrix will be used to transform the visual embeddings while preserving their norms
hadamard_matrix_visual = generate_random_orthogonal_matrix(visual_hidden_size, device=model.device).double()
hadamard_matrix_model = generate_random_orthogonal_matrix(model_hidden_size, device=model.device).double()


import torch.nn as nn
class Qwen2RMSNormNoWeight(nn.Module):
    """
    A modified version of RMSNorm that doesn't apply weights.
    
    This is used to replace the original normalization layers after 
    absorbing their weights into the subsequent layers.
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

def embedding_hook(module, input, output):
    """
    Hook function to transform patch embeddings using the orthogonal matrix.
    This applies the basis transformation to the output of the patch embedding layer.
    """
    return torch.matmul(output.double(), hadamard_matrix_visual).to(torch.bfloat16)


# Process the input messages to extract vision information
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

# Prepare inputs for the model
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
).to(model.device)

# Add hooks to collect statistics during inference
model, hooks, hook = add_stat_hooks(model, MinMaxHook())

# Generate text response
generated_ids = model.generate(**inputs, max_new_tokens = 1024)

# Print the min-max values for each monitored layer
print("Layer Statistics:")
layer_dict = {}
for name, value in hook.max_vals.items():
    # print(f"Range {name}: {hook.min_vals[name]} -> {value}")
    layer_dict[name] = {
        "min": hook.min_vals[name],
        "max": value,
        "range": value - hook.min_vals[name],
    }

del hook # Remove the hook after use
# Remove the hooks
for hook in hooks:
    hook.remove()
    

# Extract and decode the generated text
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("Generated output:")
print(output_text)
print(100*"#")
# Print the number of video tokens in the input
# print(f"Number of video tokens: {(inputs['input_ids'] == model.config.video_token_id).sum()}")


def rotate_visual_model():
    # Register the hook to transform patch embeddings
    embedding_layer = model.visual.patch_embed
    hook_handle = embedding_layer.register_forward_hook(embedding_hook)

    # Optimize the model by folding normalization weights into linear layers
    # and applying basis transformation to maintain functionality while changing internal representations
    for i in range(32):  # For each of the 32 transformer blocks in the visual component
        # Fold norm1 weights into the QKV projection and replace with weight-less norm
        layernorm = model.visual.blocks[i].norm1
        model.visual.blocks[i].attn.qkv.weight.data = (model.visual.blocks[i].attn.qkv.weight.data.double() * layernorm.weight.double()).to(torch.bfloat16)
        model.visual.blocks[i].norm1 = Qwen2RMSNormNoWeight(visual_hidden_size)
        
        # Apply basis transformation to QKV projection weights
        model.visual.blocks[i].attn.qkv.weight.data = torch.matmul(hadamard_matrix_visual.T, model.visual.blocks[i].attn.qkv.weight.data.T.double()).T.to(torch.bfloat16)
        
        # Apply basis transformation to MLP down projection
        model.visual.blocks[i].mlp.down_proj.weight.data = torch.matmul(hadamard_matrix_visual.T, model.visual.blocks[i].mlp.down_proj.weight.data.double()).to(torch.bfloat16)
        model.visual.blocks[i].mlp.down_proj.bias.data = torch.matmul(hadamard_matrix_visual.T, model.visual.blocks[i].mlp.down_proj.bias.data.double()).to(torch.bfloat16)


    for i in range(32):  # Second pass through blocks to handle remaining transformations
        # Fold norm2 weights into MLP gate and up projections
        layernorm = model.visual.blocks[i].norm2
        model.visual.blocks[i].mlp.gate_proj.weight.data = (model.visual.blocks[i].mlp.gate_proj.weight.data.double() * layernorm.weight.double()).to(torch.bfloat16)
        model.visual.blocks[i].mlp.up_proj.weight.data = (model.visual.blocks[i].mlp.up_proj.weight.data.double() * layernorm.weight.double()).to(torch.bfloat16)
        model.visual.blocks[i].norm2 = Qwen2RMSNormNoWeight(visual_hidden_size)
        
        # Apply basis transformation to attention projection
        model.visual.blocks[i].attn.proj.weight.data = torch.matmul(hadamard_matrix_visual.T, model.visual.blocks[i].attn.proj.weight.data.double()).to(torch.bfloat16)
        model.visual.blocks[i].attn.proj.bias.data = torch.matmul(hadamard_matrix_visual.T, model.visual.blocks[i].attn.proj.bias.data.double()).to(torch.bfloat16)
        
        # Apply basis transformation to MLP gate and up projections
        model.visual.blocks[i].mlp.gate_proj.weight.data = torch.matmul(model.visual.blocks[i].mlp.gate_proj.weight.data.double(), hadamard_matrix_visual).to(torch.bfloat16)
        model.visual.blocks[i].mlp.up_proj.weight.data = torch.matmul(model.visual.blocks[i].mlp.up_proj.weight.data.double(), hadamard_matrix_visual).to(torch.bfloat16)

    # Fold normalization weights into the merger MLP
    layernorm = model.visual.merger.ln_q
    model.visual.merger.mlp[0].weight.data = (model.visual.merger.mlp[0].weight.data.double() * layernorm.weight.double().repeat(4)).to(torch.bfloat16)
    model.visual.merger.ln_q = Qwen2RMSNormNoWeight(visual_hidden_size)
    # Apply block-wise transformations to the merger MLP weights
    weight_matrix = model.visual.merger.mlp[0].weight.data.clone()
    rotated_weight = weight_matrix.clone()

    # Process the weight matrix in blocks corresponding to spatial_merge_size
    for row_block in range(spatial_merge_size ** 2):  
        for col_block in range(spatial_merge_size ** 2):  
            row_start = row_block * visual_hidden_size
            row_end = row_start + visual_hidden_size
            col_start = col_block * visual_hidden_size
            col_end = col_start + visual_hidden_size
            
            # Extract the block and apply the transformation
            block = weight_matrix[row_start:row_end, col_start:col_end]
            rotated_block = torch.matmul(block.double(), hadamard_matrix_visual).to(torch.float32)
            
            # Update in the rotated weight matrix
            rotated_weight[row_start:row_end, col_start:col_end] = rotated_block

    # Apply the transformed weights
    model.visual.merger.mlp[0].weight.data = rotated_weight.to(torch.bfloat16)
    
from tqdm import tqdm 
def rotate_language_model():
    model.model.embed_tokens.weight.data = torch.matmul(model.model.embed_tokens.weight.data.double(), hadamard_matrix_model).to(torch.bfloat16)
    
    layernorm = model.model.norm
    model.lm_head.weight.data = (model.lm_head.weight.data.double() * layernorm.weight.double()).to(torch.bfloat16)
    model.lm_head.weight.data = torch.matmul(model.lm_head.weight.data.double(), hadamard_matrix_model.to(model.lm_head.weight.data.device)).to(torch.bfloat16)
    model.model.norm = Qwen2RMSNormNoWeight(model_hidden_size)
    
    for i in tqdm(range(28), desc="Rotating the LLM qkv"):  # For each of the 28 transformer blocks in the language component
        # Fold norm1 weights into the QKV projection and replace with weight-less norm
        layernorm = model.model.layers[i].input_layernorm
        model.model.layers[i].self_attn.q_proj.weight.data = (model.model.layers[i].self_attn.q_proj.weight.data.double() * layernorm.weight.double()).to(torch.bfloat16)
        model.model.layers[i].self_attn.k_proj.weight.data = (model.model.layers[i].self_attn.k_proj.weight.data.double() * layernorm.weight.double()).to(torch.bfloat16)
        model.model.layers[i].self_attn.v_proj.weight.data = (model.model.layers[i].self_attn.v_proj.weight.data.double() * layernorm.weight.double()).to(torch.bfloat16)
        model.model.layers[i].input_layernorm = Qwen2RMSNormNoWeight(model_hidden_size)
        print(layernorm.weight.shape)
        
        # Apply basis transformation to q,k,v projection weights
        model.model.layers[i].self_attn.q_proj.weight.data = torch.matmul(hadamard_matrix_model.T.to(model.model.layers[i].self_attn.q_proj.weight.data.device), model.model.layers[i].self_attn.q_proj.weight.data.T.double()).T.to(torch.bfloat16)
        model.model.layers[i].self_attn.k_proj.weight.data = torch.matmul(hadamard_matrix_model.T.to(model.model.layers[i].self_attn.k_proj.weight.data.device), model.model.layers[i].self_attn.k_proj.weight.data.T.double()).T.to(torch.bfloat16)
        model.model.layers[i].self_attn.v_proj.weight.data = torch.matmul(hadamard_matrix_model.T.to(model.model.layers[i].self_attn.v_proj.weight.data.device), model.model.layers[i].self_attn.v_proj.weight.data.T.double()).T.to(torch.bfloat16)
        
        # Apply basis transformation to MLP down projection
        model.model.layers[i].mlp.down_proj.weight.data = torch.matmul(hadamard_matrix_model.T.to(model.model.layers[i].mlp.down_proj.weight.data.device), model.model.layers[i].mlp.down_proj.weight.data.double()).to(torch.bfloat16)

    
    for i in tqdm(range(28), desc="Rotating the LLM MLP"):   # Second pass through blocks to handle remaining transformations
        # Fold norm2 weights into MLP gate and up projections
        layernorm = model.model.layers[i].post_attention_layernorm
        model.model.layers[i].mlp.gate_proj.weight.data = (model.model.layers[i].mlp.gate_proj.weight.data.double() * layernorm.weight.double()).to(torch.bfloat16)
        model.model.layers[i].mlp.up_proj.weight.data = (model.model.layers[i].mlp.up_proj.weight.data.double() * layernorm.weight.double()).to(torch.bfloat16)
        model.model.layers[i].post_attention_layernorm = Qwen2RMSNormNoWeight(model_hidden_size)
        
        # Apply basis transformation to attention projection
        model.model.layers[i].self_attn.o_proj.weight.data = torch.matmul(hadamard_matrix_model.T.to(model.model.layers[i].self_attn.o_proj.weight.data.device), model.model.layers[i].self_attn.o_proj.weight.data.double()).to(torch.bfloat16)
        
        # Apply basis transformation to MLP gate and up projections
        model.model.layers[i].mlp.gate_proj.weight.data = torch.matmul(model.model.layers[i].mlp.gate_proj.weight.data.double(), hadamard_matrix_model.to(model.model.layers[i].mlp.gate_proj.weight.data.device)).to(torch.bfloat16)
        model.model.layers[i].mlp.up_proj.weight.data = torch.matmul(model.model.layers[i].mlp.up_proj.weight.data.double(), hadamard_matrix_model.to(model.model.layers[i].mlp.up_proj.weight.data.device)).to(torch.bfloat16)

rotate_visual_model() # Rotate the visual model
# rotate_language_model() # Rotate the language model

# Add hooks to collect statistics during inference
model, hooks, hook = add_stat_hooks(model, MinMaxHook())

# Generate text response
generated_ids = model.generate(**inputs, max_new_tokens = 1024)

# Print the min-max values for each monitored layer
print("Layer Statistics:")
for name, value in hook.max_vals.items():
    # print(f"Range {name}: {hook.min_vals[name]} -> {value}")
    layer_dict[name] |= {
        "rotated_min": hook.min_vals[name],
        "rotated_max": value,
        "rotated_range": value - hook.min_vals[name]
    }

del hook
# Remove the hooks
for hook in hooks:
    hook.remove()
    

# Extract and decode the generated text
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("After Rotation Generated output:")
print(output_text)
import pandas as pd
df = pd.DataFrame(layer_dict).T
print(df)
df.to_csv("layer_statistics.csv", index=True, header=True)
model.save_pretrained("/auto/work/sw/kapil/models/durga--Qwen2.5-VL--rotated")
