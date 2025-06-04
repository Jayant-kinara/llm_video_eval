import torch
from .get_hadamard import sylvester_hadamard_20, sylvester_hadamard_28

class Qwen2RMSNormNoWeight(torch.nn.Module):
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

def rotate_left(linear_layer: torch.nn.Linear, hadamard_matrix: torch.Tensor):
    '''
    Applies H^T W
    '''
    # this is necessary for bigger models, where its loaded on both gpus
    if linear_layer.weight.device != hadamard_matrix.device:
        hadamard_matrix = hadamard_matrix.to(linear_layer.weight.device)
    linear_layer.weight.data = torch.matmul(linear_layer.weight.data.double(), hadamard_matrix).to(torch.bfloat16)

def rotate_right(linear_layer: torch.nn.Linear, hadamard_matrix: torch.Tensor):
    '''
    Applies W H
    '''
    # this is necessary for bigger models, where its loaded on both gpus
    if linear_layer.weight.device != hadamard_matrix.device:
        hadamard_matrix = hadamard_matrix.to(linear_layer.weight.device)
    linear_layer.weight.data = torch.matmul(hadamard_matrix.T, linear_layer.weight.data.double()).to(torch.bfloat16)

def merge_lnorm_weights(linear_layer: torch.nn.Linear, lnorm: torch.Tensor):
    linear_layer.weight.data = (linear_layer.weight.data.double() * lnorm.weight.double()).to(torch.bfloat16)


def rotate_visual_model(model):
    visual_hidden_size = 1280
    k = 6
    block_had_size = 20
    assert(block_had_size * (2**k) == visual_hidden_size)
    num_hidden_layers = 32
    spatial_merge_size = 2
    hadamard_matrix_visual: torch.Tensor = sylvester_hadamard_20(k).double().cuda()
    old_forward = model.visual.patch_embed.forward
    rotation = torch.nn.Linear(visual_hidden_size, visual_hidden_size, False, model.device, dtype = torch.bfloat16)
    rotation.load_state_dict({
        "weight": hadamard_matrix_visual.T.bfloat16()
    })
    model.visual.patch_embed.register_module('rotation', rotation)
    def new_forward(pixels):
        output = old_forward(pixels)
        return model.visual.patch_embed.rotation(output)
    model.visual.patch_embed.forward = new_forward

    # Optimize the model by folding normalization weights into linear layers
    # and applying basis transformation to maintain functionality while changing internal representations
    for i in range(num_hidden_layers):  # For each of the 32 transformer blocks in the visual component
        # Fold norm1 weights into the QKV projection and replace with weight-less norm
        norm1 = model.visual.blocks[i].norm1
        merge_lnorm_weights(model.visual.blocks[i].attn.qkv, norm1)
        model.visual.blocks[i].norm1 = Qwen2RMSNormNoWeight(visual_hidden_size).to(model.visual.blocks[i].norm1.weight.device)
        
        # Apply basis transformation to QKV projection weights
        rotate_left(model.visual.blocks[i].attn.qkv, hadamard_matrix_visual) 
        
        # Apply basis transformation to MLP down projection
        rotate_right(model.visual.blocks[i].mlp.down_proj, hadamard_matrix_visual)
        model.visual.blocks[i].mlp.down_proj.bias.data = torch.matmul(hadamard_matrix_visual.T, model.visual.blocks[i].mlp.down_proj.bias.data.double()).to(torch.bfloat16)


    for i in range(num_hidden_layers):  # Second pass through blocks to handle remaining transformations
        # Fold norm2 weights into MLP gate and up projections
        norm2 = model.visual.blocks[i].norm2
        # model.visual.blocks[i].mlp.gate_proj.weight = rotate_left(model.visual.blocks[i].mlp.gate_proj, hadamard_matrix_visual)
        merge_lnorm_weights(model.visual.blocks[i].mlp.gate_proj, norm2)
        merge_lnorm_weights(model.visual.blocks[i].mlp.up_proj, norm2)
        model.visual.blocks[i].norm2 = Qwen2RMSNormNoWeight(visual_hidden_size).to(model.visual.blocks[i].norm2.weight.device)
        
        # Apply basis transformation to attention projection
        rotate_right(model.visual.blocks[i].attn.proj, hadamard_matrix_visual)
        model.visual.blocks[i].attn.proj.bias.data = torch.matmul(hadamard_matrix_visual.T, model.visual.blocks[i].attn.proj.bias.data.double()).to(torch.bfloat16)
        
        # Apply basis transformation to MLP gate and up projections
        rotate_left(model.visual.blocks[i].mlp.gate_proj, hadamard_matrix_visual)
        rotate_left(model.visual.blocks[i].mlp.up_proj, hadamard_matrix_visual)

    # Fold normalization weights into the merger MLP
    layernorm = model.visual.merger.ln_q
    model.visual.merger.mlp[0].weight.data = (model.visual.merger.mlp[0].weight.data.double() * layernorm.weight.repeat(4).double()).to(torch.bfloat16)
    model.visual.merger.ln_q = Qwen2RMSNormNoWeight(visual_hidden_size).to(model.visual.merger.ln_q.weight.device)
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

    # rotate last mlp 1 again with llm hadamard
    hadamard_matrix_llm: torch.Tensor = sylvester_hadamard_28(7).double().cuda()
    rotate_right(model.visual.merger.mlp[2], hadamard_matrix_llm)

def rotate_llm_model(model):
    # Register the hook to transform patch embeddings
    # embedding_layer = model.language_model.patch_embed
    llm_hidden_size = 3584
    k = 7
    block_had_size = 28
    assert(block_had_size * (2**k) == llm_hidden_size)
    n_llm_decoders = 28
    hadamard_matrix_llm: torch.Tensor = sylvester_hadamard_28(7).double().cuda()
    model.language_model.embed_tokens.weight.data = torch.matmul(model.language_model.embed_tokens.weight.data.double(), hadamard_matrix_llm).to(torch.bfloat16)

    # Optimize the model by folding normalization weights into linear layers
    # and applying basis transformation to maintain functionality while changing internal representations
    for i in range(n_llm_decoders):  # For each of the 32 transformer layers in the visual component
        #Fold norm1 weights into the QKV projection and replace with weight-less norm
        input_layernorm = model.language_model.layers[i].input_layernorm
        merge_lnorm_weights(model.language_model.layers[i].self_attn.q_proj, input_layernorm)
        merge_lnorm_weights(model.language_model.layers[i].self_attn.k_proj, input_layernorm)
        merge_lnorm_weights(model.language_model.layers[i].self_attn.v_proj, input_layernorm)
        model.language_model.layers[i].input_layernorm = Qwen2RMSNormNoWeight(llm_hidden_size).to(model.language_model.layers[i].input_layernorm.weight.device)
        
        # Apply basis transformation to QKV projection weights
        rotate_left(model.language_model.layers[i].self_attn.q_proj, hadamard_matrix_llm)
        rotate_left(model.language_model.layers[i].self_attn.k_proj, hadamard_matrix_llm)
        rotate_left(model.language_model.layers[i].self_attn.v_proj, hadamard_matrix_llm)
        
        # Apply basis transformation to MLP down projection
        rotate_right(model.language_model.layers[i].mlp.down_proj, hadamard_matrix_llm)

    for i in range(n_llm_decoders):  # Second pass through layers to handle remaining transformations
        #Fold norm2 weights into MLP gate and up projections
        post_layernorm = model.language_model.layers[i].post_attention_layernorm
        merge_lnorm_weights(model.language_model.layers[i].mlp.gate_proj, post_layernorm)
        merge_lnorm_weights(model.language_model.layers[i].mlp.up_proj, post_layernorm)
        model.language_model.layers[i].post_attention_layernorm = Qwen2RMSNormNoWeight(llm_hidden_size).to(model.language_model.layers[i].post_attention_layernorm.weight.device)
        
        # Apply basis transformation to attention projection
        rotate_right(model.language_model.layers[i].self_attn.o_proj, hadamard_matrix_llm)
        
        # Apply basis transformation to MLP gate and up projections
        rotate_left(model.language_model.layers[i].mlp.gate_proj, hadamard_matrix_llm)
        rotate_left(model.language_model.layers[i].mlp.up_proj, hadamard_matrix_llm)
    
    final_layernorm = model.language_model.norm
    merge_lnorm_weights(model.lm_head, final_layernorm)
    rotate_left(model.lm_head, hadamard_matrix_llm)
    model.language_model.norm = Qwen2RMSNormNoWeight(llm_hidden_size).to(model.language_model.norm.weight.device)