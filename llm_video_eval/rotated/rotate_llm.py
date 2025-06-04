from transformers import Qwen2_5_VLForConditionalGeneration
import torch
from get_hadamard import sylvester_hadamard_28
from torch import nn
import sys 
sys.path.append("../")
from inference import Qwen2_5_VL_HF_Inference
from constants import MODEL_ID, MODEL_ID_GPT4_DEQ_PATH, CACHE_DIR, videos, prompts

llm_hidden_size = 3584
hadamard_matrix_llm: torch.Tensor = sylvester_hadamard_28(7).double().cuda()
model: Qwen2_5_VLForConditionalGeneration = None
n_llm_decoders = 28

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
    
def rotate_llm_model(model):
    # Register the hook to transform patch embeddings
    # embedding_layer = model.language_model.patch_embed
    hadamard_matrix_llm: torch.Tensor = sylvester_hadamard_28(7).double().cuda()
    old_forward = model.language_model.layers[0].forward
    rotation = torch.nn.Linear(llm_hidden_size, llm_hidden_size, False, model.device, dtype = torch.bfloat16)
    rotation.load_state_dict({
        "weight": hadamard_matrix_llm.T.bfloat16()
    })
    model.language_model.layers[0].register_module('rotation', rotation)
    def new_forward(*args, **kwargs):
        rotated = model.language_model.layers[0].rotation(*args)
        return old_forward(rotated, **kwargs)
    model.language_model.layers[0].forward = new_forward
    # hook_handle = embedding_layer.register_forward_hook(embedding_hook)

    # Optimize the model by folding normalization weights into linear layers
    # and applying basis transformation to maintain functionality while changing internal representations
    for i in range(n_llm_decoders):  # For each of the 32 transformer layers in the visual component
        hadamard_matrix_llm = hadamard_matrix_llm.to(model.language_model.layers[i].self_attn.q_proj.weight.device)
        #Fold norm1 weights into the QKV projection and replace with weight-less norm
        input_layernorm = model.language_model.layers[i].input_layernorm
        model.language_model.layers[i].self_attn.q_proj.weight.data = (model.language_model.layers[i].self_attn.q_proj.weight.data.double() * input_layernorm.weight.double()).to(torch.bfloat16)
        model.language_model.layers[i].self_attn.k_proj.weight.data = (model.language_model.layers[i].self_attn.k_proj.weight.data.double() * input_layernorm.weight.double()).to(torch.bfloat16)
        model.language_model.layers[i].self_attn.v_proj.weight.data = (model.language_model.layers[i].self_attn.v_proj.weight.data.double() * input_layernorm.weight.double()).to(torch.bfloat16)
        model.language_model.layers[i].input_layernorm = Qwen2RMSNormNoWeight(llm_hidden_size)
        
        # Apply basis transformation to QKV projection weights
        model.language_model.layers[i].self_attn.q_proj.weight.data = torch.matmul(model.language_model.layers[i].self_attn.q_proj.weight.data.double(), hadamard_matrix_llm).to(torch.bfloat16)
        model.language_model.layers[i].self_attn.k_proj.weight.data = torch.matmul(model.language_model.layers[i].self_attn.k_proj.weight.data.double(), hadamard_matrix_llm).to(torch.bfloat16)       
        model.language_model.layers[i].self_attn.v_proj.weight.data = torch.matmul(model.language_model.layers[i].self_attn.v_proj.weight.data.double(), hadamard_matrix_llm).to(torch.bfloat16)
        
        # Apply basis transformation to MLP down projection
        model.language_model.layers[i].mlp.down_proj.weight.data = torch.matmul(hadamard_matrix_llm.T, model.language_model.layers[i].mlp.down_proj.weight.data.double()).to(torch.bfloat16)

    for i in range(n_llm_decoders):  # Second pass through layers to handle remaining transformations
        hadamard_matrix_llm = hadamard_matrix_llm.to(model.language_model.layers[i].self_attn.q_proj.weight.device)
        #Fold norm2 weights into MLP gate and up projections
        post_layernorm = model.language_model.layers[i].post_attention_layernorm
        model.language_model.layers[i].mlp.gate_proj.weight.data = (model.language_model.layers[i].mlp.gate_proj.weight.data.double() * post_layernorm.weight.double()).to(torch.bfloat16)
        model.language_model.layers[i].mlp.up_proj.weight.data = (model.language_model.layers[i].mlp.up_proj.weight.data.double() * post_layernorm.weight.double()).to(torch.bfloat16)
        model.language_model.layers[i].post_attention_layernorm = Qwen2RMSNormNoWeight(llm_hidden_size)
        
        # Apply basis transformation to attention projection
        model.language_model.layers[i].self_attn.o_proj.weight.data = torch.matmul(hadamard_matrix_llm.T, model.language_model.layers[i].self_attn.o_proj.weight.data.double()).to(torch.bfloat16)
        
        # Apply basis transformation to MLP gate and up projections
        model.language_model.layers[i].mlp.gate_proj.weight.data = torch.matmul(model.language_model.layers[i].mlp.gate_proj.weight.data.double(),hadamard_matrix_llm).to(torch.bfloat16)
        model.language_model.layers[i].mlp.up_proj.weight.data = torch.matmul(model.language_model.layers[i].mlp.up_proj.weight.data.double(),hadamard_matrix_llm).to(torch.bfloat16)
    
    final_layernorm = model.language_model.norm
    model.lm_head.weight.data = (model.lm_head.weight.data.double() * final_layernorm.weight.double()).to(torch.bfloat16)
    model.lm_head.weight.data = torch.matmul(model.lm_head.weight.data.double(),hadamard_matrix_llm).to(torch.bfloat16)
    model.language_model.norm = Qwen2RMSNormNoWeight(llm_hidden_size)

    
    # # Fold normalization weights into the merger MLP
    # layernorm = model.language_model.merger.ln_q
    # model.language_model.merger.mlp[0].weight.data = (model.language_model.merger.mlp[0].weight.data.double() * layernorm.weight.double().repeat(4)).to(torch.bfloat16)
    # model.language_model.merger.ln_q = Qwen2RMSNormNoWeight(visual_hidden_size)
    # # Apply block-wise transformations to the merger MLP weights
    # weight_matrix = model.language_model.merger.mlp[0].weight.data.clone()
    # rotated_weight = weight_matrix.clone()

    # # Process the weight matrix in layers corresponding to spatial_merge_size
    # for row_block in range(spatial_merge_size ** 2):  
    #     for col_block in range(spatial_merge_size ** 2):  
    #         row_start = row_block * visual_hidden_size
    #         row_end = row_start + visual_hidden_size
    #         col_start = col_block * visual_hidden_size
    #         col_end = col_start + visual_hidden_size
            
    #         # Extract the block and apply the transformation
    #         block = weight_matrix[row_start:row_end, col_start:col_end]
    #         rotated_block = torch.matmul(block.double(), hadamard_matrix_visual).to(torch.float32)
            
    #         # Update in the rotated weight matrix
    #         rotated_weight[row_start:row_end, col_start:col_end] = rotated_block

    # Apply the transformed weights
    # model.language_model.merger.mlp[0].weight.data = rotated_weight.to(torch.bfloat16)

if __name__ == '__main__':
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID_GPT4_DEQ_PATH, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=CACHE_DIR, attn_implementation = "eager"
        )
    rotate_llm_model(model)
    inference = Qwen2_5_VL_HF_Inference(model, MODEL_ID)
    out = inference.infer(videos[-1], prompts[0])
    print(out)