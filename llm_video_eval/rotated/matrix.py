# import torch

# def generate_random_orthogonal_matrix(dim, dtype=torch.float32, device='cpu'):
#     """
#     Generates a random orthogonal matrix of shape (dim, dim).
#     """
#     # Create a random matrix
#     random_matrix = torch.randn((dim, dim), dtype=dtype, device=device)
#     # Perform QR decomposition
#     q, r = torch.linalg.qr(random_matrix)
#     # Ensure a uniform distribution over orthogonal matrices
#     d = torch.diag(r)
#     ph = d.sign()
#     q *= ph
#     return q

# m = generate_random_orthogonal_matrix(6)
# transpose_m = m.T
# result = torch.matmul(m, transpose_m)
# sample = torch.randint(100,(6, 6)).float()
# sample2 = torch.randint(10,(8, 6)).float()
# prod = torch.matmul(sample, sample2.T)

# prod1 = torch.matmul(sample, m)
# prod2 = torch.matmul(m.T, sample2.T)
# prod3 = torch.matmul(prod1, prod2)
# diff = prod3 - prod
# print(diff)


# #########################


import torch
import numpy as np

# Create an example Hadamard-like matrix (normalized)
# def create_hadamard_matrix(n):
#     """Create a Hadamard matrix of size n x n."""
#     # Start with the 1x1 Hadamard matrix
#     H = torch.ones(1, 1)
    
#     # Calculate size needed (next power of 2)
#     import math
#     m = 2 ** math.ceil(math.log2(n))
    
#     # Generate Hadamard matrix using Sylvester's construction
#     while H.shape[0] < m:
#         H = torch.cat([
#             torch.cat([H, H], dim=1),
#             torch.cat([H, -H], dim=1)
#         ], dim=0)
    
#     # Normalize
#     H = H / torch.sqrt(torch.tensor(m, dtype=torch.float32))
    
#     # Truncate if necessary
#     if m > n:
#         H = H[:n, :n]
    
#     return H

def generate_random_orthogonal_matrix(dim, dtype=torch.float64, device='cpu'):
    """
    Generates a random orthogonal matrix of shape (dim, dim).
    """
    # Create a random matrix
    random_matrix = torch.randn((dim, dim), dtype=dtype, device=device)
    # Perform QR decomposition
    q, r = torch.linalg.qr(random_matrix)
    # Ensure a uniform distribution over orthogonal matrices
    d = torch.diag(r)
    ph = d.sign()
    q *= ph
    return q

# Set random seed for reproducibility
torch.manual_seed(42)

# Create sample data
# Visual transformer output: 20 rows (patches) with 6 features each
visual_output = torch.randint(100, (20, 6), dtype=torch.float32)
weight_matrix = torch.randint(10, (24, 24), dtype=torch.float32)

reshaped_visual_output = visual_output.reshape(-1, 24)
original_matmul = torch.matmul(reshaped_visual_output, weight_matrix.T)
print("Original matmul : ", original_matmul.shape)
print(original_matmul[0][:-1])

# Create the 6x6 Hadamard matrix for rotation
hadamard = generate_random_orthogonal_matrix(6).double()
print("Hadamard matrix:")
print(hadamard)
print("\nVerify Hadamard is orthogonal (should be close to identity):")
print(torch.matmul(hadamard, hadamard.T))

rotated_visual_output = torch.matmul(visual_output.double(), hadamard).to(torch.float32)
rotated_visual_output = rotated_visual_output.reshape(-1, 24)

# Step 4: Apply inverse rotation to the weight matrix
# For a 24x24 matrix with 6x6 rotations, we need to process 16 blocks
rotated_weight = weight_matrix.clone()
for row_block in range(4):  # 24/6 = 4
    for col_block in range(4):  # 24/6 = 4
        row_start = row_block * 6
        row_end = row_start + 6
        col_start = col_block * 6
        col_end = col_start + 6
        
        # Get the block
        block = weight_matrix[row_start:row_end, col_start:col_end]
        
        # Apply inverse Hadamard (which is the same as forward for Hadamard)
        rotated_block = torch.matmul(block.double(), hadamard/4).to(torch.float32)
        
        # Update in the rotated weight matrix
        rotated_weight[row_start:row_end, col_start:col_end] = rotated_block

# Apply the rotated weight matrix
rotated_output = torch.matmul(rotated_visual_output, rotated_weight.T)
print("\nRotated approach output shape:", rotated_output.shape)
print(rotated_output[0][:-1])

# Check if outputs are the same (should be very close)
diff = rotated_output - original_matmul
print("\nMax absolute difference:", torch.max(torch.abs(diff)))
print("Mean absolute difference:", torch.mean(torch.abs(diff)))

# Verification that activations have better numerical properties
# print("\nOriginal patches max value:", torch.max(torch.abs(visual_output)))
# print("Rotated patches max value:", torch.max(torch.abs(rotated_patches)))
# print("Original patches standard deviation:", torch.std(visual_output))
# print("Rotated patches standard deviation:", torch.std(rotated_patches))