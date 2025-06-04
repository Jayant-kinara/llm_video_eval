import torch
import numpy as np

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

hadamard = generate_random_orthogonal_matrix(6).double()
print("Hadamard matrix:")
print(hadamard)
print("\nVerify Hadamard is orthogonal (should be close to identity):")
print(torch.matmul(hadamard, hadamard.T))

sample = torch.randint(100,(6, 6)).float()
print("\nSample matrix:")
print(sample)
result = torch.matmul(hadamard, sample.double())
result = torch.matmul(result, hadamard.T)
print("\nResult of Hadamard matrix multiplication:")
print(result)