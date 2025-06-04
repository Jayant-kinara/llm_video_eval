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

H6 = generate_random_orthogonal_matrix(6).double()
example = torch.randint(100, (6, 6)).float()
mul1 = torch.matmul(H6, example.double())
H6_24 = torch.cat([torch.cat([H6] * 4, dim=1)] * 4, dim=0)

H24_24 = torch.cat([torch.cat([H6.T / 4] * 4, dim=1)] * 4, dim=0)

result = torch.matmul(H6_24, H24_24)
print("\nResult of matrix multiplication (6×24):")
print(f"Dimensions: {result.shape}")
print(result)

