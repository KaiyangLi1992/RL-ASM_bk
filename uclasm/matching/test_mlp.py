from torch_scatter import scatter_mean
import torch

# Create a tensor to use as an index guide
index_tensor = torch.tensor([0, 0, 0, 1, 1])

# We'll create a mock Xq tensor for the purpose of this example
Xq = torch.randn(5, 10)  # Random tensor for demonstration

# Use scatter_mean from PyTorch Geometric to compute the mean for each group
group_means = scatter_mean(Xq, index_tensor,dim=0)

# Determine the number of times each group appears in the index tensor
group_counts = torch.bincount(index_tensor)

# Use repeat_interleave to expand the group means
expanded_means_pyg = torch.repeat_interleave(group_means, group_counts, dim=0)
print(expanded_means_pyg)
print(expanded_means)
