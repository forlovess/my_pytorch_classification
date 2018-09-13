import torch
tensor = torch.rand(3,3)

tensor2 = torch.rand(3,3)
torch.cat(tensor,tensor2)