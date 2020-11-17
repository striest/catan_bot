import torch

"""
Collection of utils on torch tensors
"""

def one_hot(x, n_classes):
	"""
	Note: Indexing starts at 0.
	"""
	if type(x) is int:
		out = torch.zeros(n_classes)
		out[x] = 1.0
		return out
	elif type(x) is torch.Tensor and x.dtype is torch.int64:
		size = 1 if len(x.shape) == 0 else x.shape[0]
		out = torch.zeros(size, n_classes)
		out[torch.arange(size), x] = 1.0
		return out.squeeze()
	else:
		print('Invalid input for one hot (expects int or LongTensor)')

#From rlkit: https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/pytorch_util.py
def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )
