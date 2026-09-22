import torch

def index_first_axis(x, indices):
    return x[indices]

def pad_input(hidden_states, indices, cu_seqlens, max_seqlen):
    batch, seq, dim = hidden_states.shape
    flat = hidden_states.reshape(batch * seq, dim)
    return flat[indices], cu_seqlens, batch, seq

def unpad_input(hidden_states, attention_mask):
    indices = torch.nonzero(attention_mask.reshape(-1), as_tuple=False).flatten()
    lengths = attention_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = torch.cat([torch.zeros(1, device=hidden_states.device, dtype=torch.int32), lengths.cumsum(0)])
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])[indices]
    return flat, indices, cu_seqlens, int(lengths.max().item()) if lengths.numel() else 0

def rearrange(x, *args, **kwargs):
    return x
