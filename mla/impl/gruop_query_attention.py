from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
import math

# https://ai.plainenglish.io/understanding-llama2-kv-cache-grouped-query-attention-rotary-embedding-and-more-c17e5f49a6d7
def apply_rotary_embeddings(x, freqs_complex, device):
    # (m, seq_len, num_heads, head_dim/2, 2)
    x = x.float().reshape(*x.shape[:-1], -1, 2)
    # (m, seq_len, num_heads, head_dim/2)
    x_complex = torch.view_as_complex(x)
    # (seq_len, head_dim/2) --> (1, seq_len, 1, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # multiply each complex number
    # (m, seq_len, n_heads, head_dim/2)
    x_rotated = x_complex * freqs_complex
    # convert back to the real number
    # (m, seq_len, n_heads, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (m, seq_len, n_heads, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

"""
repeat_kv 函数用于重复键和值张量的头部，使它们的形状与查询张量一致
"""
def repeat_kv(x, n_rep):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        # (m, seq_len, n_kv_heads, 1, head_dim)
        # --> (m, seq_len, n_kv_heads, n_rep, head_dim)
        # --> (m, seq_len, n_kv_heads * n_rep, head_dim)
        return (
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )

class KVCache:
    def __init__(self, max_seq_len, n_kv_heads, head_dim, device, max_batch_size = 128):
        self.cache_k = torch.zeros((max_batch_size, max_seq_len, n_kv_heads, head_dim)).to(device)
        self.cache_v = torch.zeros((max_batch_size, max_seq_len, n_kv_heads, head_dim)).to(device)

    def update(self, batch_size, start_pos, xk, xv):
        if isinstance(start_pos, torch.Tensor):
            if start_pos.numel() == 1:  # 如果 start_pos 只包含一个元素
                start_pos = start_pos.item()  # 转换为 Python 标量
            else:
                raise ValueError("start_pos should be a scalar, but it contains more than one element.")
        else:
            start_pos = int(start_pos)  # 如果它是一个数字，确保它是整数
        seq_len = int(xk.size(1))   # 确保 xk.size(1) 是整数
        
        # 更新 cache
        self.cache_k[:batch_size, start_pos :start_pos + xk.size(1)] = xk
        self.cache_v[:batch_size, start_pos :start_pos + xv.size(1)] = xv

    def get(self, batch_size, start_pos, seq_len):
        keys = self.cache_k[:batch_size,  :start_pos + seq_len]
        values = self.cache_v[:batch_size, :start_pos + seq_len]
        return keys, values
    
class LlamaAttention(nn.Module):
    def __init__(self, 
                 hidden_size: int, 
                 num_attention_heads: int, 
                 kv_lora_rank: int, 
                 v_head_dim: int, 
                 q_lora_rank: int, 
                 qk_rope_head_dim: int, 
                 qk_nope_head_dim: int, 
                 max_position_embeddings: int,
                 max_batch_size: int = 1,
                 torch_dtype: torch.dtype = torch.bfloat16,  
                 attention_bias: bool = False, 
                 *args, **kwargs):
        super().__init__()

        self.n_heads = num_attention_heads
        self.n_kv_heads = kv_lora_rank
        self.dim = hidden_size
        self.n_heads_q = self.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = v_head_dim
        
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False, dtype=torch_dtype)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False, dtype=torch_dtype)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False, dtype=torch_dtype)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False, dtype=torch_dtype)

        self.cache = KVCache(
            max_batch_size=max_batch_size,  # example default value
            max_seq_len=max_position_embeddings,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            device=kwargs.get('device', 'cuda')
        )

    def forward(self, x, start_pos, freqs_complex, extra_param=None):

        batch_size, seq_len, _ = x.shape

        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        self.cache.update(batch_size, start_pos, xk, xv)

        keys, values = self.cache.get(batch_size, start_pos, seq_len)

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        output = torch.matmul(scores, values)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.wo(output)
