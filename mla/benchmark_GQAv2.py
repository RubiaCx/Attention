from typing import Optional, Tuple, Union
from configuration_deepseek import DeepseekV2Config
from impl import *
import re
import torch
from torch import nn, Tensor
import torch.utils.benchmark as benchmark
import torch.nn.functional as F
from einops import einsum, rearrange
import time  # 导入时间模块用于测量执行时间
import math
torch.set_grad_enabled(False)

def scaled_dot_product_gqa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout: float = 0.0,
    scale: Optional[float] = None,
    mask: Optional[Tensor] = None,
    is_causal: Optional[bool] = None,
    need_weights: bool = False,
    average_attn_weights: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    # Ensure that the tensor dimensions match
    if not query.ndim == key.ndim == value.ndim == 4:
        raise ValueError(f"Expected query, key, and value to be 4-dimensional")
    # query: [batch_size, num_query_heads, seq_len, head_dim_q]
    # key/value: [batch_size, num_kv_heads, seq_len_kv, head_dim_kv]
    bsz, hq, seq_len_q, head_dim_q = query.shape
    _, hk, seq_len_kv, head_dim_kv = key.shape

    if scale is None:
        scale = head_dim_q ** 0.5
    query = query / scale

    # einsum需要严格对齐头维度
    if hq != hk:
        # expand key and value heads to match query heads
        # For GQA, keys and values are shared across groups of query heads
        # num_head_groups = hq // hk
        num_head_groups = math.gcd(hq, hk)
        key = key.unsqueeze(1).expand(-1, num_head_groups, -1, -1, -1)
        key = key.reshape(bsz, hq, seq_len_kv, head_dim_kv)
        value = value.unsqueeze(1).expand(-1, num_head_groups, -1, -1, -1)
        value = value.reshape(bsz, hq, seq_len_kv, head_dim_kv)

    # Compute attention scores
    # similarity: [batch_size, num_heads, seq_len_q, seq_len_kv]
    similarity = torch.einsum("bhqd,bhkd->bhqk", query, key)

    if is_causal:
        causal_mask = torch.tril(torch.ones((seq_len_q, seq_len_kv), device=query.device, dtype=torch.bool))
        similarity = similarity.masked_fill(~causal_mask, float('-inf'))

    if mask is not None:
        similarity = similarity.masked_fill(~mask.unsqueeze(1), float('-inf'))

    attention = F.softmax(similarity, dim=-1)
    if dropout > 0.0:
        attention = F.dropout(attention, p=dropout)

    # Compute the output
    out = torch.einsum("bhqk,bhkd->bhqd", attention, value)

    attn_weights: Optional[Tensor] = None
    if need_weights:
        attn_weights = attention.mean(dim=1) if average_attn_weights else attention

    return out, attn_weights


class MultiheadGQA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_query_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        layer_norm: bool = True,
        layer_norm_eps: float = 1e-5,
        gamma_init: float = 1.0,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.dropout = dropout
        self.layer_norm = layer_norm

        if embed_dim % num_query_heads != 0:
            raise ValueError(f"embed_dim must be divisible by num_query_heads")

        if embed_dim % num_kv_heads != 0:
            raise ValueError(f"embed_dim must be divisible by num_kv_heads")
        # 确保 head_dim 在查询和键/值之间保持一致
        self.head_dim_q = embed_dim // num_query_heads
        self.head_dim_kv = embed_dim // num_kv_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)

        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps, device=device, dtype=dtype) if layer_norm else None
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        need_weights: bool = False,
        is_causal: bool = False,
        average_attn_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        query = query.to(self.q_proj.weight.dtype)
        key = key.to(self.k_proj.weight.dtype)
        value = value.to(self.v_proj.weight.dtype)

        # Project inputs
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        bsz, seq_len_q, _ = q.shape
        _, seq_len_kv, _ = k.shape

        # Reshape for multi-head attention
        q = q.view(bsz, seq_len_q, self.num_query_heads, self.head_dim_q).transpose(1, 2)  # [bsz, num_query_heads, seq_len_q, head_dim_q]
        k = k.view(bsz, seq_len_kv, self.num_kv_heads, self.head_dim_kv).transpose(1, 2)   # [bsz, num_kv_heads, seq_len_kv, head_dim_kv]
        v = v.view(bsz, seq_len_kv, self.num_kv_heads, self.head_dim_kv).transpose(1, 2)   # [bsz, num_kv_heads, seq_len_kv, head_dim_kv]

        # Apply scaled dot-product GQA
        out, attn_weights = scaled_dot_product_gqa(
            query=q,
            key=k,
            value=v,
            is_causal=is_causal,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
        )

        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len_q, self.embed_dim)

        if self.layer_norm:
            out = self.norm(out)
        out = self.out_proj(out)

        return out, attn_weights


class AttentionMHA(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1, **kwargs):
        super(AttentionMHA, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.attn = torch.nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True, **kwargs)

    def forward(self, query, key_value, q_pos=None, kv_pos=None):
        attn_output, _ = self.attn(query, key_value, key_value)
        return self.dropout(attn_output)


class BenchmarkFixture:
    config: DeepseekV2Config
    q: torch.Tensor
    kv: torch.Tensor
    q_pos: torch.LongTensor
    kv_pos: torch.LongTensor

    def __init__(self, config: DeepseekV2Config, kv_len: int, q_len: int = 1, bsz: int = 1, dev='cuda'):
        self.config = config
        self.bsz = bsz
        self.q_len = q_len
        self.kv_len = kv_len
        self.device = torch.device(dev)

        self.q = torch.randn((self.bsz, self.q_len, config.hidden_size), dtype=config.torch_dtype, device=self.device)
        self.kv = torch.randn((self.bsz, self.kv_len, config.hidden_size), dtype=config.torch_dtype, device=self.device)
        self.q_pos = torch.randint(0, config.max_position_embeddings - 1, (self.bsz, self.q_len), dtype=torch.long, device=self.device)
        self.kv_pos = torch.arange(0, self.kv_len, dtype=torch.long, device=self.device).unsqueeze(0).repeat(self.bsz, 1)

        cfg_dict = config.to_dict()
        cfg_dict['torch_dtype'] = config.torch_dtype
        self.cfg_dict = cfg_dict

    def benchmark(self, min_run_time: float = 1.0):
        return benchmark.Timer(
            stmt='bencher.iter()',
            globals={'bencher': self},
            label=self.name(),
            sub_label=f'kv_len={self.kv_len}',
        ).blocked_autorange(min_run_time=min_run_time)

    @classmethod
    def name(cls):
        return cls.__name__.removesuffix('Bencher')

    @classmethod
    def short_name(cls):
        return re.sub('[^A-Z_]', '', cls.name())

    def cache_size(self):
        return 0

    def memory_consume(self):
        return torch.cuda.max_memory_allocated()


class MHABencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.attn = AttentionMHA(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout
        ).to(self.device)
        self.kv = self.kv.to(self.device)
        self.q = self.q.to(self.device)
        self.kv_pos = self.kv_pos.to(self.device)
        self.q_pos = self.q_pos.to(self.device)

    def iter(self):
        return self.attn.forward(self.q, self.kv, self.q_pos, self.kv_pos)

    def cache_size(self):
        return self.kv.numel() * self.kv.element_size()


class GQABencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.attn = MultiheadGQA(
            embed_dim=config.hidden_size,
            num_query_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            dropout=config.attention_dropout
        ).to(self.device)

        self.kv = self.kv.to(self.device)
        self.q = self.q.to(self.device)
        self.kv_pos = self.kv_pos.to(self.device)
        self.q_pos = self.q_pos.to(self.device)

    def iter(self):
        return self.attn.forward(self.q, self.kv, self.kv, need_weights=False)

    def cache_size(self):
        # Key and value cache size
        head_dim_kv = self.attn.head_dim_kv
        kv_heads = self.attn.num_kv_heads
        cache_elements = self.bsz * self.kv_len * kv_heads * head_dim_kv * 2  # Multiply by 2 for key and value
        element_size = self.kv.element_size()
        return cache_elements * element_size


'''
定义了不同的基准测试类
- BaselineBencher 使用 AttentionBaseline 类来计算注意力机制
- CacheDecompressedBencher 和 CacheCompressedBencher 表示不同的缓存机制实现
- 其他 Bencher 类根据需要进行定义
'''

class BaselineBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.attn = AttentionBaseline(**self.cfg_dict).to(self.device)
        self.kv = self.kv.to(self.device)
        self.q = self.q.to(self.device)
        self.kv_pos = self.kv_pos.to(self.device)
        self.q_pos = self.q_pos.to(self.device)

    def iter(self):
        return self.attn.forward(self.q, self.kv, self.q_pos, self.kv_pos)


class CacheDecompressedBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.attn = AttentionCacheDecompressed(**self.cfg_dict).to(self.device)
        k, v = self.attn.decompress_kv(self.kv, self.kv_pos)
        self.decompressed = (
            k.repeat(self.bsz, 1, 1, 1).to(self.device),
            v.repeat(self.bsz, 1, 1, 1).to(self.device)
        )
        self.q = self.q.to(self.device)
        self.q_pos = self.q_pos.to(self.device)

    def iter(self):
        k, v = self.decompressed
        return self.attn.forward(self.q, self.q_pos, k, v)

    def cache_size(self):
        k, v = self.decompressed
        return k.numel() * k.element_size() + v.numel() * v.element_size()


class CacheCompressedBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.attn = AttentionCacheCompressed(**self.cfg_dict).to(self.device)
        self.compressed = self.attn.compress_kv(self.kv, self.kv_pos).repeat(self.bsz, 1, 1).to(self.device)
        self.q = self.q.to(self.device)
        self.q_pos = self.q_pos.to(self.device)

    def iter(self):
        return self.attn.forward(self.q, self.q_pos, self.compressed)

    def cache_size(self):
        return self.compressed.numel() * self.compressed.element_size()


class AbsorbedBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.attn = AttentionAbsorbed(**self.cfg_dict).cuda()
        self.kv = self.kv.repeat(self.bsz, 1, 1)
        self.kv_pos = self.kv_pos.repeat(self.bsz, 1)
    
    def iter(self):
        return self.attn.forward(self.q, self.kv, self.q_pos, self.kv_pos)
    
    def memory_consume(self):
        return torch.cuda.max_memory_allocated()

class Absorbed_CacheCompressedBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.attn = AttentionAbsorbed_CacheCompressed(**self.cfg_dict).cuda()
        self.compressed = self.attn.compress_kv(self.kv, self.kv_pos).repeat(self.bsz, 1, 1)

    def iter(self):
        return self.attn.forward(self.q, self.q_pos, self.compressed)
    
    def cache_size(self):
        return self.compressed.numel() * self.compressed.element_size()
    
    def memory_consume(self):
        return torch.cuda.max_memory_allocated()

class Absorbed_CacheCompressed_MoveElisionBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.attn = AttentionAbsorbed_CacheCompressed_MoveElision(**self.cfg_dict).cuda()
        self.compressed = self.attn.compress_kv(self.kv, self.kv_pos).repeat(self.bsz, 1, 1)

    def iter(self):
        return self.attn.forward(self.q, self.q_pos, self.compressed)
    
    def cache_size(self):
        return self.compressed.numel() * self.compressed.element_size()
    
    def memory_consume(self):
        return torch.cuda.max_memory_allocated()

class AbsorbedMaterialized_CacheCompressed_MoveElisionBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.attn = AttentionAbsorbedMaterialized_CacheCompressed_MoveElision(**self.cfg_dict).cuda()
        self.compressed = self.attn.compress_kv(self.kv, self.kv_pos).repeat(self.bsz, 1, 1)

    def iter(self):
        return self.attn.forward(self.q, self.q_pos, self.compressed)
    
    def cache_size(self):
        return self.compressed.numel() * self.compressed.element_size()
    
    def memory_consume(self):
        return torch.cuda.max_memory_allocated()

STR_2_MODEL = {
    'MHA': MHABencher,
    'GQA': GQABencher,
    'B'       : BaselineBencher,
    'CC'      : CacheCompressedBencher,
    'CD'      : CacheDecompressedBencher,
    'A'       : AbsorbedBencher,
    'A_CC'    : Absorbed_CacheCompressedBencher,
    'A_CC_ME' : Absorbed_CacheCompressed_MoveElisionBencher,
    'AM_CC_ME': AbsorbedMaterialized_CacheCompressed_MoveElisionBencher,
}

ALL_BENCHMARKS = [
    MHABencher,
    GQABencher,
    BaselineBencher,
    CacheCompressedBencher,
    CacheDecompressedBencher,
    AbsorbedBencher,
    Absorbed_CacheCompressedBencher,
    Absorbed_CacheCompressed_MoveElisionBencher,
    AbsorbedMaterialized_CacheCompressed_MoveElisionBencher,
]


BENCHERS = {}

doc = 'Run benchmark on various attention mechanisms (MHA and GQA)\n\n'

for bencher in ALL_BENCHMARKS:
    name = bencher.name()
    short_name = bencher.short_name()
    BENCHERS[name] = bencher
    BENCHERS[short_name] = bencher
    doc += f'{short_name}\t{name}\n'

def trace_handler(prof):
    print(f"profile saved to prof_dir/timeline.json")
    prof.export_chrome_trace(f"./prof_dir/baseline.json")

def profile(mod: str, kv_len: int, bsz: int = 1, config: str = './config.json'):
    cfg = DeepseekV2Config.from_json_file(config)
    model = STR_2_MODEL[mod](cfg, kv_len, bsz=bsz)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,  # 设置为 True 以记录堆栈信息
        on_trace_ready=trace_handler,
    ) as prof:
        for i in range(15):
            with torch.profiler.record_function("inference"):
                model.iter()
                torch.cuda.synchronize()  # 确保所有 GPU 操作都完成

def main(
    gpu_id: int = 0,
    bench: str = "GQA",
    input_len: int = 1024,  
    output_len: int = 1024,  
    batch_size: int = 32,  
    config: str = './config.json',  
    csv: bool = False  #
):
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError("No GPUs are available.")
    elif gpu_id >= available_gpus:
        raise ValueError(f"Invalid GPU device ID {gpu_id}. Available GPU IDs: 0 to {available_gpus - 1}")
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(gpu_id)
    
    cfg = DeepseekV2Config.from_json_file(config)
    kv_len = input_len
    repeat = output_len
    bencher_class = BENCHERS.get(bench)
    if bencher_class is None:
        raise ValueError(f"Unknown benchmark: {bench}")
    bencher = bencher_class(cfg, kv_len=kv_len, q_len=1, bsz=batch_size, dev=device)

    start_time = time.time()
    for _ in range(repeat):
        bencher.iter()
    torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 打印执行时间
    print(f"Execution time for out_len = {repeat}: {elapsed_time * 1000:.2f} ms")

    cache_size = bencher.cache_size()
    device_name = torch.cuda.get_device_name(gpu_id)
    memory = bencher.memory_consume()

    if csv:
        print(f'{bencher.name()},{batch_size},{kv_len},{device_name},{cache_size},{elapsed_time / repeat},{elapsed_time / repeat},{elapsed_time / repeat},{elapsed_time / repeat}')
    else:
        print(f'Device: {device_name}')
        cache_size_gb = cache_size / (1024 ** 3)  # 转换为 GB
        memory_gb = memory / (1024 ** 3)          # 转换为 GB
        print(f'KV Cache: {cache_size_gb:.2f} GB')
        print(f'Memory usage: {memory_gb:.2f} GB')

main.__doc__ = doc

if __name__ == "__main__":
    import fire
    fire.Fire(main)
