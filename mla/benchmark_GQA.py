from typing import Optional, Tuple, Union
from configuration_deepseek import DeepseekV2Config
from impl import *
import re
import torch
from torch import nn, Tensor
import torch.utils.benchmark as benchmark
import torch.nn.functional as F
from einops import einsum, rearrange

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
    force_grouped: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    # Ensure that the tensor dimensions match
    if not query.ndim == key.ndim == value.ndim == 4:
        raise ValueError(f"Expected query, key, and value to be 4-dimensional")

    query = rearrange(query, "b n h d -> b h n d")
    key = rearrange(key, "b s h d -> b h s d")
    value = rearrange(value, "b s h d -> b h s d")

    bq, hq, nq, dq = query.shape
    bk, hk, nk, dk = key.shape
    if scale is None:
        scale = dq ** 0.5
    query = query / scale

    num_head_groups = hq // hk
    query = rearrange(query, "b (h g) n d -> b g h n d", g=num_head_groups)
    similarity = einsum(query, key, "b g h n d, b h s d -> b g h n s")

    if is_causal:
        mask = torch.ones((bq, nq, nk), device=query.device).tril_()
    if mask is not None:
        if mask.ndim == 2:
            mask = rearrange(mask, "b s -> b () () () s")
        elif mask.ndim == 3:
            mask = rearrange(mask, "b n s -> b () () n s")
        similarity.masked_fill_(~mask, torch.finfo(similarity.dtype).min)

    attention = F.softmax(similarity, dim=-1)
    if dropout > 0.0:
        attention = F.dropout(attention, p=dropout)

    out = einsum(attention, value, "b g h n s, b h s d -> b g h n d")
    out = rearrange(out, "b g h n d -> b n (h g) d")

    attn_weights: Optional[Tensor] = None
    if need_weights:
        attn_weights = rearrange(attention, "b g h n s -> b n s (h g)")
        if average_attn_weights:
            attn_weights = attn_weights.mean(dim=1)

    return out, attn_weights


class MultiheadGQA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        query_heads: int,
        kv_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        layer_norm: bool = True,
        layer_norm_eps: float = 1e-5,
        gamma_init: float = 1.0,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,  # 确保 dtype 是一个可选参数
    ):
        super().__init__()
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.dropout = dropout
        self.layer_norm = layer_norm

        head_dim = embed_dim // query_heads
        kv_embed_dim = embed_dim // query_heads * kv_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype) 
        self.k_proj = nn.Linear(embed_dim, kv_embed_dim, bias=bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, kv_embed_dim, bias=bias, device=device, dtype=dtype)

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
            # 确保所有张量的 dtype 一致
            query = query.to(self.q_proj.weight.dtype)
            key = key.to(self.k_proj.weight.dtype)
            value = value.to(self.v_proj.weight.dtype)

            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

            # 获取 query_heads 和每个头的维度
            h = self.query_heads
            d = v.shape[-1] // h  # 计算 head_dim

            # 将 v 重新排列成 [batch_size, seq_len, heads, head_dim]
            v = rearrange(v, "b n (h d) -> b n h d", h=h, d=d)

            # 类似地处理 query 和 key
            q = rearrange(q, "b n (h d) -> b n h d", h=h, d=d)
            k = rearrange(k, "b n (h d) -> b n h d", h=self.kv_heads, d=d)

            # 使用 scaled_dot_product_gqa 计算注意力
            x, attn = scaled_dot_product_gqa(
                query=q, key=k, value=v,
                is_causal=is_causal, need_weights=need_weights,
                average_attn_weights=average_attn_weights,
            )

            # 将输出张量重新变换回原始形状
            x = rearrange(x, "b n h d -> b n (h d)")

            if self.layer_norm:
                x = self.norm(x)
            x = self.out_proj(x)

            return x, attn


class AttentionMHA(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1, **kwargs):
        super(AttentionMHA, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.attn = torch.nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, **kwargs)

    def forward(self, query, key_value, q_pos, kv_pos):
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
        self.q = torch.randn((self.bsz, self.q_len, config.hidden_size), dtype=config.torch_dtype, device=dev)
        self.kv = torch.randn((1, kv_len, config.hidden_size), dtype=config.torch_dtype, device=dev) 
        self.q_pos = torch.randint(0, config.max_position_embeddings-1, (self.bsz, self.q_len), dtype=torch.long, device=dev)
        self.kv_pos = torch.arange(0, self.kv_len, dtype=torch.long, device=dev).unsqueeze(0)
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

class MHABencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.attn = AttentionMHA(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,  
            dropout=config.attention_dropout
        ).cuda()
        self.kv = self.kv.repeat(self.bsz, 1, 1)
        self.kv_pos = self.kv_pos.repeat(self.bsz, 1)

    def iter(self):
        return self.attn.forward(self.q, self.kv, self.q_pos, self.kv_pos)

    def cache_size(self):
        return self.kv.numel() * self.kv.element_size()

    def memory_consume(self):
        return torch.cuda.max_memory_allocated()


class GQABencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.attn = MultiheadGQA(
            embed_dim=config.hidden_size, 
            query_heads=config.num_attention_heads,  # 使用 num_attention_heads
            kv_heads=config.num_key_value_heads,     # 使用 num_key_value_heads
            dropout=config.attention_dropout
        ).cuda()
        self.kv = self.kv.repeat(self.bsz, 1, 1)
        self.kv_pos = self.kv_pos.repeat(self.bsz, 1)

    def iter(self):
        return self.attn.forward(self.q, self.kv, self.kv, need_weights=False)

    def cache_size(self):
        return self.kv.numel() * self.kv.element_size()

    def memory_consume(self):
        return torch.cuda.max_memory_allocated()


STR_2_MODEL = {
    'MHA': MHABencher,
    'GQA': GQABencher,
}

ALL_BENCHMARKS = [
    MHABencher,
    GQABencher,
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

def profile(mod:str ,kv_len: int, bsz: int = 1, config: str = './config.json'):
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
        with_stack=True,  # Enable stack recording
        on_trace_ready=trace_handler,
    ) as prof:
        for i in range(15):
            with torch.profiler.record_function("inference"):
                model.iter()
                torch.cuda.synchronize()  # Ensure GPU operations are complete
            prof.step()

# repeat = out_len
# kv_len = in_len
# 时间是repeat if语句的时间
# 必须得用HF的LLAMA的GQA……
# attn的shape和占比（ncu）
# def main(bench: str, kv_len: Optional[int] = None, bsz: Optional[int] = None, config: str = './config.json', 
#          repeat: Optional[int] = None, min_run_time: float = 1.0, csv: bool = False, mode: str = 'single'):
#     cfg = DeepseekV2Config.from_json_file(config)

#     bencher: BenchmarkFixture
#     bencher = BENCHERS[bench](cfg, kv_len=kv_len, bsz=bsz)
    
#     if repeat is not None:
#         for _ in range(repeat):
#             bencher.iter()
#         torch.cuda.synchronize()
#         return
#     result = bencher.benchmark(min_run_time=min_run_time)
#     cache_size = bencher.cache_size()
#     device_name = torch.cuda.get_device_name()
#     memory = bencher.memory_consume()
#     if csv:
#         print(f'{bencher.name()},{bsz},{kv_len},{device_name},{cache_size},{result.mean},{result.median},{result._p25},{result._p75}')
#     else:
#         print(result)
#         print(f'Device: {device_name}')
#         print(f'KV Cache: {cache_size}')
#         print(f'Memory usage: {memory}')

# main.__doc__ = doc

# if __name__ == "__main__":
#     import fire
#     fire.Fire(main)
#     # profile(mod="MHA",kv_len=2048,bsz=16)  # Profile MHA
#     # profile(mod="GQA",kv_len=2048,bsz=16)  # Profile GQA
import time  # 导入时间模块用于测量执行时间

def main(bench: str, input_len: Optional[int] = None, output_len: Optional[int] = None, bsz: Optional[int] = None, 
         config: str = './config.json', csv: bool = False, gpu_id: int = 0):
    kv_len = input_len
    repeat = output_len

    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError("No GPUs are available.")
    elif gpu_id >= available_gpus:
        raise ValueError(f"Invalid GPU device ID {gpu_id}. Available GPU IDs: 0 to {available_gpus - 1}")
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(gpu_id)  

    cfg = DeepseekV2Config.from_json_file(config)
    bencher: BenchmarkFixture
    bencher = BENCHERS[bench](cfg, kv_len=kv_len, bsz=bsz)
    
    if repeat is not None:
        start_time = time.time()
        for _ in range(repeat):
            bencher.iter()
        torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time

    # 打印执行时间
    print(f"Execution time for out_len =  {repeat}: {elapsed_time * 1000:.2f} ms")

    # 继续基准测试
    # result = bencher.benchmark(min_run_time=min_run_time)
    cache_size = bencher.cache_size()
    device_name = torch.cuda.get_device_name()
    memory = bencher.memory_consume()
    
    if csv:
        print(f'{bencher.name()},{bsz},{kv_len},{device_name},{cache_size},{result.mean},{result.median},{result._p25},{result._p75}')
    else:
        # print(f"Mean time: {result.mean:.4f} s")
        # print(f"Median time: {result.median:.4f} s")
        # print(f"25th percentile: {result._p25:.4f} s")
        # print(f"75th percentile: {result._p75:.4f} s")
        print(f'Device: {device_name}')
        # print(f'KV Cache: {cache_size}')
        # print(f'Memory usage: {memory}')
        cache_size_gb = cache_size / (1024 ** 3)  # 转换为 GB
        memory_gb = memory / (1024 ** 3)          # 转换为 GB
        print(f'KV Cache: {cache_size_gb:.2f} GB')
        print(f'Memory usage: {memory_gb:.2f} GB')

main.__doc__ = doc

if __name__ == "__main__":
    import fire
    fire.Fire(main)
