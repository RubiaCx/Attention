from typing import Optional
from configuration_deepseek import DeepseekV2Config
from impl import *
import re
import torch
import torch.utils.benchmark as benchmark

torch.set_grad_enabled(False)

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

'''
定义了不同的基准测试类
- BaselineBencher 使用 AttentionBaseline 类来计算注意力机制
- CacheDecompressedBencher 和 CacheCompressedBencher 表示不同的缓存机制实现
- 
'''
class BaselineBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.attn = AttentionBaseline(**self.cfg_dict).cuda()
        self.kv = self.kv.repeat(self.bsz, 1, 1)
        self.kv_pos = self.kv_pos.repeat(self.bsz, 1)
    
    def iter(self):
        return self.attn.forward(self.q, self.kv, self.q_pos, self.kv_pos)
    
    def memory_consume(self):
        return torch.cuda.max_memory_allocated()

class CacheDecompressedBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.attn = AttentionCacheDecompressed(**self.cfg_dict).cuda()
        k, v = self.attn.decompress_kv(self.kv, self.kv_pos)
        self.decompressed = k.repeat(self.bsz, 1, 1, 1), v.repeat(self.bsz, 1, 1, 1)
        
    def iter(self):
        k, v = self.decompressed
        return self.attn.forward(self.q, self.q_pos, k, v)
    
    def cache_size(self):
        k, v = self.decompressed
        return k.numel() * k.element_size() + v.numel() * v.element_size()

    def memory_consume(self):
        return torch.cuda.max_memory_allocated()

class CacheCompressedBencher(BenchmarkFixture):
    def __init__(self, config: DeepseekV2Config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.attn = AttentionCacheCompressed(**self.cfg_dict).cuda()
        self.compressed = self.attn.compress_kv(self.kv, self.kv_pos).repeat(self.bsz, 1, 1)

    
    def iter(self):
        return self.attn.forward(self.q, self.q_pos, self.compressed)
    
    def cache_size(self):
        return self.compressed.numel() * self.compressed.element_size()
    
    def memory_consume(self):
        return torch.cuda.max_memory_allocated()

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

STR_2_MODEL={
    'B':BaselineBencher,
    'CC':CacheCompressedBencher,
    'CD':CacheDecompressedBencher,
    'A':AbsorbedBencher,
    'A_CC':Absorbed_CacheCompressedBencher,
    'A_CC_ME':Absorbed_CacheCompressed_MoveElisionBencher,
    'AM_CC_ME': AbsorbedMaterialized_CacheCompressed_MoveElisionBencher,
}

ALL_BENCHMARKS = [
    BaselineBencher,
    CacheCompressedBencher,
    CacheDecompressedBencher,
    AbsorbedBencher,
    Absorbed_CacheCompressedBencher,
    Absorbed_CacheCompressed_MoveElisionBencher,
    AbsorbedMaterialized_CacheCompressed_MoveElisionBencher,
]

BENCHERS = {}

doc = 'Run benchmark on various MLA implementations\n\n'

for bencher in ALL_BENCHMARKS:
    name = bencher.name()
    short_name = bencher.short_name()
    BENCHERS[name] = bencher
    BENCHERS[short_name] = bencher
    doc += f'{short_name}\t{name}\n'

def trace_handler(prof):
    print(f"profile saved to prof_dir/timeline.json")
    prof.export_chrome_trace(f"./prof_dir/baseline.json")
    #prof.export_memory_timeline("memory_timeline.html", device="cuda:0")

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
        # with_stack=False,
        with_stack=True,  # 设置为 True 以记录堆栈信息
        on_trace_ready=trace_handler,
    ) as prof:
        for i in range(15):
            with torch.profiler.record_function("inference"):
                model.iter()
                torch.cuda.synchronize()  # 确保所有 GPU 操作都完成
            prof.step()

def main(bench: str,  kv_len: int, bsz: int = 32, config: str = '/cpfs01/user/xuhaoran/202409mla/cx-Attention/mla/config.json', repeat: Optional[int] = None, 
         min_run_time: float = 1.0, csv: bool = False):
    cfg = DeepseekV2Config.from_json_file(config)
    bencher: BenchmarkFixture
    bencher = BENCHERS[bench](cfg, kv_len, bsz=bsz)
    if repeat is not None:
        for i in range(repeat):
            bencher.iter()
            cache_size = bencher.cache_size()
            memory = bencher.memory_consume()
            
            print(f'repeat:{i} KV Cache: {cache_size}   memory usage:{memory}')
            
        torch.cuda.synchronize()
        return
    
    result = bencher.benchmark(min_run_time=min_run_time)
    cache_size = bencher.cache_size()
    device_name = torch.cuda.get_device_name()
    memory = bencher.memory_consume()
    if csv:
        print(f'{bencher.name()},{bsz},{kv_len},{device_name},{cache_size},{result.mean},{result.median},{result._p25},{result._p75}')
    else:
        print(result)
        print(f'Device: {device_name}')
        print(f'KV Cache: {cache_size}')
        print(f'memory usage:{memory}')

main.__doc__ = doc

if __name__ == "__main__":
    # import fire
    # fire.Fire(main)
    main(bench="CD",kv_len=1024, bsz=32, repeat=128)
    # main(bench="A_CC_ME",kv_len=32000,bsz=256, repeat=1)
    # profile(mod="B",kv_len=2048,bsz=16)