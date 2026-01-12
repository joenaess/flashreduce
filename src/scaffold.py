from textwrap import indent, dedent, wrap
from dataclasses import dataclass, asdict
from types import SimpleNamespace
from typing import * 
from itertools import product
import toolz as tz
import tempfile
import importlib.util
import pathlib
import os
import sys
import copy
import hashlib

FLASHREDUCE_CACHE = pathlib.Path(os.getenv('FLASHREDUCE_CACHE', 'kernel_cache')).absolute()
FLASHREDUCE_CACHE.mkdir(exist_ok=True, parents=True)
(FLASHREDUCE_CACHE / '__init__.py').touch()
if FLASHREDUCE_CACHE not in sys.path:
    sys.path.append(str(FLASHREDUCE_CACHE))

def mk_module(source_code):
    hash = hashlib.sha256(source_code.encode('utf-8')).hexdigest()
    module = f'generated_module_{hash}'
    target = FLASHREDUCE_CACHE / f'{module}.py'
    
    if target.exists():
        with open(target, 'rt') as f:
            assert f.read() == source_code, "hash equals but code mismatch: {target}"
    else:
        with open(target, 'wt') as f:
            f.write(source_code)
    try:
        return importlib.import_module(module)
        #spec = importlib.util.spec_from_file_location("generated_kernel_mod", target)
        #mod = importlib.util.module_from_spec(spec)
        #spec.loader.exec_module(mod)
        #return mod
    finally:
        print(f"Kernel loaded from: {target}")
  
def count():
    i = 0
    while True:
        yield i
        i += 1

def unsqueeze(dim, rank):
    assert 0 <= dim < rank
    if rank > 1:
        vals = ['None' for _ in range(rank)]
        vals[dim] = ':'
        return '[' + ', '.join(vals) + ']'
    else:
        return ''

def header(str, width=40):
    return f'#{f' {str} ':#^{width-2}}#'

def comment(*, body=None, title=None, width=40):
    lines = [f'#{f' {line} ': <{width-2}}#' for line in wrap(body, width-4)]
    stop = ('#' * width)
    start = header(title, width=width) if title else stop
    return '\n'.join([start, *lines, stop])

def tagged(name, tag):
    if tag is not None:
        return f'{name}_{tag}'
    else:
        return name

def render(*lines):
    builder = []
    for line in lines:
        if isinstance(line, str):
            builder.append(line)
        elif isinstance(line, Iterable):
            stuff = render(*line)
            builder.append(indent(stuff, prefix=' '*4))
    return '\n'.join(builder)

def csv(*vals, trailing_comma=True):
    if trailing_comma:
        return ', '.join(vals) + ','
    else:
        return ', '.join(vals)

def tuplify(*vals):
    return f'({csv(*vals)})'

def deepmap(sequence, *functions):
    for f in functions:
        sequence = tz.mapcat(f, sequence)
    return sequence

class Info(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def tagged(self, **kwargs):
        return copy.replace(self, **kwargs)

    def get(self, key, default=None):
        return getattr(self, key, default)

    @property
    def name(self):
        return tagged(self._name, self.get('var'))

class DimInfo(Info):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.refs = []

    def add_ref(self, name, index):
        self.refs.append([name, index])
    
    @property
    def shape_assign(self):
        shapes = {name: f'{name}.shape[{index}]' for name, index in self.refs}
        initial, *rest = list(shapes.values())
        assign = f'{self.dim} = {initial}'
        if rest:
            assertion = ' == '.join([initial, *rest])
            message = f'f"shape mismatch: {self.name} has inconsistent shapes: {', '.join([f'{name}: {{{shape}}}' for name, shape in shapes.items()])}"'
            return [assign, f'assert {assertion}, {message}']
        else:
            return [assign]

    @property
    def block(self):
        if self.tiled:
            return f'{self.name}_block'
        else:
            raise ValueError(f'Getting block from non-tiled dimension: {self}')

    @property
    def dim(self):
        return f'{self.name}_dim'

    @property
    def pid(self):
        if self.tiled:
            return f'{self.name}_pid'
        else:
            raise ValueError(f'Getting pid from non-tiled dimension: {self}')

    def offsets(self, wrapped=True):
        if self.tiled:
            offs = f'(({self.pid} * {self.block}) + tl.arange(0, {self.block}))'
            if wrapped:
                return f'({offs} % {self.dim})'
            else:
                return offs
        else:
            return f'tl.arange(0, {self.dim})'

    @property
    def bounds(self):
        return f'({self.offsets(wrapped=False)} < {self.dim})'

    @property
    def def_bounds(self):
        return f'{self.bounds} = ({self.offsets(wrapped=False)} < {self.dim})'
    
class NamedInfo(Info):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def rank(self):
        return len(self.shape)
    
    @property
    def ptr(self):
        return f'{self.name}_global_ptr'

    
    @property
    def tile(self):
        return f'{self.name}_tile'

    def stride(self, d):
        assert d in self.shape
        return f'{self.name}_stride_{d}'
    
    @property
    def strides(self):
        if self.get('contiguous'):
            accum = ['1']
            builder = []
            for d in reversed(self.shape):
                builder.append(' * '.join(accum))
                accum.append(self.shape[d].dim)
            return [f'({x})' for x in reversed(builder)]
        else:
            return [f'{self.stride(d)}' for d in self.shape]
    
    @property
    def args(self):
        ret = [self.ptr]
        if self.get('contiguous'):
            return ret
        else:
            return ret + self.strides
    
    @property
    def torch_args(self):
        ret = [self.name]
        if self.get('contiguous'):
            return ret
        else:
            return ret + [f'*{self.name}.stride()']
    
    @property
    def offsets(self):
        builder = []
        for i, stride, d in zip(count(), self.strides, self.shape.values()):
            builder.append(f'({d.offsets()} * {stride}){unsqueeze(i, self.rank)}')
        return ' + '.join(builder)
    
    @property
    def mask(self):
        builder = []
        for i, stride, d in zip(count(), self.strides, self.shape.values()):
            if d.name in 'lr':
                builder.append(f'{d.bounds}{unsqueeze(i, self.rank)}')
        assert len(builder) == 1, 'no two matrices can have both l and r dimension'
        return builder[0]
    
    @property
    def tile_ptrs(self):
        return f'{self.ptr} + {self.offsets}'

    def load(self, src=None):
        if src is None: src = self
        return f'{self.tile} = tl.load({src.tile_ptrs}, mask={src.mask})'

    def store(self, src=None):
        if src is None: src = self
        return f'tl.store({self.tile_ptrs}, {src.tile}, mask={self.mask})'
    
    def add(self, other):
        return f'{self.tile} = {self.tile} + {other.tile}'

    def atomic_add(self, src, sem='relaxed'):
        if src is None: src = self
        return f'tl.atomic_add({self.tile_ptrs}, {src.tile}, mask={self.mask}, sem="{sem}")'
    
    @property
    def shape_tuple(self):
        return tuplify(*[d.dim for d in self.shape.values()])
    
    @property
    def init(self):
        return f'{self.name} = torch.full({self.shape_tuple}, {self.initval}, dtype={self.dtype}, device=device)'

    @classmethod
    def from_info(cls, name: str, info: Info, shapes: Dict[str, DimInfo], **kwargs):
        meta = vars(copy.deepcopy(info))
        meta.pop('shape')
        return cls(_name=name, shape={d: shapes[d] for d in info.shape}, **meta, **kwargs)

def mk_kernel(
        ls: Dict[str, Info], 
        rs: Dict[str, Info], 
        ps: Dict[str, Info], 
        bs: Dict[str, Info] = None,
        code: str = None,
        dim_info: Dict[str, Dict[str, Any]] = {},
        l_blocks = [16, 32],
        r_blocks = [16, 32, 64],
        shards = [1, 4, 16, 64],
        num_stages = [3, 4],
        num_warps = [4, 8],
        ):

    assert code is not None
    
    D = {}
    for bundle in [ls, rs, ps]:
        for info in bundle.values():
            for d in info.shape:
                if d in D:
                    continue
                if d in dim_info:
                    kwargs = dim_info[d]
                else:
                    kwargs = {'tiled': d in 'lr'}
                D[d] = DimInfo(_name=d, **kwargs)

    L = [NamedInfo.from_info(f'l_{l}', info, D) for l, info in ls.items()]
    R = [NamedInfo.from_info(f'r_{r}', info, D) for r, info in rs.items()]
    P = [NamedInfo.from_info(f'p_{p}', info, D, contiguous=True) for p, info in ps.items()]

    for x in L + R:
        for i, d in enumerate(x.shape):
            D[d].add_ref(x.name, i)

    L_GRAD_TMP = [l.tagged(var='grad_tmp', contiguous=True) for l in L]
    R_GRAD_TMP = [r.tagged(var='grad_tmp', contiguous=True) for r in R]
    L_GRAD = [l.tagged(var='grad', contiguous=True, initval='0.0', dtype=l.grad_dtype) for l in L]
    R_GRAD = [r.tagged(var='grad', contiguous=True, initval='0.0', dtype=r.grad_dtype) for r in R]

    P_AGG = [p.tagged(var='agg') for p in P]
    P_TMP = [p.tagged(var='tmp') for p in P]

    if bs is None:
        B = P
    else:
        B = [NamedInfo.from_info(f'b_{b}', info, D) for b, info in bs.items()]
    
    dims = [d.dim for d in D.values()]
    bounds = [d.bounds for d in D.values() if d.tiled]
    blocks = [d.block for d in D.values() if d.tiled]

    def on_bundle(bundle, f):
        return [f(info) for info in bundle]

    def tiles(xs):
        return [x.tile for x in xs]

    def names(xs):
        return [x.name for x in xs]

    def loads(dst, src=None):
        if src is None: src = dst
        return [d.load(s) for d, s in zip(dst, src)]

    def stores(dst, src=None):
        if src is None: src = dst
        return [d.store(s) for d, s in zip(dst, src)]

    def atomic_adds(dst, src=None):
        if src is None: src = dst
        return [d.atomic_add(s) for d, s in zip(dst, src)]

    def adds(dst, src=None):
        if src is None: src = dst
        return [d.add(s) for d, s in zip(dst, src)]


    def inits(xs):
        return [x.init for x in xs]

    def do_binary_reduce(dst, l, r):
        return f'{csv(*tiles(dst))} = binary_reduce({csv(*tiles(l), *tiles(r))})'

    def do_proj_reduce(l, r, dst):
        return f'{csv(*tiles(dst))} = proj_reduce({csv(*tiles(l), *tiles(r), *bounds)})'

    def do_proj_reduce_bwd(l, r, b, l_dst, r_dst):
        return f'{csv(*tiles(l_dst), *tiles(r_dst))} = proj_reduce_bwd({csv(*tiles(l), *tiles(r), *tiles(b), *bounds)})'

    def mk_config(l_block, r_block, shards, num_stages, num_warps):
        return f'triton.Config(kwargs={{ "{D['l'].block}": {l_block}, "{D['r'].block}": {r_block}, "shards": {shards} }}, num_stages={num_stages}, num_warps={num_warps})'

    def mk_configs():
        combinations = product(l_blocks, r_blocks, shards, num_stages, num_warps)
        return 'configs=[', [mk_config(*args) +',' for args in combinations], '],'

    fwd_kernel = render(
        comment(title = 'forward kernel', body = 'generated triton language forward kernel'),
        '@triton.autotune(',
        (
            *mk_configs(),
            f'key=[{csv(*[f'"{d}"' for d in dims])}],',
            'restore_value=["lock_ptr"],',
            ')',
        ),
        "@triton.jit",
        "def fwd_kernel(",
        ((
            *[csv(*x.args) for x in L + R + P],
            'lock_ptr,', 
            *[f'{d.dim},' if d.tiled else f'{d.dim}: tl.constexpr,' for d in D.values()],
            'shards: tl.constexpr,',
            *[f'{b}: tl.constexpr,' for b in blocks],
            '):',
        ),
            comment(title='SETUP', body='set up pids, offsets, et.c.'),
            'l_pid = tl.program_id(axis=0)',
            'r_pid = tl.program_id(axis=1)',
            f'r_group = tl.cdiv({D['r'].dim}, {D['r'].block} * shards)',
            comment(title='LOAD', body='load L and (initial) R tiles, and perform the first proj_reduce'),
            *loads(L + R),
            do_proj_reduce(l=L, r=R, dst=P_AGG),
            comment(title='LOOP', body='loop over R tiles, and aggregate the intermediate P-values'),
            'for k in range(1, r_group):',
            (
                f'r_pid += shards',
                *loads(R),
                do_proj_reduce(l=L, r=R, dst=P_TMP),
                do_binary_reduce(dst=P_AGG, l=P_AGG, r=P_TMP),
            ),
            comment(title='GLOBAL UPDATE', body='Acquire lock (over L-tile), update global value with local values, store to global, then release the lock'),
            'while tl.atomic_cas(lock_ptr + l_pid, 0, 1, sem="acquire") == 1:', ['pass'],
            *loads(dst=P_TMP, src=P),
            do_binary_reduce(dst=P_AGG, l=P_AGG, r=P_TMP),
            *stores(dst=P, src=P_AGG),
            'tl.atomic_xchg(lock_ptr + l_pid, 0, sem="release")'
        ),
    )

    bwd_kernel = render(
        comment(title = 'backward kernel', body = 'generated triton language backward kernel'),
        '@triton.autotune(',
        (
            *mk_configs(),
            f'key=[{csv(*[f'"{d}"' for d in dims])}],',
            f'restore_value=[{csv(*[f'"{name}"' for name in names(L_GRAD + R_GRAD)])}],',
            ')',
        ),
        "@triton.jit",
        "def bwd_kernel(",
        ((
            *[csv(*x.args) for x in L + R + B + L_GRAD + R_GRAD],
            *[f'{d.dim},' if d.tiled else f'{d.dim}: tl.constexpr,' for d in D.values()],
            'shards: tl.constexpr,',
            *[f'{b}: tl.constexpr,' for b in blocks],
            '):',
        ),
            comment(title='SETUP', body='set up pids, offsets, et.c.'),
            'l_pid = tl.program_id(axis=0)',
            'r_pid = tl.program_id(axis=1)',
            f'r_group = tl.cdiv({D['r'].dim}, {D['r'].block} * shards)',
            comment(title='LOAD', body='load L, B, and (initial) R tiles, and perform the first proj_reduce_bwd (and R-grad update)'),
            *loads(L + R + B),
            do_proj_reduce_bwd(l=L, r=R, b=B, l_dst=L_GRAD, r_dst=R_GRAD),
            *atomic_adds(R_GRAD),
            comment(title='LOOP', body='loop over R tiles. Update global R-grad (with atomic add), and aggregate L-grads.'),
            'for k in range(1, r_group):',
            (
                f'r_pid += shards',
                *loads(R),
                do_proj_reduce_bwd(l=L, r=R, b=B, l_dst=L_GRAD_TMP, r_dst=R_GRAD),
                *atomic_adds(R_GRAD),
                *adds(dst=L_GRAD, src=L_GRAD_TMP),
            ),
            comment(title='Update L-grads', body='Update global L-grad (with atomic add).'),
            *atomic_adds(L_GRAD),
        ),
    )

    function = render(
            comment(title = 'torch Function', body = 'generated torch.autograd.Function'),
            'class TritonMonoidReduceFn(torch.autograd.Function):',
            (
                '@staticmethod',
                f'def forward({csv(*names(L), *names(R), trailing_comma=False)}):',
                (
                    f'device = {L[0].name}.device',
                    *[line for d in D.values() for line in d.shape_assign],
                    *inits(P),
                    f'grid = lambda META: (triton.cdiv({D['l'].dim}, META["{D['l'].block}"]), META["shards"])',
                    'lock = torch.zeros(l_dim, device=device, dtype=torch.int32)',
                    'fwd_kernel[grid](',
                    (
                        *[csv(*x.torch_args) for x in [*L, *R, *P]],
                        'lock,',
                        csv(*[d.dim for d in D.values()]),
                    ),
                    ')',
                    f'out = fwd_epilogue({csv(*names(P))})',
                    f'return {csv(*names(P), '*out')}'
                ),
                ''
                '@staticmethod',
                'def setup_context(ctx, inputs, outputs):',
                (
                    f'{csv(*names(P), '*_')} = outputs',
                    *[f'ctx.mark_non_differentiable({name})' for name in names(P)],
                    f'ctx.save_for_backward({csv('*inputs', *names(P))})',
                ),
                ''
                '@staticmethod',
                '@torch.autograd.function.once_differentiable',
                f'def backward({csv('ctx', *[f'_{p.name}' for p in P], '*grads_in')}):',
                (
                    f'device = {L[0].name}.device',
                    f'{csv(*names(L + R + P))} = ctx.saved_tensors',
                    f'{csv(*names(B))} = bwd_prologue({csv(*names(P))} *grads_in)',
                    *[line for d in D.values() for line in d.shape_assign],
                    *inits(L_GRAD + R_GRAD),
                    f'grid = lambda META: (triton.cdiv({D['l'].dim}, META["{D['l'].block}"]), META["shards"])',
                    'bwd_kernel[grid](',
                    (
                        *[csv(*x.torch_args) for x in [*L, *R, *B, *L_GRAD, *R_GRAD]],
                        csv(*dims),
                    ),
                    ')',
                    f'return {csv(*names(L_GRAD + R_GRAD))}',
                ),
            ),

            f'def function({csv(*names(L + R), trailing_comma=False)}):',
            (
                f'{csv(*names(P))} *out = TritonMonoidReduceFn.apply({csv(*names(L + R), trailing_comma=False)})',
                'return out'
            ),
    )


    module = mk_module(render(
            'import torch',
            'import triton.language as tl',
            'import triton',
            '',
            code,
            '',
            fwd_kernel,
            '',
            bwd_kernel,
            '',
            function,
        ))

    return module.function
