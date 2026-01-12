from flashreduce.codegen.util import *
from flashreduce.codegen.info import *
from typing import * 
from itertools import product


def mk_triton_kernel(
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
                kwargs = dim_info.get(d, {})
                D[d] = DimInfo(_name=d, tiled=d in 'lr', **kwargs)

    L = [TensorInfo.from_info(f'l_{l}', info, D) for l, info in ls.items()]
    R = [TensorInfo.from_info(f'r_{r}', info, D) for r, info in rs.items()]
    P = [TensorInfo.from_info(f'p_{p}', info, D, contiguous=True) for p, info in ps.items()]

    for x in L + R:
        for i, d in enumerate(x.shape):
            D[d].add_ref(x.name, i)

    L_GRAD = [l.tagged(var='grad', contiguous=True, initval='0.0', dtype=l.grad_dtype) for l in L]
    R_GRAD = [r.tagged(var='grad', contiguous=True, initval='0.0', dtype=r.grad_dtype) for r in R]
    L_GRAD_TMP = [l.tagged(var='grad_tmp') for l in L]
    R_GRAD_TMP = [r.tagged(var='grad_tmp') for r in R]

    P_AGG = [p.tagged(var='agg') for p in P]
    P_TMP = [p.tagged(var='tmp') for p in P]

    if bs is None:
        B = P
    else:
        B = [TensorInfo.from_info(f'b_{b}', info, D) for b, info in bs.items()]
    
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

    batched = {k: v for k, v in D.items() if v.is_batched}

    if batched:
        print(batched)
        total = ' * '.join([b.dim for b in batched.values()])
        print(f'batch_pids = {total}')
        print('batch_pid = tl.program_id(axis=2)')
        ordered = list(batched.values())
        for i, d in enumerate(ordered):
            print(f'{d.pid} = ???')

        print(csv(*[b.dim for b in batched.values()]))


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
            *[f'{d.dim},' if d.tiled or d.is_batched else f'{d.dim}: tl.constexpr,' for d in D.values()],
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
            *[f'{d.dim},' if d.tiled or d.is_batched else f'{d.dim}: tl.constexpr,' for d in D.values()],
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

    module_code = render(
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
    )

    print(module_code)

    module = mk_module(module_code)

    return module.function
