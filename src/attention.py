from scaffold import mk_kernel, Info

import torch
import triton
import triton.language as tl

code = """\
@triton.jit
def proj_reduce(q, k, v, l_bounds, r_bounds):
    logits = tl.dot(q, k.trans())
    logits = tl.where(l_bounds[:, None] & r_bounds[None, :], logits, float('-inf'))
    hi = tl.max(logits, axis=1)
    valid = hi > float('-inf')
    weights = tl.where(valid[:, None], (logits - hi[:, None]).exp(), 0.0)
    wz = weights.sum(axis=1)
    wv = tl.dot(weights.cast(v.dtype), v)
    pv = wv / tl.where(valid, wz, float('inf'))[:, None]
    pz = hi + wz.log()
    return pz, pv

@triton.jit
def binary_reduce(xz, xv, yz, yv):
    hi = tl.maximum(xz, yz)
    lo = tl.minimum(xz, yz)
    valid = hi > float('-inf')
    pz = hi + (1 + (lo - hi).exp()).log()
    pv = xv * (xz - pz).exp()[:, None] + yv * (yz - pz).exp()[:, None]
    pz = tl.where(valid, pz, hi)
    pv = tl.where(valid[:, None], pv, 0.0)
    return pz, pv

@triton.jit
def proj_reduce_bwd(q, k, v, z, dot_vg, gv, l_bounds, r_bounds):
    logits = tl.dot(q, k.trans())
    ws = (logits - z[:, None]).exp()
    
    grad_v = tl.dot(ws.trans().cast(gv.dtype), gv)

    logits = tl.dot(gv.cast(v.dtype), v.trans()) - dot_vg[:, None] * ws
    logits = logits.cast(k.dtype)

    grad_q = tl.dot(logits, k)
    grad_k = tl.dot(logits.trans(), q)

    return grad_q, grad_k, grad_v

def fwd_epilogue(pz, pv):
    return pv,

def bwd_prologue(pz, pv, gv):
    return pz, (pv * gv).sum(1), gv
"""

_kernel = mk_kernel(
    ls={'q': Info(shape='li', grad_dtype=torch.float32)}, 
    rs={'k': Info(shape='ri', grad_dtype=torch.float32), 'v': Info(shape='ro', grad_dtype=torch.float32)}, 
    ps={'z': Info(shape='l', dtype=torch.float32, initval="float('-inf')"), 'v': Info(shape='lo', dtype=torch.bfloat16, initval="0.0")},
    bs={'z': Info(shape='l'), 'dot_vg': Info(shape='l'), 'gv': Info(shape='lo')},
    code=code,
    shards=[32],
)

#@torch.compile(fullgraph=True)
def flashreduce_attention(q, k, v):
    y, = _kernel(q, k, v)
    return y

def naive_attention(q, k, v):
    return (q @ k.t()).softmax(dim=1) @ v

def hiprec_attention(q, k, v):
    return naive_attention(q.to(torch.float32), k.to(torch.float32), v.to(torch.float32))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(9, 15, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='TFLOP/s',  # Label name for the y-axis.
        plot_name='attention',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))

def benchmark(N, provider):
    D = 128
    DEVICE = 'cuda'
    DTYPE = torch.bfloat16
    q = torch.rand(N, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    k = torch.rand(N, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    v = torch.randn(N, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    mock = torch.randn(N, D, device=DEVICE, dtype=DTYPE)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: (naive_attention(q, k, v) * mock).sum().backward(), quantiles=quantiles)
        #ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_attention(q, k, v), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: (flashreduce_attention(q, k, v) * mock).sum().backward(), quantiles=quantiles)
        #ms, min_ms, max_ms = triton.testing.do_bench(lambda: flashreduce_attention(q, k, v), quantiles=quantiles)
    fps = lambda ms: N*N*D*4 * 1e-12 / (ms * 1e-3)
    return fps(ms), fps(max_ms), fps(min_ms)

if __name__ == '__main__':
    torch.random.manual_seed(0x5eed)
    N = 1024
    D = 128
    DEVICE = 'cuda'
    DTYPE = torch.bfloat16
    q = torch.rand(N, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    k = torch.rand(N, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    v = torch.randn(N, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    mock = torch.randn(N, D, device=DEVICE, dtype=DTYPE)
    y1 = flashreduce_attention(q, k, v)
    y2 = naive_attention(q, k, v)
    y3 = hiprec_attention(q, k, v)
    #(flashreduce_attention(q, k, v) * mock).sum().backward()
    print(y1)
    print(y2)
    print(y3)
    #benchmark.run(print_data=True)
