from flashreduce.codegen import mk_triton_kernel, Info, literal

import torch
import triton
import triton.language as tl

@literal
@triton.jit
def proj_reduce(q, k, v, l_bounds, r_bounds):
    logits = tl.dot(q, k.trans())
    logits = tl.where(l_bounds[:, None] & r_bounds[None, :], logits, float('-inf'))
    hi = tl.max(logits, axis=1)
    valid = hi > float('-inf')
    pz = tl.where(valid, hi + (logits - hi[:, None]).exp().sum(axis=1).log(), hi)
    weights = tl.where(valid[:, None], (logits - pz[:, None]), hi[:, None])
    pv = tl.dot(weights.exp().cast(v.dtype), v)
    return pz, pv

@literal
@triton.jit
def binary_reduce(xz, xv, yz, yv):
    hi = tl.maximum(xz, yz)
    lo = tl.minimum(xz, yz)
    valid = hi > float('-inf')
    pz = tl.where(valid, hi + (1 + (lo - hi).exp()).log(), hi)
    xw = tl.where(valid, (xz - pz).exp(), 0.0)[:, None]
    yw = tl.where(valid, (yz - pz).exp(), 0.0)[:, None]
    pv = xv * xw + yv * yw
    return pz, pv

@literal
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

@literal
def fwd_epilogue(pz, pv):
    return pv,

@literal
def bwd_prologue(pz, pv, gv):
    return pz, (pv * gv).sum(1), gv

code = '\n\n'.join([proj_reduce, proj_reduce_bwd, binary_reduce, fwd_epilogue, bwd_prologue])

_kernel = mk_triton_kernel(
    ls={'q': Info(shape='ghli', grad_dtype=torch.float32)}, 
    rs={'k': Info(shape='hri', grad_dtype=torch.float32), 'v': Info(shape='hro', grad_dtype=torch.float32)}, 
    ps={'z': Info(shape='ghl', dtype=torch.float32, initval="float('-inf')"), 'v': Info(shape='ghlo', dtype=torch.bfloat16, initval="0.0")},
    bs={'z': Info(shape='ghl'), 'dot_vg': Info(shape='ghl'), 'gv': Info(shape='ghlo')},
    code=code,
    dim_info={'h': {'batched': True}, 'g': {'batched': True}}
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
