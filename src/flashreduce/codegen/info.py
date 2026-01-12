import copy
from types import SimpleNamespace
from typing import * 

from flashreduce.codegen.util import tuplify, csv

def unsqueeze(dim, rank):
    assert 0 <= dim < rank
    if rank > 1:
        vals = ['None' for _ in range(rank)]
        vals[dim] = ':'
        return '[' + ', '.join(vals) + ']'
    else:
        return ''

def count():
    i = 0
    while True:
        yield i
        i += 1

class Info(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def tagged(self, **kwargs):
        return copy.replace(self, **kwargs)

    def get(self, key, default=None):
        return getattr(self, key, default)

    @property
    def name(self):
        var = self.get('var')
        if var is None:
            return self._name
        else:
            return f'{self._name}_{var}'

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
    def is_batched(self):
        return self.get('batched', False)


    @property
    def pid(self):
        if self.tiled | self.is_batched:
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
        elif self.is_batched:
            return f'{self.pid}'
        else:
            return f'tl.arange(0, {self.dim})'

    @property
    def bounds(self):
        return f'({self.offsets(wrapped=False)} < {self.dim})'

    @property
    def def_bounds(self):
        return f'{self.bounds} = ({self.offsets(wrapped=False)} < {self.dim})'
    
class TensorInfo(Info):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def rank(self):
        return len(self.shape)

    @property
    def inner_rank(self):
        return len([x for x in self.shape.values() if not x.is_batched])

    @property
    def ptr(self):
        return f'{self.name}_global_ptr'
    
    @property
    def tile(self):
        return f'{self.name}_tile'

    @property
    def is_contiguous(self):
        return self.get('contiguous', False)

    def dimpos(self, d):
        assert d in self.shape
        for i, n in enumerate(self.shape):
            if n == d:
                return i

    def stride(self, d):
        assert d in self.shape

        if self.is_contiguous:
            i = self.dimpos(d)
            subs = list(self.shape.values())[i+1:]
            if subs:
                return ' * '.join([s.dim for s in subs])
            else:
                return '1'
        else:
            return f'{self.name}_stride_{d}'
    
    @property
    def strides(self):
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
        dix = 0
        rank = self.inner_rank
        for d, info in self.shape.items():
            offsets = f'{info.offsets()} * {self.stride(d)}'
            if not info.is_batched:
                offsets = f'({offsets}){unsqueeze(dix, rank)}'
                dix += 1
            builder.append(offsets)
        return ' + '.join(builder)
    
    @property
    def mask(self):
        builder = []
        dix = 0
        rank = self.inner_rank
        for d, info in self.shape.items():
            if info.name in 'lr':
                builder.append(f'{info.bounds}{unsqueeze(dix, rank)}')
            elif not info.is_batched:
                dix += 1
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
