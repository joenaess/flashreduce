from textwrap import indent, dedent, wrap
from typing import * 
from itertools import product
import importlib.util
import pathlib
import os
import sys
import hashlib
import inspect
import re


###############################
# Code generation cache stuff #
###############################

def get_cache_dir():
    to_check = [
            'FLASHREDUCE_CACHE',
            'XDG_CACHE_HOME',
            ]

    name = 'flashreduce'
    
    if (dir := os.environ.get('FLASHREDUCE_CACHE')):
        dir = pathlib.Path(dir)
    elif (dir := os.environ.get('XDG_CACHE_HOME')):
        dir = pathlib.Path(dir) / name
    else:
        dir = pathlib.Path.home() / '.cache' / name
    assert dir.is_absolute()
    return dir

CACHE_DIR = get_cache_dir()

def mk_module(source_code: str):
    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    (CACHE_DIR / '__init__.py').touch()
    if (CACHE_DIR_STR := str(CACHE_DIR)) not in sys.path:
        sys.path.append(CACHE_DIR_STR)
    hash = hashlib.sha256(source_code.encode('utf-8')).hexdigest()
    module = f'generated_module_{hash}'
    target = CACHE_DIR / f'{module}.py'
    
    if target.exists():
        with open(target, 'rt') as f:
            assert f.read() == source_code, "hash equals but code mismatch: {target}"
    else:
        with open(target, 'wt') as f:
            f.write(source_code)
    try:
        return importlib.import_module(module)
    finally:
        print(f"Kernel loaded from: {target}")

#####################
# Literal decorator #
#####################

def literal(obj):
    # 1. Unwrap the object to find the original function
    target = obj
    
    # Handle Triton JIT functions specifically
    if hasattr(target, "fn"):
        target = target.fn
    
    # Handle standard Python wrappers (functools.wraps, etc.)
    while hasattr(target, "__wrapped__"):
        target = target.__wrapped__

    # 2. Get the source code of the underlying function
    try:
        source = inspect.getsource(target)
    except (TypeError, OSError) as e:
        return f"# Error retrieving source: {e}"

    # 3. Dedent to normalize indentation
    source = dedent(source)

    # 4. Clean up: Remove the @literal line itself
    lines = source.splitlines(keepends=True)
    cleaned_lines = []
    
    # Regex to match the @literal decorator (and optional parens/arguments)
    # This prevents the returned string from looking like it's recursively decorated
    dec_pattern = re.compile(r'^\s*@.*?literal.*')
    
    header_passed = False
    for line in lines:
        if not header_passed:
            if dec_pattern.match(line):
                continue
            # Once we see a line that isn't @literal (e.g. @triton or def), stop skipping
            header_passed = True
        cleaned_lines.append(line)

    return "".join(cleaned_lines)

##############################
# Helpers for code rendering #
##############################

def header(str, width=40):
    return f'#{f' {str} ':#^{width-2}}#'

def comment(*, body=None, title=None, width=40):
    lines = [f'#{f' {line} ': <{width-2}}#' for line in wrap(body, width-4)]
    stop = ('#' * width)
    start = header(title, width=width) if title else stop
    return '\n'.join([start, *lines, stop])


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

def nlcsv(*vals, trailing_comma=True):
    if trailing_comma:
        return ',\n'.join(vals) + ','
    else:
        return ',\n'.join(vals)

def tuplify(*vals):
    return f'({csv(*vals)})'

###################
# Generic helpers #
###################

def deepmap(sequence, *functions):
    for f in functions:
        sequence = [y for x in sequence for y in f(x)]
    return sequence
