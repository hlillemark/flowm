"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research
template [repo](https://github.com/buoyancy99/research-template).
By its MIT license, you must keep the above sentence in `README.md`
and the `LICENSE` file to credit the author.
"""

from typing import Optional
import torch
from torch.types import _size
import torch.nn as nn

def convert_to_dtype(arg, dtype):
    # if the arg is a iterable, convert all the tensors to the dtype recursively
    if isinstance(arg, (tuple, list)):
        return [convert_to_dtype(item, dtype) for item in arg]
    elif isinstance(arg, Tensor) and arg.dtype != dtype:
        return arg.to(dtype)
    elif isinstance(arg, dict):
        return {key: convert_to_dtype(value, dtype) for key, value in arg.items()}
    return arg


def print_module_grad_report(model, min_norm=0.0):
    """
    Print grad norms per *submodule*, using only that module's own parameters
    (recurse=False) to avoid double-counting. Sorted by total grad norm.
    """
    rows = []
    for name, module in model.named_modules():
        if name == "":  # skip root
            continue
        sq, cnt, max_pname, max_pnorm = 0.0, 0, None, 0.0
        for pname, p in module.named_parameters(recurse=False):
            if p.grad is None:
                continue
            n = p.grad.norm().item()
            sq += n * n
            cnt += 1
            if n > max_pnorm:
                max_pnorm, max_pname = n, pname
        if cnt == 0:
            continue
        total = (sq ** 0.5)
        if total >= min_norm:
            rows.append((name, type(module).__name__, cnt, total, max_pname, max_pnorm))

    rows.sort(key=lambda r: r[3], reverse=True)
    for name, cls, cnt, total, max_pname, max_pnorm in rows:
        print(f"{name:30s} [{cls:20s}]  n_params:{cnt:2d}  grad_norm:{total:.6g}  max_param:{max_pname} ({max_pnorm:.6g})")



def freeze_model(model: nn.Module) -> None:
    """Freeze the torch model"""
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def bernoulli_tensor(
    size: _size,
    p: float,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
):
    """
    Generate a tensor of the given size,
    where each element is sampled from a Bernoulli distribution with probability `p`.
    """
    return torch.bernoulli(torch.full(size, p, device=device), generator=generator)


import re
import math
import torch
from typing import Iterable, Optional, List, Dict, Any

def _safe_stats(t: torch.Tensor):
    """Return (mean, std, min, max, l2norm) on .float() without grad."""
    if t.numel() == 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), float('nan')
    with torch.no_grad():
        x = t.detach().float()
        mean = x.mean().item()
        std  = x.std(unbiased=False).item()
        minv = x.min().item()
        maxv = x.max().item()
        l2   = x.norm(2).item()
        return mean, std, minv, maxv, l2

def _fmt(x: float, width=10, prec=6):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return f"{'nan':>{width}}"
    return f"{x:>{width}.{prec}g}"

def collect_param_stats(
    model: torch.nn.Module,
    name_filter: Optional[str] = None,
    include_buffers: bool = False,
) -> List[Dict[str, Any]]:
    """
    Return a list of dicts, one per parameter (and optional buffers), with stats.
    """
    items = []
    def maybe_add(name: str, tensor: torch.Tensor, is_param: bool):
        if tensor is None:
            return
        if name_filter and not re.search(name_filter, name):
            return
        mean, std, minv, maxv, l2 = _safe_stats(tensor)
        g = tensor.grad if (is_param and isinstance(tensor, torch.nn.Parameter)) else None
        if g is not None:
            g_mean, g_std, g_min, g_max, g_l2 = _safe_stats(g)
        else:
            g_mean = g_std = g_min = g_max = g_l2 = float('nan')
        items.append(dict(
            name=name,
            shape=list(tensor.shape),
            numel=tensor.numel(),
            dtype=str(tensor.dtype).replace('torch.', ''),
            device=str(tensor.device),
            requires_grad=bool(getattr(tensor, 'requires_grad', False)),
            mean=mean, std=std, min=minv, max=maxv, l2=l2,
            g_mean=g_mean, g_std=g_std, g_min=g_min, g_max=g_max, g_l2=g_l2,
            kind='param' if is_param else 'buffer',
        ))

    # parameters
    for n, p in model.named_parameters(recurse=True):
        maybe_add(n, p, is_param=True)

    # (optional) buffers (e.g., running_mean/var in BN)
    if include_buffers:
        for n, b in model.named_buffers(recurse=True):
            maybe_add(n, b, is_param=False)

    return items

def print_param_report(
    model: torch.nn.Module,
    sort_by: str = "g_l2",
    descending: bool = True,
    name_filter: Optional[str] = None,
    include_buffers: bool = False,
    topk: Optional[int] = None,
    max_name_len: int = 48,
):
    """
    Pretty-print per-parameter statistics, optionally sorted.
    sort_by can be one of:
      'name', 'numel', 'l2', 'max', 'mean', 'std', 'g_l2', 'g_max', 'g_mean', 'g_std'
    """
    stats = collect_param_stats(model, name_filter=name_filter, include_buffers=include_buffers)

    if sort_by == "name":
        stats.sort(key=lambda d: d["name"], reverse=descending)
    else:
        key = sort_by if sort_by in stats[0] else "g_l2"
        stats.sort(key=lambda d: (float('-inf') if math.isnan(d[key]) else d[key]), reverse=descending)

    if topk is not None:
        stats = stats[:topk]

    # Header
    name_col = "name(kind)"
    headers = [
        f"{name_col:<{max_name_len}}",
        f"{'shape':>14}",
        f"{'numel':>10}",
        f"{'dtype':>9}",
        f"{'dev':>6}",
        f"{'L2':>10}",
        f"{'mean':>10}",
        f"{'std':>10}",
        f"{'min':>10}",
        f"{'max':>10}",
        f"{'g_L2':>10}",
        f"{'g_mean':>10}",
        f"{'g_std':>10}",
        f"{'g_min':>10}",
        f"{'g_max':>10}",
    ]
    line = " ".join(headers)
    print(line)
    print("-" * len(line))

    for d in stats:
        nm = d["name"]
        if len(nm) > max_name_len - 7:
            nm = nm[:max_name_len - 10] + "..."
        nm = f"{nm}({d['kind'][0]})"  # p/b
        name_cell = f"{nm:<{max_name_len}}"
        row = " ".join([
            name_cell,
            f"{str(tuple(d['shape'])):>14}",
            f"{d['numel']:>10}",
            f"{d['dtype']:>9}",
            f"{('cuda' if 'cuda' in d['device'] else 'cpu'):>6}",
            _fmt(d['l2']),
            _fmt(d['mean']),
            _fmt(d['std']),
            _fmt(d['min']),
            _fmt(d['max']),
            _fmt(d['g_l2']),
            _fmt(d['g_mean']),
            _fmt(d['g_std']),
            _fmt(d['g_min']),
            _fmt(d['g_max']),
        ])
        print(row)

    # Short legend
    print("\nLegend: kind p=param, b=buffer | g_* are gradient stats; L2 is parameter ||·||₂.\n"
          "Tip: use sort_by='l2' or 'g_l2' (default), name_filter='conv', topk=20, include_buffers=True")
