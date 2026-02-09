#!/usr/bin/env python3
import argparse
import math
import torch


def ref_flash_attn(q, k, v, causal=True):
    # q/k/v: [B,H,S,D]
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if causal:
        s = q.size(-2)
        mask = torch.triu(torch.ones(s, s, device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b', type=int, default=1)
    ap.add_argument('--h', type=int, default=2)
    ap.add_argument('--s', type=int, default=64)
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float16', 'bfloat16'])
    ap.add_argument('--causal', action='store_true')
    args = ap.parse_args()

    dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}[args.dtype]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(0)
    q = torch.randn(args.b, args.h, args.s, args.d, device=device, dtype=dtype)
    k = torch.randn(args.b, args.h, args.s, args.d, device=device, dtype=dtype)
    v = torch.randn(args.b, args.h, args.s, args.d, device=device, dtype=dtype)

    out = ref_flash_attn(q, k, v, causal=args.causal)
    print(f"ref_out shape={tuple(out.shape)} dtype={out.dtype} device={out.device}")
    print(f"samples: {out.flatten()[0].item():.6f}, {out.flatten()[out.numel()//2].item():.6f}, {out.flatten()[-1].item():.6f}")


if __name__ == '__main__':
    main()
