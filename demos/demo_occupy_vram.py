"""
demo_occupy_vram.py
-------------------
Demonstrates the functional and context-manager APIs of systools.occupy_vram.

Usage:
    python demos/demo_occupy_vram.py [target_gb]   # default: 1.0 GB
"""

import sys
import time

from systools.occupy_vram import occupy_vram, release_vram, vram_occupied

target_gb = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

print(f"\n{'='*55}")
print(f"  Demo: occupying {target_gb} GB on GPU")
print(f"{'='*55}\n")

# ── functional API ──────────────────────────────────────────────────
handles = occupy_vram(target_gb=target_gb, verbose=True)
print("\nHolding VRAM for 3 seconds …")
time.sleep(3)
release_vram(handles)

print()

# ── context-manager API ─────────────────────────────────────────────
print("Context-manager demo (512 MB):")
with vram_occupied(target_mb=512, verbose=True):
    print("  Inside context – VRAM occupied.")
print("  Outside context – VRAM released.")
