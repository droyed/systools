# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-13

### Added
- `occupy_vram()` — allocate a precise target amount of GPU VRAM
- `release_vram()` — free handles and empty the CUDA cache
- `vram_occupied()` — context manager for automatic VRAM lifecycle
- `demos/demo_occupy_vram.py` — runnable demo of both APIs
- Full README with overview, installation, quick start, API reference, and development guide
- MIT License
- Test suite (`tests/`)
- `Makefile` with `clean` target
- `pyproject.toml` packaging with optional `dev` extras
