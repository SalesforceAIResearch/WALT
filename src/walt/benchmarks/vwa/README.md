# Visual WebArena (VWA) Benchmark

This directory contains the Visual WebArena benchmark implementation for WALT.

## 📖 Documentation

**For WALT setup, installation, and usage instructions, see the main project README:**

- [🔬 Research: Reproducing Paper Results](../../../../README.md#reproducing-paper-results)
- [Main README](../../../../README.md)

**For VisualWebArena environment please refer to the original repository:**

- [VisualWebArena Environment Setup](https://github.com/web-arena-x/visualwebarena/blob/main/environment_docker/README.md)
- [VisualWebArena README](https://github.com/web-arena-x/visualwebarena/blob/main/README.md)

## 🏃 Quick Start

```bash
# Install with benchmark dependencies
uv pip install walt[benchmark]
playwright install chromium

# Configure environment
walt init  # Edit .env with VWA URLs and tokens

# Run VWA evaluation
./src/walt/benchmarks/scripts/run_vwa.sh classifieds
```

## 📁 Contents

- `aeval.py` - VWA evaluation script
- `auto_login.py` - Authentication handling
- `test_configs/` - Benchmark configuration files
- `reset.sh` - Benchmark reset logic (configure based on your setup)
- `../../browser_use/custom/evaluators/vwa/` - Task-specific evaluation logic

For detailed information about Visual WebArena, visit: https://jykoh.com/vwa
