# WebArena (WA) Benchmark

This directory contains the WebArena benchmark implementation for WALT.

## 📖 Documentation

**For WALT setup, installation, and usage instructions, see the main project README:**

- [🔬 Research: Reproducing Paper Results](../../../../README.md#reproducing-paper-results)
- [Main README](../../../../README.md)

**For WebArena environment please refer to the original repository:**

- [WebArena Environment Setup](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md)
- [WebArena README](https://github.com/web-arena-x/webarena/blob/main/README.md)

## 🏃 Quick Start

```bash
# Install with benchmark dependencies
uv pip install walt[benchmark]
playwright install chromium

# Configure environment
walt init  # Edit .env with WA URLs and tokens

# Run WA evaluation
./src/walt/benchmarks/scripts/run_wa.sh gitlab
```

## 📁 Contents

- `aeval.py` - WA evaluation script  
- `auto_login.py` - Authentication handling
- `test_configs/` - Benchmark configuration files
- `reset.sh` - Benchmark reset logic (configure based on your setup)
- `../../browser_use/custom/evaluators/wa/` - Task-specific evaluation logic

For detailed information about WebArena, visit: https://webarena.dev
