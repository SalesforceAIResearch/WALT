# Research: Reproducing WALT Paper Results

This guide explains how to reproduce the benchmark evaluation results from the WALT paper using WebArena (WA) and VisualWebArena (VWA).

## Prerequisites

1. **Set up benchmark environments:**
   - [WebArena installation guide](https://github.com/web-arena-x/webarena#installation)
   - [VisualWebArena installation guide](https://github.com/web-arena-x/visualwebarena#installation)

2. **Install WALT from source:**
   ```bash
   git clone https://github.com/salesforceairesearch/walt.git
   cd walt
   uv venv && source .venv/bin/activate
   uv pip install -e ".[benchmark]"  # Includes large ML dependencies
   playwright install chromium
   ```

3. **Configure environment:**
   ```bash
   walt init
   # Edit .env to add your API keys and benchmark URLs
   ```

   Your `.env` should include:
   ```bash
   # API Keys
   OPENAI_API_KEY=sk-...
   
   # Benchmark URLs (from your WebArena/VisualWebArena setup)
   CLASSIFIEDS=http://localhost:9980
   CLASSIFIEDS_RESET_TOKEN=...
   SHOPPING=http://localhost:7770
   SHOPPING_ADMIN=http://localhost:7780
   REDDIT=http://localhost:9999
   GITLAB=http://localhost:8023
   MAP=http://localhost:3000
   ```

## Generating tools

Tools discovered by WALT are already included in the `walt-tools` directory. To generate a tool, you can use the `walt generate` command - see [README.md](README.md) for more details.

## Running Evaluations

### Method 1: Using Experiment Configs (Recommended)

**Step 1: Generate test data**

```bash
# VWA test configs
python src/walt/benchmarks/vwa/generate_test_configs.py

# WA test data
python src/walt/benchmarks/wa/generate_test_data.py
```

**Step 2: Run evaluation**

```bash
# Visual WebArena baseline evaluation
python src/walt/benchmarks/vwa/aeval.py --config experiment_configs/vwa_baseline.yaml

# With discovered tools
python src/walt/benchmarks/vwa/aeval.py --config experiment_configs/vwa_with_tools.yaml
```

### Method 2: Using CLI Arguments

```bash
# VWA evaluation
python src/walt/benchmarks/vwa/aeval.py \
  --split classifieds \
  --test_config_base_dir src/walt/benchmarks/vwa/test_configs/visualwebarena \
  --browser_agent_model gpt-5-mini \
  --max_steps 50 \
  --result_dir outputs/vwa-classifieds \
  --headless

# WA evaluation
python src/walt/benchmarks/wa/aeval.py \
  --split gitlab \
  --test_config_base_dir src/walt/benchmarks/wa/test_configs/webarena \
  --browser_agent_model gpt-5-mini \
  --max_steps 50 \
  --result_dir outputs/wa-gitlab
```

## Creating Custom Experiment Configs

Experiment configs provide a clean, reproducible way to define evaluation runs. Create a YAML file in `configs/experiments/`:

```yaml
# configs/experiments/my_experiment.yaml
name: "My Custom VWA Experiment"
description: "Testing with GPT-5 and custom settings"

tags:
  - vwa
  - gpt5
  - custom

llm:
  agent_model: gpt-5-mini
  temperature: 0.1

agent:
  max_steps: 100
  use_stealth: true
  use_vision: true

benchmark:
  type: vwa  # or "wa"
  website: shopping
  task_list: configs/visualwebarena/test_shopping.json
  generate_test_data: true

execution:
  parallel: 4
  save_traces: true

output:
  dir: outputs/my-experiment
  generate_report: true
```

Run it:
```bash
python src/walt/benchmarks/vwa/aeval.py --config configs/experiments/my_experiment.yaml
```

## Tool Discovery for Benchmarks

To discover tools from benchmark websites:

```bash
# Discover tools from VWA classifieds
walt discover classifieds --benchmark vwa --output walt-tools/classifieds/

# Discover tools from WA GitLab
walt discover gitlab --benchmark wa --output walt-tools/gitlab/

# Customize discovery
walt discover shopping --benchmark vwa --max-tools 15 --focus search
```

**Note:** Discovery requires benchmark environments to be running and authenticated.

## Benchmark Utilities

### Reset Environments

```bash
# Reset a specific domain
./src/walt/benchmarks/scripts/reset_benchmark.sh classifieds

# Reset all domains
./src/walt/benchmarks/scripts/reset_benchmark.sh all
```

## Benchmark Structure

### VisualWebArena (`src/walt/benchmarks/vwa/`)

- `aeval.py` - Main evaluation script
- `auto_login.py` - Automatic authentication
- `generate_test_configs.py` - Generate test configs from .env
- `configs/` - Task configurations
- `data/` - Supporting data files

### WebArena (`src/walt/benchmarks/wa/`)

- `aeval.py` - Main evaluation script
- `auto_login.py` - Automatic authentication
- `generate_test_data.py` - Generate test data from .env
- `config_files/` - Task configurations

## Common Issues

### Issue: "DATASET environment variable not set"

**Solution:** Make sure you've run `walt init` and configured your `.env` file with benchmark URLs.

### Issue: Authentication failures

**Solution:** 
1. Verify benchmark services are running
2. Check credentials in `.env`
3. Run auto-login manually:
   ```bash
   python src/walt/benchmarks/vwa/auto_login.py
   python src/walt/benchmarks/wa/auto_login.py
   ```

### Issue: Test configs not found

**Solution:** Run the generation scripts first:
```bash
python src/walt/benchmarks/vwa/generate_test_configs.py
python src/walt/benchmarks/wa/generate_test_data.py
```

## Evaluation Metrics

Evaluations automatically compute:
- **Success Rate**: Percentage of tasks completed successfully
- **Average Steps**: Mean number of steps per task
- **Execution Time**: Total and per-task execution time

Results are saved to the output directory specified in your config or via `--result_dir`.

## Tips for Running Large-Scale Evaluations

1. **Use parallel execution:**
   ```yaml
   execution:
     parallel: 8  # Run 8 tasks in parallel
   ```

2. **Save traces for debugging:**
   ```yaml
   execution:
     save_traces: true
     save_screenshots: false  # Screenshots take a lot of space
   ```

3. **Use headless mode** to avoid GUI overhead (default for most configs)

4. **Monitor resources** - browser instances can be memory-intensive

5. **Reset environments** between major evaluation runs to ensure clean state

## Citation

If you use WALT in your research, please cite:

```bibtex
@article{puri2024walt,
  title={WALT: Towards Automating Web Agents through Actions-as-Tools},
  author={Puri, Viraj and others},
  journal={arXiv preprint},
  year={2024}
}
```

---

For general WALT usage (not benchmarks), see the main [README.md](README.md).

