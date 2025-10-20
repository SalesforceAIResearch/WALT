# WALT: Web Agents that Learn Tools

> **W**eb **A**gents that **L**earn **T**ools - Automatic tool discovery from websites for LLM agents

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://www.arxiv.org/abs/2510.01524)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**WALT** enables LLM agents to automatically discover and learn reusable tools from any website. Point WALT at a website, and it will explore, understand, and generate ready-to-use tool definitions.

<p align="center">
  <img src="walt-overview.png" alt="WALT Overview">
</p>

---

## 🚀 Quick Start

### Installation

```bash
# Install uv (faster than pip)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install WALT (ideally inside a venv)
uv venv && source .venv/bin/activate
uv pip install walt
playwright install chromium

# Set up configuration
walt init  # Creates .env file for API keys
```

### Basic Usage

```bash
# Run agent with tools
walt agent "find and return the URL of the cheapest blue kayak" \
  --tools walt-tools/classifieds/ \
  --start-url http://localhost:9980

# Discover new tools from any website
walt discover --url https://example.com

# Or generate a specific tool (faster!)
walt generate --url https://zillow.com --goal "Search for homes with filters"

# List available tools
walt list walt-tools/shopping/

# Start an MCP server
walt serve walt-tools/classifieds/ --port 8000

# Record a demonstration
walt record https://example.com --name my_tool
```

---

## 🐍 Python SDK

WALT can be used programmatically for tool discovery and agent execution:

```python
# Tool discovery
from walt.tools.discovery import propose, generate
import asyncio

async def discover_tools():
    class Args:
        base_url = "https://example.com"
        output_dir = "my-tools"
        llm = "gpt-5-mini"
        planner_llm = "gpt-5-mini"
        auth_file = None  # Optional: path to Playwright storage_state.json
        max_processes = 16
        
    args = Args()
    
    # Phase 1: Discover candidates
    tools = await propose.discover_candidates(args)
    
    # Phase 2: Generate tools
    await generate.generate_tools(args, tools)

asyncio.run(discover_tools())
```

```python
# Agent with tools
from walt.browser_use.custom.agent_zoo import VWA_Agent
from walt.browser_use.custom.browser import VWABrowser, BrowserConfig
from walt.browser_use import Controller
from walt.tools.discovery.register import register_tools_from_directory
from langchain_openai import ChatOpenAI

async def run_agent():
    # Setup browser and controller
    browser = VWABrowser(BrowserConfig(headless=False))
    controller = Controller()
    
    # Load tools
    register_tools_from_directory(
        controller=controller,
        tool_dir="walt-tools/classifieds/",
        llm=ChatOpenAI(model="gpt-5-mini")
    )
    
    # Create and run agent
    agent = VWA_Agent(
        task="Find the cheapest blue kayak",
        llm=ChatOpenAI(model="gpt-5-mini"),
        browser=browser,
        controller=controller,
        max_actions_per_step=30
    )
    
    await agent.run()
    await browser.close()

asyncio.run(run_agent())
```

---

## 📖 CLI Commands

### `walt agent <task>`
Run an agent to complete a task, optionally using tools.

```bash
walt agent "find cheap apartments" --tools walt-tools/classifieds/ --start-url https://www.zillow.com
walt agent "book a flight to NYC" --llm gemini-2.5-flash --max-steps 100 --start-url https://www.google.com/flights
walt agent "search for blue kayaks" --save-gif kayak_search.gif  # Record as GIF
```

**Key options:** `--tools`, `--llm`, `--headless`, `--max-steps`, `--start-url`, `--save-gif`

**Recording:** Use `--save-gif <path>` to save the agent's browser interactions as an animated GIF with step-by-step actions overlaid.

### `walt discover --url <url>`
Discover and generate tools by exploring a website.

```bash
walt discover --url https://example.com
walt discover --url http://localhost:9980 --output walt-tools/mysite
walt discover --url https://example.com --auth-file .auth/state.json
walt discover --url https://example.com --llm gpt-4o --max-processes 8
```

**Key options:** `--url`, `--output`, `--llm`, `--auth-file`, `--max-processes`, `--force-regenerate`

**Note:** To reproduce results on research benchmarks, see [BENCHMARKS.md](BENCHMARKS.md).

### `walt generate --url <url> --goal <goal>`
Generate a specific tool without exploration (when you know what you want).

```bash
walt generate --url https://airbnb.com --goal "Search for homes available in a location for provided dates and guest details"
walt generate --url https://zillow.com --goal "View property details" -o walt-tools/zillow/
walt generate --url https://example.com --goal "Book appointment" --auth-file .auth/state.json
```

**Key options:** `--url`, `--goal`, `--output`, `--llm`, `--auth-file`

**Use case:** When you already know what tool you need and don't want to wait for exploratory discovery.

### `walt record <url>`
Record a human demonstration and convert it to a tool.

```bash
walt record https://example.com --name search_products
```

### `walt serve <tool_dir>`
Start an MCP server with your tools.

```bash
walt serve walt-tools/shopping/ --port 8000
```

### `walt list [tool_dir]`
List discovered tools.

```bash
walt list                           # All tools
walt list walt-tools/classifieds/   # Specific directory
walt list --detailed                # Detailed table view
```

The [examples/](examples/) directory contains detailed examples of how to use WALT, including:
- [01_simple_discovery.py](examples/01_simple_discovery.py) - Simple tool discovery
- [02_agent_with_tools.py](examples/02_agent_with_tools.py) - Using an agent with discovered tools
- [03_advanced_tool_use.py](examples/03_advanced_tool_use.py) - Advanced tool usage patterns


---

## 📦 Tool Format

WALT tools are JSON files with a simple structure:

```json
{
  "name": "search_products",
  "description": "Search for products on the site",
  "inputs": {
    "query": {
      "type": "string",
      "description": "Search query",
      "required": true
    }
  },
  "steps": [
    {
      "type": "navigation",
      "url": "https://example.com"
    },
    {
      "type": "input",
      "cssSelector": "#search-box",
      "text": "{query}"
    },
    {
      "type": "click",
      "cssSelector": "#search-button"
    },
    {
      "type": "extract_page_content",
      "goal": "Extract search results"
    }
  ]
}
```

**Step types:**
- **Deterministic:** `navigation`, `click`, `input`, `select_change`, `key_press`, `scroll`
- **Agentic:** `extract_page_content`, `wait_for_page_load`

See [`walt-tools/`](walt-tools/) for 50 pre-discovered examples.

---

## 🛠️ Development

### Install from Source

```bash
git clone https://github.com/salesforceairesearch/walt.git
cd walt
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
playwright install chromium
```



### Project Structure

```
walt/
├── src/walt/
│   ├── browser_use/         # Browser automation
│   ├── tools/               # Tool system (discovery, execution, demonstration)
│   ├── benchmarks/          # WebArena/VisualWebArena evaluation
│   ├── cli.py               # CLI entry point
│   └── config.py            # Configuration system
├── experiment_configs/
│   └── ...                  # Experiment & benchmark configs
├── walt-tools/              # Pre-discovered tools
└── examples/                # Example scripts
```

### Configuration

Use experiment configs to define reproducible evaluation runs:

```yaml
# experiment_configs/my_experiment.yaml
name: "My Experiment"
llm:
  agent_model: gpt-5
agent:
  max_steps: 100
output:
  dir: outputs/my-experiment
```

Run it: `python src/walt/benchmarks/vwa/aeval.py --config experiment_configs/my_experiment.yaml`


### Reproducing Paper Results

Interested in reproducing results from our [paper](https://arxiv.org/abs/2510.01524)? See [BENCHMARKS.md](BENCHMARKS.md) for:
- WebArena and VisualWebArena setup
- Running evaluations with experiment configs
- Tool discovery for benchmarks
- Detailed configuration options

---

## 🤝 Citation

If you use WALT in your research, please cite:

```bibtex
@article{walt2025,
  title={WALT: Web Agents that Learn Tools},
  author={Viraj Prabhu, Yutong Dai, Matthew Fernandez, Jing Gu, Krithika Ramakrishnan, Yanqi Luo, Silvio Savarese, Caiming Xiong, Junnan Li, Zeyuan Chen, Ran Xu},
  journal={arXiv preprint arXiv:2510.01524},
  year={2025}
}
```

---

## 📄 License

MIT - See [LICENSE](LICENSE)

## 🙏 Acknowledgments

We are grateful to the browser-use team for the following projects upon which WALT is built:
- **[browser-use](https://github.com/browser-use/browser-use)**
- **[workflow-use](https://github.com/browser-use/workflow-use)**

We are also grateful to the WebArena and VisualWebArena teams for the benchmark datasets.