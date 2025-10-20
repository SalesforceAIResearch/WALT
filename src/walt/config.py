"""
Experiment configuration system for WALT.

This module provides a config-driven approach to running experiments,
replacing scattered shell scripts with declarative YAML configs.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml


@dataclass
class LLMConfig:
    """LLM configuration."""
    agent_model: str = "gpt-5-mini"
    planner_model: Optional[str] = None  # Defaults to agent_model if not set
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    
    def __post_init__(self):
        if self.planner_model is None:
            self.planner_model = self.agent_model


@dataclass
class AgentConfig:
    """Agent behavior configuration."""
    max_steps: int = 50
    use_stealth: bool = True
    system_prompt: Optional[str] = None  # Path to custom system prompt
    use_vision: bool = True
    max_actions_per_step: int = 10


@dataclass
class BenchmarkConfig:
    """Benchmark-specific configuration."""
    type: str  # "vwa" or "wa"
    website: Optional[str] = None  # For discovery tasks
    task_list: Optional[str] = None  # Path to task list file
    reset_environment: bool = False
    authenticate: bool = True
    generate_test_data: bool = True


@dataclass
class ExecutionConfig:
    """Execution parameters."""
    parallel: int = 1
    timeout: Optional[int] = None  # Per-task timeout in seconds
    max_retries: int = 0
    save_traces: bool = True
    save_screenshots: bool = False


@dataclass
class OutputConfig:
    """Output configuration."""
    dir: str = "results"
    log_level: str = "INFO"
    generate_report: bool = True
    save_raw_results: bool = True


@dataclass
class DiscoveryConfig:
    """Tool discovery configuration."""
    max_tools: int = 10
    focus: Optional[str] = None  # e.g., "search", "content", etc.
    skip_reset: bool = False
    force_regenerate: bool = False


@dataclass
class ExperimentConfig:
    """
    Main experiment configuration.
    
    This class represents a complete experiment configuration, loaded from YAML.
    It's designed to replace shell scripts with long argument lists.
    
    Example:
        config = ExperimentConfig.load("configs/experiments/vwa_baseline.yaml")
        # Use config values to run experiment
    """
    name: str
    description: str = ""
    
    # Core configs
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    benchmark: Optional[BenchmarkConfig] = None
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    discovery: Optional[DiscoveryConfig] = None
    
    # Additional metadata
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    
    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """
        Load experiment config from YAML file.
        
        Args:
            path: Path to YAML config file (relative to repo root or absolute)
            
        Returns:
            ExperimentConfig instance
            
        Example:
            config = ExperimentConfig.load("configs/experiments/vwa_baseline.yaml")
        """
        # Handle relative paths from repo root
        if not os.path.isabs(path):
            # Assume path is relative to repo root
            repo_root = Path(__file__).parent.parent.parent
            path = str(repo_root / path)
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Convert dictionary to ExperimentConfig."""
        # Extract nested configs
        llm_data = data.get('llm', {})
        agent_data = data.get('agent', {})
        benchmark_data = data.get('benchmark')
        execution_data = data.get('execution', {})
        output_data = data.get('output', {})
        discovery_data = data.get('discovery')
        
        return cls(
            name=data['name'],
            description=data.get('description', ''),
            llm=LLMConfig(**llm_data),
            agent=AgentConfig(**agent_data),
            benchmark=BenchmarkConfig(**benchmark_data) if benchmark_data else None,
            execution=ExecutionConfig(**execution_data),
            output=OutputConfig(**output_data),
            discovery=DiscoveryConfig(**discovery_data) if discovery_data else None,
            tags=data.get('tags', []),
            notes=data.get('notes'),
        )
    
    def save(self, path: str) -> None:
        """
        Save config to YAML file.
        
        Args:
            path: Output path for YAML file
        """
        data = self._to_dict()
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        data = {
            'name': self.name,
            'description': self.description,
            'llm': {
                'agent_model': self.llm.agent_model,
                'planner_model': self.llm.planner_model,
                'temperature': self.llm.temperature,
            },
            'agent': {
                'max_steps': self.agent.max_steps,
                'use_stealth': self.agent.use_stealth,
                'use_vision': self.agent.use_vision,
                'max_actions_per_step': self.agent.max_actions_per_step,
            },
            'execution': {
                'parallel': self.execution.parallel,
                'save_traces': self.execution.save_traces,
                'save_screenshots': self.execution.save_screenshots,
            },
            'output': {
                'dir': self.output.dir,
                'log_level': self.output.log_level,
                'generate_report': self.output.generate_report,
            },
        }
        
        # Add optional configs
        if self.agent.system_prompt:
            data['agent']['system_prompt'] = self.agent.system_prompt
        
        if self.llm.max_tokens:
            data['llm']['max_tokens'] = self.llm.max_tokens
        
        if self.benchmark:
            data['benchmark'] = {
                'type': self.benchmark.type,
                'reset_environment': self.benchmark.reset_environment,
                'authenticate': self.benchmark.authenticate,
                'generate_test_data': self.benchmark.generate_test_data,
            }
            if self.benchmark.website:
                data['benchmark']['website'] = self.benchmark.website
            if self.benchmark.task_list:
                data['benchmark']['task_list'] = self.benchmark.task_list
        
        if self.discovery:
            data['discovery'] = {
                'max_tools': self.discovery.max_tools,
                'skip_reset': self.discovery.skip_reset,
                'force_regenerate': self.discovery.force_regenerate,
            }
            if self.discovery.focus:
                data['discovery']['focus'] = self.discovery.focus
        
        if self.execution.timeout:
            data['execution']['timeout'] = self.execution.timeout
        
        if self.tags:
            data['tags'] = self.tags
        
        if self.notes:
            data['notes'] = self.notes
        
        return data


def get_prompt_path(prompt_name: str) -> Path:
    """
    Get the full path to a prompt file in prompts/.
    
    Args:
        prompt_name: Name of the prompt file (e.g., "runtime/agent.md")
        
    Returns:
        Path to the prompt file
        
    Example:
        path = get_prompt_path("runtime/agent.md")
        content = path.read_text()
    """
    # Path(__file__) = walt/src/walt/config.py
    # We want walt/src/walt/prompts/
    walt_package_root = Path(__file__).parent
    return walt_package_root / "prompts" / prompt_name


def load_prompt(prompt_name: str) -> str:
    """
    Load a prompt from prompts/ directory (legacy function - prefer importing from walt.prompts).
    
    Args:
        prompt_name: Name of the prompt file (e.g., "runtime/agent.md")
        
    Returns:
        Prompt content as string
        
    Example:
        prompt = load_prompt("agent_system.md")
    """
    return get_prompt_path(prompt_name).read_text()

