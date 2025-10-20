"""
WALT CLI

Main command-line interface for WALT (Web Agents that Learn Tools).
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="walt",
    help="🪄 WALT: Web Agents that Learn Tools - Automatic tool discovery from websites",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()


@app.command()
def init():
    """Initialize WALT configuration with .env file."""
    console.print("[bold cyan]🚀 Initializing WALT configuration[/bold cyan]")

    env_content = """# WALT Configuration

# ==============================================================================
# LLM API Keys (OpenAI by default,configure according to your LLM provider)
# ==============================================================================

OPENAI_API_KEY=your-openai-key-here
# ANTHROPIC_API_KEY=your-anthropic-key-here
# GOOGLE_API_KEY=your-google-key-here

# ==============================================================================
# Benchmark URLs (For research reproduction only)
# ==============================================================================

# Uncomment and configure based on your benchmark setup:

# VisualWebArena URLs
# DATASET=visualwebarena
# CLASSIFIEDS=http://localhost:9980
# CLASSIFIEDS_RESET_TOKEN=4b61655535e7ed388f0d40a93600254c
# SHOPPING=http://localhost:7770  
# REDDIT=http://localhost:9999
# WIKIPEDIA=http://localhost:8888
# HOMEPAGE=http://localhost:4399

# WebArena URLs  
# GITLAB=http://localhost:8023
# MAP=http://localhost:3000
# SHOPPING_ADMIN=http://localhost:7780/admin

# ==============================================================================
# Logging & Telemetry
# ==============================================================================

ANONYMIZED_TELEMETRY=false
BROWSER_USE_LOGGING_LEVEL=info

# ==============================================================================
# Advanced Settings
# ==============================================================================

# Browser settings
# HEADLESS=true

# Performance
# MAX_STEPS=30
# MAX_PROCESSES=16
"""

    if Path(".env").exists():
        console.print("[yellow]⚠️  .env file already exists[/yellow]")
        overwrite = typer.confirm("Overwrite existing .env file?")
        if not overwrite:
            console.print("[dim]Cancelled[/dim]")
            return

    with open(".env", "w") as f:
        f.write(env_content)

    console.print("[green]✅ Created .env file[/green]")
    console.print("[dim]Please edit .env and add your OPENAI_API_KEY[/dim]")


@app.command()
def version():
    """Show WALT version."""
    from walt import __version__

    console.print(f"[bold cyan]WALT[/bold cyan] version {__version__}")


@app.command()
def discover(
    url: str = typer.Option(
        ..., "--url", help="Base URL to discover tools from (e.g., https://example.com)"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory for discovered tools"
    ),
    llm: str = typer.Option("gpt-5-mini", "--llm", help="LLM model to use"),
    planner_llm: Optional[str] = typer.Option(
        None, "--planner-llm", help="Planner LLM model (defaults to same as --llm)"
    ),
    auth_file: Optional[str] = typer.Option(
        None, "--auth-file", help="Playwright storage_state JSON file for authentication"
    ),
    max_processes: int = typer.Option(16, "--max-processes", "-p", help="Max concurrent processes"),
    force_regenerate: bool = typer.Option(
        False, "--force-regenerate", help="Force regeneration of existing tools"
    ),
    skip_test: bool = typer.Option(False, "--skip-test", help="Skip testing generated tools"),
    optimize: bool = typer.Option(False, "--optimize", help="Generate optimized versions of tools"),
):
    """
    Discover and generate tools from any website.

    Examples:
        walt discover --url https://example.com
        walt discover --url http://localhost:9980 --output walt-tools/mysite
        walt discover --url https://example.com --auth-file .auth/state.json
        walt discover --url https://example.com --llm gpt-4o --max-processes 8

    The command automatically:
    1. Explores the website to discover possible tools
    2. Generates tool definitions with parameters
    3. Tests each tool to verify it works
    4. Saves tools to the output directory
    """
    console.print(f"[bold cyan]🔍 Discovering tools from:[/bold cyan] {url}")

    # Build args for generic discovery system
    from types import SimpleNamespace
    
    args = SimpleNamespace(
        url=url,
        base_url=url,
        llm=llm,
        planner_llm=planner_llm or llm,
        auth_file=auth_file,
        max_processes=max_processes,
        force_regenerate=force_regenerate,
        test=not skip_test,
        optimize=optimize,
        discover=True,
        generate=True
    )

    # Derive output directory from URL if not specified
    if not output_dir:
        domain = url.replace("https://", "").replace("http://", "").split("/")[0]
        args.output_dir = f"walt-tools/{domain}"
    else:
        args.output_dir = output_dir

    if auth_file:
        console.print(f"[dim]🔑 Using authentication: {auth_file}[/dim]")

    # Run discovery
    try:
        asyncio.run(discovery_main_async(args))
        console.print(f"\n[bold green]✅ Discovery complete![/bold green]")
        console.print(f"[dim]Tools saved to: {args.output_dir}[/dim]")
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Discovery interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


async def discovery_main_async(args):
    """Run the generic discovery pipeline."""
    from walt.tools.discovery import propose, generate
    import os

    os.makedirs(args.output_dir, exist_ok=True)

    console.print(f"[dim]📁 Output directory: {args.output_dir}[/dim]")

    # Phase 1: Discovery
    console.print("\n[bold cyan]🔍 Phase 1: Discovering candidate tools...[/bold cyan]")
    tools_json = await propose.discover_candidates(args)
    console.print(f"[green]✅ Found {len(tools_json)} candidate tools[/green]")

    # Phase 2: Generation
    console.print("\n[bold cyan]🚀 Phase 2: Generating tools...[/bold cyan]")
    tools_json = propose.load_existing_candidates(args)
    if not tools_json:
        console.print("[yellow]⚠️  No candidates found[/yellow]")
        return

    success_count = await generate.generate_tools(args, tools_json)
    console.print(
        f"[green]✅ Generated {success_count}/{len(tools_json)} tools successfully[/green]"
    )


async def generate_main_async(args, goals: list[str]):
    """Run targeted tool generation without exploration."""
    from walt.tools.discovery import generate
    import os
    import json
    import re

    os.makedirs(args.output_dir, exist_ok=True)

    console.print(f"[dim]📁 Output directory: {args.output_dir}[/dim]")

    # Create candidate tools from goals
    tools_json = []
    for goal in goals:
        # Sanitize goal to create a tool name
        tool_name = re.sub(r"[^a-z0-9_]+", "_", goal.lower())[:50].strip("_")

        tools_json.append(
            {
                "name": tool_name,
                "description": goal,
                "start_url": args.base_url,
                "elements": [],  # No specific elements - agent will figure it out
            }
        )

    # Save candidates file (for consistency with discover)
    exploration_file = os.path.join(args.output_dir, "exploration_result.json")
    with open(exploration_file, "w") as f:
        json.dump({"tools": tools_json}, f, indent=4)

    console.print(f"[green]✅ Created {len(tools_json)} tool candidate(s) from goals[/green]")

    # Phase 2: Generation
    console.print("\n[bold cyan]🚀 Generating tools...[/bold cyan]")
    success_count = await generate.generate_tools(args, tools_json)
    console.print(
        f"[green]✅ Generated {success_count}/{len(tools_json)} tools successfully[/green]"
    )


@app.command()
def generate(
    url: str = typer.Option(
        ..., "--url", help="Base URL for the tools (e.g., https://example.com)"
    ),
    goal: str = typer.Option(
        ..., "--goal", help="Tool goal/description (e.g., 'Search for homes with filters')"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory for generated tools"
    ),
    llm: str = typer.Option("gpt-5-mini", "--llm", help="LLM model to use"),
    planner_llm: Optional[str] = typer.Option(
        None, "--planner-llm", help="Planner LLM model (defaults to same as --llm)"
    ),
    auth_file: Optional[str] = typer.Option(
        None, "--auth-file", help="Playwright storage_state JSON file for authentication"
    ),
    max_processes: int = typer.Option(16, "--max-processes", "-p", help="Max concurrent processes"),
    force_regenerate: bool = typer.Option(
        False, "--force-regenerate", help="Force regeneration of existing tools"
    ),
    skip_test: bool = typer.Option(False, "--skip-test", help="Skip testing generated tools"),
):
    """
    Generate a specific tool from a website without exploration.

    Use this when you already know what tool you want to create.
    For exploratory discovery, use 'walt discover' instead.

    Examples:
        walt generate --url https://zillow.com --goal "Search for homes with price filters"
        walt generate --url https://zillow.com --goal "View property details" -o walt-tools/zillow/
        walt generate --url https://example.com --goal "Book appointment" --auth-file .auth/state.json
    """
    goals = [goal]
    
    console.print(f"[bold cyan]🎯 Generating tool from:[/bold cyan] {url}")
    console.print(f"[dim]Goal: {goal}[/dim]")

    # Build args for generation
    from types import SimpleNamespace
    
    args = SimpleNamespace(
        url=url,
        base_url=url,
        llm=llm,
        planner_llm=planner_llm or llm,
        auth_file=auth_file,
        max_processes=max_processes,
        force_regenerate=force_regenerate,
        test=not skip_test,
        optimize=False  # Skip optimization for targeted generation
    )

    # Derive output directory from URL if not specified
    if not output_dir:
        domain = url.replace("https://", "").replace("http://", "").split("/")[0]
        args.output_dir = f"walt-tools/{domain}"
    else:
        args.output_dir = output_dir

    if auth_file:
        console.print(f"[dim]🔑 Using authentication: {auth_file}[/dim]")

    # Run targeted generation
    try:
        asyncio.run(generate_main_async(args, goals))
        console.print(f"\n[bold green]✅ Generation complete![/bold green]")
        console.print(f"[dim]Tools saved to: {args.output_dir}[/dim]")
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Generation interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        raise typer.Exit(1)


@app.command()
def serve(
    tool_dir: str = typer.Argument(..., help="Directory containing tool JSON files"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to run MCP server on"),
):
    """
    Start an MCP server with discovered tools.

    Examples:
        walt serve walt-tools/classifieds/
        walt serve my-tools/ --port 9000
    """
    console.print(f"[bold cyan]🚀 Starting MCP server with tools from:[/bold cyan] {tool_dir}")

    # Check if directory exists
    tool_path = Path(tool_dir)
    if not tool_path.exists():
        console.print(f"[red]❌ Directory not found:[/red] {tool_dir}")
        raise typer.Exit(1)

    # Count tool files
    tool_files = list(tool_path.glob("*.tool.json"))
    if not tool_files:
        console.print(f"[yellow]⚠️  No .tool.json files found in {tool_dir}[/yellow]")
        console.print("[dim]Hint: Run 'walt discover <url>' first to generate tools[/dim]")
        raise typer.Exit(1)

    console.print(f"[green]Found {len(tool_files)} tools[/green]")

    # Import and run MCP server
    try:
        from walt.tools.mcp.service import run_mcp_server

        asyncio.run(run_mcp_server(tool_dir, port))
    except ImportError:
        console.print("[red]❌ MCP server not available[/red]")
        console.print("[dim]This feature is currently in development[/dim]")
        raise typer.Exit(1)


def infer_provider(model_name: str) -> str:
    """Infer LLM provider from model name."""
    model_lower = model_name.lower()
    if model_lower.startswith(("gpt-", "o1-", "o3-", "o4-", "gpt-5")):
        return "openai"
    elif model_lower.startswith(("gemini-", "models/gemini")):
        return "google"
    elif "bedrock" in model_lower or model_lower.startswith(("claude-", "anthropic")):
        return "bedrock"
    else:
        # Default to OpenAI
        console.print(f"[yellow]⚠️  Unknown model prefix, defaulting to OpenAI[/yellow]")
        return "openai"


@app.command()
def agent(
    task: str = typer.Argument(..., help="Task for the agent to perform"),
    tools: Optional[str] = typer.Option(
        None, "--tools", "-t", help="Directory with tool JSON files"
    ),
    start_url: Optional[str] = typer.Option(
        None,
        "--start-url",
        "-u",
        help="Starting URL for the agent (if not provided, LLM will infer)",
    ),
    llm: str = typer.Option("gpt-5-mini", "--llm", help="LLM model to use"),
    planner_llm: Optional[str] = typer.Option(
        "gpt-5", "--planner-llm", help="LLM for planning (defaults to gpt-5)"
    ),
    planner_interval: int = typer.Option(
        15, "--planner-interval", help="Run planner every N steps"
    ),
    headless: bool = typer.Option(
        False, "--headless/--headed", help="Run browser in headless mode"
    ),
    stealth: bool = typer.Option(
        True, "--stealth/--no-stealth", help="Use patchright for bot detection evasion"
    ),
    max_steps: int = typer.Option(30, "--max-steps", help="Maximum agent steps"),
    save_gif: Optional[str] = typer.Option(
        None,
        "--save-gif",
        "-g",
        help="Save agent history as GIF (provide path or use default 'agent_history.gif')",
    ),
):
    """
    Run an agent with optional tool augmentation.

    Examples:
        walt agent "find me the cheapest blue kayak, and return its URL" --tools walt-tools/classifieds/ --start-url http:://localhost:9980
        walt agent "book a flight to NYC" --llm gpt-5-mini --headed --save-gif booking.gif
        walt agent "check my email" --start-url https://gmail.com
        walt agent "post on social media" --start-url https://twitter.com --tools walt-tools/social/
    """
    console.print(f"[bold cyan]🤖 Running agent:[/bold cyan] {task}")
    if start_url:
        console.print(f"[dim]Start URL: {start_url}[/dim]")
    if tools:
        console.print(f"[dim]Tools: {tools}[/dim]")

    # Import here to avoid slow startup
    from walt.browser_use.browser.browser import BrowserConfig
    from walt.browser_use.custom.utils import create_llm
    from walt.browser_use.custom.agent_zoo import VWA_Agent
    from walt.browser_use.custom.eval_envs.VWA import (
        VWABrowser,
        VWABrowserContext,
        VWABrowserContextConfig,
    )

    async def run_agent():
        provider = infer_provider(llm)
        llm_instance = create_llm(provider, llm, temperature=0.0)

        # Create planner LLM
        planner_provider = infer_provider(planner_llm)
        planner_llm_instance = create_llm(planner_provider, planner_llm, temperature=0.0)

        # Create VWA browser (for full VWA_Agent support)
        browser_config = BrowserConfig(headless=headless, use_stealth=stealth)
        browser = VWABrowser(browser_config)

        # Load tools and register multimodal skills
        from walt.browser_use import Controller
        from walt.browser_use.custom.skills import register_generic_skills

        controller = Controller()

        # Register multimodal skills (extract_content with vision, etc.)
        register_generic_skills(controller)

        # Load tools if provided
        tool_count = 0
        if tools:
            from walt.tools.discovery.register import register_tools_from_directory

            tool_count = register_tools_from_directory(
                controller=controller,
                tool_dir=tools,
                llm=llm_instance,
                logger=console.log,
            )
            console.print(f"[green]Loaded {tool_count} tools[/green]")

        # Build extended system message from centralized prompts
        from walt.prompts import build_extended_system_message

        extend_system_message = build_extended_system_message(
            tool_count=tool_count,
        )

        # Create browser context (VWA-style for multimodal support)
        context_config = VWABrowserContextConfig(
            browser_window_size={"width": 1280, "height": 720},
            trace_path=None,
        )
        browser_context = VWABrowserContext(
            browser=browser, config=context_config, som_color="black_transparent"
        )

        # Set up verification callback
        from walt.browser_use.custom.skills import verify_with_judge
        from walt.browser_use.custom.skills.models import VerifyAction
        from walt.browser_use.agent.views import AgentHistoryList, ActionResult

        async def verify_callback(
            task_str: str, task_image, agent_history: AgentHistoryList, browser_ctx
        ) -> ActionResult:
            """Verify task completion after agent marks done."""
            params = VerifyAction(
                task=task_str,
                task_image_paths=task_image,
                score_threshold=5,  # Pass if score >= 5
            )
            return await verify_with_judge(params, agent_history, browser_ctx)

        # Prepare the task string (include start_url if provided)
        full_task = task
        if start_url:
            full_task = f"{task}\n\nStart by navigating to: {start_url}"

        # Determine GIF output path
        gif_output = False
        if save_gif is not None:
            gif_output = save_gif if save_gif else "agent_history.gif"

        # Use VWA_Agent for multimodal + verification support
        agent = VWA_Agent(
            task=full_task,
            task_image=None,  # CLI doesn't support task images yet
            llm=llm_instance,
            browser=browser,  # Pass browser explicitly
            browser_context=browser_context,
            controller=controller,
            planner_llm=planner_llm_instance,
            planner_interval=planner_interval,
            extend_system_message=extend_system_message,
            expose_tool_actions=tool_count > 0,  # Enable tool-aware planner
            expose_multimodal_actions=True,
            max_actions_per_step=max_steps,
            register_done_callback=verify_callback,
            generate_gif=gif_output,
            retry_delay=1,
        )

        # Show detailed agent configuration
        console.print(f"\n[bold cyan]🤖 Starting WALT Agent[/bold cyan]")
        console.print(f"[dim]╭─ Configuration[/dim]")
        console.print(f"[dim]├─[/dim] [bold]LLM:[/bold] {llm} ({provider})")
        console.print(
            f"[dim]├─[/dim] [bold]Planner:[/bold] {planner_llm} ({planner_provider}) [every {planner_interval} steps]"
        )
        console.print(
            f"[dim]├─[/dim] [bold]Browser:[/bold] {'Headless' if headless else 'Headed'} mode ({'Stealth' if stealth else 'Standard'})"
        )
        console.print(f"[dim]├─[/dim] [bold]Max Steps:[/bold] {max_steps} actions per step")
        if start_url:
            console.print(f"[dim]├─[/dim] [bold]Start URL:[/bold] {start_url}")
        if tools:
            console.print(
                f"[dim]├─[/dim] [bold]Tools:[/bold] {tool_count} custom tools loaded from {tools}"
            )
        else:
            console.print(
                f"[dim]├─[/dim] [bold]Tools:[/bold] Default browser actions only (click, type, navigate, etc.)"
            )
        if gif_output:
            console.print(f"[dim]├─[/dim] [bold]GIF Output:[/bold] {gif_output}")
        console.print(f"[dim]╰─[/dim] [bold]Task:[/bold] {task}")
        console.print()

        history, final_page = await agent.run()
        console.print(f"\n[bold green]✅ Task completed[/bold green]")

        if gif_output:
            console.print(f"[green]📹 GIF saved to {gif_output}[/green]")

        # Show final result if available
        if history.is_done() and history.final_result():
            final_text = history.final_result()
            if final_text and len(final_text) < 200:
                console.print(f"[dim]Result: {final_text}[/dim]")

        # Clean up browser resources with timeout protection (following aeval.py pattern)
        # Note: agent.run() already closes browser/context in its finally block,
        # but we add explicit cleanup here as a safety measure
        try:
            await asyncio.wait_for(browser_context.close(), timeout=30.0)
        except asyncio.TimeoutError:
            console.print("[yellow]⚠️  Context cleanup timed out[/yellow]")
        except Exception:
            pass  # Ignore cleanup errors

        try:
            await asyncio.wait_for(browser.close(), timeout=30.0)
        except asyncio.TimeoutError:
            console.print("[yellow]⚠️  Browser cleanup timed out[/yellow]")
        except Exception:
            pass  # Ignore cleanup errors

        # Force garbage collection to help cleanup
        import gc

        gc.collect()

    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Interrupted by user[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"[bold red]❌ Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command(name="list")
def list_tools(
    tool_dir: str = typer.Argument("walt-tools/", help="Directory containing tools"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed information"),
):
    """
    List discovered tools.

    Examples:
        walt list
        walt list walt-tools/classifieds/ --detailed
    """
    tool_path = Path(tool_dir)

    if not tool_path.exists():
        console.print(f"[yellow]⚠️  Directory not found:[/yellow] {tool_dir}")
        console.print("[dim]Run 'walt discover <url>' to create tools[/dim]")
        raise typer.Exit(1)

    # Find all tool files
    tool_files = list(tool_path.rglob("*.tool.json"))

    if not tool_files:
        console.print(f"[yellow]No tools found in {tool_dir}[/yellow]")
        raise typer.Exit(0)

    console.print(f"[bold cyan]Found {len(tool_files)} tools in {tool_dir}[/bold cyan]\n")

    if detailed:
        # Detailed view with table
        from walt.tools.schema import ToolDefinitionSchema

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Tool Name", style="green")
        table.add_column("Description")
        table.add_column("Steps", justify="right")

        for tool_file in sorted(tool_files):
            try:
                schema = ToolDefinitionSchema.load_from_json(str(tool_file))
                table.add_row(
                    tool_file.stem.replace(".tool", ""),
                    (
                        schema.description[:60] + "..."
                        if len(schema.description) > 60
                        else schema.description
                    ),
                    str(len(schema.steps)),
                )
            except Exception as e:
                table.add_row(tool_file.stem, f"[red]Error: {e}[/red]", "-")

        console.print(table)
    else:
        # Simple list view
        for tool_file in sorted(tool_files):
            console.print(f"  • {tool_file.relative_to(tool_path)}")


@app.command()
def record(
    url: str = typer.Argument(..., help="Website URL to record demonstration on"),
    output: str = typer.Option(
        "recording.tool.json", "--output", "-o", help="Output file for tool"
    ),
    name: str = typer.Option(None, "--name", "-n", help="Tool name"),
    description: str = typer.Option(None, "--description", "-d", help="Tool description"),
):
    """
    Record a human demonstration and convert to a tool.

    Examples:
        walt record https://example.com --name search_product
        walt record https://site.com -o my-tool.json -n "Book Flight"
    """
    console.print(f"[bold cyan]🎥 Recording demonstration on:[/bold cyan] {url}")
    console.print("[dim]A browser will open. Perform your task, then close the browser.[/dim]\n")

    try:
        from walt.tools.recorder.service import record_tool

        result = asyncio.run(
            record_tool(
                url=url,
                output_file=output,
                tool_name=name,
                tool_description=description,
            )
        )

        if result:
            console.print(f"\n[bold green]✅ Tool saved to:[/bold green] {output}")
            console.print(f"[dim]Steps recorded: {result['step_count']}[/dim]")
        else:
            console.print("[yellow]⚠️  Recording cancelled or failed[/yellow]")
            raise typer.Exit(1)

    except ImportError as e:
        console.print("[red]❌ Recorder not available[/red]")
        console.print(f"[dim]Import error: {e}[/dim]")
        raise typer.Exit(1)
    except EnvironmentError as e:
        console.print("[red]❌ Recorder requires a graphical display[/red]")
        console.print(f"[yellow]{str(e)}[/yellow]")
        raise typer.Exit(1)
    except FileNotFoundError as e:
        console.print("[red]❌ Recorder extension not found[/red]")
        console.print(f"[yellow]{str(e)}[/yellow]")
        console.print(
            "\n[dim]The recorder feature requires a Chrome extension that captures browser actions.[/dim]"
        )
        console.print("[dim]This extension is not included in the standard distribution.[/dim]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]❌ Error:[/bold red] {e}")
        raise typer.Exit(1)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
