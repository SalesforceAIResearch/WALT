#!/usr/bin/env python3
"""
Generic tool discovery system for any website.
Discovers and generates browser automation tools from website functionality.

For benchmark-specific usage (VWA/WA), use the scripts in benchmarks/<benchmark>/discover.py
"""
import asyncio
import argparse
import os

os.environ["ANONYMIZED_TELEMETRY"] = "false"

from walt.tools.discovery import propose, generate


async def main():
    """
    Generic CLI for tool discovery on any website.
    
    Usage:
        # Discover and generate tools
        python -m walt.tools.discovery.main --url https://example.com --discover --generate
        
        # With authentication
        python -m walt.tools.discovery.main --url https://example.com --auth-file .auth/state.json --discover --generate
    """
    parser = argparse.ArgumentParser(
        description="Discover and generate browser automation tools from any website"
    )
    
    # Required arguments
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Base URL to discover tools for (e.g., https://example.com)",
    )
    
    # Pipeline phases
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Phase 1: Discover candidate tools by exploring the website",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Phase 2: Generate tools from candidates (requires --discover first)",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration of tools even if they already exist",
    )
    
    # Optional configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for discovered tools (default: outputs/<domain>)",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="gpt-5",
        help="LLM model for browser agent (default: gpt-5)",
    )
    parser.add_argument(
        "--planner-llm",
        type=str,
        default="gpt-5",
        help="LLM model for planner (default: gpt-5)",
    )
    parser.add_argument(
        "--auth-file",
        type=str,
        help="Path to Playwright storage_state JSON file for authentication",
    )
    parser.add_argument(
        "--max-processes",
        type=int,
        default=16,
        help="Maximum number of concurrent processes for tool generation (default: 16)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default=True,
        help="Test generated tools (default: True)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Generate optimized versions of tools",
    )

    args = parser.parse_args()
    
    # Derive output directory from URL if not specified
    if not args.output_dir:
        domain = args.url.replace("https://", "").replace("http://", "").split("/")[0]
        args.output_dir = f"outputs/{domain}-discovered"
    
    args.base_url = args.url
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🎯 Tool Discovery for: {args.base_url}")
    print(f"📁 Output directory: {args.output_dir}")
    if args.auth_file:
        print(f"🔑 Using authentication: {args.auth_file}")
    
    # Phase 1: Discovery
    if args.discover:
        print("\n🔍 Phase 1: Discovering candidate tools...")
        tools = await propose.discover_candidates(args)
        print(f"✅ Discovery complete: {len(tools)} candidates found")
    
    # Phase 2: Generation
    if args.generate:
        print("\n🚀 Phase 2: Generating tools...")
        
        # Load candidates
        tools_json = propose.load_existing_candidates(args)
        if not tools_json:
            print("❌ No candidates found. Run with --discover first.")
            return
        
        # Generate tools (includes testing and optimization)
        success_count = await generate.generate_tools(args, tools_json)
        print(f"✅ Generation complete: {success_count}/{len(tools_json)} tools successful")


if __name__ == "__main__":
    asyncio.run(main())
