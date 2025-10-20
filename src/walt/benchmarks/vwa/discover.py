#!/usr/bin/env python3
"""
VWA-specific tool discovery wrapper.
Handles VWA environment setup, authentication, and domain resets, then calls generic discovery.
"""
import asyncio
import argparse
import os
import sys
import subprocess
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from walt.tools.discovery import propose, generate


# VWA-specific configuration - load from environment
WEBSITE_URLS = {
    "classifieds": os.environ.get("VWA_CLASSIFIEDS", "http://localhost:9980"),
    "reddit": os.environ.get("VWA_REDDIT", "http://localhost:9999"),
    "shopping": os.environ.get("VWA_SHOPPING", "http://localhost:7770"),
}


async def run_vwa_domain_reset(website: str):
    """Run VWA domain reset script."""
    script_dir = os.path.dirname(__file__)
    reset_script = os.path.join(script_dir, f"scripts/reset_{website}.sh")
    
    if not os.path.exists(reset_script):
        print(f"⚠️  Reset script not found: {reset_script}")
        return
    
    print(f"🔄 Resetting {website} environment...")
    try:
        result = subprocess.run(
            ["bash", reset_script],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            print(f"✅ {website} environment reset successfully")
        else:
            print(f"⚠️  Reset completed with warnings")
    except Exception as e:
        print(f"⚠️  Reset failed: {e}")


async def setup_vwa_authentication(website: str, auth_file: str) -> bool:
    """Run VWA auto-login script."""
    try:
        from walt.benchmarks.vwa.auto_login import get_site_comb_from_filepath
    except ImportError:
        print("❌ VWA auto_login module not found")
        return False
    
    auth_dir = os.path.dirname(auth_file)
    os.makedirs(auth_dir, exist_ok=True)
    
    comb = get_site_comb_from_filepath(auth_file)
    script_dir = os.path.dirname(__file__)
    auto_login_script = os.path.join(script_dir, "helpers/auto_login.py")
    cmd = [
        "python3",
        auto_login_script,
        "--auth_folder", auth_dir,
        "--site_list", *comb,
    ]
    
    print(f"🔑 Running VWA authentication for {website}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0 and os.path.exists(auth_file):
            print(f"✅ Authentication successful")
            return True
        else:
            print(f"❌ Authentication failed")
            return False
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        return False


async def main():
    """VWA discovery CLI wrapper."""
    parser = argparse.ArgumentParser(
        description="VWA tool discovery - wrapper around generic discovery"
    )
    
    # VWA-specific arguments
    parser.add_argument(
        "--website",
        required=True,
        choices=["classifieds", "reddit", "shopping"],
        help="VWA website to discover tools for",
    )
    parser.add_argument(
        "--skip-reset",
        action="store_true",
        help="Skip VWA environment reset before discovery",
    )
    
    # Pipeline phases
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Phase 1: Discover candidate tools",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Phase 2: Generate tools from candidates",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration of existing tools",
    )
    
    # Optional configuration
    parser.add_argument(
        "--llm",
        default="gpt-5",
        help="LLM model for browser agent (default: gpt-5)",
    )
    parser.add_argument(
        "--planner-llm",
        default="gpt-5",
        help="LLM model for planner (default: gpt-5)",
    )
    parser.add_argument(
        "--max-processes",
        type=int,
        default=16,
        help="Maximum concurrent processes for generation (default: 16)",
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
    
    # VWA-specific setup
    print(f"🎯 VWA Tool Discovery for: {args.website}")
    print(f"=" * 60)
    
    # 1. Reset environment
    if not args.skip_reset:
        await run_vwa_domain_reset(args.website)
    else:
        print("⏭️  Skipping domain reset")
    
    # 2. Setup authentication
    auth_file = f".auth/{args.website}_state.json"
    auth_success = await setup_vwa_authentication(args.website, auth_file)
    
    if not auth_success:
        print("💥 CRITICAL: Authentication failed")
        print("   Cannot proceed without valid authentication")
        sys.exit(1)
    
    # 3. Prepare args for generic discovery
    base_url = WEBSITE_URLS[args.website]
    output_dir = f"outputs/vwa_{args.website}"
    
    # Create args namespace for generic discovery
    discovery_args = argparse.Namespace(
        base_url=base_url,
        output_dir=output_dir,
        llm=args.llm,
        planner_llm=args.planner_llm,
        auth_file=auth_file,
        max_processes=args.max_processes,
        test=args.test,
        optimize=args.optimize,
        force_regenerate=args.force_regenerate,
    )
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"🌐 Base URL: {base_url}")
    print(f"🔑 Authentication: {auth_file}")
    
    # 4. Run generic discovery pipeline
    if args.discover:
        print("\n🔍 Phase 1: Discovering candidate tools...")
        tools = await propose.discover_candidates(discovery_args)
        print(f"✅ Discovery complete: {len(tools)} candidates found")
    
    if args.generate:
        print("\n🚀 Phase 2: Generating tools...")
        tools_json = propose.load_existing_candidates(discovery_args)
        if not tools_json:
            print("❌ No candidates found. Run with --discover first.")
            return
        
        success_count = await generate.generate_tools(discovery_args, tools_json)
        print(f"✅ Generation complete: {success_count}/{len(tools_json)} tools successful")
    
    print(f"\n{'=' * 60}")
    print(f"✨ VWA discovery complete!")


if __name__ == "__main__":
    asyncio.run(main())

