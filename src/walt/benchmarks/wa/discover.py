#!/usr/bin/env python3
"""
WebArena (WA) tool discovery wrapper.
Handles WA environment setup, authentication, and configuration, then calls generic discovery.
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


# WA-specific configuration - load from environment
WA_HOST_IP = os.environ.get("WA_HOST_IP", "3.12.136.212")

WEBSITE_URLS = {
    "reddit": os.environ.get("WA_REDDIT", f"http://{WA_HOST_IP}:9999"),
    "shopping": os.environ.get("WA_SHOPPING", f"http://{WA_HOST_IP}:7770"),
    "shopping_admin": os.environ.get("WA_SHOPPING_ADMIN", f"http://{WA_HOST_IP}:7780/admin"),
    "gitlab": os.environ.get("WA_GITLAB", f"http://{WA_HOST_IP}:8023"),
    "wikipedia": os.environ.get("WA_WIKIPEDIA", f"http://{WA_HOST_IP}:8888"),
    "map": os.environ.get("WA_MAP", f"http://{WA_HOST_IP}:3000"),
    "homepage": os.environ.get("WA_HOMEPAGE", f"http://{WA_HOST_IP}:4399"),
}


async def run_wa_environment_setup(website: str):
    """Run WA environment setup if needed."""
    # WA typically doesn't need domain reset like VWA
    # Add any WA-specific environment setup here
    print(f"🔧 WA environment ready for {website}")


async def setup_wa_authentication(website: str, auth_file: str) -> bool:
    """Run WA auto-login script."""
    try:
        from walt.benchmarks.wa.auto_login import get_site_comb_from_filepath
    except ImportError:
        print("❌ WA auto_login module not found")
        return False

    auth_dir = os.path.dirname(auth_file)
    os.makedirs(auth_dir, exist_ok=True)

    # Remove wa_ prefix if present for site list
    site_name = website.replace("wa_", "")
    comb = get_site_comb_from_filepath(auth_file)

    script_dir = os.path.dirname(__file__)
    auto_login_script = os.path.join(script_dir, "helpers/auto_login.py")
    cmd = [
        "python3",
        auto_login_script,
        "--auth_folder",
        auth_dir,
        "--site_list",
        *comb,
    ]

    print(f"🔑 Running WA authentication for {website}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0 and os.path.exists(auth_file):
            print(f"✅ Authentication successful")
            return True
        else:
            print(f"⚠️  Authentication may have issues, continuing anyway...")
            return True  # WA auth is less strict
    except Exception as e:
        print(f"⚠️  Authentication error: {e}, continuing anyway...")
        return True  # WA can sometimes work without perfect auth


async def main():
    """WA discovery CLI wrapper."""
    parser = argparse.ArgumentParser(
        description="WebArena tool discovery - wrapper around generic discovery"
    )

    # WA-specific arguments
    parser.add_argument(
        "--website",
        required=True,
        choices=list(WEBSITE_URLS.keys()),
        help="WA website to discover tools for",
    )
    parser.add_argument(
        "--skip-auth",
        action="store_true",
        help="Skip WA authentication (use existing session or no auth)",
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

    # WA-specific setup
    print(f"🎯 WebArena Tool Discovery for: {args.website}")
    print(f"=" * 60)

    # 1. Setup environment
    await run_wa_environment_setup(args.website)

    # 2. Setup authentication (if not skipped)
    auth_file = None
    if not args.skip_auth:
        auth_file = f".auth/wa_{args.website}_state.json"
        auth_success = await setup_wa_authentication(args.website, auth_file)

        if not auth_success:
            print("⚠️  Warning: Authentication may not be optimal")
            print("   Continuing with discovery anyway...")
    else:
        print("⏭️  Skipping authentication")

    # 3. Prepare args for generic discovery
    base_url = WEBSITE_URLS[args.website]
    output_dir = f"outputs/wa_{args.website}"

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
    if auth_file:
        print(f"🔑 Authentication: {auth_file}")
    else:
        print(f"🔓 No authentication")

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
    print(f"✨ WebArena discovery complete!")


if __name__ == "__main__":
    asyncio.run(main())
