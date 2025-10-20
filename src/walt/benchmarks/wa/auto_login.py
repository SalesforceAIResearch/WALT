"""Script to automatically login each website"""

import argparse
import glob
import json
import os
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

from playwright.sync_api import sync_playwright
from dotenv import load_dotenv

load_dotenv()

# Import available customer accounts for shopping site and Reddit accounts
import sys

sys.path.append(os.path.dirname(__file__))
from accounts_config import (
    CUSTOMER_ACCOUNTS,
    get_account_by_index,
    get_total_accounts,
    REDDIT_ACCOUNTS,
    get_reddit_account_by_index,
    get_total_reddit_accounts,
)

from browser_use.custom.evaluators.wa.env_config import (
    ACCOUNTS,
    GITLAB,
    REDDIT,
    SHOPPING,
    SHOPPING_ADMIN,
)

HEADLESS = True
SLOW_MO = 0


SITES = ["gitlab", "shopping", "shopping_admin", "reddit"]
URLS = [
    f"{GITLAB}/-/profile",
    f"{SHOPPING}/wishlist/",
    f"{SHOPPING_ADMIN}/dashboard",
    f"{REDDIT}/user/{ACCOUNTS['reddit']['username']}/account",
]
EXACT_MATCH = [True, True, True, True]
KEYWORDS = ["", "", "Dashboard", "Delete"]


def is_expired(
    storage_state: Path, url: str, keyword: str, url_exact: bool = True
) -> bool:
    """Test whether the cookie is expired"""
    if not storage_state.exists():
        return True

    # First check cookie expiry times
    try:
        with open(storage_state, "r") as f:
            state_data = json.load(f)

        current_time = time.time()
        expired_cookies = []

        for cookie in state_data.get("cookies", []):
            expires = cookie.get("expires", -1)
            # Handle session cookies (expires: -1) as valid
            if expires != -1 and expires < current_time:
                expired_cookies.append(cookie["name"])

        # If many cookies are expired, likely the auth is stale
        if len(expired_cookies) > len(state_data.get("cookies", [])) // 2:
            print(f"  ⚠️  Many expired cookies in {storage_state}: {expired_cookies}")

    except Exception as e:
        print(f"  ⚠️  Could not check cookie expiry for {storage_state}: {e}")

    # Then test actual authentication by loading the page
    context_manager = sync_playwright()
    playwright = context_manager.__enter__()
    browser = playwright.chromium.launch(headless=True, slow_mo=SLOW_MO)
    context = browser.new_context(storage_state=storage_state)
    page = context.new_page()
    page.goto(url)
    time.sleep(1)
    d_url = page.url
    content = page.content()
    context_manager.__exit__()
    if keyword:
        return keyword not in content
    else:
        if url_exact:
            return d_url != url
        else:
            return url not in d_url


async def is_expired_async(
    storage_state: Path, url: str, keyword: str, url_exact: bool = True
) -> bool:
    """Test whether the cookie is expired (async version)"""
    if not storage_state.exists():
        return True

    # First check cookie expiry times
    try:
        with open(storage_state, "r") as f:
            state_data = json.load(f)

        current_time = time.time()
        expired_cookies = []

        for cookie in state_data.get("cookies", []):
            expires = cookie.get("expires", -1)
            # Handle session cookies (expires: -1) as valid
            if expires != -1 and expires < current_time:
                expired_cookies.append(cookie["name"])

        # If many cookies are expired, likely the auth is stale
        if len(expired_cookies) > len(state_data.get("cookies", [])) // 2:
            print(f"  ⚠️  Many expired cookies in {storage_state}: {expired_cookies}")

    except Exception as e:
        print(f"  ⚠️  Could not check cookie expiry for {storage_state}: {e}")

    # Then test actual authentication by loading the page
    from playwright.async_api import async_playwright

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True, slow_mo=SLOW_MO)
        context = await browser.new_context(storage_state=storage_state)
        page = await context.new_page()
        await page.goto(url)
        await page.wait_for_timeout(1000)  # Replace time.sleep with async wait
        d_url = page.url
        content = await page.content()
        await browser.close()

        if keyword:
            return keyword not in content
        else:
            if url_exact:
                return d_url != url
            else:
                return url not in d_url


def get_shopping_account_for_task(
    process_id: int = None, task_info: str = None
) -> dict:
    """
    Get a shopping account using hash-based distribution to ensure good spread across all accounts.

    Args:
        process_id: The process ID (for multiprocessing distribution)
        task_info: Task-specific info like config file path (for additional entropy)

    Returns:
        Dictionary with 'email' and 'password' keys
    """
    # Create a hash seed from available information
    hash_input = ""
    if process_id is not None:
        hash_input += str(process_id)
    if task_info:
        hash_input += str(task_info)

    # Fallback to current time if no input provided
    if not hash_input:
        import time

        hash_input = str(int(time.time() * 1000000))  # microsecond precision

    # Generate hash and map to account index
    hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
    account_index = hash_value % get_total_accounts()

    account = get_account_by_index(account_index)
    print(
        f"Selected shopping account {account_index} ({account['email']}) for process_id={process_id}, task_info={task_info}"
    )

    return account


def get_reddit_account_for_task(process_id: int = None, task_info: str = None) -> dict:
    """
    Get a Reddit account using hash-based distribution to ensure good spread across all accounts.

    Args:
        process_id: The process ID (for multiprocessing distribution)
        task_info: Task-specific info like config file path (for additional entropy)

    Returns:
        Dictionary with 'email' and 'password' keys (note: email field represents username for Reddit)
    """
    # Create a hash seed from available information
    hash_input = ""
    if process_id is not None:
        hash_input += str(process_id)
    if task_info:
        hash_input += str(task_info)

    # Fallback to current time if no input provided
    if not hash_input:
        import time

        hash_input = str(int(time.time() * 1000000))  # microsecond precision

    # Generate hash and map to account index
    hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
    account_index = hash_value % get_total_reddit_accounts()

    account = get_reddit_account_by_index(account_index)
    print(
        f"Selected Reddit account {account_index} ({account['email']}) for process_id={process_id}, task_info={task_info}"
    )

    return account


def renew_comb(
    comb: list[str],
    auth_folder: str = "./.auth",
    shopping_email: str = None,
    shopping_password: str = None,
    process_id: int = None,
    task_info: str = None,
) -> None:
    context_manager = sync_playwright()
    playwright = context_manager.__enter__()
    browser = playwright.chromium.launch(headless=HEADLESS)
    context = browser.new_context()
    page = context.new_page()

    if "shopping" in comb:
        # Use provided credentials or select account via hash distribution
        if shopping_email and shopping_password:
            username = shopping_email
            password = shopping_password
        else:
            account = get_shopping_account_for_task(process_id, task_info)
            username = account["email"]
            password = account["password"]
        page.goto(f"{SHOPPING}/customer/account/login/")
        page.get_by_label("Email", exact=True).fill(username)
        page.get_by_label("Password", exact=True).fill(password)
        page.get_by_role("button", name="Sign In").click()

    if "reddit" in comb:
        # Use distributed account selection for Reddit to prevent rate limit conflicts
        account = get_reddit_account_for_task(process_id, task_info)
        username = account[
            "email"
        ]  # Note: email field contains username for Reddit accounts
        password = account["password"]
        page.goto(f"{REDDIT}/login")
        page.get_by_label("Username").fill(username)
        page.get_by_label("Password").fill(password)
        page.get_by_role("button", name="Log in").click()

    if "shopping_admin" in comb:
        username = ACCOUNTS["shopping_admin"]["username"]
        password = ACCOUNTS["shopping_admin"]["password"]
        page.goto(f"{SHOPPING_ADMIN}")
        page.get_by_placeholder("user name").fill(username)
        page.get_by_placeholder("password").fill(password)
        page.get_by_role("button", name="Sign in").click()

    if "gitlab" in comb:
        username = ACCOUNTS["gitlab"]["username"]
        password = ACCOUNTS["gitlab"]["password"]
        page.goto(f"{GITLAB}/users/sign_in")
        page.get_by_test_id("username-field").click()
        page.get_by_test_id("username-field").fill(username)
        page.get_by_test_id("username-field").press("Tab")
        page.get_by_test_id("password-field").fill(password)
        page.get_by_test_id("sign-in-button").click()

    context.storage_state(path=f"{auth_folder}/{'.'.join(comb)}_state.json")
    print(f"Saved auth state to {auth_folder}/{'.'.join(comb)}_state.json")

    # Store account metadata for auth
    if "shopping" in comb or "reddit" in comb:
        auth_file = f"{auth_folder}/{'.'.join(comb)}_state.json"
        try:
            with open(auth_file, "r") as f:
                data = json.load(f)

            # Add metadata
            if "metadata" not in data:
                data["metadata"] = {}
            data["metadata"]["account_email"] = (
                username if "username" in locals() else "unknown"
            )
            if process_id is not None:
                data["metadata"]["process_id"] = process_id
            if task_info:
                data["metadata"]["task_info"] = task_info

            with open(auth_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not store account metadata: {e}")

    context_manager.__exit__()


def get_site_comb_from_filepath(file_path: str) -> list[str]:
    comb = os.path.basename(file_path).rsplit("_", 1)[0].split(".")
    return comb


def main(
    auth_folder: str = "./.auth",
    target_sites: list = None,
    process_id: int = None,
    task_info: str = None,
) -> None:
    """Generate auth files. If target_sites is specified, only generate for those sites."""

    # Create auth folder (like VWA version does)
    os.makedirs(auth_folder, exist_ok=True)

    # Determine what to generate based on target_sites or auth_folder name
    if target_sites:
        sites_to_generate = target_sites
        print(f"🎯 Generating auth for specified sites: {sites_to_generate}")
    else:
        # Infer from auth folder name (like the auth pool script does)
        auth_dir_name = os.path.basename(auth_folder).lower()

        if "shopping_admin" in auth_dir_name or (
            "shopping" in auth_dir_name and "admin" in auth_dir_name
        ):
            sites_to_generate = ["shopping_admin"]
            print(
                f"🎯 Detected shopping_admin workload from folder name, generating: {sites_to_generate}"
            )
        elif "shopping" in auth_dir_name:
            sites_to_generate = ["shopping"]
            print(
                f"🎯 Detected shopping workload from folder name, generating: {sites_to_generate}"
            )
        elif "reddit" in auth_dir_name:
            sites_to_generate = ["reddit"]
            print(
                f"🎯 Detected reddit workload from folder name, generating: {sites_to_generate}"
            )
        elif "gitlab" in auth_dir_name:
            sites_to_generate = ["gitlab"]
            print(
                f"🎯 Detected gitlab workload from folder name, generating: {sites_to_generate}"
            )
        else:
            # Default: generate all combinations (like before)
            sites_to_generate = SITES
            print(
                f"🔧 No specific workload detected, generating all combinations for: {sites_to_generate}"
            )

    max_workers = 8
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if len(sites_to_generate) == len(SITES):
            # Generate all combinations (original behavior)
            pairs = list(combinations(SITES, 2))
            for pair in pairs:
                # TODO[shuyanzh] auth don't work on these two sites
                if "reddit" in pair and (
                    "shopping" in pair or "shopping_admin" in pair
                ):
                    continue
                executor.submit(
                    renew_comb,
                    list(sorted(pair)),
                    auth_folder=auth_folder,
                    process_id=process_id,
                    task_info=task_info,
                )

            for site in SITES:
                executor.submit(
                    renew_comb,
                    [site],
                    auth_folder=auth_folder,
                    process_id=process_id,
                    task_info=task_info,
                )
        else:
            # Generate only for target sites (optimized)
            for site in sites_to_generate:
                if site in SITES:
                    executor.submit(
                        renew_comb,
                        [site],
                        auth_folder=auth_folder,
                        process_id=process_id,
                        task_info=task_info,
                    )
                else:
                    print(f"⚠️  Warning: Unknown site '{site}', skipping")

    futures = []
    cookie_files = list(glob.glob(f"{auth_folder}/*.json"))
    future_to_file = {}  # Track which future corresponds to which file

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for c_file in cookie_files:
            comb = get_site_comb_from_filepath(c_file)
            for cur_site in comb:
                url = URLS[SITES.index(cur_site)]
                keyword = KEYWORDS[SITES.index(cur_site)]
                match = EXACT_MATCH[SITES.index(cur_site)]
                future = executor.submit(is_expired, Path(c_file), url, keyword, match)
                futures.append(future)
                future_to_file[future] = c_file

    # Validate cookies - show details about what's failing
    failed_cookies = set()
    total_validations = len(futures)
    passed_validations = 0

    for future in futures:
        c_file = future_to_file[future]
        try:
            if future.result():
                print(f"❌ Cookie validation failed: {c_file}")
                failed_cookies.add(c_file)
            else:
                print(f"✅ Cookie validation passed: {c_file}")
                passed_validations += 1
        except Exception as e:
            print(f"⚠️  Warning: Could not validate cookie {c_file}: {e}")
            failed_cookies.add(c_file)

    print(f"\n🔍 Validation Summary:")
    print(f"   Total validations: {total_validations}")
    print(f"   Passed: {passed_validations}")
    print(f"   Failed: {total_validations - passed_validations}")
    print(f"   Unique files with failures: {len(failed_cookies)}")

    if failed_cookies:
        print(f"⚠️  Files with validation issues: {sorted(failed_cookies)}")
        print(
            "🔧 Note: Validation issues may be temporary - auth files might still work for tasks"
        )
    else:
        print("✅ All cookies passed validation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site_list", nargs="+", default=[])
    parser.add_argument("--auth_folder", type=str, default="./.auth")
    parser.add_argument(
        "--target_sites",
        nargs="+",
        help="Only generate auth for specified sites (e.g., --target_sites reddit shopping)",
    )
    parser.add_argument(
        "--process_id", type=int, help="Process ID for account distribution"
    )
    parser.add_argument(
        "--task_info",
        type=str,
        help="Task info for account distribution (e.g., config file path)",
    )
    args = parser.parse_args()

    if args.site_list:
        if "all" in args.site_list:
            main(
                auth_folder=args.auth_folder,
                target_sites=args.target_sites,
                process_id=args.process_id,
                task_info=args.task_info,
            )
        else:
            renew_comb(
                args.site_list,
                auth_folder=args.auth_folder,
                process_id=args.process_id,
                task_info=args.task_info,
            )
    else:
        main(
            auth_folder=args.auth_folder,
            target_sites=args.target_sites,
            process_id=args.process_id,
            task_info=args.task_info,
        )
