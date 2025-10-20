import os
import signal
import errno
import functools
import time
import logging
import asyncio
import sys
import json
from pathlib import Path
from playwright.async_api import (
    Playwright,
    async_playwright,
)
from playwright.async_api import Browser as PlaywrightBrowser
import requests
from dataclasses import dataclass
from typing import Optional, TypedDict, Any

from walt.browser_use.browser.browser import BrowserConfig
from walt.browser_use.browser.context import BrowserContextConfig

from walt.browser_use.custom.browser_zoo import BrowserBugFix
from walt.browser_use.custom.browser_context_zoo import BrowserContextBugFix, BrowserSessionBugFix
from walt.browser_use.custom.utils import robust_page_navigation_with_fallback


logger = logging.getLogger(__name__)



def atimeout(seconds=2, error_message=os.strerror(errno.ETIME)):
    def decorator(afunc):
        async def awrap(*args, **kwargs):
            try:
                start_time = time.time()
                result = await asyncio.wait_for(afunc(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError as e:
                logger.error(
                    f"func {afunc.__name__} timed out after {time.time() - start_time} seconds")
                raise TimeoutError(error_message)
            file_path = sys.modules[afunc.__module__].__file__
            logger.debug(
                f"Function {afunc.__name__} from {file_path} took {time.time() - start_time} seconds")
            return result
        return awrap

    return decorator


def aretry_timeout(num_retry: int = 2):
    def decorator(afunc):
        async def awrap(*args, **kwargs):
            for i in range(num_retry):
                try:
                    return await afunc(*args, **kwargs)
                except TimeoutError as e:
                    logger.error(f"Retry {i + 1}/{num_retry} failed")
                    continue
            raise TimeoutError(f"Retry {num_retry} failed within time limit")

        return awrap
    return decorator


class WABrowser(BrowserBugFix):
    def __init__(self, config: BrowserConfig):
        super().__init__(config)

    async def _init(self):
        """Initialize the browser session"""
        context_manager = async_playwright()
        playwright = await context_manager.start()
        browser = await self._setup_browser(playwright)

        self.context_manager = context_manager
        self.playwright = playwright
        self.playwright_browser = browser

        return self.playwright_browser

    async def asetup(self) -> None:
        # get a new browser created as playwright.chromium.launch
        await self._init()
        return


@dataclass
class WABrowserContextConfig(BrowserContextConfig):
    geolocation: Optional[str] = None
    storage_state: Optional[str] = None


class WABrowserContext(BrowserContextBugFix):
    def __init__(self, *args: Any, **kwargs: Any,):
        super().__init__(*args, **kwargs)

    async def _create_context(self, browser: PlaywrightBrowser):
        """Creates a new browser context with anti-detection measures and loads cookies if available."""
        if self.browser.config.cdp_url and len(browser.contexts) > 0:
            context = browser.contexts[0]
        elif self.browser.config.chrome_instance_path and len(browser.contexts) > 0:
            # Connect to existing Chrome instance instead of creating new one
            context = browser.contexts[0]
        else:
            # Original code for creating new context
            context = await browser.new_context(
                viewport=self.config.browser_window_size,
                no_viewport=False,
                user_agent=self.config.user_agent,
                java_script_enabled=True,
                bypass_csp=self.config.disable_security,
                ignore_https_errors=self.config.disable_security,
                record_video_dir=self.config.save_recording_path,
                record_video_size=self.config.browser_window_size,
                locale=self.config.locale,
                device_scale_factor=1,
                geolocation=self.config.geolocation,
                storage_state=self.config.storage_state
            )

        if self.config.trace_path:
            await context.tracing.start(screenshots=True, snapshots=True, sources=True)

        # Load cookies if they exist
        if self.config.cookies_file and os.path.exists(self.config.cookies_file):
            with open(self.config.cookies_file, 'r') as f:
                cookies = json.load(f)
                logger.info(
                    f'Loaded {len(cookies)} cookies from {self.config.cookies_file}')
                await context.add_cookies(cookies)

        # Expose anti-detection scripts
        await context.add_init_script(
            """
            // Webdriver property
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });

            // Languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US']
            });

            // Plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });

            // Chrome runtime
            window.chrome = { runtime: {} };

            // Permissions
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
            (function () {
                const originalAttachShadow = Element.prototype.attachShadow;
                Element.prototype.attachShadow = function attachShadow(options) {
                    return originalAttachShadow.call(this, { ...options, mode: "open" });
                };
            })();
            """
        )

        # Set longer timeouts for multiprocessing scenarios
        context.set_default_timeout(90000)  # 90 seconds instead of default 30
        context.set_default_navigation_timeout(90000)  # 90 seconds for navigation

        return context


class WAEnv:
    def __init__(self,
                 browser: WABrowser,
                 context: WABrowserContext) -> None:
        self.browser = browser
        self.context = context
        self.reset_finished = False

    @aretry_timeout(num_retry=6)
    @atimeout(seconds=60)  # maybe environment is resetting
    async def asetup(self,  config_file: Path | None = None) -> None:
        if self.reset_finished:
            await self.browser.context_manager.__aexit__()
        await self.browser.asetup()

        if config_file:
            with open(config_file, "r") as f:
                instance_config = json.load(f)
        else:
            instance_config = {}

        storage_state = instance_config.get("storage_state", None)
        start_url = instance_config.get("start_url", None)
        geolocation = instance_config.get("geolocation", None)
        self.context.config.storage_state = storage_state
        self.context.config.geolocation = geolocation

        # Use custom viewport size if specified in the config, otherwise use the default.
        self.context.config.browser_window_size.update(
            instance_config.get("viewport_size", {}))
        playwright_browser = await self.browser.get_playwright_browser()
        context = await self.context._create_context(playwright_browser)
        self.context._add_new_page_listener(context)
        if start_url:
            start_urls = start_url.split(" |AND| ")
            for url in start_urls:
                logger.debug(f'starting new page at: {url}')
                page = await context.new_page()
                
                # Use robust navigation with connection reset handling
                success = await robust_page_navigation_with_fallback(
                    page=page,
                    url=url,
                    timeout=60000,  # Reverted from 300000 to 60000 (1 min) to prevent timeout budget exhaustion
                    wait_until="domcontentloaded",
                    max_retries=5
                )
                
                if not success:
                    logger.error(f"Failed to navigate to {url} after multiple attempts with robust navigation")
                    raise Exception(f"Failed to navigate to {url} after multiple attempts")
                            
            logger.info("bring the first page to front")
            # set the first page as the current page
            page = context.pages[0]
            await page.bring_to_front()

        # Instead of calling _update_state(), create an empty initial state
        initial_state = self.context._get_initial_state(page)
        session = BrowserSessionBugFix(
            context=context,
            current_page=page,
            cached_state=initial_state,
        )
        self.context.session = session
        return

    async def areset(
        self,
        *,
        options: dict[str, str] | None = None,
    ):
        """
        Reset the environment.
        :param options: options for the environment. The current supported options are:
            - "storage_state": the storage state of the browser. It is a file path to a json file.
        """
        if self.reset_finished:
            await self.browser.context_manager.__aexit__()

        if options is not None and "config_file" in options:
            config_file = Path(options["config_file"])
            if config_file.exists():
                await self.asetup(config_file=config_file)
            else:
                raise ValueError(f"Config file {config_file} does not exist.")
        else:
            await self.asetup()
        self.reset_finished = True
        return
