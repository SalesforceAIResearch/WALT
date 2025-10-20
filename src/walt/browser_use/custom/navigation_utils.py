import asyncio
import logging
from typing import Optional, Union
from playwright.async_api import Page

logger = logging.getLogger(__name__)

async def robust_page_navigation(
    page: Page,
    url: str,
    timeout: int = 60000,
    wait_until: str = "domcontentloaded",
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    connection_reset_specific_retries: int = 3,
) -> bool:
    """
    Robust page navigation with connection reset error handling.
    
    Args:
        page: Playwright page object
        url: URL to navigate to
        timeout: Navigation timeout in milliseconds
        wait_until: When to consider navigation successful
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff
        max_delay: Maximum delay between retries
        connection_reset_specific_retries: Extra retries specifically for connection reset errors
    
    Returns:
        bool: True if navigation successful, False otherwise
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Try navigation with timeout
            await page.goto(url, timeout=timeout, wait_until=wait_until)
            
            # Wait for page to be ready
            await page.wait_for_load_state("networkidle", timeout=30000)
            
            logger.debug(f"Successfully navigated to {url} on attempt {attempt + 1}")
            return True
            
        except Exception as e:
            last_error = e
            error_msg = str(e).lower()
            
            # Check if this is a connection reset error
            is_connection_reset = any(keyword in error_msg for keyword in [
                "connection_reset", "err_connection_reset", "net::err_connection_reset",
                "connection reset", "connection was reset", "connection closed",
                "network error", "connection refused", "connection timed out"
            ])
            
            # Check if this is a timeout error
            is_timeout = any(keyword in error_msg for keyword in [
                "timeout", "timed out", "navigation timeout"
            ])
            
            if is_connection_reset:
                logger.warning(f"Connection reset error on attempt {attempt + 1}/{max_retries} for {url}: {e}")
                
                # For connection reset errors, we get extra retries
                if attempt < max_retries - 1 or attempt < connection_reset_specific_retries:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.info(f"Retrying navigation to {url} after {delay:.1f}s delay...")
                    await asyncio.sleep(delay)
                    continue
                    
            elif is_timeout:
                logger.warning(f"Timeout error on attempt {attempt + 1}/{max_retries} for {url}: {e}")
                
                if attempt < max_retries - 1:
                    delay = min(base_delay * (1.5 ** attempt), max_delay)
                    logger.info(f"Retrying navigation to {url} after {delay:.1f}s delay...")
                    await asyncio.sleep(delay)
                    continue
                    
            else:
                logger.error(f"Non-recoverable error on attempt {attempt + 1} for {url}: {e}")
                if attempt < max_retries - 1:
                    delay = min(base_delay * (1.2 ** attempt), max_delay)
                    await asyncio.sleep(delay)
                    continue
                    
            # If we've exhausted retries, break
            break
    
    # If we get here, all retries failed
    logger.error(f"Failed to navigate to {url} after {max_retries} attempts. Last error: {last_error}")
    return False


async def robust_page_navigation_with_fallback(
    page: Page,
    url: str,
    timeout: int = 60000,
    wait_until: str = "domcontentloaded",
    max_retries: int = 5,
    fallback_wait_until: str = "load",
) -> bool:
    """
    Robust page navigation with fallback strategies.
    
    First tries with the specified wait_until strategy, then falls back to a more lenient one.
    """
    # Try with the primary strategy
    success = await robust_page_navigation(
        page=page,
        url=url,
        timeout=timeout,
        wait_until=wait_until,
        max_retries=max_retries
    )
    
    if success:
        return True
    
    # If primary strategy failed, try with fallback
    logger.info(f"Primary navigation strategy failed for {url}, trying fallback strategy...")
    
    success = await robust_page_navigation(
        page=page,
        url=url,
        timeout=timeout,
        wait_until=fallback_wait_until,
        max_retries=max(2, max_retries // 2)  # Fewer retries for fallback
    )
    
    return success 