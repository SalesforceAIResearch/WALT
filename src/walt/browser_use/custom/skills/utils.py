"""
Base Utilities for Skills

Shared utility functions used by skills across all platforms.
Includes both web operations and content processing utilities.
"""

import asyncio
import aiohttp
import re
import os
import json
import io
import base64
from base64 import b64encode
from typing import List, Dict, Optional, Any
from walt.browser_use.agent.views import ActionResult
from walt.browser_use.custom.utils import pil_to_b64

# =============================================================================
# WEB OPERATION UTILITIES
# =============================================================================


async def safe_navigate(page, url: str, timeout: int = 30000) -> ActionResult:
    """
    Navigate to a URL with error handling.

    Args:
        page: Browser page instance
        url: URL to navigate to
        timeout: Navigation timeout in milliseconds

    Returns:
        ActionResult with navigation status
    """
    try:
        await page.goto(url)
        await page.wait_for_load_state("networkidle", timeout=timeout)
        return ActionResult(
            extracted_content=f"Successfully navigated to {url}",
            include_in_memory=True,
        )
    except Exception as e:
        return ActionResult(
            extracted_content=f"Navigation failed: {str(e)}",
            include_in_memory=True,
        )


async def safe_post_request(
    page,
    url: str,
    form_data: Dict[str, Any],
    csrf_token_name: str = "csrf_token",
    extract_secret: bool = False,
) -> ActionResult:
    """
    Submit a POST request using JavaScript evaluation with automatic token handling.
    This approach avoids CSRF token staleness by extracting and using tokens at runtime.

    Args:
        page: Browser page instance
        url: URL to POST to
        form_data: Form data to submit
        csrf_token_name: Name of CSRF token field to extract and use
        extract_secret: Whether to extract and include secret token

    Returns:
        ActionResult with POST operation status and response details
    """
    try:
        # Make POST request with form data using page.evaluate
        result = await page.evaluate(
            """
            async (args) => {
                const { url, params, extractSecret, csrfTokenName } = args;
                
                // Enhanced CSRF token extraction with multiple strategies using the provided token name
                let csrfToken = '';
                let tokenSource = 'not_found';
                
                // Strategy 1: Look for token with the specified name in any form
                const tokenInput = document.querySelector(`input[name="${csrfTokenName}"]`);
                if (tokenInput) {
                    csrfToken = tokenInput.value;
                    tokenSource = 'form_input';
                } else {
                    // Strategy 2: Look in all forms for any hidden inputs with the token name
                    const forms = document.querySelectorAll('form');
                    for (const form of forms) {
                        const formTokenInput = form.querySelector(`input[name="${csrfTokenName}"]`);
                        if (formTokenInput) {
                            csrfToken = formTokenInput.value;
                            tokenSource = 'form_hidden_input';
                            break;
                        }
                    }
                    
                    // Strategy 3: Look for any hidden input that might be a token (fallback)
                    if (!csrfToken) {
                        const hiddenInputs = document.querySelectorAll('input[type="hidden"]');
                        for (const input of hiddenInputs) {
                            if (input.name === csrfTokenName || input.name.includes('token')) {
                                csrfToken = input.value;
                                tokenSource = 'hidden_input_generic';
                                break;
                            }
                        }
                    }
                }
                
                // Enhanced secret token extraction if requested
                let secret = '';
                let secretSource = 'not_found';
                if (extractSecret) {
                    // Strategy 1: Look for secret in hidden form inputs
                    const secretInput = document.querySelector('input[name="secret"]');
                    if (secretInput) {
                        secret = secretInput.value;
                        secretSource = 'form_input';
                    } else {
                        // Strategy 2: Look in all forms for secret inputs
                        const forms = document.querySelectorAll('form');
                        for (const form of forms) {
                            const secretFormInput = form.querySelector('input[name="secret"]');
                            if (secretFormInput) {
                                secret = secretFormInput.value;
                                secretSource = 'form_hidden_input';
                                break;
                            }
                        }
                        
                        // Strategy 3: Check URL parameters for secret
                        if (!secret) {
                            const urlParams = new URLSearchParams(window.location.search);
                            const urlSecret = urlParams.get('secret');
                            if (urlSecret) {
                                secret = urlSecret;
                                secretSource = 'url_parameter';
                            }
                        }
                        
                        // Strategy 4: Look for secret in any management links
                        if (!secret) {
                            const managementLinks = document.querySelectorAll('a[href*="secret="]');
                            for (const link of managementLinks) {
                                const href = link.href;
                                const secretMatch = href.match(/[?&]secret=([^&]+)/);
                                if (secretMatch) {
                                    secret = secretMatch[1];
                                    secretSource = 'management_link';
                                    break;
                                }
                            }
                        }
                    }
                }
                
                const formData = new FormData();
                
                // Add CSRF token if found
                if (csrfToken) {
                    formData.append(csrfTokenName, csrfToken);
                }
                
                // Add secret token if found and requested
                if (extractSecret && secret) {
                    formData.append('secret', secret);
                }
                
                // Add all the parameters
                Object.keys(params).forEach(key => {
                    if (params[key] !== null && params[key] !== undefined) {
                        formData.append(key, params[key]);
                    }
                });
                
                // For item_add operations, ensure we have empty fields that browser sends
                if (params.action && params.action.includes('item_add')) {
                    // Add empty fields that were in working browser request
                    if (!formData.has('cityArea')) formData.append('cityArea', '');
                    if (!formData.has('cityAreaId')) formData.append('cityAreaId', '');
                    if (!formData.has('address')) formData.append('address', '');
                    if (!formData.has('contactPhone')) formData.append('contactPhone', '');
                    if (!formData.has('contactOther')) formData.append('contactOther', '');
                }
                
                const response = await fetch(url, {
                    method: 'POST',
                    body: formData,
                    credentials: 'include'
                });
                
                const responseText = await response.text();
                
                return {
                    status: response.status,
                    statusText: response.statusText,
                    url: response.url,
                    redirected: response.redirected,
                    body: responseText.substring(0, 2000),
                    csrf_token_found: csrfToken ? 'YES' : 'NO',
                    secret_found: extractSecret && secret ? 'YES' : 'NO',
                    csrf_token_length: csrfToken ? csrfToken.length : 0,
                    csrf_token_preview: csrfToken ? csrfToken.substring(0, 8) + '...' : 'none',
                    secret_preview: extractSecret && secret ? secret.substring(0, 8) + '...' : 'none',
                    token_source: tokenSource,
                    secret_source: extractSecret ? secretSource : 'not_requested'
                };
            }
        """,
            {
                "url": url,
                "params": form_data,
                "extractSecret": extract_secret,
                "csrfTokenName": csrf_token_name,
            },
        )

        if result["status"] >= 200 and result["status"] < 300:
            # Navigate to the response URL to see the updated page
            if result.get("redirected") and result.get("url"):
                await page.goto(result["url"])
                await page.wait_for_load_state("networkidle", timeout=30000)
            else:
                # If no redirect, refresh the current page
                await page.reload()
                await page.wait_for_load_state("networkidle", timeout=30000)

            return ActionResult(
                extracted_content=f"POST successful (status {result['status']}). Response analysis: {result}",
                include_in_memory=True,
            )
        else:
            return ActionResult(
                extracted_content=f"POST failed with status {result['status']}. Response: {result}",
                include_in_memory=True,
            )

    except Exception as e:
        return ActionResult(
            extracted_content=f"POST operation exception: {str(e)}",
            include_in_memory=True,
        )


def modify_url_params(current_url: str, params: Dict[str, Any]) -> str:
    """
    Modify URL parameters in an existing URL.

    Args:
        current_url: Current URL to modify
        params: Parameters to add/update

    Returns:
        Modified URL with updated parameters
    """
    target_url = current_url

    for key, value in params.items():
        param_pattern = f"[?&]{key}=[^&]*"

        if re.search(param_pattern, target_url):
            # Replace existing parameter
            target_url = re.sub(f"([?&]){key}=[^&]*", f"\\1{key}={value}", target_url)
        else:
            # Add new parameter
            separator = "&" if "?" in target_url else "?"
            target_url = f"{target_url}{separator}{key}={value}"

    return target_url


async def get_next_page(page, page_param: str = "page") -> ActionResult:
    """
    Navigate to the next page by incrementing a page parameter.

    Args:
        page: Browser page instance
        page_param: Name of the page parameter to increment

    Returns:
        ActionResult with navigation status
    """
    try:
        current_url = page.url

        # Extract current page number
        page_match = re.search(f"[?&]{page_param}=(\\d+)", current_url)
        current_page = int(page_match.group(1)) if page_match else 1
        next_page = current_page + 1

        # Build next page URL
        next_url = modify_url_params(current_url, {page_param: next_page})

        # Navigate to next page
        return await safe_navigate(page, next_url)

    except Exception as e:
        return ActionResult(
            extracted_content=f"Pagination failed: {str(e)}",
            include_in_memory=True,
        )


async def verify_login_success(page, success_indicators: list = None) -> bool:
    """
    Verify if login was successful by checking URL and page content.

    Args:
        page: Browser page instance
        success_indicators: List of strings to look for that indicate success

    Returns:
        True if login appears successful, False otherwise
    """
    try:
        if success_indicators:
            page_content = await page.content()
            page_content_lower = page_content.lower()
            return any(
                indicator.lower() in page_content_lower
                for indicator in success_indicators
            )

        return False

    except Exception:
        return False


# =============================================================================
# CONTENT PROCESSING UTILITIES
# =============================================================================


def is_asset_image(url: str) -> bool:
    """
    Filter out decorative/asset images that aren't content-relevant.
    Returns True if the image should be skipped.
    """
    # Common asset patterns to skip
    asset_patterns = [
        r"/icons?/",
        r"/assets/",
        r"/static/",
        r"/css/",
        r"/images/ui/",
        r"/img/ui/",
        r"logo",
        r"avatar",
        r"favicon",
        r"spinner",
        r"loading",
        r"banner",
        r"header",
        r"footer",
        r"button",
        r"arrow",
        r"_icon",
        r"\.ico$",
    ]

    url_lower = url.lower()
    return any(re.search(pattern, url_lower) for pattern in asset_patterns)


async def download_image_as_b64(
    url: str, timeout: int = 10, max_size_mb: float = 0.5
) -> Optional[str]:
    """
    Download an image from URL, compress if needed, and return as base64 data URL.
    Returns None if download fails.

    Args:
        url: Image URL to download
        timeout: Request timeout in seconds
        max_size_mb: Maximum file size in MB after compression
    """
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()

                    # Skip if original is too large (>2MB)
                    if len(content) > 2 * 1024 * 1024:
                        return None

                    # Process and compress image
                    compressed_content = _compress_image(content, max_size_mb)
                    if compressed_content is None:
                        return None

                    # Get content type
                    content_type = response.headers.get("content-type", "image/jpeg")
                    if not content_type.startswith("image/"):
                        content_type = "image/jpeg"

                    b64_content = b64encode(compressed_content).decode("utf-8")
                    return f"data:{content_type};base64,{b64_content}"
                return None
    except Exception:
        return None


def _compress_image(image_content: bytes, max_size_mb: float = 0.5) -> Optional[bytes]:
    """
    Aggressively compress and resize image for rudimentary understanding only.

    Args:
        image_content: Raw image bytes
        max_size_mb: Maximum size in MB (default 0.5MB = 500KB)

    Returns:
        Compressed image bytes or None if processing fails
    """
    try:
        from PIL import Image
        import io

        # Open image
        img = Image.open(io.BytesIO(image_content))

        # Convert to RGB if needed (for JPEG compatibility)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # Much smaller dimensions for rudimentary understanding (max 400x300)
        max_width, max_height = 400, 300
        if img.width > max_width or img.height > max_height:
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

        # Save with aggressive compression
        output = io.BytesIO()
        max_bytes = int(max_size_mb * 1024 * 1024)

        # More aggressive quality reduction for smaller files
        output.seek(0)
        output.truncate()
        img.save(output, format="JPEG", quality=10, optimize=True)
        # if too large, try to compress more
        if output.tell() > max_bytes:
            for attempt_quality in [60, 45, 30, 20, 15]:
                output.seek(0)
                output.truncate()
                img.save(output, format="JPEG", quality=attempt_quality, optimize=True)

                if output.tell() <= max_bytes:
                    break
            else:
                # If still too large, skip this image
                return None

        return output.getvalue()

    except Exception:
        return None


async def process_images_parallel(
    urls: List[str], max_images: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Process multiple images in parallel, returning successful downloads.

    Args:
        urls: List of absolute image URLs to process
        max_images: Maximum number of images to process (None for no limit)

    Returns:
        List of dicts with 'url' and 'b64' keys for successful downloads
    """
    if max_images:
        urls = urls[:max_images]

    # Download all images concurrently
    tasks = [download_image_as_b64(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter successful downloads
    processed_images = []
    for url, result in zip(urls, results):
        if isinstance(result, str) and result:  # Successful download
            processed_images.append({"url": url, "b64": result})

    return processed_images


def extract_image_urls_from_content(
    content: str, filter_assets: bool = True
) -> List[str]:
    """
    Extract absolute image URLs from markdown/HTML content.

    Args:
        content: Text content to search for image URLs
        filter_assets: Whether to filter out asset/decorative images

    Returns:
        List of absolute image URLs (http:// or https:// only)
    """
    # Pattern to find absolute image URLs only
    patterns = [
        r"!\[.*?\]\((https?://[^)]+)\)",  # Markdown images: ![alt](http://url)
        r'<img[^>]+src=["\']https?://([^"\']+)["\']',  # HTML img tags with absolute URLs
        r'(https?://[^\s"\']+\.(?:jpg|jpeg|png|gif|webp|svg))',  # Direct absolute URLs
    ]

    urls = []
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        urls.extend(matches)

    # Remove duplicates and filter assets
    unique_urls = []
    seen = set()
    for url in urls:
        if url not in seen and url.startswith(("http://", "https://")):
            seen.add(url)
            if not filter_assets or not is_asset_image(url):
                unique_urls.append(url)

    return unique_urls


def create_interleaved_content(
    content: str, processed_images: List[Dict[str, str]]
) -> List[Dict]:
    """
    Create interleaved text and image content for LLM processing.
    Uses different strategies based on the model to avoid reasoning loops.

    Args:
        content: Original text content
        processed_images: List of processed images with 'url' and 'b64' keys
        model_name: Name of the model being used (for model-specific optimizations)

    Returns:
        List of content blocks with 'type' and content
    """
    if not processed_images:
        return [{"type": "text", "text": content}]

    # Create a mapping of URLs to base64 data
    url_to_b64 = {img["url"]: img["b64"] for img in processed_images}

    # Find all absolute image URLs in content and create interleaved structure
    image_pattern = r'(https?://[^\s"\']+\.(?:jpg|jpeg|png|gif|webp|svg))'
    interleaved_content = []
    last_idx = 0

    for match in re.finditer(image_pattern, content, re.IGNORECASE):
        start, end = match.span()
        url = match.group(0)

        # Add preceding text
        if start > last_idx:
            text_block = content[last_idx:start].strip()
            if text_block:
                interleaved_content.append({"type": "text", "text": text_block})

        # Add image if we successfully processed it
        if url in url_to_b64:
            interleaved_content.append(
                {"type": "image_url", "image_url": {"url": url_to_b64[url]}}
            )
        else:
            # Fallback to URL as text if image processing failed
            interleaved_content.append({"type": "text", "text": url})

        last_idx = end

    # Add remaining text
    if last_idx < len(content):
        remaining_text = content[last_idx:].strip()
        if remaining_text:
            interleaved_content.append({"type": "text", "text": remaining_text})

    return interleaved_content


# =============================================================================
# VERIFICATION UTILITIES
# =============================================================================


def encode_image_from_path(image_path: str) -> str:
    """Convert image file path to base64 data URL using existing pil_to_b64 utility"""
    # Clean path if it starts with "../"
    if image_path.startswith("../"):
        image_path = image_path[3:]

    from PIL import Image

    image = Image.open(image_path)
    return pil_to_b64(image)


def build_message_with_images(text: str, image_paths: list[str]) -> list[dict]:
    """Build message content with text and images for LangChain messages"""
    content = [{"type": "text", "text": text}]

    for path in image_paths:
        if not path or not os.path.exists(path):
            print(f"Warning: Image path does not exist: {path}")
            continue
        try:
            content.append(
                {"type": "image_url", "image_url": {"url": encode_image_from_path(path)}}
            )
        except Exception as e:
            print(f"Warning: Failed to encode image {path}: {e}")
            continue

    return content

def extract_score_from_response(response: str) -> tuple[int, str]:
    """Extract score and reasoning from JSON response"""

    try:
        data = json.loads(response.strip())
        score = data.get("score", 0)
        reasoning = data.get("reasoning", "")
        return int(score), reasoning
    except Exception as e:
        print(f"Warning: Failed to parse score JSON: {e}")
        return 0, "Failed to extract score"


def format_action_history(last_actions: list, action_thoughts: list = None) -> str:
    """Format action history with optional reasoning"""
    if action_thoughts:
        return "\n".join(
            f"{i+1}. {action}. Reasoning: {action_thought}"
            for i, (action, action_thought) in enumerate(
                zip(last_actions, action_thoughts)
            )
        )
    else:
        return "\n".join(f"{i+1}. {action}" for i, action in enumerate(last_actions))
