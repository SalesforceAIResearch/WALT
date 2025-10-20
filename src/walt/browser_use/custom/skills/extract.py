"""
Generic Extract Content Skill

Browser-agnostic content extraction skill that works across all platforms.
"""

import markdownify
from typing import Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from walt.browser_use.agent.views import ActionResult
from walt.browser_use.agent.message_manager.utils import extract_json_from_model_output

# Import browser context types
from walt.browser_use.custom.eval_envs.VWA import VWABrowserContext
from walt.browser_use.custom.eval_envs.WA import WABrowserContext
from .models import ExtractContentAction
from .utils import (
    extract_image_urls_from_content,
    process_images_parallel,
    create_interleaved_content,
)


async def extract_content(
    params: ExtractContentAction,
    browser: Union["VWABrowserContext", "WABrowserContext"],
    page_extraction_llm: BaseChatModel,
) -> ActionResult:
    """
    Extracts goal-relevant information from page HTML contents that may contain both text and image URLs.
    Page contents are first converted to an interleaved multimodal format and passed to a VLM.
    Ideal for pages with lists (eg. search results) as well as single listing pages.

    Works with both VWA and WA browser contexts.

    Example usage:
    - extract_content(goal="Find the first listing that has an image with an animal in it")
    - extract_content(goal="Do any of the listings feature a blue kayak?")
    - extract_content(goal="What animal is on the large logo featured in the middle of the page?")
    """

    try:
        page = await browser.get_current_page()
        
        # Wait for page to be fully loaded before extracting
        try:
            await page.wait_for_load_state('networkidle', timeout=5000)
        except Exception:
            # Fallback to domcontentloaded if networkidle times out
            try:
                await page.wait_for_load_state('domcontentloaded', timeout=3000)
            except Exception:
                pass  # Continue anyway if both fail
        
        content = markdownify.markdownify(await page.content())

        # Original prompt for OpenAI and other models
        prompt_text = f"""Your task is to extract goal-relevant content from a page represented as interleaved text and images. Extraction goal: {params.goal}
        
        Instructions:
        - Extract all relevant information related to the goal from the interleaved input. If the goal is vague, summarize the entire page
        - If the page contains a list of items, extract information for EACH item. If the page contains a single listing, extract information for the listing.
        - Return your response as a JSON list of extracted items. Each item should include relevant text and image (if applicable) content.
        Page content follows below:"""

        # Start with the prompt text
        message_content = [{"type": "text", "text": prompt_text}]

        image_urls = extract_image_urls_from_content(content, filter_assets=True)

        # Process images in parallel
        processed_images = await process_images_parallel(
            image_urls, max_images=params.max_images
        )

        # Add the interleaved content (text and images from the page)
        interleaved_content = create_interleaved_content(content, processed_images)
        message_content.extend(interleaved_content)

        output = page_extraction_llm.invoke([HumanMessage(content=message_content)])
        extracted_content = extract_json_from_model_output(output.content)
        
        # Format output for better readability
        import json
        try:
            formatted = json.dumps(extracted_content, indent=2)
            print("=" * 80)
            print("📄 EXTRACTED CONTENT:")
            print("=" * 80)
            print(formatted)
            print("=" * 80)
        except:
            # Fallback if json formatting fails
            print(f"📄 Extracted from page:\n{extracted_content}\n")
        
        msg = f"📄 Extracted from page:\n{extracted_content}\n"
        return ActionResult(extracted_content=msg, include_in_memory=True)

    except Exception as e:
        print(f"Error extracting content: {e}")

        try:
            page = await browser.get_current_page()
            content = markdownify.markdownify(await page.content())
            msg = f"📄 Extracted from page:\n{content}\n"
            print(msg)
            return ActionResult(extracted_content=msg, include_in_memory=True)
        except Exception as fallback_error:
            return ActionResult(
                extracted_content=f"Error extracting content: {str(fallback_error)}",
                include_in_memory=True,
            )
