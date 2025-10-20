"""
Browser-Use Generic Skills

Centralized collection of generic skills that work across all benchmarks and platforms.
These skills are browser-agnostic and can be used with VWA, WA, or any other browser context.
"""

from walt.browser_use import Controller
from walt.browser_use.agent.views import ActionResult

# Import the generic skills
from .extract import extract_content as extract_content_impl
from .verify import verify_with_judge
from .models import ExtractContentAction, VerifyAction

# Make the main functions available at package level
__all__ = [
    "extract_content_impl",
    "verify_with_judge", 
    "ExtractContentAction",
    "VerifyAction",
    "register_generic_skills",
]


def register_generic_skills(controller: Controller, browser_type: str = "auto"):
    """
    Register all generic skills with the controller.
    
    Args:
        controller: Browser-use controller to register skills with
        browser_type: Type of browser context ("vwa", "wa", or "auto" for auto-detection)
    
    Returns:
        int: Number of skills registered
    """    
    skills_registered = 0
    
    # Register extract_content skill
    @controller.action(
        "Extract goal-relevant content from the current page, including both text and images. "
        "Ideal for processing search results, product listings, or any complex page content.",
        param_model=ExtractContentAction,
    )
    async def extract_content(params: ExtractContentAction, browser, page_extraction_llm) -> ActionResult:
        return await extract_content_impl(params, browser, page_extraction_llm)
    
    skills_registered += 1
        
    return skills_registered