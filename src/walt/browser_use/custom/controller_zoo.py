from walt.browser_use.controller.service import Controller
from typing import Any, Optional
from importlib import resources
import time
from walt.browser_use.controller.views import (
	ClickElementAction,
	InputTextAction,
)
from walt.browser_use.custom.browser_context_zoo import BrowserContextBugFix
from walt.browser_use.agent.views import ActionModel, ActionResult
import logging

logger = logging.getLogger(__name__)
class ControllerBugFix(Controller):
  def __init__(self, *args: Any, **kwargs: Any,):
    super().__init__(*args, **kwargs)
    self.cursor_js = resources.read_text('walt.browser_use.custom.js', 'showCursor.js')
    self.highlight_js = resources.read_text('walt.browser_use.custom.js', 'highlightBox.js')

    @self.registry.action('Click element', param_model=ClickElementAction)
    async def click_element(params: ClickElementAction, browser: BrowserContextBugFix):
      session = await browser.get_session()
      page = await browser.get_current_page()


      if params.index not in await browser.get_selector_map():
        raise Exception(f'Element with index {params.index} does not exist - retry or use alternative actions')

      element_node = await browser.get_dom_element_by_index(params.index)
      initial_pages = len(session.context.pages)

      # if element has file uploader then dont click
      if await browser.is_file_uploader(element_node):
        msg = f'Index {params.index} - has an element which opens file upload dialog. To upload files please use a specific function to upload files '
        logger.info(msg)
        return ActionResult(extracted_content=msg, include_in_memory=True)

      msg = None

      try:
        # element_node.viewport_coordinates['center']
        await self.add_effects(page=page, element_node=element_node)
        download_path = await browser._click_element_node(element_node)
        if download_path:
          msg = f'💾  Downloaded file to {download_path}'
        else:
          # msg = f'🖱️  Clicked button with index {params.index}: {element_node.get_all_text_till_next_clickable_element(max_depth=2)}'
          msg = f'🖱️  Clicked button with index {params.index}: {element_node.get_all_text_till_next_clickable_element(max_depth=2)}'

        logger.info(msg)
        logger.debug(f'Element xpath: {element_node.xpath}')
        if len(session.context.pages) > initial_pages:
          new_tab_msg = 'New tab opened - switching to it'
          msg += f' - {new_tab_msg}'
          logger.info(new_tab_msg)
          await browser.switch_to_tab(-1)
        return ActionResult(extracted_content=msg, include_in_memory=True)
      except Exception as e:
        logger.warning(f'Element not clickable with index {params.index} - most likely the page changed')
        return ActionResult(error=str(e))
      

    @self.registry.action('Input text into a input interactive element',param_model=InputTextAction,)
    async def input_text(params: InputTextAction, browser: BrowserContextBugFix, has_sensitive_data: bool = False):
      page = await browser.get_current_page()
      if params.index not in await browser.get_selector_map():
        raise Exception(f'Element index {params.index} does not exist - retry or use alternative actions')

      element_node = await browser.get_dom_element_by_index(params.index)
      await self.add_effects(page=page, element_node=element_node)
      await browser._input_text_element_node(element_node, params.text)
      if not has_sensitive_data:
        msg = f'⌨️  Input {params.text} into index {params.index}: {element_node.get_all_text_till_next_clickable_element()}'
      else:
        msg = f'⌨️  Input sensitive data into index {params.index}: {element_node.get_all_text_till_next_clickable_element()}'
      logger.info(msg)
      logger.debug(f'Element xpath: {element_node.xpath}')
      return ActionResult(extracted_content=msg, include_in_memory=True)
    
  async def add_effects(self, page, element_node):
    cursor_args = {
      'targetX': element_node.viewport_coordinates['center']['x'], 
      'targetY': element_node.viewport_coordinates['center']['y'],
      'speed':400,
      'color': '#00A1E0'
      }
    highlight_args = {
      'left':element_node.viewport_coordinates['topLeft']['x'],
      'top':element_node.viewport_coordinates['topLeft']['y'],
      'width':element_node.viewport_coordinates['width'],
      'height':element_node.viewport_coordinates['height'],
      'color':'#00A1E0'
    }
    time.sleep(1)
    await page.evaluate(self.cursor_js, cursor_args)
    time.sleep(1)
    await page.evaluate(self.highlight_js, highlight_args)
    time.sleep(1)