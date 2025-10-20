from walt.browser_use.dom.views import DOMElementNode, DOMBaseNode, DOMTextNode
from typing import Any

class DomElementNodeBugFix(DOMElementNode):
  def __init__(
    self,
    *args: Any,
    **kwargs: Any,
  ):
    super().__init__(*args, **kwargs)

  def get_all_text_till_next_clickable_element(self, max_depth: int = -1) -> str:
    text_parts = []

    def collect_text(node: DOMBaseNode, current_depth: int) -> None:
      if max_depth != -1 and current_depth > max_depth:
        return

      # Skip this branch if we hit a highlighted element (except for the current node)
      if isinstance(node, DOMElementNode) and node != self and node.highlight_index is not None:
        return

      if isinstance(node, DOMTextNode):
        text_parts.append(node.text)
      elif isinstance(node, DOMElementNode):
        for child in node.children:
          collect_text(child, current_depth + 1)
      if len(text_parts) == 0 and isinstance(node, DOMElementNode):
        for child in node.parent.children:
          if 'tag_name' in child and child.tag_name == 'label':
            text_parts.append(child.innerText)

      collect_text(self, 0)
      return '\n'.join(text_parts).strip()