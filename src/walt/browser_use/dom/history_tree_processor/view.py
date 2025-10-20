from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel


@dataclass
class HashedDomElement:
	"""
	Hash of the dom element to be used as a unique identifier
	"""

	branch_path_hash: str
	attributes_hash: str
	xpath_hash: str
	# text_hash: str


class Coordinates(BaseModel):
	x: int
	y: int


class CoordinateSet(BaseModel):
	top_left: Coordinates
	top_right: Coordinates
	bottom_left: Coordinates
	bottom_right: Coordinates
	center: Coordinates
	width: int
	height: int


class ViewportInfo(BaseModel):
	width: int
	height: int
	scroll_x: int = 0
	scroll_y: int = 0



@dataclass
class DOMHistoryElement:
	tag_name: str
	xpath: str
	highlight_index: Optional[int]
	entire_parent_branch_path: list[str]
	attributes: dict[str, str]
	shadow_root: bool = False
	css_selector: Optional[str] = None
	page_coordinates: Optional[CoordinateSet] = None
	viewport_coordinates: Optional[CoordinateSet] = None
	viewport_info: Optional[ViewportInfo] = None

	def to_dict(self) -> dict:
		page_coordinates = self.page_coordinates.model_dump() if self.page_coordinates else None
		viewport_coordinates = self.viewport_coordinates.model_dump() if self.viewport_coordinates else None
		try:
			viewport_info = self.viewport_info.model_dump() if self.viewport_info else None
		except:
			# print("has error using model_dump for viewport_info, need to fix it later. Now just save the width and height as a dict.")
			# this is becuase the viewport_info is actually use the definition from walt.browser_use.dom.service.ViewportInfo, which is a simple dataclass
			viewport_info = {'width': self.viewport_info.width, 'height': self.viewport_info.height}

		return {
			'tag_name': self.tag_name,
			'xpath': self.xpath,
			'highlight_index': self.highlight_index,
			'entire_parent_branch_path': self.entire_parent_branch_path,
			'attributes': self.attributes,
			'shadow_root': self.shadow_root,
			'css_selector': self.css_selector,
			'page_coordinates': page_coordinates,
			'viewport_coordinates': viewport_coordinates,
			'viewport_info': viewport_info,
		}
