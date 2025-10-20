from __future__ import annotations

import base64
import io
import logging
import os
import platform
from typing import TYPE_CHECKING, Optional

from walt.browser_use.agent.views import (
	AgentHistoryList,
)

if TYPE_CHECKING:
	from PIL import Image, ImageFont

logger = logging.getLogger(__name__)


def create_history_gif(
	task: str,
	history: AgentHistoryList,
	#
	output_path: str = 'agent_history.gif',
	duration: int = 3000,
	show_goals: bool = True,
	show_task: bool = True,
	show_logo: bool = False,
	font_size: int = 28,  # Reduced from 40
	title_font_size: int = 40,  # Reduced from 56
	goal_font_size: int = 32,  # Reduced from 44
	margin: int = 30,  # Reduced from 40
	line_spacing: float = 1.3,  # Reduced from 1.5
) -> None:
	"""Create a GIF from the agent's history with overlaid task and goal text."""
	if not history.history:
		logger.warning('No history to create GIF from')
		return

	from PIL import Image, ImageFont

	images = []

	# if history is empty or first screenshot is None, we can't create a gif
	if not history.history or not history.history[0].state.screenshot:
		logger.warning('No history or first screenshot to create GIF from')
		return

	# Try to load nicer fonts
	try:
		# Try different font options in order of preference
		font_options = ['Helvetica', 'Arial', 'DejaVuSans', 'Verdana']
		font_loaded = False

		for font_name in font_options:
			try:
				if platform.system() == 'Windows':
					# Need to specify the abs font path on Windows
					font_name = os.path.join(os.getenv('WIN_FONT_DIR', 'C:\\Windows\\Fonts'), font_name + '.ttf')
				regular_font = ImageFont.truetype(font_name, font_size)
				title_font = ImageFont.truetype(font_name, title_font_size)
				goal_font = ImageFont.truetype(font_name, goal_font_size)
				font_loaded = True
				break
			except OSError:
				continue

		if not font_loaded:
			raise OSError('No preferred fonts found')

	except OSError:
		regular_font = ImageFont.load_default()
		title_font = ImageFont.load_default()

		goal_font = regular_font

	# Load logo if requested
	logo = None
	if show_logo:
		try:
			logo = Image.open('./static/browser-use.png')
			# Resize logo to be small (e.g., 40px height)
			logo_height = 150
			aspect_ratio = logo.width / logo.height
			logo_width = int(logo_height * aspect_ratio)
			logo = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
		except Exception as e:
			logger.warning(f'Could not load logo: {e}')

	# Create task frame if requested
	if show_task and task:
		task_frame = _create_task_frame(
			task,
			history.history[0].state.screenshot,
			title_font,  # type: ignore
			regular_font,  # type: ignore
			logo,
			line_spacing,
		)
		images.append(task_frame)

	# Process each history item
	for i, item in enumerate(history.history, 1):
		if not item.state.screenshot:
			continue

		# Convert base64 screenshot to PIL Image
		img_data = base64.b64decode(item.state.screenshot)
		image = Image.open(io.BytesIO(img_data))

		if show_goals and item.model_output:
			# Format actions to show (similar to qualitative results figure)
			action_text = ""
			if item.model_output.action:
				# Show up to 2 actions to keep text readable
				actions_to_show = item.model_output.action[:2]
				action_lines = []
				has_done_action = False
				
				for action in actions_to_show:
					# Get the action name and key details
					action_dict = action.model_dump(exclude_unset=True)
					for action_name, action_params in action_dict.items():
						if action_name == "interacted_element":
							continue
						
						# Check if this is a done action
						if action_name == "done":
							has_done_action = True
							action_lines.append("done()")
							continue
						
						# Format like: action_name(key_param=value)
						if isinstance(action_params, dict):
							# Get most relevant parameters
							if action_name.startswith("tool_"):
								# For tools, show function-call style
								tool_name = action_name.replace("tool_", "")
								# Get first 2-3 key parameters
								params = []
								for k, v in list(action_params.items())[:3]:
									if v is not None and k not in ['index']:
										if isinstance(v, str):
											v_str = v[:40] + "..." if len(v) > 40 else v
											params.append(f"{k}={v_str}")
										else:
											params.append(f"{k}={v}")
								param_str = ", ".join(params)
								action_lines.append(f"{tool_name}({param_str})")
							elif "index" in action_params:
								# For click/input actions with index
								action_lines.append(f"{action_name}(id={action_params['index']})")
							elif "text" in action_params:
								text = action_params["text"][:30] + "..." if len(action_params["text"]) > 30 else action_params["text"]
								action_lines.append(f"{action_name}(text={text})")
							elif "url" in action_params:
								action_lines.append(f"{action_name}(url={action_params['url'][:40]}...)")
							elif "goal" in action_params:
								goal = action_params["goal"][:50] + "..." if len(action_params["goal"]) > 50 else action_params["goal"]
								action_lines.append(f"{action_name}(goal={goal})")
							else:
								action_lines.append(action_name)
						else:
							action_lines.append(action_name)
				
				# If this is the last step with done action, try to add verification result
				if has_done_action and i == len(history.history) and item.result:
					# Check if there's a verification result
					for result in item.result:
						if result.extracted_content and "whether_agent_succeeded_according_to_judge" in result.extracted_content:
							try:
								import json
								judge_data = json.loads(result.extracted_content)
								judge_verdict = judge_data.get('whether_agent_succeeded_according_to_judge', 'unknown')
								if judge_verdict == 'success':
									action_lines.append("✅ Verified: Task completed")
								elif judge_verdict == 'partial':
									action_lines.append("⚠️ Verified: Partially completed")
								else:
									action_lines.append("❌ Verified: Task failed")
							except:
								pass
				
				if len(item.model_output.action) > 2:
					action_lines.append(f"... +{len(item.model_output.action) - 2} more")
				
				action_text = "\n".join(action_lines) if action_lines else item.model_output.current_state.next_goal
			else:
				action_text = item.model_output.current_state.next_goal
			
			image = _add_overlay_to_image(
				image=image,
				step_number=i,
				goal_text=action_text,
				regular_font=regular_font,  # type: ignore
				title_font=title_font,  # type: ignore
				margin=margin,
				logo=logo,
			)

		images.append(image)

	if images:
		# Save the GIF
		images[0].save(
			output_path,
			save_all=True,
			append_images=images[1:],
			duration=duration,
			loop=0,
			optimize=False,
		)
		logger.info(f'Created GIF at {output_path}')
	else:
		logger.warning('No images found in history to create GIF')


def _create_task_frame(
	task: str,
	first_screenshot: str,
	title_font: 'ImageFont.FreeTypeFont',
	regular_font: 'ImageFont.FreeTypeFont',
	logo: Optional[Image.Image] = None,
	line_spacing: float = 1.5,
) -> 'Image.Image':
	"""Create initial frame showing the task."""
	from PIL import Image, ImageDraw, ImageFont

	img_data = base64.b64decode(first_screenshot)
	template = Image.open(io.BytesIO(img_data))
	image = Image.new('RGB', template.size, (0, 0, 0))
	draw = ImageDraw.Draw(image)

	# Calculate vertical center of image
	center_y = image.height // 2

	# Draw task text with increased font size
	margin = 140  # Increased margin
	max_width = image.width - (2 * margin)
	larger_font = ImageFont.truetype(regular_font.path, regular_font.size + 16)  # Increase font size more
	wrapped_text = _wrap_text(task, larger_font, max_width)

	# Calculate line height with spacing
	line_height = larger_font.size * line_spacing

	# Split text into lines and draw with custom spacing
	lines = wrapped_text.split('\n')
	total_height = line_height * len(lines)

	# Start position for first line
	text_y = center_y - (total_height / 2) + 50  # Shifted down slightly

	for line in lines:
		# Get line width for centering
		line_bbox = draw.textbbox((0, 0), line, font=larger_font)
		text_x = (image.width - (line_bbox[2] - line_bbox[0])) // 2

		draw.text(
			(text_x, text_y),
			line,
			font=larger_font,
			fill=(255, 255, 255),
		)
		text_y += line_height

	# Add logo if provided (top right corner)
	if logo:
		logo_margin = 20
		logo_x = image.width - logo.width - logo_margin
		image.paste(logo, (logo_x, logo_margin), logo if logo.mode == 'RGBA' else None)

	return image


def _add_overlay_to_image(
	image: 'Image.Image',
	step_number: int,
	goal_text: str,
	regular_font: 'ImageFont.FreeTypeFont',
	title_font: 'ImageFont.FreeTypeFont',
	margin: int,
	logo: Optional['Image.Image'] = None,
	display_step: bool = True,
	text_color: tuple[int, int, int, int] = (255, 255, 255, 255),
	text_box_color: tuple[int, int, int, int] = (0, 0, 0, 255),
) -> 'Image.Image':
	"""Add step number and goal overlay to an image."""
	from PIL import Image, ImageDraw

	image = image.convert('RGBA')
	txt_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
	draw = ImageDraw.Draw(txt_layer)
	if display_step:
		# Add step number (bottom left)
		step_text = str(step_number)
		step_bbox = draw.textbbox((0, 0), step_text, font=title_font)
		step_width = step_bbox[2] - step_bbox[0]
		step_height = step_bbox[3] - step_bbox[1]

		# Position step number in bottom left
		x_step = margin + 10  # Slight additional offset from edge
		y_step = image.height - margin - step_height - 10  # Slight offset from bottom

		# Draw rounded rectangle background for step number
		padding = 20  # Increased padding
		step_bg_bbox = (
			x_step - padding,
			y_step - padding,
			x_step + step_width + padding,
			y_step + step_height + padding,
		)
		draw.rounded_rectangle(
			step_bg_bbox,
			radius=15,  # Add rounded corners
			fill=text_box_color,
		)

		# Draw step number
		draw.text(
			(x_step, y_step),
			step_text,
			font=title_font,
			fill=text_color,
		)

	# Draw goal text (centered, bottom)
	max_width = image.width - (4 * margin)
	wrapped_goal = _wrap_text(goal_text, title_font, max_width)
	goal_bbox = draw.multiline_textbbox((0, 0), wrapped_goal, font=title_font)
	goal_width = goal_bbox[2] - goal_bbox[0]
	goal_height = goal_bbox[3] - goal_bbox[1]

	# Center goal text horizontally, place above step number
	x_goal = (image.width - goal_width) // 2
	y_goal = y_step - goal_height - padding * 4  # More space between step and goal

	# Draw rounded rectangle background for goal
	padding_goal = 25  # Increased padding for goal
	goal_bg_bbox = (
		x_goal - padding_goal,  # Remove extra space for logo
		y_goal - padding_goal,
		x_goal + goal_width + padding_goal,
		y_goal + goal_height + padding_goal,
	)
	draw.rounded_rectangle(
		goal_bg_bbox,
		radius=15,  # Add rounded corners
		fill=text_box_color,
	)

	# Draw goal text
	draw.multiline_text(
		(x_goal, y_goal),
		wrapped_goal,
		font=title_font,
		fill=text_color,
		align='center',
	)

	# Add logo if provided (top right corner)
	if logo:
		logo_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
		logo_margin = 20
		logo_x = image.width - logo.width - logo_margin
		logo_layer.paste(logo, (logo_x, logo_margin), logo if logo.mode == 'RGBA' else None)
		txt_layer = Image.alpha_composite(logo_layer, txt_layer)

	# Composite and convert
	result = Image.alpha_composite(image, txt_layer)
	return result.convert('RGB')


def _wrap_text(text: str, font: 'ImageFont.FreeTypeFont', max_width: int) -> str:
	"""
	Wrap text to fit within a given width.

	Args:
	    text: Text to wrap
	    font: Font to use for text
	    max_width: Maximum width in pixels

	Returns:
	    Wrapped text with newlines
	"""
	words = text.split()
	lines = []
	current_line = []

	for word in words:
		current_line.append(word)
		line = ' '.join(current_line)
		bbox = font.getbbox(line)
		if bbox[2] > max_width:
			if len(current_line) == 1:
				lines.append(current_line.pop())
			else:
				current_line.pop()
				lines.append(' '.join(current_line))
				current_line = [word]

	if current_line:
		lines.append(' '.join(current_line))

	return '\n'.join(lines)
