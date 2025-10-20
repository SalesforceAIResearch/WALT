import asyncio
import json
import pathlib
from typing import Optional

# Import lightweight view models (just Pydantic schemas)
from walt.tools.recorder.views import (
	HttpRecordingStoppedEvent,
	HttptoolUpdateEvent,
	RecorderEvent,
	ToolDefinitionSchema,
)

# Heavy imports (Browser, FastAPI, uvicorn) are done inside methods to keep module load fast

# Path Configuration 
# The extension is in the tool-use directory (sibling to walt)
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
# Navigate from walt/src/walt/tools/recorder -> walt -> browser-use-dev -> tool-use/extension
EXT_DIR = SCRIPT_DIR.parent.parent.parent.parent.parent / 'tool-use' / 'extension' / '.output' / 'chrome-mv3'
USER_DATA_DIR = SCRIPT_DIR / 'user_data_dir'


class RecordingService:
	def __init__(self):
		# Import FastAPI here - only when RecordingService is instantiated
		from fastapi import FastAPI
		
		self.event_queue: asyncio.Queue[RecorderEvent] = asyncio.Queue()
		self.last_tool_update_event: Optional[HttptoolUpdateEvent] = None
		self.browser = None
		self.browser_context = None

		self.final_tool_output: Optional[ToolDefinitionSchema] = None
		self.recording_complete_event = asyncio.Event()
		self.final_tool_processed_lock = asyncio.Lock()
		self.final_tool_processed_flag = False

		self.app = FastAPI(title='Temporary Recording Event Server')
		self.app.add_api_route('/event', self._handle_event_post, methods=['POST'], status_code=202)
		# -- DEBUGGING --
		# Turn this on to debug requests
		# @self.app.middleware("http")
		# async def log_requests(request: Request, call_next):
		#     print(f"[Debug] Incoming request: {request.method} {request.url}")
		#     try:
		#         # Read request body
		#         body = await request.body()
		#         print(f"[Debug] Request body: {body.decode('utf-8', errors='replace')}")
		#         response = await call_next(request)
		#         print(f"[Debug] Response status: {response.status_code}")
		#         return response
		#     except Exception as e:
		#         print(f"[Error] Error processing request: {str(e)}")

		self.uvicorn_server_instance = None
		self.server_task: Optional[asyncio.Task] = None
		self.browser_task: Optional[asyncio.Task] = None
		self.event_processor_task: Optional[asyncio.Task] = None

	async def _handle_event_post(self, event_data: RecorderEvent):
		if isinstance(event_data, HttptoolUpdateEvent):
			self.last_tool_update_event = event_data
		await self.event_queue.put(event_data)
		return {'status': 'accepted', 'message': 'Event queued for processing'}

	async def _process_event_queue(self):
		print('[Service] Event processing task started.')
		try:
			while True:
				event = await self.event_queue.get()
				print(f'[Service] Event Received: {event.type}')
				if isinstance(event, HttptoolUpdateEvent):
					# self.last_tool_update_event is already updated in _handle_event_post
					pass
				elif isinstance(event, HttpRecordingStoppedEvent):
					print('[Service] RecordingStoppedEvent received, processing final tool...')
					await self._capture_and_signal_final_tool('RecordingStoppedEvent')
				self.event_queue.task_done()
		except asyncio.CancelledError:
			print('[Service] Event processing task cancelled.')
		except Exception as e:
			print(f'[Service] Error in event processing task: {e}')

	async def _capture_and_signal_final_tool(self, trigger_reason: str):
		processed_this_call = False
		async with self.final_tool_processed_lock:
			if not self.final_tool_processed_flag and self.last_tool_update_event:
				print(f'[Service] Capturing final tool (Trigger: {trigger_reason}).')
				self.final_tool_output = self.last_tool_update_event.payload
				self.final_tool_processed_flag = True
				processed_this_call = True

		if processed_this_call:
			print('[Service] Final tool captured. Setting recording_complete_event.')
			self.recording_complete_event.set()  # Signal completion to the main method

			# If processing was due to RecordingStoppedEvent, also try to close the browser/context
			if trigger_reason == 'RecordingStoppedEvent':
				print('[Service] Attempting to close browser due to RecordingStoppedEvent...')
				try:
					if getattr(self, 'browser_context', None):
						await self.browser_context.close()
					if getattr(self, 'browser', None):
						await self.browser.close()
					print('[Service] Browser close command issued.')
				except Exception as e_close:
					print(f'[Service] Error closing browser on recording stop: {e_close}')

	async def _launch_browser_and_wait(self, start_url: Optional[str] = None, login: bool = False):
		"""Launch the browser and wait for it to be closed or recording to stop."""
		# Import browser dependencies only when launching
		from walt.browser_use import Browser
		from walt.browser_use.browser.browser import BrowserConfig
		from walt.browser_use.browser.context import BrowserContextConfig
		
		print(f'[Service] Attempting to load extension from: {EXT_DIR}')
		if not EXT_DIR.exists() or not EXT_DIR.is_dir():
			print(f'[Service] ERROR: Extension directory not found: {EXT_DIR}')
			self.recording_complete_event.set()  # Signal failure
			return

		# Ensure user data dir exists
		USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
		print(f'[Service] Using browser user data directory: {USER_DATA_DIR}')

		try:
			# Create and configure browser with extension flags (older browser-use API)
			config = BrowserConfig(
				headless=False,
				extra_chromium_args=[
					f'--disable-extensions-except={str(EXT_DIR.resolve())}',
					f'--load-extension={str(EXT_DIR.resolve())}',
					'--no-default-browser-check',
					'--no-first-run',
					f'--user-data-dir={str(USER_DATA_DIR.resolve())}',
				],
			)
			self.browser = Browser(config=config)
			# Create a dedicated context so the extension/recorder captures actions
			self.browser_context = await self.browser.new_context(BrowserContextConfig())
			page = await self.browser_context.get_current_page()
			# goto start url if present

			if login:  # run auto login
				username = "blake.sullivan@gmail.com"
				password = "Password.123"
				await page.goto(f"http://localhost:9980/index.php?page=login")
				await page.locator("#email").fill(username)
				await page.locator("#password").fill(password)
				await page.get_by_role("button", name="Log in").click()

			if start_url:
				await page.goto(start_url)
			print('[Service] Browser launched. Waiting for close or recording stop...')

			# Wait for browser to be closed manually or recording to stop
			# Poll to check if context/page is still valid
			while True:
				try:
					await self.browser_context.get_current_page()
					await asyncio.sleep(1)  # Poll every second
				except Exception:
					# Browser is likely closed
					print('[Service] Browser appears to be closed or inaccessible.')
					break

		except asyncio.CancelledError:
			print('[Service] Browser task cancelled.')
			if self.browser:
				try:
					await self.browser.close()
				except:
					pass  # Best effort
			raise  # Re-raise to be caught by gather
		except Exception as e:
			print(f'[Service] Error in browser task: {e}')
		finally:
			print('[Service] Browser task finalization.')
			# self.browser = None
			# This call ensures that if browser is closed manually, we still try to capture.
			await self._capture_and_signal_final_tool('BrowserTaskEnded')

	async def capture_tool(
		self, start_url: Optional[str] = None, login: bool = False
	) -> Optional[ToolDefinitionSchema]:
		"""Capture a tool from the browser."""
		# Import uvicorn only when capturing
		import uvicorn
		
		print('[Service] Starting capture_tool session...')
		# Reset state for this session
		self.last_tool_update_event = None
		self.final_tool_output = None
		self.recording_complete_event.clear()
		self.final_tool_processed_flag = False

		# Start background tasks
		self.event_processor_task = asyncio.create_task(self._process_event_queue())
		self.browser_task = asyncio.create_task(self._launch_browser_and_wait(start_url=start_url, login=login))

		# Configure and start Uvicorn server
		config = uvicorn.Config(self.app, host='127.0.0.1', port=7331, log_level='warning', loop='asyncio')
		self.uvicorn_server_instance = uvicorn.Server(config)
		self.server_task = asyncio.create_task(self.uvicorn_server_instance.serve())
		print('[Service] Uvicorn server task started.')

		try:
			print('[Service] Waiting for recording to complete...')
			await self.recording_complete_event.wait()
			print('[Service] Recording complete event received. Proceeding to cleanup.')
		except asyncio.CancelledError:
			print('[Service] capture_tool task was cancelled externally.')
		finally:
			print('[Service] Starting cleanup phase...')

			# 1. Stop Uvicorn server
			if self.uvicorn_server_instance and self.server_task and not self.server_task.done():
				print('[Service] Signaling Uvicorn server to shut down...')
				self.uvicorn_server_instance.should_exit = True
				try:
					await asyncio.wait_for(self.server_task, timeout=5)  # Give server time to shut down
				except asyncio.TimeoutError:
					print('[Service] Uvicorn server shutdown timed out. Cancelling task.')
					self.server_task.cancel()
				except asyncio.CancelledError:  # If capture_tool itself was cancelled
					pass
				except Exception as e_server_shutdown:
					print(f'[Service] Error during Uvicorn server shutdown: {e_server_shutdown}')

			# 2. Stop browser task (and ensure browser/context are closed)
			if self.browser_task and not self.browser_task.done():
				print('[Service] Cancelling browser task...')
				self.browser_task.cancel()
				try:
					await self.browser_task
				except asyncio.CancelledError:
					pass
				except Exception as e_browser_cancel:
					print(f'[Service] Error awaiting cancelled browser task: {e_browser_cancel}')

			# Close context first, then browser (old API cleanup)
			try:
				if getattr(self, 'browser_context', None):
					print('[Service] Ensuring browser context is closed in cleanup...')
					await self.browser_context.close()
			except Exception as e_ctx_close:
				print(f'[Service] Error closing browser context in final cleanup: {e_ctx_close}')
			try:
				if getattr(self, 'browser', None):
					print('[Service] Ensuring browser is closed in cleanup...')
					await self.browser.close()
			except Exception as e_browser_close:
				print(f'[Service] Error closing browser in final cleanup: {e_browser_close}')
				# self.browser = None

			# 3. Stop event processor task
			if self.event_processor_task and not self.event_processor_task.done():
				print('[Service] Cancelling event processor task...')
				self.event_processor_task.cancel()
				try:
					await self.event_processor_task
				except asyncio.CancelledError:
					pass
				except Exception as e_ep_cancel:
					print(f'[Service] Error awaiting cancelled event processor task: {e_ep_cancel}')

			print('[Service] Cleanup phase complete.')

		if self.final_tool_output:
			print('[Service] Returning captured tool.')
		else:
			print('[Service] No tool captured or an error occurred.')
		return self.final_tool_output


async def main_service_runner():  # Example of how to run the service
	service = RecordingService()
	tool_data = await service.capture_tool()
	if tool_data:
		print('\n--- CAPTURED tool DATA (from main_service_runner) ---')
		# Assuming ToolDefinitionSchema has model_dump_json or similar
		try:
			print(tool_data.model_dump_json(indent=2))
		except AttributeError:
			print(json.dumps(tool_data, indent=2))  # Fallback for plain dicts if model_dump_json not present
		print('-----------------------------------------------------')
	else:
		print('No tool data was captured by the service.')


async def record_tool(
	url: Optional[str] = None,
	output_file: str = "recording.tool.json",
	tool_name: Optional[str] = None,
	tool_description: Optional[str] = None,
	login: bool = False,
) -> Optional[dict]:
	"""
	Record a tool by launching a browser with recording extension.
	
	Args:
		url: Starting URL to navigate to
		output_file: Path to save the recorded tool JSON
		tool_name: Name for the tool (optional)
		tool_description: Description for the tool (optional)
		login: Whether to auto-login before recording
		
	Returns:
		Dictionary with recording metadata or None if failed
	"""
	import os
	
	# Check if running in a headless environment FIRST before any heavy imports
	if not os.environ.get('DISPLAY'):
		raise EnvironmentError(
			"No display found (DISPLAY environment variable not set).\n"
			"The recorder requires a graphical display to show the browser and extension UI.\n\n"
			"Solutions:\n"
			"  1. Run on a machine with a GUI/desktop environment\n"
			"  2. Use X11 forwarding: ssh -X user@host\n"
			"  3. Use Xvfb (virtual display): xvfb-run walt record <url>\n"
			"  4. Use pre-discovered tools: walt list walt-tools/"
		)
	
	# Check if extension exists
	if not EXT_DIR.exists() or not EXT_DIR.is_dir():
		raise FileNotFoundError(
			f"Chrome extension not found at: {EXT_DIR}\n"
			f"The recorder requires a browser extension that is not included in the repository.\n"
			f"Please check the documentation for extension setup instructions."
		)
	
	# Environment checks passed - instantiate the service
	# FastAPI and browser imports happen inside RecordingService methods
	service = RecordingService()
	tool_data = await service.capture_tool(start_url=url, login=login)
	
	if tool_data:
		# Update name and description if provided
		if tool_name:
			tool_data.name = tool_name
		if tool_description:
			tool_data.description = tool_description
			
		# Save to file
		output_path = pathlib.Path(output_file)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		
		with open(output_path, 'w') as f:
			json.dump(tool_data.model_dump(), f, indent=2)
		
		return {
			"success": True,
			"output_file": str(output_path),
			"step_count": len(tool_data.steps) if hasattr(tool_data, 'steps') else 0,
		}
	
	return None


if __name__ == '__main__':
	# This allows running service.py directly for testing
	try:
		asyncio.run(main_service_runner())
	except KeyboardInterrupt:
		print('Service runner interrupted by user.')
