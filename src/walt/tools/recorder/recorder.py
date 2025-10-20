# Example for your new main runner (e.g., in recorder.py or a new script)
import asyncio

from walt.tools.recorder.service import RecordingService  # Adjust import path if necessary


async def run_recording():
	service = RecordingService()
	print('Starting recording session via service...')
	tool_schema = await service.capture_tool()

	if tool_schema:
		print('\n--- MAIN SCRIPT: CAPTURED tool ---')
		try:
			print(tool_schema.model_dump_json(indent=2))
		except AttributeError:
			# Fallback if model_dump_json isn't available (e.g. if it's a dict)
			import json

			print(json.dumps(tool_schema, indent=2))  # Ensure schema is serializable
		print('------------------------------------')
	else:
		print('MAIN SCRIPT: No tool was captured.')


if __name__ == '__main__':
	try:
		asyncio.run(run_recording())
	except KeyboardInterrupt:
		print('Main recording script interrupted.')
	except Exception as e:
		print(f'An error occurred in the main recording script: {e}')
