from typing import Any, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
import logging
import asyncio
import time
from walt.browser_use.custom.evaluators.vwa import image_utils
import json
from playwright.async_api import Page
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_community.callbacks.manager import (
    get_openai_callback,
    get_bedrock_anthropic_callback,
)
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import ValidationError

from walt.browser_use.agent.service import Agent
from walt.browser_use.agent.views import (
    ActionResult,
    AgentError,
    AgentHistory,
    AgentHistoryList,
    AgentOutput,
    AgentSettings,
    AgentState,
    AgentStepInfo,
    StepMetadata,
    ToolCallingMethod,
)

from walt.browser_use.utils import time_execution_async
from walt.browser_use.agent.message_manager.utils import save_conversation
from walt.browser_use.telemetry.views import AgentStepTelemetryEvent, AgentEndTelemetryEvent
from walt.browser_use.agent.gif import create_history_gif
from walt.browser_use.agent.prompts import AgentMessagePrompt, PlannerPrompt, SystemPrompt
from walt.browser_use.agent.message_manager.utils import (
    convert_input_messages,
    extract_json_from_model_output,
    save_conversation,
)
from walt.browser_use.controller.registry.views import ActionModel
from walt.browser_use.custom.utils import pil_to_b64

load_dotenv()
logger = logging.getLogger(__name__)


def get_usage(cb):
    return {
        "prompt_tokens": cb.prompt_tokens,
        "prompt_tokens_cached": cb.prompt_tokens_cached,
        "completion_tokens": cb.completion_tokens,
        "reasoning_tokens": cb.reasoning_tokens,
        "total_tokens": cb.total_tokens,
        "total_cost": cb.total_cost,
    }


def log_response(response: AgentOutput, step: str) -> None:
    """Utility function to log the model's response."""

    if "Success" in response.current_state.evaluation_previous_goal:
        emoji = "👍"
    elif "Failed" in response.current_state.evaluation_previous_goal:
        emoji = "⚠"
    else:
        emoji = "🤷"
    res_str = ""
    res_str += f"📍 Step {step}\n\n"
    logger.info(f"{emoji} Eval: {response.current_state.evaluation_previous_goal}")
    res_str += f"{emoji} Eval: {response.current_state.evaluation_previous_goal}\n"
    logger.info(f"🧠 Memory: {response.current_state.memory}")
    res_str += f"🧠 Memory: {response.current_state.memory}\n"
    logger.info(f"🎯 Next goal: {response.current_state.next_goal}")
    res_str += f"🎯 Next goal: {response.current_state.next_goal}\n"
    for i, action in enumerate(response.action):
        logger.info(
            f"🛠️  Action {i + 1}/{len(response.action)}: {action.model_dump_json(exclude_unset=True)}"
        )
        res_str += f"🛠️  Action {i + 1}/{len(response.action)}: {action.model_dump_json(exclude_unset=True)}\n"
    return res_str


class AgentWithMessageTracker(Agent):
    def __init__(
        self,
        break_after_intial_actions: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.trajectory_jsonfied = {}
        self.break_after_intial_actions = break_after_intial_actions
        self.total_tabs_opened = 0

    def get_jsonfied_trajectory(self):
        return self.trajectory_jsonfied

    def _add_action_results(
        self, result: list[ActionResult]
    ) -> Dict[str, List[Dict[str, Any]]]:
        controller_message = {"action_result": [], "action_error": []}
        if result:
            for r in result:
                if r.include_in_memory:
                    if r.extracted_content:
                        msg = HumanMessage(
                            content="Action result: " + str(r.extracted_content)
                        )
                        controller_message["action_result"].append(msg.model_dump())
                        controller_message["action_error"].append(None)
                    if r.error:
                        last_line = r.error.split("\n")[-1]
                        msg = HumanMessage(content="Action error: " + last_line)
                        try:
                            controller_message["action_error"][-1] = msg.model_dump()
                        except IndexError:
                            # error when the list is empty
                            controller_message["action_error"].append(msg.model_dump())

        return controller_message

    async def _run_planner(self):
        """Run the planner to analyze state and suggest next steps"""
        # Skip planning if no planner_llm is set
        if not self.settings.planner_llm:
            return None

        # Create planner message history using full message history
        planner_messages = [
            PlannerPrompt(
                self.controller.registry.get_prompt_description()
            ).get_system_message(),
            *self._message_manager.get_messages()[
                1:
            ],  # Use full message history except the first (system message)
        ]

        if not self.settings.use_vision_for_planner and self.settings.use_vision:
            last_state_message: HumanMessage = planner_messages[-1]
            # remove image from last state message
            new_msg = ""
            if isinstance(last_state_message.content, list):
                for msg in last_state_message.content:
                    if msg["type"] == "text":  # type: ignore
                        new_msg += msg["text"]  # type: ignore
                    elif msg["type"] == "image_url":  # type: ignore
                        continue  # type: ignore
            else:
                new_msg = last_state_message.content

            planner_messages[-1] = HumanMessage(content=new_msg)

        planner_messages = convert_input_messages(
            planner_messages, self.planner_model_name
        )

        # Get planner output
        provider = self.llm.__class__.__name__
        if "openai" in provider.lower():
            callback_context = get_openai_callback()
            with callback_context as cb:
                response = await self.settings.planner_llm.ainvoke(planner_messages)
            usage = get_usage(cb)
        else:
            response = await self.settings.planner_llm.ainvoke(planner_messages)
            if "usage_metadata" in response.content:
                usage = response.content["usage_metadata"]
            else:
                usage = {}
        logger.info(f"Planner Usage: {usage}")
        plan = str(response.content)
        # if deepseek-reasoner, remove think tags
        if self.planner_model_name == "deepseek-reasoner":
            plan = self._remove_think_tags(plan)
        try:
            plan_json = json.loads(plan)
            logger.info(f"Planning Analysis:\n{json.dumps(plan_json, indent=4)}")
        except json.JSONDecodeError:
            logger.info(f"Planning Analysis:\n{plan}")
        except Exception as e:
            logger.debug(f"Error parsing planning analysis: {e}")
            logger.info(f"Plan: {plan}")

        return plan, usage

    @time_execution_async("--get_next_action (agent)")
    async def get_next_action(
        self, input_messages: list[BaseMessage]
    ) -> Union[AgentOutput, Dict[str, Any]]:
        """Get next action from LLM based on current state"""

        input_messages = self._convert_input_messages(input_messages)
        provider = self.llm.__class__.__name__
        if self.tool_calling_method == "raw":
            if "openai" in provider.lower():
                with get_openai_callback() as cb:
                    output = self.llm.invoke(input_messages)
                usage = get_usage(cb)
                logger.info(f"Get Next Action Usage: {usage}")
            else:
                output = self.llm.invoke(input_messages)
                if "usage_metadata" in output.content:
                    usage = output.content["usage_metadata"]
                else:
                    usage = {}
            # TODO: currently invoke does not return reasoning_content, we should override invoke
            output.content = self._remove_think_tags(str(output.content))
            try:
                parsed_json = extract_json_from_model_output(output.content)
                parsed = self.AgentOutput(**parsed_json)
            except (ValueError, ValidationError) as e:
                logger.warning(f"Failed to parse model output: {output} {str(e)}")
                raise ValueError("Could not parse response.")

        elif self.tool_calling_method is None:
            structured_llm = self.llm.with_structured_output(
                self.AgentOutput, include_raw=True
            )
            if "openai" in provider.lower():
                with get_openai_callback() as cb:
                    response: dict[str, Any] = await structured_llm.ainvoke(input_messages)  # type: ignore
                usage = get_usage(cb)
                logger.info(f"Get Next Action Usage: {usage}")
            else:
                response: dict[str, Any] = await structured_llm.ainvoke(input_messages)  # type: ignore
                usage = {}
            parsed: AgentOutput | None = response["parsed"]
        else:
            structured_llm = self.llm.with_structured_output(
                self.AgentOutput, include_raw=True, method=self.tool_calling_method
            )
            if "openai" in provider.lower():
                with get_openai_callback() as cb:
                    response: dict[str, Any] = await structured_llm.ainvoke(input_messages)  # type: ignore
                usage = get_usage(cb)
                logger.info(f"Get Next Action Usage: {usage}")
            else:
                response: dict[str, Any] = await structured_llm.ainvoke(input_messages)  # type: ignore
                usage = {}
            parsed: AgentOutput | None = response["parsed"]

        if parsed is None:
            logger.warning(f"Failed to parse response: {response}.")
            raise ValueError(f"Could not parse response.")

        # cut the number of actions to max_actions_per_step if needed
        if len(parsed.action) > self.settings.max_actions_per_step:
            parsed.action = parsed.action[: self.settings.max_actions_per_step]

        logged_res = log_response(parsed, self.state.n_steps)

        return parsed, usage, logged_res

    @time_execution_async("--multi-act (agent)")
    async def multi_act(
        self,
        actions: list[ActionModel],
        check_for_new_elements: bool = True,
    ) -> list[ActionResult]:
        """Execute multiple actions"""
        results = []

        cached_selector_map = await self.browser_context.get_selector_map()
        cached_path_hashes = set(
            e.hash.branch_path_hash for e in cached_selector_map.values()
        )

        await self.browser_context.remove_highlights()

        for i, action in enumerate(actions):
            if check_for_new_elements:
                if action.get_index() is not None and i != 0:
                    new_state = await self.browser_context.get_state()
                    new_path_hashes = set(
                        e.hash.branch_path_hash for e in new_state.selector_map.values()
                    )
                    if not new_path_hashes.issubset(cached_path_hashes):
                        # next action requires index but there are new elements on the page
                        msg = (
                            f"Something new appeared after action {i} / {len(actions)}"
                        )
                        logger.info(msg)
                        results.append(
                            ActionResult(extracted_content=msg, include_in_memory=True)
                        )
                        break

            await self._raise_if_stopped_or_paused()

            result = await self.controller.act(
                action,
                self.browser_context,
                self.settings.page_extraction_llm,
                self.sensitive_data,
                self.settings.available_file_paths,
                context=self.context,
            )

            results.append(result)

            logger.debug(f"Executed action {i + 1} / {len(actions)}")
            if results[-1].is_done or results[-1].error or i == len(actions) - 1:
                break

            await asyncio.sleep(self.browser_context.config.wait_between_actions)
            # hash all elements. if it is a subset of cached_state its fine - else break (new elements on page)

        return results

    @time_execution_async("--step (agent)")
    async def step(self, step_info: Optional[AgentStepInfo] = None) -> None:
        """Execute one step of the task"""
        # self.state.n_steps starts from 1
        logger.info(f"📍 Step {self.state.n_steps}")
        state = None
        model_output = None
        result: list[ActionResult] = []
        step_start_time = time.time()
        tokens = 0
        logged_res = ""
        self.trajectory_jsonfied[self.state.n_steps] = {}

        try:
            state = await self.browser_context.get_state()

            await self._raise_if_stopped_or_paused()

            self._message_manager.add_state_message(
                state, self.state.last_result, step_info, self.settings.use_vision
            )

            # Run planner at specified intervals if planner is configured
            if (
                self.settings.planner_llm
                and self.state.n_steps % self.settings.planner_interval == 0
            ):
                # llm call
                plan, planner_usage = await self._run_planner()
                # add plan before last state message
                self._message_manager.add_plan(plan, position=-1)
                self.trajectory_jsonfied[self.state.n_steps]["get_plan"] = {
                    "plan": plan,
                    "usage": planner_usage,
                }

            if step_info and step_info.is_last_step():
                # Add last step warning if needed
                msg = 'Now comes your last step. Use only the "done" action now. No other actions - so here your action sequence musst have length 1.'
                msg += '\nIf the task is not yet fully finished as requested by the user, set success in "done" to false! E.g. if not all steps are fully completed.'
                msg += '\nIf the task is fully finished, set success in "done" to true.'
                msg += "\nInclude everything you found out for the ultimate task in the done text."
                logger.info("Last step finishing up")
                self._message_manager._add_message_with_tokens(
                    HumanMessage(content=msg)
                )
                self.AgentOutput = self.DoneAgentOutput

            input_messages = self._message_manager.get_messages()
            # cumulative input tokens
            tokens = self._message_manager.state.history.current_tokens
            self.trajectory_jsonfied[self.state.n_steps]["input_messages"] = {
                "contents": [m.model_dump() for m in input_messages],
                # as a backup in case the callback context is not available
                "token_count": sum(
                    [self.message_manager._count_tokens(m) for m in input_messages]
                ),
                "cumulative_input_token_count": tokens,
            }

            try:
                # llm call
                model_output, get_next_action_usage, logged_res = (
                    await self.get_next_action(input_messages)
                )
                self.trajectory_jsonfied[self.state.n_steps]["get_next_action"] = {
                    "usage": get_next_action_usage,
                }

                self.state.n_steps += 1

                if self.register_new_step_callback:
                    await self.register_new_step_callback(
                        state, model_output, self.state.n_steps
                    )

                if self.settings.save_conversation_path:
                    target = (
                        self.settings.save_conversation_path
                        + f"_{self.state.n_steps}.txt"
                    )
                    save_conversation(
                        input_messages,
                        model_output,
                        target,
                        self.settings.save_conversation_path_encoding,
                    )

                self._message_manager._remove_last_state_message()  # we dont want the whole state in the chat history

                await self._raise_if_stopped_or_paused()

                self._message_manager.add_model_output(model_output)

                message_history = self.message_manager.get_messages()
                # the self.state.n_steps was increased after the get_next_action call
                self.trajectory_jsonfied[self.state.n_steps - 1]["output_messages"] = {
                    # contains state description + action prediction
                    "tool_call_message": message_history[-2].model_dump(),
                    # just empty response
                    "tool_response": message_history[-1].model_dump(),
                    # as a backup in case the callback context is not available
                    "token_count": sum(
                        [
                            self.message_manager._count_tokens(m)
                            for m in message_history[-2:]
                        ]
                    ),
                }
            except Exception as e:
                # model call failed, remove last state message from history
                self._message_manager._remove_last_state_message()
                raise e

            result: list[ActionResult] = await self.multi_act(
                model_output.action, check_for_new_elements=False
            )

            # check if the if total number of tabs have changed after the action
            await self.browser_context._wait_for_page_and_frames_load()
            session = await self.browser_context.get_session()
            if len(session.context.pages) != self.total_tabs_opened:
                if len(session.context.pages) > self.total_tabs_opened:
                    logger.info("New tab opened, switching to it")
                    page = session.context.pages[-1]
                    # this will write to self.browser_context.session.current_page
                    session.current_page = page
                self.total_tabs_opened = len(session.context.pages)
                logger.info("resetting the total tabs opened")
                # sleep for 1 second to avoid busy-waiting
                await asyncio.sleep(1)
                logger.info(f"Total tabs opened after action: {self.total_tabs_opened}")

            self.state.last_result = result

            if len(result) > 0 and result[-1].is_done:
                logger.info(f"📄 Result: {result[-1].extracted_content}")
                logged_res += f"📄 Result: {result[-1].extracted_content}\n"

            self.state.consecutive_failures = 0

        except InterruptedError:
            logger.debug("Agent paused")
            self.state.last_result = [
                ActionResult(
                    error="The agent was paused - now continuing actions might need to be repeated",
                    include_in_memory=True,
                )
            ]
            return
        except Exception as e:
            result = await self._handle_step_error(e)
            self.state.last_result = result

        finally:
            # the self.state.n_steps was increased after the get_next_action call
            self.trajectory_jsonfied[self.state.n_steps - 1]["controller_messages"] = (
                self._add_action_results(result)
            )
            step_end_time = time.time()
            actions = (
                [a.model_dump(exclude_unset=True) for a in model_output.action]
                if model_output
                else []
            )
            self.telemetry.capture(
                AgentStepTelemetryEvent(
                    agent_id=self.state.agent_id,
                    step=self.state.n_steps,
                    actions=actions,
                    consecutive_failures=self.state.consecutive_failures,
                    step_error=(
                        [r.error for r in result if r.error]
                        if result
                        else ["No result"]
                    ),
                )
            )
            if not result:
                return

            if state:
                metadata = StepMetadata(
                    step_number=self.state.n_steps,
                    step_start_time=step_start_time,
                    step_end_time=step_end_time,
                    input_tokens=tokens,
                )
                self._make_history_item(model_output, state, result, metadata)
            return logged_res, self.state.history.is_done()

    @time_execution_async("--run (agent)")
    async def run(self, max_steps: int = 100) -> Tuple[AgentHistoryList, Page]:
        """Execute the task with maximum number of steps"""
        try:
            self._log_agent_run()

            # Execute initial actions if provided
            if self.initial_actions:
                result = await self.multi_act(
                    self.initial_actions, check_for_new_elements=False
                )
                self.trajectory_jsonfied[0] = {
                    "controller_messages": self._add_action_results(result)
                }
                self.state.last_result = result
                if self.break_after_intial_actions:
                    logger.info("🏁 Finished initial actions and exit.")
                    return self.state.history, self.browser_context.session.current_page

            # track the total tabs opened after the initial actions
            try:
                self.total_tabs_opened = len(self.browser_context.session.context.pages)
                logger.info(
                    f"Total tabs opened after initial actions: {self.total_tabs_opened}"
                )
            except Exception as e:
                logger.info(
                    f"Error getting total tabs opened after initial actions: {e}; likely there is no browser context as initial actions are not provided"
                )
                self.total_tabs_opened = 0
            for step in range(max_steps):
                # Check if we should stop due to too many failures
                if self.state.consecutive_failures >= self.settings.max_failures:
                    logger.error(
                        f"❌ Stopping due to {self.settings.max_failures} consecutive failures"
                    )
                    break

                # Check control flags before each step
                if self.state.stopped:
                    logger.info("Agent stopped")
                    break

                while self.state.paused:
                    await asyncio.sleep(0.2)  # Small delay to prevent CPU spinning
                    if self.state.stopped:  # Allow stopping while paused
                        break

                step_info = AgentStepInfo(step_number=step, max_steps=max_steps)
                await self.step(step_info)

                if self.state.history.is_done():
                    if self.settings.validate_output and step < max_steps - 1:
                        if not await self._validate_output():
                            continue

                    await self.log_completion()
                    result, rationale = await self.judge_task()
                    if result:
                        logger.info(f"📋 Judge reasoning: {rationale[:200]}..." if len(rationale) > 200 else f"📋 Judge reasoning: {rationale}")
                        break
                    else:
                        logger.warning(f"📋 Judge feedback: {rationale[:200]}..." if len(rationale) > 200 else f"📋 Judge feedback: {rationale}")
                        last_result = self.state.history.history[-1].result[-1]
                        last_result.is_done = False
                        last_result.success = False
                        # Add judge rationale to the agent's context
                        last_result.extracted_content = f"Your final answer of ``{last_result.extracted_content}'' has been judged as likely incorrect with the following feedback: {rationale}. Carefully review the feedback to identify any potential mistakes in your previous steps and try again."

            # Check if we hit the step limit without completing
            if step == max_steps - 1 and not self.state.history.is_done():
                logger.info("❌ Failed to complete task in maximum steps")

            current_page = self.browser_context.session.current_page
            return self.state.history, current_page
        finally:
            logger.debug(
                f"Agent {self.state.agent_id} finished with {self.state.n_steps} steps using {self.state.history.total_input_tokens()} input tokens in {self.state.history.total_duration_seconds()} seconds"
            )
            self.telemetry.capture(
                AgentEndTelemetryEvent(
                    agent_id=self.state.agent_id,
                    is_done=self.state.history.is_done(),
                    success=self.state.history.is_successful(),
                    steps=self.state.n_steps,
                    max_steps_reached=self.state.n_steps >= max_steps,
                    errors=self.state.history.errors(),
                    total_input_tokens=self.state.history.total_input_tokens(),
                    total_duration_seconds=self.state.history.total_duration_seconds(),
                )
            )

            if not self.injected_browser_context:
                await self.browser_context.close()

            if not self.injected_browser and self.browser:
                await self.browser.close()

            if self.settings.generate_gif:
                output_path: str = "agent_history.gif"
                if isinstance(self.settings.generate_gif, str):
                    output_path = self.settings.generate_gif

                create_history_gif(
                    task=self.task, history=self.state.history, output_path=output_path
                )

    async def _execute_history_step(
        self, history_item: AgentHistory, delay: float
    ) -> list[ActionResult]:
        """Execute a single step from history with element validation"""
        state = await self.browser_context.get_state()
        if not state or not history_item.model_output:
            raise ValueError("Invalid state or model output")
        updated_actions = []
        for i, action in enumerate(history_item.model_output.action):
            updated_action = await self._update_action_indices(
                history_item.state.interacted_element[i],
                action,
                state,
            )
            updated_actions.append(updated_action)

            if updated_action is None:
                raise ValueError(f"Could not find matching element {i} in current page")

        result = await self.multi_act(updated_actions, check_for_new_elements=False)
        # check if the if total number of tabs have changed after the action
        await self.browser_context._wait_for_page_and_frames_load()
        session = await self.browser_context.get_session()
        if len(session.context.pages) != self.total_tabs_opened:
            if len(session.context.pages) > self.total_tabs_opened:
                logger.info("New tab opened, switching to it")
                page = session.context.pages[-1]
                # this will write to self.browser_context.session.current_page
                session.current_page = page
            self.total_tabs_opened = len(session.context.pages)
            logger.info("resetting the total tabs opened")
            # sleep for 1 second to avoid busy-waiting
            await asyncio.sleep(1)
            logger.info(f"Total tabs opened after action: {self.total_tabs_opened}")

        await asyncio.sleep(delay)
        return result

    async def rerun_history(
        self,
        history: AgentHistoryList,
        max_retries: int = 3,
        skip_failures: bool = True,
        delay_between_actions: float = 2.0,
    ) -> list[ActionResult]:
        """
        Rerun a saved history of actions with error handling and retry logic.

        Args:
                history: The history to replay
                max_retries: Maximum number of retries per action
                skip_failures: Whether to skip failed actions or stop execution
                delay_between_actions: Delay between actions in seconds

        Returns:
                List of action results
        """
        # Execute initial actions if provided
        if self.initial_actions:
            result = await self.multi_act(self.initial_actions)
            self.state.last_result = result
        try:
            self.total_tabs_opened = len(self.browser_context.session.context.pages)
            logger.info(
                f"Total tabs opened after initial actions: {self.total_tabs_opened}"
            )
        except Exception as e:
            logger.error(f"Error getting total tabs opened after initial actions: {e}")
            self.total_tabs_opened = 0

        results = []

        for i, history_item in enumerate(history.history):
            goal = (
                history_item.model_output.current_state.next_goal
                if history_item.model_output
                else ""
            )
            logger.info(f"Replaying step {i + 1}/{len(history.history)}: goal: {goal}")

            if (
                not history_item.model_output
                or not history_item.model_output.action
                or history_item.model_output.action == [None]
            ):
                logger.warning(f"Step {i + 1}: No action to replay, skipping")
                results.append(ActionResult(error="No action to replay"))
                continue

            retry_count = 0
            while retry_count < max_retries:
                try:
                    result = await self._execute_history_step(
                        history_item, delay_between_actions
                    )
                    results.extend(result)
                    break

                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        error_msg = f"Step {i + 1} failed after {max_retries} attempts: {str(e)}"
                        logger.error(error_msg)
                        if not skip_failures:
                            results.append(ActionResult(error=error_msg))
                            raise RuntimeError(error_msg)
                    else:
                        logger.warning(
                            f"Step {i + 1} failed (attempt {retry_count}/{max_retries}), retrying..."
                        )
                        await asyncio.sleep(delay_between_actions)

        return results


class AgentWithCustomPlanner(AgentWithMessageTracker):
    def __init__(self, *args: Any, **kwargs: Any):
        self.planner_inputs = kwargs.pop("planner_inputs", {})
        self.planner_config = kwargs.pop("planner_config", {})
        # default to true
        if "include_init_state_observation" not in self.planner_inputs:
            self.planner_inputs["include_init_state_observation"] = True
        super().__init__(*args, **kwargs)
        self.trajectory_jsonfied = {}
        self.plan_history = []

    def _flatten_messages(
        self, messages: list[BaseMessage], omit_system_message: bool = False
    ) -> str:
        text = ""
        for msg in messages:
            role = msg.type
            if omit_system_message and role == "system":
                text += (
                    "\n"
                    + "-" * 50
                    + f"{role}"
                    + "-" * 50
                    + "\n"
                    + "<...system message omitted...>"
                    + "\n"
                )
                continue
            content = msg.content
            if isinstance(content, str):
                if content != "":
                    if len(content) <= 300:
                        text += "\n" + "-" * 50 + f"{role}" + "-" * 50 + "\n" + content
                    else:
                        text += (
                            "\n"
                            + "-" * 50
                            + f"{role}"
                            + "-" * 50
                            + "\n"
                            + content[:300]
                            + "\n<...truncated for brevity...>\n"
                        )
                else:
                    if role == "tool":
                        text += (
                            "\n"
                            + "-" * 50
                            + f"{role}"
                            + "-" * 50
                            + "\n"
                            + f"tool call id: {msg.tool_call_id}"
                        )
                    elif role == "ai":
                        agent_output = msg.tool_calls[0]["args"]
                        text += (
                            "\n" + "-" * 50 + f"{role}" + "-" * 50 + f"\n{agent_output}"
                        )
            elif isinstance(content, list):
                text += "\n" + "-" * 50 + f"{role}" + "-" * 50 + "\n"
                for item in content:
                    if item["type"] == "text":
                        if len(item["text"]) <= 300:
                            text += item["text"]
                        else:
                            text += (
                                item["text"][:300] + "\n<...truncated for brevity...>\n"
                            )
                    elif item["type"] == "image_url":
                        text += f"\n[screenshot placeholder]\n"
            else:
                continue
        return text

    async def _run_planner(self, verify=False):
        if not self.settings.planner_llm:
            return None
        from walt.prompts import get_planner_prompt

        planner_prompt = get_planner_prompt(use_tools=self.expose_tool_actions)
        
        # If using tools, append the actual action descriptions
        if self.expose_tool_actions:
            action_descriptions = (
                self.controller.registry.registry.get_high_level_action_description()
            )
            planner_prompt = planner_prompt + f"\n\n## These are the actions available to the agent executing your plan: \n{action_descriptions}."
        
        system_message = SystemMessage(content=planner_prompt)
        retrived_narrative_memory_list = self.planner_inputs.get(
            "retrived_narrative_memory", []
        )
        task_description = f"\nYour ultimate task is ```{self.task}```"
        for idx, item in enumerate(retrived_narrative_memory_list):
            if idx == 0:
                task_description += f"\n## Similar task and experience:\n"
            task_description += f"Sample {idx+1}:\nTASK:{item['task']}\nEXPERIENCE:\n{item['experience']}"
        task_description += f"\n\n## This task information\n"
        if self.planner_inputs.get("include_init_state_observation", True):
            planner_messages = [
                system_message,
                HumanMessage(content=task_description),
                *self._message_manager.get_messages()[
                    5:
                ],  # task history memory + browser state + last action result
            ]
        else:
            planner_messages = [
                system_message,
                HumanMessage(content=task_description),
                *self._message_manager.get_messages()[5:-1],
                HumanMessage(content="\n[Task history memory ends]"),
            ]
        if self.plan_history:
            planner_messages.append(
                HumanMessage(
                    content="## previous plan for this task made by you\n"
                    + self.plan_history[-1]
                )
            )
        if not self.settings.use_vision_for_planner and self.settings.use_vision:
            if len(self.plan_history) == 0:
                # if there is no previous plan, the state message is the last
                last_state_message: HumanMessage = planner_messages[-1]
            else:
                # if there is a previous plan, the state message is the second last
                last_state_message: HumanMessage = planner_messages[-2]
            # remove image from last state message
            new_msg = ""
            if isinstance(last_state_message.content, list):
                for msg in last_state_message.content:
                    if msg["type"] == "text":  # type: ignore
                        new_msg += msg["text"]  # type: ignore
                    elif msg["type"] == "image_url":  # type: ignore
                        continue  # type: ignore
            else:
                new_msg = last_state_message.content
            if len(self.plan_history) == 0:
                planner_messages[-1] = HumanMessage(content=new_msg)
            else:
                planner_messages[-2] = HumanMessage(content=new_msg)

        planner_messages = convert_input_messages(
            planner_messages, self.planner_model_name
        )
        # logger.debug(f'Planner messages:\n{self._flatten_messages(planner_messages, omit_system_message=True)}')
        # Get planner output
        provider = self.llm.__class__.__name__
        if "openai" in provider.lower():
            callback_context = get_openai_callback()
            with callback_context as cb:
                response = await self.settings.planner_llm.ainvoke(planner_messages)
            usage = get_usage(cb)
        else:
            response = await self.settings.planner_llm.ainvoke(planner_messages)
            usage = {}
        logger.info(f"Planner Usage: {usage}")
        plan = str(response.content)
        # if deepseek-reasoner, remove think tags
        if self.planner_model_name == "deepseek-reasoner":
            plan = self._remove_think_tags(plan)
        return plan, usage

    @time_execution_async("--step (agent)")
    async def step(self, step_info: Optional[AgentStepInfo] = None) -> None:
        """Execute one step of the task"""
        # self.state.n_steps starts from 1
        logger.info(f"📍 Step {self.state.n_steps}")
        state = None
        model_output = None
        result: list[ActionResult] = []
        step_start_time = time.time()
        tokens = 0
        self.trajectory_jsonfied[self.state.n_steps] = {}
        # import pdb; pdb.set_trace()
        try:
            state = await self.browser_context.get_state()

            await self._raise_if_stopped_or_paused()

            self._message_manager.add_state_message(
                state, self.state.last_result, step_info, self.settings.use_vision
            )

            # Run planner at specified intervals
            # if (self.state.n_steps - 1) % self.settings.planner_interval == 0:
            condtion_1 = self.state.n_steps == 1
            if self.state.last_result:
                condtion_2 = (
                    self.state.last_result[-1].extracted_content
                    == "Re-plan signal received"
                )
            else:
                condtion_2 = False
            condtion_3 = (self.state.n_steps - 1) % self.settings.planner_interval == 0

            flag = condtion_1 or condtion_2 or condtion_3
            if self.settings.planner_llm and flag:
                # llm call
                plan, planner_usage = await self._run_planner()
                plan = (
                    "\nBelow is the plan for the current task:\n[plan start]\n"
                    + plan
                    + "\n[plan end]"
                )
                # add plan before last state message
                self.plan_history.append(plan)
                # add the latest plan before the latest state message
                self._message_manager.add_plan(plan, position=-1, as_ai_message=False)
                self.trajectory_jsonfied[self.state.n_steps]["get_plan"] = {
                    "plan": plan,
                    "usage": planner_usage,
                }
                logger.info(f"🗒️ Planner:\n{plan}")
            # not the time to make a new plan; add the last plan again back to the history for the action prediction
            elif self.settings.planner_llm:
                logger.info(
                    "🗒️ Planner: Not the time to make a new plan, so the last plan will be used again"
                )
                self._message_manager.add_plan(
                    self.plan_history[-1], position=-1, as_ai_message=False
                )

            if step_info and step_info.is_last_step():
                # Add last step warning if needed
                msg = 'Now comes your last step. Use only the "done" action now. No other actions - so here your action sequence musst have length 1.'
                msg += '\nIf the task is not yet fully finished as requested by the user, set success in "done" to false! E.g. if not all steps are fully completed.'
                msg += '\nIf the task is fully finished, set success in "done" to true.'
                msg += "\nInclude everything you found out for the ultimate task in the done text."
                logger.info("Last step finishing up")
                self._message_manager._add_message_with_tokens(
                    HumanMessage(content=msg)
                )
                self.AgentOutput = self.DoneAgentOutput

            input_messages = self._message_manager.get_messages()
            # logger.debug(f'Input messages:\n{self._flatten_messages(input_messages, omit_system_message=True)}')
            # cumulative input tokens
            tokens = self._message_manager.state.history.current_tokens
            self.trajectory_jsonfied[self.state.n_steps]["input_messages"] = {
                "contents": [m.model_dump() for m in input_messages],
                # as a backup in case the callback context is not available
                "token_count": sum(
                    [self.message_manager._count_tokens(m) for m in input_messages]
                ),
                "cumulative_input_token_count": tokens,
            }
            try:
                # llm call
                model_output, get_next_action_usage, logged_res = (
                    await self.get_next_action(input_messages)
                )
                self.trajectory_jsonfied[self.state.n_steps]["get_next_action"] = {
                    "usage": get_next_action_usage,
                }

                self.state.n_steps += 1

                if self.register_new_step_callback:
                    await self.register_new_step_callback(
                        state, model_output, self.state.n_steps
                    )

                if self.settings.save_conversation_path:
                    target = (
                        self.settings.save_conversation_path
                        + f"_{self.state.n_steps}.txt"
                    )
                    save_conversation(
                        input_messages,
                        model_output,
                        target,
                        self.settings.save_conversation_path_encoding,
                    )

                self._message_manager._remove_last_state_message()  # we dont want the whole state in the chat history
                self._message_manager._remove_last_plan()  # we dont want the last plan in the chat history, we will add it agian in the next step
                await self._raise_if_stopped_or_paused()

                self._message_manager.add_model_output(model_output)

                message_history = self.message_manager.get_messages()
                # the self.state.n_steps was increased after the get_next_action call
                self.trajectory_jsonfied[self.state.n_steps - 1]["output_messages"] = {
                    # contains state description + action prediction
                    "tool_call_message": message_history[-2].model_dump(),
                    # just empty response
                    "tool_response": message_history[-1].model_dump(),
                    # as a backup in case the callback context is not available
                    "token_count": sum(
                        [
                            self.message_manager._count_tokens(m)
                            for m in message_history[-2:]
                        ]
                    ),
                }
            except Exception as e:
                # model call failed, remove last state message from history
                self._message_manager._remove_last_state_message()
                self._message_manager._remove_last_plan()  # we dont want the last plan in the chat history, we will add it agian in the next step
                raise e

            result: list[ActionResult] = await self.multi_act(
                model_output.action, check_for_new_elements=False
            )

            # check if the if total number of tabs have changed after the action
            await self.browser_context._wait_for_page_and_frames_load()
            session = await self.browser_context.get_session()
            if len(session.context.pages) != self.total_tabs_opened:
                if len(session.context.pages) > self.total_tabs_opened:
                    logger.info("New tab opened, switching to it")
                    page = session.context.pages[-1]
                    # this will write to self.browser_context.session.current_page
                    session.current_page = page
                self.total_tabs_opened = len(session.context.pages)
                logger.info("resetting the total tabs opened")
                # sleep for 1 second to avoid busy-waiting
                await asyncio.sleep(1)
                logger.info(f"Total tabs opened after action: {self.total_tabs_opened}")

            self.state.last_result = result

            if len(result) > 0 and result[-1].is_done:
                logger.info(f"📄 Result: {result[-1].extracted_content}")

            self.state.consecutive_failures = 0

        except InterruptedError:
            logger.debug("Agent paused")
            self.state.last_result = [
                ActionResult(
                    error="The agent was paused - now continuing actions might need to be repeated",
                    include_in_memory=True,
                )
            ]
            return
        except Exception as e:
            result = await self._handle_step_error(e)
            self.state.last_result = result

        finally:
            # the self.state.n_steps was increased after the get_next_action call
            if (self.state.n_steps - 1) not in self.trajectory_jsonfied:
                self.trajectory_jsonfied[self.state.n_steps - 1] = {}
            self.trajectory_jsonfied[self.state.n_steps - 1]["controller_messages"] = (
                self._add_action_results(result)
            )
            step_end_time = time.time()
            actions = (
                [a.model_dump(exclude_unset=True) for a in model_output.action]
                if model_output
                else []
            )
            self.telemetry.capture(
                AgentStepTelemetryEvent(
                    agent_id=self.state.agent_id,
                    step=self.state.n_steps,
                    actions=actions,
                    consecutive_failures=self.state.consecutive_failures,
                    step_error=(
                        [r.error for r in result if r.error]
                        if result
                        else ["No result"]
                    ),
                )
            )
            if not result:
                return

            if state:
                metadata = StepMetadata(
                    step_number=self.state.n_steps,
                    step_start_time=step_start_time,
                    step_end_time=step_end_time,
                    input_tokens=tokens,
                )
                self._make_history_item(model_output, state, result, metadata)


class AgentOnAthena(AgentWithCustomPlanner):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    @time_execution_async("--step (agent)")
    async def step(self, step_info: Optional[AgentStepInfo] = None):
        """Execute one step of the task"""
        # self.state.n_steps starts from 1
        logger.info(f"📍 Step {self.state.n_steps}")
        state = None
        model_output = None
        result: list[ActionResult] = []
        step_start_time = time.time()
        tokens = 0
        planner_res_str = ""
        self.trajectory_jsonfied[self.state.n_steps] = {}
        try:
            state = await self.browser_context.get_state()

            await self._raise_if_stopped_or_paused()

            self._message_manager.add_state_message(
                state, self.state.last_result, step_info, self.settings.use_vision
            )

            # Run planner at specified intervals
            # if (self.state.n_steps - 1) % self.settings.planner_interval == 0:
            condtion_1 = self.state.n_steps == 1
            if self.state.last_result:
                condtion_2 = (
                    self.state.last_result[-1].extracted_content
                    == "Re-plan signal received"
                )
            else:
                condtion_2 = False
            condtion_3 = (self.state.n_steps - 1) % self.settings.planner_interval == 0
            flag = condtion_1 or condtion_2 or condtion_3
            if self.settings.planner_llm and flag:
                # llm call
                plan, planner_usage = await self._run_planner()
                plan = (
                    "\nBelow is the plan for the current task:\n[plan start]\n"
                    + plan
                    + "\n[plan end]"
                )
                # add plan before last state message
                self.plan_history.append(plan)
                # add the latest plan before the latest state message
                self._message_manager.add_plan(plan, position=-1, as_ai_message=False)
                self.trajectory_jsonfied[self.state.n_steps]["get_plan"] = {
                    "plan": plan,
                    "usage": planner_usage,
                }
                # logger.info(f'🗒️ Planner:\n{plan}')
                planner_res_str += f"📍 Step {self.state.n_steps}\n\n"
                planner_res_str += f"🗒️ Planner:\n{plan}\n"
                self.state.n_steps += 1

            # not the time to make a new plan; add the last plan again back to the history for the action prediction
            elif self.settings.planner_llm:
                logger.info(
                    "🗒️ Planner: Not the time to make a new plan, so the last plan will be used again"
                )
                self._message_manager.add_plan(
                    self.plan_history[-1], position=-1, as_ai_message=False
                )

            if step_info and step_info.is_last_step():
                # Add last step warning if needed
                msg = 'Now comes your last step. Use only the "done" action now. No other actions - so here your action sequence musst have length 1.'
                msg += '\nIf the task is not yet fully finished as requested by the user, set success in "done" to false! E.g. if not all steps are fully completed.'
                msg += '\nIf the task is fully finished, set success in "done" to true.'
                msg += "\nInclude everything you found out for the ultimate task in the done text."
                logger.info("Last step finishing up")
                self._message_manager._add_message_with_tokens(
                    HumanMessage(content=msg)
                )
                self.AgentOutput = self.DoneAgentOutput

            input_messages = self._message_manager.get_messages()
            # logger.debug(f'Input messages:\n{self._flatten_messages(input_messages, omit_system_message=True)}')
            # cumulative input tokens
            tokens = self._message_manager.state.history.current_tokens
            self.trajectory_jsonfied[self.state.n_steps]["input_messages"] = {
                "contents": [m.model_dump() for m in input_messages],
                # as a backup in case the callback context is not available
                "token_count": sum(
                    [self.message_manager._count_tokens(m) for m in input_messages]
                ),
                "cumulative_input_token_count": tokens,
            }
            if planner_res_str == "":
                try:
                    # llm call
                    model_output, get_next_action_usage, logged_res = (
                        await self.get_next_action(input_messages)
                    )
                    self.trajectory_jsonfied[self.state.n_steps]["get_next_action"] = {
                        "usage": get_next_action_usage,
                    }

                    self.state.n_steps += 1

                    if self.register_new_step_callback:
                        await self.register_new_step_callback(
                            state, model_output, self.state.n_steps
                        )

                    if self.settings.save_conversation_path:
                        target = (
                            self.settings.save_conversation_path
                            + f"_{self.state.n_steps}.txt"
                        )
                        save_conversation(
                            input_messages,
                            model_output,
                            target,
                            self.settings.save_conversation_path_encoding,
                        )

                    self._message_manager._remove_last_state_message()  # we dont want the whole state in the chat history
                    self._message_manager._remove_last_plan()  # we dont want the last plan in the chat history, we will add it agian in the next step
                    await self._raise_if_stopped_or_paused()

                    self._message_manager.add_model_output(model_output)

                    message_history = self.message_manager.get_messages()
                    # the self.state.n_steps was increased after the get_next_action call
                    self.trajectory_jsonfied[self.state.n_steps - 1][
                        "output_messages"
                    ] = {
                        # contains state description + action prediction
                        "tool_call_message": message_history[-2].model_dump(),
                        # just empty response
                        "tool_response": message_history[-1].model_dump(),
                        # as a backup in case the callback context is not available
                        "token_count": sum(
                            [
                                self.message_manager._count_tokens(m)
                                for m in message_history[-2:]
                            ]
                        ),
                    }
                except Exception as e:
                    # model call failed, remove last state message from history
                    self._message_manager._remove_last_state_message()
                    self._message_manager._remove_last_plan()  # we dont want the last plan in the chat history, we will add it agian in the next step
                    raise e

                # logger.debug(f'After remove state and plan messages:\n{self._flatten_messages(self._message_manager.get_messages(), omit_system_message=True)}')

                result: list[ActionResult] = await self.multi_act(
                    model_output.action, check_for_new_elements=False
                )

                # check if the if total number of tabs have changed after the action
                await self.browser_context._wait_for_page_and_frames_load()
                session = await self.browser_context.get_session()
                if len(session.context.pages) != self.total_tabs_opened:
                    if len(session.context.pages) > self.total_tabs_opened:
                        logger.info("New tab opened, switching to it")
                        page = session.context.pages[-1]
                        # this will write to self.browser_context.session.current_page
                        session.current_page = page
                    self.total_tabs_opened = len(session.context.pages)
                    logger.info("resetting the total tabs opened")
                    # sleep for 1 second to avoid busy-waiting
                    await asyncio.sleep(1)
                    logger.info(
                        f"Total tabs opened after action: {self.total_tabs_opened}"
                    )

                self.state.last_result = result

                if len(result) > 0 and result[-1].is_done:
                    logger.info(f"📄 Result: {result[-1].extracted_content}")

                self.state.consecutive_failures = 0

        except InterruptedError:
            logger.debug("Agent paused")
            self.state.last_result = [
                ActionResult(
                    error="The agent was paused - now continuing actions might need to be repeated",
                    include_in_memory=True,
                )
            ]
            print("FAILED HERE")
            return "", False
        except Exception as e:
            result = await self._handle_step_error(e)
            self.state.last_result = result

        finally:
            if planner_res_str != "":
                return planner_res_str, self.state.history.is_done()
            # the self.state.n_steps was increased after the get_next_action call
            if (self.state.n_steps - 1) not in self.trajectory_jsonfied:
                self.trajectory_jsonfied[self.state.n_steps - 1] = {}
            self.trajectory_jsonfied[self.state.n_steps - 1]["controller_messages"] = (
                self._add_action_results(result)
            )
            step_end_time = time.time()
            actions = (
                [a.model_dump(exclude_unset=True) for a in model_output.action]
                if model_output
                else []
            )
            self.telemetry.capture(
                AgentStepTelemetryEvent(
                    agent_id=self.state.agent_id,
                    step=self.state.n_steps,
                    actions=actions,
                    consecutive_failures=self.state.consecutive_failures,
                    step_error=(
                        [r.error for r in result if r.error]
                        if result
                        else ["No result"]
                    ),
                )
            )
            if not result:
                print("RETURN HERE")
                return "", False

            if state:
                metadata = StepMetadata(
                    step_number=self.state.n_steps,
                    step_start_time=step_start_time,
                    step_end_time=step_end_time,
                    input_tokens=tokens,
                )
                self._make_history_item(model_output, state, result, metadata)
            print("SUCCESS? HERE")
            return logged_res, self.state.history.is_done()


from walt.browser_use.custom.message_manager_zoo import PlainMessageManager


class LLMAgent:
    def __init__(
        self,
        llm: BaseChatModel,
        system_prompt: str,
    ):
        self.llm = llm
        self.provider = self.llm.__class__.__name__
        system_message = SystemMessage(content=system_prompt)
        self._message_manager = PlainMessageManager(system_message=system_message)

    async def aget_response(
        self, instruction: str, reset_message_manager: bool = False
    ):
        if reset_message_manager:
            self._message_manager.reset()
        self._message_manager.prepare_llm_input(instruction)
        llm_input = self._message_manager.get_messages()
        if "openai" in self.provider.lower():
            callback_context = get_openai_callback()

            with callback_context as cb:
                response = await self.llm.ainvoke(llm_input)
            usage = get_usage(cb)
        else:
            response = await self.llm.ainvoke(llm_input)
            usage = {}
        return response, usage

    def get_response(self, instruction: str, reset_message_manager: bool = False):
        if reset_message_manager:
            self._message_manager.reset()
        self._message_manager.prepare_llm_input(instruction)
        llm_input = self._message_manager.get_messages()
        if "openai" in self.provider.lower():
            callback_context = get_openai_callback()
            with callback_context as cb:
                response = self.llm.invoke(llm_input)
            usage = get_usage(cb)
        else:
            response = self.llm.invoke(llm_input)
            usage = {}
        return response, usage


from walt.browser_use.custom.message_manager_zoo import MessageManagerWithImages
from walt.browser_use.agent.message_manager.service import MessageManagerSettings
from walt.browser_use.custom.eval_envs.VWA import VWABrowser, VWABrowserContext
from walt.browser_use.controller.service import Controller
from walt.browser_use.telemetry.service import ProductTelemetry
from walt.browser_use.agent.service import Context
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, TypeVar
from walt.browser_use.browser.views import (
    BrowserError,
    BrowserState,
    URLNotAllowedError,
)
from PIL.Image import Image
from walt.browser_use.custom.browser_zoo import BrowserBugFix
from walt.browser_use.custom.browser_context_zoo import BrowserContextBugFix


class VWA_Agent(AgentWithCustomPlanner):
    def __init__(
        self,
        task: str,
        task_image: List[Image] | None,
        llm: BaseChatModel,
        # Optional parameters
        browser: VWABrowser | None = None,
        browser_context: VWABrowserContext | None = None,
        controller: Controller[Context] = Controller(),
        # Initial agent run parameters
        sensitive_data: Optional[Dict[str, str]] = None,
        initial_actions: Optional[List[Dict[str, Dict[str, Any]]]] = None,
        # Cloud Callbacks
        register_new_step_callback: (
            Callable[["BrowserState", "AgentOutput", int], Awaitable[None]] | None
        ) = None,
        register_done_callback: (
            Callable[["AgentHistoryList"], Awaitable[None]] | None
        ) = None,
        register_external_agent_status_raise_error_callback: (
            Callable[[], Awaitable[bool]] | None
        ) = None,
        # Agent settings
        use_vision: bool = True,
        use_vision_for_planner: bool = False,
        save_conversation_path: Optional[str] = None,
        save_conversation_path_encoding: Optional[str] = "utf-8",
        max_failures: int = 2,
        retry_delay: int = 5,
        override_system_message: Optional[str] = None,
        extend_system_message: Optional[str] = None,
        max_input_tokens: int = 128000,
        validate_output: bool = False,
        message_context: Optional[str] = None,
        generate_gif: bool | str = False,
        available_file_paths: Optional[list[str]] = None,
        include_attributes: list[str] = [
            "title",
            "type",
            "name",
            "role",
            "aria-label",
            "placeholder",
            "value",
            "alt",
            "aria-expanded",
            "data-date-format",
        ],
        max_actions_per_step: int = 10,
        tool_calling_method: Optional[ToolCallingMethod] = "auto",
        page_extraction_llm: Optional[BaseChatModel] = None,
        planner_llm: Optional[BaseChatModel] = None,
        planner_interval: int = 1,  # Run planner every N steps
        planner_inputs={},
        planner_config={},
        # Inject state
        injected_agent_state: Optional[AgentState] = None,
        #
        context: Context | None = None,
        expose_api_actions: bool = False,
        expose_multimodal_actions: bool = False,
        expose_tool_actions: bool = False,
    ):
        if page_extraction_llm is None:
            page_extraction_llm = llm

        # Core components
        self.task = task
        self.task_image = task_image
        self.llm = llm
        self.controller = controller
        self.sensitive_data = sensitive_data

        self.settings = AgentSettings(
            use_vision=use_vision,
            use_vision_for_planner=use_vision_for_planner,
            save_conversation_path=save_conversation_path,
            save_conversation_path_encoding=save_conversation_path_encoding,
            max_failures=max_failures,
            retry_delay=retry_delay,
            override_system_message=override_system_message,
            extend_system_message=extend_system_message,
            max_input_tokens=max_input_tokens,
            validate_output=validate_output,
            message_context=message_context,
            generate_gif=generate_gif,
            available_file_paths=available_file_paths,
            include_attributes=include_attributes,
            max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method,
            page_extraction_llm=page_extraction_llm,
            planner_llm=planner_llm,
            planner_interval=planner_interval,
        )

        # Initialize state
        self.state = injected_agent_state or AgentState()

        # Action setup
        self._setup_action_models()
        self._set_browser_use_version_and_source()
        self.initial_actions = (
            self._convert_initial_actions(initial_actions) if initial_actions else None
        )

        # Model setup
        self._set_model_names()

        # for models without tool calling, add available actions to context
        self.available_actions = self.controller.registry.get_prompt_description()

        self.tool_calling_method = self._set_tool_calling_method()
        self.settings.message_context = self._set_message_context()

        # Initialize message manager with state
        self._message_manager = MessageManagerWithImages(
            task=task,
            task_image=task_image,
            system_message=SystemPrompt(
                action_description=self.available_actions,
                max_actions_per_step=self.settings.max_actions_per_step,
                override_system_message=override_system_message,
                extend_system_message=extend_system_message,
            ).get_system_message(),
            settings=MessageManagerSettings(
                max_input_tokens=self.settings.max_input_tokens,
                include_attributes=self.settings.include_attributes,
                message_context=self.settings.message_context,
                sensitive_data=sensitive_data,
                available_file_paths=self.settings.available_file_paths,
            ),
            state=self.state.message_manager_state,
        )

        # Browser setup
        self.injected_browser = browser is not None
        self.injected_browser_context = browser_context is not None
        self.browser = (
            browser
            if browser is not None
            else (None if browser_context else BrowserBugFix())
        )
        if browser_context:
            self.browser_context = browser_context
        elif self.browser:
            self.browser_context = BrowserContextBugFix(
                browser=self.browser, config=self.browser.config.new_context_config
            )
        else:
            self.browser = BrowserBugFix()
            self.browser_context = BrowserContextBugFix(browser=self.browser)

        # Callbacks
        self.register_new_step_callback = register_new_step_callback
        self.register_done_callback = register_done_callback
        self.register_external_agent_status_raise_error_callback = (
            register_external_agent_status_raise_error_callback
        )
        # Context
        self.context = context

        # Telemetry
        self.telemetry = ProductTelemetry()

        if self.settings.save_conversation_path:
            logger.info(
                f"Saving conversation to {self.settings.save_conversation_path}"
            )

        # customizationed
        self.trajectory_jsonfied = {}
        self.break_after_intial_actions = False
        self.total_tabs_opened = 0
        self.planner_inputs = planner_inputs
        self.planner_config = planner_config
        self.plan_history = []
        self.expose_api_actions = expose_api_actions
        self.expose_tool_actions = expose_tool_actions

    async def _run_planner(self, verify=False):
        if not self.settings.planner_llm:
            return None

        from walt.prompts import get_planner_prompt
        
        # Get base planner prompt (tool-aware if tools are available)
        planner_prompt = get_planner_prompt(use_tools=self.expose_tool_actions)
        
        # If using tools, append the actual action descriptions
        if self.expose_tool_actions:
            action_descriptions = (
                self.controller.registry.registry.get_high_level_action_description()
            )
            planner_prompt = planner_prompt + f"\n\n## These are the actions available to the agent executing your plan: \n{action_descriptions}."

        system_message = SystemMessage(content=planner_prompt)
        logger.debug("=" * 80)
        logger.debug("PLANNER SYSTEM MESSAGE:")
        logger.debug("=" * 80)
        logger.debug(planner_prompt)
        logger.debug("=" * 80)

        retrived_narrative_memory_list = self.planner_inputs.get(
            "retrived_narrative_memory", []
        )

        if self.task_image and len(self.task_image) > 0:

            content = [
                {
                    "type": "text",
                    "text": f'Your ultimate task is: """{self.task}""". This task involves image input(s) provided below.',
                }
            ]
            for idx, pil_image in enumerate(self.task_image):
                content.extend(
                    [
                        {"type": "text", "text": f"Input Image {idx + 1}:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": pil_to_b64(pil_image)},
                        },
                    ]
                )
        else:
            content = [
                {"type": "text", "text": f'Your ultimate task is: """{self.task}""".'}
            ]
        similar_task_description = ""
        for idx, item in enumerate(retrived_narrative_memory_list):
            if idx == 0:
                similar_task_description += f"\n## Similar task and experience:\n"
            similar_task_description += f"Sample {idx+1}:\nTASK:{item['task']}\nEXPERIENCE:\n{item['experience']}"
        similar_task_description += f"\n\n## This task information\n"

        content.extend(
            [
                {"type": "text", "text": similar_task_description},
            ]
        )

        planner_messages = [
            system_message,
            HumanMessage(content=content),
            *self._message_manager.get_messages()[
                5:
            ],  # task history memory + browser state + last action result
        ]
        if self.plan_history:
            planner_messages.append(
                HumanMessage(
                    content="## previous plan for this task made by you\n"
                    + self.plan_history[-1]
                )
            )
        if not self.settings.use_vision_for_planner and self.settings.use_vision:
            if len(self.plan_history) == 0:
                # if there is no previous plan, the state message is the last
                last_state_message: HumanMessage = planner_messages[-1]
            else:
                # if there is a previous plan, the state message is the second last
                last_state_message: HumanMessage = planner_messages[-2]
            # remove image from last state message
            new_msg = ""
            if isinstance(last_state_message.content, list):
                for msg in last_state_message.content:
                    if msg["type"] == "text":  # type: ignore
                        new_msg += msg["text"]  # type: ignore
                    elif msg["type"] == "image_url":  # type: ignore
                        continue  # type: ignore
            else:
                new_msg = last_state_message.content
            if len(self.plan_history) == 0:
                planner_messages[-1] = HumanMessage(content=new_msg)
            else:
                planner_messages[-2] = HumanMessage(content=new_msg)
        planner_messages = convert_input_messages(
            planner_messages, self.planner_model_name
        )

        # logger.debug(f'Planner messages:\n{self._flatten_messages(planner_messages, omit_system_message=True)}')
        # Get planner output

        provider = self.llm.__class__.__name__
        if "openai" in provider.lower():
            callback_context = get_openai_callback()
            with callback_context as cb:
                response = await self.settings.planner_llm.ainvoke(planner_messages)
            usage = get_usage(cb)
        else:
            response = await self.settings.planner_llm.ainvoke(planner_messages)
            usage = {}
        logger.info(f"Planner Usage: {usage}")
        plan = str(response.content)
        # if deepseek-reasoner, remove think tags
        if self.planner_model_name == "deepseek-reasoner":
            plan = self._remove_think_tags(plan)
        return plan, usage


class WA_Agent(VWA_Agent):
    def __init__(self, *args: Any, **kwargs: Any):
        if "task_image" in kwargs:
            raise RuntimeError("task_image is not supported for WA_Agent; set to None")
        kwargs["task_image"] = None
        super().__init__(*args, **kwargs)
