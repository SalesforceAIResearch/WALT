"""base class for vwa evaluation"""
import json
import re
import time
import logging
from pathlib import Path
from typing import Any, Optional, Union, TypedDict, Dict
from urllib.parse import urljoin
import numpy as np
import numpy.typing as npt

import evaluate  # type: ignore[import]
import requests
from beartype import beartype
from beartype.door import is_bearable
from dataclasses import dataclass
from nltk.tokenize import word_tokenize  # type: ignore
from PIL import Image
from playwright.async_api import Page
from walt.browser_use.custom.evaluators.vwa import image_utils
from walt.browser_use.custom.evaluators.vwa.utils_sgv import (
    PseudoPage, llm_fuzzy_match, llm_ua_match,
    get_query_text,  # these async functions are called by await eval(xxx)
    get_query_text_lowercase,
    gitlab_get_project_memeber_role,
    reddit_get_latest_comment_content_by_username,
    reddit_get_latest_comment_obj_by_username,
    reddit_get_parent_comment_username_of_latest_comment_by_username,
    reddit_get_post_url,
    shopping_get_latest_order_url,
    shopping_get_num_reviews,
    shopping_get_order_product_name_list,
    shopping_get_order_product_option,
    shopping_get_order_product_quantity,
    shopping_get_product_attributes,
    shopping_get_product_price,
    shopping_get_rating_as_percentage,
    shopping_get_sku_latest_review_author,
    shopping_get_sku_latest_review_rating,
    shopping_get_sku_latest_review_text
)


Observation = str | npt.NDArray[np.uint8]


@dataclass
class Action:
    answer: str

    def __getitem__(self, item):
        # compatbile with the original TypedDict class
        if not isinstance(item, str):
            raise TypeError(f"attribute name must be string, not '{type(item).__name__}'")
        return getattr(self, item)

    def __setitem__(self, key, value):
        # compatbile with the original TypedDict class
        setattr(self, key, value)
        return

    def __contains__(self, key):
        # compatbile with the original TypedDict class
        return hasattr(self, key)


# class StateInfo(TypedDict):
#     observation: dict[str, Observation]
#     info: Dict[str, Any]


# Trajectory = list[Union[Action, StateInfo]]
Trajectory = list[Action]

logger = logging.getLogger("evaluation")


@beartype
class AEvaluator(object):
    def __init__(self, eval_tag: str = "", fuzzy_match_provider: str = "openai") -> None:
        self.eval_tag = eval_tag
        self.fuzzy_match_provider = fuzzy_match_provider

    async def __acall__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page | PseudoPage
    ) -> float:
        raise NotImplementedError

    @staticmethod
    def get_last_action(trajectory: Trajectory) -> Action:
        try:
            last_action = trajectory[-1]
            is_bearable(last_action, Action)
        except Exception:
            raise ValueError(
                "The last element of trajectory should be an action, add a fake stop action if needed"
            )

        return last_action  # type: ignore[return-value]


@beartype
class NumericEvaluator(AEvaluator):
    """Check if the numerical relationship is correct"""

    @staticmethod
    @beartype
    def str_2_int(s: str) -> Optional[int]:
        try:
            s = s.strip()
            if "," in s:
                s = s.replace(",", "")

            return int(s)
        except ValueError:
            # Return None if the string cannot be converted to int
            print(f"[NumericEvaluator error]: Cannot convert {s} to int")
            return None

    @staticmethod
    @beartype
    def compare_inequality(
        value: Union[int, float], inequality: str, tol: float = 1e-8
    ) -> bool:
        """
        Compare a value (int or float) against an inequality string.

        Args:
        - value (int/float): The value to be compared.
        - inequality (str): Inequality in the form of "< 700", ">= 300", etc.
        - tol (float): Tolerance for floating point comparisons.

        Returns:
        - bool: True if the value satisfies the inequality, False otherwise.
        """
        # Extract the operator and the number from the inequality string
        ops = {
            "<=": lambda x, y: x <= y + tol,
            ">=": lambda x, y: x >= y - tol,
            "==": lambda x, y: abs(x - y) <= tol,
            "<": lambda x, y: x < y + tol,
            ">": lambda x, y: x > y - tol,
        }

        for op, func in ops.items():
            if op in inequality:
                _, num = inequality.split(op)
                return func(value, float(num.strip()))

        raise ValueError(f"Invalid inequality string: {inequality}")


@beartype
class StringEvaluator(AEvaluator):
    """Check whether the answer is correct with:
    exact match: the answer is exactly the same as the reference answer
    must include: each phrase in the reference answer must be included in the answer
    fuzzy match: the answer is similar to the reference answer, using LLM judge
    """

    @staticmethod
    def clean_answer(answer: str | None) -> str:
        # Handle None or empty answer gracefully
        if answer is None:
            return ""
        if not isinstance(answer, str):
            answer = str(answer)
            
        if answer.startswith("'") and answer.endswith("'"):
            answer = answer[1:-1]
        elif answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]

        # Normalize dashes to simple dash (matches custom VWA)
        answer = re.sub(r"(\w+)[\u2010-\u2015\u2212-](\w+)", r"\1-\2", answer)

        return answer.lower()

    @staticmethod
    @beartype
    def exact_match(ref: str, pred: Union[str, int]) -> float:
        if isinstance(pred, int):
            pred = str(pred)
        return float(
            StringEvaluator.clean_answer(pred)
            == StringEvaluator.clean_answer(ref)
        )

    @staticmethod
    @beartype
    def _num_safe_match(ref: str, ans: str):
        try:
            ref_float = float(ref)
            ans_float = float(ans)
            return ref_float == ans_float
        except ValueError:
            return ref == ans

    @staticmethod
    @beartype
    def must_include(ref: str, pred: str, tokenize: bool = False) -> float:
        clean_ref = StringEvaluator.clean_answer(ref)
        clean_pred = StringEvaluator.clean_answer(pred)
        # tokenize the answer if the ref is a single word
        # prevent false positive (e.g, 0) # NOTE: from TreeSearchLMs https://github.com/kohjingyu/search-agents/blob/7c35ac9eb7fda663d821449efdfd44d360fd0e18/evaluation_harness/evaluators.py#L160C4-L160C42
        if tokenize and len(clean_ref) == 1 and len(word_tokenize(clean_ref)) == 1:
            tok_pred = word_tokenize(clean_pred)
            return float(clean_ref in tok_pred)
        else:
            return float(clean_ref in clean_pred)

    @staticmethod
    @beartype
    def must_exclude(ref: str, pred: str) -> float:
        """Returns 1 if pred is not in ref, and 0 otherwise"""
        clean_ref = StringEvaluator.clean_answer(ref)
        clean_pred = StringEvaluator.clean_answer(pred)
        # tokenize the answer if the ref is a single word
        # prevent false positive (e.g, 0) # NOTE: from TreeSearchLMs https://github.com/kohjingyu/search-agents/blob/7c35ac9eb7fda663d821449efdfd44d360fd0e18/evaluation_harness/evaluators.py#L177C1-L177C42
        if len(word_tokenize(clean_ref)) == 1:
            tok_pred = word_tokenize(clean_pred)
            return float(clean_ref not in tok_pred)
        else:
            return float(clean_ref not in clean_pred)

    @staticmethod
    @beartype
    def fuzzy_match(ref: str, pred: str, intent: str, fuzzy_match_provider: str = "openai") -> float:
        return llm_fuzzy_match(pred, ref, intent, provider=fuzzy_match_provider)

    @staticmethod
    @beartype
    def ua_match(ref: str, pred: str, intent: str, fuzzy_match_provider: str = "openai") -> float:
        return llm_ua_match(pred, ref, intent, provider=fuzzy_match_provider)

    async def __acall__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page | PseudoPage | None = None
    ) -> float:
        with open(config_file, "r") as f:
            configs = json.load(f)

        last_action = self.get_last_action(trajectory)
        
        # Add debugging for None answer issues
        if last_action is None:
            raise ValueError("get_last_action returned None - no actions found in trajectory")
        if "answer" not in last_action:
            raise ValueError(f"No 'answer' key found in last_action. Available attributes: {dir(last_action)}")
        if last_action["answer"] is None:
            # Log this but don't fail - let clean_answer handle it
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"last_action['answer'] is None. Full last_action: {last_action}")
            
        pred = self.clean_answer(last_action["answer"])

        score = 1.0
        for approach, value in configs["eval"]["reference_answers"].items():
            match approach:
                case "exact_match":
                    score *= self.exact_match(ref=value, pred=pred)
                case "required_values":
                    required_values = value
                    assert isinstance(required_values, list)
                    pred = NumericEvaluator.str_2_int(pred)
                    if pred is None:
                        score = 0.0
                    else:
                        for v in required_values:
                            value_or = v.split(" |OR| ")
                            score *= any(
                                [
                                    NumericEvaluator.compare_inequality(
                                        pred, value
                                    )
                                    for value in value_or
                                ]
                            )
                case "must_include":
                    assert isinstance(value, list)
                    for must_value in value:
                        value_or = must_value.split(" |OR| ")
                        score *= any([self.must_include(ref=v, pred=pred)
                                     for v in value_or])
                case "must_exclude":
                    assert isinstance(value, list)
                    for must_excl_value in value:
                        score *= self.must_exclude(
                            ref=must_excl_value, pred=pred
                        )
                case "one_of":
                    assert isinstance(value, list)
                    found = False
                    for one_of_value in value:
                        one_of_value = self.clean_answer(one_of_value)
                        if one_of_value in pred:
                            found = True
                            break
                    score = score * found
                case "fuzzy_match":
                    intent = configs["intent"]
                    if value == "N/A":
                        # if the instruction only asks the model to generate N/A when encountering an unachievable task
                        # without more concrete reasons
                        score *= self.exact_match(ref=value, pred=pred)
                        # if the instruction also asks the model to generate the reason why the task is unachievable
                        # this should be the default as it will prevent false positive N/A`
                        if score != 1:
                            score = 1.0 * self.ua_match(
                                intent=configs["intent"],
                                ref=configs["eval"]["string_note"],
                                pred=pred,
                            )
                    else:
                        assert isinstance(value, list)
                        for reference in value:
                            score *= self.fuzzy_match(
                                ref=reference, pred=pred, intent=intent, fuzzy_match_provider=self.fuzzy_match_provider
                            )
        return score


@beartype
class StringSoftEvaluator(AEvaluator):
    """Use text generation metrics such as BLEU, ROUGE, etc. to evaluate the answer"""
    m = evaluate.load("rouge")

    async def __acall__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page | PseudoPage | None = None
    ) -> float:
        with open(config_file, "r") as f:
            configs = json.load(f)

        last_action = self.get_last_action(trajectory)
        pred = last_action["answer"]
        ref = configs["eval"]["reference_answers"]
        # rouge
        m = self.m
        rouge = m.compute(predictions=[pred], references=[ref])
        return float(rouge["rouge1"])


@beartype
class URLExactEvaluator(AEvaluator):
    """Check whether the URL is exactly the same as of the reference URLs"""

    async def __acall__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page | PseudoPage
    ) -> float:
        with open(config_file, "r") as f:
            configs = json.load(f)

        def clean_url(url: str) -> str:
            url = str(url)
            # Replace http://localhost with http://127.0.0.1 to keep things consistent across evals.
            url = url.replace("localhost", "127.0.0.1")
            if url.endswith("/"):
                url = url[:-1]
            return url

        pred = clean_url(page.url)
        ref_urls = configs["eval"]["reference_url"].split(" |OR| ")
        ref_urls = [clean_url(url) for url in ref_urls]
        matching_rule = configs["eval"].get("url_note", "EXACT")
        if matching_rule == "EXACT":
            if pred in ref_urls:
                return 1.0
            else:
                return 0.0
        elif matching_rule == "GOLD in PRED":
            if any([ref in pred for ref in ref_urls]):
                return 1.0
            else:
                return 0.0
        else:
            raise ValueError(f"Unknown matching rule: {matching_rule}")


@beartype
class HTMLContentExactEvaluator(AEvaluator):
    """Check whether the contents appear in the page"""

    async def __acall__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page | PseudoPage
    ) -> float:
        with open(config_file, "r") as f:
            configs = json.load(f)

        targets = configs["eval"]["program_html"]

        score = 1.0
        for target in targets:
            target_url: str = target["url"]  # which url to check
            if target_url.startswith("func"):
                func = target_url.split("func:")[1]
                func = func.replace("__last_url__", page.url)
                target_url = await eval(func)
                # logger.debug(f"calling {func} to get target_url: {target_url}")

            locator: str = target["locator"]  # js element locator

            # navigate to that url
            if target_url != "last":
                await page.goto(target_url)
                time.sleep(3)  # TODO [shuyanzh]: fix this hard-coded sleep

            # empty, use the full page
            if not locator.strip():
                selected_element = await page.content()
            # use JS to select the element
            elif locator.startswith("document.") or locator.startswith(
                "[...document."
            ):
                if "prep_actions" in target:
                    try:
                        for prep_action in target["prep_actions"]:
                            await page.evaluate(f"() => {prep_action}")
                    except Exception:
                        pass
                try:
                    selected_element = str(await page.evaluate(f"() => {locator}"))
                    if not selected_element:
                        selected_element = ""
                except Exception:
                    # the page is wrong, return empty
                    selected_element = ""
            elif locator.startswith("lambda:"):
                try:
                    locator = locator.lstrip("lambda:")
                    selected_element = await page.evaluate(locator)
                    if not selected_element:
                        selected_element = None
                except Exception:
                    # the page is wrong, return empty
                    selected_element = None
            # run program to call API
            elif locator.startswith("func:"):  # a helper function
                func = locator.split("func:")[1]
                func = func.replace("__page__", "page")
                selected_element = await eval(func)
                # logger.debug(
                #     f"calling {func} to get selected_element: {selected_element}")
            else:
                raise ValueError(f"Unknown locator: {locator}")

            # If the selected element is None, then the page is wrong
            if selected_element is None:
                score = 0.0
                break

            if "exact_match" in target["required_contents"]:
                required_contents = target["required_contents"]["exact_match"]
                score *= StringEvaluator.exact_match(
                    ref=required_contents, pred=selected_element
                )
            elif "must_include" in target["required_contents"]:
                required_contents = target["required_contents"]["must_include"]
                assert isinstance(required_contents, list)
                for content in required_contents:
                    content_or = content.split(" |OR| ")
                    score *= any(
                        [
                            StringEvaluator.must_include(
                                ref=content, pred=selected_element
                            )
                            for content in content_or
                        ]
                    )
            elif "must_exclude" in target["required_contents"]:
                required_contents = target["required_contents"]["must_exclude"]
                assert isinstance(required_contents, list)
                for content in required_contents:
                    assert " |OR| " not in content
                    score *= StringEvaluator.must_exclude(
                        content, pred=selected_element
                    )
            elif "required_values" in target["required_contents"]:
                required_values = target["required_contents"][
                    "required_values"
                ]
                assert isinstance(required_values, list)
                if isinstance(selected_element, str):
                    selected_element = NumericEvaluator.str_2_int(
                        selected_element
                    )
                if selected_element is None:
                    score = 0.0
                else:
                    for value in required_values:
                        value_or = value.split(" |OR| ")
                        score *= any(
                            [
                                NumericEvaluator.compare_inequality(
                                    selected_element, value
                                )
                                for value in value_or
                            ]
                        )
            elif "fuzzy_match" in target["required_contents"]:
                targets = target["required_contents"]["fuzzy_match"]
                assert isinstance(targets, str)
                targets = targets.split(" |OR| ")
                for target in targets:
                    score *= max(
                        [
                            StringEvaluator.fuzzy_match(
                                ref=target,
                                pred=selected_element,
                                intent="NOT USED",
                                fuzzy_match_provider=self.fuzzy_match_provider,
                            )
                        ]
                    )
            else:
                raise ValueError(
                    f"Unknown required_contents: {target['required_contents'].keys()}"
                )

        return score


@beartype
class PageImageEvaluator(AEvaluator):
    """Check whether the answer is correct by querying a vision model."""

    def __init__(self, captioning_fn, fuzzy_match_provider: str = "openai"):
        self.captioning_fn = captioning_fn
        self.fuzzy_match_provider = fuzzy_match_provider
        # Default to 0.8 as the threshold for similarity to account for compression, resizing, etc
        # This might be too generous but we bias towards minimizing false negatives.
        self.ssim_threshold = 0.8
        return

    async def __acall__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page | PseudoPage | None = None
    ) -> float:
        with open(config_file, "r") as f:
            configs = json.load(f)

        for query in configs["eval"]["page_image_query"]:
            locator: str = query["eval_image_class"]
            target_url: str = query["eval_image_url"]
            if target_url.startswith("func"):
                func = target_url.split("func:")[1]
                func = func.replace("__last_url__", page.url)
                target_url = await eval(func)
                # logger.debug(f"calling {func} to get target_url: {target_url}")

            # navigate to that url
            if target_url != "last":
                await page.goto(target_url)
                time.sleep(3)  # TODO(jykoh): fix this hard-coded sleep

            # empty, use the full page
            if not locator.strip():
                images = await page.get_by_role("img").all()
            # use JS to select the element
            elif locator.startswith("."):
                # Get all img children under the locator
                elements = await page.query_selector_all(locator)
                images = []
                for element in elements:
                    is_img = await element.evaluate(
                        'element => element.tagName === "IMG"'
                    )
                    if is_img:
                        images.append(element)
                    else:
                        images.extend(await element.query_selector_all("img"))
            else:
                raise ValueError(f"Unknown locator: {locator}")

            if images == []:
                return 0.0

            all_image_pixels = []
            for image in images:
                try:
                    # Get image from URL.
                    image_url = await image.get_attribute("src")
                    if not image_url.startswith(
                        ("http://", "https://", "www.")
                    ):
                        image_url = urljoin(page.url, image_url)
                    image = Image.open(
                        requests.get(image_url, stream=True).raw
                    )
                    all_image_pixels.append(image)
                except Exception as e:
                    print("[WARNING]: ", e)

            score = 1.0
            if all_image_pixels == []:
                return 0.0
            else:
                # Run the VQA eval on the image elements.
                eval_vqas = query.get("eval_vqa", [])
                assert (
                    len(eval_vqas) > 0 or "eval_fuzzy_image_match" in query
                ), "eval_vqa must have at least 2 questions or eval_fuzzy_image_match must be True"
                for qa in eval_vqas:
                    question, answer = qa["question"], qa["answer"]
                    prompt = f"Q: {question} A:"
                    pred_ans = self.captioning_fn(
                        all_image_pixels, [prompt] * len(all_image_pixels)
                    )
                    score *= float(
                        any(
                            [answer.lower() in ans.lower() for ans in pred_ans]
                        )
                    )

                if "eval_fuzzy_image_match" in query:
                    ssim_threshold = query.get(
                        "ssim_threshold", self.ssim_threshold
                    )
                    exact_match_imgs = query["eval_fuzzy_image_match"].split(
                        " |OR| "
                    )
                    all_exact_match_pixels = []

                    for exact_match_img in exact_match_imgs:
                        if exact_match_img.startswith("http"):
                            exact_match_pixels = Image.open(
                                requests.get(exact_match_img, stream=True).raw
                            )
                        else:
                            exact_match_pixels = Image.open(exact_match_img)
                        all_exact_match_pixels.append(exact_match_pixels)

                    # Check if any of the images on the page match
                    found_exact_match = False
                    for exact_match_pixels in all_exact_match_pixels:
                        for image_pixels in all_image_pixels:
                            ssim = image_utils.get_image_ssim(
                                image_pixels, exact_match_pixels
                            )
                            if ssim > ssim_threshold:
                                found_exact_match = True
                                break
                    score *= float(found_exact_match)

        return score


class AEvaluatorComb:
    def __init__(self, evaluators: list[AEvaluator]) -> None:
        self.evaluators = evaluators

    async def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page | PseudoPage
    ) -> float:

        score = 1.0
        for evaluator in self.evaluators:
            cur_score = await evaluator.__acall__(trajectory, config_file, page)
            score *= cur_score

        return score


@beartype
class TrajectoryURLExactEvaluator(AEvaluator):
    """Check whether the trajectory is correct"""

    async def __acall__(self, trajectory: Trajectory, config_file: Path | str, page: Page | PseudoPage) -> float:
        with open(config_file, "r") as f:
            configs = json.load(f)

        def clean_url(url: str) -> str:
            url = str(url)
            # Replace http://localhost with http://127.0.0.1 to keep things consistent across evals.
            # url = url.replace("localhost", "127.0.0.1")

            # The above doesnt work if hosting websites on other machines.
            # this will find the common endpoint and map it to 127.0.0.1
            # url = map_endpoint_to_target(url, "127.0.0.1")  # Comment out for browser-use

            if url.endswith("/"):
                url = url[:-1]
            return url

        # Parse into AND-groups of OR-alternatives
        # NOTE: This is brittle. Using string form to minimize modifications to config files pattern.
        # Should refactor config files pattern and this logic.
        ref_urls = configs["eval"]["trajectory_url_match"]["reference_url"]
        matching_rule = configs["eval"]["trajectory_url_match"].get("url_note", "EXACT")

        ref_groups = [[clean_url(u.strip()) for u in group.split(" |OR| ")] for group in ref_urls.split(" |AND| ")]

        # Get URLs visited in trajectory
        trajectory_urls = [
            clean_url(s["info"]["page"].url) for s in trajectory[::2] if "info" in s and "page" in s["info"]
        ]

        # Check if all reference URLs are in the trajectory
        if matching_rule == "EXACT":
            ok = all(any(ref in trajectory_urls for ref in group) for group in ref_groups)
        elif matching_rule == "GOLD in PRED":
            ok = all(any(any(ref in url for url in trajectory_urls) for ref in group) for group in ref_groups)
        else:
            raise ValueError(f"Unknown matching rule: {matching_rule}")

        return 1.0 if ok else 0.0


@beartype
def evaluator_router(
    config_file: Path | str, captioning_fn=None, fuzzy_match_prov=None
) -> AEvaluatorComb:
    """Router to get the evaluator class"""
    with open(config_file, "r") as f:
        configs = json.load(f)

    eval_types = configs["eval"]["eval_types"]
    evaluators: list[AEvaluatorComb] = []
    for eval_type in eval_types:
        match eval_type:
            case "string_match":
                evaluators.append(StringEvaluator(fuzzy_match_provider=fuzzy_match_prov or "openai"))
            case "url_match":
                evaluators.append(URLExactEvaluator(fuzzy_match_provider=fuzzy_match_prov or "openai"))
            case "program_html":
                evaluators.append(HTMLContentExactEvaluator(fuzzy_match_provider=fuzzy_match_prov or "openai"))
            case "page_image_query":
                evaluators.append(PageImageEvaluator(captioning_fn, fuzzy_match_provider=fuzzy_match_prov or "openai"))
            case "trajectory_url_match":
                evaluators.append(TrajectoryURLExactEvaluator(fuzzy_match_provider=fuzzy_match_prov or "openai"))
            case _:
                raise ValueError(f"eval_type {eval_type} is not supported")

    return AEvaluatorComb(evaluators)
