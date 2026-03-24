# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import json
from typing import Any

import litellm
from pydantic import BaseModel
from scripts_evaluation.evaluate_with_openai import (
    GRADER_TEMPLATE as GRADER_TEMPLATE_OPENAI,
)
from scripts_evaluation.evaluate_with_openai import (
    compute_citation_metrics,
    extract_citations_from_response,
    parse_judge_response,
)
from search_agent.prompts import GRADER_TEMPLATE_QWEN

from ...core.context import try_get_context
from ...core.types import SessionScore
from ...utils.settings import get_settings

_settings = get_settings()


class BrowseCompEvaluator(BaseModel):
    eval_model_id: str
    sampling_params: dict[str, Any] = {}
    grader_template: str

    def evaluate_response(
        self,
        agent_response,
        instance,
        retrieved_docids_set=None,
        tool_call_counts=None,
    ) -> SessionScore:
        question = instance["query"]
        correct_answer = instance["gold_answer"]
        positives_for_query = instance["evidence_docs"]
        cited_docids = []
        confidence = None
        retrieval_recall = None
        extracted_final_answer = None
        judge_textual_response = None

        # compute retrieval recall
        if retrieved_docids_set is not None:
            retrieval_recall = len(retrieved_docids_set.intersection(set(positives_for_query))) / float(
                len(positives_for_query)
            )

        if not agent_response:  # run was halted without a final answer
            is_successful = False
            score = 0
            is_complete = False
            parse_error = False
            judge_usage = None

        else:  # call judge
            is_complete = True
            prompt = self.create_judge_prompt(question, agent_response, correct_answer)
            judge_response = litellm.completion(
                model=self.eval_model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_output_tokens,
                litellm_metadata={"context": try_get_context()},
                **self.sampling_params,
            )
            choice = judge_response["choices"][0]
            judge_textual_response = choice.get("message").content
            judge_usage = judge_response.usage.copy()
            answer_metrics = parse_judge_response(judge_textual_response)
            parse_error = bool(answer_metrics["parse_error"])

            if parse_error:
                is_successful = False
                score = 0
            else:
                is_successful = answer_metrics["correct"]
                score = int(is_successful) if is_successful is not None else 0
                confidence = (answer_metrics.get("confidence", 100),)
                extracted_final_answer = answer_metrics.get("extracted_final_answer")
                cited_docids = extract_citations_from_response(agent_response)

        citation_metrics_positives = compute_citation_metrics(cited_docids, positives_for_query)
        scores = {
            "Accuracy": score,
            "Retrieval_recall": retrieval_recall,
            "Citation_metrics_positives": citation_metrics_positives,
            "Confidence": confidence,
        }
        meta_data = {
            "instance": instance.copy(),
            "retrieved_docids": list(retrieved_docids_set) if retrieved_docids_set else [],
            "response": agent_response,
            "extracted_final_answer": extracted_final_answer,
            "judge_model": self.eval_model_id,
            "is_complete": is_complete,
            "judge_parse_error": parse_error,
            "tool_call_counts": tool_call_counts,
        }
        if parse_error:
            meta_data["judge_raw_response"] = judge_textual_response
        all_scores = SessionScore(
            score=int(is_successful),
            success=is_successful,
            is_finished=is_complete,
            session_metrics=scores,
            session_metadata=meta_data,
        )

        return all_scores, judge_usage

    def create_judge_prompt(self, question: str, response: str, correct_answer: str) -> str:
        return self.grader_template.format(question=question, response=response, correct_answer=correct_answer)


class BrowseCompEvaluatorOpenai(BrowseCompEvaluator):
    max_output_tokens: int = 1024
    grader_template: str = GRADER_TEMPLATE_OPENAI
    eval_model_id: str = "openai/Azure/gpt-4.1"


class BrowsecompEvaluatorQwen(BrowseCompEvaluator):
    max_output_tokens: int = 4096
    temperature: int = 0.7
    top_p: float = 0.8
    top_k: int = 20
    eval_model_id: str = "Qwen/Qwen3-32B"

    def model_post_init(self, __context) -> None:
        self.grader_template = GRADER_TEMPLATE_QWEN
        self.sampling_params = {
            "top_p": self.top_p,
            "temperature": self.temperature,
            "top_k": self.top_k,
        }


if __name__ == "__main__":
    e = BrowseCompEvaluatorOpenai()  # eval_model_id="watsonx/openai/gpt-oss-120b")
    instance = {
        "gold_answer": "2015",
        "query": "When did someone was born",
        "evidence_docs": ["1", "4"],
    }
    response = {
        "explanation": "as stated in docs [1] [5]",
        "exact_answer": "1925",
        "confidence": 0.8,
    }
    response = json.dumps(response)

    r = e.evaluate_response(response, instance)
    print(r[0])
