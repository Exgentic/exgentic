# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import datetime
from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, ClassVar, Mapping

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scripts_evaluation.evaluate_with_openai import (
    calculate_calibration_error,
)

from .search_tool_handler import BCPSearchToolHandler
from ...core.actions import ActionsHandler
from .browsecomp_eval import (
    BrowseCompEvaluator,
    BrowseCompEvaluatorOpenai,
    BrowsecompEvaluatorQwen,
)
from .searcher_cache import SearchDiskCacheSession
from ...utils.cost import CostReport, LiteLLMCostReport
from ...core import Benchmark, Session
from ...core.types import (
    Action,
    MessageAction,
    FinishAction,
    SingleAction,
    ActionType,
    SingleObservation,
    Observation,
    BenchmarkResults,
    SessionScore,
    EmptyObservation,
    SessionIndex,
)
from ...adapters.executors.executer import make_executer
from ...utils.settings import ExecuterName, ExgenticSettings, get_settings


# Paper-reported total for the full BrowseCompPlus dataset.
DEFAULT_TOTAL_TASKS = 830


class BrowseCompPlusSearchArgs(BaseModel):
    query: str


class BrowseCompPlusSearchAction(SingleAction):
    name: Literal["search"] = "search"
    arguments: BrowseCompPlusSearchArgs


class BrowseCompPlusGetDocumentsArgs(BaseModel):
    docid: str


class BrowseCompPlusGetDocAction(SingleAction):
    name: Literal["get_document"] = "get_document"
    arguments: BrowseCompPlusGetDocumentsArgs


class BrowseCompPlusFinishArgs(BaseModel):
    exact_answer: str = Field(description="Your succinct, final answer")
    explanation: str = Field(
        description="Your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20]."
    )
    confidence: float = Field(
        description="Your confidence score between 0% and 100% for your answer"
    )


class BrowseCompPlusFinishAction(FinishAction):
    name: Literal["submit"] = "submit"
    arguments: BrowseCompPlusFinishArgs


class BrowseCompPlusSession(Session):
    _done: bool
    evaluator: BrowseCompEvaluator

    def __init__(
        self,
        settings: ExgenticSettings,
        instance: Dict[str, Any],
        searcher_params: Dict[str, Any],
        judge_model: str = "openai/Azure/gpt-4.1",
        max_interactions: Optional[int] = 100,
        session_id: str | None = None,
    ) -> None:
        if session_id is not None:
            self._session_id = session_id
        self._instance = instance.copy()
        self._task_id = instance["task_id"]
        self._done = False

        # Initialize search tool handler using singleton service
        self._init_search_tool_handler(searcher_params)

        self._registry = ActionsHandler(logger=self.logger)
        self.set_action_types()
        self._response = None
        self.evaluator = self.get_evaluator(judge_model)
        self.retrieved_docids = set()
        self.tool_call_count = defaultdict(lambda: 0)
        self.model_usage = None
        self.total_actions_executed = 0
        self.max_interactions = max_interactions
        self.logger.info(f"Running for query id {self.task_id}")
        super().__init__()

    def _init_search_tool_handler(self, searcher_params: Dict[str, Any]) -> None:
        """Initialize search tool handler using singleton service."""
        from .search_service import get_search_service
        from searcher.searchers import SearcherType

        # Get searcher from singleton service
        search_service = get_search_service()
        searcher_class = SearcherType.get_searcher_class(
            searcher_params["searcher_type"]
        )

        # Extract searcher-specific args
        searcher_args = {
            k: v
            for k, v in searcher_params.items()
            if k
            not in [
                "searcher_type",
                "max_snippet_length",
                "top_k_docs",
                "include_get_document",
                "full_doc_max_tokens",
            ]
        }

        # Pass session's logger to the service
        searcher_obj = search_service.get_or_create_searcher(
            searcher_type=searcher_params["searcher_type"],
            searcher_class=searcher_class,
            logger=self.logger,
            **searcher_args,
        )

        # Create search tool handler
        self.search_tool_handler = BCPSearchToolHandler(
            searcher=searcher_obj,
            snippet_max_tokens=searcher_params["max_snippet_length"],
            k=searcher_params["top_k_docs"],
            include_get_document=searcher_params["include_get_document"],
            full_doc_max_tokens=searcher_params["full_doc_max_tokens"],
        )

    @property
    def task(self) -> str:
        get_doc_str = (
            "and document expansion "
            if self.search_tool_handler.include_get_document
            else ""
        )
        return f"""Answer the provided question by performing search {get_doc_str}as needed, and submit your final answer.
Question: {self._instance["query"]}
Note:
- The question has an answer discoverable through proper search.
- The question requires putting together information from different sources.

Your performance is scored based on:
    1. Most importantly, the correctness of the answer you assembled from different searches.
    2. Your effective use of search and your ability to retrieve all relevant information for the question.
    3. How efficiently you find all the relevant information, using as few searches as possible.

Important: During your work, Do NOT interact with the user or send any messages at any point — messages will be ignored and are NOT considered a valid final answer. The ONLY acceptable way to finish is by calling 'submit' with the required structured fields.

Finish the session always by calling `submit`. If you fail to find the answer, submit with exact_answer: "Can't find the answer."."""

    @property
    def context(self) -> Dict[str, Any]:
        return {}

    @property
    def actions(self) -> List["ActionType"]:
        return self._registry.actions

    @property
    def task_id(self) -> str:
        """Task identifier"""
        return str(self._instance["query_id"])

    def _to_observation(
        self, raw: Any, invoking: Optional[List[SingleAction]] = None
    ) -> Observation:
        return SingleObservation(invoking_actions=invoking or [], result=raw)

    def start(self):
        return EmptyObservation()

    def record_single_action(self, action: SingleAction) -> None:
        self.logger.info(
            f"Received *{action.name}* action with arguments: {action.arguments}"
        )
        self.tool_call_count[action.name] += 1
        self.total_actions_executed += 1

    def step(self, action: "Action") -> Optional["Observation"]:
        if action is None:
            self._done = True

        if self.total_actions_executed >= self.max_interactions:
            self.logger.info(
                f"Reached maximal limit of {self.total_actions_executed} allowed actions"
            )
            self._done = True

        if self._done:
            return None

        observation = self._registry.execute(action)
        return observation

    # Action handlers ------------------------------------------------------------
    def _handle_search(self, action: SingleAction) -> Any:
        self.record_single_action(action)
        result = self.get_search_result(action)
        # keep retrieved docs
        self.record_retrieved_docids(result)
        return result

    def _handle_get_document(self, action: SingleAction) -> Optional[SingleAction]:
        self.record_single_action(action)
        args_dict = self.get_arguments_dict(action.arguments)
        return self.search_tool_handler.execute_tool("get_document", args_dict)

    def _handle_finish(self, action: SingleAction) -> Optional[SingleAction]:
        self.record_single_action(action)
        final_response = self.get_arguments_dict(action.arguments)
        self._response = json.dumps(final_response)
        self._done = True
        return None

    def done(self) -> bool:
        return self._done

    def score(self) -> SessionScore:
        results, self.model_usage = self.evaluator.evaluate_response(
            agent_response=self._response,
            instance=self._instance,
            retrieved_docids_set=self.retrieved_docids,
            tool_call_counts=self.tool_call_count,
        )
        return results

    def get_cost(self) -> CostReport:
        if not self.model_usage:
            return LiteLLMCostReport.initialize_empty(
                model_name=self.evaluator.eval_model_id
            )

        return LiteLLMCostReport.from_token_counts(
            model_name=self.evaluator.eval_model_id,
            input_tokens=self.model_usage["prompt_tokens"],
            output_tokens=self.model_usage["completion_tokens"],
        )

    def close(self) -> None:
        super().close()
        # Persist minimal results for aggregation
        sc = self.score()
        self.save_results(sc.model_dump())

    def set_action_types(self):
        k = self.search_tool_handler.k
        n_tokens = self.search_tool_handler.snippet_max_tokens
        self._registry.add_action(
            name="search",
            description=f"Perform a search on a knowledge source: supply a single 'query' string; the action retrieves the {k} top most relevant results, each trimmed to {n_tokens} tokens.",
            action_cls=BrowseCompPlusSearchAction,
            handler=self._handle_search,
        )
        self._registry.add_action(
            name="submit",
            description="Submit final answer and complete",
            action_cls=BrowseCompPlusFinishAction,
            handler=self._handle_finish,
            is_finish=True,
        )

        if self.search_tool_handler.include_get_document:
            self._registry.add_action(
                name="get_document",
                description="Retrieve the full document using its document id",
                action_cls=BrowseCompPlusGetDocAction,
                handler=self._handle_get_document,
            )
        # todo: agents fail in practice without this option, even with  explicit instructions. should we keep it?
        self._registry.add_action(
            name="message",
            description="Send the final answer as a message to the user",
            action_cls=MessageAction,
            handler=self._handle_finish,
            is_hidden=True,
            is_message=True,
        )

    def get_evaluator(self, eval_model_id):
        if "gpt" in eval_model_id:
            return BrowseCompEvaluatorOpenai(eval_model_id=eval_model_id)
        elif eval_model_id == "Qwen/Qwen3-32B":
            return BrowsecompEvaluatorQwen()  # Currently not supported
        else:
            raise ValueError(f"Invalid eval_model_id: {eval_model_id}")

    # Robustly extract the answer from pydantic model or dict
    def get_arguments_dict(self, args):
        if isinstance(args, BaseModel):
            return args.model_dump()  # Pydantic v2
        elif isinstance(args, Mapping):
            return dict(args)
        else:
            return str(getattr(args, "value", {}))

    def get_retrieved_docids(self, result: str):
        try:
            result = json.loads(result)
            retrieved_docids = set([result.get("docid") for result in result])
            self.retrieved_docids = self.retrieved_docids | retrieved_docids
        except json.decoder.JSONDecodeError:
            self.logger.error(f"Failed to retrieve docids: {result}")

    def record_retrieved_docids(self, result: str):
        try:
            result = json.loads(result)
            retrieved_docids = set([result.get("docid") for result in result])
            self.retrieved_docids = self.retrieved_docids | retrieved_docids
        except json.decoder.JSONDecodeError:
            self.logger.error(f"Failed to retrieve docids: {result}")

    def get_search_result(self, act):
        args_dict = self.get_arguments_dict(act.arguments)
        searcher_params = self.get_searcher_params()
        search_cache = SearchDiskCacheSession(args_dict["query"], **searcher_params)
        search_cache.logger = self.logger
        result = search_cache.handle_start_fetch_results()
        if result:
            self.logger.info(f"Retrieved docs from cache: {result}")
        else:
            result = self.search_tool_handler.execute_tool(act.name, args_dict)
            search_cache.cache_results(result)
            self.logger.info(f"Retrieved docs from searcher: {result}")
        return result

    def get_searcher_params(self):
        session_searcher = self.search_tool_handler.searcher

        def get_att_or_none(obj, att):
            return getattr(obj, att) if hasattr(obj, att) else None

        return {
            "n": self.search_tool_handler.snippet_max_tokens,
            "k": self.search_tool_handler.k,
            "search_type": self.search_tool_handler.searcher.search_type,
            "search_model": get_att_or_none(session_searcher.args, "model_name"),
            "normalize": get_att_or_none(session_searcher.args, "normalize"),
        }


class BrowseCompPlusBenchmark(Benchmark, BaseModel):
    display_name: ClassVar[str] = "BrowseCompPlus"
    slug_name: ClassVar[str] = "browsecompplus"
    model_config = ConfigDict(arbitrary_types_allowed=True)
    subset: Literal["main"] = "main"

    executer: Optional[ExecuterName] = "inprocess"  # Code is threadsafe

    # Internals
    _dataset: List[Dict[str, Any]] | None = None
    _task_lookup: Dict[str, Dict[str, Any]] | None = None

    # Agent inference params (for logging)
    inference_model: str = "N/A"

    # searcher params
    searcher_type: str = "faiss"  # "bm25" or "faiss"
    searcher_model_name: str = "Qwen/Qwen3-Embedding-8B"  # Used for faiss only
    max_snippet_length: int = 512
    top_k_docs: int = 5
    include_get_document: bool = True
    normalize_search: bool = True
    full_doc_max_tokens: int = 2048
    max_interactions: Optional[int] = 100

    # bcp evaluation params
    judge_model: str = "openai/Azure/gpt-4.1"

    def model_post_init(self, __context) -> None:
        # Initialize dataset and task lookup
        self._ensure_dataset()

    def list_tasks(self) -> List[str]:  # type: ignore[override]
        self._ensure_dataset()
        return [str(item["query_id"]) for item in self._dataset]

    @property
    def assets_dir(self):
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        return script_dir / "assets"

    def extract_dataset(self):
        data_path = self.assets_dir / "data" / "browsecomp_plus_decrypted_docids.jsonl"
        if not data_path.exists():
            raise Exception(
                f"{data_path} does not exist. Benchmark must be set up before use."
            )
        instances = pd.read_json(path_or_buf=data_path, lines=True).to_dict(
            orient="records"
        )

        def proces_instance(instance):
            processed_instance = {
                "query_id": instance["query_id"],
                "query": instance["query"],
                "gold_answer": instance["answer"],
            }
            for k in ["gold_docs", "evidence_docs", "negative_docs"]:
                processed_instance[k] = instance[k]
            return processed_instance

        instances = [proces_instance(instance) for instance in instances]
        return instances

    def _ensure_dataset(self) -> None:
        if self._dataset is None:
            self._dataset = self.extract_dataset()
            self._task_lookup = {str(item["query_id"]): item for item in self._dataset}

    def _get_searcher_params(self) -> Dict[str, Any]:
        """Get searcher parameters to pass to session."""
        index_dir = self.assets_dir / "indexes"
        if self.searcher_type == "bm25":
            index_dir = index_dir / "bm25"
            searcher_args = {"index_path": str(index_dir)}
        else:
            model_dir_name = self.searcher_model_name.lower().split("/")[-1]
            index_dir = index_dir / model_dir_name
            if not os.path.exists(index_dir):
                available_models = os.listdir(self.assets_dir / "indexes")
                raise FileNotFoundError(
                    f"Index dir for BrowseCompPlus benchmark {index_dir} does not exist. "
                    f"Please select an available embedding model out of {available_models}"
                )
            index_path = index_dir / "corpus.shard*_of_4.pkl"
            searcher_args = {
                "index_path": str(index_path),
                "model_name": self.searcher_model_name,
                "normalize": self.normalize_search,
            }

        return {
            "searcher_type": self.searcher_type,
            "max_snippet_length": self.max_snippet_length,
            "top_k_docs": self.top_k_docs,
            "include_get_document": self.include_get_document,
            "full_doc_max_tokens": self.full_doc_max_tokens,
            **searcher_args,
        }

    def create_session(self, index: SessionIndex) -> BrowseCompPlusSession:
        self._ensure_dataset()
        task_id = index.task_id
        if self._task_lookup is None or task_id not in self._task_lookup:
            raise KeyError(f"Unknown BrowseCompPlus task id '{task_id}'.")
        instance = {"task_id": task_id, **self._task_lookup[task_id]}
        session_id = index.session_id
        executer = make_executer(
            self.executer,
            BrowseCompPlusSession,
            get_settings(),
            instance,
            self._get_searcher_params(),
            max_interactions=self.max_interactions,
            session_id=session_id,
            judge_model=self.judge_model,
        )
        proxy = executer.get_proxy()
        return proxy  # type: ignore[return-value]

    def aggregate_sessions(self, sessions: List[SessionIndex]) -> BenchmarkResults:
        # Aggregate per-session scores written by sessions
        scores: List[float] = []
        retrieval_recalls: List[float] = []
        confidence_list: List[float] = []
        tool_call_counts_list: List[Dict[str, float]] = []
        correctness: List[float] = []
        for paths in self.get_sessions_paths(sessions):
            fp = paths.benchmark_results
            if not fp.exists():
                raise FileNotFoundError(
                    f"Missing results for planned session '{paths.session_id}' at {fp}"
                )

            with open(fp, "r", encoding="utf-8-sig") as f:
                payload = json.load(f)
            if not payload:
                raise ValueError(
                    f"Empty benchmark results for session '{paths.session_id}' at {fp}"
                )

            s = float(payload["score"])  # minimal: assume exists
            scores.append(s)
            metrics = payload.get("session_metrics", {})
            if not metrics:
                print(f"No metrics for session '{paths.session_id}' at {fp}")

            retrieval_recall = metrics.get("Retrieval_recall", 0)
            retrieval_recalls.append(
                float(retrieval_recall) if retrieval_recall is not None else 0
            )
            correctness.append(payload.get("success", 0))

            confidence = metrics.get("Confidence")
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0
            confidence_list.append(confidence)

            metadata = payload.get("session_metadata", {})
            if not metadata:
                print(f"No metadata for session '{paths.session_id}' at {fp}")
            tool_call_counts_list.append(metadata.get("tool_call_counts", {}))

        avg = sum(scores) / len(scores) if scores else 0.0
        avg_retrieval_recalls = (
            sum(retrieval_recalls) / len(retrieval_recalls)
            if retrieval_recalls
            else 0.0
        )
        tools_keys = set().union(*tool_call_counts_list)
        avg_tool_use_counts = {
            k: sum(d.get(k, 0) for d in tool_call_counts_list)
            / len(tool_call_counts_list)
            for k in tools_keys
        }

        calibration_error = None
        # calibration error only comupted for a large number of examples
        if len(correctness) >= 100:
            try:
                calibration_error = calculate_calibration_error(
                    confidences=confidence_list, correctness=correctness
                )
            except Exception:
                print(
                    f"Failed to calculate calibration error for session '{paths.session_id}' at {fp}"
                )
                calibration_error = 0
        metrics = {
            "LLM": self.inference_model,
            "Accuracy (%)": avg,
            "Recall (%)": avg_retrieval_recalls,
            "avg_tool_stats": avg_tool_use_counts,
            "Calibration Error (%)": calibration_error,
            "Retriever": self.searcher_model_name,
            "Link": "change me when submitting",
            "Evaluation Date": datetime.datetime.now().date().isoformat(),
        }
        return BenchmarkResults(
            benchmark_name="BrowseCompPlus",
            total_tasks=len(sessions),
            score=avg,
            metrics=metrics,
        )
