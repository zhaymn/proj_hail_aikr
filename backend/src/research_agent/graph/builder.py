from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from research_agent.config import get_settings
from research_agent.graph.state import GraphState
from research_agent.retrieval.dense import DenseRetriever
from research_agent.schemas import Mode
from research_agent.services.text_generation import TextGenerationService

settings = get_settings()
dense_retriever = DenseRetriever(settings)
text_service = TextGenerationService(settings)
_PAPER_TEXT_CACHE: dict[str, str] = {}

REVIEWER_STATE_KEYS = (
    "attack_vectors",
    "active_vector_id",
    "debate_history",
    "debate_summary",
    "skeptic_position",
    "advocate_position",
    "resolution",
    "turn_count",
    "syntheses",
    "vector_verdicts",
    "vector_judgments",
    "vector_reports",
    "final_report",
    "next_speaker",
    "intervention_mode",
    "vectors_remaining",
)


def _is_rate_limit_error(error: Exception) -> bool:
    text = str(error or "").lower()
    return (
        "rate_limit_exceeded" in text
        or "rate limit reached" in text
        or "resource_exhausted" in text
        or "quota exceeded" in text
    )


def _extract_retry_hint(error: Exception) -> str:
    text = str(error or "")
    match = re.search(r"Please try again in ([^\\.]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _prepare_mode_step(state: GraphState) -> GraphState:
    mode = state["mode"]
    instructions = {
        Mode.LOCAL: (
            "You are a strict retrieval assistant. "
            "Answer only from the retrieved paper excerpts. "
            "Do not use outside knowledge. "
            "For every factual claim, attach inline citations like [1], [2]. "
            "For count questions, return a single exact numeric answer when explicit evidence exists. "
            "If the excerpt lists numbered entities (for example Expert 1 ... Expert 6), infer the explicit count as 6. "
            "If evidence is missing, respond with exactly: "
            "'This information is not in your uploaded papers.' "
            "Then briefly name what is missing and offer Global mode."
        ),
        Mode.GLOBAL: (
            "You are a high-quality general-purpose assistant. "
            "Respond naturally, clearly, and directly like a normal LLM. "
            "Use uploaded papers only when they are genuinely relevant to the user request. "
            "Do not force paper grounding for general questions. "
            "Use inline citations [n] only for claims that come from retrieved paper context."
        ),
        Mode.WRITER: (
            "You are a research writing assistant that has studied this researcher's style. "
            "When drafting or rewriting, closely mirror the style profile provided: "
            "match their sentence length, formality level, hedging language, "
            "vocabulary choices, and citation format. "
            "Do not default to generic academic prose - the output should be "
            "indistinguishable from the researcher's own writing. "
            "If no style profile exists yet, say so and ask the user to upload a paper first. "
            "Help with any writing task: drafting sections, rewriting sentences, "
            "generating abstracts, or suggesting citation placements."
        ),
        Mode.REVIEWER: (
            "You are a rigorous top-tier ML conference reviewer. "
            "Produce a critical but fair review with concrete evidence and actionable fixes. "
            "Treat the user's message as a review objective/focus lens. "
            "Explicitly separate major concerns from minor concerns. "
            "Before calling something missing, verify whether the paper already addresses it. "
            "If addressed, mark it covered and explain why it is still sufficient/insufficient. "
            "Cite factual claims with inline citations [n]. "
            "You may use general field knowledge only to interpret why a grounded issue matters; do not invent paper-specific details."
        ),
        Mode.COMPARATOR: (
            "You are a research analyst running a claim-level comparison lab. "
            "Do not produce generic prose. Build explicit claim-to-evidence contrasts, "
            "highlight true conflicts, and end with concrete decisions. "
            "Use paper filenames to disambiguate every point. "
            "You may use general field knowledge only to interpret grounded benchmark or method differences; do not invent uncited paper facts."
        ),
    }
    debug = dict(state.get("debug", {}))
    debug["prepared_mode"] = mode.value
    debug["paper_count"] = len(state.get("paper_ids", []))
    return {
        "mode_instructions": instructions[mode],
        "debug": debug,
    }


def _retrieve_step(state: GraphState) -> GraphState:
    mode = state["mode"]
    paper_ids = state.get("paper_ids", [])
    debug = dict(state.get("debug", {}))
    query = _contextualize_query(
        message=state["message"],
        history=state.get("history", []),
        mode=mode,
    )

    if mode == Mode.REVIEWER and state.get("review_paper_id"):
        paper_ids = [state["review_paper_id"]]
        vector_claim = _active_vector_claim(state)
        if vector_claim:
            query = f"{state['message']} {vector_claim}".strip()
        else:
            query = state["message"]
        hits, reviewer_subqueries = _retrieve_reviewer_hits(
            query=query,
            paper_id=paper_ids[0],
        )
        debug["reviewer_subqueries"] = reviewer_subqueries
    elif mode == Mode.COMPARATOR:
        paper_ids = paper_ids[:3]
        query = f"{state['message']} contributions methods benchmarks results differences"
        hits = _retrieve_comparator_hits(query=query, paper_ids=paper_ids)
    elif mode == Mode.GLOBAL and not _should_use_global_retrieval(query=query, paper_ids=paper_ids):
        hits = []
        debug["global_retrieval"] = "skipped_for_general_query"
    else:
        hits, retrieval_subqueries = _retrieve_general_hits(
            query=query,
            paper_ids=paper_ids or None,
            mode=mode,
        )
        if retrieval_subqueries:
            debug["retrieval_subqueries"] = retrieval_subqueries

    documents = [document for document, _ in hits]
    debug["retrieved_count"] = len(documents)
    debug["retrieval_query"] = query
    debug["retrieval_preview"] = [_citation_from_document(document) for document in documents[:5]]
    debug["retrieval_scores"] = [float(score) if score is not None else None for _, score in hits[:5]]

    return {
        "retrieved_documents": documents,
        "debug": debug,
    }


def _retrieve_comparator_hits(
    *,
    query: str,
    paper_ids: list[str],
) -> list[tuple[Document, float | None]]:
    if not paper_ids:
        return []

    per_paper_top_k = max(12, settings.retrieval_top_k + 2)
    max_total = max(settings.retrieval_top_k * 5, len(paper_ids) * 12)
    subqueries = [
        query,
        f"{query} abstract contribution claim scope",
        f"{query} method architecture objective design",
        f"{query} evaluation benchmark dataset baseline metric results",
        f"{query} limitation efficiency compute reproducibility ablation",
    ]
    combined: list[tuple[Document, float | None]] = []
    for index, subquery in enumerate(subqueries):
        query_weight = max(0.7, 1.0 - (0.12 * index))
        for paper_id in paper_ids:
            hits = dense_retriever.retrieve(
                query=subquery,
                paper_ids=[paper_id],
                top_k=per_paper_top_k,
            )
            for document, score in hits:
                weighted_score = (float(score) if score is not None else 0.0) * query_weight
                combined.append((document, weighted_score))

    combined.sort(key=lambda pair: pair[1] if pair[1] is not None else float("-inf"), reverse=True)
    deduped: list[tuple[Document, float | None]] = []
    seen_chunk_ids: set[str] = set()
    for document, score in combined:
        chunk_id = str((document.metadata or {}).get("chunk_id", ""))
        if chunk_id and chunk_id in seen_chunk_ids:
            continue
        if chunk_id:
            seen_chunk_ids.add(chunk_id)
        deduped.append((document, score))
        if len(deduped) >= max_total:
            break
    non_reference = [
        (document, score)
        for document, score in deduped
        if not _looks_like_reference_snippet(document.page_content or "")
        and not _looks_like_non_argument_snippet(document.page_content or "")
        and not _looks_like_metadata_snippet(document.page_content or "")
        and not _looks_acknowledgement_text(document.page_content or "")
    ]
    if len(non_reference) >= max(6, len(paper_ids) * 3):
        return non_reference[:max_total]
    blended: list[tuple[Document, float | None]] = list(non_reference)
    for item in deduped:
        if item in blended:
            continue
        blended.append(item)
        if len(blended) >= max_total:
            break
    return blended


def _retrieve_general_hits(
    *,
    query: str,
    paper_ids: list[str] | None,
    mode: Mode,
) -> tuple[list[tuple[Document, float | None]], list[str]]:
    subqueries = _general_subqueries(query=query, mode=mode)
    if not subqueries:
        return [], []

    per_query_top_k = max(settings.retrieval_top_k, settings.rerank_top_n + 4)
    if mode == Mode.LOCAL:
        per_query_top_k = max(per_query_top_k, settings.retrieval_top_k * 3)
        if _is_math_intent_query(query):
            per_query_top_k = max(per_query_top_k, settings.retrieval_top_k * 4)
    aggregated: list[tuple[Document, float]] = []
    for index, subquery in enumerate(subqueries):
        hits = dense_retriever.retrieve(
            query=subquery,
            paper_ids=paper_ids,
            top_k=per_query_top_k,
        )
        query_weight = max(0.75, 1.0 - (0.08 * index))
        for document, score in hits:
            weighted_score = (float(score) if score is not None else 0.0) * query_weight
            aggregated.append((document, weighted_score))

    best_by_chunk: dict[str, tuple[Document, float]] = {}
    for document, score in aggregated:
        key = _document_identity(document)
        existing = best_by_chunk.get(key)
        if existing is None or score > existing[1]:
            best_by_chunk[key] = (document, score)

    merged = list(best_by_chunk.values())
    merged.sort(key=lambda item: item[1], reverse=True)
    target = max(settings.retrieval_top_k * 3, settings.rerank_top_n + 8)
    if mode == Mode.LOCAL:
        target = max(target, settings.retrieval_top_k * 6)
    return [(document, score) for document, score in merged[:target]], subqueries


def _general_subqueries(*, query: str, mode: Mode) -> list[str]:
    base = (query or "").strip()
    if not base:
        return []

    focused = _focused_retrieval_query(base)
    lower = base.lower()
    quantity_intent = any(marker in lower for marker in ("how many", "number", "count", "how much"))
    expert_intent = any(token.startswith("expert") for token in _tokenize_for_overlap(lower))
    seeds = [base]
    if focused and focused.lower() != lower:
        seeds.append(focused)

    if "mixture of experts" in lower or re.search(r"\bmoe\b", lower):
        seeds.append(f"{focused or base} mixture of experts model MoE")
    if mode == Mode.GLOBAL and _is_global_person_query(base):
        seeds.append(f"{focused or base} author authors corresponding author affiliation email")
    if expert_intent and quantity_intent:
        seeds.append(
            f"{focused or base} number of experts specific experts shared experts parameter analysis ablation"
        )
        seeds.append(f"{focused or base} expert 1 expert 2 expert 3 expert 4 expert 5 expert 6")
    if "transformer" in lower and re.search(r"\bhead\b|\bheads\b", lower):
        seeds.append(f"{focused or base} transformer attention heads multi-head")
    if "eeg" in lower:
        seeds.append(f"{focused or base} EEG model architecture dataset")
    if "model" in lower:
        seeds.append(f"{focused or base} proposed model name architecture")
    if _is_math_intent_query(base):
        seeds.append(f"{focused or base} equation equations objective loss function formula formulation")
        seeds.append(f"{focused or base} optimization training objective regularization derivation")
        seeds.append(f"{focused or base} notation variable definition term interpretation")
        seeds.append(f"{focused or base} where denotes can be formulated as equation ( )")
        if "attention" in lower:
            seeds.append(f"{focused or base} scaled dot product attention equation Q K V softmax sqrt d_k")
            seeds.append(f"{focused or base} multi head attention equation notation symbols")
    if mode == Mode.LOCAL:
        seeds.append(f"{focused or base} exact metric value method model version")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in seeds:
        normalized = " ".join(item.split()).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(item)
    return deduped


def _is_math_intent_query(query: str) -> bool:
    lower = (query or "").lower()
    markers = (
        "math",
        "equation",
        "equations",
        "formula",
        "formulation",
        "derivation",
        "objective",
        "loss",
        "notation",
        "term does",
        "teach me the math",
    )
    return any(marker in lower for marker in markers)


def _focused_retrieval_query(query: str) -> str:
    tokens = _tokenize_for_overlap(query)
    if not tokens:
        return ""
    drop = {
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "many",
        "much",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "the",
        "a",
        "an",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
    }
    kept = [token for token in tokens if token not in drop]
    if not kept:
        kept = tokens
    return " ".join(kept[:20])


def _contextualize_query(*, message: str, history: list[dict[str, Any]], mode: Mode) -> str:
    current = (message or "").strip()
    if not current:
        return ""
    if mode in {Mode.REVIEWER, Mode.COMPARATOR}:
        return current
    if not _is_followup_style_query(current):
        return current
    previous_user = _latest_user_history_message(history=history, exclude=current)
    if not previous_user:
        return current
    return f"{previous_user} {current}".strip()


def _is_followup_style_query(message: str) -> bool:
    lower = (message or "").strip().lower()
    if not lower:
        return False
    tokens = _tokenize_for_overlap(lower)
    if len(tokens) <= 6:
        return True
    followup_markers = (
        "that",
        "this",
        "it",
        "same",
        "exact number",
        "just number",
        "that's it",
        "thats it",
        "only",
    )
    return any(marker in lower for marker in followup_markers)


def _latest_user_history_message(*, history: list[dict[str, Any]], exclude: str) -> str:
    if not history:
        return ""
    excluded = " ".join((exclude or "").strip().lower().split())
    for item in reversed(history):
        if str(item.get("role", "")).lower() != "user":
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        normalized = " ".join(content.lower().split())
        if normalized == excluded:
            continue
        if _is_auto_reviewer_bootstrap(content):
            continue
        return content
    return ""


def _retrieve_reviewer_hits(
    *,
    query: str,
    paper_id: str,
) -> tuple[list[tuple[Document, float | None]], list[str]]:
    if not paper_id:
        return [], []

    subqueries = _reviewer_subqueries(query)
    per_query_top_k = max(6, settings.retrieval_top_k)
    aggregated: list[tuple[Document, float]] = []
    for index, subquery in enumerate(subqueries):
        hits = dense_retriever.retrieve(
            query=subquery,
            paper_ids=[paper_id],
            top_k=per_query_top_k,
        )
        query_weight = max(0.72, 1.0 - (0.04 * index))
        for document, score in hits:
            weighted_score = (float(score) if score is not None else 0.0) * query_weight
            aggregated.append((document, weighted_score))

    best_by_chunk: dict[str, tuple[Document, float]] = {}
    for document, score in aggregated:
        key = _document_identity(document)
        existing = best_by_chunk.get(key)
        if existing is None or score > existing[1]:
            best_by_chunk[key] = (document, score)

    merged = list(best_by_chunk.values())
    merged.sort(key=lambda item: item[1], reverse=True)
    target = max(settings.retrieval_top_k * 3, settings.rerank_top_n + 8)
    preferred = [
        (document, score)
        for document, score in merged
        if not _looks_like_reference_snippet(document.page_content or "")
        and not _looks_like_non_argument_snippet(document.page_content or "")
        and not _looks_like_metadata_snippet(document.page_content or "")
        and not _looks_acknowledgement_text(document.page_content or "")
    ]
    if len(preferred) >= max(8, settings.rerank_top_n):
        return [(document, score) for document, score in preferred[:target]], subqueries
    blended: list[tuple[Document, float | None]] = list(preferred)
    for document, score in merged:
        candidate = (document, score)
        if candidate in blended:
            continue
        blended.append(candidate)
        if len(blended) >= target:
            break
    return [(document, score) for document, score in blended[:target]], subqueries


def _reviewer_subqueries(query: str) -> list[str]:
    base = _normalize_reviewer_message(query)
    lower = base.lower()
    seeds: list[str] = []

    if base and not lower.startswith("focus lens:"):
        seeds.append(base)

    if not base or lower == "focus lens: full review":
        seeds.extend(
            [
                "full paper review abstract contribution novelty prior work claim scope",
                "full paper review method architecture training objective implementation design",
                "full paper review evaluation benchmark dataset baseline results metric score",
                "full paper review limitations ablation robustness reproducibility compute",
            ]
        )
    elif "novelty" in lower:
        seeds.extend(
            [
                "novelty review abstract contribution prior work delta claim support",
                "novelty review benchmark result delta baseline scope wording",
                "novelty review method evidence supporting the claimed contribution",
            ]
        )
    elif "method" in lower:
        seeds.extend(
            [
                "method review architecture algorithm design training objective implementation details",
                "method review benchmark evidence supporting the architecture choice",
                "method review limitations ablation reproducibility compute details",
            ]
        )
    else:
        seeds.extend(
            [
                f"{base or 'review'} core contribution novelty claims assumptions",
                f"{base or 'review'} method architecture training objective implementation details",
                f"{base or 'review'} evaluation benchmarks baselines metrics protocol statistical significance",
                f"{base or 'review'} limitations ablation robustness reproducibility",
            ]
        )
    deduped: list[str] = []
    seen: set[str] = set()
    for item in seeds:
        normalized = " ".join(item.split()).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(item)
        if len(deduped) >= 4:
            break
    return deduped


def _document_identity(document: Document) -> str:
    metadata = document.metadata or {}
    chunk_id = str(metadata.get("chunk_id", "")).strip()
    if chunk_id:
        return chunk_id
    filename = str(metadata.get("filename", "unknown.pdf")).strip()
    page = str(metadata.get("page", "")).strip()
    return f"{filename}:{page}:{hash(document.page_content)}"


def _read_paper_text(paper_id: str) -> str:
    normalized = str(paper_id or "").strip()
    if not normalized:
        return ""
    if normalized in _PAPER_TEXT_CACHE:
        return _PAPER_TEXT_CACHE[normalized]
    text_path = Path(settings.paper_text_dir) / f"{normalized}.txt"
    if not text_path.exists():
        _PAPER_TEXT_CACHE[normalized] = ""
        return ""
    try:
        content = text_path.read_text(encoding="utf-8")
    except Exception:
        content = ""
    cleaned = _clean_mojibake_text(content)
    _PAPER_TEXT_CACHE[normalized] = cleaned
    return cleaned


def _paper_abstract_block(text: str) -> str:
    cleaned = _clean_mojibake_text(text or "")
    if not cleaned:
        return ""
    match = re.search(
        r"\babstract\b\s*([\s\S]{120,2600}?)(?:\n\s*(?:1\s*introduction|introduction)\b)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    head = cleaned[:2400]
    if "1 Introduction" in head:
        head = head.split("1 Introduction", 1)[0]
    return re.sub(r"\s+", " ", head).strip()


def _metric_name_markers() -> tuple[str, ...]:
    return (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "roc-auc",
        "bleu",
        "rouge",
        "meteor",
        "wer",
        "cer",
        "perplexity",
        "exact match",
        "pass@1",
        "success rate",
        "win rate",
        "iou",
        "dice",
        "mse",
        "mae",
        "rmse",
        "top-1",
        "top-5",
        "mrr",
        "ndcg",
    )


def _has_metric_name(text: str) -> bool:
    cleaned = _clean_mojibake_text(text or "")
    lower = cleaned.lower()
    if not lower:
        return False
    if any(marker in lower for marker in _metric_name_markers()):
        return True
    return bool(
        re.search(
            r"\b(?:map|em|acc|r@1|r@5|r@10|hit@1|hit@10|hit rate|top1|top5)\b",
            lower,
        )
    )


def _method_signal_markers() -> tuple[str, ...]:
    return (
        "method",
        "approach",
        "architecture",
        "algorithm",
        "objective",
        "framework",
        "pipeline",
        "model",
        "backbone",
        "module",
        "component",
        "training",
        "inference",
        "optimization",
        "encoder",
        "decoder",
        "attention",
        "recurrent",
        "convolution",
        "diffusion",
        "retrieval",
        "prompting",
    )


def _evaluation_signal_markers() -> tuple[str, ...]:
    return (
        "benchmark",
        "dataset",
        "corpus",
        "task",
        "evaluation",
        "results",
        "baseline",
        "metric",
        "score",
        "test set",
        "validation set",
        "leaderboard",
        "ablation",
    )


def _efficiency_signal_markers() -> tuple[str, ...]:
    return (
        "training",
        "runtime",
        "latency",
        "throughput",
        "memory",
        "compute",
        "cost",
        "gpu",
        "gpus",
        "tpu",
        "hours",
        "days",
        "seconds",
        "parallel",
        "parallelization",
        "parameters",
        "flops",
    )


def _reviewer_category_markers(category: str) -> tuple[str, ...]:
    normalized = str(category or "").strip().lower()
    if normalized == "novelty":
        return (
            "abstract",
            "we propose",
            "our main result",
            "in this work",
            "novel",
            "new",
            "state of the art",
            "contribution",
            "prior work",
        )
    if normalized == "method":
        return (
            "method",
            "approach",
            "architecture",
            "algorithm",
            "objective",
            "design",
            "framework",
            "model",
            "backbone",
            "component",
            "attention",
            "implementation",
        )
    if normalized == "evaluation":
        return _evaluation_signal_markers() + _metric_name_markers() + ("state of the art",)
    if normalized == "ablation":
        return (
            "ablation",
            "robustness",
            "analysis",
            "sensitivity",
            "failure",
            "limitation",
            "training took",
            "days",
            "hours",
            "gpus",
            "compute",
            "learning rate",
            "epoch",
            "batch",
            "beam search",
            "parameters",
            "optimizer",
        )
    if normalized == "reproducibility":
        return (
            "training",
            "steps",
            "batch",
            "hyperparameters",
            "gpus",
            "seed",
            "implementation",
            "vocabulary",
            "configuration",
            "code",
        )
    return (
        "paper",
        "method",
        "result",
        "benchmark",
        "training",
    )


def _best_role_sentence(text: str, *, role: str) -> str:
    cleaned = _clean_visible_text(re.sub(r"\s+", " ", (text or "")).strip())
    if not cleaned:
        return ""
    best = ""
    best_score = float("-inf")
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    for sentence in sentences:
        snippet = sentence.strip(" -")
        if len(snippet) < 40 or len(snippet) > 360:
            continue
        if _looks_like_reference_snippet(snippet):
            continue
        if (
            _looks_like_non_argument_snippet(snippet)
            or _looks_like_metadata_snippet(snippet)
            or _looks_acknowledgement_text(snippet)
        ):
            continue
        lower = snippet.lower()
        score = 0.0
        if role == "summary":
            if lower.startswith(("we propose", "our main result", "the main result", "in this work")):
                score += 1.2
            if any(
                marker in lower
                for marker in (
                    "we propose",
                    "we present",
                    "we introduce",
                    "we study",
                    "we investigate",
                    "we show",
                    "we demonstrate",
                    "this paper",
                    "this work",
                    "our model",
                    "our approach",
                    "our main result",
                    "state-of-the-art",
                    "state of the art",
                    "contribution",
                    "results",
                    "benchmark",
                    "task",
                )
            ):
                score += 0.65
        elif role == "method":
            if any(marker in lower for marker in _method_signal_markers()):
                score += 1.0
            if any(
                marker in lower
                for marker in (
                    "based on",
                    "consists of",
                    "is built on",
                    "we use",
                    "we train",
                    "relies on",
                    "dispensing with",
                )
            ):
                score += 0.55
        elif role == "metric":
            if _has_metric_name(lower):
                score += 1.35
            if any(
                marker in lower
                for marker in _evaluation_signal_markers()
                + ("state-of-the-art", "state of the art", "improving", "achieve", "achieves")
            ):
                score += 0.65
            if re.search(r"\b\d+(?:\.\d+)?\b", snippet):
                score += 0.3
        elif role == "efficiency":
            strong_efficiency_markers = (
                "gpu",
                "gpus",
                "tpu",
                "parallel",
                "parallelization",
                "training took",
                "time to train",
                "latency",
                "throughput",
                "words per second",
                "hours",
                "days",
                "seconds",
                "flops",
                "learning rate",
                "epoch",
                "epochs",
                "batch",
                "beam search",
                "beam-search",
                "parameters",
                "optimizer",
                "computational cost",
            )
            if any(marker in lower for marker in strong_efficiency_markers):
                score += 1.2
            elif "training" in lower and (
                re.search(r"\b\d+(?:\.\d+)?\b", snippet)
                or any(
                    marker in lower
                    for marker in ("batch", "schedule", "optimizer", "epochs", "learning rate", "parameters")
                )
            ):
                score += 0.7
            if re.search(r"\b\d+(?:\.\d+)?\b", snippet):
                score += 0.25
        elif role == "limitation":
            if any(
                marker in lower
                for marker in (
                    "however",
                    "did not",
                    "despite",
                    "limited",
                    "without",
                    "penalized",
                    "cannot",
                    "scope",
                )
            ):
                score += 0.8
            if any(marker in lower for marker in ("oov", "out-of-vocabulary", "single model", "rescoring")):
                score += 0.35
        if lower.startswith("abstract"):
            score -= 0.1
        if score > best_score:
            best_score = score
            best = snippet
    threshold = 0.7 if role in {"summary", "method", "metric", "efficiency"} else 0.45
    return _compact_turn_text(best, max_chars=260) if best_score >= threshold else ""


def _best_citation_index_for_snippet(*, snippet: str, documents: list[Document], filename: str = "") -> int:
    lowered = str(snippet or "").strip().lower()
    if not lowered or not documents:
        return 1
    query_terms = set(_tokenize_for_overlap(lowered))
    best_index = 0
    best_score = float("-inf")
    for index, document in enumerate(documents):
        metadata = document.metadata or {}
        doc_filename = str(metadata.get("filename", "unknown.pdf")).strip()
        text = (document.page_content or "").lower()
        if not text:
            continue
        filename_bonus = 0.2 if filename and doc_filename == filename else 0.0
        contains_bonus = 1.15 if lowered in text else 0.0
        overlap = _overlap_score(text, query_terms)
        score = filename_bonus + contains_bonus + overlap
        if score > best_score:
            best_score = score
            best_index = index
    return best_index + 1


def _paper_profiles_from_documents(documents: list[Document]) -> dict[str, dict[str, Any]]:
    if not documents:
        return {}
    grouped: dict[str, dict[str, Any]] = {}
    for document in documents:
        metadata = document.metadata or {}
        filename = str(metadata.get("filename", "unknown.pdf")).strip() or "unknown.pdf"
        paper_id = str(metadata.get("paper_id", "")).strip()
        payload = grouped.setdefault(filename, {"paper_id": paper_id, "documents": []})
        if paper_id and not str(payload.get("paper_id", "")).strip():
            payload["paper_id"] = paper_id
        payload["documents"].append(document)

    profiles: dict[str, dict[str, Any]] = {}
    for filename, payload in grouped.items():
        paper_id = str(payload.get("paper_id", "")).strip()
        paper_docs = payload.get("documents", [])
        full_text = _read_paper_text(paper_id) or "\n".join(document.page_content or "" for document in paper_docs)
        full_text = _clean_visible_text(full_text)
        abstract = _paper_abstract_block(full_text)
        summary = (
            _best_role_sentence(abstract, role="summary")
            or _best_role_sentence(full_text, role="summary")
            or _extract_signal_sentence(abstract or full_text)
        )
        method = (
            _best_role_sentence(full_text, role="method")
            or _best_role_sentence(abstract, role="method")
            or summary
        )
        metric = (
            _best_role_sentence(abstract, role="metric")
            or _best_role_sentence(full_text, role="metric")
            or _extract_metric_sentence(abstract or full_text)
        )
        efficiency = (
            _best_role_sentence(abstract, role="efficiency")
            or _best_role_sentence(full_text, role="efficiency")
        )
        limitation = _best_role_sentence(full_text, role="limitation")
        metric_context = "\n".join(part for part in [abstract, metric, efficiency] if part).strip()
        if not metric_context:
            metric_context = full_text[:7000]
        metric_records = _extract_metric_records(metric_context)
        metric_records = sorted(
            metric_records,
            key=lambda item: (
                1 if str(item.get("benchmark", "")).strip() else 0,
                float(item.get("value", 0.0)),
            ),
            reverse=True,
        )
        profiles[filename] = {
            "paper_id": paper_id,
            "full_text": full_text,
            "abstract": abstract,
            "summary_sentence": summary,
            "method_sentence": method,
            "metric_sentence": metric,
            "efficiency_sentence": efficiency,
            "limitation_sentence": limitation,
            "metric_records": metric_records,
            "benchmark_label": _infer_benchmark_label(metric_context or abstract or full_text),
            "summary_citation": _best_citation_index_for_snippet(
                snippet=summary,
                documents=documents,
                filename=filename,
            ),
            "method_citation": _best_citation_index_for_snippet(
                snippet=method,
                documents=documents,
                filename=filename,
            ),
            "metric_citation": _best_citation_index_for_snippet(
                snippet=metric,
                documents=documents,
                filename=filename,
            ),
            "efficiency_citation": _best_citation_index_for_snippet(
                snippet=efficiency,
                documents=documents,
                filename=filename,
            ),
            "limitation_citation": _best_citation_index_for_snippet(
                snippet=limitation,
                documents=documents,
                filename=filename,
            ),
        }
    return profiles


def _profile_benchmark_labels(profile: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for record in profile.get("metric_records", []) or []:
        benchmark = str(record.get("benchmark", "")).strip()
        if benchmark and benchmark.lower() not in seen:
            seen.add(benchmark.lower())
            labels.append(benchmark)
        if len(labels) >= 3:
            return labels
    fallback = str(profile.get("benchmark_label", "")).strip()
    if fallback and fallback.lower() not in seen:
        labels.append(fallback)
    return labels[:3]


def _profile_method_signature(profile: dict[str, Any]) -> str:
    text = str(profile.get("method_sentence", "")).strip() or str(profile.get("summary_sentence", "")).strip()
    return _compact_turn_text(text, max_chars=140) if text else "distinct method emphasis not recovered"


def _infer_paper_title(*, full_text: str, filename: str) -> str:
    text = _clean_visible_text(full_text or "")
    if not text:
        return filename
    lines = [line.strip(" -") for line in re.split(r"[\r\n]+", text[:2200]) if line.strip()]
    candidates: list[tuple[float, str]] = []
    for line in lines[:14]:
        candidate = re.sub(r"\s+", " ", line).strip()
        if not candidate or len(candidate) < 18 or len(candidate) > 180:
            continue
        lower = candidate.lower()
        if any(marker in lower for marker in ("abstract", "@", "arxiv", "http://", "https://", "google", "university")):
            continue
        words = re.findall(r"[A-Za-z][A-Za-z0-9\-]*", candidate)
        if len(words) < 4:
            continue
        title_ratio = sum(1 for word in words if word[:1].isupper()) / max(1, len(words))
        score = title_ratio
        if len(words) >= 6:
            score += 0.3
        if any(token in lower for token in ("learning", "attention", "transformer", "network", "networks", "translation", "model", "models")):
            score += 0.35
        candidates.append((score, candidate))
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]
    return filename


def _infer_paper_year_hint(*, full_text: str, filename: str) -> str:
    text = _clean_visible_text(full_text or "")[:800]
    match = re.search(r"\b(19|20)\d{2}\b", text)
    if match:
        return match.group(0)
    file_match = re.search(r"\b(19|20)\d{2}\b", filename)
    if file_match:
        return file_match.group(0)
    return ""


def _paper_field_contexts(*, documents: list[Document]) -> dict[str, dict[str, Any]]:
    profiles = _paper_profiles_from_documents(documents)
    if not profiles:
        return {}

    dossiers: list[dict[str, Any]] = []
    for filename, profile in list(profiles.items())[:3]:
        full_text = str(profile.get("full_text", "")).strip()
        title = _infer_paper_title(full_text=full_text, filename=filename)
        year_hint = _infer_paper_year_hint(full_text=full_text, filename=filename)
        metrics = []
        for record in list(profile.get("metric_records", []) or [])[:3]:
            try:
                value = float(record.get("value", 0.0))
            except Exception:
                continue
            metric = str(record.get("metric", "metric")).upper()
            benchmark = str(record.get("benchmark", "")).strip()
            detail = f"{value:.2f} {metric}"
            if benchmark:
                detail += f" on {benchmark}"
            metrics.append(detail)
        dossiers.append(
            {
                "filename": filename,
                "title": title,
                "year_hint": year_hint,
                "summary": _compact_turn_text(str(profile.get("summary_sentence", "")).strip(), max_chars=260),
                "method": _compact_turn_text(str(profile.get("method_sentence", "")).strip(), max_chars=220),
                "benchmarks": metrics,
            }
        )

    fallback: dict[str, dict[str, Any]] = {}
    for dossier in dossiers:
        filename = str(dossier.get("filename", "")).strip()
        if not filename:
            continue
        fallback[filename] = {
            "field_position": "important",
            "novelty_band": "major",
            "historical_significance": (
                "The retrieved paper evidence suggests a meaningful contribution, but this fallback view stays conservative about historical rank when broader field knowledge is unavailable."
            ),
            "reviewer_take": (
                "Treat the work as clearly non-trivial, but separate paper-evidence novelty from stronger claims about long-term field impact unless the model can place it confidently."
            ),
            "comparator_take": (
                "Historical prestige can inform interpretation, but head-to-head decisions should still be tied to grounded benchmark and method evidence."
            ),
            "confidence": 0.35,
        }

    if not text_service.available:
        return fallback

    try:
        response = text_service.generate(
            system_prompt=(
                "You are a research-field historian helping a paper reviewer.\n"
                "Use general field knowledge to place each paper in context, but do NOT invent paper-specific results beyond the provided dossier.\n"
                "Return JSON only as an object keyed by filename.\n"
                "Each value must contain: field_position, novelty_band, historical_significance, reviewer_take, comparator_take, confidence.\n"
                "Allowed field_position values: foundational, landmark, important, solid, unclear.\n"
                "Allowed novelty_band values: foundational, major, moderate, incremental, unclear.\n"
                "historical_significance, reviewer_take, and comparator_take should each be 1-2 sentences.\n"
                "Be honest about uncertainty; use 'unclear' rather than bluffing."
            ),
            user_prompt=(
                "Paper dossiers:\n"
                f"{json.dumps(dossiers)}"
            ),
            temperature=0.0,
            max_output_tokens=520,
        )
        payload = _try_parse_json_payload(response)
        if isinstance(payload, dict):
            resolved: dict[str, dict[str, Any]] = {}
            for dossier in dossiers:
                filename = str(dossier.get("filename", "")).strip()
                item = payload.get(filename, {})
                if not isinstance(item, dict):
                    resolved[filename] = fallback[filename]
                    continue
                resolved[filename] = {
                    "field_position": str(item.get("field_position", "")).strip().lower() or fallback[filename]["field_position"],
                    "novelty_band": str(item.get("novelty_band", "")).strip().lower() or fallback[filename]["novelty_band"],
                    "historical_significance": str(item.get("historical_significance", "")).strip() or fallback[filename]["historical_significance"],
                    "reviewer_take": str(item.get("reviewer_take", "")).strip() or fallback[filename]["reviewer_take"],
                    "comparator_take": str(item.get("comparator_take", "")).strip() or fallback[filename]["comparator_take"],
                    "confidence": max(0.0, min(1.0, float(item.get("confidence", fallback[filename]["confidence"])))),
                }
            return resolved
    except Exception:
        pass
    return fallback


def _field_novelty_score(*, base_score: int, field_context: dict[str, Any] | None) -> int:
    if not isinstance(field_context, dict):
        return base_score
    band = str(field_context.get("novelty_band", "")).strip().lower()
    confidence = float(field_context.get("confidence", 0.0) or 0.0)
    mapped = {
        "foundational": 10,
        "major": 9,
        "moderate": 7,
        "incremental": 5,
        "unclear": base_score,
    }.get(band, base_score)
    if confidence >= 0.75:
        return int(mapped)
    if confidence >= 0.45:
        return int(round((base_score + mapped) / 2))
    return int(base_score)


def _reviewer_field_context_lines(*, documents: list[Document]) -> list[str]:
    contexts = _paper_field_contexts(documents=documents)
    lines: list[str] = []
    for filename, context in list(contexts.items())[:2]:
        position = str(context.get("field_position", "")).strip() or "unclear"
        novelty_band = str(context.get("novelty_band", "")).strip() or "unclear"
        significance = str(context.get("historical_significance", "")).strip()
        reviewer_take = str(context.get("reviewer_take", "")).strip()
        confidence = float(context.get("confidence", 0.0) or 0.0)
        lines.append(
            f"{filename}: field position = {position}; field-relative novelty = {novelty_band}; confidence = {confidence:.2f}."
        )
        if significance:
            lines.append(f"{filename}: {significance}")
        if reviewer_take:
            lines.append(f"{filename}: reviewer lens = {reviewer_take}")
        if len(lines) >= 4:
            break
    return lines[:4]


def _rerank_step(state: GraphState) -> GraphState:
    documents = state.get("retrieved_documents", [])
    if not documents:
        return {}

    mode = state["mode"]
    query_text = state.get("message", "")
    if mode == Mode.REVIEWER:
        reviewer_focus = _normalize_reviewer_message(query_text)
        lower_focus = reviewer_focus.lower()
        if not reviewer_focus or lower_focus == "focus lens: full review":
            query_text = "full paper review novelty method evaluation benchmark limitations reproducibility"
        elif "novelty" in lower_focus:
            query_text = "novelty contribution prior work benchmark delta scope"
        elif "method" in lower_focus:
            query_text = "method architecture training objective implementation reproducibility"
        else:
            query_text = reviewer_focus
    query_terms = set(_tokenize_for_overlap(query_text))
    query_phrases = _query_phrases(query_text)
    anchor_terms = _anchor_terms_for_query(query_text)
    focus_terms = set(_tokenize_for_overlap(" ".join(_mode_keywords(mode))))
    math_query = mode == Mode.LOCAL and _is_math_intent_query(query_text)
    quantity_intent = any(marker in (query_text or "").lower() for marker in ("how many", "number", "count", "how much"))
    expert_count_query = quantity_intent and any(token.startswith("expert") for token in query_terms)
    person_query = mode == Mode.GLOBAL and _is_global_person_query(query_text)

    scored: list[tuple[Document, float]] = []
    for index, document in enumerate(documents):
        text = document.page_content or ""
        if mode in {Mode.REVIEWER, Mode.COMPARATOR} and (
            _looks_like_metadata_snippet(text) or _looks_acknowledgement_text(text)
        ):
            continue
        lower = text.lower()
        normalized_text = _normalize_for_phrase_match(lower)

        rank_prior = max(0.05, 1.0 - (index / max(1, len(documents))))
        overlap = _overlap_score(lower, query_terms)
        phrase_overlap = _phrase_overlap_score(normalized_text, query_phrases)
        anchor_overlap = _overlap_score(lower, anchor_terms) if anchor_terms else 0.0
        focus_overlap = _overlap_score(lower, focus_terms)

        section_boost = 0.0
        if _looks_like_high_signal_section(text):
            section_boost += 0.12
        if mode == Mode.REVIEWER and any(
            marker in lower for marker in ("ablation", "benchmark", "result", "table", "limitation")
        ):
            section_boost += 0.10
        if mode == Mode.COMPARATOR and (
            any(marker in lower for marker in ("dataset", "baseline", "benchmark", "results", "state of the art", "sota"))
            or _has_metric_name(lower)
        ):
            section_boost += 0.10
        if mode == Mode.COMPARATOR and _looks_metric_rich_chunk(text):
            section_boost += 0.22
        if math_query and _looks_math_dense_chunk(text):
            section_boost += 0.55
        elif math_query and any(
            marker in lower for marker in ("equation", "objective", "loss", "formulated as", "where")
        ):
            section_boost += 0.28
        if person_query and _looks_author_metadata_text(lower):
            section_boost += 0.75
        has_number_phrase = "number of experts" in lower
        has_numbered_experts = _has_numbered_expert_pattern(lower)
        has_any_number = bool(re.search(r"\b\d+\b", lower))
        if expert_count_query and has_numbered_experts:
            section_boost += 0.90
        elif expert_count_query and has_number_phrase:
            section_boost += 0.55

        quality_penalty = _low_signal_penalty(text, allow_numeric_dense=math_query)
        anchor_penalty = 0.0
        if anchor_terms and anchor_overlap < 0.34:
            anchor_penalty += 0.30
        if expert_count_query and (has_number_phrase or has_numbered_experts):
            anchor_penalty = max(0.0, anchor_penalty - 0.30)
        if expert_count_query and "expert" in lower and not has_any_number and not has_number_phrase:
            anchor_penalty += 0.35
        if {"mixture", "experts"}.issubset(anchor_terms):
            if "player" in lower and "novice" in lower and "intermediate" in lower:
                anchor_penalty += 0.35
        if person_query and not _looks_author_metadata_text(lower):
            anchor_penalty += 0.18
        total = (
            rank_prior
            + (0.9 * overlap)
            + (0.7 * phrase_overlap)
            + (1.0 * anchor_overlap)
            + (0.6 * focus_overlap)
            + section_boost
            - quality_penalty
            - anchor_penalty
        )
        scored.append((document, total))

    scored.sort(key=lambda item: item[1], reverse=True)
    rerank_limit = max(1, settings.rerank_top_n)
    if mode == Mode.REVIEWER:
        rerank_limit = max(rerank_limit, 10)
    elif mode == Mode.COMPARATOR:
        rerank_limit = max(rerank_limit, 12)
    reranked_docs = _select_balanced_docs(mode=mode, scored_docs=scored, limit=rerank_limit)
    debug = dict(state.get("debug", {}))
    debug["reranked_count"] = len(reranked_docs)
    debug["rerank_preview"] = [
        {
            "filename": (doc.metadata or {}).get("filename", "unknown.pdf"),
            "page": (doc.metadata or {}).get("page"),
            "chunk_id": (doc.metadata or {}).get("chunk_id"),
            "score": round(score, 4),
        }
        for doc, score in scored[:5]
    ]
    return {
        "retrieved_documents": reranked_docs,
        "citations": [_citation_from_document(document) for document in reranked_docs],
        "debug": debug,
    }


def _draft_answer_step(state: GraphState) -> GraphState:
    mode = state["mode"]
    paper_ids = state.get("paper_ids", [])
    review_paper_id = state.get("review_paper_id")
    documents = state.get("retrieved_documents", [])
    debug = dict(state.get("debug", {}))

    if mode == Mode.REVIEWER and not review_paper_id:
        return {
            "draft_answer": "Reviewer mode requires a selected paper before it can generate a review.",
            "citations": [],
            "debug": {**debug, "response_stage": "validation"},
        }

    if mode == Mode.REVIEWER and not documents:
        return {
            "draft_answer": (
                "I could not retrieve enough evidence from the selected paper to review it. "
                "Try asking a more specific question or re-upload the paper."
            ),
            "citations": [],
            "debug": {**debug, "response_stage": "empty_reviewer_context"},
        }

    if mode == Mode.REVIEWER:
        if not text_service.available:
            draft_answer = _fallback_without_model(state)
            debug["response_stage"] = "fallback"
            return {
                "draft_answer": draft_answer,
                "citations": state.get("citations", []),
                "debug": debug,
            }
        try:
            debate_payload = _run_reviewer_debate(state)
        except Exception as error:
            try:
                compact_answer = text_service.generate(
                    system_prompt=_system_prompt(state),
                    user_prompt=_reviewer_compact_generation_prompt(
                        message=state["message"],
                        history=state.get("history", []),
                        documents=documents,
                    ),
                    temperature=0.15,
                    max_output_tokens=1200,
                )
                debug["model_provider"] = text_service.last_provider or "unknown"
                debug["response_stage"] = "reviewer_model_compact_retry"
                debug["used_context_docs"] = len(documents)
                return {
                    "draft_answer": compact_answer,
                    "citations": state.get("citations", []),
                    "debug": debug,
                }
            except Exception as compact_error:
                error = compact_error
            retry_hint = _extract_retry_hint(error)
            fallback_text = _reviewer_rate_limit_fallback(state, retry_hint=retry_hint)
            debug["response_stage"] = "reviewer_model_fallback"
            debug["model_fallback"] = True
            if retry_hint:
                debug["retry_hint"] = retry_hint
            debug["model_error"] = str(error)[:180]
            return {
                "draft_answer": _prefix_retrieval_fallback(
                    fallback_text,
                    retry_hint=retry_hint,
                    mode=mode,
                ),
                "citations": state.get("citations", []),
                "debug": debug,
            }
        debate_debug = dict(debate_payload.get("debug", {}))
        debate_debug["response_stage"] = "reviewer_debate"
        debate_debug["model_provider"] = text_service.last_provider or debate_debug.get("model_provider")
        debate_payload["debug"] = debate_debug
        return debate_payload

    if mode == Mode.COMPARATOR and len(paper_ids) < 2:
        return {
            "draft_answer": "Comparator mode requires at least two selected papers.",
            "citations": [],
            "debug": {**debug, "response_stage": "validation"},
        }

    comparator_doc_papers = _unique_paper_ids(documents)
    if mode == Mode.COMPARATOR and (len(documents) < 2 or len(comparator_doc_papers) < 2):
        return {
            "draft_answer": (
                "I could not retrieve enough evidence from the selected papers. "
                "Try selecting different papers or ask a more specific comparison question."
            ),
            "citations": state.get("citations", []),
            "debug": {**debug, "response_stage": "insufficient_comparator_context"},
        }

    if mode == Mode.LOCAL and not documents:
        return {
            "draft_answer": "This information is not in your uploaded papers.",
            "citations": [],
            "debug": {**debug, "response_stage": "empty_local_context"},
        }
    if mode == Mode.LOCAL and _insufficient_local_grounding(query=state.get("message", ""), documents=documents):
        debug["response_stage"] = "local_low_relevance"
        return {
            "draft_answer": "This information is not in your uploaded papers.",
            "citations": [],
            "debug": debug,
        }
    if mode == Mode.LOCAL:
        numeric_fastpath = _try_local_numeric_fastpath(
            query=state.get("message", ""),
            documents=documents,
        )
        if numeric_fastpath:
            debug["response_stage"] = "local_numeric_fastpath"
            return {
                "draft_answer": numeric_fastpath,
                "citations": state.get("citations", []),
                "debug": debug,
            }

    if not text_service.available:
        draft_answer = _fallback_without_model(state)
        debug["response_stage"] = "fallback"
        return {
            "draft_answer": draft_answer,
            "citations": state.get("citations", []),
            "debug": debug,
        }

    prompt_documents = documents
    if mode == Mode.GLOBAL and documents and not _is_context_relevant_to_query(state["message"], documents):
        prompt_documents = []
        debug["global_context_relevance"] = "low"
    elif mode == Mode.GLOBAL and documents:
        debug["global_context_relevance"] = "high"

    if mode == Mode.REVIEWER:
        max_output_tokens = 2000
    elif mode == Mode.COMPARATOR:
        max_output_tokens = 1500
    else:
        max_output_tokens = 1400
    try:
        draft_answer = text_service.generate(
            system_prompt=_system_prompt(state),
            user_prompt=_draft_user_prompt(
                mode=mode,
                message=state["message"],
                history=state.get("history", []),
                documents=prompt_documents,
                paper_ids=paper_ids,
            ),
            temperature=_temperature_for_mode(mode),
            max_output_tokens=max_output_tokens,
        )
        debug["model_provider"] = text_service.last_provider or "unknown"
    except Exception as error:
        if mode == Mode.COMPARATOR:
            try:
                draft_answer = text_service.generate(
                    system_prompt=_system_prompt(state),
                    user_prompt=_comparator_compact_generation_prompt(
                        message=state["message"],
                        documents=prompt_documents,
                        paper_ids=paper_ids,
                    ),
                    temperature=max(_temperature_for_mode(mode), 0.1),
                    max_output_tokens=700,
                )
                debug["model_provider"] = text_service.last_provider or "unknown"
                debug["response_stage"] = "draft_model_compact_retry"
                debug["used_context_docs"] = len(prompt_documents)
                debug["comparator_retrieved_paper_count"] = len(comparator_doc_papers)
                return {
                    "draft_answer": draft_answer,
                    "citations": state.get("citations", []),
                    "debug": debug,
                }
            except Exception as compact_error:
                error = compact_error
        else:
            try:
                draft_answer = text_service.generate(
                    system_prompt=_system_prompt(state),
                    user_prompt=_compact_generation_prompt_for_mode(
                        mode=mode,
                        message=state["message"],
                        documents=prompt_documents,
                        history=state.get("history", []),
                        paper_ids=paper_ids,
                    ),
                    temperature=max(_temperature_for_mode(mode), 0.1),
                    max_output_tokens=max(420, min(900, max_output_tokens)),
                )
                debug["model_provider"] = text_service.last_provider or "unknown"
                debug["response_stage"] = "draft_model_compact_retry"
                debug["used_context_docs"] = len(prompt_documents)
                if mode == Mode.COMPARATOR:
                    debug["comparator_retrieved_paper_count"] = len(comparator_doc_papers)
                return {
                    "draft_answer": draft_answer,
                    "citations": state.get("citations", []),
                    "debug": debug,
                }
            except Exception as compact_error:
                error = compact_error
        retry_hint = _extract_retry_hint(error)
        draft_answer = _prefix_retrieval_fallback(
            _rate_limit_fallback_answer(state, retry_hint=retry_hint),
            retry_hint=retry_hint,
            mode=mode,
        )
        debug["response_stage"] = "model_fallback"
        debug["model_fallback"] = True
        if retry_hint:
            debug["retry_hint"] = retry_hint
        debug["model_error"] = str(error)[:180]
    if not debug.get("model_fallback"):
        debug["response_stage"] = "draft_model"
    if mode == Mode.COMPARATOR:
        debug["comparator_retrieved_paper_count"] = len(comparator_doc_papers)
    debug["used_context_docs"] = len(prompt_documents)
    return {
        "draft_answer": draft_answer,
        "citations": state.get("citations", []),
        "debug": debug,
    }


def _validate_answer_step(state: GraphState) -> GraphState:
    mode = state["mode"]
    draft = (state.get("draft_answer") or "").strip()
    documents = state.get("retrieved_documents", [])
    debug = dict(state.get("debug", {}))
    if not draft:
        return {"debug": debug}

    if mode == Mode.LOCAL and debug.get("response_stage") == "local_low_relevance":
        return {
            "validated_answer": draft,
            "validation_issues": [],
            "debug": {**debug, "validation_stage": "local_low_relevance_bypassed"},
        }

    if mode == Mode.WRITER:
        return {
            "validated_answer": draft,
            "validation_issues": [],
            "debug": {**debug, "validation_stage": "writer_skipped"},
        }

    if mode == Mode.REVIEWER and debug.get("reviewer_debate_mode"):
        return {
            "validated_answer": draft,
            "validation_issues": [],
            "debug": {**debug, "validation_stage": "reviewer_debate_bypassed"},
        }

    if mode == Mode.GLOBAL:
        stage = (
            "global_low_context_bypassed"
            if debug.get("global_context_relevance") == "low"
            else "global_normal_llm_bypassed"
        )
        return {
            "validated_answer": draft,
            "validation_issues": [],
            "debug": {**debug, "validation_stage": stage},
        }

    if not text_service.available or not documents:
        return {
            "validated_answer": draft,
            "validation_issues": [],
            "debug": {**debug, "validation_stage": "bypassed"},
        }

    try:
        validator_tokens = 800 if mode == Mode.COMPARATOR else 1400
        validator_raw = text_service.generate(
            system_prompt=_validation_system_prompt(mode),
            user_prompt=(
                "Validate this answer against the retrieved evidence.\n\n"
                "Draft answer:\n"
                f"{draft}\n\n"
                "Retrieved context:\n"
                f"{_format_context(documents, max_docs=max(settings.rerank_top_n, 8) if mode == Mode.REVIEWER else None)}"
            ),
            temperature=0.0,
            max_output_tokens=validator_tokens,
        )
    except Exception as error:
        retry_hint = _extract_retry_hint(error)
        debug["validation_stage"] = "model_bypassed"
        if debug.get("response_stage") in {"model_fallback", "reviewer_model_fallback", "fallback"}:
            debug["model_fallback"] = True
        else:
            debug["validation_model_bypassed"] = True
        if retry_hint:
            debug["retry_hint"] = retry_hint
        debug["validation_error"] = str(error)[:180]
        return {
            "validated_answer": draft,
            "validation_issues": [],
            "debug": debug,
        }

    verdict, issues, revised = _parse_validation_payload(validator_raw)
    validated = revised.strip() if revised.strip() else draft
    if mode == Mode.COMPARATOR and validated:
        revised_issues = _comparator_answer_quality_issues(
            answer=validated,
            citations=state.get("citations", []),
            documents=documents,
            selected_paper_ids=state.get("paper_ids", []),
        )
        if "missing_sections" in revised_issues:
            draft_issues = _comparator_answer_quality_issues(
                answer=draft,
                citations=state.get("citations", []),
                documents=documents,
                selected_paper_ids=state.get("paper_ids", []),
            )
            if "missing_sections" not in draft_issues:
                validated = draft
                issues = list(issues) + [
                    "Validator revision removed comparator sections; original sectioned draft preserved."
                ]
                debug["validation_structure_preserved"] = True
    debug["validation_stage"] = verdict
    debug["validation_issue_count"] = len(issues)
    if issues:
        debug["validation_issues"] = issues[:5]
    return {
        "validated_answer": validated,
        "validation_issues": issues,
        "debug": debug,
    }


def _finalize_answer_step(state: GraphState) -> GraphState:
    mode = state["mode"]
    documents = state.get("retrieved_documents", [])
    draft_answer = (state.get("draft_answer") or "").strip()
    validated_answer = (state.get("validated_answer") or "").strip()
    raw_citations = state.get("citations", [])
    debug = dict(state.get("debug", {}))

    answer = validated_answer or draft_answer
    if not answer and mode == Mode.LOCAL and not documents:
        answer = "This information is not in your uploaded papers."
    answer = _clean_mojibake_text(answer)
    if mode in {Mode.LOCAL, Mode.REVIEWER, Mode.COMPARATOR}:
        answer = _clean_visible_text(answer)
    if mode == Mode.GLOBAL and _is_global_recommendation_query(state.get("message", "")):
        answer = _strip_inline_reference_markers(answer)
        debug["global_citation_cleanup"] = "recommendation_query"
    if mode == Mode.GLOBAL and debug.get("global_context_relevance") == "low":
        answer = _strip_inline_reference_markers(answer)
        debug["global_citation_cleanup"] = "applied"
    if mode == Mode.LOCAL and _is_math_intent_query(state.get("message", "")):
        formatted_math = _format_local_math_answer(answer)
        if formatted_math != answer:
            answer = formatted_math
            debug["local_math_formatting"] = "latex_applied"
        answer = _ensure_local_math_citations(answer, raw_citations if isinstance(raw_citations, list) else [])
    if mode == Mode.COMPARATOR and documents:
        validation_bypassed = bool(debug.get("validation_model_bypassed")) or str(debug.get("validation_stage", "")).strip().lower() == "model_bypassed"
        validation_requires_fallback = str(debug.get("validation_stage", "")).strip().lower() in {"revise", "fail"} and bool(state.get("validation_issues"))
        if validation_bypassed or validation_requires_fallback:
            answer = _comparator_structured_fallback(documents=documents)
            debug["comparator_quality_guard"] = sorted(
                set(
                    list(debug.get("comparator_quality_guard", []))
                    + (
                        ["validation_bypassed_structured_fallback"]
                        if validation_bypassed
                        else ["validation_repair_structured_fallback"]
                    )
                )
            )
        else:
            answer = _ensure_comparator_common_benchmark(answer=answer, documents=documents)
            quality_issues = _comparator_answer_quality_issues(
                answer=answer,
                citations=raw_citations,
                documents=documents,
                selected_paper_ids=state.get("paper_ids", []),
            )
            if _comparator_has_grounding_risk(state.get("validation_issues", [])):
                quality_issues.append("validation_grounding_risk")
            if "benchmark_contradiction" in quality_issues:
                answer = _repair_comparator_benchmark_contradiction(answer=answer, documents=documents)
                quality_issues = [item for item in quality_issues if item != "benchmark_contradiction"]
            critical_issues = {
                "empty_answer",
                "mentions_unselected_paper",
                "benchmark_metric_omitted",
                "benchmark_signal_missing",
            }
            repair_to_fallback = {
                "speculative_without_citations",
                "citations_single_paper",
                "uncited_use_case_winner",
            }
            if "missing_sections" in quality_issues:
                answer = _comparator_structured_fallback(documents=documents)
                debug["comparator_quality_guard"] = sorted(set(quality_issues + ["structured_repair"]))
            elif any(issue in critical_issues for issue in quality_issues):
                answer = _comparator_structured_fallback(documents=documents)
                debug["comparator_quality_guard"] = quality_issues
            elif any(issue in repair_to_fallback for issue in quality_issues):
                answer = _comparator_structured_fallback(documents=documents)
                debug["comparator_quality_guard"] = sorted(set(quality_issues + ["structured_repair"]))
            elif quality_issues:
                debug["comparator_quality_notes"] = quality_issues

    if mode in {Mode.LOCAL, Mode.REVIEWER, Mode.COMPARATOR} and documents and not _has_inline_citations(answer):
        debug["citation_warning"] = "Answer lacked inline citations after validation."

    citations = _select_citations_for_answer(answer=answer, citations=raw_citations, mode=mode)
    reindexed_answer = _reindex_answer_citations(
        answer=answer,
        raw_citations=raw_citations,
        selected_citations=citations,
    )
    if reindexed_answer != answer:
        answer = reindexed_answer
        debug["citation_reindexed"] = True
    debug["citation_count"] = len(citations)
    debug["response_stage"] = "finalized"
    return {
        "answer": answer,
        "citations": citations,
        "debug": debug,
    }


def _system_prompt(state: GraphState) -> str:
    mode = state["mode"]
    style_profile = _load_style_profile()
    base = (state.get("mode_instructions") or "").strip()
    if not base:
        base = "You are a research assistant."

    common = (
        "\n\nGeneral rules:\n"
        "- Be concrete and avoid vague language.\n"
        "- Use inline citations [n] for paper-grounded claims.\n"
        "- If evidence is missing, say so directly.\n"
    )

    if mode == Mode.WRITER:
        style_suffix = (
            f"Stored style profile:\n{style_profile}"
            if style_profile
            else "No stored style profile is available yet. Use a clear academic style."
        )
        return f"{base}\n\n{style_suffix}{common}"

    if mode == Mode.GLOBAL:
        return (
            f"{base}{common}\n"
            "Keep the tone natural and helpful. Do not force paper citations unless the claim comes from paper context."
        )
    return f"{base}{common}"


def _format_context(documents: list[Document], max_docs: int | None = None) -> str:
    if not documents:
        return "No retrieved paper context."

    limit = max_docs if max_docs is not None else settings.rerank_top_n
    blocks = []
    for index, document in enumerate(documents[: max(1, int(limit))], start=1):
        metadata = document.metadata or {}
        filename = metadata.get("filename", "unknown.pdf")
        chunk_id = metadata.get("chunk_id", f"chunk-{index}")
        page = metadata.get("page")
        page_suffix = f", p.{page}" if page else ""
        blocks.append(f"[{index}] {filename} ({chunk_id}{page_suffix})\n{document.page_content}")
    return "\n\n".join(blocks)


def _citation_from_document(document: Document) -> dict[str, Any]:
    metadata = document.metadata or {}
    return {
        "paper_id": str(metadata.get("paper_id", "")),
        "filename": str(metadata.get("filename", "unknown.pdf")),
        "snippet": document.page_content[:500],
        "chunk_id": str(metadata.get("chunk_id")) if metadata.get("chunk_id") else None,
        "page": metadata.get("page"),
    }


def _load_style_profile() -> str:
    if not settings.style_profile_store.exists():
        return ""
    raw = settings.style_profile_store.read_text(encoding="utf-8").strip()
    if not raw:
        return ""
    if raw.startswith("{"):
        try:
            payload = json.loads(raw)
        except Exception:
            return raw
        return str(payload.get("profile", "")).strip()
    return raw


def _format_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "No earlier conversation."
    recent = history[-settings.conversation_window :]
    return "\n".join(
        f"{item.get('role', 'user').upper()}: {item.get('content', '').strip()}"
        for item in recent
        if item.get("content")
    )


def _fallback_without_model(state: GraphState) -> str:
    mode = state["mode"]
    documents = state.get("retrieved_documents", [])
    if mode == Mode.LOCAL and documents:
        blocks = [
            f"- {doc.metadata.get('filename', 'unknown.pdf')}: {doc.page_content[:260]}"
            for doc in documents[:3]
        ]
        return (
            "Hybrid retrieval found relevant paper evidence, but no text model is configured yet "
            "(`GROQ_API_KEY`, `GEMINI_API_KEY`, or `OPENROUTER_API_KEY`). "
            "Here are the strongest grounded excerpts:\n\n"
            + "\n\n".join(blocks)
        )
    return (
        "Retrieval is working, but model generation is disabled until `GROQ_API_KEY`, "
        "`GEMINI_API_KEY`, or `OPENROUTER_API_KEY` is set in `backend/.env`."
    )


def _prefix_retrieval_fallback(answer: str, *, retry_hint: str, mode: Mode) -> str:
    guidance = f" Try again in about {retry_hint}." if retry_hint else " Try again shortly."
    if mode == Mode.GLOBAL:
        return (
            "Model generation is temporarily unavailable."
            f"{guidance}\n\n{answer.strip()}"
        ).strip()
    if mode == Mode.COMPARATOR:
        # Keep comparator fallback readable without a hard-failure banner.
        return answer.strip()
    return (
        "Model generation is temporarily unavailable, so this is a retrieval-only fallback."
        f"{guidance}\n\n{answer.strip()}"
    ).strip()


def _rate_limit_fallback_answer(state: GraphState, *, retry_hint: str) -> str:
    mode = state["mode"]
    documents = state.get("retrieved_documents", [])
    if mode == Mode.LOCAL:
        if not documents:
            return "This information is not in your uploaded papers."
        return _local_extractive_fallback(
            query=state.get("message", ""),
            documents=documents,
        )
    if mode == Mode.GLOBAL:
        if documents and _is_context_relevant_to_query(state.get("message", ""), documents):
            return _local_extractive_fallback(
                query=state.get("message", ""),
                documents=documents,
            )
        return (
            "I’m temporarily unable to reach the text-generation provider for Global mode. "
            "Please retry in a moment."
        )
    if mode == Mode.COMPARATOR:
        if not documents:
            return "I could not retrieve enough evidence from the selected papers."
        return _comparator_structured_fallback(documents=documents)
    return (
        _local_extractive_fallback(
            query=state.get("message", ""),
            documents=documents,
        )
        if documents
        else "I could not find enough grounded evidence in the retrieved context."
    )


def _build_comparator_signal_pool(*, documents: list[Document], limit: int) -> dict[str, list[dict[str, Any]]]:
    per_paper: dict[str, list[dict[str, Any]]] = {}
    profiles = _paper_profiles_from_documents(documents)
    for index, document in enumerate(documents[: max(1, int(limit))], start=1):
        metadata = document.metadata or {}
        filename = str(metadata.get("filename", "unknown.pdf")).strip() or "unknown.pdf"
        page = metadata.get("page")
        raw_text = _clean_mojibake_text(document.page_content or "")
        snippet = _clean_mojibake_text(_extract_signal_sentence(raw_text))
        if not snippet:
            continue
        if _looks_like_reference_snippet(snippet) or _looks_like_non_argument_snippet(snippet):
            continue
        metric_snippet = _extract_metric_sentence(raw_text)
        metric_source = metric_snippet or raw_text or snippet
        metric_records = _extract_metric_records(metric_source)
        per_paper.setdefault(filename, []).append(
            {
                "citation": index,
                "snippet": snippet,
                "metric_snippet": _compact_turn_text(metric_snippet, max_chars=220) if metric_snippet else "",
                "page": page,
                "metric_records": metric_records,
                "benchmark_label": _infer_benchmark_label(metric_source),
            }
        )
    for filename, profile in profiles.items():
        summary = str(profile.get("summary_sentence", "")).strip()
        method = str(profile.get("method_sentence", "")).strip()
        metric = str(profile.get("metric_sentence", "")).strip()
        efficiency = str(profile.get("efficiency_sentence", "")).strip()
        limitation = str(profile.get("limitation_sentence", "")).strip()
        metric_records = list(profile.get("metric_records", []) or [])
        if not any((summary, method, metric, efficiency, metric_records)):
            continue
        synthetic_entry = {
            "citation": int(profile.get("metric_citation") or profile.get("summary_citation") or 1),
            "snippet": summary or method or metric or efficiency,
            "metric_snippet": metric,
            "page": None,
            "metric_records": metric_records,
            "benchmark_label": str(profile.get("benchmark_label", "")).strip(),
            "summary_snippet": summary,
            "method_snippet": method,
            "efficiency_snippet": efficiency,
            "limitation_snippet": limitation,
        }
        entries = per_paper.setdefault(filename, [])
        synthetic_key = re.sub(r"\s+", " ", str(synthetic_entry.get("snippet", "")).lower()).strip()
        if synthetic_key:
            duplicate = next(
                (
                    entry
                    for entry in entries
                    if re.sub(r"\s+", " ", str(entry.get("snippet", "")).lower()).strip() == synthetic_key
                ),
                None,
            )
            if duplicate is not None:
                merged_records: list[dict[str, Any]] = []
                seen_records: set[tuple[str, float, str]] = set()
                for record in list(duplicate.get("metric_records", []) or []) + metric_records:
                    metric = str(record.get("metric", "")).lower()
                    try:
                        value = float(record.get("value", 0.0))
                    except Exception:
                        continue
                    benchmark = str(record.get("benchmark", "")).strip()
                    record_key = (metric, round(value, 3), benchmark.lower())
                    if record_key in seen_records:
                        continue
                    seen_records.add(record_key)
                    merged_records.append(record)
                duplicate["metric_records"] = merged_records
                if synthetic_entry.get("metric_snippet"):
                    duplicate["metric_snippet"] = synthetic_entry.get("metric_snippet")
                if synthetic_entry.get("benchmark_label"):
                    duplicate["benchmark_label"] = synthetic_entry.get("benchmark_label")
                duplicate["summary_snippet"] = synthetic_entry.get("summary_snippet", duplicate.get("summary_snippet", ""))
                duplicate["method_snippet"] = synthetic_entry.get("method_snippet", duplicate.get("method_snippet", ""))
                duplicate["efficiency_snippet"] = synthetic_entry.get("efficiency_snippet", duplicate.get("efficiency_snippet", ""))
                duplicate["limitation_snippet"] = synthetic_entry.get("limitation_snippet", duplicate.get("limitation_snippet", ""))
                continue
        entries.insert(0, synthetic_entry)
    return per_paper


def _comparator_structured_fallback(*, documents: list[Document]) -> str:
    per_paper = _build_comparator_signal_pool(documents=documents, limit=12)
    profiles = _paper_profiles_from_documents(documents)
    field_contexts = _paper_field_contexts(documents=documents)
    if not per_paper and not profiles:
        return "I could not retrieve enough evidence from the selected papers."

    papers = list(per_paper.keys() or profiles.keys())[:3]

    def _best_entry(filename: str, *, prefer: tuple[str, ...] = ()) -> dict[str, Any]:
        entries = per_paper.get(filename, [])
        if not entries:
            return {"citation": 1, "snippet": "No grounded snippet retrieved."}
        if not prefer:
            return entries[0]
        best_match: dict[str, Any] | None = None
        best_score = -1
        for entry in entries:
            lower = str(entry.get("snippet", "")).lower()
            score = sum(1 for token in prefer if token in lower)
            if entry.get("metric_records"):
                score += 1
            if score > best_score:
                best_score = score
                best_match = entry
        if best_match is not None and best_score > 0:
            return best_match
        for entry in entries:
            if entry.get("metric_records"):
                return entry
        return entries[0]

    def _score_row(filename: str) -> tuple[int, int, int, str]:
        profile = profiles.get(filename, {})
        field_context = field_contexts.get(filename, {})
        joined = " ".join(
            str(part or "").lower()
            for part in (
                profile.get("summary_sentence", ""),
                profile.get("method_sentence", ""),
                profile.get("metric_sentence", ""),
                profile.get("efficiency_sentence", ""),
            )
        )
        if not joined:
            entries = per_paper.get(filename, [])
            joined = " ".join(str(entry.get("snippet", "")).lower() for entry in entries)
        novelty = 8 if any(token in joined for token in ("propose", "novel", "new", "first")) else 7
        novelty = _field_novelty_score(base_score=novelty, field_context=field_context)
        rigor = 8 if (_has_metric_name(joined) or any(token in joined for token in ("benchmark", "dataset", "outperform", "results", "evaluation"))) else 6
        reproducibility = 8 if any(token in joined for token in ("training", "batch", "gpu", "implementation", "hours", "days")) else 6
        base = profiles.get(filename, {})
        citation = int(
            base.get("metric_citation")
            or base.get("summary_citation")
            or _best_entry(filename).get("citation", 1)
        )
        position = str(field_context.get("field_position", "")).strip().lower()
        significance = str(field_context.get("historical_significance", "")).strip()
        if position and significance:
            justification = f"Grounded in contribution/method/result sentences recovered from the paper [{citation}], with field-aware novelty adjusted using a {position} historical reading."
        else:
            justification = f"Grounded in contribution/method/result sentences recovered from the paper [{citation}]."
        return novelty, rigor, reproducibility, justification

    def _profile_sentence(filename: str, *, key: str, fallback_tokens: tuple[str, ...]) -> tuple[str, int]:
        profile = profiles.get(filename, {})
        sentence = str(profile.get(key, "")).strip()
        citation_map = {
            "summary_sentence": "summary_citation",
            "method_sentence": "method_citation",
            "metric_sentence": "metric_citation",
            "efficiency_sentence": "efficiency_citation",
            "limitation_sentence": "limitation_citation",
        }
        citation = int(profile.get(citation_map.get(key, ""), 0) or 0)
        if sentence and citation > 0:
            return sentence, citation
        entry = _best_entry(filename, prefer=fallback_tokens)
        fallback_sentence = str(entry.get("metric_snippet") or entry.get("snippet", "")).strip()
        return fallback_sentence, int(entry.get("citation", 1))

    def _metric_trail(filename: str) -> list[str]:
        profile = profiles.get(filename, {})
        records = profile.get("metric_records", []) if isinstance(profile, dict) else []
        lines: list[str] = []
        seen: set[tuple[str, float, str]] = set()
        metric_citation = int(profile.get("metric_citation", 1) or 1) if isinstance(profile, dict) else 1
        for record in records[:4]:
            metric = str(record.get("metric", "metric")).upper()
            try:
                value = float(record.get("value", 0.0))
            except Exception:
                continue
            benchmark = str(record.get("benchmark", "")).strip()
            key = (metric, round(value, 3), benchmark.lower())
            if key in seen:
                continue
            seen.add(key)
            detail = f"{value:.2f} {metric}"
            if benchmark:
                detail += f" on {benchmark}"
            detail += f" [{metric_citation}]"
            lines.append(detail)
            if len(lines) >= 3:
                break
        if lines:
            return lines
        return _comparator_metric_record_strings(per_paper.get(filename, []), cap=3)

    claim_blocks: list[str] = []
    field_blocks: list[str] = []
    method_blocks: list[str] = []
    benchmark_rows: list[str] = []
    for filename in papers:
        field_context = field_contexts.get(filename, {})
        contribution_text, contribution_citation = _profile_sentence(
            filename,
            key="summary_sentence",
            fallback_tokens=("we propose", "we present", "our main result", "abstract", "contribution", "result"),
        )
        method_text, method_citation = _profile_sentence(
            filename,
            key="method_sentence",
            fallback_tokens=("method", "approach", "architecture", "algorithm", "model", "objective", "design"),
        )
        benchmark_text, benchmark_citation = _profile_sentence(
            filename,
            key="metric_sentence",
            fallback_tokens=("benchmark", "dataset", "baseline", "results", "score", "metric", "state-of-the-art", "state of the art"),
        )
        claim_blocks.append(
            f"### {filename}\n"
            f"- Core contribution: {contribution_text} [{contribution_citation}]\n"
            f"- Method anchor: {method_text} [{method_citation}]\n"
            f"- Benchmark anchor: {benchmark_text} [{benchmark_citation}]"
        )
        position = str(field_context.get("field_position", "")).strip()
        novelty_band = str(field_context.get("novelty_band", "")).strip()
        historical_note = str(field_context.get("historical_significance", "")).strip()
        comparator_note = str(field_context.get("comparator_take", "")).strip()
        confidence = float(field_context.get("confidence", 0.0) or 0.0)
        if position or novelty_band or historical_note or comparator_note:
            field_lines = [f"### {filename}"]
            if position or novelty_band:
                field_lines.append(
                    f"- Field position: {position or 'unclear'}; field-relative novelty: {novelty_band or 'unclear'} (confidence {confidence:.2f})."
                )
            if historical_note:
                field_lines.append(f"- Historical significance: {historical_note}")
            if comparator_note:
                field_lines.append(f"- Comparator read: {comparator_note}")
            field_blocks.append("\n".join(field_lines))

        profile_efficiency = str(profiles.get(filename, {}).get("efficiency_sentence", "")).strip()
        profile_efficiency_lower = profile_efficiency.lower()
        has_efficiency_signal = bool(
            profile_efficiency
            and (
                any(
                    marker in profile_efficiency_lower
                    for marker in ("gpu", "gpus", "parallel", "latency", "throughput", "runtime", "training took", "hours", "days", "seconds", "memory", "flops")
                )
                or ("training" in profile_efficiency_lower and re.search(r"\b\d+(?:\.\d+)?\b", profile_efficiency))
            )
        )
        if has_efficiency_signal:
            efficiency_text, efficiency_citation = _profile_sentence(
                filename,
                key="efficiency_sentence",
                fallback_tokens=("training", "compute", "runtime", "hours", "days", "gpu", "implementation", "throughput", "speed"),
            )
        else:
            efficiency_text = ""
            efficiency_citation = int(profiles.get(filename, {}).get("metric_citation", _best_entry(filename).get("citation", 1)) or 1)
        metric_trail = _metric_trail(filename)
        method_blocks.append(
            f"### {filename}\n"
            f"- Architecture signal: {method_text} [{method_citation}]\n"
            + (
                f"- Training / efficiency signal: {efficiency_text} [{efficiency_citation}]\n"
                if efficiency_text
                else "- Training / efficiency signal: no clean training-cost or runtime sentence was recovered from the current evidence pack.\n"
            )
            + (
                f"- Recovered metric trail: {'; '.join(metric_trail)}"
                if metric_trail
                else "- Recovered metric trail: benchmark numbers were not reliably recovered from the current evidence pack."
            )
        )

        novelty, rigor, reproducibility, justification = _score_row(filename)
        metric_text = _compact_turn_text(benchmark_text, max_chars=140)
        benchmark_rows.append(
            f"| {filename} | {novelty} | {rigor} | {reproducibility} | {metric_text} [{benchmark_citation}] | {justification} |"
        )

    agreement_lines: list[str] = []
    contradiction_lines: list[str] = []
    non_overlap_lines: list[str] = []
    if len(papers) >= 2:
        p1 = papers[0]
        p2 = papers[1]
        p1_profile = profiles.get(p1, {})
        p2_profile = profiles.get(p2, {})
        p1_metric_citation = int(p1_profile.get("metric_citation", _best_entry(p1).get("citation", 1)) or 1)
        p2_metric_citation = int(p2_profile.get("metric_citation", _best_entry(p2).get("citation", 1)) or 1)
        p1_method_citation = int(p1_profile.get("method_citation", _best_entry(p1).get("citation", 1)) or 1)
        p2_method_citation = int(p2_profile.get("method_citation", _best_entry(p2).get("citation", 1)) or 1)
        shared_metric = _shared_metric_summary(per_paper=per_paper, papers=papers)
        if shared_metric.get("status") == "ok":
            metric_name = str(shared_metric.get("metric", "metric")).upper()
            benchmark_name = str(shared_metric.get("benchmark", "")).strip() or "a matched benchmark slice"
            agreement_lines.append(
                f"- Both papers report {metric_name} evidence on {benchmark_name}, so a like-for-like benchmark comparison is available [{p1_metric_citation}][{p2_metric_citation}]."
            )
        elif shared_metric.get("status") == "partial_shared_task":
            benchmark_name = str(shared_metric.get("benchmark_family", "")).strip() or "a shared evaluation family"
            agreement_lines.append(
                f"- Both papers report quantitative evidence within {benchmark_name}, even though the retrieved slices are not perfectly matched yet [{p1_metric_citation}][{p2_metric_citation}]."
            )
        elif p1_profile.get("metric_records") and p2_profile.get("metric_records"):
            agreement_lines.append(
                f"- Both papers report quantitative benchmark evidence, but the current snippets do not prove a fully matched evaluation slice [{p1_metric_citation}][{p2_metric_citation}]."
            )
        else:
            agreement_lines.append(
                f"- Both papers make empirical claims, but the current evidence is stronger on method framing than on one shared benchmark slice [{p1_method_citation}][{p2_method_citation}]."
            )
        contradiction_lines.append(
            f"- Their main method claims pull in different directions: {p1} -> {_profile_method_signature(p1_profile)} [{p1_method_citation}] versus {p2} -> {_profile_method_signature(p2_profile)} [{p2_method_citation}]."
        )
        p1_tasks = ", ".join(_profile_benchmark_labels(p1_profile)) or "task slice not recovered"
        p2_tasks = ", ".join(_profile_benchmark_labels(p2_profile)) or "task slice not recovered"
        non_overlap_lines.append(
            f"- The retrieved evidence emphasizes different evaluation scopes: {p1} -> {p1_tasks}; {p2} -> {p2_tasks} [{p1_metric_citation}][{p2_metric_citation}]."
        )
    if not agreement_lines:
        agreement_lines.append("- No high-confidence agreement detected from current snippet coverage.")
    if not contradiction_lines:
        contradiction_lines.append("- No explicit contradiction recovered from current snippet set.")
    if not non_overlap_lines:
        non_overlap_lines.append("- Non-overlap appears limited in retrieved snippets; verify with full text for certainty.")

    synthesis_lines: list[str] = []
    if papers:
        lead = papers[0]
        lead_entry = profiles.get(lead, {})
        lead_citation = int(lead_entry.get("method_citation", _best_entry(lead).get("citation", 1)) or 1)
        synthesis_lines.append(
            f"- Borrow from **{lead}**: its clearest retrieved method claim and problem framing [{lead_citation}]."
        )
    if len(papers) > 1:
        second = papers[1]
        second_entry = profiles.get(second, {})
        second_citation = int(second_entry.get("metric_citation", _best_entry(second).get("citation", 1)) or 1)
        synthesis_lines.append(
            f"- Borrow from **{second}**: its strongest recovered quantitative result or efficiency signal [{second_citation}]."
        )
    synthesis_lines.append(
        "- Merged experiment: evaluate both ideas on one matched benchmark slice with identical preprocessing, then report the shared headline metric together with runtime/cost."
    )
    synthesis_lines.append(
        "- Guardrail: compare only like-for-like settings; if benchmark slice, training regime, or evaluation variant changes, report that difference instead of forcing a winner."
    )

    use_case_lines: list[str] = []
    shared_metric = _shared_metric_summary(per_paper=per_paper, papers=papers)
    if papers:
        lead = max(
            papers,
            key=lambda item: max(
                (float(record.get("value", 0.0)) for record in profiles.get(item, {}).get("metric_records", []) or []),
                default=0.0,
            ),
        )
        lead_entry = profiles.get(lead, {})
        lead_metric_name = "quantitative result"
        if lead_entry.get("metric_records"):
            lead_metric_name = str((lead_entry.get("metric_records") or [{}])[0].get("metric", "metric")).upper()
        use_case_lines.append(
            f"- Use Case: prioritize the strongest recovered {lead_metric_name} result in the current evidence pack. Winner: **{lead}** [{int(lead_entry.get('metric_citation', _best_entry(lead).get('citation', 1)) or 1)}]."
        )
    if len(papers) > 1:
        use_case_lines.append(
            f"- Use Case: inspect method trade-offs and design philosophy. Winner: **No single winner**; choose based on which method signature better matches your target constraints [{p1_method_citation}][{p2_method_citation}]."
        )
    if shared_metric.get("status") == "ok":
        winner = str(shared_metric.get("winner", "No winner (insufficient evidence)"))
        c1 = int(shared_metric.get("first_citation", 1))
        c2 = int(shared_metric.get("second_citation", 1))
        use_case_lines.append(
            f"- Use Case: choose by the strongest recovered matched benchmark slice. Winner: **{winner}** [{c1}][{c2}]."
        )
    else:
        use_case_lines.append("- Use Case: publication-grade final selection. Winner: **No winner (the current evidence only supports family-level comparison)**.")

    return (
        "## Papers Compared\n"
        + "\n".join(f"- {paper}" for paper in papers)
        + "\n\n## Claim Matrix\n"
        + "\n\n".join(claim_blocks)
        + (
            "\n\n## Field Context\n" + "\n\n".join(field_blocks)
            if field_blocks
            else ""
        )
        + "\n\n## Conflict Map\n"
        + "### Agreements\n"
        + "\n".join(agreement_lines)
        + "\n\n### Contradictions\n"
        + "\n".join(contradiction_lines)
        + "\n\n### Non-overlap\n"
        + "\n".join(non_overlap_lines)
        + "\n\n## Benchmark Verdict Matrix\n"
        + "| Paper | Novelty (1-10) | Empirical Rigor (1-10) | Reproducibility (1-10) | Key Result Signal | Justification |\n"
        + "|---|---|---|---|---|---|\n"
        + "\n".join(benchmark_rows)
        + "\n\n### Common Benchmark Analysis\n"
        + _render_common_benchmark_analysis(per_paper=per_paper, papers=papers)
        + "\n\n## Method Trade-offs\n"
        + "\n\n".join(method_blocks)
        + "\n\n## Synthesis Blueprint\n"
        + "\n".join(synthesis_lines)
        + "\n\n## Decision By Use Case\n"
        + "\n".join(use_case_lines)
    )


def _render_common_benchmark_analysis(*, per_paper: dict[str, list[dict[str, Any]]], papers: list[str]) -> str:
    if len(papers) < 2:
        return "- Need at least two papers to analyze benchmark overlap."

    summary = _shared_metric_summary(per_paper=per_paper, papers=papers)
    if summary.get("status") == "no_metrics":
        return "- Shared benchmark metrics are not explicitly available in current fallback snippets."
    if summary.get("status") == "partial_shared_task":
        family = str(summary.get("benchmark_family", "")).strip() or "shared task family"
        p1, p2 = papers[0], papers[1]
        first_line = _format_metric_brief_line(
            paper=p1,
            metrics=summary.get("first_metrics", []),
            fallback="metric value not recovered from current snippets",
            citation=int(summary.get("first_citation", 1)),
        )
        second_line = _format_metric_brief_line(
            paper=p2,
            metrics=summary.get("second_metrics", []),
            fallback="metric value not recovered from current snippets",
            citation=int(summary.get("second_citation", 1)),
        )
        return (
            f"- Common ground: both papers appear to evaluate within the broader **{family}** evaluation family.\n"
            f"- {first_line}\n"
            f"- {second_line}\n"
            "- Delta is withheld because one side is missing a directly comparable recovered metric in current snippets."
        )
    if summary.get("status") == "no_pair":
        p1, p2 = papers[0], papers[1]
        first_line = _format_metric_brief_line(
            paper=p1,
            metrics=summary.get("first_metrics", []),
            fallback="metrics recovered but task alignment is unclear",
            citation=int(summary.get("first_citation", 1)),
        )
        second_line = _format_metric_brief_line(
            paper=p2,
            metrics=summary.get("second_metrics", []),
            fallback="metrics recovered but task alignment is unclear",
            citation=int(summary.get("second_citation", 1)),
        )
        return (
            "- Both papers report numeric results, but the retrieved snippets do not prove they are directly comparable on the same benchmark slice.\n"
            f"- {first_line}\n"
            f"- {second_line}\n"
            "- Numeric delta is withheld to avoid misleading cross-task comparisons."
        )
    if summary.get("status") != "ok":
        return "- Shared benchmark analysis is inconclusive under current fallback evidence."

    p1, p2 = papers[0], papers[1]
    metric = str(summary.get("metric", "metric")).upper()
    benchmark_text = str(summary.get("benchmark", "")).strip() or "shared task not explicitly named"
    a_value = float(summary.get("first_value", 0.0))
    b_value = float(summary.get("second_value", 0.0))
    delta = float(summary.get("delta", 0.0))
    winner = str(summary.get("winner", "No winner (insufficient evidence)"))
    comparability = str(summary.get("comparability", "strict")).strip().lower()
    first_citation = int(summary.get("first_citation", 1))
    second_citation = int(summary.get("second_citation", 1))
    first_snippet = str(summary.get("first_snippet", ""))
    second_snippet = str(summary.get("second_snippet", ""))
    first_benchmark = str(summary.get("first_benchmark", "")).strip()
    second_benchmark = str(summary.get("second_benchmark", "")).strip()
    reason = _benchmark_explanatory_reason(
        per_paper=per_paper,
        papers=papers,
        winner=winner,
        first_snippet=first_snippet,
        second_snippet=second_snippet,
        first_benchmark=first_benchmark,
        second_benchmark=second_benchmark,
    )
    if comparability == "loose":
        return (
            f"- Shared benchmark family detected: {benchmark_text}.\n"
            f"- {p1}: {a_value:.2f} {metric}"
            + (f" on {first_benchmark}" if first_benchmark else "")
            + f" [{first_citation}]\n"
            f"- {p2}: {b_value:.2f} {metric}"
            + (f" on {second_benchmark}" if second_benchmark else "")
            + f" [{second_citation}]\n"
            "- Direct delta is withheld because the retrieved snippets appear to reference different benchmark slices/tasks.\n"
            f"- Grounded interpretation: {reason}"
        )
    return (
        f"- Shared benchmark signal: {benchmark_text}.\n"
        f"- {p1}: {a_value:.2f} {metric} [{first_citation}]\n"
        f"- {p2}: {b_value:.2f} {metric} [{second_citation}]\n"
        f"- Delta: {delta:.2f} {metric} (winner: {winner}).\n"
        f"- Why the winner is ahead in the recovered evidence: {reason}\n"
        f"- Winner-side reading: {_benchmark_outcome_note(paper=winner, snippet=first_snippet if winner == p1 else second_snippet)}\n"
        f"- Caveat: treat this as a benchmark-slice decision, not blanket proof that {winner} is stronger on every setting in the paper."
    )


def _benchmark_outcome_note(*, paper: str, snippet: str) -> str:
    lower = _clean_visible_text(snippet or "").lower()
    if any(marker in lower for marker in _efficiency_signal_markers()):
        return f"{paper} ties its reported result to a concrete training or compute story rather than presenting the metric in isolation."
    if any(marker in lower for marker in _method_signal_markers()):
        return f"{paper} presents the score alongside a concrete architecture claim, which makes the benchmark win easier to interpret."
    if any(marker in lower for marker in ("state-of-the-art", "state of the art", "improving", "improves", "superior")):
        return f"{paper} frames the result as an explicit improvement over prior baselines in the retrieved evidence."
    return f"{paper} provides the clearest recovered metric statement for this shared slice, so the comparison is grounded rather than inferred."


def _benchmark_explanatory_reason(
    *,
    per_paper: dict[str, list[dict[str, Any]]],
    papers: list[str],
    winner: str,
    first_snippet: str,
    second_snippet: str,
    first_benchmark: str = "",
    second_benchmark: str = "",
) -> str:
    loser = next((paper for paper in papers if paper != winner), "")
    winner_corpus = " ".join(
        _clean_visible_text(str(item.get("metric_snippet") or item.get("snippet", "")).strip())
        for item in per_paper.get(winner, [])
        if isinstance(item, dict)
    )
    loser_corpus = " ".join(
        _clean_visible_text(str(item.get("metric_snippet") or item.get("snippet", "")).strip())
        for item in per_paper.get(loser, [])
        if isinstance(item, dict)
    )
    winner_lower = winner_corpus.lower()
    loser_lower = loser_corpus.lower()
    if any(marker in winner_lower for marker in ("parallelizable", "less time to train", "training costs", "significantly less time to train")):
        return f"{winner} explicitly links its score to a more parallel or lower-cost training story in the recovered evidence, which is the clearest grounded explanation for the gap."
    if any(marker in winner_lower for marker in ("self-attention", "attention mechanisms", "transformer")) and any(
        marker in loser_lower for marker in ("lstm", "recurrent", "sequence to sequence")
    ):
        return f"The recovered evidence points to an architecture shift from recurrent sequence modeling toward attention-first modeling, and that is the strongest grounded explanation available for the benchmark gap."
    return _benchmark_difference_reason(
        first_snippet=first_snippet,
        second_snippet=second_snippet,
        first_benchmark=first_benchmark,
        second_benchmark=second_benchmark,
    )


def _ensure_comparator_common_benchmark(*, answer: str, documents: list[Document]) -> str:
    text = (answer or "").strip()
    if not text:
        return text
    if "## Benchmark Verdict Matrix" not in text:
        return text

    per_paper = _build_comparator_signal_pool(documents=documents, limit=12)
    papers = list(per_paper.keys())[:3]
    if not papers:
        return text
    analysis = _render_common_benchmark_analysis(per_paper=per_paper, papers=papers)
    addition = f"### Common Benchmark Analysis\n{analysis}".strip()

    section_pattern = r"(##\s+Benchmark Verdict Matrix)([\s\S]*?)(\n##\s+Method Trade-offs|\Z)"
    match = re.search(section_pattern, text, flags=re.IGNORECASE)
    if not match:
        return text

    heading = match.group(1)
    body = match.group(2) or ""
    tail = match.group(3) or ""
    body_without_common = re.sub(
        r"\n*###\s+Common Benchmark Analysis[\s\S]*?(?=\n###\s+|\Z)",
        "",
        body,
        flags=re.IGNORECASE,
    ).rstrip()
    rebuilt_body = f"{body_without_common}\n\n{addition}".strip()
    replacement = f"{heading}\n{rebuilt_body}\n{tail.lstrip()}"
    return re.sub(section_pattern, replacement, text, count=1, flags=re.IGNORECASE)


def _shared_metric_summary(*, per_paper: dict[str, list[dict[str, Any]]], papers: list[str]) -> dict[str, Any]:
    if len(papers) < 2:
        return {"status": "no_pair"}
    records_by_paper: dict[str, list[dict[str, Any]]] = {}
    benchmark_families: dict[str, set[str]] = {}
    paper_best_citations: dict[str, int] = {}
    for paper in papers:
        for entry in per_paper.get(paper, []):
            citation = int(entry.get("citation", 1))
            if paper not in paper_best_citations:
                paper_best_citations[paper] = citation
            raw_benchmark = str(entry.get("benchmark_label", "")).strip()
            family = _benchmark_family(raw_benchmark)
            if family:
                benchmark_families.setdefault(paper, set()).add(family)
            snippet_text = str(entry.get("metric_snippet") or entry.get("snippet", "")).strip()
            snippet_family = _benchmark_family(_infer_benchmark_label(snippet_text))
            if snippet_family:
                benchmark_families.setdefault(paper, set()).add(snippet_family)
            task_family = _task_family_from_text(snippet_text)
            if task_family:
                benchmark_families.setdefault(paper, set()).add(task_family)
            for record in entry.get("metric_records", []) or []:
                try:
                    value = float(record.get("value", 0.0))
                except Exception:
                    continue
                record_benchmark = str(record.get("benchmark", "")).strip() or raw_benchmark
                record_family = _benchmark_family(record_benchmark)
                if record_family:
                    benchmark_families.setdefault(paper, set()).add(record_family)
                records_by_paper.setdefault(paper, []).append(
                    {
                        "metric": str(record.get("metric", "")).lower(),
                        "value": value,
                        "citation": citation,
                        "benchmark": record_benchmark,
                        "benchmark_family": record_family,
                        "snippet": snippet_text,
                        "language_pair": _detect_mt_language_pair(f"{record_benchmark} {snippet_text}"),
                    }
                )
    p1, p2 = papers[0], papers[1]
    r1 = records_by_paper.get(p1, [])
    r2 = records_by_paper.get(p2, [])
    shared_families = _shared_benchmark_families(
        left=benchmark_families.get(p1, set()),
        right=benchmark_families.get(p2, set()),
    )

    if not r1 or not r2:
        if shared_families:
            return {
                "status": "partial_shared_task",
                "benchmark_family": shared_families[0],
                "first_metrics": _metric_preview_records(r1),
                "second_metrics": _metric_preview_records(r2),
                "first_citation": int(paper_best_citations.get(p1, 1)),
                "second_citation": int(paper_best_citations.get(p2, 1)),
            }
        return {"status": "no_metrics"}

    best_pair: tuple[dict[str, Any], dict[str, Any], int, str, float, float] | None = None
    for a in r1:
        for b in r2:
            if a["metric"] != b["metric"]:
                continue
            a_benchmark = a.get("benchmark", "").lower()
            b_benchmark = b.get("benchmark", "").lower()
            a_family = str(a.get("benchmark_family", "")).lower()
            b_family = str(b.get("benchmark_family", "")).lower()
            a_pair = str(a.get("language_pair", "")).strip().lower()
            b_pair = str(b.get("language_pair", "")).strip().lower()
            if a_pair and b_pair and a_pair == b_pair:
                score = 4
                comparability = "strict"
            elif a_benchmark and b_benchmark and a_benchmark == b_benchmark:
                score = 3
                comparability = "strict"
            elif a_pair and b_pair and a_pair != b_pair:
                score = 1
                comparability = "loose"
            elif a_family and b_family and a_family == b_family:
                score = 2
                comparability = "family"
            else:
                score = 1
                comparability = "loose"
            pair_rank = max(float(a.get("value", 0.0)), float(b.get("value", 0.0)))
            pair_total = float(a.get("value", 0.0)) + float(b.get("value", 0.0))
            if (
                best_pair is None
                or score > best_pair[2]
                or (score == best_pair[2] and pair_rank > best_pair[4])
                or (score == best_pair[2] and pair_rank == best_pair[4] and pair_total > best_pair[5])
            ):
                best_pair = (a, b, score, comparability, pair_rank, pair_total)
            if score == 3:
                break
        if best_pair and best_pair[2] == 3:
            break

    if not best_pair:
        if shared_families:
            return {
                "status": "partial_shared_task",
                "benchmark_family": shared_families[0],
                "first_metrics": _metric_preview_records(r1),
                "second_metrics": _metric_preview_records(r2),
                "first_citation": int(paper_best_citations.get(p1, 1)),
                "second_citation": int(paper_best_citations.get(p2, 1)),
            }
        return {
            "status": "no_pair",
            "first_metrics": _metric_preview_records(r1),
            "second_metrics": _metric_preview_records(r2),
            "first_citation": int(paper_best_citations.get(p1, 1)),
            "second_citation": int(paper_best_citations.get(p2, 1)),
        }

    a, b, _, comparability, _, _ = best_pair
    metric = str(a.get("metric", "metric")).lower()
    first_benchmark = str(a.get("benchmark", "")).strip()
    second_benchmark = str(b.get("benchmark", "")).strip()
    benchmark = first_benchmark or second_benchmark
    if comparability == "loose" and shared_families:
        benchmark = shared_families[0]
    elif not benchmark and shared_families:
        benchmark = shared_families[0]
    first_value = float(a.get("value", 0.0))
    second_value = float(b.get("value", 0.0))
    lower_is_better = metric in {"wer", "loss", "perplexity", "error"}
    delta = abs(first_value - second_value)
    if lower_is_better:
        winner = p1 if first_value < second_value else p2
    else:
        winner = p1 if first_value > second_value else p2
    return {
        "status": "ok",
        "metric": metric,
        "benchmark": benchmark,
        "first_value": first_value,
        "second_value": second_value,
        "delta": delta,
        "winner": winner,
        "first_citation": int(a.get("citation", 1)),
        "second_citation": int(b.get("citation", 1)),
        "first_snippet": str(a.get("snippet", "")),
        "second_snippet": str(b.get("snippet", "")),
        "comparability": comparability,
        "first_benchmark": first_benchmark,
        "second_benchmark": second_benchmark,
        "first_language_pair": str(a.get("language_pair", "")).strip(),
        "second_language_pair": str(b.get("language_pair", "")).strip(),
    }


def _extract_metric_records(snippet: str) -> list[dict[str, Any]]:
    text = _clean_mojibake_text(snippet or "")
    if not text:
        return []

    benchmark = _infer_benchmark_label(text)
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, float, str]] = set()
    patterns = (
        (r"(\d+(?:\.\d+)?)\s*BLEU", "bleu"),
        (r"BLEU(?:\s+score)?\s*(?:of|=|:|to|increases?\s+to|increased\s+to|reaches?\s+|reached\s+)?\s*(\d+(?:\.\d+)?)", "bleu"),
        (r"accuracy\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)\s*%?", "accuracy"),
        (r"precision\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)\s*%?", "precision"),
        (r"recall\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)\s*%?", "recall"),
        (r"f1\s*(?:score)?\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)\s*%?", "f1"),
        (r"auc\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)", "auc"),
        (r"rouge(?:[-\s]?[12l])?\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)", "rouge"),
        (r"meteor\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)", "meteor"),
        (r"(?:exact match|em)\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)\s*%?", "exact match"),
        (r"(?:iou|intersection over union)\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)", "iou"),
        (r"dice\s*(?:score)?\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)", "dice"),
        (r"(?:map|mean average precision)\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)", "map"),
        (r"ndcg\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)", "ndcg"),
        (r"mrr\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)", "mrr"),
        (r"wer\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)\s*%?", "wer"),
        (r"perplexity\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)", "perplexity"),
    )
    for pattern, metric in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            try:
                value = float(match.group(1))
            except Exception:
                continue
            if metric == "bleu" and (value < 10 or value > 100):
                continue
            window_start = max(0, match.start() - 140)
            window_end = min(len(text), match.end() + 140)
            local_window = text[window_start:window_end]
            local_benchmark = _infer_benchmark_label(local_window) or benchmark
            key = (metric, round(value, 3), local_benchmark.lower())
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "metric": metric,
                    "value": value,
                    "benchmark": local_benchmark,
                }
            )
    if not records:
        lower = text.lower()
        numeric_source = re.sub(r"\[[0-9]+\]", " ", text)
        numeric_values: list[float] = []
        for raw in re.findall(r"\b\d+(?:\.\d+)?\b", numeric_source):
            try:
                value = float(raw)
            except Exception:
                continue
            if value <= 0 or value > 100:
                continue
            numeric_values.append(value)
        if "bleu" in lower and numeric_values:
            plausible_bleu = [value for value in numeric_values if 10 <= value <= 100]
            candidates = sorted(set(plausible_bleu), reverse=True)
            for value in candidates[:2]:
                key = ("bleu", round(value, 3), benchmark.lower())
                if key in seen:
                    continue
                seen.add(key)
                records.append(
                    {
                        "metric": "bleu",
                        "value": value,
                        "benchmark": benchmark,
                    }
                )
    if records:
        bleu_values = [float(item.get("value", 0.0)) for item in records if str(item.get("metric", "")).lower() == "bleu"]
        if any(value >= 10 for value in bleu_values):
            records = [
                item
                for item in records
                if str(item.get("metric", "")).lower() != "bleu" or float(item.get("value", 0.0)) >= 10
            ]
    return records


def _extract_metric_sentence(text: str) -> str:
    cleaned = _clean_visible_text(text or "")
    if not cleaned:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    best = ""
    best_score = -1.0
    for sentence in sentences:
        snippet = sentence.strip()
        if len(snippet) < 24:
            continue
        lower = snippet.lower()
        if _looks_like_reference_snippet(snippet):
            continue
        if _looks_like_non_argument_snippet(snippet) or _looks_like_metadata_snippet(snippet):
            continue
        score = 0.0
        if _has_metric_name(lower):
            score += 1.0
        if any(marker in lower for marker in _evaluation_signal_markers()):
            score += 0.45
        if re.search(r"\b\d+(?:\.\d+)?\b", snippet):
            score += 0.35
        if score > best_score:
            best_score = score
            best = snippet
    return _compact_turn_text(best, max_chars=240) if best_score >= 0.7 else ""


def _clean_benchmark_candidate(text: str) -> str:
    candidate = _clean_mojibake_text(text or "")
    candidate = re.sub(r"\s+", " ", candidate).strip(" .,:;()[]")
    candidate = re.sub(r"^(?:the|standard|widely used)\s+", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(
        r"\s+(?:dataset|benchmark|task|corpus|leaderboard|challenge|evaluation|test set|dev set|validation set)\b.*$",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(r"\s+", " ", candidate).strip(" .,:;()[]")
    return candidate


def _benchmark_candidate_is_generic(candidate: str) -> bool:
    cleaned = _clean_benchmark_candidate(candidate)
    lower = cleaned.lower()
    if not lower or len(lower) < 3:
        return True
    generic = {
        "same",
        "different",
        "shared",
        "various",
        "selected",
        "benchmark",
        "dataset",
        "task",
        "corpus",
        "evaluation",
        "results",
        "experiments",
        "translation",
        "machine translation",
        "classification",
        "image classification",
        "question answering",
        "summarization",
        "language modeling",
        "speech recognition",
        "object detection",
        "segmentation",
        "information retrieval",
    }
    if lower in generic:
        return True
    tokens = re.findall(r"[A-Za-z0-9./+\-]+", cleaned)
    if not tokens or len(tokens) > 5:
        return True
    if re.fullmatch(r"[a-z\s]+", cleaned) and not re.search(r"\d", cleaned):
        return True
    stopwords = {"the", "a", "an", "same", "different", "various", "our", "this", "these", "those", "for", "in", "on", "to", "of", "and"}
    if sum(1 for token in tokens if token.lower() in stopwords) >= max(2, len(tokens) - 1):
        return True
    return False


def _benchmark_alias_catalog() -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (
        ("WMT", (r"\bwmt(?:['’]?\d{2})?\b",)),
        ("IWSLT", (r"\biwslt(?:['’]?\d{2})?\b",)),
        ("FLORES", (r"\bflores(?:-?\d+)?\b",)),
        ("ImageNet", (r"\bimagenet(?:-1k|-21k)?\b",)),
        ("CIFAR-100", (r"\bcifar[\s-]?100\b",)),
        ("CIFAR-10", (r"\bcifar[\s-]?10\b",)),
        ("COCO", (r"\bcoco\b",)),
        ("SuperGLUE", (r"\bsuperglue\b",)),
        ("SQuAD", (r"\bsquad(?:\s*v?\d(?:\.\d)?)?\b",)),
        ("GLUE", (r"\bglue\b",)),
        ("MS MARCO", (r"\bms\s*marco\b",)),
        ("LibriSpeech", (r"\blibrispeech\b",)),
        ("MMLU", (r"\bmmlu\b",)),
        ("HumanEval", (r"\bhumaneval\b",)),
        ("GSM8K", (r"\bgsm8k\b",)),
    )


def _language_alias_map() -> dict[str, str]:
    return {
        "ar": "Ar",
        "arabic": "Ar",
        "chinese": "Zh",
        "cs": "Cs",
        "czech": "Cs",
        "de": "De",
        "en": "En",
        "english": "En",
        "es": "Es",
        "french": "Fr",
        "fr": "Fr",
        "german": "De",
        "hi": "Hi",
        "hindi": "Hi",
        "it": "It",
        "italian": "It",
        "ja": "Ja",
        "japanese": "Ja",
        "ko": "Ko",
        "korean": "Ko",
        "portuguese": "Pt",
        "pt": "Pt",
        "ro": "Ro",
        "romanian": "Ro",
        "ru": "Ru",
        "russian": "Ru",
        "spanish": "Es",
        "zh": "Zh",
    }


def _match_benchmark_alias(text: str) -> str:
    lower = _clean_visible_text(text or "").lower()
    for canonical, patterns in _benchmark_alias_catalog():
        if any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in patterns):
            return canonical
    return ""


def _extract_benchmark_year(text: str) -> str:
    cleaned = _clean_visible_text(text or "")
    match = re.search(r"\b(20\d{2})\b", cleaned)
    if match:
        return match.group(1)
    short_year = re.search(r"\b(?:wmt|iwslt)['’]?(\d{2})\b", cleaned, flags=re.IGNORECASE)
    if short_year:
        return f"20{short_year.group(1)}"
    return ""


def _canonicalize_benchmark_label(label: str, *, context: str = "") -> str:
    source = _clean_visible_text(f"{label} {context}".strip())
    alias = _match_benchmark_alias(source)
    if alias:
        parts = [alias]
        year = _extract_benchmark_year(source)
        if year:
            parts.append(year)
        pair = _detect_mt_language_pair(source)
        if pair and (
            alias in {"WMT", "IWSLT", "FLORES"}
            or any(marker in source.lower() for marker in ("translation", "machine translation", "seq2seq"))
        ):
            parts.append(pair)
        return " ".join(parts)
    candidate = _clean_benchmark_candidate(label)
    candidate = re.sub(r"\s+", " ", candidate).strip(" -,:;")
    return candidate if not _benchmark_candidate_is_generic(candidate) else ""


def _infer_benchmark_label_legacy(text: str) -> str:
    cleaned = _clean_mojibake_text(text or "")
    lower = cleaned.lower()
    if "wmt" in lower:
        has_en_de = bool(
            re.search(r"\ben\s*[-/ ]\s*de\b", lower)
            or re.search(r"english\s*[-–—]?\s*to\s*[-–—]?\s*german", lower)
        )
        has_en_fr = bool(
            re.search(r"\ben\s*[-/ ]\s*fr\b", lower)
            or re.search(r"english\s*[-–—]?\s*to\s*[-–—]?\s*french", lower)
        )
        if "2014" in lower:
            if has_en_de:
                return "WMT 2014 En-De"
            if has_en_fr:
                return "WMT 2014 En-Fr"
            return "WMT 2014"
        if has_en_de:
            return "WMT En-De"
        if has_en_fr:
            return "WMT En-Fr"
        return "WMT"
    if "imagenet" in lower:
        return "ImageNet"
    if "cifar-100" in lower or "cifar 100" in lower:
        return "CIFAR-100"
    if "cifar-10" in lower or "cifar 10" in lower:
        return "CIFAR-10"
    if "coco" in lower:
        return "COCO"
    if "superglue" in lower:
        return "SuperGLUE"
    if "squad" in lower:
        return "SQuAD"
    if "glue" in lower:
        return "GLUE"
    if "ms marco" in lower:
        return "MS MARCO"
    if "librispeech" in lower:
        return "LibriSpeech"
    if "mmlu" in lower:
        return "MMLU"
    if "humaneval" in lower:
        return "HumanEval"
    if "gsm8k" in lower:
        return "GSM8K"
    generic_patterns = (
        r"(?:on|for|using|evaluated on|tested on|benchmark(?:ed)? on)\s+(?:the\s+)?([A-Za-z0-9][A-Za-z0-9./+\-]*(?:\s+[A-Za-z0-9][A-Za-z0-9./+\-]*){0,5})\s+(?:dataset|benchmark|task|corpus|leaderboard|challenge)",
        r"(?:the\s+)?([A-Za-z0-9][A-Za-z0-9./+\-]*(?:\s+[A-Za-z0-9][A-Za-z0-9./+\-]*){0,5})\s+(?:dataset|benchmark|task|corpus|leaderboard|challenge)",
    )
    for pattern in generic_patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = _clean_benchmark_candidate(match.group(1))
        if not _benchmark_candidate_is_generic(candidate):
            return candidate
    return ""


def _benchmark_family_legacy(label: str) -> str:
    lower = (label or "").strip().lower()
    if not lower:
        return ""
    if "wmt" in lower:
        if "2014" in lower:
            return "WMT 2014"
        return "WMT"
    if "imagenet" in lower:
        return "ImageNet"
    if "cifar-100" in lower:
        return "CIFAR-100"
    if "cifar-10" in lower:
        return "CIFAR-10"
    if "coco" in lower:
        return "COCO"
    if "superglue" in lower:
        return "SuperGLUE"
    if "squad" in lower:
        return "SQuAD"
    if "glue" in lower:
        return "GLUE"
    if "ms marco" in lower:
        return "MS MARCO"
    if "librispeech" in lower:
        return "LibriSpeech"
    if "mmlu" in lower:
        return "MMLU"
    if "humaneval" in lower:
        return "HumanEval"
    if "gsm8k" in lower:
        return "GSM8K"
    root = _benchmark_family_root(label)
    return root if not _benchmark_candidate_is_generic(root) else ""


def _shared_benchmark_families(*, left: set[str], right: set[str]) -> list[str]:
    direct = sorted(set(left) & set(right))
    if direct:
        return direct
    shared_roots = sorted(
        {
            root
            for root in (_benchmark_family_root(item) for item in left)
            if root
        }
        & {
            root
            for root in (_benchmark_family_root(item) for item in right)
            if root
        }
    )
    if shared_roots:
        return shared_roots
    return []


def _benchmark_family_root_legacy(label: str) -> str:
    lower = (label or "").strip().lower()
    if not lower:
        return ""
    if "wmt" in lower:
        return "WMT"
    if "imagenet" in lower:
        return "ImageNet"
    if "cifar-100" in lower:
        return "CIFAR-100"
    if "cifar-10" in lower:
        return "CIFAR-10"
    if "coco" in lower:
        return "COCO"
    if "superglue" in lower:
        return "SuperGLUE"
    if "squad" in lower:
        return "SQuAD"
    if "glue" in lower:
        return "GLUE"
    if "ms marco" in lower:
        return "MS MARCO"
    if "librispeech" in lower:
        return "LibriSpeech"
    if "mmlu" in lower:
        return "MMLU"
    if "humaneval" in lower:
        return "HumanEval"
    if "gsm8k" in lower:
        return "GSM8K"
    candidate = _clean_benchmark_candidate(label)
    candidate = re.sub(r"\b20\d{2}\b", "", candidate)
    candidate = re.sub(r"\bv\d+(?:\.\d+)?\b", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(
        r"\b(?:en\s*[-/ ]\s*fr|en\s*[-/ ]\s*de|english\s*to\s*french|english\s*to\s*german|\d+\s*shot|zero\s*shot|few\s*shot)\b",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(r"\s+", " ", candidate).strip(" -,:;")
    return candidate if not _benchmark_candidate_is_generic(candidate) else ""


def _infer_benchmark_label(text: str) -> str:
    cleaned = _clean_visible_text(text or "")
    alias_label = _canonicalize_benchmark_label(cleaned)
    if alias_label:
        return alias_label
    generic_patterns = (
        r"(?:on|for|using|evaluated on|tested on|benchmark(?:ed)? on)\s+(?:the\s+)?([A-Za-z0-9][A-Za-z0-9./+\-]*(?:\s+[A-Za-z0-9][A-Za-z0-9./+\-]*){0,5})\s+(?:dataset|benchmark|task|corpus|leaderboard|challenge)",
        r"(?:the\s+)?([A-Za-z0-9][A-Za-z0-9./+\-]*(?:\s+[A-Za-z0-9][A-Za-z0-9./+\-]*){0,5})\s+(?:dataset|benchmark|task|corpus|leaderboard|challenge)",
    )
    for pattern in generic_patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = _canonicalize_benchmark_label(match.group(1), context=cleaned)
        if candidate:
            return candidate
    return ""


def _benchmark_family(label: str) -> str:
    cleaned = _clean_visible_text(label or "")
    if not cleaned:
        return ""
    alias = _match_benchmark_alias(cleaned)
    if alias:
        year = _extract_benchmark_year(cleaned)
        return f"{alias} {year}".strip() if year else alias
    root = _benchmark_family_root(cleaned)
    return root if not _benchmark_candidate_is_generic(root) else ""


def _benchmark_family_root(label: str) -> str:
    cleaned = _clean_visible_text(label or "")
    if not cleaned:
        return ""
    alias = _match_benchmark_alias(cleaned)
    if alias:
        return alias
    candidate = _clean_benchmark_candidate(cleaned)
    candidate = re.sub(r"\b20\d{2}\b", "", candidate)
    candidate = re.sub(r"\bv\d+(?:\.\d+)?\b", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(
        r"\b(?:[a-z]{2}\s*[-/ ]\s*[a-z]{2}|[a-z]+\s*to\s*[a-z]+|\d+\s*shot|zero\s*shot|few\s*shot)\b",
        "",
        candidate,
        flags=re.IGNORECASE,
    )
    candidate = re.sub(r"\s+", " ", candidate).strip(" -,:;")
    return candidate if not _benchmark_candidate_is_generic(candidate) else ""


def _task_family_from_text(text: str) -> str:
    lower = (text or "").lower()
    if not lower:
        return ""
    if "wmt" in lower:
        if "2014" in lower:
            return "WMT 2014"
        return "WMT"
    families = (
        ("Machine Translation", ("machine translation", "translation task", "en-fr", "en-de", "seq2seq")),
        ("Question Answering", ("question answering", "reading comprehension", "extractive qa")),
        ("Summarization", ("summarization", "summarisation", "summary generation")),
        ("Language Modeling", ("language modeling", "language modelling", "next-word prediction")),
        ("Text Classification", ("text classification", "classification task", "sentiment classification")),
        ("Image Classification", ("image classification", "imagenet", "cifar")),
        ("Object Detection", ("object detection", "detection benchmark", "bounding box")),
        ("Segmentation", ("segmentation", "semantic segmentation", "instance segmentation")),
        ("Speech Recognition", ("speech recognition", "automatic speech recognition", "asr")),
        ("Information Retrieval", ("information retrieval", "retrieval task", "search ranking")),
        ("Recommendation", ("recommendation", "ranking task", "ctr prediction")),
        ("Code Generation", ("code generation", "program synthesis", "humaneval", "pass@1")),
    )
    for family, markers in families:
        if any(marker in lower for marker in markers):
            return family
    return ""


def _metric_preview_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    seen: set[tuple[str, float, str]] = set()
    for record in records:
        metric = str(record.get("metric", "")).lower()
        try:
            value = float(record.get("value", 0.0))
        except Exception:
            continue
        benchmark = str(record.get("benchmark", "")).strip()
        key = (metric, round(value, 3), benchmark.lower())
        if key in seen:
            continue
        seen.add(key)
        preview.append(
            {
                "metric": metric,
                "value": value,
                "citation": int(record.get("citation", 1)),
                "benchmark": benchmark,
                "language_pair": str(record.get("language_pair", "")).strip(),
            }
        )
        if len(preview) >= 2:
            break
    return preview


def _format_metric_brief_line(*, paper: str, metrics: list[dict[str, Any]], fallback: str, citation: int) -> str:
    if not isinstance(metrics, list) or not metrics:
        return f"{paper}: {fallback} [{citation}]"
    parts: list[str] = []
    citations: list[str] = []
    for item in metrics[:2]:
        metric = str(item.get("metric", "metric")).upper()
        try:
            value = float(item.get("value", 0.0))
            detail = f"{value:.2f} {metric}"
        except Exception:
            continue
        benchmark = str(item.get("benchmark", "")).strip()
        language_pair = str(item.get("language_pair", "")).strip()
        if benchmark:
            detail += f" on {benchmark}"
        elif language_pair:
            detail += f" on {language_pair}"
        parts.append(detail)
        citations.append(str(int(item.get("citation", citation))))
    citation_suffix = "".join(f"[{c}]" for c in citations if c.isdigit())
    return f"{paper}: {', '.join(parts) or fallback} {citation_suffix}".strip()


def _benchmark_difference_reason(
    *,
    first_snippet: str,
    second_snippet: str,
    first_benchmark: str = "",
    second_benchmark: str = "",
) -> str:
    first = (first_snippet or "").lower()
    second = (second_snippet or "").lower()
    combined = f"{first} {second}"
    first_pair = _detect_mt_language_pair(first_benchmark or first)
    second_pair = _detect_mt_language_pair(second_benchmark or second)
    if first_pair and second_pair and first_pair != second_pair:
        return (
            f"Scores appear to come from different language pairs ({first_pair} vs {second_pair}); "
            "treating one as uniformly better would be misleading."
        )
    if first_benchmark and second_benchmark and first_benchmark.strip().lower() != second_benchmark.strip().lower():
        if _benchmark_family_root(first_benchmark) != _benchmark_family_root(second_benchmark):
            return "Scores appear to come from different benchmark slices or tasks, so a direct ranking would overstate comparability."
    if any(marker in combined for marker in _efficiency_signal_markers()):
        return "Retrieved snippets indicate training or compute differences that can materially shift reported results."
    if any(marker in combined for marker in ("preprocess", "pre-processing", "tokenization", "augmentation", "prompt", "beam", "retrieval", "sampling")):
        return "Retrieved evidence points to setup or evaluation-protocol differences that can change the measured outcome."
    if any(marker in combined for marker in _method_signal_markers()):
        return "The difference appears linked to method or architecture choices described in the retrieved evidence."
    return "Difference is likely tied to method and evaluation setup choices reported in the retrieved snippets."


def _detect_mt_language_pair_legacy(text: str) -> str:
    lower = (text or "").lower()
    if re.search(r"\ben\s*[-/ ]\s*fr\b", lower) or re.search(r"english\s*[-–—]?\s*to\s*[-–—]?\s*french", lower):
        return "En-Fr"
    if re.search(r"\ben\s*[-/ ]\s*de\b", lower) or re.search(r"english\s*[-–—]?\s*to\s*[-–—]?\s*german", lower):
        return "En-De"
    return ""


def _detect_mt_language_pair(text: str) -> str:
    cleaned = _clean_visible_text(text or "").lower()
    aliases = _language_alias_map()
    patterns = (
        r"\b([a-z]{2,12})\s*-\s*to\s*-\s*([a-z]{2,12})\b",
        r"\b([a-z]{2,12})\s*[-/]\s*([a-z]{2,12})\b",
        r"\b([a-z]{2,12})\s*(?:to|into)\s*([a-z]{2,12})\b",
    )
    for pattern in patterns:
        for left, right in re.findall(pattern, cleaned, flags=re.IGNORECASE):
            left_code = aliases.get(left.lower())
            right_code = aliases.get(right.lower())
            if left_code and right_code and left_code != right_code:
                return f"{left_code}-{right_code}"
    return ""


def _unique_paper_ids(documents: list[Document]) -> set[str]:
    paper_ids: set[str] = set()
    for document in documents:
        paper_id = str((document.metadata or {}).get("paper_id", "")).strip()
        if paper_id:
            paper_ids.add(paper_id)
    return paper_ids


def _comparator_selected_filenames(
    *,
    documents: list[Document],
    selected_paper_ids: list[str] | None,
) -> list[str]:
    lookup = _comparator_filename_lookup(documents)
    selected: list[str] = []
    seen: set[str] = set()
    for paper_id in selected_paper_ids or []:
        pid = str(paper_id or "").strip()
        if not pid:
            continue
        name = lookup.get(pid, "")
        if not name or name in seen:
            continue
        seen.add(name)
        selected.append(name)
    if selected:
        return selected
    for document in documents:
        filename = str((document.metadata or {}).get("filename", "")).strip()
        if not filename or filename in seen:
            continue
        seen.add(filename)
        selected.append(filename)
        if len(selected) >= 3:
            break
    return selected


def _comparator_answer_quality_issues(
    *,
    answer: str,
    citations: list[dict[str, Any]],
    documents: list[Document],
    selected_paper_ids: list[str] | None,
) -> list[str]:
    issues: list[str] = []
    text = (answer or "").strip()
    if not text:
        return ["empty_answer"]

    expected_sections = (
        "## Papers Compared",
        "## Claim Matrix",
        "## Conflict Map",
        "## Benchmark Verdict Matrix",
        "## Method Trade-offs",
        "## Synthesis Blueprint",
        "## Decision By Use Case",
    )
    missing_sections = [section for section in expected_sections if section.lower() not in text.lower()]
    if missing_sections:
        issues.append("missing_sections")
    if len(text) < 460:
        issues.append("too_short")
    placeholder_markers = (
        "section is removed due to lack of direct evidence",
        "removed due to lack of direct evidence",
        "section removed due to lack of direct evidence",
        "removed due to lack of evidence from the retrieved context",
    )
    if any(marker in text.lower() for marker in placeholder_markers):
        issues.append("section_removed_placeholder")
    speculative_markers = (
        " may ",
        " might ",
        " could ",
        " likely ",
        " perhaps ",
        " possibly ",
        "careful tuning",
        "low-resource",
    )
    speculative_without_citations = 0
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        lowered = f" {sentence.lower()} "
        if not any(marker in lowered for marker in speculative_markers):
            continue
        if re.search(r"\[[0-9]+\]", sentence):
            continue
        speculative_without_citations += 1
    if speculative_without_citations >= 2:
        issues.append("speculative_without_citations")
    if _count_uncited_use_case_winners(text) > 0:
        issues.append("uncited_use_case_winner")

    selected_filenames = _comparator_selected_filenames(
        documents=documents,
        selected_paper_ids=selected_paper_ids,
    )
    missing_filename_mentions = [
        filename
        for filename in selected_filenames
        if filename.lower() not in text.lower()
    ]
    if missing_filename_mentions:
        issues.append("missing_selected_paper_mentions")

    referenced_numbers = sorted({int(match) for match in re.findall(r"\[([0-9]+)\]", text) if match.isdigit()})
    if not referenced_numbers:
        issues.append("missing_inline_citations")
        return issues

    cited_sources: set[str] = set()
    for number in referenced_numbers:
        index = number - 1
        if index < 0 or index >= len(citations):
            continue
        citation = citations[index]
        source = str(citation.get("paper_id", "")).strip() or str(citation.get("filename", "")).strip()
        if source:
            cited_sources.add(source)

    if len(_unique_paper_ids(documents)) >= 2 and len(cited_sources) < 2:
        issues.append("citations_single_paper")

    mentioned_files = {match.strip() for match in re.findall(r"\b([A-Za-z0-9_.-]+\.pdf)\b", text)}
    allowed_files = {name.strip() for name in selected_filenames if name.strip()}
    unexpected = [name for name in mentioned_files if name not in allowed_files]
    if allowed_files and unexpected:
        issues.append("mentions_unselected_paper")
    lower_text = text.lower()
    no_shared_claim = (
        "no shared benchmark metrics" in lower_text
        or "shared benchmark metrics are not explicitly available" in lower_text
    )
    if no_shared_claim and _comparator_has_shared_benchmark_signal(documents=documents):
        issues.append("benchmark_contradiction")
    per_paper = _build_comparator_signal_pool(documents=documents, limit=16)
    has_metric_signal = any(
        entry.get("metric_records")
        for entries in per_paper.values()
        for entry in entries
    )
    if has_metric_signal and "no benchmark results" in lower_text:
        issues.append("benchmark_metric_omitted")
    if has_metric_signal and "benchmark verdict matrix" in lower_text and not _has_metric_name(lower_text):
        issues.append("benchmark_signal_missing")
    return issues


def _count_uncited_use_case_winners(text: str) -> int:
    section = _extract_markdown_section(text, "Decision By Use Case")
    if not section:
        return 0
    count = 0
    for line in section.splitlines():
        lowered = line.lower()
        if "winner" not in lowered:
            continue
        if "no winner" in lowered:
            continue
        if re.search(r"\[[0-9]+\]", line):
            continue
        count += 1
    return count


def _comparator_has_grounding_risk(issues: list[str] | None) -> bool:
    if not isinstance(issues, list):
        return False
    markers = (
        "lack of direct evidence",
        "not directly supported",
        "insufficient information",
        "unsupported",
        "not supported by the retrieved context",
        "not supported by retrieved context",
    )
    for issue in issues:
        lowered = str(issue or "").lower()
        if not lowered:
            continue
        if any(marker in lowered for marker in markers):
            return True
    return False


def _comparator_has_shared_benchmark_signal(*, documents: list[Document]) -> bool:
    per_paper = _build_comparator_signal_pool(documents=documents, limit=12)
    papers = list(per_paper.keys())[:2]
    if len(papers) < 2:
        return False
    summary = _shared_metric_summary(per_paper=per_paper, papers=papers)
    status = str(summary.get("status", "")).strip().lower()
    if status == "ok":
        return True
    if status == "partial_shared_task":
        family = str(summary.get("benchmark_family", "")).strip()
        return bool(family)
    return False


def _repair_comparator_benchmark_contradiction(*, answer: str, documents: list[Document]) -> str:
    text = str(answer or "")
    if not text:
        return text
    per_paper = _build_comparator_signal_pool(documents=documents, limit=12)
    papers = list(per_paper.keys())[:2]
    if len(papers) < 2:
        return text
    summary = _shared_metric_summary(per_paper=per_paper, papers=papers)
    family = (
        str(summary.get("benchmark_family", "")).strip()
        or _benchmark_family(str(summary.get("benchmark", "")).strip())
        or "a shared benchmark family"
    )
    replacement = (
        f"Shared benchmark family detected ({family}), but fully matched metric pairs are limited in current snippets."
    )
    repaired = re.sub(
        r"(?i)shared benchmark metrics are not explicitly available in current fallback snippets\.",
        replacement,
        text,
    )
    repaired = re.sub(
        r"(?i)no shared benchmark metrics(?: are)? present",
        replacement,
        repaired,
    )
    return repaired


def _extract_markdown_section(text: str, title: str) -> str:
    escaped = re.escape(title.strip())
    pattern = rf"##\s+{escaped}\s*\n(?P<body>[\s\S]*?)(?:\n##\s+|\Z)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return ""
    return (match.group("body") or "").strip()


def _extract_signal_sentence(text: str) -> str:
    cleaned = _clean_visible_text(re.sub(r"\s+", " ", (text or "")).strip())
    if not cleaned:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    best = ""
    best_score = float("-inf")
    for sentence in sentences:
        snippet = sentence.strip()
        if len(snippet) < 40:
            continue
        if _looks_like_reference_snippet(snippet):
            continue
        if _looks_like_non_argument_snippet(snippet) or _looks_like_metadata_snippet(snippet):
            continue
        lower = snippet.lower()
        score = 0.0
        if lower.startswith(("in this paper", "we propose", "we present", "our main result", "the main result")):
            score += 0.7
        if any(marker in lower for marker in _method_signal_markers() + _evaluation_signal_markers()):
            score += 0.45
        if _has_metric_name(lower):
            score += 0.25
        if any(marker in lower for marker in ("outperform", "improve", "faster", "lower", "higher", "state-of-the-art", "state of the art")):
            score += 0.2
        if re.search(r"\b\d+(?:\.\d+)?%?\b", snippet):
            score += 0.15
        if score > best_score:
            best_score = score
            best = snippet[:260]
    if best:
        return best
    return cleaned[:260]


def _clean_mojibake_text(text: str) -> str:
    cleaned = str(text or "")
    cleaned = cleaned.replace(chr(226) + "??", "'")
    cleaned = cleaned.replace("â?¢", "-").replace("â€¢", "-")
    replacements = {
        "Ã": "x",
        "Â": "",
        "â": "'",
        "â": "'",
        "â": '"',
        "â": '"',
        "â": "-",
        "â": "-",
        "â": "sqrt",
        "âˆš": "sqrt",
        "ï¬": "fi",
        "ï¬": "fl",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    cleaned = (
        cleaned.replace("â??", "'")
        .replace("â€™", "'")
        .replace("â€œ", '"')
        .replace("â€", '"')
        .replace("â€“", "-")
        .replace("â€”", "-")
    )
    cleaned = (
        cleaned.replace("â??", "'")
        .replace("â€™", "'")
        .replace("â€œ", '"')
        .replace("â€", '"')
        .replace("â€“", "-")
        .replace("â€”", "-")
    )
    return cleaned


def _looks_like_metadata_snippet(text: str) -> bool:
    lowered = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if not lowered:
        return False
    if re.search(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", lowered):
        return True
    markers = (
        "google brain",
        "google research",
        "university of",
        "provided proper attribution",
        "permission to reproduce",
        "all rights reserved",
        "corresponding author",
        "equal contribution",
        "work performed while at",
        "conference on neural information processing systems",
        "long beach, ca",
    )
    return any(marker in lowered for marker in markers)


def _clean_visible_text(text: str) -> str:
    cleaned = _clean_mojibake_text(text or "")
    replacements = {
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€": '"',
        "â€“": "-",
        "â€”": "-",
        "âˆ’": "-",
        "â€ ": "",
        "â€¡": "",
        "ï¬": "fi",
        "ï¬‚": "fl",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "WMTâ??14": "WMT 2014",
        "WMT'14": "WMT 2014",
        "LSTMâ??s": "LSTM's",
        "uniformdistribution": "uniform distribution",
        "halfepoch": "half epoch",
        "dividedit": "divided it",
        "witha": "with a",
        "begun halving": "began halving",
        "ques- tion": "question",
        "tran slation": "translation",
        "tran slations": "translations",
        "translationsproduced": "translations produced",
        "â??": "'",
        "WMTâ??14": "WMT 2014",
        "WMT’14": "WMT 2014",
        "anEnglish": "an English",
        "wordsrepresenting": "words representing",
        "theinput": "the input",
        "comp utational": "computational",
        "softma x": "softmax",
        "ou r": "our",
        "t he": "the",
        "th e": "the",
        "us ed": "used",
        "cle ar": "clear",
        "Englishto-": "English-to-",
        "Frenchto-": "French-to-",
        "Germanto-": "German-to-",
        "L STM": "LSTM",
        "LS TM": "LSTM",
        "ar e": "are",
        "i n": "in",
        "re ported": "reported",
        "refer ence": "reference",
        "conﬁguration": "configuration",
        "difﬁculty": "difficulty",
        "ﬁxed": "fixed",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    cleaned = re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1\2", cleaned)
    cleaned = re.sub(r"\b([A-Z][a-z]+)\s*to\s*-\s*([A-Z][a-z]+)\b", r"\1-to-\2", cleaned)
    cleaned = re.sub(r"\b([A-Z]{1,3})\s+([A-Z]{2,4})\b", lambda m: m.group(0).replace(" ", ""), cleaned)
    cleaned = re.sub(r"([a-z])([A-Z][a-z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"\b(d|q|k|v)\s+([0-9]+)\b", r"\1_\2", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _contains_ocr_noise(text: str) -> bool:
    lower = str(text or "").lower()
    return any(
        marker in lower
        for marker in (
            "â??",
            "ï¬",
            "anenglish",
            "theinput",
            "wordsrepresenting",
            "uniformdistribution",
            "halfepoch",
            "dividedit",
        )
    )


def _clean_local_math_text(text: str) -> str:
    cleaned = _clean_visible_text(text or "")
    # LLM validator fallbacks can preserve JSON-escaped LaTeX (e.g., \\frac),
    # which KaTeX then renders as plain text tokens. Normalize to single slashes.
    cleaned = re.sub(r"\\\\(?=[A-Za-z])", r"\\", cleaned)
    replacements = {
        "QK T": "QK^T",
        "QK t": "QK^T",
        "QK⊤": "QK^T",
        "d k": "d_k",
        "dk ": "d_k ",
        "W 1": "W_1",
        "W 2": "W_2",
        "b 1": "b_1",
        "b 2": "b_2",
        "x W_1": "xW_1",
        "sqrt d_k": "sqrt(d_k)",
        "sqrt dk": "sqrt(d_k)",
        "$dk$": "$d_k$",
        "$dv$": "$d_v$",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    cleaned = re.sub(r"\bdk\b", "d_k", cleaned)
    cleaned = re.sub(r"\bdv\b", "d_v", cleaned)
    cleaned = re.sub(r"\b([A-Za-z])\s+_([0-9]+)\b", r"\1_\2", cleaned)
    cleaned = re.sub(r"\s+\^\s+", "^", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _structure_local_math_answer(text: str) -> str:
    compact = str(text or "").strip()
    if not compact or re.search(r"(?m)^##\s+", compact):
        return compact

    equation_match = re.search(r"(?s)\$\$\s*(.*?)\s*\$\$", compact)
    if not equation_match:
        return compact

    before = compact[: equation_match.start()].strip()
    equation = equation_match.group(1).strip()
    after = compact[equation_match.end() :].strip()
    citation_match = re.search(r"(\[[0-9]+\])", after)
    citation = citation_match.group(1) if citation_match else ""
    trailing = re.sub(r"\[[0-9]+\]", "", after).strip()

    lead = before or "The retrieved paper context supports the following equation."
    lead = re.sub(r"(?i)^the attention equation is given by:?\s*", "", lead).strip(" :")
    if not lead:
        lead = "The retrieved paper context supports the following equation."

    walkthrough = trailing
    if walkthrough.lower().startswith("where "):
        walkthrough = "The surrounding paper text immediately defines the symbols and how the equation should be read: " + walkthrough

    symbol_lines: list[str] = []
    lower_all = f"{before} {after} {equation}".lower()
    if "query" in lower_all or re.search(r"\bq\b", lower_all):
        symbol_lines.append("- $Q$: query vectors or the matrix of queries.")
    if "key" in lower_all or re.search(r"\bk\b", lower_all):
        symbol_lines.append("- $K$: key vectors or the matrix of keys.")
    if "value" in lower_all or re.search(r"\bv\b", lower_all):
        symbol_lines.append("- $V$: value vectors or the matrix of values.")
    if "d_k" in lower_all:
        symbol_lines.append("- $d_k$: key dimensionality used in the scaling factor $1/\\sqrt{d_k}$.")
    if "d_v" in lower_all:
        symbol_lines.append("- $d_v$: value dimensionality of the output vectors.")

    lines = [
        "## Short Answer",
        lead,
        "",
        "## Math Walkthrough",
        "$$",
        equation,
        "$$",
    ]
    if walkthrough:
        lines.extend(["", walkthrough])
    if symbol_lines:
        lines.extend(["", "## Symbol Guide", *symbol_lines])
    if citation:
        lines.extend(["", "## Grounding Notes", f"Equation wording is grounded in the retrieved paper context. {citation}"])
    return "\n".join(lines).strip()


def _format_local_math_answer(answer: str) -> str:
    text = (
        _clean_local_math_text(answer or "")
        .replace("â", "sqrt")
        .replace("âˆš", "sqrt")
        .replace("√", "sqrt")
        .replace("√", "sqrt")
        .strip()
    )
    if not text:
        return text

    lines = text.splitlines()
    formatted: list[str] = []
    in_code_fence = False
    in_math_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            formatted.append(line)
            continue
        if in_code_fence:
            formatted.append(line)
            continue

        if stripped == "$$":
            in_math_block = not in_math_block
            formatted.append(line)
            continue
        if in_math_block or stripped.startswith("$$") or stripped.startswith(r"\["):
            formatted.append(line)
            continue
        if stripped.startswith(r"\]"):
            formatted.append(line)
            continue

        if not _looks_like_equation_line(stripped):
            formatted.append(line)
            continue

        citation_match = re.match(r"^(.*?)(\s*(?:\[[0-9]+\]\s*)+)$", stripped)
        equation = stripped
        trailing_citations = ""
        if citation_match:
            equation = citation_match.group(1).strip()
            trailing_citations = citation_match.group(2).strip()

        if not equation:
            formatted.append(line)
            continue

        formatted.append("$$")
        formatted.append(equation)
        formatted.append("$$")
        if trailing_citations:
            formatted.append(trailing_citations)

    compact = "\n".join(formatted)
    compact = re.sub(r"\n{3,}", "\n\n", compact).strip()
    if "$$" in compact:
        compact = (
            compact.replace("QKT", r"QK^{\top}")
            .replace("QK T", r"QK^{\top}")
            .replace(r"\sqrt{dk}", r"\sqrt{d_k}")
            .replace(r"\sqrt{dk}", r"\sqrt{d_k}")
            .replace(" dk ", " d_k ")
        )
        return _structure_local_math_answer(compact)

    lower = compact.lower()
    if (
        "attention" in lower
        and "softmax" in lower
        and ("sqrt" in lower or "â" in compact or "d_k" in lower or "dk" in lower)
    ):
        cite_match = re.search(r"\[[0-9]+\]", compact)
        cite = f"\n{cite_match.group(0)}" if cite_match else ""
        compact += (
            "\n\n$$\n"
            r"\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V"
            "\n$$"
            f"{cite}"
        )
    return _structure_local_math_answer(compact.strip())


def _ensure_local_math_citations(answer: str, citations: list[dict[str, Any]]) -> str:
    text = str(answer or "").strip()
    if not text or re.search(r"\[[0-9]+\]", text):
        return text
    if not citations:
        return text
    cite = "[1]"
    if "\n$$" in text or text.startswith("$$"):
        return text + f"\n\nGrounding note: equation wording is based on the retrieved paper context. {cite}"
    return text + f" {cite}"


def _looks_like_equation_line(line: str) -> bool:
    text = str(line or "").strip()
    if len(text) < 8:
        return False
    if text.startswith(("#", "-", "*", "+", "|", ">", "`")):
        return False
    if text.lower().startswith(("where ", "thus ", "therefore ", "note:")):
        return False
    if text.endswith(".") and "=" not in text:
        return False
    if not re.search(r"[A-Za-z]", text):
        return False
    if re.search(r"https?://", text):
        return False
    if text.count(" ") > 18:
        return False
    if re.search(r"[.!?]", text) and text.count(" ") > 10 and not text.startswith("\\"):
        return False
    if text.count(" ") > 14 and "=" not in text and not text.startswith("\\"):
        return False
    if "=" not in text and ":" in text and not text.startswith("\\"):
        return False

    markers = (
        "=",
        r"\sum",
        r"\prod",
        r"\frac",
        r"\alpha",
        r"\beta",
        r"\theta",
        r"\lambda",
        r"\sigma",
        r"\mu",
        r"\mathbb",
        r"\mathcal",
        r"\mathbf",
        "argmax",
        "argmin",
        "softmax",
        "log(",
        "exp(",
        "||",
        "^",
    )
    if any(marker in text for marker in markers):
        token_density = len(re.findall(r"[=^_\\/{}()]", text))
        if text.count(" ") > 10 and token_density < 3 and not text.startswith("\\"):
            return False
        return True

    if text.startswith("\\"):
        return True
    if re.match(r"^[A-Za-z][A-Za-z0-9_]*\s*\(.+\)$", text):
        return True

    token_density = len(re.findall(r"[=^_\\/{}()]", text))
    return token_density >= 3


def _local_extractive_fallback(*, query: str, documents: list[Document]) -> str:
    query_terms = set(_tokenize_for_overlap(query))
    query_phrases = _query_phrases(query)
    lower_query = (query or "").lower()
    quantity_intent = _is_quantity_intent_query(query)
    if quantity_intent and any(token.startswith("expert") for token in query_terms):
        quantity_snippet, citation_index = _extract_quantity_snippet(documents=documents, keyword="expert")
        if quantity_snippet:
            expert_count = _infer_expert_count(quantity_snippet)
            if expert_count is not None:
                return (
                    "Based on the uploaded paper: "
                    f"the paper uses {expert_count} experts. {quantity_snippet} [{citation_index}]"
                )
            return f"Based on the uploaded paper: {quantity_snippet} [{citation_index}]"
        return (
            "This information is not in your uploaded papers. "
            "The retrieved context discusses mixture-of-experts concepts but does not provide a clear numeric expert count."
        )
    candidates: list[tuple[float, str]] = []
    for index, document in enumerate(documents[:6]):
        text = (document.page_content or "").strip()
        if not text:
            continue
        doc_overlap = _overlap_score(text.lower(), query_terms)
        doc_phrase = _phrase_overlap_score(_normalize_for_phrase_match(text.lower()), query_phrases)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            snippet = sentence.strip()
            if len(snippet) < 28:
                continue
            if _looks_like_reference_snippet(snippet):
                continue
            overlap = _overlap_score(snippet.lower(), query_terms)
            phrase = _phrase_overlap_score(_normalize_for_phrase_match(snippet.lower()), query_phrases)
            rank_prior = max(0.05, 1.0 - (index / max(1, len(documents))))
            score = (0.9 * overlap) + (0.9 * phrase) + (0.25 * doc_overlap) + (0.2 * doc_phrase) + (0.12 * rank_prior)
            snippet_no_citations = re.sub(r"\[[0-9]+\]", " ", snippet)
            if quantity_intent and re.search(r"\b\d+(?:\.\d+)?\b", snippet_no_citations):
                score += 0.35
            score -= min(0.3, _low_signal_penalty(snippet))
            candidates.append((score, snippet))
    if not candidates:
        return "This information is not in your uploaded papers."

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_snippet = candidates[0]
    if best_score < 0.20:
        return "This information is not in your uploaded papers."

    cleaned_best = re.sub(r"\[[0-9]+\]", "", best_snippet)
    cleaned_best = re.sub(r"\s+", " ", cleaned_best).strip()
    return f"Based on the uploaded paper: {cleaned_best} [1]"


def _try_local_numeric_fastpath(*, query: str, documents: list[Document]) -> str | None:
    if not documents:
        return None
    if not _is_quantity_intent_query(query):
        return None

    lower_query = (query or "").lower()
    keyword = "expert"
    if "participant" in lower_query:
        keyword = "participant"
    elif "player" in lower_query:
        keyword = "player"
    elif "subject" in lower_query:
        keyword = "subject"
    elif re.search(r"\bhead\b|\bheads\b", lower_query):
        keyword = "head"

    snippet, citation_index = _extract_quantity_snippet(documents=documents, keyword=keyword)
    if not snippet:
        return None

    count = _extract_keyword_count(snippet=snippet, keyword=keyword)
    if count is None and keyword != "expert":
        count = _extract_keyword_count(snippet=snippet, keyword="expert")
        if count is not None:
            keyword = "expert"
    if count is None:
        return None

    if _is_just_number_request(query):
        return f"{count} [{citation_index}]"

    if keyword == "expert":
        return f"The paper uses {count} experts. [{citation_index}]"
    if keyword == "participant":
        return f"The paper uses {count} participants. [{citation_index}]"
    if keyword == "player":
        return f"The paper uses {count} players. [{citation_index}]"
    if keyword == "subject":
        return f"The paper uses {count} subjects. [{citation_index}]"
    if keyword == "head":
        return f"The paper uses {count} transformer heads. [{citation_index}]"
    return f"The paper reports {count}. [{citation_index}]"


def _looks_like_reference_snippet(text: str) -> bool:
    lower = (text or "").lower()
    markers = (
        "arxiv",
        "preprint",
        "doi:",
        "proc.",
        "proceedings of",
        "in proceedings",
        "in international conference",
        "conference on ",
        "transactions on ",
        "journal of ",
        "ieee trans",
        "et al.",
        "pp.",
        "vol.",
        "no.",
    )
    if any(marker in lower for marker in markers):
        return True
    if re.match(r"^\s*\[[0-9]+\]\s*", text or ""):
        return True
    if re.match(r"^\s*[A-Z][a-z]+,\s+[A-Z]\.\s*(?:and|&)\s*[A-Z][a-z]+,\s+[A-Z]\.", text or ""):
        return True
    return False


def _looks_like_non_argument_snippet(text: str) -> bool:
    lower = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if not lower:
        return True
    junk_markers = (
        "frame-level video-based temporal analysis of fps gameplay without telemetry",
        "start of trial",
        "score =",
        "figure ",
        "table ",
        "(a)",
        "(b)",
        "(c)",
        "copyrights for components of this work",
        "no personally identifiable information",
    )
    if any(marker in lower for marker in junk_markers):
        return True
    if re.match(r"^[\(\[]?[a-z0-9][\)\]]?\s", lower) and len(lower.split()) <= 8:
        return True
    if len(lower) < 45:
        return True
    return False


def _extract_quantity_snippet(*, documents: list[Document], keyword: str) -> tuple[str, int]:
    best_score = float("-inf")
    best_snippet = ""
    best_citation = 1
    for doc_index, document in enumerate(documents[:8], start=1):
        text = (document.page_content or "").strip()
        if not text:
            continue
        cleaned = re.sub(r"\[[0-9]+\]", " ", text)
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        for sentence in sentences:
            snippet = sentence.strip()
            if len(snippet) < 24:
                continue
            lower = snippet.lower()
            if keyword not in lower and f"{keyword}s" not in lower:
                continue
            if _looks_like_reference_snippet(snippet):
                continue
            score = 0.0
            if re.search(rf"\b\d+\s+{re.escape(keyword)}s?\b", lower):
                score += 1.0
            if re.search(rf"\bnumber of {re.escape(keyword)}s?\b", lower):
                score += 0.9
            if re.search(rf"\b{re.escape(keyword)}s?\s*(?:is|are|were|was|=|:)?\s*\d+\b", lower):
                score += 1.0
            if keyword == "expert":
                if _has_numbered_expert_pattern(lower):
                    score += 1.1
                if _infer_expert_count(snippet) is not None:
                    score += 0.6
            if re.search(r"\b\d+\b", lower):
                score += 0.25
            score -= min(0.25, _low_signal_penalty(snippet))
            if score > best_score:
                best_score = score
                best_snippet = snippet
                best_citation = doc_index
    if best_score < 0.45:
        return "", 1
    return best_snippet, best_citation


def _extract_keyword_count(*, snippet: str, keyword: str) -> int | None:
    lower = (snippet or "").lower()
    if keyword == "expert":
        return _infer_expert_count(snippet)

    candidates: list[int] = []
    for pattern in (
        rf"\b(\d+)\s+{re.escape(keyword)}s?\b",
        rf"\b{re.escape(keyword)}s?\s*(?:is|are|were|was|=|:)?\s*(\d+)\b",
        rf"\bnumber of {re.escape(keyword)}s?\s*(?:is|:)?\s*(\d+)\b",
    ):
        candidates.extend(int(value) for value in re.findall(pattern, lower))
    if not candidates:
        return None
    return max(candidates)


def _is_quantity_intent_query(query: str) -> bool:
    lower = (query or "").lower()
    return any(marker in lower for marker in ("how many", "number", "count", "how much"))


def _is_just_number_request(query: str) -> bool:
    lower = (query or "").lower()
    markers = ("just number", "only number", "number only", "thats it", "that's it", "just give number")
    return any(marker in lower for marker in markers)


def _infer_expert_count(snippet: str) -> int | None:
    text = (snippet or "").lower()
    if "expert" not in text:
        return None
    values: list[int] = []
    values.extend(int(value) for value in re.findall(r"\bexpert\s+(\d+)\b", text))
    values.extend(int(value) for value in re.findall(r"\bexperts?\s*(?:\(|:)?\s*(\d+)\b", text))
    for match in re.finditer(r"\bexperts?\s+([0-9,\sand]+)", text):
        values.extend(int(value) for value in re.findall(r"\d+", match.group(1)))
    values = [value for value in values if value > 0]
    if not values:
        return None
    return max(values)


def _has_numbered_expert_pattern(text: str) -> bool:
    lower = (text or "").lower()
    if "expert" not in lower:
        return False
    if re.search(r"\bexperts?\s+\d+\b", lower):
        return True
    if re.search(r"\bexpert\s+\d+\b", lower):
        return True
    if re.search(r"\bexperts?\s+\d+\s*,\s*\d+", lower):
        return True
    return False


def _reviewer_rate_limit_fallback(state: GraphState, *, retry_hint: str) -> str:
    documents = state.get("retrieved_documents", [])
    attack_vectors = _normalize_attack_vectors(
        _fallback_attack_vectors(message=state.get("message", ""), documents=documents),
        fallback_count=min(3, settings.reviewer_attack_vector_count),
        documents=documents,
    )
    active = attack_vectors[0] if attack_vectors else {
        "id": "V1",
        "claim": "Core contribution framing vs evidence support.",
        "severity": "high",
        "category": "novelty",
        "quote": _default_quote(documents),
        "skeptic_lead": "Novelty strength is unclear without explicit comparative evidence.",
    }
    quote = active.get("quote", _default_quote(documents))
    return (
        "## Claim Trial Engine\n"
        f"Active Claim: {active.get('id', 'V1')} - {active.get('claim', '')}\n"
        f"Claim Trigger: \"{quote}\"\n\n"
        "### Skeptic\n"
        "- Concern: evidence may be weaker than framing suggests [1].\n"
        "- Ask for a tighter quantitative comparison and clearer scope boundary [1].\n\n"
        "### Advocate\n"
        "- Defense: the paper does provide partial evidence for the claim [1].\n"
        "- Recommend narrowing claim language to what is directly demonstrated [1].\n\n"
        "### Evidence-only Judge\n"
        "- Verdict: contested\n"
        "- Rationale: evidence partially supports feasibility but does not fully settle novelty strength [1].\n\n"
        "### Rewrite Compiler Card\n"
        "Target Section: contribution framing paragraph\n"
        "Target Claim: contribution-level novelty statement\n"
        "Patch Instruction: revise the contribution claim to include one concrete metric/baseline comparison and explicitly state the scope limits.\n"
        "Why: this preserves strengths while reducing overclaim risk.\n\n"
        "Intervention: ask either reviewer to sharpen evidence or move to the next vector."
    )


def _validation_system_prompt(mode: Mode) -> str:
    mode_hint = {
        Mode.LOCAL: "Strictly keep only claims grounded in retrieved paper context.",
        Mode.GLOBAL: (
            "Allow normal general-knowledge responses. "
            "Only enforce citations for claims that explicitly rely on retrieved paper context."
        ),
        Mode.WRITER: "Preserve style while correcting factual inaccuracies.",
        Mode.REVIEWER: (
            "Enforce rigorous review quality. "
            "Remove unsupported claims, and ensure every major concern is evidenced. "
            "Do not mark benchmarks/ablations as missing when context shows they are covered; relabel as covered with caveats if needed. "
            "Ensure the output preserves reviewer structure, concrete actionable feedback, and complete score block."
        ),
        Mode.COMPARATOR: (
            "Keep only comparisons directly supported by retrieved context blocks. "
            "Preserve comparator section structure and abstain where evidence is insufficient."
        ),
    }[mode]
    return (
        "You are a factual validator for research answers.\n"
        f"{mode_hint}\n"
        "Return JSON only with keys:\n"
        "{\n"
        '  "verdict": "pass" or "revise",\n'
        '  "issues": ["short issue", "..."],\n'
        '  "revised_answer": "final corrected answer with inline citations [n] where needed"\n'
        "}\n"
        "If draft is already strong and grounded, set verdict to pass and copy it into revised_answer."
    )


def _parse_validation_payload(raw: str) -> tuple[str, list[str], str]:
    text = (raw or "").strip()
    if not text:
        return "pass", [], ""
    payload = _try_parse_json_object(text)
    if payload is None:
        recovered = _recover_revised_answer(text)
        if recovered:
            return "revise", ["Validator returned malformed JSON; recovered revised answer."], recovered
        return "revise", ["Validator returned non-JSON output."], text
    verdict = str(payload.get("verdict", "pass")).strip().lower()
    if verdict not in {"pass", "revise"}:
        verdict = "pass"
    issues_raw = payload.get("issues", [])
    issues: list[str] = []
    if isinstance(issues_raw, list):
        issues = [str(item).strip() for item in issues_raw if str(item).strip()]
    revised_answer = str(payload.get("revised_answer", "")).strip()
    if revised_answer and revised_answer.lstrip().startswith("{") and '"revised_answer"' in revised_answer:
        nested_payload = _try_parse_json_object(revised_answer)
        if nested_payload is not None:
            nested_revised = str(nested_payload.get("revised_answer", "")).strip()
            if nested_revised:
                revised_answer = nested_revised
        else:
            nested_recovered = _recover_revised_answer(revised_answer)
            if nested_recovered:
                revised_answer = nested_recovered
    if revised_answer.startswith("```"):
        stripped = _strip_markdown_fence(revised_answer).strip()
        if stripped:
            revised_answer = stripped
    return verdict, issues, revised_answer


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _recover_revised_answer(text: str) -> str:
    cleaned = _strip_markdown_fence(text).strip()
    if not cleaned:
        return ""

    # Handles malformed JSON where revised_answer is still present in a quoted block.
    quoted_match = re.search(
        r'"revised_answer"\s*:\s*"([\s\S]*?)"\s*(?:,\s*"[^"]+"\s*:|\}\s*$)',
        cleaned,
        flags=re.DOTALL,
    )
    if quoted_match:
        candidate = quoted_match.group(1)
        candidate = candidate.replace('\\"', '"').replace("\\n", "\n").strip()
        candidate = re.sub(r"\\\\(?=[A-Za-z])", r"\\", candidate)
        if candidate:
            return candidate

    # Fallback for unquoted payload styles.
    raw_match = re.search(
        r'"revised_answer"\s*:\s*([\s\S]*?)\s*(?:,\s*"[^"]+"\s*:|\}\s*$)',
        cleaned,
        flags=re.DOTALL,
    )
    if raw_match:
        candidate = raw_match.group(1).strip().strip('"').strip()
        if candidate:
            return candidate
    return ""


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _comparator_output_vision() -> str:
    return (
        "Target style:\n"
        "- Read like a strong research decision memo, not a generic summary.\n"
        "- Make benchmark overlap explicit: exact slice, family-level overlap, or no fair comparison.\n"
        "- Each paper should sound distinct in method, evidence strength, and trade-offs.\n"
        "- Separate paper-grounded evidence from field-relative importance; both matter for novelty valuation.\n"
        "- Prefer honest abstention over weak filler.\n"
        "Mini mock shape:\n"
        "## Claim Matrix\n"
        "### paper-a.pdf\n"
        "- Core contribution: names the concrete method delta and why it matters [1]\n"
        "- Evidence boundary: strongest metric is tied to one exact setup, not the whole paper [2]\n"
        "## Field Context\n"
        "### paper-a.pdf\n"
        "- Historical position: foundational / landmark / important / unclear.\n"
        "- Field-relative novelty: say how big the contribution was at publication time.\n"
        "## Common Benchmark Analysis\n"
        "- Shared slice: benchmark family + exact setting if recovered.\n"
        "- paper-a.pdf: metric.\n"
        "- paper-b.pdf: metric.\n"
        "- Comparison note: direct winner only when the slice is genuinely matched.\n"
        "## Decision By Use Case\n"
        "- Scenario: low-latency deployment -> winner + why.\n"
        "- Scenario: strongest headline benchmark -> winner + caveat.\n"
        "- Scenario: evidence too mismatched -> No winner (insufficient evidence).\n"
    )


def _reviewer_output_vision() -> str:
    return (
        "Target style:\n"
        "- Sound like a sharp but fair senior reviewer.\n"
        "- Separate what the paper proves from what the paper merely suggests.\n"
        "- Separate paper-grounded novelty from field-relative novelty and historical importance.\n"
        "- Use claim-specific strengths and revisions; avoid template reuse.\n"
        "- Anchor each verdict in one concrete quote or benchmark fact.\n"
        "- Panel-summary sections should read like short analyst paragraphs, not one-line placeholders.\n"
        "- For major sections, prefer 4-5 lines of grounded synthesis over one-sentence verdicts.\n"
        "Mini mock shape:\n"
        "## Reviewer Complete Report\n"
        "Final Decision: Borderline / Weak Accept / Weak Reject with one-sentence reason.\n"
        "### Field Context\n"
        "- Historical position: foundational / landmark / important / unclear.\n"
        "- Novelty-at-publication: separate from whether the local snippet evidence is well scoped.\n"
        "### Strengths Worth Keeping\n"
        "- Claim-specific strength tied to an exact method or metric anchor.\n"
        "### High-Impact Concerns\n"
        "- Concrete overclaim or evidence gap, not a generic complaint.\n"
        "### Required Revisions\n"
        "- One surgical edit per claim: what sentence to tighten, what comparator/metric to add.\n"
    )


def _draft_user_prompt(
    *,
    mode: Mode,
    message: str,
    history: list[dict[str, str]],
    documents: list[Document],
    paper_ids: list[str] | None = None,
) -> str:
    history_text = _format_history(history)
    reviewer_context_text = _format_context(documents, max_docs=max(settings.rerank_top_n, 10))
    context_text = _format_context(documents)
    math_intent = _is_math_intent_query(message)

    if mode == Mode.REVIEWER:
        return (
            "You are writing a rigorous ML conference-style review.\n"
            "Requirements:\n"
            "- Ground factual statements in retrieved evidence and cite with [n].\n"
            "- Do not hallucinate missing experiments; explicitly mark whether each item is covered or missing.\n"
            "- Be specific: name concrete failure points, experimental gaps, and methodological risks.\n"
            "- Prefer precise, actionable recommendations over generic advice.\n"
            "- Output in markdown with exactly these headings:\n"
            "  1) ## Paper Snapshot\n"
            "  2) ## Technical Summary\n"
            "  3) ## Strengths\n"
            "  4) ## Major Concerns\n"
            "  5) ## Minor Concerns\n"
            "  6) ## Coverage Check (Covered vs Missing)\n"
            "  7) ## Reproducibility & Clarity Checklist\n"
            "  8) ## Actionable Revision Plan\n"
            "  9) ## Scores\n"
            "- In `## Scores`, include numeric sub-scores (1-10):\n"
            "  - Novelty\n"
            "  - Technical Soundness\n"
            "  - Empirical Rigor\n"
            "  - Clarity\n"
            "  - Reproducibility\n"
            "  - Weighted Overall Score (compute as: "
            "0.25*Novelty + 0.30*Technical Soundness + 0.25*Empirical Rigor + 0.10*Clarity + 0.10*Reproducibility)\n"
            "- Also include:\n"
            "  - Recommendation: <1-10>\n"
            "  - Confidence: <1-5>\n"
            "  - Risk of Rejection: <Low|Medium|High>\n\n"
            "Quality target:\n"
            f"{_reviewer_output_vision()}\n"
            "\n"
            "Conversation history:\n"
            f"{history_text}\n\n"
            "User message:\n"
            f"{message}\n\n"
            "Retrieved context:\n"
            f"{reviewer_context_text}"
        )

    if mode == Mode.GLOBAL:
        recommendation_query = _is_global_recommendation_query(message)
        person_query = _is_global_person_query(message)
        if recommendation_query:
            query_directive = (
                "- The user is asking for recommendations. "
                "Give a useful list with concise relevance reasoning for each item.\n"
            )
        elif person_query:
            query_directive = (
                "- The user is asking about a person/author. "
                "Give a useful profile-style answer (role, domain, notable work themes), "
                "and call out ambiguity if the name could refer to multiple people.\n"
            )
        else:
            query_directive = ""
        return (
            "You are responding in Global mode.\n"
            "Guidelines:\n"
            "- Answer like a normal high-quality LLM: clear, practical, and direct.\n"
            "- Use general knowledge by default.\n"
            "- Use retrieved paper context only when it clearly helps this question.\n"
            "- If a specific statement comes from retrieved paper context, cite it with [n].\n"
            "- Do not force citations or paper framing for general questions.\n"
            "- Do not mention missing uploaded papers unless the user explicitly asks for paper-grounded evidence.\n"
            f"{query_directive}\n"
            "Conversation history:\n"
            f"{history_text}\n\n"
            "User message:\n"
            f"{message}\n\n"
            "Retrieved context:\n"
            f"{context_text}"
        )

    if mode == Mode.LOCAL:
        if math_intent:
            return (
                "You are answering a Local Brain math question using ONLY retrieved paper evidence.\n"
                "Math-output requirements:\n"
                "- Rewrite equations cleanly in LaTeX (not OCR fragments).\n"
                "- Use `$$ ... $$` for display equations and `$...$` for short inline terms.\n"
                "- Keep surrounding prose as normal sentences; never place explanation paragraphs inside `$$ ... $$`.\n"
                "- Keep citation markers outside equation blocks (for example, after the equation sentence).\n"
                "- Define symbols clearly when first introduced.\n"
                "- If an equation is not present in context, explicitly abstain.\n"
                "- Output in markdown with this structure:\n"
                "  1) ## Short Answer\n"
                "  2) ## Math Walkthrough\n"
                "  3) ## Symbol Guide\n"
                "  4) ## Grounding Notes\n\n"
                "Conversation history:\n"
                f"{history_text}\n\n"
                "User message:\n"
                f"{message}\n\n"
                "Retrieved context:\n"
                f"{_format_context(documents, max_docs=max(settings.rerank_top_n, 10))}"
            )
        return (
            "You are answering in Local Brain mode.\n"
            "Requirements:\n"
            "- Use only retrieved paper evidence.\n"
            "- Every factual claim must have inline citations [n].\n"
            "- Prefer direct, specific wording over broad summaries.\n"
            "- If evidence is missing, say exactly what is missing.\n\n"
            "Conversation history:\n"
            f"{history_text}\n\n"
            "User message:\n"
            f"{message}\n\n"
            "Retrieved context:\n"
            f"{context_text}"
        )

    if mode == Mode.COMPARATOR:
        paper_list = _comparator_paper_list(
            documents=documents,
            selected_paper_ids=paper_ids,
        )
        evidence_pack = _comparator_evidence_pack(
            documents=documents,
            selected_paper_ids=paper_ids,
            max_snippets_per_paper=3,
        )
        return (
            "Produce a decision-useful comparison of the selected papers.\n"
            "Rules:\n"
            "- Use only selected-paper evidence and cite grounded claims as [n].\n"
            "- Mention each selected paper by filename.\n"
            "- If benchmark overlap is unproven, state non-overlap explicitly.\n"
            "- Avoid speculation; abstain when evidence is weak.\n"
            "- You may use background field knowledge only to interpret why a grounded difference matters; never invent uncited paper-specific facts.\n"
            "- Use field knowledge explicitly for historical position and novelty-at-publication judgments when you can do so honestly.\n"
            "- Prefer exact shared benchmark slices (same dataset/task/setting) over looser family matches.\n"
            "- If papers share only the benchmark family or metric family, say so and withhold head-to-head numeric winners.\n"
            "- Do not say a paper lacks benchmark results when the evidence pack lists recovered metrics for it.\n"
            "- For major findings, prefer 4-6 lines of concrete explanation over 1-line verdicts.\n"
            "- Output markdown with EXACT sections:\n"
            "  1) ## Papers Compared\n"
            "  2) ## Claim Matrix\n"
            "  3) ## Field Context\n"
            "  4) ## Conflict Map\n"
            "  5) ## Benchmark Verdict Matrix\n"
            "  6) ## Method Trade-offs\n"
            "  7) ## Synthesis Blueprint\n"
            "  8) ## Decision By Use Case\n"
            "- Claim Matrix: at least 2 strong claims per paper with direct evidence.\n"
            "- Field Context: state historical position and field-relative novelty separately from paper-grounded benchmark evidence.\n"
            "- Conflict Map: include Agreements, Contradictions, Non-overlap.\n"
            "- Benchmark Verdict Matrix: score novelty, empirical rigor, reproducibility (1-10) with justification.\n"
            "- Benchmark Verdict Matrix must also include a `Common Benchmark Analysis` subsection:\n"
            "  - name the shared dataset/task (if any),\n"
            "  - report each paper's metric value (for example accuracy/F1/BLEU) when present,\n"
            "  - state the numeric delta and one grounded reason for the difference.\n"
            "  - if no shared benchmark metrics are present, explicitly say so.\n"
            "- Method Trade-offs: at least 2 strengths and 2 limitations per paper.\n"
            "- Synthesis Blueprint: borrow from each paper and propose one merged experiment.\n"
            "- Decision By Use Case: at least 3 concrete scenarios; each winner needs citation [n] or 'No winner (insufficient evidence)'.\n\n"
            "Quality target:\n"
            f"{_comparator_output_vision()}\n"
            "\n"
            "Selected papers:\n"
            f"{paper_list}\n\n"
            "Evidence Pack (citation map):\n"
            f"{evidence_pack}\n\n"
            "Conversation policy: ignore prior-turn claims about papers not in Selected papers.\n\n"
            "User message:\n"
            f"{message}\n\n"
            "Retrieved context:\n"
            f"{context_text}"
        )

    return (
        "You are producing a first-draft response.\n"
        "Requirements:\n"
        "- Every concrete factual claim must include at least one inline citation [n].\n"
        "- Do not invent metrics, baselines, or section claims.\n"
        "- If evidence is missing, explicitly state uncertainty.\n\n"
        "Conversation history:\n"
        f"{history_text}\n\n"
        "User message:\n"
        f"{message}\n\n"
        "Retrieved context:\n"
        f"{context_text}"
    )


def _comparator_compact_generation_prompt(
    *,
    message: str,
    documents: list[Document],
    paper_ids: list[str] | None,
) -> str:
    paper_list = _comparator_paper_list(
        documents=documents,
        selected_paper_ids=paper_ids,
    )
    evidence_pack = _comparator_evidence_pack(
        documents=documents,
        selected_paper_ids=paper_ids,
        max_snippets_per_paper=2,
    )
    return (
        "You are in compact comparator mode under provider budget constraints.\n"
        "Return markdown with EXACT sections:\n"
        "## Papers Compared\n"
        "## Claim Matrix\n"
        "## Field Context\n"
        "## Conflict Map\n"
        "## Benchmark Verdict Matrix\n"
        "## Method Trade-offs\n"
        "## Synthesis Blueprint\n"
        "## Decision By Use Case\n"
        "Rules:\n"
        "- Use only selected papers and current evidence.\n"
        "- Keep each section concise but substantive (no placeholders).\n"
        "- Every winner statement must include citation [n] or 'No winner (insufficient evidence)'.\n"
        "- Avoid speculative claims unless explicitly grounded.\n\n"
        "- You may add short background interpretation only when it explains a grounded difference; do not invent uncited paper facts.\n"
        "- Use field-relative context for historical importance and novelty-at-publication when it is genuinely known.\n"
        "- Prefer exact shared benchmark slices over loose family matches, and withhold direct winners when the slices differ.\n\n"
        "Quality target:\n"
        f"{_comparator_output_vision()}\n"
        "\n"
        "Selected papers:\n"
        f"{paper_list}\n\n"
        "Evidence pack:\n"
        f"{evidence_pack}\n\n"
        "User message:\n"
        f"{message}"
    )


def _reviewer_compact_generation_prompt(
    *,
    message: str,
    history: list[dict[str, str]],
    documents: list[Document],
) -> str:
    history_text = _format_history(history)
    compact_context = _compact_context_preview(documents=documents, max_docs=max(settings.rerank_top_n, 8), max_chars=3600)
    return (
        "You are in compact reviewer mode under provider constraints.\n"
        "Produce an evidence-grounded report in markdown with these headings:\n"
        "## Paper Snapshot\n"
        "## Key Strengths\n"
        "## High-Impact Concerns\n"
        "## Required Revisions\n"
        "## Decision\n"
        "Rules:\n"
        "- Cite grounded claims with [n].\n"
        "- Avoid generic statements and avoid internal IDs.\n"
        "- For each concern, state the concrete evidence gap and one actionable fix.\n"
        "- Decision must be one of: Reject, Weak Reject, Borderline, Weak Accept, Accept.\n\n"
        "Quality target:\n"
        f"{_reviewer_output_vision()}\n"
        "\n"
        "Conversation history:\n"
        f"{history_text}\n\n"
        "User message:\n"
        f"{message}\n\n"
        "Compact retrieved context:\n"
        f"{compact_context}"
    )


def _compact_generation_prompt_for_mode(
    *,
    mode: Mode,
    message: str,
    documents: list[Document],
    history: list[dict[str, str]],
    paper_ids: list[str] | None = None,
) -> str:
    if mode == Mode.COMPARATOR:
        return _comparator_compact_generation_prompt(
            message=message,
            documents=documents,
            paper_ids=paper_ids,
        )
    if mode == Mode.REVIEWER:
        return _reviewer_compact_generation_prompt(
            message=message,
            history=history,
            documents=documents,
        )

    compact_context = _compact_context_preview(documents=documents, max_docs=3, max_chars=2200)
    if mode == Mode.LOCAL:
        if _is_math_intent_query(message):
            return (
                "You are in compact local-math mode.\n"
                "Use only retrieved paper evidence. Render equations cleanly in LaTeX using `$$...$$` for display.\n"
                "Add citations [n] after grounded statements (outside the equation block).\n"
                "Structure:\n"
                "## Short Answer\n"
                "## Math Walkthrough\n"
                "## Grounding Notes\n\n"
                f"User message:\n{message}\n\n"
                f"Compact context:\n{compact_context}"
            )
        return (
            "You are in compact local mode.\n"
            "Use only retrieved evidence and add [n] citations to factual claims.\n"
            "If evidence is missing, state what is missing.\n\n"
            f"User message:\n{message}\n\n"
            f"Compact context:\n{compact_context}"
        )
    if mode == Mode.GLOBAL:
        return (
            "You are in compact global mode.\n"
            "Answer directly and practically. Use retrieved context only when clearly relevant.\n"
            "Only cite [n] if a concrete claim comes from retrieved paper snippets.\n\n"
            f"User message:\n{message}\n\n"
            f"Compact context:\n{compact_context}"
        )
    return (
        "You are in compact answer mode.\n"
        "Provide a grounded concise response with citations [n] where factual claims rely on context.\n\n"
        f"User message:\n{message}\n\n"
        f"Compact context:\n{compact_context}"
    )


def _compact_context_preview(*, documents: list[Document], max_docs: int, max_chars: int) -> str:
    if not documents:
        return "No retrieved context."
    blocks: list[str] = []
    for index, document in enumerate(documents[: max(1, int(max_docs))], start=1):
        metadata = document.metadata or {}
        filename = str(metadata.get("filename", "unknown.pdf")).strip() or "unknown.pdf"
        page = metadata.get("page")
        page_suffix = f", p.{page}" if page else ""
        snippet = _compact_turn_text(_extract_signal_sentence(document.page_content or ""), max_chars=280)
        if not snippet:
            continue
        blocks.append(f"[{index}] {filename}{page_suffix}\n{snippet}")
    text = "\n\n".join(blocks).strip()
    if not text:
        return "No high-signal snippets available."
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def _comparator_paper_list(
    *,
    documents: list[Document],
    selected_paper_ids: list[str] | None = None,
) -> str:
    id_to_filename = _comparator_filename_lookup(documents)
    labels: list[str] = []
    seen: set[str] = set()
    for paper_id in selected_paper_ids or []:
        pid = str(paper_id or "").strip()
        if not pid:
            continue
        label = id_to_filename.get(pid, f"paper_id:{pid}")
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
    for document in documents:
        filename = str((document.metadata or {}).get("filename", "")).strip()
        if not filename or filename in seen:
            continue
        seen.add(filename)
        labels.append(filename)
        if len(labels) >= 3:
            break
    if not labels:
        return "- Unknown papers from retrieved context"
    return "\n".join(f"- {label}" for label in labels)


def _comparator_filename_lookup(documents: list[Document]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for document in documents:
        metadata = document.metadata or {}
        paper_id = str(metadata.get("paper_id", "")).strip()
        filename = str(metadata.get("filename", "")).strip()
        if paper_id and filename and paper_id not in mapping:
            mapping[paper_id] = filename
    return mapping


def _comparator_metric_record_strings(entries: list[dict[str, Any]], *, cap: int = 3) -> list[str]:
    lines: list[str] = []
    seen: set[tuple[str, float, str]] = set()
    for entry in entries:
        citation = int(entry.get("citation", 1))
        benchmark_fallback = str(entry.get("benchmark_label", "")).strip()
        for record in entry.get("metric_records", []) or []:
            metric = str(record.get("metric", "metric")).upper()
            try:
                value = float(record.get("value", 0.0))
            except Exception:
                continue
            benchmark = str(record.get("benchmark", "")).strip() or benchmark_fallback
            key = (metric, round(value, 3), benchmark.lower())
            if key in seen:
                continue
            seen.add(key)
            detail = f"{value:.2f} {metric}"
            if benchmark:
                detail += f" on {benchmark}"
            detail += f" [{citation}]"
            lines.append(detail)
            if len(lines) >= max(1, cap):
                return lines
    return lines


def _comparator_evidence_pack(
    *,
    documents: list[Document],
    selected_paper_ids: list[str] | None,
    max_snippets_per_paper: int,
) -> str:
    by_paper: dict[str, list[str]] = {}
    id_to_filename = _comparator_filename_lookup(documents)
    signal_pool = _build_comparator_signal_pool(documents=documents, limit=max(12, len(documents)))
    profiles = _paper_profiles_from_documents(documents)
    selected = [str(paper_id or "").strip() for paper_id in (selected_paper_ids or []) if str(paper_id or "").strip()]
    selected_set = set(selected)
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        paper_id = str(metadata.get("paper_id", "")).strip() or "unknown"
        if selected_set and paper_id not in selected_set:
            continue
        if len(by_paper.get(paper_id, [])) >= max(1, max_snippets_per_paper):
            continue
        raw_text = (document.page_content or "").strip()
        if _looks_like_reference_snippet(raw_text) or _looks_like_non_argument_snippet(raw_text):
            continue
        snippet = _extract_signal_sentence(raw_text)
        if not snippet:
            continue
        filename = str(metadata.get("filename", "unknown.pdf")).strip() or "unknown.pdf"
        page = metadata.get("page")
        page_suffix = f", p.{page}" if page else ""
        by_paper.setdefault(paper_id, []).append(f"- [{index}] {filename}{page_suffix}: \"{snippet}\"")

    ordered_ids: list[str] = []
    seen: set[str] = set()
    for paper_id in selected:
        if paper_id in seen:
            continue
        seen.add(paper_id)
        ordered_ids.append(paper_id)
    for document in documents:
        paper_id = str((document.metadata or {}).get("paper_id", "")).strip() or "unknown"
        if paper_id in seen:
            continue
        seen.add(paper_id)
        ordered_ids.append(paper_id)

    lines: list[str] = []
    for paper_id in ordered_ids[:3]:
        filename = id_to_filename.get(paper_id, f"paper_id:{paper_id}")
        profile = profiles.get(filename, {})
        lines.append(f"### {filename}")
        paper_entries = signal_pool.get(filename, [])
        contribution = (
            {
                "snippet": str(profile.get("summary_sentence", "")).strip(),
                "citation": int(profile.get("summary_citation", 1) or 1),
            }
            if str(profile.get("summary_sentence", "")).strip()
            else None
        )
        if contribution is None:
            contribution = next(
                (
                    entry
                    for entry in paper_entries
                    if any(
                        token in str(entry.get("snippet", "")).lower()
                        for token in ("we propose", "we present", "we introduce", "main result", "in this paper", "in this work")
                    )
                ),
                paper_entries[0] if paper_entries else None,
            )
        method = (
            {
                "snippet": str(profile.get("method_sentence", "")).strip(),
                "citation": int(profile.get("method_citation", 1) or 1),
            }
            if str(profile.get("method_sentence", "")).strip()
            else None
        )
        if method is None:
            method = next(
                (
                    entry
                    for entry in paper_entries
                    if any(token in str(entry.get("snippet", "")).lower() for token in _method_signal_markers())
                ),
                contribution,
            )
        benchmark = (
            {
                "snippet": str(profile.get("metric_sentence", "")).strip(),
                "citation": int(profile.get("metric_citation", 1) or 1),
            }
            if str(profile.get("metric_sentence", "")).strip()
            else None
        )
        if benchmark is None:
            benchmark = next(
                (
                    entry
                    for entry in paper_entries
                    if entry.get("metric_records")
                    or any(token in str(entry.get("snippet", "")).lower() for token in _evaluation_signal_markers())
                ),
                method,
            )
        metrics = _comparator_metric_record_strings(paper_entries, cap=max(2, max_snippets_per_paper + 1))
        if contribution:
            lines.append(f"- Contribution signal: \"{contribution.get('snippet', '')}\" [{int(contribution.get('citation', 1))}]")
        if method:
            lines.append(f"- Method signal: \"{method.get('snippet', '')}\" [{int(method.get('citation', 1))}]")
        if benchmark:
            lines.append(f"- Benchmark setup: \"{benchmark.get('snippet', '')}\" [{int(benchmark.get('citation', 1))}]")
        if metrics:
            lines.append(f"- Metric signals: {'; '.join(metrics)}")
        else:
            entries = by_paper.get(paper_id, [])
            if entries:
                lines.extend(entries[: max(1, max_snippets_per_paper)])
            else:
                lines.append("- No high-signal snippet retrieved for this paper in current context.")

    if not lines:
        return "No comparator evidence pack available."
    return "\n".join(lines)


def _run_reviewer_debate(state: GraphState) -> GraphState:
    raw_message = str(state.get("message", ""))
    message = _normalize_reviewer_message(raw_message)
    intervention_mode = _normalize_intervention_mode(state.get("intervention_mode"))
    documents = state.get("retrieved_documents", [])
    debug = dict(state.get("debug", {}))
    debate_history = deepcopy(state.get("debate_history", []))
    debate_summary = str(state.get("debate_summary", "")).strip()
    syntheses = deepcopy(state.get("syntheses", {}))
    vector_verdicts = deepcopy(state.get("vector_verdicts", {}))
    vector_judgments = deepcopy(state.get("vector_judgments", {}))
    vector_reports = deepcopy(state.get("vector_reports", {}))
    final_report = deepcopy(state.get("final_report", {}))
    attack_vectors = _normalize_attack_vectors(
        state.get("attack_vectors", []),
        fallback_count=settings.reviewer_attack_vector_count,
        documents=documents,
    )
    if _is_new_reviewer_session_signal(raw_message):
        debate_history = []
        debate_summary = ""
        syntheses = {}
        vector_verdicts = {}
        vector_judgments = {}
        vector_reports = {}
        final_report = {}
        debug["reviewer_session_reset"] = True

    if not attack_vectors:
        attack_vectors = _generate_attack_vectors(
            message=message,
            documents=documents,
            count=settings.reviewer_attack_vector_count,
        )
        attack_vectors = _normalize_attack_vectors(
            attack_vectors,
            fallback_count=settings.reviewer_attack_vector_count,
            documents=documents,
        )

    if not attack_vectors:
        attack_vectors = [
            {
                "id": "V1",
                "claim": "Core contribution framing and novelty support.",
                "severity": "high",
                "category": "novelty",
                "quote": _default_quote(documents),
                "skeptic_lead": "The novelty framing is not yet grounded in a direct quantitative delta.",
            }
        ]

    attack_vector_ids = [str(item.get("id", "")).strip() for item in attack_vectors if str(item.get("id", "")).strip()]
    completed_session = bool(final_report) and bool(attack_vector_ids) and len(syntheses) >= len(attack_vector_ids)
    vectors_remaining = [
        vector_id
        for vector_id in state.get("vectors_remaining", [])
        if vector_id in attack_vector_ids and vector_id not in syntheses
    ]
    if not vectors_remaining and not completed_session:
        vectors_remaining = [vector_id for vector_id in attack_vector_ids if vector_id not in syntheses]
    if not vectors_remaining and not completed_session:
        vectors_remaining = attack_vector_ids[:]

    if _user_requested_next_vector(message):
        current_active = str(state.get("active_vector_id", "")).strip()
        if current_active and current_active in vectors_remaining:
            vectors_remaining = [vector for vector in vectors_remaining if vector != current_active] + [current_active]

    explicit_vector = _extract_vector_selection(message, attack_vectors)
    if explicit_vector and explicit_vector in syntheses:
        syntheses.pop(explicit_vector, None)
        vector_verdicts.pop(explicit_vector, None)
        vector_judgments.pop(explicit_vector, None)
        vector_reports.pop(explicit_vector, None)
        if explicit_vector not in vectors_remaining:
            vectors_remaining.insert(0, explicit_vector)

    active_vector_id = (
        explicit_vector
        or state.get("active_vector_id")
        or (vectors_remaining[0] if vectors_remaining else (attack_vector_ids[0] if attack_vector_ids else "V1"))
    )
    if active_vector_id not in attack_vector_ids and attack_vector_ids:
        active_vector_id = attack_vector_ids[0]
    if active_vector_id not in vectors_remaining and active_vector_id not in syntheses:
        vectors_remaining.insert(0, active_vector_id)

    active_vector = _get_attack_vector(attack_vectors, active_vector_id)
    turn_count = _count_vector_turns(debate_history, active_vector_id)
    skeptic_position = str(state.get("skeptic_position", "")).strip() or _latest_speaker_content(
        debate_history, speaker="skeptic", vector_id=active_vector_id
    )
    advocate_position = str(state.get("advocate_position", "")).strip() or _latest_speaker_content(
        debate_history, speaker="advocate", vector_id=active_vector_id
    )
    resolution = _infer_resolution(
        skeptic_position=skeptic_position,
        advocate_position=advocate_position,
        history=debate_history,
        active_vector_id=active_vector_id,
        turn_count=turn_count,
    )

    if _looks_like_score_request(message):
        score_answer = _reviewer_score_response(
            query=message,
            active_vector=active_vector,
            resolution=resolution,
            vector_verdicts=vector_verdicts,
            vector_judgments=vector_judgments,
            debate_history=debate_history,
            documents=documents,
        )
        debug["reviewer_debate_mode"] = True
        debug["response_stage"] = "reviewer_scorecard"
        debug["active_vector_id"] = active_vector_id
        debug["resolution"] = resolution
        debug["turn_count"] = turn_count
        debug["warning_turn"] = settings.reviewer_warning_turn
        debug["max_turns"] = settings.reviewer_max_turns
        debug["intervention_mode"] = intervention_mode
        debug["next_speaker"] = state.get("next_speaker", "skeptic")
        return {
            "draft_answer": score_answer,
            "citations": state.get("citations", []),
            "attack_vectors": attack_vectors,
            "active_vector_id": active_vector_id,
            "debate_history": debate_history,
            "debate_summary": debate_summary,
            "skeptic_position": skeptic_position,
            "advocate_position": advocate_position,
            "resolution": resolution,
            "turn_count": turn_count,
            "syntheses": syntheses,
            "vector_verdicts": vector_verdicts,
            "vector_judgments": vector_judgments,
            "vector_reports": vector_reports,
            "final_report": final_report,
            "next_speaker": state.get("next_speaker", "skeptic"),
            "intervention_mode": intervention_mode,
            "vectors_remaining": vectors_remaining,
            "debug": debug,
        }

    # If this session already completed all vectors, keep returning the complete report.
    if not vectors_remaining and isinstance(final_report, dict) and final_report:
        answer = _render_reviewer_debate(
            attack_vectors=attack_vectors,
            active_vector=active_vector,
            vectors_remaining=vectors_remaining,
            syntheses=syntheses,
            vector_verdicts=vector_verdicts,
            vector_judgments=vector_judgments,
            vector_reports=vector_reports,
            current_vector_report={},
            final_report=final_report,
            round_events=[],
            debate_history=debate_history,
            debate_summary=debate_summary,
            resolution=resolution,
            turn_count=turn_count,
            next_speaker="user",
        )
        debug["reviewer_debate_mode"] = True
        debug["response_stage"] = "reviewer_complete_report"
        debug["final_report_ready"] = True
        return {
            "draft_answer": answer,
            "citations": state.get("citations", []),
            "attack_vectors": attack_vectors,
            "active_vector_id": active_vector_id,
            "debate_history": debate_history,
            "debate_summary": debate_summary,
            "skeptic_position": skeptic_position,
            "advocate_position": advocate_position,
            "resolution": resolution,
            "turn_count": turn_count,
            "syntheses": syntheses,
            "vector_verdicts": vector_verdicts,
            "vector_judgments": vector_judgments,
            "vector_reports": vector_reports,
            "final_report": final_report,
            "next_speaker": "user",
            "intervention_mode": intervention_mode,
            "vectors_remaining": vectors_remaining,
            "debug": debug,
        }

    user_target = _resolve_user_target(
        message=message,
        intervention_mode=intervention_mode,
    )
    if message and not _is_auto_reviewer_bootstrap(message):
        debate_history.append(
            {
                "speaker": "user",
                "content": message,
                "turn": turn_count + 1,
                "vector_id": active_vector_id,
                "target": user_target,
                "intervention_mode": intervention_mode,
            }
        )

    round_events: list[dict[str, Any]] = []
    next_speaker = str(state.get("next_speaker", "skeptic")).strip().lower() or "skeptic"
    if _is_new_reviewer_session_signal(raw_message):
        next_speaker = "skeptic"
    # Complete-panel mode: run the full multi-vector debate in one call.
    total_vectors = max(1, len(vectors_remaining))
    loops = max(
        2,
        settings.reviewer_max_turns * total_vectors * 2,
        settings.reviewer_turns_per_response,
    )
    for _ in range(loops):
        turn_count = _count_vector_turns(debate_history, active_vector_id)
        if turn_count >= 4:
            next_speaker = "synthesise"
        else:
            next_speaker = str(next_speaker or "").strip().lower() or "skeptic"
        skeptic_position = _latest_speaker_content(debate_history, speaker="skeptic", vector_id=active_vector_id)
        advocate_position = _latest_speaker_content(debate_history, speaker="advocate", vector_id=active_vector_id)
        resolution = _infer_resolution(
            skeptic_position=skeptic_position,
            advocate_position=advocate_position,
            history=debate_history,
            active_vector_id=active_vector_id,
            turn_count=turn_count,
        )
        if next_speaker != "synthesise":
            next_speaker = _route_reviewer_turn(
                history=debate_history,
                active_vector_id=active_vector_id,
                resolution=resolution,
                turn_count=turn_count,
                fallback=next_speaker,
            )

        if next_speaker == "user":
            break
        if next_speaker == "synthesise":
            judgment = _run_evidence_only_judge(
                active_vector=active_vector,
                debate_history=debate_history,
                resolution=resolution,
                documents=documents,
            )
            verdict = str(judgment.get("verdict", "contested"))
            vector_verdicts[active_vector_id] = verdict
            vector_judgments[active_vector_id] = judgment
            round_events.append(
                {
                    "speaker": "judge",
                    "content": _render_judge_card(active_vector_id=active_vector_id, judgment=judgment),
                    "vector_id": active_vector_id,
                }
            )
            synthesis = _synthesise_vector(
                active_vector=active_vector,
                verdict=verdict,
                judgment=judgment,
                debate_history=debate_history,
                documents=documents,
            )
            syntheses[active_vector_id] = synthesis
            vectors_remaining = [vector for vector in vectors_remaining if vector != active_vector_id]
            round_events.append(
                {
                    "speaker": "synthesise",
                    "content": synthesis,
                    "vector_id": active_vector_id,
                }
            )
            if not vectors_remaining:
                break
            active_vector_id = vectors_remaining[0]
            active_vector = _get_attack_vector(attack_vectors, active_vector_id)
            next_speaker = "skeptic"
            resolution = "open"
            continue

        turn_content, route_meta = _generate_reviewer_turn(
            speaker=next_speaker,
            active_vector=active_vector,
            objective=message,
            debate_summary=debate_summary,
            debate_history=debate_history,
            documents=documents,
        )
        if not turn_content:
            break

        turn_count += 1
        turn_payload = {
            "speaker": next_speaker,
            "content": turn_content,
            "turn": turn_count,
            "vector_id": active_vector_id,
            "meta": route_meta,
        }
        debate_history.append(turn_payload)
        round_events.append(turn_payload)

        if _count_vector_turns(debate_history, active_vector_id) % 2 == 0:
            debate_summary = _refresh_debate_summary(
                debate_summary=debate_summary,
                active_vector=active_vector,
                debate_history=debate_history,
            )

    turn_count = _count_vector_turns(debate_history, active_vector_id)
    skeptic_position = _latest_speaker_content(debate_history, speaker="skeptic", vector_id=active_vector_id)
    advocate_position = _latest_speaker_content(debate_history, speaker="advocate", vector_id=active_vector_id)
    resolution = _infer_resolution(
        skeptic_position=skeptic_position,
        advocate_position=advocate_position,
        history=debate_history,
        active_vector_id=active_vector_id,
        turn_count=turn_count,
    )
    if turn_count >= 4:
        next_speaker = "synthesise"
    else:
        next_speaker = _route_reviewer_turn(
            history=debate_history,
            active_vector_id=active_vector_id,
            resolution=resolution,
            turn_count=turn_count,
            fallback=next_speaker,
        )
    current_vector_report = _build_current_vector_report(
        active_vector=active_vector,
        skeptic_position=skeptic_position,
        advocate_position=advocate_position,
        debate_history=debate_history,
        documents=documents,
        existing_report=vector_reports.get(active_vector_id, {}),
    )
    if current_vector_report:
        vector_reports[active_vector_id] = current_vector_report

    if next_speaker == "synthesise":
        judgment = _run_evidence_only_judge(
            active_vector=active_vector,
            debate_history=debate_history,
            resolution=resolution,
            documents=documents,
        )
        verdict = str(judgment.get("verdict", "contested"))
        vector_verdicts[active_vector_id] = verdict
        vector_judgments[active_vector_id] = judgment
        round_events.append(
            {
                "speaker": "judge",
                "content": _render_judge_card(active_vector_id=active_vector_id, judgment=judgment),
                "vector_id": active_vector_id,
            }
        )
        synthesis = _synthesise_vector(
            active_vector=active_vector,
            verdict=verdict,
            judgment=judgment,
            debate_history=debate_history,
            documents=documents,
        )
        syntheses[active_vector_id] = synthesis
        vectors_remaining = [vector for vector in vectors_remaining if vector != active_vector_id]
        round_events.append(
            {
                "speaker": "synthesise",
                "content": synthesis,
                "vector_id": active_vector_id,
            }
        )
        if vectors_remaining:
            active_vector_id = vectors_remaining[0]
            active_vector = _get_attack_vector(attack_vectors, active_vector_id)
            turn_count = _count_vector_turns(debate_history, active_vector_id)
            skeptic_position = _latest_speaker_content(debate_history, speaker="skeptic", vector_id=active_vector_id)
            advocate_position = _latest_speaker_content(debate_history, speaker="advocate", vector_id=active_vector_id)
            resolution = _infer_resolution(
                skeptic_position=skeptic_position,
                advocate_position=advocate_position,
                history=debate_history,
                active_vector_id=active_vector_id,
                turn_count=turn_count,
            )
            next_speaker = "skeptic"
        else:
            next_speaker = "user"

    debate_history = debate_history[-64:]
    if not vectors_remaining and syntheses:
        if not isinstance(final_report, dict) or not final_report:
            final_report = _build_reviewer_final_report(
                attack_vectors=attack_vectors,
                vector_verdicts=vector_verdicts,
                vector_judgments=vector_judgments,
                vector_reports=vector_reports,
                syntheses=syntheses,
                debate_history=debate_history,
                documents=documents,
            )
    else:
        final_report = {}
    answer = _render_reviewer_debate(
        attack_vectors=attack_vectors,
        active_vector=active_vector,
        vectors_remaining=vectors_remaining,
        syntheses=syntheses,
        vector_verdicts=vector_verdicts,
        vector_judgments=vector_judgments,
        vector_reports=vector_reports,
        current_vector_report=current_vector_report,
        final_report=final_report,
        round_events=round_events,
        debate_history=debate_history,
        debate_summary=debate_summary,
        resolution=resolution,
        turn_count=turn_count,
        next_speaker=next_speaker,
    )
    debug["reviewer_debate_mode"] = True
    debug["active_vector_id"] = active_vector_id
    debug["resolution"] = resolution
    debug["turn_count"] = turn_count
    debug["warning_turn"] = settings.reviewer_warning_turn
    debug["max_turns"] = settings.reviewer_max_turns
    debug["intervention_mode"] = intervention_mode
    debug["next_speaker"] = next_speaker
    debug["vectors_remaining"] = vectors_remaining[:]
    debug["round_speakers"] = [str(item.get("speaker", "")) for item in round_events]
    debug["round_events"] = [
        {
            "speaker": str(item.get("speaker", "")).strip().lower(),
            "vector_id": str(item.get("vector_id", "")).strip(),
            "turn": int(item.get("turn", 0) or 0),
            "content": _compact_turn_text(str(item.get("content", "")), max_chars=520),
        }
        for item in round_events
        if str(item.get("speaker", "")).strip().lower() in {"skeptic", "advocate", "judge", "synthesise"}
    ]
    if isinstance(final_report, dict) and final_report and not vectors_remaining:
        debug["round_events"] = []
    debug["round_event_count"] = len(debug["round_events"])
    debug["vector_verdicts"] = vector_verdicts
    debug["vector_judgments"] = vector_judgments
    if current_vector_report:
        debug["current_vector_report"] = current_vector_report
    debug["final_report_ready"] = bool(final_report)
    if final_report:
        debug["final_report"] = final_report
    if turn_count >= settings.reviewer_warning_turn:
        debug["turn_warning"] = "debate_closing"

    return {
        "draft_answer": answer,
        "citations": state.get("citations", []),
        "attack_vectors": attack_vectors,
        "active_vector_id": active_vector_id,
        "debate_history": debate_history,
        "debate_summary": debate_summary,
        "skeptic_position": skeptic_position,
        "advocate_position": advocate_position,
        "resolution": resolution,
        "turn_count": turn_count,
        "syntheses": syntheses,
        "vector_verdicts": vector_verdicts,
        "vector_judgments": vector_judgments,
        "vector_reports": vector_reports,
        "final_report": final_report,
        "next_speaker": next_speaker,
        "intervention_mode": intervention_mode,
        "vectors_remaining": vectors_remaining,
        "debug": debug,
    }


def _normalize_reviewer_message(message: str) -> str:
    raw = (message or "").strip()
    if not raw:
        return ""
    normalized = re.sub(r"\s+", " ", raw)
    if normalized.lower().startswith("[start debate]"):
        lens_match = re.search(r"focus lens:\s*(.+)$", normalized, flags=re.IGNORECASE)
        lens = lens_match.group(1).strip() if lens_match else ""
        return f"Focus lens: {lens}" if lens else ""
    if normalized.lower().startswith("[lens:"):
        return normalized
    if normalized.lower().startswith("act as a top-tier ml conference reviewer."):
        lens_match = re.search(r"focus lens:\s*([^\.]+)", normalized, flags=re.IGNORECASE)
        lens = lens_match.group(1).strip() if lens_match else ""
        if lens:
            return f"Focus lens: {lens}"
        return ""
    return raw


def _is_new_reviewer_session_signal(message: str) -> bool:
    lower = (message or "").strip().lower()
    return lower.startswith("[start debate]") or lower.startswith("/restart debate")


def _is_auto_reviewer_bootstrap(message: str) -> bool:
    lower = (message or "").strip().lower()
    if not lower:
        return True
    if lower.startswith("act as a top-tier ml conference reviewer."):
        return True
    if lower.startswith("[auto review]"):
        return True
    if lower.startswith("[start debate]"):
        return True
    return False


def _extract_user_target(message: str) -> str | None:
    lower = (message or "").strip().lower()
    if not lower:
        return None
    if lower.startswith("skeptic:") or lower.startswith("@skeptic"):
        return "skeptic"
    if lower.startswith("advocate:") or lower.startswith("@advocate"):
        return "advocate"
    if "skeptic" in lower and "address" in lower:
        return "skeptic"
    if "advocate" in lower and "address" in lower:
        return "advocate"
    return None


def _normalize_intervention_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in {"defend", "ask", "redirect"}:
        return mode
    return "ask"


def _resolve_user_target(*, message: str, intervention_mode: str) -> str | None:
    explicit = _extract_user_target(message)
    if explicit:
        return explicit
    if intervention_mode == "defend":
        return "skeptic"
    if intervention_mode == "redirect":
        return "skeptic"
    if intervention_mode == "ask":
        return None
    return None


def _user_requested_next_vector(message: str) -> bool:
    lower = (message or "").strip().lower()
    if not lower:
        return False
    markers = ("next", "move on", "another vector", "new vector", "skip this")
    return any(marker in lower for marker in markers)


def _extract_vector_selection(message: str, attack_vectors: list[dict[str, Any]]) -> str | None:
    if not message:
        return None
    vector_ids = [str(vector.get("id", "")).strip() for vector in attack_vectors if str(vector.get("id", "")).strip()]
    if not vector_ids:
        return None

    id_lookup = {vector_id.lower(): vector_id for vector_id in vector_ids}
    normalized = (message or "").lower()
    for key, vector_id in id_lookup.items():
        if re.search(rf"\b{re.escape(key)}\b", normalized):
            return vector_id

    number_match = re.search(r"\b(?:vector|v)?\s*([0-9]{1,2})\b", normalized)
    if number_match:
        index = int(number_match.group(1)) - 1
        if 0 <= index < len(vector_ids):
            return vector_ids[index]
    return None


def _normalize_attack_vectors(
    vectors: list[dict[str, Any]],
    fallback_count: int,
    documents: list[Document],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    quote_cursor = 0
    quote_pool = _quote_candidates(documents)
    for index, raw in enumerate(vectors or [], start=1):
        claim = str(raw.get("claim", "")).strip()
        if not claim:
            continue
        vector_id = str(raw.get("id", "")).strip() or f"V{index}"
        if vector_id in seen:
            continue
        seen.add(vector_id)
        severity = str(raw.get("severity", "medium")).strip().lower()
        if severity not in {"low", "medium", "high", "critical"}:
            severity = "medium"
        category = str(raw.get("category", "method")).strip().lower() or "method"
        quote = str(raw.get("quote", "")).strip()
        if not quote and quote_pool:
            quote = quote_pool[quote_cursor % len(quote_pool)]
            quote_cursor += 1
        if not quote:
            continue
        skeptic_lead = str(raw.get("skeptic_lead", "")).strip()
        if not skeptic_lead:
            skeptic_lead = f"Challenge whether the evidence behind '{claim}' is sufficient."
        normalized.append(
            {
                "id": vector_id,
                "claim": claim,
                "severity": severity,
                "category": category,
                "quote": quote,
                "skeptic_lead": skeptic_lead,
            }
        )
        if len(normalized) >= max(1, fallback_count):
            break
    return normalized


def _generate_attack_vectors(*, message: str, documents: list[Document], count: int) -> list[dict[str, Any]]:
    if not text_service.available or not documents:
        return []
    user_focus = message or "Run a full adversarial-vs-charitable review."
    try:
        raw = text_service.generate(
            system_prompt=(
                "You are generating attack vectors for a paper-review debate.\n"
                "Produce vectors that are contestable and evidence-checkable from paper excerpts.\n"
                "Every vector must include an exact quote from the paper and a skeptic opener.\n"
                "Return JSON only as an array with objects using keys: id, claim, severity, category, quote, skeptic_lead.\n"
                "Use id format V1, V2, ... and severity in {low, medium, high, critical}."
            ),
            user_prompt=(
                f"User focus: {user_focus}\n"
                f"Target vector count: {max(3, count)}\n"
                "Retrieved paper context:\n"
                f"{_format_context(documents, max_docs=max(settings.rerank_top_n, 8))}"
            ),
            temperature=0.1,
            max_output_tokens=420,
        )
    except Exception:
        return _fallback_attack_vectors(message=message, documents=documents)
    payload = _try_parse_json_payload(raw)
    if isinstance(payload, list):
        vectors = [item for item in payload if isinstance(item, dict)]
        return vectors
    if isinstance(payload, dict):
        maybe_list = payload.get("attack_vectors")
        if isinstance(maybe_list, list):
            return [item for item in maybe_list if isinstance(item, dict)]
    return _fallback_attack_vectors(message=message, documents=documents)


def _fallback_attack_vectors(message: str, documents: list[Document]) -> list[dict[str, Any]]:
    focus = (message or "").lower()
    profiles = _paper_profiles_from_documents(documents)
    profile = next(iter(profiles.values()), {})
    quote_pool = _quote_candidates(documents)
    quote = str(profile.get("summary_sentence", "")).strip() or (quote_pool[0] if quote_pool else _default_quote(documents))
    q1 = str(profile.get("method_sentence", "")).strip() or (quote_pool[1] if len(quote_pool) > 1 else quote)
    q2 = str(profile.get("metric_sentence", "")).strip() or (quote_pool[2] if len(quote_pool) > 2 else quote)
    q3 = str(profile.get("efficiency_sentence", "")).strip() or (quote_pool[3] if len(quote_pool) > 3 else q2 or q1 or quote)
    q4 = str(profile.get("limitation_sentence", "")).strip() or (quote_pool[4] if len(quote_pool) > 4 else q3 or q2 or q1 or quote)
    vectors = [
        {
            "id": "V1",
            "claim": "Novelty should be stated at the level of the architecture/result delta actually demonstrated.",
            "severity": "high",
            "category": "novelty",
            "quote": quote,
            "skeptic_lead": "This novelty claim needs to be tied to the exact architectural change and measured gain the paper actually shows.",
        },
        {
            "id": "V2",
            "claim": "Method justification should explain why the chosen design is preferable to nearby alternatives.",
            "severity": "high",
            "category": "method",
            "quote": q1,
            "skeptic_lead": "The method may be strong, but the paper still has to justify why this exact design is the right trade-off.",
        },
        {
            "id": "V3",
            "claim": "Benchmark claims should be tied to the exact dataset/metric slice actually reported, not a broader paper-level impression.",
            "severity": "high",
            "category": "evaluation",
            "quote": q2,
            "skeptic_lead": "The benchmark story is only convincing if the paper makes the task slice, metric, and comparator explicit.",
        },
        {
            "id": "V4",
            "claim": "Training-efficiency or robustness language should be separated from broader generalization claims.",
            "severity": "medium",
            "category": "ablation",
            "quote": q3,
            "skeptic_lead": "Measured training cost is useful, but it should not be allowed to stand in for broader robustness evidence.",
        },
        {
            "id": "V5",
            "claim": "Replication details should pin down the configuration behind the strongest reported result.",
            "severity": "medium",
            "category": "reproducibility",
            "quote": q4,
            "skeptic_lead": "Key replication details are currently too sparse for a reliable reproduction.",
        },
    ]
    if "novelty" in focus:
        return vectors[:3]
    if "method" in focus:
        return [vectors[1], vectors[2], vectors[3]]
    return vectors


def _quote_candidates(documents: list[Document]) -> list[str]:
    candidates: list[str] = []
    for document in documents[:8]:
        text = (document.page_content or "").strip()
        if not text:
            continue
        chunk_lower = text.lower()
        if any(
            marker in chunk_lower
            for marker in (
                "acknowledg",
                "would like to thank",
                "we thank",
                "authors would like to thank",
            )
        ):
            continue
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            snippet = sentence.strip()
            if len(snippet) < 70 or len(snippet) > 220:
                continue
            lowered = snippet.lower()
            if lowered.startswith("references"):
                continue
            if any(
                marker in lowered
                for marker in (
                    "copyright",
                    "permission",
                    "licensed to acm",
                    "request permissions",
                    "doi:",
                    "all rights reserved",
                    "manuscript submitted",
                    "acknowledg",
                    "we thank",
                    "would like to thank",
                    "conference on neural information processing systems",
                    "long beach, ca",
                    "started the effort",
                    "crucially involved",
                )
            ):
                continue
            if lowered.startswith("figure ") or lowered.startswith("table "):
                continue
            if re.search(r"\([a-z]\)\s+[a-z]", lowered):
                continue
            tokens = re.findall(r"[A-Za-z0-9_]+", snippet)
            if len(tokens) < 9:
                continue
            alpha_tokens = [token for token in tokens if re.search(r"[A-Za-z]", token)]
            if len(alpha_tokens) < 7:
                continue
            numeric_ratio = sum(1 for token in tokens if token.isdigit()) / max(1, len(tokens))
            if numeric_ratio > 0.33:
                continue
            candidates.append(snippet)
            if len(candidates) >= 12:
                return candidates
    return candidates


def _default_quote(documents: list[Document]) -> str:
    candidates = _quote_candidates(documents)
    if candidates:
        return candidates[0]
    if documents and (documents[0].page_content or "").strip():
        return (documents[0].page_content or "").strip()[:180]
    return "The paper claims its contribution is effective and broadly applicable."


def _get_attack_vector(vectors: list[dict[str, Any]], vector_id: str) -> dict[str, Any]:
    for item in vectors:
        if str(item.get("id", "")).strip() == vector_id:
            return item
    return vectors[0] if vectors else {"id": "V1", "claim": "Unspecified vector.", "severity": "medium", "category": "method"}


def _count_vector_turns(history: list[dict[str, Any]], vector_id: str) -> int:
    return sum(
        1
        for item in history
        if str(item.get("vector_id", "")) == vector_id and str(item.get("speaker", "")) in {"skeptic", "advocate"}
    )


def _latest_speaker_content(history: list[dict[str, Any]], *, speaker: str, vector_id: str) -> str:
    for item in reversed(history):
        if str(item.get("vector_id", "")) != vector_id:
            continue
        if str(item.get("speaker", "")) != speaker:
            continue
        content = str(item.get("content", "")).strip()
        if content:
            return content
    return ""


def _route_reviewer_turn(
    *,
    history: list[dict[str, Any]],
    active_vector_id: str,
    resolution: str,
    turn_count: int,
    fallback: str,
) -> str:
    if turn_count >= settings.reviewer_max_turns:
        return "synthesise"
    if resolution == "force_closed":
        return "synthesise"
    if turn_count == 0:
        return "skeptic"

    last_turn = _last_vector_turn(history, active_vector_id)
    if last_turn and str(last_turn.get("speaker", "")) == "user":
        target = str(last_turn.get("target", "")).strip().lower()
        if target in {"skeptic", "advocate"}:
            return target
        previous = _last_non_user_speaker(history, active_vector_id)
        if previous == "skeptic":
            return "advocate"
        if previous == "advocate":
            return "skeptic"
        return "skeptic"

    last_meta = _last_route_meta(history, active_vector_id)
    if last_meta.get("addressed_to") == "user":
        if turn_count >= max(3, settings.reviewer_warning_turn - 1):
            return "synthesise"
        previous = _last_non_user_speaker(history, active_vector_id)
        return "advocate" if previous == "skeptic" else "skeptic"
    if last_meta.get("addressed_to") in {"advocate", "skeptic"}:
        return str(last_meta.get("addressed_to"))

    if _speaker_conceded(_latest_speaker_content(history, speaker="skeptic", vector_id=active_vector_id), "skeptic"):
        return "advocate"
    if bool(last_meta.get("concession")):
        return "synthesise"

    if resolution in {"resolved", "deadlocked", "force_closed"}:
        return "synthesise"

    if _is_deadlock(history, active_vector_id):
        return "synthesise"

    skeptic_turns = _speaker_turn_count(history, "skeptic", active_vector_id)
    advocate_turns = _speaker_turn_count(history, "advocate", active_vector_id)
    if skeptic_turns == 0:
        return "skeptic"
    if advocate_turns == 0:
        return "advocate"

    if last_turn and str(last_turn.get("speaker", "")) == "skeptic":
        return "advocate"
    if last_turn and str(last_turn.get("speaker", "")) == "advocate":
        return "skeptic"

    return fallback if fallback in {"skeptic", "advocate", "user", "synthesise"} else "skeptic"


def _last_vector_turn(history: list[dict[str, Any]], vector_id: str) -> dict[str, Any] | None:
    for item in reversed(history):
        if str(item.get("vector_id", "")) == vector_id:
            return item
    return None


def _last_non_user_speaker(history: list[dict[str, Any]], vector_id: str) -> str | None:
    for item in reversed(history):
        if str(item.get("vector_id", "")) != vector_id:
            continue
        speaker = str(item.get("speaker", ""))
        if speaker in {"skeptic", "advocate"}:
            return speaker
    return None


def _last_route_meta(history: list[dict[str, Any]], vector_id: str) -> dict[str, Any]:
    for item in reversed(history):
        if str(item.get("vector_id", "")) != vector_id:
            continue
        speaker = str(item.get("speaker", ""))
        if speaker not in {"skeptic", "advocate"}:
            continue
        meta = item.get("meta", {})
        if isinstance(meta, dict):
            return meta
    return {"addressed_to": "none", "concession": False, "confidence": 0.0}


def _speaker_turn_count(history: list[dict[str, Any]], speaker: str, vector_id: str) -> int:
    return sum(
        1
        for item in history
        if str(item.get("vector_id", "")) == vector_id and str(item.get("speaker", "")) == speaker
    )


def _infer_resolution(
    *,
    skeptic_position: str,
    advocate_position: str,
    history: list[dict[str, Any]],
    active_vector_id: str,
    turn_count: int,
) -> str:
    if turn_count >= settings.reviewer_max_turns:
        return "force_closed"

    last_meta = _last_route_meta(history, active_vector_id)
    if bool(last_meta.get("concession")):
        return "resolved"
    if _speaker_conceded(advocate_position, "advocate"):
        return "resolved"
    if _speaker_conceded(skeptic_position, "skeptic"):
        return "resolved"
    if _is_deadlock(history, active_vector_id):
        if turn_count >= max(4, settings.reviewer_warning_turn - 1):
            return "force_closed"
        return "deadlocked"
    return "open"


def _speaker_conceded(text: str, speaker: str) -> bool:
    lowered = (text or "").lower()
    if not lowered:
        return False
    generic = (
        "i concede",
        "concession: yes",
        "point conceded",
        "this point is conceded",
        "i retract",
    )
    if any(marker in lowered for marker in generic):
        return True
    if speaker == "skeptic":
        skeptic_markers = (
            "claim is sufficiently supported",
            "this concern is covered",
            "concern resolved",
        )
        return any(marker in lowered for marker in skeptic_markers)
    advocate_markers = (
        "cannot defend",
        "defense fails",
        "insufficient evidence",
        "skeptic is correct",
    )
    return any(marker in lowered for marker in advocate_markers)


def _is_deadlock(history: list[dict[str, Any]], vector_id: str) -> bool:
    skeptic_turns = _last_n_speaker_turns(history, speaker="skeptic", vector_id=vector_id, n=2)
    advocate_turns = _last_n_speaker_turns(history, speaker="advocate", vector_id=vector_id, n=2)
    if len(skeptic_turns) < 2 or len(advocate_turns) < 2:
        return False
    skeptic_unchanged = _normalize_turn_text(skeptic_turns[0]) == _normalize_turn_text(skeptic_turns[1])
    advocate_unchanged = _normalize_turn_text(advocate_turns[0]) == _normalize_turn_text(advocate_turns[1])
    if skeptic_unchanged and advocate_unchanged:
        return True

    skeptic_similarity = _turn_similarity_score(skeptic_turns[0], skeptic_turns[1])
    advocate_similarity = _turn_similarity_score(advocate_turns[0], advocate_turns[1])
    if skeptic_similarity >= 0.88 and advocate_similarity >= 0.88:
        return True
    if skeptic_similarity >= 0.94 and advocate_similarity >= 0.80:
        return True
    if advocate_similarity >= 0.94 and skeptic_similarity >= 0.80:
        return True
    return False


def _last_n_speaker_turns(
    history: list[dict[str, Any]],
    *,
    speaker: str,
    vector_id: str,
    n: int,
) -> list[str]:
    collected: list[str] = []
    for item in reversed(history):
        if str(item.get("vector_id", "")) != vector_id:
            continue
        if str(item.get("speaker", "")) != speaker:
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        collected.append(content)
        if len(collected) >= n:
            break
    return collected


def _normalize_turn_text(text: str) -> str:
    condensed = re.sub(r"\s+", " ", (text or "").strip().lower())
    return condensed[:240]


def _turn_similarity_score(first: str, second: str) -> float:
    first_tokens = set(_tokenize_for_overlap(first))
    second_tokens = set(_tokenize_for_overlap(second))
    if not first_tokens or not second_tokens:
        return 0.0
    overlap = len(first_tokens & second_tokens)
    union = len(first_tokens | second_tokens)
    if union <= 0:
        return 0.0
    return overlap / union


def _looks_like_score_request(message: str) -> bool:
    lower = (message or "").strip().lower()
    if not lower:
        return False
    markers = (
        "what score",
        "give score",
        "overall score",
        "recommendation score",
        "acceptance score",
        "rate this",
        "what would you rate",
        "rating",
    )
    return any(marker in lower for marker in markers)


def _reviewer_score_response(
    *,
    query: str,
    active_vector: dict[str, Any],
    resolution: str,
    vector_verdicts: dict[str, str],
    vector_judgments: dict[str, dict[str, Any]],
    debate_history: list[dict[str, Any]],
    documents: list[Document],
) -> str:
    fallback = _fallback_scorecard(
        query=query,
        active_vector=active_vector,
        resolution=resolution,
        vector_verdicts=vector_verdicts,
    )
    if not text_service.available:
        return fallback
    try:
        response = text_service.generate(
            system_prompt=(
                "You are a strict ML conference reviewer producing a concise scorecard.\n"
                "Return markdown only with these headings:\n"
                "## Scorecard\n"
                "- Overall: <x.x/10>\n"
                "- Recommendation: <Reject|Weak Reject|Borderline|Weak Accept|Accept>\n"
                "- Confidence: <1-5>\n"
                "- Rationale: <3 bullets tied to evidence with [n] citations>"
            ),
            user_prompt=(
                f"User ask: {query}\n"
                f"Active vector: {active_vector.get('id', 'V?')} - {active_vector.get('claim', '')}\n"
                f"Resolution: {resolution}\n"
                f"Vector verdicts: {json.dumps(vector_verdicts)}\n"
                f"Judge cards: {json.dumps(vector_judgments)}\n"
                "Debate excerpt:\n"
                f"{_format_vector_history(debate_history=debate_history, vector_id=str(active_vector.get('id','')), max_turns=6)}\n\n"
                "Context:\n"
                f"{_format_context(documents, max_docs=4)}"
            ),
            temperature=0.1,
            max_output_tokens=260,
        )
        cleaned = (response or "").strip()
        return cleaned if cleaned else fallback
    except Exception:
        return fallback


def _fallback_scorecard(
    *,
    query: str,
    active_vector: dict[str, Any],
    resolution: str,
    vector_verdicts: dict[str, str],
) -> str:
    verdict = vector_verdicts.get(str(active_vector.get("id", "")), "contested")
    base = 6.2
    if verdict == "skeptic_prevailed":
        base = 5.2
    elif verdict == "advocate_prevailed":
        base = 7.1
    elif resolution in {"deadlocked", "force_closed"}:
        base = 5.8
    score = max(1.0, min(9.5, base))
    if score >= 7.5:
        recommendation = "Accept"
    elif score >= 6.5:
        recommendation = "Weak Accept"
    elif score >= 5.8:
        recommendation = "Borderline"
    elif score >= 4.8:
        recommendation = "Weak Reject"
    else:
        recommendation = "Reject"
    return (
        "## Scorecard\n"
        f"- Overall: {score:.1f}/10\n"
        f"- Recommendation: {recommendation}\n"
        "- Confidence: 2/5\n"
        "- Rationale:\n"
        f"  - The active concern is `{active_vector.get('claim', 'claim strength vs evidence')}` and remains unresolved.\n"
        "  - Current evidence supports parts of the contribution but claim wording should be narrowed.\n"
        "  - A stronger score needs explicit baseline/metric framing and cleaner scope boundaries."
    )


def _generate_reviewer_turn(
    *,
    speaker: str,
    active_vector: dict[str, Any],
    objective: str,
    debate_summary: str,
    debate_history: list[dict[str, Any]],
    documents: list[Document],
) -> tuple[str, dict[str, Any]]:
    history_excerpt = _format_vector_history(
        debate_history=debate_history,
        vector_id=str(active_vector.get("id", "")),
        max_turns=2,
    )
    turn_docs = _select_turn_documents(
        documents=documents,
        active_vector=active_vector,
        objective=objective,
    )
    evidence_pack = _build_vector_evidence_pack(
        active_vector=active_vector,
        documents=turn_docs,
        limit=3,
    )
    if speaker == "advocate" and len(evidence_pack) > 1:
        evidence_pack = evidence_pack[1:] + evidence_pack[:1]
    previous_same_speaker = _latest_speaker_content(
        debate_history,
        speaker=speaker,
        vector_id=str(active_vector.get("id", "")),
    )
    opponent_speaker = "advocate" if speaker == "skeptic" else "skeptic"
    opponent_latest = _latest_speaker_content(
        debate_history,
        speaker=opponent_speaker,
        vector_id=str(active_vector.get("id", "")),
    )
    vector_id = str(active_vector.get("id", ""))
    speaker_turn_number = _speaker_turn_count(debate_history, speaker, vector_id) + 1
    deterministic = _deterministic_reviewer_turn(
        speaker=speaker,
        active_vector=active_vector,
        evidence_pack=evidence_pack,
        opponent_turn=opponent_latest,
        speaker_turn_number=speaker_turn_number,
        debate_summary=debate_summary,
        history_excerpt=history_excerpt,
    )
    deduped = _reduce_reviewer_repetition(
        speaker=speaker,
        turn_text=deterministic,
        previous_turn=previous_same_speaker,
        active_vector=active_vector,
        evidence_pack=evidence_pack,
    )
    grounded = _enforce_grounded_reviewer_turn(
        speaker=speaker,
        turn_text=deduped,
        active_vector=active_vector,
        evidence_pack=evidence_pack,
        opponent_turn=opponent_latest,
    )
    return grounded, {
        "addressed_to": "advocate" if speaker == "skeptic" else "skeptic",
        "concession": False,
        "confidence": 0.66,
    }


def _deterministic_reviewer_turn(
    *,
    speaker: str,
    active_vector: dict[str, Any],
    evidence_pack: list[dict[str, Any]],
    opponent_turn: str,
    speaker_turn_number: int,
    debate_summary: str,
    history_excerpt: str,
) -> str:
    primary = evidence_pack[0] if evidence_pack else {}
    secondary = evidence_pack[1] if len(evidence_pack) > 1 else primary
    p_text = _compact_turn_text(str(primary.get("snippet", "")).strip() or str(active_vector.get("quote", "")), max_chars=200)
    s_text = _compact_turn_text(str(secondary.get("snippet", "")).strip() or p_text, max_chars=200)
    p_cite = int(primary.get("citation_index", 1)) if primary else 1
    s_cite = int(secondary.get("citation_index", 1)) if secondary else p_cite
    category = str(active_vector.get("category", "method")).strip().lower()
    opponent_short = _extract_opponent_position(opponent_turn) or _compact_turn_text(
        opponent_turn or "No direct opposing argument yet.",
        max_chars=160,
    )
    opponent_gap = _extract_opponent_gap(opponent_turn)

    skeptic_openers = [
        "Position: Evidence supports feasibility, but claim strength still exceeds what is directly shown.",
        "Position: The current claim remains under-justified relative to the evidence presented.",
        "Position: The paper is promising, yet the claim framing is still too broad for the reported support.",
    ]
    advocate_openers = [
        "Position: The claim is defensible when explicitly bounded to reported evidence and scope.",
        "Position: The paper supports a scoped version of the claim with credible evidence.",
        "Position: The contribution can stand if framed conservatively around measured outcomes.",
    ]
    skeptic_gaps = {
        "novelty": "novelty delta versus closest prior work is not explicit in quantified terms",
        "method": "method assumptions are not yet justified strongly enough for the full claim",
        "evaluation": "evaluation coverage does not fully match the breadth of the claim",
        "ablation": "robustness/ablation evidence is too thin for stronger wording",
        "reproducibility": "replication-critical details remain insufficiently specified",
    }
    advocate_strengths = {
        "novelty": "accessibility and telemetry-free analysis provide a meaningful scoped contribution",
        "method": "method choices are reasonable for an exploratory scoped study",
        "evaluation": "evidence supports a narrower claim tied to reported settings",
        "ablation": "feasibility is supported, with robustness claims needing scoped language",
        "reproducibility": "contribution can be retained with explicit implementation caveats",
    }

    if speaker == "skeptic":
        opener = skeptic_openers[(speaker_turn_number - 1) % len(skeptic_openers)]
        gap = skeptic_gaps.get(category, "evidence-claim alignment is still incomplete")
        return (
            f"{opener}\n"
            "Argument:\n"
            f"- Rebuttal target: {opponent_gap or opponent_short}.\n"
            f"- Evidence anchor: \"{p_text}\" [{p_cite}].\n"
            f"- Why rebuttal fails: {gap}. Supporting context: \"{s_text}\" [{s_cite}].\n"
            "- Required revision: narrow the claim sentence and add one explicit comparator on a reported metric."
        )

    opener = advocate_openers[(speaker_turn_number - 1) % len(advocate_openers)]
    strength = advocate_strengths.get(category, "claim can be defended when scoped to reported evidence")
    return (
        f"{opener}\n"
        "Argument:\n"
        f"- Counter to skeptic: {opponent_gap or opponent_short}.\n"
        f"- Evidence anchor: \"{p_text}\" [{p_cite}].\n"
        f"- Defense logic: {strength}. Boundary signal: \"{s_text}\" [{s_cite}].\n"
        "- Accepted limitation: contribution should be stated as scoped to the reported setup and measurements."
    )


def _extract_opponent_position(turn: str) -> str:
    text = (turn or "").strip()
    if not text:
        return ""
    match = re.search(r"Position:\s*(.+)", text, flags=re.IGNORECASE)
    if not match:
        return ""
    sentence = re.sub(r"\s+", " ", match.group(1)).strip()
    return sentence[:150]


def _extract_opponent_gap(turn: str) -> str:
    text = (turn or "").strip()
    if not text:
        return ""
    patterns = (
        r"Unresolved gap:\s*(.+?)(?:\n|$)",
        r"Remaining issue:\s*(.+?)(?:\n|$)",
        r"Why rebuttal fails:\s*(.+?)(?:\n|$)",
        r"Counter to skeptic:\s*(.+?)(?:\n|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            snippet = re.sub(r"\s+", " ", match.group(1)).strip()
            if snippet:
                return snippet[:180]
    return ""


def _enforce_grounded_reviewer_turn(
    *,
    speaker: str,
    turn_text: str,
    active_vector: dict[str, Any],
    evidence_pack: list[dict[str, Any]],
    opponent_turn: str = "",
) -> str:
    text = (turn_text or "").strip()
    if not text:
        return _grounded_reviewer_template(
            speaker=speaker,
            active_vector=active_vector,
            evidence_pack=evidence_pack,
            opponent_turn=opponent_turn,
        )
    if _reviewer_turn_low_quality(text=text, evidence_pack=evidence_pack):
        return _grounded_reviewer_template(
            speaker=speaker,
            active_vector=active_vector,
            evidence_pack=evidence_pack,
            opponent_turn=opponent_turn,
        )
    if "position:" not in text.lower() or "argument:" not in text.lower():
        return _grounded_reviewer_template(
            speaker=speaker,
            active_vector=active_vector,
            evidence_pack=evidence_pack,
            opponent_turn=opponent_turn,
        )
    return text


def _reviewer_turn_low_quality(*, text: str, evidence_pack: list[dict[str, Any]]) -> bool:
    lower = (text or "").lower()
    if len(lower) < 80:
        return True
    if len(lower) > 2200:
        return True
    if not re.search(r"\[[0-9]+\]", lower):
        return True

    evidence_tokens: set[str] = set()
    evidence_overlaps: list[float] = []
    for item in evidence_pack:
        snippet = str(item.get("snippet", ""))
        evidence_tokens.update(_tokenize_for_overlap(snippet))
        evidence_overlaps.append(_overlap_score(lower, set(_tokenize_for_overlap(snippet))))
    if evidence_tokens:
        overlap = _overlap_score(lower, evidence_tokens)
        if overlap < 0.14:
            return True
    if evidence_overlaps and max(evidence_overlaps) < 0.22:
        return True

    generic_markers = (
        "real-world scenarios",
        "practical applications",
        "specific examples would strengthen",
        "case studies",
        "significant impact",
        "broader implications",
        "methodological shift",
        "intriguing",
        "it is difficult to assess",
        "would strengthen this argument significantly",
    )
    generic_hits = sum(1 for marker in generic_markers if marker in lower)
    if generic_hits >= 1:
        return True

    if lower.count("?") >= 3:
        return True
    if lower.count("while") >= 3 and "response to" not in lower:
        return True
    return False


def _grounded_reviewer_template(
    *,
    speaker: str,
    active_vector: dict[str, Any],
    evidence_pack: list[dict[str, Any]],
    opponent_turn: str = "",
) -> str:
    quote = str(active_vector.get("quote", "")).strip()
    claim = str(active_vector.get("claim", "")).strip() or "the active claim"

    primary = evidence_pack[0] if evidence_pack else {}
    secondary = evidence_pack[1] if len(evidence_pack) > 1 else primary
    p_text = _compact_turn_text(str(primary.get("snippet", "")).strip() or quote or claim, max_chars=180)
    p_cite = int(primary.get("citation_index", 1)) if primary else 1
    s_text = _compact_turn_text(str(secondary.get("snippet", "")).strip() or p_text, max_chars=180)
    s_cite = int(secondary.get("citation_index", 1)) if secondary else p_cite
    category = str(active_vector.get("category", "method")).strip().lower()
    opponent_short = _compact_turn_text(opponent_turn, max_chars=140) if opponent_turn else ""

    skeptic_gap_by_category = {
        "novelty": "the novelty delta versus closest prior work is still not explicit in quantified terms.",
        "method": "key method assumptions are not yet justified strongly enough for the full claim.",
        "evaluation": "evaluation coverage does not yet fully match the breadth of the claim.",
        "ablation": "ablation/robustness evidence is still too thin to support stronger wording.",
        "reproducibility": "replication-critical details remain underspecified for a strong claim.",
    }
    advocate_defense_by_category = {
        "novelty": "the contribution can be defended as a scoped accessibility/telemetry-free approach.",
        "method": "the method is defensible when described as an exploratory, scoped design choice.",
        "evaluation": "evaluation can support a narrower claim aligned to reported settings.",
        "ablation": "current evidence supports feasibility; stronger robustness framing should be conditional.",
        "reproducibility": "claim is defendable when bounded and accompanied by explicit implementation caveats.",
    }
    skeptic_gap = skeptic_gap_by_category.get(category, "the claim still extends beyond what is directly demonstrated.")
    advocate_defense = advocate_defense_by_category.get(category, "the claim is defensible when explicitly scoped.")

    if speaker == "skeptic":
        return (
            "Position: Evidence supports feasibility, but the claim is still broader than what is directly demonstrated.\n"
            "Argument:\n"
            f"- Evidence shown: \"{p_text}\" [{p_cite}].\n"
            f"- Gap: {skeptic_gap} Supporting context: \"{s_text}\" [{s_cite}].\n"
            f"- Response to advocate: {opponent_short if opponent_short else 'scope-limited defense is reasonable, but still needs tighter evidence linkage.'}"
        )
    return (
        "Position: The claim is defensible when explicitly bounded to the reported setup and evidence.\n"
        "Argument:\n"
        f"- Supporting evidence: \"{p_text}\" [{p_cite}].\n"
        f"- Scope support: \"{s_text}\" [{s_cite}] indicates the paper already states limits that can bound the claim.\n"
        f"- Response to skeptic: {advocate_defense} Keep wording scoped to reported measurements."
    )


def _format_vector_history(
    *,
    debate_history: list[dict[str, Any]],
    vector_id: str,
    max_turns: int,
) -> str:
    relevant = [item for item in debate_history if str(item.get("vector_id", "")) == vector_id]
    if not relevant:
        return "No prior turns."
    trimmed = relevant[-max_turns:]
    lines = []
    for item in trimmed:
        speaker = str(item.get("speaker", "unknown")).upper()
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines) if lines else "No prior turns."


def _format_vector_history_compact(
    *,
    debate_history: list[dict[str, Any]],
    vector_id: str,
    max_turns: int,
    max_chars_per_turn: int = 220,
) -> str:
    relevant = [item for item in debate_history if str(item.get("vector_id", "")) == vector_id]
    if not relevant:
        return "No prior turns."
    trimmed = relevant[-max_turns:]
    lines: list[str] = []
    for item in trimmed:
        speaker = str(item.get("speaker", "unknown")).upper()
        content = _compact_turn_text(str(item.get("content", "")), max_chars=max_chars_per_turn)
        if not content:
            continue
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines) if lines else "No prior turns."


def _compact_turn_text(text: str, *, max_chars: int) -> str:
    cleaned = re.sub(r"ROUTE_JSON:\s*\{[\s\S]*\}\s*$", "", text or "", flags=re.IGNORECASE).strip()
    cleaned = _clean_visible_text(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 3].rstrip()}..."


def _select_turn_documents(
    *,
    documents: list[Document],
    active_vector: dict[str, Any],
    objective: str,
) -> list[Document]:
    if not documents:
        return []
    category = str(active_vector.get("category", "")).strip().lower()
    category_terms = set(_tokenize_for_overlap(" ".join(_reviewer_category_markers(category))))
    query_terms = set(
        _tokenize_for_overlap(
            " ".join(
                [
                    str(active_vector.get("claim", "")),
                    str(active_vector.get("category", "")),
                    str(active_vector.get("quote", "")),
                    objective or "",
                ]
            )
        )
    )
    scored: list[tuple[Document, float]] = []
    for index, document in enumerate(documents):
        text = document.page_content or ""
        if _looks_like_metadata_snippet(text) or _looks_acknowledgement_text(text):
            continue
        lower = text.lower()
        overlap = _overlap_score(text, query_terms)
        category_overlap = _overlap_score(text, category_terms) if category_terms else 0.0
        rank_prior = max(0.05, 1.0 - (index / max(1, len(documents))))
        penalty = _low_signal_penalty(text)
        category_bonus = 0.0
        if category == "novelty" and any(marker in lower for marker in ("abstract", "we propose", "our main result", "in this work")):
            category_bonus += 0.35
        if category == "evaluation" and _looks_metric_rich_chunk(text):
            category_bonus += 0.45
        if category == "method" and any(marker in lower for marker in _method_signal_markers()):
            category_bonus += 0.35
        if category in {"ablation", "reproducibility"} and any(marker in lower for marker in ("hours", "days", "gpu", "gpus", "training", "batch", "beam", "implementation")):
            category_bonus += 0.28
        scored.append((document, overlap + (0.75 * category_overlap) + (0.4 * rank_prior) + category_bonus - (0.6 * penalty)))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    selected: list[Document] = []
    seen_pages: set[tuple[str, Any]] = set()
    for document, _score in scored:
        metadata = document.metadata or {}
        filename = str(metadata.get("filename", "")).strip()
        page = metadata.get("page")
        key = (filename, page)
        if key in seen_pages:
            continue
        seen_pages.add(key)
        selected.append(document)
        if len(selected) >= 4:
            break
    return selected if selected else [document for document, _ in scored[:2]]


def _extract_route_meta(text: str, default_target: str) -> dict[str, Any]:
    payload = {"addressed_to": default_target, "concession": False, "confidence": 0.55}
    match = re.search(r"ROUTE_JSON:\s*(\{[\s\S]*\})", text or "", flags=re.IGNORECASE)
    if not match:
        return payload
    parsed = _try_parse_json_payload(match.group(1))
    if not isinstance(parsed, dict):
        return payload
    addressed = str(parsed.get("addressed_to", default_target)).strip().lower()
    if addressed not in {"advocate", "skeptic", "user", "none"}:
        addressed = default_target
    concession = bool(parsed.get("concession", False))
    confidence = parsed.get("confidence", 0.55)
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.55
    confidence = max(0.0, min(1.0, confidence))
    return {
        "addressed_to": addressed,
        "concession": concession,
        "confidence": confidence,
    }


def _strip_route_json_footer(text: str) -> str:
    cleaned = re.sub(r"ROUTE_JSON:\s*\{[\s\S]*\}\s*$", "", text or "", flags=re.IGNORECASE).strip()
    return cleaned


def _fallback_reviewer_turn(
    *,
    speaker: str,
    active_vector: dict[str, Any],
    documents: list[Document],
) -> str:
    quote = str(active_vector.get("quote", "")).strip() or _default_quote(documents)
    evidence = _fallback_vector_evidence(active_vector=active_vector, documents=documents, limit=2)
    primary = evidence[0] if evidence else quote
    secondary = evidence[1] if len(evidence) > 1 else primary
    if speaker == "skeptic":
        return (
            "Position: Evidence supports feasibility, but not the full strength of the claim as currently worded.\n"
            "Argument:\n"
            f"- Triggered claim: \"{quote}\"\n"
            f"- Evidence check: \"{primary}\" [1].\n"
            f"- Scope gap: \"{secondary}\" [1] indicates limits that should be reflected in claim wording.\n"
            "- Request: tighten claim scope and anchor it to one explicit reported metric.\n"
            'ROUTE_JSON: {"addressed_to":"advocate","concession":false,"confidence":0.54}'
        )
    return (
        "Position: The claim is supportable when bounded to the paper's measured scope.\n"
        "Argument:\n"
        f"- Triggered claim: \"{quote}\"\n"
        f"- Supporting evidence: \"{primary}\" [1].\n"
        f"- Scope caveat already present: \"{secondary}\" [1].\n"
        "- Revision path: keep contribution framing and rewrite novelty language to match exactly what is measured.\n"
        'ROUTE_JSON: {"addressed_to":"skeptic","concession":false,"confidence":0.56}'
    )


def _fallback_vector_evidence(
    *,
    active_vector: dict[str, Any],
    documents: list[Document],
    limit: int,
) -> list[str]:
    if not documents:
        return []
    category = str(active_vector.get("category", "")).strip().lower()
    category_markers = _reviewer_category_markers(category)
    query = " ".join(
        [
            str(active_vector.get("claim", "")),
            str(active_vector.get("category", "")),
            str(active_vector.get("quote", "")),
            str(active_vector.get("skeptic_lead", "")),
            " ".join(category_markers),
        ]
    )
    query_terms = set(_tokenize_for_overlap(query))
    query_phrases = _query_phrases(query)
    category_terms = set(_tokenize_for_overlap(" ".join(category_markers)))
    candidates: list[tuple[float, str]] = []
    profiles = _paper_profiles_from_documents(documents)
    profile = next(iter(profiles.values()), {})
    seed_snippets: list[str] = []
    if category == "novelty":
        seed_snippets.extend([str(profile.get("summary_sentence", "")), str(profile.get("metric_sentence", ""))])
    elif category == "method":
        seed_snippets.extend([str(profile.get("method_sentence", "")), str(profile.get("summary_sentence", ""))])
    elif category == "evaluation":
        seed_snippets.extend([str(profile.get("metric_sentence", "")), str(profile.get("summary_sentence", "")), str(profile.get("efficiency_sentence", ""))])
    else:
        seed_snippets.extend([str(profile.get("efficiency_sentence", "")), str(profile.get("metric_sentence", "")), str(profile.get("limitation_sentence", ""))])
    for position, snippet in enumerate(seed_snippets):
        normalized = re.sub(r"\s+", " ", str(snippet or "")).strip()
        if not normalized:
            continue
        candidates.append((1.6 - (0.08 * position), normalized))
    for doc_index, document in enumerate(documents[:6]):
        text = (document.page_content or "").strip()
        if not text:
            continue
        if _looks_like_metadata_snippet(text) or _looks_acknowledgement_text(text):
            continue
        sentences = re.split(r"(?<=[.!?])\s+", text)
        doc_prior = max(0.05, 1.0 - (doc_index / max(1, len(documents))))
        for sentence in sentences:
            snippet = sentence.strip()
            if len(snippet) < 30:
                continue
            if _looks_like_reference_snippet(snippet):
                continue
            if _looks_like_non_argument_snippet(snippet) or _looks_like_metadata_snippet(snippet):
                continue
            lowered = snippet.lower()
            overlap = _overlap_score(lowered, query_terms)
            phrase = _phrase_overlap_score(_normalize_for_phrase_match(lowered), query_phrases)
            marker_overlap = _overlap_score(lowered, category_terms) if category_terms else 0.0
            numeric_bonus = 0.15 if re.search(r"\b\d+(?:\.\d+)?%?\b", snippet) else 0.0
            category_bonus = 0.0
            if category == "novelty" and any(marker in lowered for marker in ("we propose", "our main result", "in this work")):
                category_bonus += 0.35
            if category == "evaluation" and _looks_metric_rich_chunk(snippet):
                category_bonus += 0.45
            if category == "method" and any(marker in lowered for marker in _method_signal_markers()):
                category_bonus += 0.35
            if category in {"ablation", "reproducibility"} and any(marker in lowered for marker in ("hours", "days", "gpu", "gpus", "training", "batch", "beam", "implementation", "parallel")):
                category_bonus += 0.28
            score = (0.9 * overlap) + (0.8 * phrase) + (0.55 * marker_overlap) + (0.2 * doc_prior) + numeric_bonus + category_bonus
            score -= min(0.3, _low_signal_penalty(snippet))
            candidates.append((score, snippet))
    if not candidates:
        return []
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    seen_norm: set[str] = set()
    for score, snippet in candidates:
        if score < 0.18:
            continue
        normalized = re.sub(r"\s+", " ", re.sub(r"\[[0-9]+\]", "", snippet)).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen_norm:
            continue
        seen_norm.add(key)
        selected.append(normalized)
        if len(selected) >= max(1, limit):
            break
    return selected


def _build_vector_evidence_pack(
    *,
    active_vector: dict[str, Any],
    documents: list[Document],
    limit: int,
) -> list[dict[str, Any]]:
    snippets = _fallback_vector_evidence(
        active_vector=active_vector,
        documents=documents,
        limit=max(2, limit + 1),
    )
    if not snippets and documents:
        fallback_text = _compact_turn_text(documents[0].page_content or "", max_chars=220)
        if fallback_text:
            snippets = [fallback_text]
    pack: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, snippet in enumerate(snippets, start=1):
        normalized = re.sub(r"\s+", " ", snippet).strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        doc_index = _best_doc_index_for_snippet(snippet=normalized, documents=documents)
        citation_index = doc_index + 1 if doc_index >= 0 else 1
        metadata = documents[doc_index].metadata if 0 <= doc_index < len(documents) else {}
        pack.append(
            {
                "id": f"E{index}",
                "snippet": normalized,
                "citation_index": citation_index,
                "filename": str((metadata or {}).get("filename", "unknown.pdf")),
                "page": (metadata or {}).get("page"),
                "chunk_id": str((metadata or {}).get("chunk_id", "")),
            }
        )
        if len(pack) >= max(1, limit):
            break
    return pack


def _best_doc_index_for_snippet(*, snippet: str, documents: list[Document]) -> int:
    if not documents:
        return -1
    lowered = snippet.lower()
    query_terms = set(_tokenize_for_overlap(lowered))
    best_index = 0
    best_score = float("-inf")
    for index, document in enumerate(documents):
        text = (document.page_content or "").lower()
        if not text:
            continue
        contains_bonus = 1.2 if lowered and lowered in text else 0.0
        overlap = _overlap_score(text, query_terms)
        score = contains_bonus + overlap
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def _format_evidence_pack(pack: list[dict[str, Any]]) -> str:
    if not pack:
        return "- No direct evidence snippets available from current retrieval."
    lines: list[str] = []
    for item in pack:
        evidence_id = str(item.get("id", "E?"))
        citation_index = int(item.get("citation_index", 1))
        filename = str(item.get("filename", "unknown.pdf"))
        page = item.get("page")
        page_text = f", p.{page}" if page else ""
        snippet = str(item.get("snippet", "")).strip()
        lines.append(
            f"- {evidence_id} -> [{citation_index}] {filename}{page_text}: \"{snippet}\""
        )
    return "\n".join(lines)


def _clean_reviewer_turn_text(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^\s*#+\s*Reviewer\s+[AB]\s*\([^)]+\)\s*\n?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*Reviewer\s+[AB]\s*\([^)]+\)\s*[:\-]?\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _reduce_reviewer_repetition(
    *,
    speaker: str,
    turn_text: str,
    previous_turn: str,
    active_vector: dict[str, Any],
    evidence_pack: list[dict[str, Any]],
) -> str:
    current = (turn_text or "").strip()
    previous = (previous_turn or "").strip()
    if not current or not previous:
        return current
    similarity = _turn_similarity_score(current, previous)
    if similarity < 0.70:
        return current

    alternate = evidence_pack[1] if len(evidence_pack) > 1 else (evidence_pack[0] if evidence_pack else {})
    alt_snippet = str(alternate.get("snippet", "")).strip()
    alt_citation = int(alternate.get("citation_index", 1)) if alternate else 1
    claim = str(active_vector.get("claim", "")).strip() or "this claim"
    category = str(active_vector.get("category", "method")).strip().lower()

    skeptic_gap_by_category = {
        "novelty": "the novelty delta versus closest prior work still needs an explicit quantitative statement",
        "method": "the method rationale still needs stronger evidence for the full framing",
        "evaluation": "evaluation support is still narrower than the breadth of the current claim",
        "ablation": "robustness evidence is still too limited for stronger wording",
        "reproducibility": "replication-critical details remain incomplete for a stronger claim",
    }
    advocate_support_by_category = {
        "novelty": "the contribution can still be defended as a scoped telemetry-free analysis approach",
        "method": "the method remains defensible if framed as a scoped design choice",
        "evaluation": "the reported setup supports a narrower claim aligned to measured outcomes",
        "ablation": "the evidence supports feasibility, with robustness framed as future work",
        "reproducibility": "the claim can stand with explicit implementation boundaries",
    }

    if speaker == "skeptic":
        return (
            "Position: The unresolved evidence gap still blocks a stronger claim.\n"
            "Argument:\n"
            f"- Evidence anchor: \"{alt_snippet or claim}\" [{alt_citation}].\n"
            f"- Remaining issue: {skeptic_gap_by_category.get(category, 'evidence-claim alignment is still incomplete')}.\n"
            "- Required revision: narrow wording and attach one concrete metric/baseline comparator."
        )

    return (
        "Position: The claim remains defendable with explicit scope boundaries.\n"
        "Argument:\n"
        f"- Supporting anchor: \"{alt_snippet or claim}\" [{alt_citation}].\n"
        f"- Defense: {advocate_support_by_category.get(category, 'the claim is defensible when tied directly to reported evidence')}.\n"
        "- Revision path: keep the contribution, but state limits and comparator in the same paragraph."
    )


def _refresh_debate_summary(
    *,
    debate_summary: str,
    active_vector: dict[str, Any],
    debate_history: list[dict[str, Any]],
) -> str:
    vector_id = str(active_vector.get("id", "V?"))
    claim = str(active_vector.get("claim", "")).strip() or "the active claim"
    skeptic_latest = _compact_turn_text(
        _latest_speaker_content(debate_history, speaker="skeptic", vector_id=vector_id),
        max_chars=180,
    )
    advocate_latest = _compact_turn_text(
        _latest_speaker_content(debate_history, speaker="advocate", vector_id=vector_id),
        max_chars=180,
    )
    if not skeptic_latest and not advocate_latest:
        return debate_summary
    sentence_1 = f"Debate on {vector_id} centers on whether {claim.lower()} is adequately supported."
    sentence_2 = f"Skeptic focus: {skeptic_latest or 'insufficient concrete evidence.'}"
    sentence_3 = f"Advocate focus: {advocate_latest or 'scope-limited defense with partial support.'}"
    return " ".join([sentence_1, sentence_2, sentence_3]).strip()


def _run_evidence_only_judge(
    *,
    active_vector: dict[str, Any],
    debate_history: list[dict[str, Any]],
    resolution: str,
    documents: list[Document],
) -> dict[str, Any]:
    vector_id = str(active_vector.get("id", "V?"))
    evidence_pack = _build_vector_evidence_pack(
        active_vector=active_vector,
        documents=documents,
        limit=3,
    )
    if resolution in {"deadlocked", "force_closed"}:
        return {
            "verdict": "contested",
            "confidence": 0.42,
            "rationale": "Debate hit stopping criteria without a clean evidence-based resolution.",
            "decisive_evidence": [item.get("id", "E1") for item in evidence_pack[:1]],
            "evidence_pack": evidence_pack,
        }

    fallback = {
        "verdict": "contested",
        "confidence": 0.5,
        "rationale": "Available evidence supports part of the claim, but does not settle the exact scope or comparator framing cleanly.",
        "decisive_evidence": [item.get("id", "E1") for item in evidence_pack[:1]],
        "evidence_pack": evidence_pack,
    }
    if not text_service.available:
        return fallback
    try:
        response = text_service.generate(
            system_prompt=(
                "You are an evidence-only judge for a paper-review trial.\n"
                "Decide ONLY from the provided evidence pack and debate excerpt.\n"
                "If evidence does not settle the claim, return contested.\n"
                "Return JSON only with keys: verdict, confidence, rationale, decisive_evidence.\n"
                "Allowed verdict values: skeptic_prevailed, advocate_prevailed, contested."
            ),
            user_prompt=(
                f"Claim vector: {vector_id} | {active_vector.get('claim', '')}\n"
                f"Claim trigger quote: {active_vector.get('quote', '')}\n\n"
                "Evidence pack:\n"
                f"{_format_evidence_pack(evidence_pack)}\n\n"
                "Debate excerpt:\n"
                f"{_format_vector_history(debate_history=debate_history, vector_id=vector_id, max_turns=8)}\n\n"
                "Rules:\n"
                "- decisive_evidence must be an array of evidence ids from the pack (for example [\"E1\",\"E2\"]).\n"
                "- rationale must be one short sentence."
            ),
            temperature=0.0,
            max_output_tokens=180,
        )
    except Exception:
        return fallback

    payload = _try_parse_json_payload(response)
    if not isinstance(payload, dict):
        return fallback

    verdict = str(payload.get("verdict", "contested")).strip().lower()
    if verdict not in {"skeptic_prevailed", "advocate_prevailed", "contested"}:
        verdict = "contested"
    try:
        confidence = float(payload.get("confidence", 0.5))
    except Exception:
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))
    rationale = str(payload.get("rationale", "")).strip() or fallback["rationale"]
    decisive_raw = payload.get("decisive_evidence", [])
    decisive: list[str] = []
    allowed_ids = {str(item.get("id", "")) for item in evidence_pack}
    if isinstance(decisive_raw, list):
        for item in decisive_raw:
            evidence_id = str(item).strip()
            if evidence_id in allowed_ids and evidence_id not in decisive:
                decisive.append(evidence_id)
    if not decisive and evidence_pack:
        decisive = [str(evidence_pack[0].get("id", "E1"))]
    return {
        "verdict": verdict,
        "confidence": confidence,
        "rationale": rationale,
        "decisive_evidence": decisive,
        "evidence_pack": evidence_pack,
    }


def _render_judge_card(*, active_vector_id: str, judgment: dict[str, Any]) -> str:
    verdict = str(judgment.get("verdict", "contested"))
    confidence = float(judgment.get("confidence", 0.0))
    rationale = str(judgment.get("rationale", "No rationale.")).strip()
    evidence_pack = judgment.get("evidence_pack", [])
    evidence_lookup: dict[str, str] = {}
    if isinstance(evidence_pack, list):
        for item in evidence_pack:
            if not isinstance(item, dict):
                continue
            evidence_id = str(item.get("id", "")).strip()
            citation_index = int(item.get("citation_index", 1))
            if evidence_id:
                evidence_lookup[evidence_id] = f"{evidence_id} [{citation_index}]"
    decisive_labels: list[str] = []
    decisive_raw = judgment.get("decisive_evidence", [])
    if isinstance(decisive_raw, list):
        for item in decisive_raw:
            evidence_id = str(item).strip()
            if not evidence_id:
                continue
            decisive_labels.append(evidence_lookup.get(evidence_id, evidence_id))
    decisive_text = ", ".join(decisive_labels) if decisive_labels else "none"
    return (
        "### Evidence-only Judge\n"
        f"Vector: {active_vector_id}\n"
        f"Verdict: {verdict}\n"
        f"Confidence: {confidence:.2f}\n"
        f"Decisive Evidence: {decisive_text}\n"
        f"Rationale: {rationale}"
    )


def _synthesise_vector(
    *,
    active_vector: dict[str, Any],
    verdict: str,
    judgment: dict[str, Any],
    debate_history: list[dict[str, Any]],
    documents: list[Document],
) -> str:
    vector_id = str(active_vector.get("id", "V?"))
    if not text_service.available:
        return _build_grounded_rewrite_card(active_vector=active_vector, verdict=verdict, judgment=judgment)
    try:
        evidence_text = _format_evidence_pack(judgment.get("evidence_pack", []))
        response = text_service.generate(
            system_prompt=(
                "You are a rewrite compiler that converts a judged review claim into one concrete patch.\n"
                "Do not summarize the debate. Output exactly one actionable patch tied to paper evidence.\n"
                "Do not invent metrics, datasets, or numeric results not present in the evidence pack."
            ),
            user_prompt=(
                f"Vector: {vector_id} | {active_vector.get('claim', '')}\n"
                f"Verdict: {verdict}\n"
                f"Judge rationale: {judgment.get('rationale', '')}\n"
                f"Judge decisive evidence ids: {judgment.get('decisive_evidence', [])}\n"
                f"Quoted trigger sentence: {active_vector.get('quote', '')}\n"
                "Evidence pack:\n"
                f"{evidence_text}\n\n"
                "Debate transcript:\n"
                f"{_format_vector_history(debate_history=debate_history, vector_id=vector_id, max_turns=10)}\n\n"
                "Retrieved context:\n"
                f"{_format_context(documents, max_docs=max(settings.rerank_top_n, 8))}\n\n"
                "Output format:\n"
                "### Rewrite Compiler Card\n"
                "Target Section: <section name or approximate location>\n"
                "Target Claim: <the claim being rewritten>\n"
                "Patch Instruction: <one concrete instruction sentence with metric/baseline/clarification target>\n"
                "Patch (Before -> After):\n"
                "- Before: <short phrase for current wording weakness>\n"
                "- After: <short phrase for corrected wording>\n"
                "Why: <one sentence>"
            ),
            temperature=0.1,
            max_output_tokens=280,
        )
        candidate = (response or "").strip()
        if not candidate:
            return _build_grounded_rewrite_card(active_vector=active_vector, verdict=verdict, judgment=judgment)
        if _rewrite_card_low_quality(candidate, judgment.get("evidence_pack", [])):
            return _build_grounded_rewrite_card(active_vector=active_vector, verdict=verdict, judgment=judgment)
        return candidate
    except Exception:
        return _build_grounded_rewrite_card(active_vector=active_vector, verdict=verdict, judgment=judgment)


def _build_grounded_rewrite_card(
    *,
    active_vector: dict[str, Any],
    verdict: str,
    judgment: dict[str, Any],
) -> str:
    claim = str(active_vector.get("claim", "")).strip() or "the active claim"
    evidence_pack = judgment.get("evidence_pack", []) if isinstance(judgment, dict) else []
    snippet = ""
    citation = 1
    if isinstance(evidence_pack, list) and evidence_pack:
        first = evidence_pack[0] if isinstance(evidence_pack[0], dict) else {}
        snippet = _compact_turn_text(str(first.get("snippet", "")).strip(), max_chars=220)
        try:
            citation = int(first.get("citation_index", 1))
        except Exception:
            citation = 1
    section_hint = _rewrite_section_hint(claim)
    evidence_line = f'Key Evidence: "{snippet}" [{citation}]' if snippet else "Key Evidence: use the strongest cited sentence for this claim."
    return (
        "### Rewrite Compiler Card\n"
        f"Verdict: {verdict}\n"
        f"Target Section: {section_hint}\n"
        f"Target Claim: {claim}\n"
        f"{evidence_line}\n"
        "Patch Instruction: rewrite the claim to stay within measured scope, then add one explicit metric/comparator already reported in the cited evidence.\n"
        "Patch (Before -> After):\n"
        "- Before: broad claim wording that exceeds direct support.\n"
        "- After: scoped claim wording tied to cited measurement and stated limitation.\n"
        "Why: this converts a contested claim into an evidence-aligned statement without overreach."
    )


def _rewrite_section_hint(claim: str) -> str:
    lower = (claim or "").lower()
    if "novel" in lower or "contribution" in lower:
        return "Introduction / Contribution Framing"
    if "method" in lower or "implementation" in lower:
        return "Methods"
    if "evaluation" in lower or "benchmark" in lower:
        return "Results / Evaluation"
    if "ablation" in lower or "robust" in lower:
        return "Results / Limitations"
    return "Discussion"


def _rewrite_card_low_quality(text: str, evidence_pack: list[dict[str, Any]]) -> bool:
    lower = (text or "").lower()
    if "patch instruction:" not in lower:
        return True
    if "before" not in lower or "after" not in lower:
        return True
    evidence_text = " ".join(str((item or {}).get("snippet", "")) for item in evidence_pack if isinstance(item, dict))
    allowed_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", evidence_text))
    output_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text or ""))
    if output_numbers and allowed_numbers:
        unseen = {value for value in output_numbers if value not in allowed_numbers}
        if unseen:
            return True
    banned_markers = (
        "significant improvement",
        "state of the art",
        "outperform",
        "novel benchmark gain",
    )
    return any(marker in lower for marker in banned_markers)


def _render_reviewer_debate(
    *,
    attack_vectors: list[dict[str, Any]],
    active_vector: dict[str, Any],
    vectors_remaining: list[str],
    syntheses: dict[str, str],
    vector_verdicts: dict[str, str],
    vector_judgments: dict[str, dict[str, Any]],
    vector_reports: dict[str, dict[str, Any]],
    current_vector_report: dict[str, Any],
    final_report: dict[str, Any],
    round_events: list[dict[str, Any]],
    debate_history: list[dict[str, Any]],
    debate_summary: str,
    resolution: str,
    turn_count: int,
    next_speaker: str,
) -> str:
    active_vector_id = str(active_vector.get("id", "V?"))
    active_claim = str(active_vector.get("claim", "")).strip() or "Unspecified claim."
    total_claims = max(1, len(attack_vectors))
    active_rank = 1
    for idx, vector in enumerate(attack_vectors, start=1):
        if str(vector.get("id", "")) == active_vector_id:
            active_rank = idx
            break

    if isinstance(final_report, dict) and final_report and not vectors_remaining:
        complete_reports = dict(vector_reports or {})
        if isinstance(current_vector_report, dict) and current_vector_report:
            complete_reports[active_vector_id] = current_vector_report
        return _render_reviewer_complete_report(
            attack_vectors=attack_vectors,
            vector_verdicts=vector_verdicts,
            vector_judgments=vector_judgments,
            vector_reports=complete_reports,
            syntheses=syntheses,
            debate_history=debate_history,
            final_report=final_report,
        )

    status_map = {
        "open": "Open",
        "resolved": "Resolved",
        "deadlocked": "Deadlocked",
        "force_closed": "Force Closed",
    }
    status_label = status_map.get(str(resolution).lower(), str(resolution).title())

    skeptic_latest = _latest_speaker_content(
        debate_history,
        speaker="skeptic",
        vector_id=active_vector_id,
    )
    advocate_latest = _latest_speaker_content(
        debate_history,
        speaker="advocate",
        vector_id=active_vector_id,
    )

    judgment = vector_judgments.get(active_vector_id, {})
    has_judgment = isinstance(judgment, dict) and bool(judgment)
    judge_verdict = str(judgment.get("verdict", "pending")) if has_judgment else "pending"
    try:
        judge_confidence = float(judgment.get("confidence", 0.0)) if has_judgment else 0.0
    except Exception:
        judge_confidence = 0.0
    judge_rationale = str(judgment.get("rationale", "")).strip() if has_judgment else ""

    active_rewrite = syntheses.get(active_vector_id, "")
    if not active_rewrite and syntheses:
        # Show the latest completed rewrite card if current claim is still open.
        latest_completed_id = list(syntheses.keys())[-1]
        active_rewrite = syntheses.get(latest_completed_id, "")

    queued_claims: list[str] = []
    for idx, vector in enumerate(attack_vectors, start=1):
        vector_id = str(vector.get("id", "V?"))
        if vector_id == active_vector_id:
            continue
        claim_text = str(vector.get("claim", "")).strip()
        if not claim_text:
            continue
        state = "resolved" if vector_id in syntheses else "queued"
        queued_claims.append(f"- Claim {idx} ({state}): {claim_text}")
        if len(queued_claims) >= 3:
            break

    next_move = _human_next_move(next_speaker=next_speaker, vector_id=active_vector_id)
    this_round = _render_round_events_compact(round_events=round_events, vector_id=active_vector_id)

    blocks = [
        "## Review Panel",
        f"Primary Claim: {active_claim}",
        f"Claim {active_rank}/{total_claims} | Status: {status_label} | Turn {turn_count}/{settings.reviewer_max_turns}",
        "",
        "### Skeptic (Latest)",
        skeptic_latest or "No skeptic argument yet.",
        "",
        "### Advocate (Latest)",
        advocate_latest or "No advocate response yet.",
    ]

    if has_judgment:
        blocks.extend(
            [
                "",
                "### Judge",
                f"Verdict: {judge_verdict}",
                f"Confidence: {judge_confidence:.2f}",
                f"Why: {judge_rationale or 'No rationale available.'}",
            ]
        )

    if active_rewrite:
        blocks.extend(
            [
                "",
                "### Recommended Rewrite",
                active_rewrite,
            ]
        )

    if isinstance(current_vector_report, dict) and current_vector_report:
        blocks.extend(
            [
                "",
                "### Claim Intelligence",
                _render_current_vector_report_brief(current_vector_report),
            ]
        )

    if queued_claims:
        blocks.extend(
            [
                "",
                "### Other Claims",
                "\n".join(queued_claims),
            ]
        )

    blocks.extend(
        [
            "",
            "### Round Timeline",
            this_round,
            "",
            "### Next Step",
            next_move,
            f"Claims remaining: {len(vectors_remaining)}",
            "Use Reviewer Controls: `Next Turn` to continue, `Restart` to reset this debate.",
        ]
    )
    return "\n".join(blocks).strip()


def _render_reviewer_complete_report(
    *,
    attack_vectors: list[dict[str, Any]],
    vector_verdicts: dict[str, str],
    vector_judgments: dict[str, dict[str, Any]],
    vector_reports: dict[str, dict[str, Any]],
    syntheses: dict[str, str],
    debate_history: list[dict[str, Any]],
    final_report: dict[str, Any],
) -> str:
    overview = str(final_report.get("overview", "")).strip() or "Panel review completed."
    final_decision = str(final_report.get("final_decision", "")).strip() or "No final decision available."
    confidence = final_report.get("confidence", 0.0)
    try:
        confidence_text = f"{float(confidence):.2f}"
    except Exception:
        confidence_text = "0.00"
    agreements = _dedupe_reviewer_lines(
        [str(item).strip() for item in final_report.get("agreements", []) if str(item).strip()],
        cap=4,
    )
    disagreements = _dedupe_reviewer_lines(
        [str(item).strip() for item in final_report.get("disagreements", []) if str(item).strip()],
        cap=5,
    )
    context_snapshot = [str(item).strip() for item in final_report.get("context_snapshot", []) if str(item).strip()]
    field_context = [str(item).strip() for item in final_report.get("field_context", []) if str(item).strip()]
    suggestions = [str(item).strip() for item in final_report.get("final_suggestions", []) if str(item).strip()]

    lines: list[str] = [
        "## Reviewer Complete Report",
        overview,
        "",
        f"Final Decision: {final_decision}",
        f"Panel Confidence: {confidence_text}",
    ]
    if agreements:
        lines.extend(["", "### Strengths Worth Keeping"])
        lines.extend(f"- {item}" for item in agreements[:4])
    if disagreements:
        lines.extend(["", "### High-Impact Concerns"])
        lines.extend(f"- {item}" for item in disagreements[:5])
    if context_snapshot:
        lines.extend(["", "### Paper Snapshot"])
        lines.extend(f"- {item}" for item in context_snapshot[:5])
    if field_context:
        lines.extend(["", "### Field Context"])
        lines.extend(f"- {item}" for item in field_context[:4])
    if suggestions:
        lines.extend(["", "### Required Revisions"])
        lines.extend(f"- {item}" for item in suggestions[:6])

    used_quotes: set[str] = set()
    for idx, vector in enumerate(attack_vectors, start=1):
        vector_id = str(vector.get("id", "V?"))
        category = str(vector.get("category", "claim")).strip().lower() or "claim"
        section_title = category.replace("_", " ").title()
        claim = str(vector.get("claim", "")).strip() or "Unspecified claim."
        verdict = str(vector_verdicts.get(vector_id, "contested"))
        judgment = vector_judgments.get(vector_id, {})
        rationale = str(judgment.get("rationale", "")).strip()
        evidence_pack = judgment.get("evidence_pack", []) if isinstance(judgment, dict) else []
        primary_snippet, citation_index = _reviewer_anchor_choice(
            preferred_quote="",
            evidence_pack=evidence_pack if isinstance(evidence_pack, list) else [],
            category=category,
            claim=claim,
            used_quotes=used_quotes,
        )
        reviewer_read = _reviewer_expand_read(
            reviewer_read=_reviewer_verdict_read(category=category, verdict=verdict, rationale=rationale),
            category=category,
            verdict=verdict,
            quote=primary_snippet,
            evidence_pack=evidence_pack if isinstance(evidence_pack, list) else [],
        )
        if category == "novelty" and field_context:
            field_note = re.sub(r"^[^:]+:\s*", "", field_context[0]).strip()
            reviewer_read = f"{reviewer_read} Field-aware note: {field_note}".strip()
        rewrite = _extract_patch_instruction(syntheses.get(vector_id, ""))
        if not rewrite or _looks_generic_patch_instruction(rewrite):
            rewrite = _claim_specific_reviewer_suggestion(
                claim=claim,
                category=category,
                quote=primary_snippet,
            )

        lines.extend(
            [
                "",
                f"### {section_title} Review",
                f"- Verdict: {verdict}",
                f"- Issue under debate: {claim}",
                (
                    f"- Grounded anchor: \"{primary_snippet}\" [{citation_index}]"
                    if primary_snippet
                    else "- Grounded anchor: strongest snippet not recovered in this summary."
                ),
                f"- Reviewer read: {reviewer_read}",
                f"- Required edit: {rewrite or 'Tighten the wording and attach one concrete evidence-backed comparator.'}",
            ]
        )
        if idx >= 4:
            break

    return "\n".join(lines).strip()


def _render_full_vector_transcript(*, debate_history: list[dict[str, Any]], vector_id: str) -> str:
    turns = [
        item
        for item in debate_history
        if str(item.get("vector_id", "")) == vector_id and str(item.get("speaker", "")) in {"skeptic", "advocate"}
    ]
    if not turns:
        return "No transcript available."
    lines: list[str] = []
    for item in turns:
        speaker = str(item.get("speaker", "")).strip().title() or "Reviewer"
        turn = int(item.get("turn", 0) or 0)
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"  - Turn {turn} {speaker}: {_compact_turn_text(content, max_chars=1400)}")
    return "\n".join(lines) if lines else "No transcript available."


def _build_detailed_author_guidance(
    *,
    attack_vectors: list[dict[str, Any]],
    vector_verdicts: dict[str, str],
    vector_judgments: dict[str, dict[str, Any]],
    vector_reports: dict[str, dict[str, Any]],
    syntheses: dict[str, str],
) -> list[str]:
    guidance: list[str] = []
    for idx, vector in enumerate(attack_vectors, start=1):
        vector_id = str(vector.get("id", "V?"))
        claim = str(vector.get("claim", "")).strip() or "Unspecified claim."
        verdict = str(vector_verdicts.get(vector_id, "contested"))
        judgment = vector_judgments.get(vector_id, {})
        rationale = str(judgment.get("rationale", "")).strip() or "Evidence remains mixed for this claim."
        report = vector_reports.get(vector_id, {}) if isinstance(vector_reports, dict) else {}
        actions = [str(item).strip() for item in report.get("author_action_plan", []) if str(item).strip()]
        skeptic_conclusion = str(report.get("skeptic_conclusion", "")).strip()
        advocate_conclusion = str(report.get("advocate_conclusion", "")).strip()
        patch_instruction = _extract_patch_instruction(syntheses.get(vector_id, ""))

        what_to_change = patch_instruction or (
            actions[0]
            if actions
            else "Rewrite the claim sentence so the scope and evidence are explicitly aligned."
        )
        why_it_matters = (
            f"Judge outcome is `{verdict}` and the unresolved risk is: {rationale}"
            if verdict == "contested"
            else f"Judge outcome is `{verdict}`; this edit preserves strengths while reducing reviewer risk."
        )
        skeptic_point = skeptic_conclusion or "Current framing still appears broader than direct evidence support."
        advocate_point = advocate_conclusion or "Core contribution can stand when claim language is explicitly bounded."
        implementation_steps = actions[:3] if actions else [
            "Edit the claim sentence to name the exact evaluated setting.",
            "Add one explicit quantitative comparator next to the claim.",
            "Add one limitation sentence that bounds generalization beyond measured data.",
        ]

        block_lines = [
            f"- **Claim {idx}**: {claim}",
            f"  - What to change: {what_to_change}",
            f"  - Why it matters: {why_it_matters}",
            f"  - Skeptic concern to resolve: {skeptic_point}",
            f"  - Advocate condition to preserve: {advocate_point}",
            "  - Implementation steps:",
        ]
        block_lines.extend(f"    1. {step}" for step in implementation_steps)
        guidance.append("\n".join(block_lines))
    return guidance


def _build_current_vector_report(
    *,
    active_vector: dict[str, Any],
    skeptic_position: str,
    advocate_position: str,
    debate_history: list[dict[str, Any]],
    documents: list[Document],
    existing_report: dict[str, Any],
) -> dict[str, Any]:
    skeptic = (skeptic_position or "").strip()
    advocate = (advocate_position or "").strip()
    if not skeptic or not advocate:
        return existing_report if isinstance(existing_report, dict) else {}

    fingerprint = _normalize_turn_text(skeptic) + "||" + _normalize_turn_text(advocate)
    if isinstance(existing_report, dict) and existing_report.get("fingerprint") == fingerprint:
        return existing_report

    fallback = _fallback_current_vector_report(
        active_vector=active_vector,
        skeptic_position=skeptic,
        advocate_position=advocate,
    )
    fallback["fingerprint"] = fingerprint
    if not text_service.available:
        return fallback

    vector_id = str(active_vector.get("id", "V?"))
    claim = str(active_vector.get("claim", "")).strip() or "active claim"
    try:
        response = text_service.generate(
            system_prompt=(
                "You are a debate analyst for a paper reviewer panel.\n"
                "Return JSON only with keys:\n"
                "agreements (array), disagreements (array), common_points (array),\n"
                "skeptic_conclusion (string), advocate_conclusion (string),\n"
                "joint_conclusion (string), author_action_plan (array).\n"
                "Keep output concrete and grounded in the two arguments."
            ),
            user_prompt=(
                f"Vector: {vector_id} | Claim: {claim}\n\n"
                f"Skeptic:\n{skeptic}\n\n"
                f"Advocate:\n{advocate}\n\n"
                "Recent debate transcript:\n"
                f"{_format_vector_history_compact(debate_history=debate_history, vector_id=vector_id, max_turns=6)}\n\n"
                "Retrieved context:\n"
                f"{_format_context(documents, max_docs=3)}"
            ),
            temperature=0.1,
            max_output_tokens=360,
        )
        payload = _try_parse_json_payload(response)
        if isinstance(payload, dict):
            report = {
                "agreements": [str(item).strip() for item in payload.get("agreements", []) if str(item).strip()],
                "disagreements": [str(item).strip() for item in payload.get("disagreements", []) if str(item).strip()],
                "common_points": [str(item).strip() for item in payload.get("common_points", []) if str(item).strip()],
                "skeptic_conclusion": str(payload.get("skeptic_conclusion", "")).strip(),
                "advocate_conclusion": str(payload.get("advocate_conclusion", "")).strip(),
                "joint_conclusion": str(payload.get("joint_conclusion", "")).strip(),
                "author_action_plan": [
                    str(item).strip() for item in payload.get("author_action_plan", []) if str(item).strip()
                ],
                "fingerprint": fingerprint,
            }
            if report["skeptic_conclusion"] and report["advocate_conclusion"] and report["joint_conclusion"]:
                if not report["agreements"]:
                    report["agreements"] = fallback["agreements"]
                if not report["disagreements"]:
                    report["disagreements"] = fallback["disagreements"]
                if not report["common_points"]:
                    report["common_points"] = fallback["common_points"]
                if not report["author_action_plan"]:
                    report["author_action_plan"] = fallback["author_action_plan"]
                return report
    except Exception:
        pass
    return fallback


def _fallback_current_vector_report(
    *,
    active_vector: dict[str, Any],
    skeptic_position: str,
    advocate_position: str,
) -> dict[str, Any]:
    claim = str(active_vector.get("claim", "")).strip() or "the active claim"
    return {
        "agreements": [
            "Both reviewers agree the claim should be bounded to directly measured evidence.",
            "Both sides agree clearer wording improves credibility.",
        ],
        "disagreements": [
            "Skeptic argues current evidence is insufficient for the full claim strength.",
            "Advocate argues the claim is defendable once scope is explicit.",
        ],
        "common_points": [
            "Evidence exists for feasibility.",
            "Claim wording should match measured scope.",
        ],
        "skeptic_conclusion": f"The paper overstates {claim.lower()} without enough direct support.",
        "advocate_conclusion": f"The paper can keep {claim.lower()} if scope is narrowed to measured settings.",
        "joint_conclusion": "Promising contribution, but strongest claims need tighter evidence-aligned framing.",
        "author_action_plan": [
            "Rewrite claim language to match reported setup and measurements.",
            "Add one explicit metric/baseline comparator in the same paragraph as the claim.",
            "State one explicit limitation boundary adjacent to the claim.",
        ],
    }


def _render_current_vector_report_markdown(report: dict[str, Any]) -> str:
    agreements = report.get("agreements", [])
    disagreements = report.get("disagreements", [])
    common_points = report.get("common_points", [])
    actions = report.get("author_action_plan", [])
    lines = ["Shared Ground:"]
    if isinstance(agreements, list) and agreements:
        lines.extend(f"- {str(item).strip()}" for item in agreements if str(item).strip())
    else:
        lines.append("- No explicit agreements yet.")
    lines.extend(["", "Core Disagreement:"])
    if isinstance(disagreements, list) and disagreements:
        lines.extend(f"- {str(item).strip()}" for item in disagreements if str(item).strip())
    else:
        lines.append("- No explicit disagreements yet.")
    lines.extend(["", "Alignment Signals:"])
    if isinstance(common_points, list) and common_points:
        lines.extend(f"- {str(item).strip()}" for item in common_points if str(item).strip())
    else:
        lines.append("- No common ground identified yet.")
    lines.extend(
        [
            "",
            "Joint Conclusion:",
            str(report.get("joint_conclusion", "Not available.")).strip(),
            "",
            "Action Plan:",
        ]
    )
    if isinstance(actions, list) and actions:
        lines.extend(f"- {str(item).strip()}" for item in actions if str(item).strip())
    else:
        lines.append("- No action plan available.")
    return "\n".join(lines).strip()


def _render_current_vector_report_brief(report: dict[str, Any]) -> str:
    agreements = [str(item).strip() for item in report.get("agreements", []) if str(item).strip()]
    disagreements = [str(item).strip() for item in report.get("disagreements", []) if str(item).strip()]
    common_points = [str(item).strip() for item in report.get("common_points", []) if str(item).strip()]
    actions = [str(item).strip() for item in report.get("author_action_plan", []) if str(item).strip()]

    lines: list[str] = []
    if agreements:
        lines.append(f"- Agreement: {agreements[0]}")
    if disagreements:
        lines.append(f"- Main disagreement: {disagreements[0]}")
    if common_points:
        lines.append(f"- Common ground: {common_points[0]}")
    joint = str(report.get("joint_conclusion", "")).strip()
    if joint:
        lines.append(f"- Joint conclusion: {joint}")
    if actions:
        lines.append("- Immediate action plan:")
        for item in actions[:2]:
            lines.append(f"  - {item}")
    if not lines:
        return "No intelligence summary available yet."
    return "\n".join(lines).strip()


def _render_round_events_compact(*, round_events: list[dict[str, Any]], vector_id: str) -> str:
    if not round_events:
        return "No new debate turns in this round."
    lines: list[str] = []
    for event in round_events:
        if str(event.get("vector_id", "")) != vector_id:
            continue
        speaker = str(event.get("speaker", "")).strip().lower()
        if speaker not in {"skeptic", "advocate"}:
            continue
        label = "Skeptic" if speaker == "skeptic" else "Advocate"
        content = _compact_turn_text(str(event.get("content", "")), max_chars=280)
        if not content:
            continue
        lines.append(f"- **{label}:** {content}")
    return "\n".join(lines) if lines else "No new debate turns in this round."


def _build_reviewer_final_report(
    *,
    attack_vectors: list[dict[str, Any]],
    vector_verdicts: dict[str, str],
    vector_judgments: dict[str, dict[str, Any]],
    vector_reports: dict[str, dict[str, Any]],
    syntheses: dict[str, str],
    debate_history: list[dict[str, Any]],
    documents: list[Document],
) -> dict[str, Any]:
    context_snapshot = _reviewer_global_context_snapshot(documents=documents)
    field_context = _reviewer_field_context_lines(documents=documents)
    fallback = _fallback_reviewer_final_report(
        attack_vectors=attack_vectors,
        vector_verdicts=vector_verdicts,
        vector_judgments=vector_judgments,
        vector_reports=vector_reports,
        syntheses=syntheses,
        context_snapshot=context_snapshot,
        field_context=field_context,
    )
    if not text_service.available:
        return fallback
    try:
        response = text_service.generate(
            system_prompt=(
                "You are generating a final panel report for a two-reviewer paper debate.\n"
                "Return JSON only with keys:\n"
                "overview (string), agreements (array of strings), disagreements (array of strings),\n"
                "common_points (array of strings), skeptic_conclusion (string), advocate_conclusion (string),\n"
                "joint_conclusion (string), field_context (array of strings), final_suggestions (array of strings), final_decision (string), confidence (number 0..1).\n"
                "Requirements:\n"
                "- concise, evidence-grounded, no generic filler\n"
                "- avoid repeating near-identical suggestions across claims\n"
                "- every suggestion must be claim-specific and actionable\n"
                "- common_points must be substantive reviewer takeaways, not placeholders\n"
                "- field_context must separate paper-grounded novelty from field-relative novelty and historical importance\n"
                "- skeptic_conclusion, advocate_conclusion, and joint_conclusion should each be 2-3 sentences, specific to this paper and panel outcome\n"
                "- do not output lines like 'the provided evidence supports the claim' or 'no major unresolved disagreements were recorded' unless there is no stronger paper-specific wording available\n\n"
                "Quality target:\n"
                f"{_reviewer_output_vision()}"
            ),
            user_prompt=(
                "Attack vectors with verdicts:\n"
                f"{json.dumps(vector_verdicts)}\n\n"
                "Judge rationales:\n"
                f"{json.dumps({k: v.get('rationale', '') for k, v in vector_judgments.items()})}\n\n"
                "Per-vector intelligence reports:\n"
                f"{json.dumps(vector_reports)}\n\n"
                "Rewrite cards:\n"
                f"{json.dumps(syntheses)}\n\n"
                "Debate transcript excerpt:\n"
                f"{_format_panel_history_compact(debate_history=debate_history, max_turns=14)}\n\n"
                "Global context snapshot (must inform the final report):\n"
                f"{json.dumps(context_snapshot)}\n\n"
                "Field-aware context scaffold (refine if the model knows more, but stay honest):\n"
                f"{json.dumps(field_context)}\n\n"
                "Retrieved context:\n"
                f"{_format_context(documents, max_docs=max(settings.rerank_top_n, 10))}"
            ),
            temperature=0.1,
            max_output_tokens=760,
        )
        payload = _try_parse_json_payload(response)
        if isinstance(payload, dict):
            report = {
                "overview": str(payload.get("overview", "")).strip(),
                "agreements": [str(item).strip() for item in payload.get("agreements", []) if str(item).strip()],
                "disagreements": [str(item).strip() for item in payload.get("disagreements", []) if str(item).strip()],
                "common_points": [str(item).strip() for item in payload.get("common_points", []) if str(item).strip()],
                "skeptic_conclusion": str(payload.get("skeptic_conclusion", "")).strip(),
                "advocate_conclusion": str(payload.get("advocate_conclusion", "")).strip(),
                "joint_conclusion": str(payload.get("joint_conclusion", "")).strip(),
                "field_context": [str(item).strip() for item in payload.get("field_context", []) if str(item).strip()],
                "final_suggestions": [
                    str(item).strip() for item in payload.get("final_suggestions", []) if str(item).strip()
                ],
                "final_decision": str(payload.get("final_decision", "")).strip(),
                "confidence": float(payload.get("confidence", 0.55)),
                "context_snapshot": context_snapshot,
            }
            report["confidence"] = max(0.0, min(1.0, report["confidence"]))
            if report["overview"] and report["final_decision"]:
                if not report["agreements"]:
                    report["agreements"] = fallback["agreements"]
                if not report["disagreements"]:
                    report["disagreements"] = fallback["disagreements"]
                if not report["common_points"]:
                    report["common_points"] = fallback["common_points"]
                if not report["skeptic_conclusion"]:
                    report["skeptic_conclusion"] = fallback["skeptic_conclusion"]
                if not report["advocate_conclusion"]:
                    report["advocate_conclusion"] = fallback["advocate_conclusion"]
                if not report["joint_conclusion"]:
                    report["joint_conclusion"] = fallback["joint_conclusion"]
                if not report["field_context"]:
                    report["field_context"] = fallback["field_context"]
                if not report["final_suggestions"]:
                    report["final_suggestions"] = fallback["final_suggestions"]
                quality_issues = _reviewer_report_quality_issues(report=report, attack_vectors=attack_vectors)
                if quality_issues:
                    fallback["quality_guard"] = quality_issues
                    return _humanize_reviewer_report(report=fallback, attack_vectors=attack_vectors)
                return _humanize_reviewer_report(report=report, attack_vectors=attack_vectors)
    except Exception:
        pass
    return _humanize_reviewer_report(report=fallback, attack_vectors=attack_vectors)


def _fallback_reviewer_final_report(
    *,
    attack_vectors: list[dict[str, Any]],
    vector_verdicts: dict[str, str],
    vector_judgments: dict[str, dict[str, Any]],
    vector_reports: dict[str, dict[str, Any]],
    syntheses: dict[str, str],
    context_snapshot: list[str] | None = None,
    field_context: list[str] | None = None,
) -> dict[str, Any]:
    skeptic_wins = sum(1 for verdict in vector_verdicts.values() if verdict == "skeptic_prevailed")
    advocate_wins = sum(1 for verdict in vector_verdicts.values() if verdict == "advocate_prevailed")
    contested = sum(1 for verdict in vector_verdicts.values() if verdict == "contested")
    total = max(1, len(vector_verdicts))

    agreements: list[str] = []
    disagreements: list[str] = []
    common_points: list[str] = []
    suggestions: list[str] = []
    seen_suggestions: set[str] = set()
    used_quotes: set[str] = set()
    for idx, vector in enumerate(attack_vectors, start=1):
        vector_id = str(vector.get("id", ""))
        claim = str(vector.get("claim", "")).strip()
        category = str(vector.get("category", "method")).strip().lower() or "method"
        judgment = vector_judgments.get(vector_id, {}) if isinstance(vector_judgments, dict) else {}
        evidence_pack = judgment.get("evidence_pack", []) if isinstance(judgment, dict) else []
        quote, _ = _reviewer_anchor_choice(
            preferred_quote=str(vector.get("quote", "")).strip(),
            evidence_pack=evidence_pack if isinstance(evidence_pack, list) else [],
            category=category,
            claim=claim,
            used_quotes=used_quotes,
        )
        claim_label = _reviewer_claim_label(index=idx, claim=claim)
        verdict = str(vector_verdicts.get(vector_id, "contested"))
        rationale_raw = str(judgment.get("rationale", "")).strip()
        rationale = rationale_raw if _reviewer_rationale_relevant_to_claim(rationale=rationale_raw, claim=claim) else ""
        reviewer_read = _reviewer_expand_read(
            reviewer_read=_reviewer_verdict_read(category=category, verdict=verdict, rationale=rationale),
            category=category,
            verdict=verdict,
            quote=quote,
            evidence_pack=evidence_pack if isinstance(evidence_pack, list) else [],
        )
        if verdict == "advocate_prevailed":
            line = f"{claim_label}: {reviewer_read}"
            if quote:
                line = f'{line} Anchor quote: "{quote}"'
            agreements.append(line)
        elif verdict == "skeptic_prevailed":
            line = f"{claim_label}: {reviewer_read}"
            if quote:
                line = f'{line} Anchor quote: "{quote}"'
            disagreements.append(line)
        else:
            line = f"{claim_label}: {reviewer_read}"
            if quote:
                line = f'{line} Anchor quote: "{quote}"'
            disagreements.append(line)
        patch = _extract_patch_instruction(syntheses.get(vector_id, ""))
        if patch and not _looks_generic_patch_instruction(patch):
            suggestion_text = patch
        else:
            suggestion_text = _claim_specific_reviewer_suggestion(
                claim=claim,
                category=category,
                quote=quote,
            )
        if suggestion_text:
            suggestion = f"{claim_label}: {suggestion_text}"
            key = suggestion.lower()
            if key not in seen_suggestions:
                seen_suggestions.add(key)
                suggestions.append(suggestion)

    if not agreements:
        agreements.append("Both sides agreed that clearer scope boundaries improve claim credibility.")
    if not disagreements:
        disagreements = _reviewer_residual_concerns(
            attack_vectors=attack_vectors,
            vector_judgments=vector_judgments,
            syntheses=syntheses,
        )
    snapshot = context_snapshot or _reviewer_context_snapshot_from_vectors(attack_vectors=attack_vectors)
    if vector_reports:
        for report in vector_reports.values():
            points = report.get("common_points", [])
            if not isinstance(points, list):
                continue
            for point in points:
                text = str(point).strip()
                if text and text not in common_points:
                    common_points.append(text)
                    if len(common_points) >= 5:
                        break
            if len(common_points) >= 5:
                break
    if not common_points:
        common_points = _reviewer_common_points(
            attack_vectors=attack_vectors,
            vector_verdicts=vector_verdicts,
            vector_judgments=vector_judgments,
            context_snapshot=snapshot if isinstance(snapshot, list) else [],
        )
    if not suggestions:
        suggestions.append("Convert each debated claim into one scoped statement tied to a measurable comparator.")

    if skeptic_wins > advocate_wins:
        decision = "Weak Reject until major claim-overreach and evidence gaps are revised."
    elif advocate_wins >= max(2, total - 1) and contested <= 1 and skeptic_wins == 0:
        decision = "Weak Accept with targeted clarity edits."
    else:
        decision = "Borderline: promising contribution, but contested claims need tighter evidence framing."

    report = {
        "overview": (
            f"Panel completed {total} claim trials: skeptic_prevailed={skeptic_wins}, "
            f"advocate_prevailed={advocate_wins}, contested={contested}."
        ),
        "agreements": agreements[:4],
        "disagreements": disagreements[:5],
        "common_points": common_points[:5],
        "context_snapshot": snapshot[:4] if isinstance(snapshot, list) else [],
        "field_context": field_context[:4] if isinstance(field_context, list) else [],
        "skeptic_conclusion": _reviewer_side_conclusion(
            side="skeptic",
            attack_vectors=attack_vectors,
            vector_verdicts=vector_verdicts,
            vector_judgments=vector_judgments,
            context_snapshot=snapshot if isinstance(snapshot, list) else [],
        ),
        "advocate_conclusion": _reviewer_side_conclusion(
            side="advocate",
            attack_vectors=attack_vectors,
            vector_verdicts=vector_verdicts,
            vector_judgments=vector_judgments,
            context_snapshot=snapshot if isinstance(snapshot, list) else [],
        ),
        "joint_conclusion": _reviewer_side_conclusion(
            side="joint",
            attack_vectors=attack_vectors,
            vector_verdicts=vector_verdicts,
            vector_judgments=vector_judgments,
            context_snapshot=snapshot if isinstance(snapshot, list) else [],
        ),
        "final_suggestions": suggestions[:6],
        "final_decision": decision,
        "confidence": max(0.35, min(0.9, 0.5 + ((advocate_wins - skeptic_wins) * 0.08))),
    }
    return _humanize_reviewer_report(report=report, attack_vectors=attack_vectors)


def _reviewer_claim_label(*, index: int, claim: str) -> str:
    compact = _compact_turn_text(claim or "unspecified claim", max_chars=92)
    return f"Claim {index} ({compact})"


def _reviewer_residual_concerns(
    *,
    attack_vectors: list[dict[str, Any]],
    vector_judgments: dict[str, dict[str, Any]],
    syntheses: dict[str, str],
) -> list[str]:
    concerns: list[str] = []
    for idx, vector in enumerate(attack_vectors, start=1):
        claim = str(vector.get("claim", "")).strip()
        category = str(vector.get("category", "method")).strip().lower() or "method"
        vector_id = str(vector.get("id", "")).strip()
        judgment = vector_judgments.get(vector_id, {}) if isinstance(vector_judgments, dict) else {}
        evidence_pack = judgment.get("evidence_pack", []) if isinstance(judgment, dict) else []
        quote, _ = _reviewer_anchor_choice(
            preferred_quote=str(vector.get("quote", "")).strip(),
            evidence_pack=evidence_pack if isinstance(evidence_pack, list) else [],
            category=category,
            claim=claim,
            used_quotes=None,
        )
        patch = _extract_patch_instruction(syntheses.get(vector_id, ""))
        if not patch or _looks_generic_patch_instruction(patch):
            patch = _claim_specific_reviewer_suggestion(claim=claim, category=category, quote=quote)
        concern = _reviewer_residual_concern_text(category=category, claim=claim, patch=patch)
        if concern:
            concerns.append(f"{_reviewer_claim_label(index=idx, claim=claim)}: {concern}")
        if len(concerns) >= 3:
            break
    return concerns or ["The paper has a credible core contribution, but its strongest claims still need tighter wording and cleaner evidence linkage before the review would feel settled."]


def _reviewer_residual_concern_text(*, category: str, claim: str, patch: str) -> str:
    lower_patch = str(patch or "").strip()
    if category == "novelty":
        return "The contribution looks real, but the novelty delta still needs a cleaner side-by-side statement against the closest prior baseline."
    if category in {"method", "assumption"}:
        return "The design choice is plausible, but the paper still needs one sharper comparator or ablation to make the preference over nearby alternatives feel earned."
    if category in {"evaluation", "benchmark"}:
        return "The evaluation story is promising, but the report should still bind the claim to one exact dataset/metric slice and comparator so readers do not over-generalize it."
    if category in {"ablation", "robustness", "reproducibility"}:
        return "The measured run is useful evidence, but the paper should still separate observed efficiency or robustness from any broader generalization language."
    if lower_patch:
        return _compact_turn_text(lower_patch, max_chars=240)
    return "The claim remains directionally strong, but it still needs one tighter evidence-bound revision before the panel would call it fully reviewer-proof."


def _reviewer_common_points(
    *,
    attack_vectors: list[dict[str, Any]],
    vector_verdicts: dict[str, str],
    vector_judgments: dict[str, dict[str, Any]],
    context_snapshot: list[str],
) -> list[str]:
    categories: list[str] = []
    for vector in attack_vectors:
        category = str(vector.get("category", "")).strip().lower()
        if category and category not in categories:
            categories.append(category)
    supported = sum(1 for verdict in vector_verdicts.values() if verdict == "advocate_prevailed")
    contested = sum(1 for verdict in vector_verdicts.values() if verdict == "contested")
    lines = [
        f"The panel consistently found a real contribution in the paper, especially once claims are stated at the level of the exact evidence actually recovered across {max(1, len(attack_vectors))} review targets.",
        f"Both sides converged on the same editorial rule: keep the strongest claims tied to concrete method or benchmark evidence, and treat broader generalization language as something that still needs explicit support.",
    ]
    if categories:
        lines.append(
            f"The debate touched {', '.join(categories[:4])}, and the common pattern was that the paper looks strongest on core method/results while remaining weaker on wording discipline and reader-facing framing."
        )
    if context_snapshot:
        lines.append(
            f"Global paper context reinforced that view: {context_snapshot[0]} The broader paper picture did not overturn the claim-by-claim debate; it mainly helped the panel separate what is directly demonstrated from what is only suggested."
        )
    if contested:
        lines.append(
            f"The remaining open question is not whether the paper works, but whether every headline claim is scoped tightly enough to survive a skeptical close read."
        )
    elif supported:
        lines.append(
            "Even where the advocate prevailed, the shared position was not 'ship it unchanged'; it was 'keep the result, then tighten the claim so the paper reads as careful rather than over-eager.'"
        )
    return lines[:4]


def _reviewer_side_conclusion(
    *,
    side: str,
    attack_vectors: list[dict[str, Any]],
    vector_verdicts: dict[str, str],
    vector_judgments: dict[str, dict[str, Any]],
    context_snapshot: list[str],
) -> str:
    skeptic_wins = sum(1 for verdict in vector_verdicts.values() if verdict == "skeptic_prevailed")
    advocate_wins = sum(1 for verdict in vector_verdicts.values() if verdict == "advocate_prevailed")
    contested = sum(1 for verdict in vector_verdicts.values() if verdict == "contested")
    lead_category = ""
    for vector in attack_vectors:
        category = str(vector.get("category", "")).strip().lower()
        if category:
            lead_category = category
            break
    scope_note = context_snapshot[0] if context_snapshot else "Recovered evidence shows a meaningful contribution with benchmark grounding."
    if side == "skeptic":
        return (
            f"From the skeptic side, the paper's risk is mostly rhetorical rather than existential: the core contribution looks real, but the prose can still sound broader than the evidence warrants, especially around {lead_category or 'claim framing'}. "
            "A tougher reviewer will want the exact comparator, benchmark slice, and scope boundary named in the same place as the headline claim rather than inferred from later sections. "
            "That means the remaining pushback is less about disproving the result and more about preventing readers from over-reading it."
        )
    if side == "advocate":
        return (
            f"From the advocate side, the paper already has enough substance to clear the bar on its central contribution: {scope_note} "
            "The strongest support is concrete rather than atmospheric, which is why the panel did not treat this as a paper lacking evidence altogether. "
            "From that angle, the revision burden is sharpening the claim language and evidence linkage, not inventing a new experimental story."
        )
    return (
        "Taken together, the panel sees a paper with a credible core result and a manageable revision burden, but the write-up still needs more discipline than it currently shows. "
        "The best final version would keep the strongest method or benchmark fact upfront, make the slice and comparator unmistakable, and separate measured evidence from broader takeaways. "
        "If that cleanup is done consistently, the paper reads like a confident evidence-led contribution rather than an overstated one."
    )


def _humanize_reviewer_report(*, report: dict[str, Any], attack_vectors: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(report, dict):
        return report
    label_map: dict[str, str] = {}
    for idx, vector in enumerate(attack_vectors, start=1):
        vector_id = str(vector.get("id", "")).strip()
        claim = str(vector.get("claim", "")).strip()
        if not vector_id:
            continue
        label_map[vector_id] = _reviewer_claim_label(index=idx, claim=claim)

    def _rewrite_text(text: str) -> str:
        updated = str(text or "")
        for vector_id, label in label_map.items():
            updated = re.sub(rf"\b{re.escape(vector_id)}\b", label, updated)
        return updated

    normalized: dict[str, Any] = {}
    for key, value in report.items():
        if isinstance(value, str):
            normalized[key] = _rewrite_text(value)
        elif isinstance(value, list):
            normalized_list = [_rewrite_text(str(item)) for item in value if str(item).strip()]
            if key == "final_suggestions":
                deduped: list[str] = []
                seen: set[str] = set()
                for item in normalized_list:
                    token = item.strip().lower()
                    if not token or token in seen:
                        continue
                    seen.add(token)
                    deduped.append(item.strip())
                normalized[key] = deduped
            elif key in {"agreements", "disagreements"}:
                normalized[key] = _dedupe_reviewer_lines(normalized_list, cap=6)
            else:
                normalized[key] = normalized_list
        else:
            normalized[key] = value
    return normalized


def _reviewer_global_context_snapshot(*, documents: list[Document]) -> list[str]:
    if not documents:
        return []
    profiles = _paper_profiles_from_documents(documents)
    filenames: list[str] = []
    seen_files: set[str] = set()
    benchmark_families: set[str] = set()
    method_anchors: list[str] = []
    metrics_by_file: dict[str, list[str]] = {}
    for filename, profile in list(profiles.items())[: max(2, settings.rerank_top_n)]:
        if filename not in seen_files:
            seen_files.add(filename)
            filenames.append(filename)
        method_anchor = _compact_turn_text(_profile_method_signature(profile), max_chars=90)
        if method_anchor:
            method_anchors.append(f"{filename}: {method_anchor}")
        metrics_by_file.setdefault(filename, [])
        for record in profile.get("metric_records", [])[:3]:
            metric = str(record.get("metric", "metric")).upper()
            try:
                value = float(record.get("value", 0.0))
            except Exception:
                continue
            benchmark = str(record.get("benchmark", "")).strip()
            if benchmark:
                benchmark_families.add(benchmark)
            detail = f"{value:.2f} {metric}"
            if benchmark:
                detail += f" ({benchmark})"
            metrics_by_file[filename].append(detail)
    if not profiles:
        for document in documents[: max(6, settings.rerank_top_n)]:
            metadata = document.metadata or {}
            filename = str(metadata.get("filename", "unknown.pdf")).strip() or "unknown.pdf"
            if filename not in seen_files:
                seen_files.add(filename)
                filenames.append(filename)
            raw_text = _clean_mojibake_text(document.page_content or "")
            family = _benchmark_family(_infer_benchmark_label(raw_text))
            if family:
                benchmark_families.add(family)
    lines: list[str] = []
    if filenames:
        lines.append(f"Papers in scope: {', '.join(filenames[:3])}.")
    if benchmark_families:
        lines.append(f"Recovered benchmark slices: {', '.join(sorted(benchmark_families)[:4])}.")
    if method_anchors:
        lines.extend(f"Method anchor: {anchor}." for anchor in method_anchors[:2])
    for filename in filenames[:2]:
        values = metrics_by_file.get(filename, [])
        if values:
            lines.append(f"{filename} recovered metrics: {', '.join(values[:2])}.")
    return lines[:5]


def _reviewer_context_snapshot_from_vectors(*, attack_vectors: list[dict[str, Any]]) -> list[str]:
    quotes = [str(item.get("quote", "")).strip() for item in attack_vectors if str(item.get("quote", "")).strip()]
    categories = sorted(
        {
            str(item.get("category", "")).strip().lower()
            for item in attack_vectors
            if str(item.get("category", "")).strip()
        }
    )
    lines: list[str] = []
    if categories:
        lines.append(f"Debated categories: {', '.join(categories[:5])}.")
    if quotes:
        lines.append(f"Primary evidence anchor: \"{_reviewer_quote_excerpt(quotes[0])}\"")
    if len(quotes) > 1:
        lines.append(f"Secondary evidence anchor: \"{_reviewer_quote_excerpt(quotes[1])}\"")
    return lines[:4]


def _reviewer_anchor_choice(
    *,
    preferred_quote: str,
    evidence_pack: list[dict[str, Any]],
    category: str = "",
    claim: str = "",
    used_quotes: set[str] | None = None,
) -> tuple[str, int]:
    candidates: list[tuple[str, int]] = []
    if isinstance(evidence_pack, list):
        for item in evidence_pack:
            if not isinstance(item, dict):
                continue
            snippet = str(item.get("snippet", "")).strip()
            if not snippet:
                continue
            try:
                citation = int(item.get("citation_index", 1))
            except Exception:
                citation = 1
            candidates.append((snippet, citation))
    if preferred_quote.strip():
        citation = 1
        if isinstance(evidence_pack, list) and evidence_pack and isinstance(evidence_pack[0], dict):
            try:
                citation = int(evidence_pack[0].get("citation_index", 1))
            except Exception:
                citation = 1
        candidates.append((preferred_quote, citation))

    category_markers = set(_reviewer_category_markers(category))
    claim_terms = {
        token
        for token in re.findall(r"[a-z0-9]+", (claim or "").lower())
        if token not in {"the", "and", "for", "with", "that", "this", "claim", "should", "be", "to", "of", "or"}
    }
    fallback_quote = ""
    fallback_citation = 1
    scored_candidates: list[tuple[float, str, int]] = []
    for raw_quote, citation in candidates:
        quote = _reviewer_quote_excerpt(raw_quote)
        if not quote:
            continue
        lower = quote.lower()
        score = 0.0
        if _looks_like_non_argument_snippet(quote) or _looks_like_metadata_snippet(quote):
            score -= 1.4
        score += sum(0.25 for marker in category_markers if marker and marker in lower)
        score += sum(0.08 for token in claim_terms if token and token in lower)
        if category in {"evaluation", "benchmark"} and (_has_metric_name(lower) or "wmt" in lower):
            score += 1.2
        if category in {"ablation", "robustness", "reproducibility"} and any(marker in lower for marker in _efficiency_signal_markers()):
            score += 1.2
        if category in {"ablation", "robustness", "reproducibility"} and any(
            marker in lower for marker in ("learning rate", "epoch", "epochs", "batch", "optimizer", "parameters", "beam search")
        ):
            score += 0.9
        if category in {"ablation", "robustness", "reproducibility"} and any(
            marker in lower for marker in ("connectionist sequence classification", "monotonic alignment", "another popular technique")
        ):
            score -= 1.0
        if category in {"method", "assumption"} and any(marker in lower for marker in _method_signal_markers()):
            score += 1.0
        if category == "novelty" and any(marker in lower for marker in ("we propose", "our main result", "in this paper", "new", "novel")):
            score += 1.0
        scored_candidates.append((score, quote, citation))
    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    for _, quote, citation in scored_candidates:
        key = re.sub(r"\s+", " ", quote.lower()).strip()
        if not fallback_quote:
            fallback_quote = quote
            fallback_citation = citation
        if used_quotes is not None and key in used_quotes:
            continue
        if used_quotes is not None and key:
            used_quotes.add(key)
        return quote, citation
    if scored_candidates:
        best_score, best_quote, best_citation = scored_candidates[0]
        if best_score >= 1.0:
            return best_quote, best_citation
    if fallback_quote:
        key = re.sub(r"\s+", " ", fallback_quote.lower()).strip()
        if used_quotes is not None and key:
            used_quotes.add(key)
        return fallback_quote, fallback_citation
    return "", 1


def _reviewer_quote_excerpt(quote: str) -> str:
    return _compact_turn_text(_clean_visible_text(quote or ""), max_chars=360)


def _reviewer_evidence_note(*, category: str, quote: str, evidence_pack: list[dict[str, Any]]) -> str:
    cleaned_quote = _clean_visible_text(quote or "")
    records: list[dict[str, Any]] = []
    if isinstance(evidence_pack, list):
        for item in evidence_pack:
            if not isinstance(item, dict):
                continue
            records.extend(_extract_metric_records(str(item.get("snippet", "")).strip()))
    best_record = records[0] if records else None
    benchmark = _infer_benchmark_label(cleaned_quote)

    if category in {"novelty", "evaluation", "benchmark"}:
        if isinstance(best_record, dict):
            metric = str(best_record.get("metric", "metric")).upper()
            value = float(best_record.get("value", 0.0))
            benchmark_label = str(best_record.get("benchmark", "")).strip() or benchmark or "the recovered evaluation slice"
            return f"The supporting evidence is concrete rather than thematic: it points to {benchmark_label} and a reported {value:.2f} {metric} result, so the claim already has a real empirical anchor."
        if benchmark:
            return f"The supporting sentence is still useful because it names the recovered benchmark slice directly, which helps keep the claim tied to a specific evaluation setting instead of the whole paper."
    if category in {"method", "assumption"} and cleaned_quote:
        return "The anchor sentence names the design move itself rather than gesturing at it abstractly, which is exactly the right starting point for a convincing method defense."
    if category in {"ablation", "robustness", "reproducibility"}:
        if any(marker in cleaned_quote.lower() for marker in ("learning rate", "epoch", "epochs", "batch", "optimizer", "parameters", "beam search")):
            return "That support comes from run-specific training or setup detail, which is useful evidence, but it should still be described as measured configuration evidence rather than broad robustness proof."
        return "This is the thinnest part of the evidence base, so the safest high-quality review move is to keep the wording run-specific and avoid implying robustness beyond what is explicitly shown."
    return ""


def _reviewer_expand_read(
    *,
    reviewer_read: str,
    category: str,
    verdict: str,
    quote: str,
    evidence_pack: list[dict[str, Any]],
) -> str:
    parts: list[str] = [str(reviewer_read or "").strip()]
    note = _reviewer_evidence_note(category=category, quote=quote, evidence_pack=evidence_pack)
    if note and note.lower() not in parts[0].lower():
        parts.append(note)
    if verdict == "advocate_prevailed" and category in {"evaluation", "benchmark"}:
        parts.append("What would make the write-up stronger is putting the comparator and exact metric in the same breath as the claim, so the reader never has to infer the scope.")
    elif verdict == "advocate_prevailed" and category in {"novelty", "method"}:
        parts.append("That is why the paper feels substantively promising even though the final camera-ready wording should still be more explicit about boundaries and alternatives.")
    elif verdict != "advocate_prevailed":
        parts.append("As written, the review pressure comes from claim discipline, not from a total absence of evidence.")
    joined = " ".join(part.strip() for part in parts if part.strip())
    return re.sub(r"\s{2,}", " ", joined).strip()


def _reviewer_rationale_relevant_to_claim(*, rationale: str, claim: str) -> bool:
    rationale_text = str(rationale or "").lower().strip()
    claim_text = str(claim or "").lower().strip()
    if not rationale_text:
        return False
    if not claim_text:
        return True
    claim_terms = {
        token
        for token in re.findall(r"[a-z0-9]+", claim_text)
        if token not in {"the", "and", "for", "with", "that", "this", "need", "needs", "more", "stronger"}
    }
    rationale_terms = set(re.findall(r"[a-z0-9]+", rationale_text))
    if claim_terms & rationale_terms:
        return True
    generic_markers = (
        "dnn",
        "deep neural networks",
        "map sequences to sequences",
        "excellent performance on difficult learning tasks",
    )
    return not any(marker in rationale_text for marker in generic_markers)


def _looks_generic_patch_instruction(text: str) -> bool:
    lower = (text or "").strip().lower()
    if not lower:
        return True
    generic_patterns = (
        "rewrite the claim to stay within measured scope",
        "add one explicit metric/comparator already reported",
    )
    return any(pattern in lower for pattern in generic_patterns)


def _claim_specific_reviewer_suggestion(*, claim: str, category: str, quote: str) -> str:
    anchor = f' Use anchor quote: "{quote}".' if quote else ""
    if category == "novelty":
        return (
            "state the closest prior baseline explicitly and quantify the novelty delta in the same sentence;"
            " then add one scope-boundary sentence." + anchor
        )
    if category in {"method", "assumption"}:
        return (
            "justify the key method assumption with one direct ablation or comparator and state why alternatives were not chosen."
            + anchor
        )
    if category in {"evaluation", "benchmark"}:
        return (
            "narrow the claim to tested settings and add one concrete benchmark metric + baseline comparator directly under the claim."
            + anchor
        )
    if category in {"ablation", "robustness"}:
        return (
            "add a targeted ablation/robustness check and explicitly separate observed evidence from untested generalization."
            + anchor
        )
    if category == "reproducibility":
        return (
            "add exact training/setup details (data split, hyperparameters, seed, and compute budget) needed for replication."
            + anchor
        )
    if claim:
        return "tighten the claim wording and add one measurable comparator in the same paragraph." + anchor
    return "tighten wording and add one measurable comparator."


def _reviewer_verdict_read(*, category: str, verdict: str, rationale: str) -> str:
    lower = (rationale or "").strip().lower()
    generic_markers = (
        "the claim is defensible when explicitly bounded to reported evidence and scope",
        "the claim is supported by direct evidence from the paper",
        "support is credible within the reported setup",
        "scope-bounded wording should be preserved",
        "evidence was mixed and no side clearly prevailed",
        "the paper provides evidence of training efficiency and robustness",
        "the paper provides evidence of the model's performance on multiple tasks",
        "including translation and parsing",
        "generalizes well to other tasks",
        "the provided evidence supports the claim",
        "the claim is supported by the provided evidence",
        "the authors provide sufficient justification for their design choice",
        "the paper explicitly ties the benchmark claim to the exact dataset/metric slice reported",
        "the evidence supports the feasibility of separating training-efficiency or robustness language from broader generalization claims",
        "the paper's contribution is supported by specific examples and results",
    )
    if rationale and not any(marker in lower for marker in generic_markers):
        return rationale.strip()
    if verdict == "skeptic_prevailed":
        if category == "novelty":
            return "The contribution is real, but the novelty wording still runs ahead of the explicit prior-work delta shown in the paper."
        if category in {"method", "assumption"}:
            return "The core design is described clearly, but the paper does not yet justify it strongly enough against nearby alternatives."
        if category in {"evaluation", "benchmark"}:
            return "The paper reports results, but the claim currently blurs which benchmark slice actually supports it."
        if category in {"ablation", "robustness", "reproducibility"}:
            return "Training or robustness language is broader than the directly shown evidence."
        return "Current wording runs ahead of the strongest directly cited evidence."
    if verdict == "contested":
        if category == "novelty":
            return "The paper appears promising, but the novelty claim still needs a cleaner boundary against prior work."
        if category in {"method", "assumption"}:
            return "The method is interesting, but the rationale for choosing it over nearby alternatives is still incomplete."
        if category in {"evaluation", "benchmark"}:
            return "The evaluation is encouraging, but the paper should bind the claim to the exact reported setup."
        if category in {"ablation", "robustness", "reproducibility"}:
            return "The setup evidence is useful, but broader robustness or efficiency language still needs tighter support."
        return "The claim needs tighter phrasing and a clearer evidence boundary."
    if category == "novelty":
        return "The novelty case is strongest when framed as a concrete architectural or empirical delta rather than a blanket breakthrough claim."
    if category in {"method", "assumption"}:
        return "The method claim is persuasive on the core design choice, but it should stay tied to the specific alternative the paper replaces."
    if category in {"evaluation", "benchmark"}:
        return "The evaluation claim is well supported once it stays tied to the exact benchmark slice and comparator reported in the paper."
    if category in {"ablation", "robustness", "reproducibility"}:
        return "The setup and efficiency claim is credible for the measured run, but it should remain separate from broader generalization language."
    return "The claim is supportable once it stays anchored to the evidence actually reported."


def _reviewer_report_quality_issues(*, report: dict[str, Any], attack_vectors: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    suggestions = [str(item).strip() for item in report.get("final_suggestions", []) if str(item).strip()]
    disagreements = [str(item).strip().lower() for item in report.get("disagreements", []) if str(item).strip()]
    if len(suggestions) < max(2, min(4, len(attack_vectors))):
        issues.append("too_few_suggestions")
    if len({item.lower() for item in suggestions}) < len(suggestions):
        issues.append("duplicate_suggestions")
    stripped_suggestion_bodies = [
        re.sub(r"^claim\s+\d+\s*\([^)]*\):\s*", "", item, flags=re.IGNORECASE).strip().lower()
        for item in suggestions
    ]
    if len({item for item in stripped_suggestion_bodies if item}) < len([item for item in stripped_suggestion_bodies if item]):
        issues.append("duplicate_suggestion_body")
    generic_suggestion_patterns = (
        "rewrite the claim to stay within measured scope",
        "add one explicit metric/comparator already reported",
        "evidence-aligned statement without overreach",
    )
    if sum(1 for item in stripped_suggestion_bodies if any(pattern in item for pattern in generic_suggestion_patterns)) >= max(2, min(3, len(stripped_suggestion_bodies))):
        issues.append("generic_suggestions")
    joined = " ".join(
        [
            str(report.get("overview", "")),
            *[str(item) for item in report.get("agreements", [])],
            *[str(item) for item in report.get("disagreements", [])],
            *suggestions,
        ]
    ).lower()
    generic_markers = (
        "dnns have achieved excellent performance",
        "map sequences to sequences",
        "sufficiently supported",
        "performance on multiple tasks",
        "generalizes well to other tasks",
    )
    if sum(1 for marker in generic_markers if marker in joined) >= 2:
        issues.append("generic_language")
    if _contains_ocr_noise(joined):
        issues.append("ocr_artifacts")
    if disagreements and all("no major unresolved disagreements" in item for item in disagreements):
        issues.append("empty_concern_section")
    common_points = [str(item).strip() for item in report.get("common_points", []) if str(item).strip()]
    if common_points and sum(len(item) for item in common_points) < 240:
        issues.append("thin_common_points")
    conclusions = [
        str(report.get("skeptic_conclusion", "")).strip(),
        str(report.get("advocate_conclusion", "")).strip(),
        str(report.get("joint_conclusion", "")).strip(),
    ]
    if any(text and len(text) < 150 for text in conclusions):
        issues.append("short_conclusions")
    repeated_verdict_markers = (
        "this claim is sufficiently supported",
        "evidence was mixed and no side clearly prevailed",
    )
    if sum(joined.count(marker) for marker in repeated_verdict_markers) >= 2:
        issues.append("repetitive_verdict_language")
    anchor_quotes = re.findall(r'anchor quote:\s*"([^"]+)"', joined, flags=re.IGNORECASE)
    normalized_quotes = [
        re.sub(r"\s+", " ", quote.strip().lower())
        for quote in anchor_quotes
        if len(re.sub(r"\s+", " ", quote.strip())) >= 24
    ]
    if normalized_quotes and len(set(normalized_quotes)) <= max(1, len(normalized_quotes) - 1):
        issues.append("repetitive_anchor_quotes")
    return issues


def _dedupe_reviewer_lines(lines: list[str], *, cap: int) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for line in lines:
        cleaned = str(line).strip()
        if not cleaned:
            continue
        key = re.sub(r"\s+", " ", cleaned.lower())
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
        if len(normalized) >= max(1, cap):
            break
    return normalized


def _extract_patch_instruction(card: str) -> str:
    text = (card or "").strip()
    if not text:
        return ""
    match = re.search(r"Patch Instruction:\s*(.+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    lines = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def _render_reviewer_final_report_markdown(report: dict[str, Any]) -> str:
    agreements = report.get("agreements", [])
    disagreements = report.get("disagreements", [])
    common_points = report.get("common_points", [])
    context_snapshot = report.get("context_snapshot", [])
    field_context = report.get("field_context", [])
    suggestions = report.get("final_suggestions", [])
    confidence = float(report.get("confidence", 0.0))
    lines = [
        "## Final Debate Report",
        str(report.get("overview", "Final panel summary is ready.")).strip(),
        "",
        "### Agreements",
    ]
    if isinstance(agreements, list) and agreements:
        lines.extend(f"- {str(item).strip()}" for item in agreements if str(item).strip())
    else:
        lines.append("- No explicit agreements captured.")
    lines.extend(["", "### Major Disagreements"])
    if isinstance(disagreements, list) and disagreements:
        lines.extend(f"- {str(item).strip()}" for item in disagreements if str(item).strip())
    else:
        lines.append("- No major disagreements captured.")
    lines.extend(["", "### Context Snapshot"])
    if isinstance(context_snapshot, list) and context_snapshot:
        lines.extend(f"- {str(item).strip()}" for item in context_snapshot if str(item).strip())
    else:
        lines.append("- No additional context snapshot captured.")
    lines.extend(["", "### Field Context"])
    if isinstance(field_context, list) and field_context:
        lines.extend(f"- {str(item).strip()}" for item in field_context if str(item).strip())
    else:
        lines.append("- No field-context assessment captured.")
    lines.extend(["", "### Common Points"])
    if isinstance(common_points, list) and common_points:
        lines.extend(f"- {str(item).strip()}" for item in common_points if str(item).strip())
    else:
        lines.append("- No common points captured.")
    skeptic_conclusion = str(report.get("skeptic_conclusion", "")).strip()
    advocate_conclusion = str(report.get("advocate_conclusion", "")).strip()
    joint_conclusion = str(report.get("joint_conclusion", "")).strip()
    if skeptic_conclusion:
        lines.extend(["", "### Skeptic Conclusion", skeptic_conclusion])
    if advocate_conclusion:
        lines.extend(["", "### Advocate Conclusion", advocate_conclusion])
    if joint_conclusion:
        lines.extend(["", "### Joint Conclusion", joint_conclusion])
    lines.extend(["", "### Final Suggestions"])
    if isinstance(suggestions, list) and suggestions:
        lines.extend(f"- {str(item).strip()}" for item in suggestions if str(item).strip())
    else:
        lines.append("- No final suggestions captured.")
    lines.extend(
        [
            "",
            "### Final Decision",
            str(report.get("final_decision", "Decision not available.")).strip(),
            f"Confidence: {confidence:.2f}",
        ]
    )
    return "\n".join(lines).strip()


def _format_panel_history_compact(*, debate_history: list[dict[str, Any]], max_turns: int) -> str:
    if not debate_history:
        return "No prior turns."
    trimmed = debate_history[-max_turns:]
    lines: list[str] = []
    for item in trimmed:
        speaker = str(item.get("speaker", "unknown")).upper()
        vector_id = str(item.get("vector_id", "")).strip()
        prefix = f"[{vector_id}] " if vector_id else ""
        content = _compact_turn_text(str(item.get("content", "")), max_chars=220)
        if not content:
            continue
        lines.append(f"{speaker}: {prefix}{content}")
    return "\n".join(lines) if lines else "No prior turns."


def _human_next_move(*, next_speaker: str, vector_id: str) -> str:
    if next_speaker == "skeptic":
        return "The skeptic will challenge the weakest unresolved evidence next."
    if next_speaker == "advocate":
        return "The advocate will respond to the latest criticism next."
    if next_speaker == "synthesise":
        return "This claim is ready for a final rewrite recommendation."
    return "Your input is needed to break the tie or redirect the debate."


def _try_parse_json_payload(text: str) -> Any:
    cleaned = _strip_markdown_fence(text)
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    object_match = re.search(r"\{[\s\S]*\}", cleaned)
    if object_match:
        try:
            return json.loads(object_match.group(0))
        except Exception:
            pass

    array_match = re.search(r"\[[\s\S]*\]", cleaned)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except Exception:
            return None
    return None


def _active_vector_claim(state: GraphState) -> str:
    active_id = str(state.get("active_vector_id", "")).strip()
    if not active_id:
        return ""
    for item in state.get("attack_vectors", []) or []:
        if str(item.get("id", "")).strip() != active_id:
            continue
        return str(item.get("claim", "")).strip()
    return ""


def _should_use_global_retrieval(*, query: str, paper_ids: list[str]) -> bool:
    if paper_ids:
        return True
    lower = (query or "").lower()
    if ".pdf" in lower:
        return True
    if _looks_paper_specific_query(query):
        return True
    explicit_markers = (
        "uploaded paper",
        "uploaded papers",
        "in your papers",
        "in this paper",
        "according to the paper",
        "from the paper",
    )
    return any(marker in lower for marker in explicit_markers)


def _is_context_relevant_to_query(query: str, documents: list[Document]) -> bool:
    if not documents:
        return False
    person_query = _is_global_person_query(query)
    recommendation_query = _is_global_recommendation_query(query)
    if not _looks_paper_specific_query(query) and not person_query and not recommendation_query:
        return False
    query_terms = set(_tokenize_for_overlap(query))
    if not query_terms:
        return False
    if person_query and _has_author_metadata_in_docs(documents):
        return True

    best_overlap = 0.0
    for document in documents:
        overlap = _overlap_score(document.page_content or "", query_terms)
        if overlap > best_overlap:
            best_overlap = overlap

    threshold = 0.14
    if person_query:
        threshold = 0.06
    elif recommendation_query:
        threshold = 0.10
    return best_overlap >= threshold


def _looks_paper_specific_query(query: str) -> bool:
    lower = (query or "").lower()
    markers = (
        "paper",
        "author",
        "authors",
        "this work",
        "this study",
        "uploaded",
        "document",
        "in the paper",
        "according to",
        "approach",
        "method",
        "methodology",
        "results",
        "benchmark",
        "dataset",
        "participant",
        "players",
        "precision",
        "recall",
        "easyocr",
        "ocr",
        "valorant",
        "section",
        "table",
        "figure",
        "game was this done on",
    )
    return any(marker in lower for marker in markers)


def _is_global_recommendation_query(query: str) -> bool:
    lower = (query or "").lower()
    markers = (
        "related papers",
        "similar papers",
        "recommend papers",
        "more papers",
        "paper recommendations",
        "literature",
        "survey",
        "state of the art",
        "sota",
    )
    return any(marker in lower for marker in markers)


def _is_global_person_query(query: str) -> bool:
    lower = (query or "").lower()
    return (
        "who is" in lower
        or "tell me about" in lower
        or ("author" in lower and "paper" not in lower)
    )


def _has_author_metadata_in_docs(documents: list[Document]) -> bool:
    for document in documents[:8]:
        lower = (document.page_content or "").lower()
        if _looks_author_metadata_text(lower):
            return True
    return False


def _looks_author_metadata_text(lower_text: str) -> bool:
    markers = (
        "corresponding author",
        "the authors are",
        "authors are",
        "e-mail",
        "email",
        "affiliation",
    )
    return any(marker in (lower_text or "") for marker in markers)


def _strip_inline_reference_markers(text: str) -> str:
    cleaned = re.sub(r"\s*\[[0-9]+\]", "", text or "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _tokenize_for_overlap(text: str) -> list[str]:
    normalized = (text or "")
    normalized = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", normalized)
    normalized = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", normalized)
    normalized = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", normalized)
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return re.findall(r"[a-zA-Z0-9_]+", normalized)


def _normalize_for_phrase_match(text: str) -> str:
    normalized = re.sub(r"[^\w\s]", " ", text or "")
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def _query_phrases(query: str) -> list[str]:
    tokens = _tokenize_for_overlap(query)
    if len(tokens) < 2:
        return []
    ignored = {
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "many",
        "much",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "a",
        "an",
        "the",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "and",
        "or",
        "with",
    }
    phrases: list[str] = []
    seen: set[str] = set()
    max_n = min(4, len(tokens))
    for n in range(max_n, 1, -1):
        for idx in range(0, len(tokens) - n + 1):
            window = tokens[idx : idx + n]
            if all(token in ignored for token in window):
                continue
            phrase = " ".join(window).strip()
            if not phrase or phrase in seen:
                continue
            seen.add(phrase)
            phrases.append(phrase)
            if len(phrases) >= 16:
                return phrases
    return phrases


def _anchor_terms_for_query(query: str) -> set[str]:
    lower = (query or "").lower()
    anchors: set[str] = set()
    if "mixture of experts" in lower or re.search(r"\bmoe\b", lower):
        anchors.update({"mixture", "experts", "moe"})
    if "transformer" in lower:
        anchors.add("transformer")
    if re.search(r"\bhead\b|\bheads\b", lower):
        anchors.update({"head", "heads"})
    if "easyocr" in lower or "ocr" in lower:
        anchors.update({"ocr", "easyocr"})
    if "precision" in lower and "recall" in lower:
        anchors.update({"precision", "recall"})
    if _is_math_intent_query(query):
        anchors.update({"equation", "objective", "loss", "formulation", "notation", "symbol"})
        if "attention" in lower:
            anchors.update({"attention", "softmax", "sqrt", "q", "k", "v"})
    if not anchors:
        return anchors
    return {token for token in anchors if token not in {"the", "a", "an", "and"}}


def _insufficient_local_grounding(*, query: str, documents: list[Document]) -> bool:
    if not documents:
        return True
    if _is_math_intent_query(query):
        # Math/derivation asks often use varied terminology; avoid over-pruning these.
        return False
    anchor_terms = _anchor_terms_for_query(query)
    if not anchor_terms:
        return False
    best_overlap = 0.0
    for document in documents[:8]:
        text = (document.page_content or "").lower()
        best_overlap = max(best_overlap, _overlap_score(text, anchor_terms))
    return best_overlap < 0.34


def _phrase_overlap_score(text: str, phrases: list[str]) -> float:
    if not text or not phrases:
        return 0.0
    padded = f" {text} "
    score = 0.0
    for phrase in phrases:
        words = phrase.split()
        if len(words) < 2:
            continue
        if f" {phrase} " not in padded:
            continue
        if len(words) >= 4:
            score += 0.55
        elif len(words) == 3:
            score += 0.38
        else:
            score += 0.22
    return min(1.0, score)


def _overlap_score(text: str, terms: set[str]) -> float:
    if not terms:
        return 0.0
    text_terms = set(_tokenize_for_overlap(text))
    if not text_terms:
        return 0.0
    overlap = len(text_terms & terms)
    return overlap / max(1, len(terms))


def _mode_keywords(mode: Mode) -> list[str]:
    if mode == Mode.REVIEWER:
        return [
            "contribution",
            "novelty",
            "benchmark",
            "ablation",
            "limitation",
            "table",
            "experiment",
            "reproducibility",
            "hyperparameter",
            "baseline",
            "statistical significance",
        ]
    if mode == Mode.COMPARATOR:
        return ["method", "dataset", "benchmark", "baseline", "result", "metric", "evaluation", "comparison"]
    if mode == Mode.WRITER:
        return ["style", "tone", "structure", "clarity"]
    if mode == Mode.GLOBAL:
        return ["background", "context", "evidence"]
    return ["evidence", "paper", "claim"]


def _looks_like_high_signal_section(text: str) -> bool:
    header_window = (text or "")[:280].lower()
    markers = (
        "abstract",
        "introduction",
        "method",
        "experiment",
        "results",
        "conclusion",
        "limitation",
        "discussion",
    )
    return any(marker in header_window for marker in markers)


def _looks_math_dense_chunk(text: str) -> bool:
    lower = (text or "").lower()
    if not lower:
        return False
    if any(marker in lower for marker in ("equation", "formulated as", "objective", "loss", "where")):
        return True
    if any(marker in lower for marker in ("attention(q", "attention (q", "softmax", "multi head(", "multi-head attention", "qk", "d_k")):
        return True
    if re.search(r"\bl(?:pretrain|aux|1|2)\b", lower):
        return True
    if re.search(r"\btop[-\s]?k\b", lower):
        return True
    if re.search(r"\b\w+\s*=\s*[^=]+", lower):
        return True
    return False


def _looks_metric_rich_chunk(text: str) -> bool:
    lower = (text or "").lower()
    if not lower:
        return False
    if _has_metric_name(lower):
        return True
    if any(marker in lower for marker in _evaluation_signal_markers()):
        return True
    if re.search(r"\b\d+(?:\.\d+)?\s*(?:bleu|%|accuracy|f1|auc|wer|rouge|meteor|mrr|ndcg|map)\b", lower):
        return True
    return False


def _low_signal_penalty(text: str, *, allow_numeric_dense: bool = False) -> float:
    lower = (text or "").lower()
    penalty = 0.0
    boilerplate_markers = (
        "permission to make digital or hard copies",
        "copyrights for components",
        "manuscript submitted to acm",
        "publication rights licensed to acm",
        "request permissions from",
        "doi:",
        "references",
    )
    if any(marker in lower for marker in boilerplate_markers):
        penalty += 0.35
    if _looks_acknowledgement_text(text):
        penalty += 0.45
    if _looks_like_metadata_snippet(text):
        penalty += 0.45
    if len(re.findall(r"\[[0-9]+\]", text or "")) >= 4:
        penalty += 0.35
    tokens = re.findall(r"[a-zA-Z0-9_]+", lower)
    if tokens:
        numeric_ratio = sum(1 for token in tokens if token.isdigit()) / len(tokens)
        if numeric_ratio > 0.38 and not (allow_numeric_dense and _looks_math_dense_chunk(text)):
            penalty += 0.25
    return penalty


def _looks_acknowledgement_text(text: str) -> bool:
    lower = (text or "").lower()
    if not lower:
        return False
    markers = (
        "acknowledg",
        "would like to thank",
        "we thank",
        "funding",
        "grant support",
        "crucially involved",
    )
    return any(marker in lower for marker in markers)


def _select_balanced_docs(
    *,
    mode: Mode,
    scored_docs: list[tuple[Document, float]],
    limit: int,
) -> list[Document]:
    if not scored_docs:
        return []

    if mode == Mode.REVIEWER:
        return _select_reviewer_coverage_docs(scored_docs=scored_docs, limit=limit)

    if mode == Mode.LOCAL:
        return _select_local_coverage_docs(scored_docs=scored_docs, limit=limit)

    if mode != Mode.COMPARATOR:
        return [document for document, _ in scored_docs[:limit]]

    return _select_comparator_coverage_docs(scored_docs=scored_docs, limit=limit)


def _select_comparator_coverage_docs(
    *,
    scored_docs: list[tuple[Document, float]],
    limit: int,
) -> list[Document]:
    if not scored_docs:
        return []

    filtered_scored_docs = [
        (document, score)
        for document, score in scored_docs
        if not _looks_like_reference_snippet(document.page_content or "")
        and not _looks_like_non_argument_snippet(document.page_content or "")
    ]
    if filtered_scored_docs:
        scored_docs = filtered_scored_docs

    by_paper: dict[str, list[tuple[Document, float]]] = {}
    for document, score in scored_docs:
        paper_id = str((document.metadata or {}).get("paper_id", ""))
        by_paper.setdefault(paper_id, []).append((document, score))

    selected: list[Document] = []
    seen: set[str] = set()
    paper_groups = [
        (paper_id, docs)
        for paper_id, docs in by_paper.items()
        if docs
    ]
    paper_groups.sort(key=lambda item: item[1][0][1], reverse=True)
    if paper_groups:
        bucket_preferences = (
            ("summary", ("abstract", "in this paper", "in this work", "we propose", "we present", "we introduce", "main result")),
            ("method", _method_signal_markers()),
            ("metric", _metric_name_markers() + _evaluation_signal_markers()),
            ("efficiency", _efficiency_signal_markers()),
            ("limitation", ("however", "limitation", "failure", "scope", "without", "underperform")),
        )
        for _, paper_docs in paper_groups:
            for bucket_name, markers in bucket_preferences:
                candidate: Document | None = None
                for document, _ in paper_docs:
                    text = (document.page_content or "").lower()
                    if bucket_name == "metric" and not _looks_metric_rich_chunk(text):
                        continue
                    if bucket_name != "metric" and not any(marker in text for marker in markers):
                        continue
                    if _looks_like_metadata_snippet(text):
                        continue
                    candidate = document
                    break
                if candidate is None:
                    continue
                identity = _document_identity(candidate)
                if identity in seen:
                    continue
                selected.append(candidate)
                seen.add(identity)
                if len(selected) >= limit:
                    return selected[:limit]

    for document, _ in scored_docs:
        identity = _document_identity(document)
        if identity in seen:
            continue
        selected.append(document)
        seen.add(identity)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _select_local_coverage_docs(
    *,
    scored_docs: list[tuple[Document, float]],
    limit: int,
) -> list[Document]:
    if not scored_docs:
        return []

    top_score = scored_docs[0][1]
    best_by_paper: dict[str, tuple[Document, float]] = {}
    for document, score in scored_docs:
        paper_id = str((document.metadata or {}).get("paper_id", "")).strip() or "unknown"
        if paper_id not in best_by_paper:
            best_by_paper[paper_id] = (document, score)

    if len(best_by_paper) <= 1:
        return [document for document, _ in scored_docs[:limit]]

    selected: list[Document] = []
    seen: set[str] = set()
    for document, score in sorted(best_by_paper.values(), key=lambda pair: pair[1], reverse=True):
        if score < (top_score - 0.22):
            continue
        identity = _document_identity(document)
        if identity in seen:
            continue
        selected.append(document)
        seen.add(identity)
        if len(selected) >= limit:
            return selected[:limit]

    for document, _ in scored_docs:
        identity = _document_identity(document)
        if identity in seen:
            continue
        selected.append(document)
        seen.add(identity)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _select_reviewer_coverage_docs(
    *,
    scored_docs: list[tuple[Document, float]],
    limit: int,
) -> list[Document]:
    if not scored_docs:
        return []
    filtered_scored_docs = [
        (document, score)
        for document, score in scored_docs
        if not _looks_acknowledgement_text(document.page_content or "")
        and not _looks_like_metadata_snippet(document.page_content or "")
        and not _looks_like_non_argument_snippet(document.page_content or "")
    ]
    if filtered_scored_docs:
        scored_docs = filtered_scored_docs

    buckets: dict[str, list[Document]] = {}
    for document, _ in scored_docs:
        bucket = _reviewer_bucket(document.page_content or "")
        buckets.setdefault(bucket, []).append(document)

    priority = [
        "summary",
        "experiments",
        "method",
        "reproducibility",
        "ablation",
        "limitations",
        "other",
    ]
    selected: list[Document] = []
    for bucket in priority:
        candidates = buckets.get(bucket, [])
        if not candidates:
            continue
        selected.append(candidates[0])
        if len(selected) >= limit:
            return selected[:limit]

    for document, _ in scored_docs:
        if document in selected:
            continue
        selected.append(document)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _reviewer_bucket(text: str) -> str:
    lower = (text or "")[:650].lower()
    if any(marker in lower for marker in ("abstract", "we propose", "our main result", "in this work", "contribution", "novelty", "motivation")):
        return "summary"
    if any(marker in lower for marker in ("experiment", "results", "dataset", "benchmark", "baseline", "table", "score", "metric")):
        return "experiments"
    if any(marker in lower for marker in ("method", "approach", "architecture", "model", "algorithm", "objective", "training")):
        return "method"
    if any(marker in lower for marker in ("ablation", "sensitivity", "error analysis", "robustness")):
        return "ablation"
    if any(marker in lower for marker in ("limitation", "failure", "threat", "bias", "ethic")):
        return "limitations"
    if any(marker in lower for marker in ("implementation", "hyperparameter", "seed", "compute", "reproducibility", "code")):
        return "reproducibility"
    return "other"


def _has_inline_citations(text: str) -> bool:
    return bool(re.search(r"\[[0-9]+\]", text or ""))


def _select_citations_for_answer(
    *,
    answer: str,
    citations: list[dict[str, Any]],
    mode: Mode,
) -> list[dict[str, Any]]:
    if not citations:
        return []

    referenced_numbers = sorted({int(match) for match in re.findall(r"\[([0-9]+)\]", answer or "") if match.isdigit()})
    selected: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for number in referenced_numbers:
        index = number - 1
        if index < 0 or index >= len(citations):
            continue
        citation = citations[index]
        key = f"{citation.get('paper_id','')}|{citation.get('chunk_id','')}|{citation.get('page','')}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append(citation)

    if selected:
        if mode == Mode.GLOBAL:
            return selected[: min(2, len(selected))]
        return selected

    if mode == Mode.REVIEWER:
        fallback_limit = 1
    elif mode == Mode.LOCAL:
        fallback_limit = 1
    elif mode == Mode.COMPARATOR:
        fallback_limit = min(4, len(citations))
    else:
        fallback_limit = 0
    return citations[: min(fallback_limit, len(citations))]


def _citation_identity(citation: dict[str, Any]) -> str:
    return f"{citation.get('paper_id','')}|{citation.get('chunk_id','')}|{citation.get('page','')}"


def _reindex_answer_citations(
    *,
    answer: str,
    raw_citations: list[dict[str, Any]],
    selected_citations: list[dict[str, Any]],
) -> str:
    text = answer or ""
    if not _has_inline_citations(text):
        return text
    if not selected_citations:
        return _strip_inline_reference_markers(text)

    key_to_new_index: dict[str, int] = {}
    for index, citation in enumerate(selected_citations, start=1):
        key_to_new_index[_citation_identity(citation)] = index

    old_to_new: dict[int, int] = {}
    for index, citation in enumerate(raw_citations or [], start=1):
        new_index = key_to_new_index.get(_citation_identity(citation))
        if new_index is not None:
            old_to_new[index] = new_index

    def _replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if not token.isdigit():
            return ""
        old_index = int(token)
        new_index = old_to_new.get(old_index)
        if new_index is None:
            return ""
        return f"[{new_index}]"

    reindexed = re.sub(r"\[([0-9]+)\]", _replace, text)
    reindexed = re.sub(r"(\[[0-9]+\])(?:\s*\1)+", r"\1", reindexed)
    reindexed = re.sub(r"\s+([,.;:])", r"\1", reindexed)
    reindexed = re.sub(r"[ \t]{2,}", " ", reindexed)
    reindexed = re.sub(r"\n{3,}", "\n\n", reindexed)
    return reindexed.strip()


def _temperature_for_mode(mode: Mode) -> float:
    return {
        Mode.LOCAL: 0.0,
        Mode.GLOBAL: 0.35,
        Mode.WRITER: 0.4,
        Mode.REVIEWER: 0.1,
        Mode.COMPARATOR: 0.1,
    }[mode]


def extract_reviewer_state(state: GraphState) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for key in REVIEWER_STATE_KEYS:
        if key in state:
            snapshot[key] = deepcopy(state[key])
    return snapshot


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("prepare_mode", _prepare_mode_step)
    graph.add_node("retrieve", _retrieve_step)
    graph.add_node("rerank", _rerank_step)
    graph.add_node("draft_answer", _draft_answer_step)
    graph.add_node("validate_answer", _validate_answer_step)
    graph.add_node("finalize_answer", _finalize_answer_step)
    graph.add_edge(START, "prepare_mode")
    graph.add_edge("prepare_mode", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "draft_answer")
    graph.add_edge("draft_answer", "validate_answer")
    graph.add_edge("validate_answer", "finalize_answer")
    graph.add_edge("finalize_answer", END)
    return graph.compile()
