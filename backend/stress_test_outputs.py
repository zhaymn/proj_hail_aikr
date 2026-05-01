from __future__ import annotations

import json
import re
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parent
SRC_ROOT = BACKEND_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from research_agent.config import get_settings
from research_agent.runtime import ResearchAgentRuntime
from research_agent.schemas import ChatRequest, HistoryMessage, Mode


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(token) > 2}


def _jaccard(a: str, b: str) -> float:
    a_tokens = _tokenize(a)
    b_tokens = _tokenize(b)
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))


def _history_message(role: str, content: str) -> HistoryMessage:
    return HistoryMessage(role=role, content=content)


def _chat(
    *,
    runtime: ResearchAgentRuntime,
    session_id: str,
    mode: Mode,
    message: str,
    paper_ids: list[str] | None = None,
    review_paper_id: str | None = None,
    intervention_mode: str | None = None,
    history: list[HistoryMessage] | None = None,
) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    response = runtime.chat(
        ChatRequest(
            session_id=session_id,
            mode=mode,
            message=message,
            paper_ids=paper_ids or [],
            review_paper_id=review_paper_id,
            intervention_mode=intervention_mode,
            history=history or [],
        )
    )
    elapsed = time.perf_counter() - start
    payload = {
        "answer": response.answer,
        "citations": [item.model_dump() for item in response.citations],
        "debug": response.debug or {},
    }
    return payload, elapsed


def _ensure_second_paper(runtime: ResearchAgentRuntime) -> None:
    papers = runtime.list_papers().papers
    if len(papers) >= 2:
        return
    candidate_paths = [
        Path(r"C:\Users\R Nishanth Reddy\Downloads\EEGMoE_A_Domain-Decoupled_Mixture-of-Experts_Model_for_Self-Supervised_EEG_Representation_Learning.pdf"),
        Path(r"C:\Users\R Nishanth Reddy\Downloads\chiplay26a-sub1417-i26.pdf"),
    ]
    to_upload: list[tuple[str, bytes]] = []
    known_names = {paper.filename for paper in papers}
    for path in candidate_paths:
        if not path.exists():
            continue
        if path.name in known_names:
            continue
        to_upload.append((path.name, path.read_bytes()))
        break
    if to_upload:
        runtime.upload_papers(to_upload)


def run_reviewer_stress(runtime: ResearchAgentRuntime, paper_id: str) -> dict[str, Any]:
    sid = f"stress-reviewer-{uuid.uuid4()}"
    history: list[HistoryMessage] = []
    max_calls = 18
    prompts = ["[Start Debate] Focus lens: Full Review"]

    turn_counts: list[int] = []
    vector_turn_progression: list[dict[str, Any]] = []
    latencies: list[float] = []
    round_event_counts: list[int] = []
    dual_speaker_rounds = 0
    fallback_rounds = 0
    skeptic_turns: list[str] = []
    advocate_turns: list[str] = []
    final_report_ready = False
    last_debug: dict[str, Any] = {}
    active_vectors: list[str] = []
    vector_switches = 0

    call_index = 0
    while call_index < max_calls:
        prompt = prompts[call_index] if call_index < len(prompts) else "next"
        payload, elapsed = _chat(
            runtime=runtime,
            session_id=sid,
            mode=Mode.REVIEWER,
            message=prompt,
            review_paper_id=paper_id,
            intervention_mode="ask",
            history=history[-8:],
        )
        answer = payload["answer"]
        debug = payload["debug"]
        latencies.append(elapsed)
        turn_counts.append(int(debug.get("turn_count", 0) or 0))
        last_debug = debug
        vector_id = str(debug.get("active_vector_id", "")).strip() or "unknown"
        active_vectors.append(vector_id)
        if len(active_vectors) >= 2 and active_vectors[-1] != active_vectors[-2]:
            vector_switches += 1
        vector_turn_progression.append(
            {
                "call": call_index + 1,
                "vector": vector_id,
                "turn": int(debug.get("turn_count", 0) or 0),
                "resolution": str(debug.get("resolution", "")),
                "round_event_count": int(debug.get("round_event_count", 0) or 0),
            }
        )

        if debug.get("model_fallback"):
            fallback_rounds += 1

        events = debug.get("round_events", [])
        round_event_counts.append(len(events) if isinstance(events, list) else 0)
        speakers = {str(item.get("speaker", "")).strip().lower() for item in events if isinstance(item, dict)}
        if "skeptic" in speakers and "advocate" in speakers:
            dual_speaker_rounds += 1
        for event in events:
            if not isinstance(event, dict):
                continue
            speaker = str(event.get("speaker", "")).strip().lower()
            content = str(event.get("content", "")).strip()
            if speaker == "skeptic" and content:
                skeptic_turns.append(content)
            if speaker == "advocate" and content:
                advocate_turns.append(content)

        if debug.get("final_report_ready"):
            final_report_ready = True

        history.append(_history_message("user", prompt))
        history.append(_history_message("assistant", answer[:4000]))
        if final_report_ready:
            break
        call_index += 1

    skeptic_repetition = 0.0
    if len(skeptic_turns) >= 2:
        skeptic_repetition = statistics.mean(
            _jaccard(skeptic_turns[i - 1], skeptic_turns[i]) for i in range(1, len(skeptic_turns))
        )
    advocate_repetition = 0.0
    if len(advocate_turns) >= 2:
        advocate_repetition = statistics.mean(
            _jaccard(advocate_turns[i - 1], advocate_turns[i]) for i in range(1, len(advocate_turns))
        )

    final_report = last_debug.get("final_report") if isinstance(last_debug, dict) else {}
    final_report_fields = {
        "overview": bool(str(final_report.get("overview", "")).strip()) if isinstance(final_report, dict) else False,
        "agreements": bool(final_report.get("agreements")) if isinstance(final_report, dict) else False,
        "disagreements": bool(final_report.get("disagreements")) if isinstance(final_report, dict) else False,
        "common_points": bool(final_report.get("common_points")) if isinstance(final_report, dict) else False,
        "skeptic_conclusion": bool(str(final_report.get("skeptic_conclusion", "")).strip())
        if isinstance(final_report, dict)
        else False,
        "advocate_conclusion": bool(str(final_report.get("advocate_conclusion", "")).strip())
        if isinstance(final_report, dict)
        else False,
        "joint_conclusion": bool(str(final_report.get("joint_conclusion", "")).strip())
        if isinstance(final_report, dict)
        else False,
        "final_suggestions": bool(final_report.get("final_suggestions")) if isinstance(final_report, dict) else False,
        "final_decision": bool(str(final_report.get("final_decision", "")).strip())
        if isinstance(final_report, dict)
        else False,
    }

    return {
        "calls": len(latencies),
        "max_calls_configured": max_calls,
        "avg_latency_s": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "p95_latency_s": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 3) if latencies else 0.0,
        "turn_progression": turn_counts,
        "vector_turn_progression": vector_turn_progression,
        "vector_switches": vector_switches,
        "resolved_vectors": len(last_debug.get("vector_verdicts", {})) if isinstance(last_debug, dict) else 0,
        "avg_round_events": round(statistics.mean(round_event_counts), 2) if round_event_counts else 0.0,
        "dual_speaker_rounds": dual_speaker_rounds,
        "fallback_rounds": fallback_rounds,
        "skeptic_avg_repetition_jaccard": round(skeptic_repetition, 3),
        "advocate_avg_repetition_jaccard": round(advocate_repetition, 3),
        "final_report_ready": final_report_ready,
        "final_report_fields": final_report_fields,
    }


def run_local_stress(runtime: ResearchAgentRuntime, paper_id: str) -> dict[str, Any]:
    sid = f"stress-local-{uuid.uuid4()}"
    history: list[HistoryMessage] = []
    queries = [
        "what version of easy ocr is used",
        "what was precision and recall",
        "how many participants were in the study",
        "which game was used",
        "summarize the core math behind their temporal distribution modeling",
        "who are the authors and affiliation",
    ]
    latencies: list[float] = []
    fallback_count = 0
    citation_count = 0
    numeric_hits = 0
    math_grounded = 0

    for query in queries:
        payload, elapsed = _chat(
            runtime=runtime,
            session_id=sid,
            mode=Mode.LOCAL,
            message=query,
            paper_ids=[paper_id],
            history=history[-8:],
        )
        latencies.append(elapsed)
        answer = str(payload["answer"])
        debug = payload["debug"]
        citations = payload["citations"]
        if debug.get("model_fallback") or "retrieval-only fallback" in answer.lower():
            fallback_count += 1
        citation_count += len(citations)
        if re.search(r"\b\d+(\.\d+)?\b", answer):
            numeric_hits += 1
        if "distribution" in answer.lower() or "ex-gaussian" in answer.lower() or "lognormal" in answer.lower():
            math_grounded += 1
        history.append(_history_message("user", query))
        history.append(_history_message("assistant", answer[:3000]))

    return {
        "calls": len(queries),
        "avg_latency_s": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "fallback_rounds": fallback_count,
        "avg_citations_per_answer": round(citation_count / max(1, len(queries)), 2),
        "answers_with_numeric_signal": numeric_hits,
        "math_grounded_answers": math_grounded,
    }


def run_global_stress(runtime: ResearchAgentRuntime, paper_id: str) -> dict[str, Any]:
    sid = f"stress-global-{uuid.uuid4()}"
    history: list[HistoryMessage] = []
    queries = [
        "what is a bird",
        "tell me something about the authors in this paper",
        "suggest 3 related papers and why",
    ]
    latencies: list[float] = []
    unrestricted_ok = 0
    fallback_count = 0
    with_citations = 0

    for query in queries:
        payload, elapsed = _chat(
            runtime=runtime,
            session_id=sid,
            mode=Mode.GLOBAL,
            message=query,
            paper_ids=[paper_id],
            history=history[-8:],
        )
        latencies.append(elapsed)
        answer = str(payload["answer"])
        debug = payload["debug"]
        citations = payload["citations"]
        if debug.get("model_fallback") or "retrieval-only fallback" in answer.lower():
            fallback_count += 1
        if "this information is not in your uploaded papers" not in answer.lower():
            unrestricted_ok += 1
        if citations:
            with_citations += 1
        history.append(_history_message("user", query))
        history.append(_history_message("assistant", answer[:3000]))

    return {
        "calls": len(queries),
        "avg_latency_s": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "fallback_rounds": fallback_count,
        "non_restricted_answers": unrestricted_ok,
        "answers_with_citations": with_citations,
    }


def run_comparator_smoke(runtime: ResearchAgentRuntime, paper_ids: list[str]) -> dict[str, Any]:
    if len(paper_ids) < 2:
        return {
            "skipped": True,
            "reason": "Need at least 2 indexed papers for comparator smoke.",
        }
    sid = f"stress-comparator-{uuid.uuid4()}"
    payload, elapsed = _chat(
        runtime=runtime,
        session_id=sid,
        mode=Mode.COMPARATOR,
        message="Run a full comparator pass with claim matrix, conflict map, benchmark verdict matrix, and decision by use case.",
        paper_ids=paper_ids[:3],
        history=[],
    )
    answer = str(payload["answer"])
    debug = payload["debug"]
    section_hits = sum(
        1
        for marker in (
            "claim matrix",
            "conflict map",
            "decision",
            "benchmark",
        )
        if marker in answer.lower()
    )
    return {
        "skipped": False,
        "latency_s": round(elapsed, 3),
        "fallback": bool(debug.get("model_fallback") or "retrieval-only fallback" in answer.lower()),
        "section_markers_hit": section_hits,
        "citation_count": len(payload["citations"]),
    }


def main() -> None:
    settings = get_settings()
    runtime = ResearchAgentRuntime(settings)
    _ensure_second_paper(runtime)

    papers = runtime.list_papers().papers
    if not papers:
        raise SystemExit("No indexed papers found. Upload at least one PDF first.")

    primary = papers[0]
    comparator_ids = [paper.paper_id for paper in papers[:3]]

    report = {
        "environment": {
            "indexed_papers": len(papers),
            "llm_available": runtime.health().llm_available,
            "primary_paper": {"paper_id": primary.paper_id, "filename": primary.filename},
            "paper_filenames": [paper.filename for paper in papers[:3]],
        },
        "reviewer": run_reviewer_stress(runtime, primary.paper_id),
        "local": run_local_stress(runtime, primary.paper_id),
        "global": run_global_stress(runtime, primary.paper_id),
        "comparator": run_comparator_smoke(runtime, comparator_ids),
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
