# LangGraph State Graph

This file documents the execution order and mode-specific branch behavior in:

- `backend/src/research_agent/graph/builder.py`
- `backend/src/research_agent/graph/state.py`

## 1) Shared Graph

```mermaid
flowchart TD
  START([START]) --> PREP[prepare_mode]
  PREP --> RETR[retrieve]
  RETR --> RERANK[rerank]
  RERANK --> DRAFT[draft_answer]
  DRAFT --> VALID[validate_answer]
  VALID --> FINAL[finalize_answer]
  FINAL --> END([END])

  DRAFT -. mode=reviewer .-> REV[Reviewer engine branch]
  DRAFT -. mode=comparator .-> COMP[Comparator branch]
```

## 2) Reviewer Branch Graph

```mermaid
flowchart TD
  R0[Load reviewer session state] --> R1[Normalize or generate attack vectors]
  R1 --> R2[Select active vector]
  R2 --> R3{Score request?}

  R3 -->|Yes| R4[Return scorecard]
  R3 -->|No| R5[Debate loop: skeptic/advocate turns]

  R5 --> R6{Synthesis condition met?}
  R6 -->|No| R5
  R6 -->|Yes| R7[Evidence-only judge verdict]
  R7 --> R8[Rewrite Compiler synthesis]
  R8 --> R9[Update vector verdict/judgment/report]
  R9 --> R10{More vectors remain?}
  R10 -->|Yes| R5
  R10 -->|No| R11[Build final_report]

  R4 --> R12[Render reviewer markdown]
  R11 --> R12
  R12 --> R13[Persist reviewer state keys]
```

## 3) Comparator Branch Graph

```mermaid
flowchart TD
  C0[Input: paper_ids and user comparison message] --> C1[Retrieve top chunks per paper]
  C1 --> C2[Merge + dedupe]
  C2 --> C3[Rerank with comparator focus]
  C3 --> C4{Enough papers and evidence?}
  C4 -->|No| C5[Structured comparator fallback]
  C4 -->|Yes| C6[Generate comparator response]
  C6 --> C7[Enforce comparator section structure]
  C5 --> C8[validate_answer]
  C7 --> C8
  C8 --> C9[finalize_answer + citation selection]
```

## 4) Exact Node Order

1. START: runtime invokes compiled graph with request payload.
2. `prepare_mode`: mode instruction + initial debug metadata.
3. `retrieve`: mode-specific retrieval strategy.
4. `rerank`: scoring and balanced selection.
5. `draft_answer`: mode branch execution (reviewer/comparator/local/global/writer).
6. `validate_answer`: validation model pass or branch-specific bypass.
7. `finalize_answer`: final answer + citations + debug.
8. END: return to runtime/API.

## 5) GraphState Fields

Core fields:

- `session_id`, `mode`, `message`, `paper_ids`, `review_paper_id`, `history`
- `mode_instructions`, `retrieved_documents`
- `draft_answer`, `validated_answer`, `validation_issues`
- `answer`, `citations`, `debug`

Reviewer fields:

- `attack_vectors`, `active_vector_id`, `vectors_remaining`
- `debate_history`, `debate_summary`
- `skeptic_position`, `advocate_position`, `resolution`, `turn_count`, `next_speaker`
- `syntheses`, `vector_verdicts`, `vector_judgments`, `vector_reports`, `final_report`
- `intervention_mode`

## 6) Persistence Notes

- Reviewer session keys are extracted with `extract_reviewer_state()` and stored in runtime memory.
- Reviewer state is scoped by `session_id::paper_id`.
- Clearing/deleting papers removes linked reviewer state.
