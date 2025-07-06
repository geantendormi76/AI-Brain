# Agent Analyzer Framework

This framework is designed for the deep analysis of long-form documents (e.g., academic papers, business reports, legal contracts).

## Core Philosophy

Following the project's core principle of "enhancing small LLMs with systemic architecture," this framework will not aim to generate high-quality summaries directly. Instead, it will employ a `MemAgent`-like stream processing workflow to transform a long document into a structured, queryable index.

## Planned Workflow

1.  **Phase 1: Asynchronous Indexing (Machine Pass)**
    *   The system will process the document in chunks, using a rolling-update mechanism to extract key entities, clauses, facts, and other configurable patterns.
    *   The output will be a structured "document index" rather than a narrative summary.

2.  **Phase 2: Interactive Analysis (Human-in-the-Loop)**
    *   Users can then perform high-precision, low-latency queries against this pre-built index, enabling efficient information retrieval and analysis.

This two-phase, human-in-the-loop approach allows us to leverage the pattern-matching strengths of small LLMs while avoiding their weaknesses in long-range reasoning and high-quality generation, thus providing significant practical value for real-world document analysis tasks.