# Klareco: AST-Native RAG Pipeline

## Guiding Principles

This document outlines the architecture for Klareco, an AI system designed to leverage the unique structural properties of the Esperanto language.

1.  **AST-First:** The Abstract Syntax Tree (AST) is the central data structure. All major components operate on the graph representation of the language.
2.  **Local-Only Generation:** The system uses a custom, locally-trained model for all text generation. It does not use any external, pre-trained LLMs for reasoning or generation.
3.  **Focused Pipeline:** The architecture is a single, streamlined pipeline for question-answering, not a multi-expert agentic system.

---

## Architecture: The AST-Native RAG Pipeline

The system processes a query in four main stages:

1.  **Structure-Aware Retrieval:** The input Esperanto question is parsed into an AST-graph. This graph's vector embedding is used to retrieve relevant documents from a corpus that has also been pre-parsed into AST-graphs. This enables a deep, structure-to-structure similarity search.

2.  **Graph-Based Reasoning:** The question graph and the retrieved context graphs are combined into a single "reasoning graph." This graph represents the full context needed to form an answer.

3.  **Structure-to-Text Generation:** The reasoning graph is fed into a custom-built **Graph-to-Sequence (Graph2Seq)** model. This model's sole job is to read the relationships in the graph and generate a coherent Esperanto sentence or paragraph that synthesizes the information.

4.  **Deparsing:** The token sequence from the generator is processed by a deparser to construct the final, well-formed Esperanto text response.

---

## Implementation Plan

This architecture will be implemented in four phases:

*   **Phase 1: Blueprint & Cleanup:** Update this `DESIGN.md`, archive old documentation, and purge all code related to the previous, more complex architecture (e.g., external LLM integrations, the orchestrator, the QA decoder).

*   **Phase 2: Self-Supervised Dataset Generation:** Create a script to automatically generate a `(Question-Graph, Context-Graph, Answer-Text)` training dataset from the existing Esperanto corpus. This involves building a rule-based Text-to-Question (T2Q) system to create realistic training examples.

*   **Phase 3: Building and Training the Graph2Seq Model:** Implement the Graph2Seq model architecture (GNN Encoder + RNN/Transformer Decoder) and the script to train it on the data from Phase 2.

*   **Phase 4: Integration & Evaluation:** Integrate the full pipeline into a new user-facing script. Create a hold-out test set and an evaluation script to measure the system's performance.
