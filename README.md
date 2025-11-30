# N8n2LangGraph Converter

A Python 3.11 CLI tool that converts n8n workflow JSON exports into standalone LangGraph + LangChain workflows that call OpenAI's Responses API (model `gpt-5`). The converter parses the workflow graph, normalizes nodes/edges, and emits a runnable Python script that mirrors the n8n logic using `StateGraph`.

## Features
- Normalizes n8n nodes, edges, and start nodes.
- Pluggable node translators (Set, IF, HTTP Request, Code stub, OpenAI/LLM).
- Generates LangGraph `StateGraph` code with routing for conditional branches.
- All LLM calls use OpenAI Responses API with configurable models (defaults to `gpt-5` and can be set to `gpt-5.1-codex-max`) via LangChain runnables.
- Optional reflection and lightweight agentic loop for OpenAI nodes in generated workflows.
- Optional LLM-assisted conversion to patch unsupported nodes and refine prompts during generation.
- Optional code formatting via `black` when installed.

## Quickstart
1. Export an n8n workflow JSON file.
2. Run the converter:
   ```bash
   python n8n_to_langgraph.py -i path/to/workflow.json -o generated_workflow.py --pretty
   ```
3. Review the summary output (nodes, edges, output path).
4. Run the generated workflow script:
   ```bash
   python generated_workflow.py --input-json '{"data": [], "context": {}}'
   ```

## Requirements
- Python 3.11+
- `openai`, `langgraph`, `langchain-core`, `requests` (runtime for generated script)
- `black` (optional, for formatting)

## Advanced options
- `--llm-model`: override the model used in generated scripts (e.g., `gpt-5.1-codex-max`).
- `--enable-reflection` and `--reflection-steps`: wrap generated LLM calls in a reflection loop for self-critique.
- `--enable-agent`: toggle a lightweight agentic helper that re-prompts the LLM with state context when translating OpenAI nodes.
- `--enable-llm-assist` and `--llm-assist-model`: use the converter's LLM helper to infer code for unsupported nodes and rewrite prompts for clarity.

## Conversion flow

```mermaid
flowchart TD
    A[Start: n8n workflow JSON] --> B[Parse nodes and edges]
    B --> C[Normalize graph structure
        - start nodes
        - node metadata
        - edge routing]
    C --> D[Translate nodes via pluggable translators
        - Set / IF / HTTP Request / Code stubs
        - OpenAI & LLM nodes]
    D --> E[Apply optional helpers
        - Reflection loop
        - Agentic re-prompting
        - LLM assist for unsupported nodes]
    E --> F[Generate LangGraph StateGraph code
        - state schema
        - routes for branches
        - OpenAI Responses API calls]
    F --> G[Format output (optional: black)]
    G --> H[Emit runnable Python script]
```

## Notes
- Unsupported n8n nodes produce stub functions that raise `NotImplementedError`.
- Code nodes are preserved as comments for manual review.
- Ensure `OPENAI_API_KEY` is set in the environment when running generated scripts.
