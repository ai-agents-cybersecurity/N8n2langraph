# N8n2LangGraph Converter

A Python 3.11 CLI tool that converts n8n workflow JSON exports into standalone LangGraph + LangChain workflows that call OpenAI's Responses API (model `gpt-5`). The converter parses the workflow graph, normalizes nodes/edges, and emits a runnable Python script that mirrors the n8n logic using `StateGraph`.

## Features
- Normalizes n8n nodes, edges, and start nodes.
- Pluggable node translators (Set, IF, HTTP Request, Code stub, OpenAI/LLM).
- Generates LangGraph `StateGraph` code with routing for conditional branches.
- All LLM calls use OpenAI Responses API with `gpt-5` via LangChain runnables.
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

## Notes
- Unsupported n8n nodes produce stub functions that raise `NotImplementedError`.
- Code nodes are preserved as comments for manual review.
- Ensure `OPENAI_API_KEY` is set in the environment when running generated scripts.
