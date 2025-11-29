# Documentation

This folder provides additional guidance for using the converter and interpreting generated workflows.

## Converter Overview
- Parses n8n JSON exports into normalized dataclasses for nodes, edges, and start nodes.
- Uses a registry of node translators to generate LangGraph node functions.
- Builds routing helpers for conditional branches and compiles a `StateGraph`.

## Generated Workflow Structure
1. Typed `WorkflowState` definition.
2. OpenAI Responses API helper `call_llm` (model `gpt-5`).
3. One function per n8n node plus routing helpers.
4. `build_graph()` to construct and compile the `StateGraph`.
5. `run_workflow()` and CLI entry for executing the compiled graph.

## Extending Translators
- Add new translators in `n8n_to_langgraph.py` and register them in `NODE_TRANSLATORS`.
- Implement `generate_node_function` to output Python code for the node.
- Use `add_conditional_edges` for multi-branch nodes to mirror n8n routing.
