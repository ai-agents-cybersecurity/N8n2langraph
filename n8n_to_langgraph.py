"""CLI tool to convert n8n workflow JSON exports into LangGraph Python scripts."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


@dataclass
class N8nNode:
    """Represents a normalized n8n node."""

    id: str
    name: str
    type: str
    parameters: Dict[str, Any]
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class N8nEdge:
    """Represents a connection between n8n nodes."""

    source_id: str
    target_id: str
    source_output_index: int = 0
    condition_label: Optional[str] = None


@dataclass
class N8nWorkflow:
    """Normalized workflow with nodes, edges, and start nodes."""

    nodes: Dict[str, N8nNode]
    edges: List[N8nEdge]
    start_nodes: List[str]


class NodeTranslator(Protocol):
    """Interface for translating n8n nodes into generated Python functions."""

    def generate_node_function(self, node: N8nNode) -> str:
        ...

    def uses_llm(self, node: N8nNode) -> bool:
        ...


class SetNodeTranslator:
    """Translator for n8n Set nodes."""

    def generate_node_function(self, node: N8nNode) -> str:
        params = json.dumps(node.parameters, indent=4)
        code = f"""
def {sanitize_identifier(node_function_name(node))}(state: WorkflowState) -> WorkflowState:
    '''Set node translated from n8n. Adds parameter values into state data.'''
    state.setdefault("data", [])
    state.setdefault("context", {{}})
    new_entry = {{"node": "{escape_string(node.name)}", "parameters": {params}}}
    state["data"].append(new_entry)
    state["last_node"] = "{escape_string(node.name)}"
    return state
"""
        return textwrap.dedent(code)

    def uses_llm(self, node: N8nNode) -> bool:
        return False


class IfNodeTranslator:
    """Translator for n8n IF nodes."""

    def generate_node_function(self, node: N8nNode) -> str:
        conditions = node.parameters.get("conditions", {})
        string_conditions = conditions.get("string", [])
        cond_snippets = []
        for cond in string_conditions:
            value1 = escape_string(str(cond.get("value1", "")))
            value2 = escape_string(str(cond.get("value2", "")))
            operation = cond.get("operation", "equal")
            if operation == "equal":
                snippet = f"str(state.get('context', {{}}).get('{value1}', '{value1}')) == '{value2}'"
            else:
                snippet = "False"
            cond_snippets.append(snippet)
        condition_expr = " and ".join(cond_snippets) if cond_snippets else "False"
        code = f"""
def {sanitize_identifier(node_function_name(node))}(state: WorkflowState) -> WorkflowState:
    '''IF node translated from n8n. Evaluates simple string equality conditions.'''
    state.setdefault("context", {{}})
    result = bool({condition_expr})
    state["context"]["if_{node.id}"] = result
    state["last_node"] = "{escape_string(node.name)}"
    return state
"""
        return textwrap.dedent(code)

    def uses_llm(self, node: N8nNode) -> bool:
        return False


class HttpRequestTranslator:
    """Translator for n8n HTTP Request nodes."""

    def generate_node_function(self, node: N8nNode) -> str:
        params = node.parameters
        method = escape_string(params.get("method", "GET").upper())
        url = escape_string(params.get("url", ""))
        headers = params.get("options", {}).get("headers", {}) or params.get("headerParameters", {})
        headers_literal = json.dumps(headers, indent=4)
        body = params.get("body", {})
        body_literal = json.dumps(body, indent=4)
        code = f"""
def {sanitize_identifier(node_function_name(node))}(state: WorkflowState) -> WorkflowState:
    '''HTTP Request node translated from n8n.'''
    import requests

    state.setdefault("data", [])
    state.setdefault("context", {{}})
    try:
        response = requests.request(
            method="{method}",
            url="{url}",
            headers={headers_literal} or None,
            json={body_literal} or None,
            timeout=30,
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        payload = response.json() if "json" in content_type else response.text
        state["data"].append({{"node": "{escape_string(node.name)}", "response": payload}})
    except Exception as exc:  # pragma: no cover - runtime safety
        state["data"].append({{"node": "{escape_string(node.name)}", "error": str(exc)}})
    state["last_node"] = "{escape_string(node.name)}"
    return state
"""
        return textwrap.dedent(code)

    def uses_llm(self, node: N8nNode) -> bool:
        return False


class CodeNodeTranslator:
    """Translator for n8n Code nodes."""

    def generate_node_function(self, node: N8nNode) -> str:
        function_code = node.parameters.get("functionCode", "")
        commented_code = "\n".join(f"    # {line}" for line in function_code.splitlines())
        code = f"""
def {sanitize_identifier(node_function_name(node))}(state: WorkflowState) -> WorkflowState:
    '''Code node translated from n8n. Manual review may be required.'''
    state.setdefault("context", {{}})
    state.setdefault("data", [])
    # Original n8n functionCode preserved for reference:
{commented_code if commented_code else '    # (no code provided)'}
    # You may execute custom logic here. For safety, this is left as a stub.
    state["data"].append({{"node": "{escape_string(node.name)}", "note": "Code execution not implemented. Manual review required."}})
    state["last_node"] = "{escape_string(node.name)}"
    return state
"""
        return textwrap.dedent(code)

    def uses_llm(self, node: N8nNode) -> bool:
        return False


class OpenAiNodeTranslator:
    """Translator for n8n OpenAI nodes using Responses API."""

    def generate_node_function(self, node: N8nNode) -> str:
        prompt = escape_string(node.parameters.get("prompt", ""))
        system_prompt = escape_string(node.parameters.get("systemMessage", ""))
        code = f"""
def {sanitize_identifier(node_function_name(node))}(state: WorkflowState) -> WorkflowState:
    '''OpenAI node translated from n8n using Responses API with gpt-5.'''
    state.setdefault("data", [])
    state.setdefault("context", {{}})
    response_text = call_llm(prompt="{prompt}", system="{system_prompt}" or None)
    state["data"].append({{"node": "{escape_string(node.name)}", "response": response_text}})
    state["context"]["openai_{node.id}"] = response_text
    state["last_node"] = "{escape_string(node.name)}"
    return state
"""
        return textwrap.dedent(code)

    def uses_llm(self, node: N8nNode) -> bool:
        return True


NODE_TRANSLATORS: Dict[str, NodeTranslator] = {
    "n8n-nodes-base.set": SetNodeTranslator(),
    "n8n-nodes-base.if": IfNodeTranslator(),
    "n8n-nodes-base.httpRequest": HttpRequestTranslator(),
    "n8n-nodes-base.code": CodeNodeTranslator(),
    "n8n-nodes-base.openAi": OpenAiNodeTranslator(),
}


DEFAULT_TRANSLATOR = CodeNodeTranslator()


def sanitize_identifier(name: str) -> str:
    """Return a valid Python identifier derived from the provided string."""

    sanitized = re.sub(r"\W+", "_", name)
    sanitized = sanitized.strip("_")
    if sanitized and sanitized[0].isdigit():
        sanitized = f"n_{sanitized}"
    return sanitized or "node"


def escape_string(text: str) -> str:
    """Escape quotes for safe embedding in generated code."""

    return text.replace("\\", "\\\\").replace('"', '\\"')


def node_function_name(node: N8nNode) -> str:
    """Generate a function name for a node."""

    return f"node_{sanitize_identifier(node.name or node.id)}"


def load_workflow(path: str) -> N8nWorkflow:
    """Load and normalize an n8n workflow from JSON."""

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_nodes = data.get("nodes") or []
    if not raw_nodes:
        raise ValueError("Workflow JSON contains no nodes.")

    nodes: Dict[str, N8nNode] = {}
    id_by_name: Dict[str, str] = {}
    for raw_node in raw_nodes:
        node_id = raw_node.get("id") or raw_node.get("name")
        if not node_id:
            raise ValueError(f"Node missing id and name: {raw_node}")
        name = raw_node.get("name") or node_id
        node_type = raw_node.get("type", "unknown")
        parameters = raw_node.get("parameters", {})
        node = N8nNode(id=str(node_id), name=str(name), type=str(node_type), parameters=parameters, raw=raw_node)
        nodes[node.id] = node
        id_by_name[node.name] = node.id

    edges: List[N8nEdge] = []
    connections = data.get("connections", {}) or {}
    for source_name, conn_info in connections.items():
        source_id = id_by_name.get(source_name, source_name)
        main_outputs = conn_info.get("main") or []
        for output_index, targets in enumerate(main_outputs):
            for target in targets or []:
                target_name = target.get("node")
                if target_name is None:
                    continue
                target_id = id_by_name.get(target_name, target_name)
                condition_label = target.get("type") if target.get("type") not in (None, "main") else None
                edge = N8nEdge(
                    source_id=str(source_id),
                    target_id=str(target_id),
                    source_output_index=int(target.get("index", output_index)),
                    condition_label=condition_label,
                )
                edges.append(edge)

    incoming: Dict[str, int] = {node_id: 0 for node_id in nodes}
    for edge in edges:
        incoming[edge.target_id] = incoming.get(edge.target_id, 0) + 1

    start_nodes: List[str] = []
    for node_id, node in nodes.items():
        if incoming.get(node_id, 0) == 0 or "trigger" in node.type.lower() or "webhook" in node.type.lower():
            start_nodes.append(node_id)
    if not start_nodes:
        # fall back to first node
        start_nodes = [next(iter(nodes.keys()))]

    return N8nWorkflow(nodes=nodes, edges=edges, start_nodes=start_nodes)


def generate_node_function(node: N8nNode) -> Tuple[str, bool]:
    """Generate Python code for a node function and return (code, uses_llm)."""

    translator = NODE_TRANSLATORS.get(node.type, DEFAULT_TRANSLATOR)
    if translator is DEFAULT_TRANSLATOR and node.type not in NODE_TRANSLATORS:
        logger.warning("Unsupported node type '%s'. Generating stub Code node.", node.type)
    code = translator.generate_node_function(node)
    return code, translator.uses_llm(node)


def generate_routing_function(node_id: str, edges: List[N8nEdge]) -> str:
    """Generate a routing function for conditional edges from a node."""

    mapping_lines = []
    for edge in edges:
        label = edge.condition_label or f"out_{edge.source_output_index}"
        mapping_lines.append(f"    if label == '{label}':\n        return '{edge.target_id}'")
    mapping_body = "\n".join(mapping_lines)
    code = f"""
def route_from_{sanitize_identifier(node_id)}(state: WorkflowState) -> str:
    '''Routing function for node {node_id}. Update the label as needed.'''
    label = state.get("context", {{}}).get("route_{node_id}")
    if label is None:
        # Default to first available label
        label = "{edges[0].condition_label or f'out_{edges[0].source_output_index}'}"
    {mapping_body}
    return '{edges[0].target_id}'
"""
    return textwrap.dedent(code)


def generate_python(workflow: N8nWorkflow) -> str:
    """Generate the Python script implementing the workflow using LangGraph."""

    lines: List[str] = []
    lines.append("from __future__ import annotations")
    lines.append("import argparse")
    lines.append("import json")
    lines.append("from typing import TypedDict, Any, Dict, List")
    lines.append("from openai import OpenAI")
    lines.append("from langgraph.graph import StateGraph")
    lines.append("import os")
    lines.append("")
    lines.append("class WorkflowState(TypedDict, total=False):")
    lines.append("    data: List[Dict[str, Any]]")
    lines.append("    context: Dict[str, Any]")
    lines.append("    last_node: str")
    lines.append("")
    lines.append("client = OpenAI()")
    lines.append("")
    lines.append("def call_llm(prompt: str, system: str | None = None) -> str:")
    lines.append("    \"\"\"Call OpenAI Responses API with model gpt-5.\"\"\"")
    lines.append("    messages = []")
    lines.append("    if system:\n        messages.append({\"role\": \"system\", \"content\": system})")
    lines.append("    messages.append({\"role\": \"user\", \"content\": prompt})")
    lines.append("    response = client.responses.create(model=\"gpt-5\", input=messages)")
    lines.append("    choice = response.output[0]")
    lines.append("    return choice.content[0].text")
    lines.append("")

    sorted_nodes = sorted(workflow.nodes.values(), key=lambda n: n.name)
    for node in sorted_nodes:
        func_code, uses_llm = generate_node_function(node)
        lines.append(func_code)

    # Edges grouping
    edges_by_source: Dict[str, List[N8nEdge]] = {}
    for edge in workflow.edges:
        edges_by_source.setdefault(edge.source_id, []).append(edge)

    # Routing functions for conditional edges
    for source_id, edges in edges_by_source.items():
        if len(edges) > 1:
            lines.append(generate_routing_function(source_id, edges))

    # build_graph definition
    lines.append("def build_graph():")
    lines.append("    graph = StateGraph(WorkflowState)")
    for node in sorted_nodes:
        lines.append(f"    graph.add_node('{node.id}', {sanitize_identifier(node_function_name(node))})")

    for start_id in workflow.start_nodes:
        lines.append(f"    graph.set_entry_point('{start_id}')")

    for source_id, edges in edges_by_source.items():
        if len(edges) == 1:
            target = edges[0].target_id
            lines.append(f"    graph.add_edge('{source_id}', '{target}')")
        else:
            label_map = {edge.condition_label or f"out_{edge.source_output_index}": edge.target_id for edge in edges}
            lines.append(f"    graph.add_conditional_edges('{source_id}', route_from_{sanitize_identifier(source_id)}, {label_map})")

    lines.append("    return graph.compile()")
    lines.append("")
    lines.append("def run_workflow(initial_state: Dict[str, Any] | None = None) -> Dict[str, Any]:")
    lines.append("    state = initial_state or {'data': [], 'context': {}}");
    lines.append("    app = build_graph()")
    lines.append("    result = app.invoke(state)")
    lines.append("    return result")
    lines.append("")
    lines.append("def main():")
    lines.append("    parser = argparse.ArgumentParser(description='Run generated LangGraph workflow.')")
    lines.append("    parser.add_argument('--input-json', help='Initial state as JSON string', default=None)")
    lines.append("    args = parser.parse_args()")
    lines.append("    initial_state = json.loads(args.input_json) if args.input_json else None")
    lines.append("    final_state = run_workflow(initial_state)")
    lines.append("    print(json.dumps(final_state, indent=2))")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    main()")

    return "\n".join(lines)


def prettify_code(path: Path) -> None:
    """Optional code formatting using black when available."""

    try:
        subprocess.run([sys.executable, "-m", "black", str(path)], check=True)
    except FileNotFoundError:
        logger.info("Black not installed; skipping formatting.")
    except subprocess.CalledProcessError as exc:  # pragma: no cover - external command
        logger.warning("Black formatting failed: %s", exc)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the converter."""

    parser = argparse.ArgumentParser(description="Convert n8n workflow JSON to LangGraph Python script")
    parser.add_argument("--input", "-i", required=True, help="Path to n8n workflow JSON file")
    parser.add_argument("--output", "-o", required=True, help="Path to save generated Python script")
    parser.add_argument("--pretty", action="store_true", help="Format generated code with black if available")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args(argv)


def write_output(code: str, output_path: str) -> None:
    """Write generated code to output path."""

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(code, encoding="utf-8")


def summarize(workflow: N8nWorkflow, output_path: str) -> None:
    """Print summary of conversion."""

    print(f"Converted {len(workflow.nodes)} nodes")
    print(f"Created {len(workflow.edges)} edges")
    print(f"Generated file: {output_path}")


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the converter CLI."""

    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s:%(message)s")

    try:
        workflow = load_workflow(args.input)
    except FileNotFoundError:
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed to load workflow: %s", exc)
        sys.exit(1)

    try:
        code = generate_python(workflow)
    except Exception as exc:
        logger.error("Failed to generate Python code: %s", exc)
        sys.exit(1)

    write_output(code, args.output)

    if args.pretty:
        prettify_code(Path(args.output))

    summarize(workflow, args.output)


if __name__ == "__main__":
    main()
