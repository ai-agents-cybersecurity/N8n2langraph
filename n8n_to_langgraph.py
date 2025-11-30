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


@dataclass
class GenerationOptions:
    """Options that influence generated workflow behavior."""

    llm_model: str = "gpt-5"
    enable_reflection: bool = False
    reflection_steps: int = 1
    enable_agent: bool = False


class NodeTranslator(Protocol):
    """Interface for translating n8n nodes into generated Python functions."""

    def generate_node_function(self, node: N8nNode) -> str:
        """Return Python code implementing the node."""

    def uses_llm(self, node: N8nNode) -> bool:
        """Return True if the node requires LLM support."""


class SetNodeTranslator:
    """Translator for n8n Set nodes."""

    def generate_node_function(self, node: N8nNode) -> str:
        values = node.parameters.get("values", {})
        assignments = values.get("string", []) or []
        writes: List[str] = []
        for entry in assignments:
            name = entry.get("name")
            value = entry.get("value", "")
            if name is None:
                continue
            writes.append(
                f"    state.setdefault(\"context\", {{}})['{escape_string(str(name))}'] = '{escape_string(str(value))}'"
            )
        writes_code = "\n".join(writes) if writes else "    # No explicit set operations captured from n8n parameters"
        template = """
def {func_name}(state: WorkflowState) -> WorkflowState:
    '''Set node translated from n8n. Adds parameter values into state data and context.'''
    state.setdefault("data", [])
    state.setdefault("context", {})
{writes}
    state["data"].append({{"node": "{node_name}", "set": state.get("context", {{}})}})
    state["last_node"] = "{node_name}"
    return state
"""
        code = template.format(
            func_name=sanitize_identifier(node_function_name(node)),
            writes=writes_code,
            node_name=escape_string(node.name),
        )
        return textwrap.dedent(code)

    def uses_llm(self, node: N8nNode) -> bool:
        return False


class IfNodeTranslator:
    """Translator for n8n IF nodes."""

    def generate_node_function(self, node: N8nNode) -> str:
        conditions = node.parameters.get("conditions", {})
        string_conditions = conditions.get("string", [])
        cond_snippets: List[str] = []
        for cond in string_conditions:
            value1 = escape_string(str(cond.get("value1", "")))
            value2 = escape_string(str(cond.get("value2", "")))
            operation = cond.get("operation", "equal")
            left = f"state.get('context', {{}}).get('{value1}', '{value1}')"
            if operation == "equal":
                snippet = f"str({left}) == '{value2}'"
            elif operation == "notEqual":
                snippet = f"str({left}) != '{value2}'"
            else:
                snippet = "False"
            cond_snippets.append(snippet)
        condition_expr = " and ".join(cond_snippets) if cond_snippets else "False"
        template = """
def {func_name}(state: WorkflowState) -> WorkflowState:
    '''IF node translated from n8n. Evaluates simple string conditions.'''
    state.setdefault("context", {})
    result = bool({condition})
    state["context"]["if_{node_id}"] = result
    state["last_node"] = "{node_name}"
    return state
"""
        code = template.format(
            func_name=sanitize_identifier(node_function_name(node)),
            condition=condition_expr,
            node_id=node.id,
            node_name=escape_string(node.name),
        )
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
        body = params.get("body", {}) or params.get("jsonParameters", {}) or {}
        body_literal = json.dumps(body, indent=4)
        template = """
def {func_name}(state: WorkflowState) -> WorkflowState:
    '''HTTP Request node translated from n8n.'''
    import requests

    state.setdefault("data", [])
    state.setdefault("context", {})
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
        state["data"].append({{"node": "{node_name}", "response": payload}})
    except Exception as exc:  # pragma: no cover - runtime safety
        state["data"].append({{"node": "{node_name}", "error": str(exc)}})
    state["last_node"] = "{node_name}"
    return state
"""
        code = template.format(
            func_name=sanitize_identifier(node_function_name(node)),
            method=method,
            url=url,
            headers_literal=headers_literal,
            body_literal=body_literal,
            node_name=escape_string(node.name),
        )
        return textwrap.dedent(code)

    def uses_llm(self, node: N8nNode) -> bool:
        return False


class CodeNodeTranslator:
    """Translator for n8n Code nodes."""

    def generate_node_function(self, node: N8nNode) -> str:
        function_code = node.parameters.get("functionCode", "")
        commented_code = "\n".join(f"    # {line}" for line in function_code.splitlines())
        template = """
def {func_name}(state: WorkflowState) -> WorkflowState:
    '''Code node translated from n8n. Manual review may be required.'''
    state.setdefault("context", {})
    state.setdefault("data", [])
    # Original n8n functionCode preserved for reference:
{code_block}
    # You may execute custom logic here. For safety, this is left as a stub.
    state["data"].append({{"node": "{node_name}", "note": "Code execution not implemented. Manual review required."}})
    state["last_node"] = "{node_name}"
    return state
"""
        code = template.format(
            func_name=sanitize_identifier(node_function_name(node)),
            code_block=commented_code if commented_code else "    # (no code provided)",
            node_name=escape_string(node.name),
        )
        return textwrap.dedent(code)

    def uses_llm(self, node: N8nNode) -> bool:
        return False


class OpenAiNodeTranslator:
    """Translator for n8n OpenAI nodes using Responses API."""

    def __init__(self, options: GenerationOptions):
        self.options = options

    def generate_node_function(self, node: N8nNode) -> str:
        prompt = escape_string(node.parameters.get("prompt", ""))
        system_prompt = escape_string(node.parameters.get("systemMessage", ""))
        template = """
def {func_name}(state: WorkflowState) -> WorkflowState:
    '''OpenAI node translated from n8n using Responses API with configurable models and reflection.'''
    from langchain_core.runnables import RunnableLambda

    state.setdefault("data", [])
    state.setdefault("context", {})

    if ENABLE_AGENT:
        response_text = agentic_llm(prompt="{prompt}", system="{system_prompt}" or None, state=state)
    else:
        llm_step = RunnableLambda(
            lambda user_prompt: call_llm(
                prompt=user_prompt,
                system="{system_prompt}" or None,
                model=DEFAULT_MODEL,
                reflection=ENABLE_REFLECTION,
                reflection_steps=REFLECTION_STEPS,
            )
        )
        response_text = llm_step.invoke("{prompt}")

    state["data"].append({{"node": "{node_name}", "response": response_text}})
    state["context"]["openai_{node_id}"] = response_text
    state["last_node"] = "{node_name}"
    return state
"""
        code = template.format(
            func_name=sanitize_identifier(node_function_name(node)),
            system_prompt=system_prompt,
            prompt=prompt,
            node_name=escape_string(node.name),
            node_id=node.id,
        )
        return textwrap.dedent(code)

    def uses_llm(self, node: N8nNode) -> bool:
        return True


class UnsupportedNodeTranslator:
    """Fallback translator that raises NotImplementedError at runtime."""

    def generate_node_function(self, node: N8nNode) -> str:
        template = """
def {func_name}(state: WorkflowState) -> WorkflowState:
    '''Unsupported node type stub.'''
    raise NotImplementedError("Node type {node_type} is not supported yet.")
"""
        code = template.format(
            func_name=sanitize_identifier(node_function_name(node)),
            node_type=escape_string(node.type),
        )
        return textwrap.dedent(code)

    def uses_llm(self, node: N8nNode) -> bool:
        return False


def get_translators(options: GenerationOptions) -> Dict[str, NodeTranslator]:
    """Return node translators configured with generation options."""

    return {
        "n8n-nodes-base.set": SetNodeTranslator(),
        "n8n-nodes-base.if": IfNodeTranslator(),
        "n8n-nodes-base.httpRequest": HttpRequestTranslator(),
        "n8n-nodes-base.code": CodeNodeTranslator(),
        "n8n-nodes-base.openAi": OpenAiNodeTranslator(options),
    }


DEFAULT_TRANSLATOR = UnsupportedNodeTranslator()


def sanitize_identifier(name: str) -> str:
    """Return a valid Python identifier derived from the provided string."""

    sanitized = re.sub(r"\W+", "_", name)
    sanitized = sanitized.strip("_")
    if sanitized and sanitized[0].isdigit():
        sanitized = f"n_{sanitized}"
    return sanitized or "node"


def escape_string(text: str) -> str:
    """Escape quotes for safe embedding in generated code."""

    return text.replace("\\", "\\\\").replace("\"", "\\\"")


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
        start_nodes = [next(iter(nodes.keys()))]

    return N8nWorkflow(nodes=nodes, edges=edges, start_nodes=start_nodes)


def generate_node_function(node: N8nNode, options: GenerationOptions) -> Tuple[str, bool]:
    """Generate Python code for a node function and return (code, uses_llm)."""

    translators = get_translators(options)
    translator = translators.get(node.type, DEFAULT_TRANSLATOR)
    if translator is DEFAULT_TRANSLATOR and node.type not in translators:
        logger.warning("Unsupported node type '%s'. Generating stub node.", node.type)
    code = translator.generate_node_function(node)
    return code, translator.uses_llm(node)


def generate_routing_function(node: N8nNode, edges: List[N8nEdge]) -> str:
    """Generate a routing function for conditional edges from a node."""

    default_label = edges[0].condition_label or f"out_{edges[0].source_output_index}"
    label_lines = []
    for edge in edges:
        label = edge.condition_label or f"out_{edge.source_output_index}"
        label_lines.append(f"    if label == '{label}':\n        return '{edge.target_id}'")
    mapping_body = "\n".join(label_lines)
    route_hint = f"state.get('context', {{}}).get('route_{node.id}')"
    if "if" in node.type.lower():
        route_hint = f"state.get('context', {{}}).get('if_{node.id}')"
    template = """
def {func_name}(state: WorkflowState) -> str:
    '''Routing function for node {node_name}. Update the label as needed.'''
    label = state.get("context", {{}}).get("route_{node_id}")
    if label is None:
        label = {route_hint}
    if label is None:
        label = "{default_label}"
    if isinstance(label, bool):
        label = "true" if label else "false"
    label = str(label)
{mapping_body}
    return '{fallback_target}'
"""
    code = template.format(
        func_name=f"route_from_{sanitize_identifier(node.id)}",
        node_name=node.name,
        node_id=node.id,
        route_hint=route_hint,
        default_label=default_label,
        mapping_body=mapping_body,
        fallback_target=edges[-1].target_id,
    )
    return textwrap.dedent(code)


def generate_python(workflow: N8nWorkflow, options: GenerationOptions) -> str:
    """Generate the Python script implementing the workflow using LangGraph."""

    lines: List[str] = []
    append = lines.append
    append("from __future__ import annotations")
    append("import argparse")
    append("import json")
    append("from typing import TypedDict, Any, Dict, List")
    append("from openai import OpenAI")
    append("from langgraph.graph import StateGraph")
    append("import os")
    append("import textwrap")
    append("")
    append("class WorkflowState(TypedDict, total=False):")
    append("    data: List[Dict[str, Any]]")
    append("    context: Dict[str, Any]")
    append("    last_node: str")
    append("")
    append(f"DEFAULT_MODEL = \"{escape_string(options.llm_model)}\"")
    append(f"ENABLE_REFLECTION = {options.enable_reflection}")
    append(f"REFLECTION_STEPS = {max(1, options.reflection_steps)}")
    append(f"ENABLE_AGENT = {options.enable_agent}")
    append("")
    append("client = OpenAI()")
    append("")
    append("def _invoke_llm(messages: List[Dict[str, str]], model: str) -> str:")
    append("    response = client.responses.create(model=model, input=messages)")
    append("    choice = response.output[0]")
    append("    return choice.content[0].text")
    append("")
    append("def call_llm(prompt: str, system: str | None = None, model: str = DEFAULT_MODEL, reflection: bool = False, reflection_steps: int = 1) -> str:")
    append("    \"\"\"Call OpenAI Responses API with optional reflection loops.\"\"\"")
    append("    messages = []")
    append("    if system:\n        messages.append({\"role\": \"system\", \"content\": system})")
    append("    messages.append({\"role\": \"user\", \"content\": prompt})")
    append("    draft = _invoke_llm(messages, model=model)")
    append("    if not reflection:")
    append("        return draft")
    append("    critique = None")
    append("    for _ in range(max(1, reflection_steps)):")
    append("        critique_prompt = textwrap.dedent(f\"\"\"Review and, if needed, critique the following draft. Return only actionable suggestions.\nDraft:\n{draft}\"\"\")")
    append("        critique_messages = []")
    append("        if system:\n            critique_messages.append({\"role\": \"system\", \"content\": system})")
    append("        critique_messages.append({\"role\": \"user\", \"content\": critique_prompt})")
    append("        critique = _invoke_llm(critique_messages, model=model)")
    append("        revision_prompt = textwrap.dedent(f\"\"\"Improve the previous draft using the critique. Return the revised answer directly.\nDraft:\n{draft}\nCritique:\n{critique}\"\"\")")
    append("        revision_messages = []")
    append("        if system:\n            revision_messages.append({\"role\": \"system\", \"content\": system})")
    append("        revision_messages.append({\"role\": \"user\", \"content\": revision_prompt})")
    append("        draft = _invoke_llm(revision_messages, model=model)")
    append("    return draft")
    append("")
    append("def agentic_llm(prompt: str, system: str | None, state: WorkflowState) -> str:")
    append("    \"\"\"Simple agent loop that reflects and uses context hints for reasoning.\"\"\"")
    append("    context_hint = json.dumps(state.get(\"context\", {}), ensure_ascii=False)")
    append("    enriched_prompt = textwrap.dedent(\"\"\"" "You are an agent solving a task with access to prior context.\n" "Context: {context}\nTask: {task}\"\"\").format(context=context_hint, task=prompt)")
    append("    draft = call_llm(enriched_prompt, system=system, model=DEFAULT_MODEL, reflection=True, reflection_steps=max(1, REFLECTION_STEPS))")
    append("    followup_prompt = textwrap.dedent(\"\"\"" "Re-evaluate the draft answer. If it is complete, return it verbatim. " "If it can be improved using context, return the improved version.\nDraft:\n{draft}\"\"\").format(draft=draft)")
    append("    final_answer = call_llm(followup_prompt, system=system, model=DEFAULT_MODEL, reflection=False)")
    append("    return final_answer")
    append("")

    sorted_nodes = sorted(workflow.nodes.values(), key=lambda n: n.name)
    for node in sorted_nodes:
        func_code, _ = generate_node_function(node, options)
        append(func_code)

    edges_by_source: Dict[str, List[N8nEdge]] = {}
    for edge in workflow.edges:
        edges_by_source.setdefault(edge.source_id, []).append(edge)

    for source_id, edges in edges_by_source.items():
        if len(edges) > 1 or any(edge.condition_label for edge in edges):
            append(generate_routing_function(workflow.nodes.get(source_id, N8nNode(source_id, source_id, "", {}, {})), edges))

    append("def build_graph():")
    append("    graph = StateGraph(WorkflowState)")
    for node in sorted_nodes:
        append(f"    graph.add_node('{node.id}', {sanitize_identifier(node_function_name(node))})")

    if workflow.start_nodes:
        append(f"    graph.set_entry_point('{workflow.start_nodes[0]}')")

    for source_id, edges in edges_by_source.items():
        if len(edges) == 1 and not edges[0].condition_label:
            append(f"    graph.add_edge('{source_id}', '{edges[0].target_id}')")
        else:
            label_map = {edge.condition_label or f"out_{edge.source_output_index}": edge.target_id for edge in edges}
            append(
                f"    graph.add_conditional_edges('{source_id}', route_from_{sanitize_identifier(source_id)}, {label_map})"
            )

    append("    return graph.compile()")
    append("")
    append("def run_workflow(initial_state: Dict[str, Any] | None = None) -> Dict[str, Any]:")
    append("    state = initial_state or {'data': [], 'context': {}}")
    append("    app = build_graph()")
    append("    result = app.invoke(state)")
    append("    return result")
    append("")
    append("def main():")
    append("    parser = argparse.ArgumentParser(description='Run generated LangGraph workflow.')")
    append("    parser.add_argument('--input-json', help='Initial state as JSON string', default=None)")
    append("    args = parser.parse_args()")
    append("    initial_state = json.loads(args.input_json) if args.input_json else None")
    append("    final_state = run_workflow(initial_state)")
    append("    print(json.dumps(final_state, indent=2))")
    append("")
    append("if __name__ == '__main__':")
    append("    main()")

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
    parser.add_argument("--llm-model", default="gpt-5", help="Model name to embed in generated scripts (e.g., gpt-5.1-codex-max)")
    parser.add_argument(
        "--enable-reflection",
        action="store_true",
        help="Wrap LLM calls in a reflection loop inside generated scripts",
    )
    parser.add_argument(
        "--reflection-steps",
        type=int,
        default=1,
        help="Number of reflection iterations when enabled",
    )
    parser.add_argument(
        "--enable-agent",
        action="store_true",
        help="Use the lightweight agentic LLM helper in generated OpenAI nodes",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args(argv)


def write_output(code: str, output_path: str) -> None:
    """Write generated code to output path."""

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(code, encoding="utf-8")


def summarize(workflow: N8nWorkflow, output_path: str, options: GenerationOptions) -> None:
    """Print summary of conversion."""

    print(f"Converted {len(workflow.nodes)} nodes")
    print(f"Created {len(workflow.edges)} edges")
    print(f"Generated file: {output_path}")
    print(f"LLM model: {options.llm_model}")
    if options.enable_agent:
        print("Agentic LLM: enabled")
    if options.enable_reflection:
        print(f"Reflection steps: {max(1, options.reflection_steps)}")


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

    options = GenerationOptions(
        llm_model=args.llm_model,
        enable_reflection=args.enable_reflection,
        reflection_steps=args.reflection_steps,
        enable_agent=args.enable_agent,
    )

    try:
        code = generate_python(workflow, options)
    except Exception as exc:
        logger.error("Failed to generate Python code: %s", exc)
        sys.exit(1)

    write_output(code, args.output)

    if args.pretty:
        prettify_code(Path(args.output))

    summarize(workflow, args.output, options)


if __name__ == "__main__":
    main()
