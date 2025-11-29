# Examples

Use this directory to store sample n8n exports and generated Python workflows for testing.

## Suggested Workflow
1. Place an exported n8n JSON file here (e.g., `sample_workflow.json`).
2. Run the converter from the project root:
   ```bash
   python n8n_to_langgraph.py -i examples/sample_workflow.json -o examples/sample_workflow.py --pretty
   ```
3. Execute the generated script to verify behavior:
   ```bash
   python examples/sample_workflow.py --input-json '{"data": [], "context": {}}'
   ```

## Tips
- Populate `state['context']` with values that IF/branch nodes expect.
- Review Code node comments for manual adaptation before executing in production.
- Ensure `OPENAI_API_KEY` is available when running workflows that use LLM nodes.
