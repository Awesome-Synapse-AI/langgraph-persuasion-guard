# langgraph-persuasion-guard

Provider-agnostic persuasion guard built with LangChain and LangGraph.

Implementation code lives under `src/langgraph_persuasion_guard` and follows the design in `implementation-guide.md`.

You can configure models either via environment variables or directly in Python.

Quick start:

```bash
pip install -e .
python examples/run_demo.py
```

PowerShell example:

```powershell
$env:MODEL_NAME = "openai:gpt-4o-mini"
python examples/run_demo.py
```

Python config example (no environment variables required):

```python
from langgraph_persuasion_guard import create_persuasion_guard

agent = create_persuasion_guard(
    default_model="gpt-4o-mini",
    default_provider="openai",
    use_env=False,
)
```
