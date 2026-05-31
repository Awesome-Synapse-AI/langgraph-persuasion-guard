# langgraph-persuasion-guard

Provider-agnostic persuasion guard built with LangChain and LangGraph.

Implementation code lives under `src/langgraph_persuasion_guard` and follows the design in `implementation-guide.md`.

You can configure models via:
- global environment defaults (`MODEL_NAME`, optional `MODEL_PROVIDER`)
- role-specific environment variables (`ROUTER_*`, `SANITIZER_*`, `EXECUTOR_*`, `CHAT_*`)
- Python arguments (`default_model`, `default_provider`, `role_model_overrides`)

Quick start:

```bash
pip install -e .
python examples/run_demo.py
```

PowerShell example:

```powershell
$env:MODEL_NAME = "gpt-4o-mini"
$env:MODEL_PROVIDER = "openai"
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

Role override example:

```python
from langgraph_persuasion_guard import RoleModelConfig, create_persuasion_guard

agent = create_persuasion_guard(
    default_model="gpt-4o-mini",
    default_provider="openai",
    role_model_overrides={
        "router": RoleModelConfig(model="gpt-4o-mini", model_provider="openai", temperature=0.0),
        "chat": RoleModelConfig(model="gpt-4o", model_provider="openai", temperature=0.7),
    },
)
```
