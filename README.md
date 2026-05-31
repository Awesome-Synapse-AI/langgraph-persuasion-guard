# langgraph-persuasion-guard

Provider-agnostic persuasion guard built with LangChain and LangGraph.

Implementation code lives under `src/langgraph_persuasion_guard` and follows the design in `implementation-guide.md`.

## Graph Flow

```mermaid
flowchart TD
    S([START])
    R[router_node<br/>classify latest chat message]
    SG[sanitizer_gate_node<br/>decide requires_sanitizer]
    SN[sanitizer_node<br/>build genesis brief + handoff]
    E[executor_node<br/>run isolated task execution]
    C[chat_node<br/>normal chat + topic summary]
    X([END])

    S -->|phase==EXECUTION AND execution_history[-1] is HumanMessage| SG
    S -->|otherwise| R

    R -->|phase==EXECUTION (router_decision.is_task=true)| SG
    R -->|phase==CHAT (router_decision.is_task=false)| C

    SG -->|sanitizer_required=true| SN
    SG -->|sanitizer_required=false| E

    SN --> E
    E --> X
    C --> X
```

Routing details from code:
- `route_from_start`: skips router and resumes execution at `sanitizer_gate_node` only for execution follow-up turns where the latest `execution_history` item is a `HumanMessage`.
- `route_after_router`: sends task-initiation to execution path (`sanitizer_gate_node`), otherwise to chat path (`chat_node`).
- `route_after_sanitizer_gate`: runs `sanitizer_node` when sanitization is required, else executes directly with `executor_node`.

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
