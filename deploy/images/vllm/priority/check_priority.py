"""The build gate of vllm-patches/priority.patch, run inside the image (it ships no pytest): the Anthropic /v1/messages
request carries `priority` into the chat request the engine runs, and a priority on a server that does not schedule by
it is REFUSED (ValueError → HTTP 400) instead of being ignored — while one that does passes the check."""
from types import SimpleNamespace

from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.v1.engine.input_processor import InputProcessor

req = AnthropicMessagesRequest(model="m", max_tokens=4, messages=[{"role": "user", "content": "hi"}], priority=-1)
assert AnthropicServingMessages._build_base_request(req, [{"role": "user", "content": "hi"}]).priority == -1
plain = AnthropicMessagesRequest(model="m", max_tokens=4, messages=[{"role": "user", "content": "hi"}])
assert AnthropicServingMessages._build_base_request(plain, []).priority == 0, "none named: the engine's default 0"


class Past(Exception):
    """Raised by the stubbed next step: the priority check let the request through."""


def run(policy: str, priority: int) -> str:
    proc = InputProcessor.__new__(InputProcessor)
    proc.scheduler_config = SimpleNamespace(policy=policy)
    proc._validate_params = lambda *a, **k: (_ for _ in ()).throw(Past())
    try:
        proc.process_inputs("r", "p", None, (), priority=priority)
    except Past:
        return "passed"
    except ValueError as exc:
        return f"refused: {exc}"
    return "?"


assert run("fcfs", -1).startswith("refused: priority -1 needs --scheduling-policy priority"), run("fcfs", -1)
assert run("fcfs", 0) == "passed", "no priority named: nothing to refuse"
assert run("priority", -1) == "passed"
print("priority patch: ok")
