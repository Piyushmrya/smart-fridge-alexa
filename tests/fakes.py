"""A scripted stand-in for Claude on Bedrock, so Strands code paths can be tested with no AWS key and no cost."""
from strands.models import Model


class ScriptedModel(Model):
    """script: list of steps, one per model call: ("tool", name, json_input) | ("text", string) | ("raise", Exception)."""
    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def get_config(self): return {"model_id": "fake.anthropic.claude-sonnet-4-6"}
    def update_config(self, **kwargs): pass
    async def structured_output(self, *a, **k): raise NotImplementedError

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        step = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if step[0] == "raise":
            raise step[1]
        yield {"messageStart": {"role": "assistant"}}
        if step[0] == "tool":
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": f"t{self.calls}", "name": step[1]}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": step[2]}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": step[1]}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": {"usage": {"inputTokens": 1000, "outputTokens": 100, "totalTokens": 1100}, "metrics": {"latencyMs": 1}}}
