"""Web front-end for the fake Alexa+.   Run: python -m simulator.app  -> http://127.0.0.1:8898"""
import os
from pathlib import Path
from collections import defaultdict

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from fridge_log import log
from . import agent

app = FastAPI(title="Alexa+ Simulator")
STATIC = Path(__file__).parent / "static"
SESSIONS: dict[str, list[dict]] = defaultdict(list)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    session_id: str = "default"


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/api/chat")
async def chat(body: ChatIn):
    hist = SESSIONS[body.session_id][-12:]
    out = await agent.chat(list(hist), body.message)
    SESSIONS[body.session_id] += [{"role": "user", "content": body.message}, {"role": "assistant", "content": out["reply"]}]
    return out


@app.post("/api/reset")
def reset(session_id: str = "default"):
    SESSIONS.pop(session_id, None)
    return {"ok": True}


@app.get("/api/status")
async def status():
    try:
        info = await agent.deploy() if agent._tools_cache is None else agent._server_info
        return {"connected": True, "mcp_url": agent.MCP_URL, "provider": agent.PROVIDER, **info}
    except Exception as e:
        return JSONResponse({"connected": False, "mcp_url": agent.MCP_URL, "provider": agent.PROVIDER, "error": str(e)}, status_code=200)


@app.post("/api/redeploy")
async def redeploy():
    """Equivalent of `alexa-ai deploy`: Alexa+ only re-reads tools on deployment."""
    return await agent.deploy()


@app.get("/api/fridge")
async def fridge():
    """Live fridge panel -- fetched THROUGH the MCP server, same path Alexa+ uses."""
    trace: list = []
    try:
        async with agent.mcp_session() as (s, _):
            await agent.call_tool(s, "get_fridge_inventory", {}, trace)
        return trace[0]["result"]
    except Exception as e:
        return {"items": [], "error": str(e)}


if __name__ == "__main__":
    log("SIM", f"Alexa+ simulator on http://127.0.0.1:{os.environ.get('SIM_PORT', '8898')}  (brain: {agent.PROVIDER}, MCP: {agent.MCP_URL})")
    uvicorn.run(app, host=os.environ.get("SIM_HOST", "127.0.0.1"), port=int(os.environ.get("SIM_PORT", "8898")), log_level="warning")
