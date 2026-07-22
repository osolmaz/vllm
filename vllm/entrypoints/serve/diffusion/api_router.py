# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from vllm.logger import init_logger
from vllm.v1.engine.diffusion_events import DiffusionEventBroadcaster

logger = init_logger(__name__)

router = APIRouter()


def _broadcaster(request: Request) -> DiffusionEventBroadcaster:
    engine_client = request.app.state.engine_client
    broadcaster = getattr(engine_client, "diffusion_event_broadcaster", None)
    if broadcaster is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Diffusion canvas streaming is not enabled. Start the server "
                "with --diffusion-stream-canvas and a diffusion model."
            ),
        )
    return broadcaster


@router.get("/v1/diffusion/events")
async def diffusion_events(raw_request: Request) -> StreamingResponse:
    """SSE side channel with intermediate diffusion canvas states.

    Each event is a JSON object `{"request_id", "step", "text"}` holding the
    detokenized canvas (partially denoised token block) that was scheduled
    for one denoising step. Committed tokens keep flowing through the normal
    OpenAI-compatible completion stream; this endpoint only exposes the
    intermediate states for observability and visualization.
    """
    broadcaster = _broadcaster(raw_request)

    async def event_stream():
        async for event in broadcaster.subscribe():
            payload = json.dumps(
                {
                    "request_id": event.request_id,
                    "step": event.step,
                    "text": event.text,
                }
            )
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def attach_router(app: FastAPI):
    app.include_router(router)
