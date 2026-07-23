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
async def diffusion_events(
    raw_request: Request, request_id: str | None = None
) -> StreamingResponse:
    """SSE side channel with intermediate diffusion canvas states.

    Each event is a JSON object `{"request_id", "step", "block", "text"}`
    holding the detokenized canvas (partially denoised token block) that was
    scheduled for one denoising step; `block` is the commit ordinal of that
    block, letting clients discard snapshots of an already-committed block
    that raced the commit on this separate connection. Committed tokens keep
    flowing through the normal OpenAI-compatible completion stream; this
    endpoint only exposes the intermediate states for observability and
    visualization.

    The optional `request_id` query parameter scopes the stream to a single
    request. Clients that set the `X-Request-Id` header on their completion
    request know the id (`chatcmpl-<header>`) up front and should subscribe
    scoped, so they never receive canvas states of other clients' requests.
    An unscoped subscription streams every request on the server and is
    intended for server-side observability.
    """
    broadcaster = _broadcaster(raw_request)

    async def event_stream():
        async for event in broadcaster.subscribe():
            if request_id is not None and event.request_id != request_id:
                continue
            payload = json.dumps(
                {
                    "request_id": event.request_id,
                    "step": event.step,
                    "block": event.block,
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
