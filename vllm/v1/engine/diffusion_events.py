# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger(__name__)

# Per-subscriber queue depth. Denoising steps arrive at ~10-100 Hz per
# request; a slow consumer loses the oldest snapshots, which is acceptable
# because every event is a full canvas snapshot, not a delta.
_SUBSCRIBER_QUEUE_SIZE = 256


@dataclass(frozen=True)
class DiffusionCanvasEvent:
    """One intermediate canvas snapshot of a diffusion request.

    `text` is the detokenized canvas scheduled for a denoising step: accepted
    tokens mixed with the sampler's renoise tokens. `step` counts denoising
    steps observed for the request since the frontend started tracking it.
    `block` is the commit ordinal of the block the snapshot belongs to (how
    many commits the request had streamed when it was taken): the canvas feed
    and the completion stream travel on separate connections, so clients use
    it to discard snapshots of a block whose commit they already received.
    """

    request_id: str
    step: int
    block: int
    text: str


class DiffusionEventBroadcaster:
    """Fan-out of diffusion canvas events to SSE subscribers.

    publish() is called from the engine output-processing loop and must not
    block: when a subscriber's queue is full, the oldest queued snapshot is
    evicted so a slow consumer always drains toward the current canvas state
    (each event is a self-contained snapshot).
    """

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[DiffusionCanvasEvent]] = set()

    @property
    def has_subscribers(self) -> bool:
        return bool(self._subscribers)

    def publish(self, event: DiffusionCanvasEvent) -> None:
        for queue in self._subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Evict the oldest snapshot in favor of the newest; both
                # publish() and the consumer run on the same event loop, so
                # these two operations cannot race.
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(event)

    async def subscribe(self) -> AsyncIterator[DiffusionCanvasEvent]:
        queue: asyncio.Queue[DiffusionCanvasEvent] = asyncio.Queue(
            maxsize=_SUBSCRIBER_QUEUE_SIZE
        )
        self._subscribers.add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)
