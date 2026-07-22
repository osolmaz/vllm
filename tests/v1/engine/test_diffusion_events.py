# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from dataclasses import dataclass

import pytest
from transformers import AutoTokenizer

from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.engine.diffusion_events import (
    DiffusionCanvasEvent,
    DiffusionEventBroadcaster,
)
from vllm.v1.engine.output_processor import OutputProcessor

# Deliberately ungated so the test runs without HF credentials.
_TOKENIZER_NAME = "gpt2"
_PROMPT_TEXT = "The quick brown fox jumps over the lazy dog."
_GENERATION_TEXT = " It kept running through the quiet morning field."


@dataclass
class _Vectors:
    tokenizer: AutoTokenizer
    prompt_tokens: list[int]
    prompt_string: str
    generation_tokens: list[int]


@pytest.fixture(scope="module")
def vectors() -> _Vectors:
    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)
    return _Vectors(
        tokenizer=tokenizer,
        prompt_tokens=tokenizer(_PROMPT_TEXT).input_ids,
        prompt_string=_PROMPT_TEXT,
        generation_tokens=tokenizer(_GENERATION_TEXT).input_ids,
    )


def _make_request(request_id: str, prompt_tokens: list[int]) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=f"{request_id}-int",
        external_req_id=request_id,
        prompt_token_ids=prompt_tokens,
        mm_features=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        sampling_params=SamplingParams(
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
            output_kind=RequestOutputKind.DELTA,
            stop=[],
            include_stop_str_in_output=False,
        ),
        pooling_params=None,
    )


@pytest.mark.asyncio
async def test_broadcaster_fanout_and_overflow():
    broadcaster = DiffusionEventBroadcaster()
    assert not broadcaster.has_subscribers

    received: list[DiffusionCanvasEvent] = []

    async def consume(n: int):
        async for event in broadcaster.subscribe():
            received.append(event)
            if len(received) >= n:
                break

    task = asyncio.create_task(consume(2))
    # Let the subscriber register.
    await asyncio.sleep(0)
    assert broadcaster.has_subscribers

    broadcaster.publish(DiffusionCanvasEvent("req-1", 1, "noise a"))
    broadcaster.publish(DiffusionCanvasEvent("req-1", 2, "noise b"))
    await asyncio.wait_for(task, timeout=5)

    assert [event.step for event in received] == [1, 2]
    assert received[0].request_id == "req-1"

    # Subscriber went away; publishing must not raise.
    await asyncio.sleep(0)
    assert not broadcaster.has_subscribers
    broadcaster.publish(DiffusionCanvasEvent("req-1", 3, "noise c"))


@pytest.mark.asyncio
async def test_output_processor_publishes_canvas_events(vectors):
    broadcaster = DiffusionEventBroadcaster()
    output_processor = OutputProcessor(
        vectors.tokenizer,
        log_stats=False,
        diffusion_event_broadcaster=broadcaster,
    )
    request = _make_request("request-0", vectors.prompt_tokens)
    output_processor.add_request(request, vectors.prompt_string)

    events: list[DiffusionCanvasEvent] = []

    async def consume(n: int):
        async for event in broadcaster.subscribe():
            events.append(event)
            if len(events) >= n:
                break

    task = asyncio.create_task(consume(2))
    await asyncio.sleep(0)

    canvas = vectors.generation_tokens[:8]
    denoise_step = EngineCoreOutput(
        request_id=request.request_id,
        new_token_ids=[],
        diffusion_canvas_token_ids=canvas,
    )

    # Two denoising steps: no completion output must be produced, but two
    # canvas events must be broadcast with increasing step numbers.
    processed = output_processor.process_outputs([denoise_step, denoise_step])
    assert processed.request_outputs == []
    await asyncio.wait_for(task, timeout=5)

    assert [event.step for event in events] == [1, 2]
    assert events[0].request_id == "request-0"
    expected_text = vectors.tokenizer.decode(canvas, skip_special_tokens=True)
    assert events[0].text == expected_text

    # A commit step (real tokens, no canvas) flows through the normal path.
    commit = EngineCoreOutput(
        request_id=request.request_id,
        new_token_ids=vectors.generation_tokens[:2],
    )
    output_processor.process_outputs([commit])


def test_output_processor_skips_detokenization_without_subscribers(vectors):
    broadcaster = DiffusionEventBroadcaster()
    output_processor = OutputProcessor(
        vectors.tokenizer,
        log_stats=False,
        diffusion_event_broadcaster=broadcaster,
    )
    request = _make_request("request-0", vectors.prompt_tokens)
    output_processor.add_request(request, vectors.prompt_string)

    denoise_step = EngineCoreOutput(
        request_id=request.request_id,
        new_token_ids=[],
        diffusion_canvas_token_ids=vectors.generation_tokens[:4],
    )
    processed = output_processor.process_outputs([denoise_step])
    assert processed.request_outputs == []

    # Step counting continues even when nobody is subscribed.
    req_state = output_processor.request_states[request.request_id]
    assert req_state.diffusion_step == 1
