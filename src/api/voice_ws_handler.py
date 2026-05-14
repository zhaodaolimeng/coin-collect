#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WebSocket 双工通话处理器"""
import asyncio
import json
import logging
import subprocess
import traceback

import numpy as np

from core.voice.ws_adapters import WebSocketAudioSource, WebSocketAudioOutput
from core.voice.pipeline import DuplexCallPipeline, PipelineState, PipelineConfig
from core.voice.vad import SileroVAD

logger = logging.getLogger(__name__)


def _load_audio_file(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """用 ffmpeg 加载音频文件（支持 MP3/WAV），返回 float32 单声道 numpy 数组"""
    result = subprocess.run(
        ['ffmpeg', '-y', '-i', file_path, '-f', 'f32le', '-acodec', 'pcm_f32le',
         '-ar', str(target_sr), '-ac', '1', '-'],
        capture_output=True,
        timeout=60,
    )
    if result.returncode != 0 or len(result.stdout) == 0:
        raise RuntimeError(f"ffmpeg 加载音频失败: {result.stderr.decode()}")
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


async def handle_duplex_ws(websocket, chatbot):
    """处理 WebSocket 双工通话连接"""
    source = WebSocketAudioSource(sample_rate=16000, block_size=2048)
    vad = SileroVAD(sample_rate=16000, frame_duration_ms=128, energy_threshold=0.5, voice_frames=3, silence_frames=3)
    session_id = getattr(chatbot, "session_id", "")

    # 1. 获取问候文本（稍后在 ready 消息中发送）
    try:
        greeting_text, _ = await chatbot.process(use_tts=False)
    except Exception:
        greeting_text = "Halo, selamat siang. Ada yang bisa saya bantu?"
    logger.info(f"[Duplex] session={session_id[:8]} greeting={greeting_text[:50]}...")

    # 2. 构建管线
    async def send_audio_chunk(data: bytes, _sr: int):
        try:
            await websocket.send_bytes(data)
        except Exception:
            pass

    output = WebSocketAudioOutput(source, send_chunk=send_audio_chunk)
    config = PipelineConfig(sample_rate=16000, block_size=2048, silence_duration=0.3, max_speech_duration=15.0)
    pipeline = DuplexCallPipeline(chatbot, source, output, None, None, vad, config=config)

    def on_state(old, new):
        asyncio.ensure_future(_safe_send_json(websocket, {
            "type": "state",
            "from": old.name,
            "to": new.name,
        }))

    pipeline.on_state_change = on_state

    def on_debug(msg: str):
        asyncio.ensure_future(_safe_send_json(websocket, {
            "type": "debug",
            "text": msg,
        }))

    pipeline.on_debug = on_debug

    # 3. 同步加载 TTS + ASR（管线启动前完成，消除竞态）
    async def _init_tts(greeting_text):
        from core.voice.tts import TTSManager
        tts = TTSManager()
        result = await tts.synthesize(greeting_text)
        audio = None
        if result.success:
            if result.audio_data is not None:
                audio = np.frombuffer(result.audio_data, dtype=np.float32) \
                    if isinstance(result.audio_data, bytes) else result.audio_data
            elif result.audio_file:
                audio = _load_audio_file(result.audio_file)
            if audio is not None and len(audio) > 0:
                peak = float(np.max(np.abs(audio)))
                audio = audio / max(peak, 1.0) * 0.95
        return tts, audio

    async def _init_asr(chatbot):
        from core.voice.asr import ASRPipeline
        corrector = getattr(chatbot, "asr_corrector", None)
        return await ASRPipeline.create(model_size="small", corrector=corrector)

    tts_task = asyncio.create_task(_init_tts(greeting_text))
    asr_task = asyncio.create_task(_init_asr(chatbot))
    try:
        tts, greeting_audio = await tts_task
        pipeline._tts = tts
        logger.info(f"[Duplex] TTS loaded, greeting_audio={'ready' if greeting_audio is not None else 'none'}")
    except Exception as e:
        logger.warning(f"[Duplex] TTS loading failed: {e}")
        greeting_audio = None
    try:
        pipeline._asr = await asr_task
        logger.info("[Duplex] ASR loaded and injected into pipeline")
    except Exception as e:
        logger.warning(f"[Duplex] ASR loading failed: {e}")

    # 4. 发送 ready 消息，通知前端模型加载完成
    await _safe_send_json(websocket, {
        "type": "ready",
        "text": greeting_text,
        "session_id": session_id,
    })

    # 5. 启动管线后注入问候音频（start() 设为 LISTENING，注入后切到 RESPONDING）
    await pipeline.start()
    if greeting_audio is not None:
        pipeline.inject_greeting_audio(greeting_audio)
        logger.info(f"[Duplex] Greeting audio injected: {len(greeting_audio)} samples")

    pipeline_task = asyncio.create_task(_run_pipeline(pipeline, websocket))
    text_task = asyncio.create_task(_stream_text_events(pipeline, websocket))

    # 6. 消息循环 — 持续读取客户端音频 + JSON 控制消息
    chunk_count = 0
    last_chunk_log = 0
    try:
        while True:
            if pipeline.state == PipelineState.CLOSED:
                logger.info("[Duplex] Pipeline closed, ending message loop")
                break

            try:
                message = await websocket.receive()
            except Exception as e:
                logger.info(f"[Duplex] WebSocket receive ended: {type(e).__name__}: {e}")
                break

            if message.get("type") == "websocket.disconnect":
                logger.info(f"[Duplex] Client disconnected, code={message.get('code')}")
                break

            data = message.get("text") or message.get("bytes")
            if data is None:
                continue

            if isinstance(data, bytes):
                source.feed_chunk(data)
                chunk_count += 1
                if chunk_count - last_chunk_log >= 50:
                    last_chunk_log = chunk_count
                    qs = source._queue.qsize()
                    rms = source.current_rms()
                    logger.info(f"[Duplex] received {chunk_count} audio chunks, "
                                f"queue size={qs}, RMS={rms:.4f}")
                    on_debug(f"[音频] 已接收{chunk_count}块 queue={qs} RMS={rms:.4f}")
            elif isinstance(data, str):
                try:
                    msg = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if msg.get("type") == "interrupt":
                    on_debug(f"[客户端] 收到interrupt消息, 当前状态={pipeline.state.name}")
                    pipeline.handle_interruption()
                    await _safe_send_json(websocket, {"type": "interrupted"})
                elif msg.get("type") == "stop":
                    on_debug("[客户端] 收到stop消息")
                    break
    except Exception as e:
        logger.error(f"[Duplex] Message loop exception: {type(e).__name__}: {e}")
    finally:
        pipeline_task.cancel()
        text_task.cancel()
        tts_task.cancel()
        asr_task.cancel()
        try:
            await pipeline.stop()
        except Exception:
            pass
        asr = pipeline._asr
        if asr:
            try:
                asr.shutdown()
            except Exception:
                pass


async def _run_pipeline(pipeline, websocket):
    """后台轮询管线 step()"""
    step_count = 0
    try:
        while pipeline.state != PipelineState.CLOSED:
            result = await pipeline.step()
            if result:
                step_count += 1
                if step_count % 100 == 0:
                    logger.debug(f"[Duplex] step={step_count} state={pipeline.state.name}")
                if result.asr_text:
                    logger.info(f"[Duplex] ASR text: {result.asr_text[:80]}")
                    await _safe_send_json(websocket, {"type": "asr", "text": result.asr_text})
                if result.agent_text:
                    logger.info(f"[Duplex] Agent text: {result.agent_text[:80]}")
                    await _safe_send_json(websocket, {"type": "agent_text", "text": result.agent_text})
            await asyncio.sleep(0.01)
        logger.info(f"[Duplex] Pipeline ended after {step_count} steps")
        await _safe_send_json(websocket, {"type": "closed"})
    except asyncio.CancelledError:
        logger.info(f"[Duplex] Pipeline cancelled after {step_count} steps")
    except Exception:
        logger.error(f"Pipeline error: {traceback.format_exc()}")


async def _stream_text_events(pipeline, websocket):
    """检测 agent_text 变化并推送"""
    last_agent = None
    try:
        while pipeline.state != PipelineState.CLOSED:
            current = getattr(pipeline, "_current_agent_text", None)
            if current and current != last_agent:
                last_agent = current
                await _safe_send_json(websocket, {"type": "agent_text", "text": current})
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        pass


async def _safe_send_json(websocket, data: dict):
    try:
        await websocket.send_text(json.dumps(data, ensure_ascii=False))
    except Exception:
        pass
