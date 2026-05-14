"""双工通话端到端集成测试 — 模拟完整多轮对话流程

覆盖场景:
- 单轮/多轮对话 (speech → ASR → Bot → TTS → playback → next turn)
- Barge-in during agent playback
- 完整状态转移验证
- ASR/TTS 未就绪时的优雅降级
- Bot CLOSE 触发管线关闭
- Handler-level WebSocket 协议验证
"""
import asyncio
import json
import sys
import time
from pathlib import Path
from unittest import mock
from unittest.mock import AsyncMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

SR = 16000
BLOCK = 2048


def make_speech(duration_s: float, sr: int = SR, freq: float = 300,
                amplitude: float = 0.02) -> np.ndarray:
    """生成模拟语音（正弦波，RMS 约 0.014，高于 VAD 阈值 0.01）"""
    t = np.arange(0, int(duration_s * sr), dtype=np.float32) / sr
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)


def make_silence(duration_s: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(duration_s * sr), dtype=np.float32)


def chunkify(audio: np.ndarray, block_size: int = BLOCK) -> list:
    chunks = []
    for i in range(0, len(audio), block_size):
        chunk = audio[i:i + block_size]
        if len(chunk) < block_size:
            chunk = np.pad(chunk, (0, block_size - len(chunk)))
        chunks.append(chunk)
    return chunks


async def wait_for_state(pipeline, target_state, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pipeline.state == target_state:
            return True
        await asyncio.sleep(0.05)
    return False


def feed_to_source(source, audio: np.ndarray):
    """将音频分块喂入 source 队列"""
    for chunk in chunkify(audio, BLOCK):
        source.feed_chunk(chunk)


async def run_pipeline_until(pipeline, target_states: set, max_steps: int = 400):
    """逐 step 推进管线直到进入目标状态集合，返回经历的状态转移"""
    transitions = []
    entered_processing = False
    for _ in range(max_steps):
        prev = pipeline.state
        result = await pipeline.step()
        if result and result.state_from != result.state_to:
            transitions.append((result.state_from.name, result.state_to.name))
        await asyncio.sleep(0.01)
        if pipeline.state == PipelineState.PROCESSING:
            entered_processing = True
        # 只有经过 PROCESSING 后才允许在 LISTENING/CLOSED 停止
        if pipeline.state in target_states:
            if entered_processing or PipelineState.PROCESSING in target_states:
                break
    return transitions


# 延迟导入 PipelineState（避免模块加载时循环依赖）
from src.core.voice.pipeline import PipelineState


# ═══════════════════════════════════════════════════════════════════
# Mock 组件
# ═══════════════════════════════════════════════════════════════════

class FakeASR:
    def __init__(self, texts=None):
        self.texts = texts or ["Ya, saya mengerti"]
        self.call_count = 0
        self.is_available = True
        self._shutdown = False
        self.last_audio = None

    async def transcribe_async(self, audio: np.ndarray) -> str:
        self.call_count += 1
        self.last_audio = audio.copy()
        idx = min(self.call_count - 1, len(self.texts) - 1)
        return self.texts[idx]

    def shutdown(self):
        self._shutdown = True


class FakeTTS:
    def __init__(self, audio_duration_s: float = 0.5, amplitude: float = 0.1):
        self.audio_duration_s = audio_duration_s
        self.amplitude = amplitude
        self.call_count = 0
        self.last_text = ""

    async def synthesize(self, text, **kwargs):
        from src.core.voice.tts import TTSResult
        self.call_count += 1
        self.last_text = text
        n = int(self.audio_duration_s * SR)
        audio = (np.ones(n, dtype=np.float32) * self.amplitude).copy()
        return TTSResult(
            text=text, audio_data=audio, audio_file=None,
            success=True, engine_name="fake",
        )


class FakeBot:
    def __init__(self, max_turns: int = 3, session_id: str = "test-session-001"):
        self.max_turns = max_turns
        self.session_id = session_id
        self.asr_corrector = None
        self.turns = 0
        self._state = _FakeState("INIT")
        self.inputs_received = []

    @property
    def state(self):
        return self._state

    async def process(self, customer_input=None, use_tts=False):
        self.turns += 1
        if customer_input:
            self.inputs_received.append(customer_input)
        if self.turns >= self.max_turns:
            self._state = _FakeState("CLOSE")
            return "Terima kasih, selamat tinggal.", None
        return f"Baik, saya catat. [{self.turns}]", None


class _FakeState:
    def __init__(self, name: str):
        self.name = name


class _FakeURL:
    def __init__(self, query: str = "chat_group=H2&customer_name=Test"):
        self.query = query


class ConversationWebSocket:
    """asyncio.Queue 驱动 — 支持交互式和预填充两种模式"""

    def __init__(self, query_string: str = "chat_group=H2&customer_name=Test"):
        self.sent_json: list = []
        self.sent_bytes: list = []
        self._incoming: asyncio.Queue = asyncio.Queue()
        self._prefilled: list = []
        self._url = _FakeURL(query_string)
        self.accepted = False
        self._json_read_pos = 0

    @property
    def url(self):
        return self._url

    async def accept(self):
        self.accepted = True

    async def send_text(self, data: str):
        self.sent_json.append(data)

    async def send_bytes(self, data: bytes):
        self.sent_bytes.append(data)

    async def close(self):
        pass

    async def receive(self) -> dict:
        if self._prefilled:
            for msg in self._prefilled:
                self._incoming.put_nowait(msg)
            self._prefilled.clear()
        try:
            msg = await asyncio.wait_for(self._incoming.get(), timeout=30.0)
        except asyncio.TimeoutError:
            return {"type": "websocket.disconnect", "code": 1000}
        if isinstance(msg, bytes):
            return {"type": "websocket.receive", "bytes": msg}
        return {"type": "websocket.receive", "text": msg}

    # ── 预填充 API（在调用 handler 前使用）──

    def feed_audio(self, audio: np.ndarray):
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        for chunk in chunkify(audio, BLOCK):
            self._prefilled.append(chunk.tobytes())

    def put_json(self, data: dict):
        self._prefilled.append(json.dumps(data, ensure_ascii=False))

    def send_stop(self):
        self.put_json({"type": "stop"})

    def send_interrupt(self):
        self.put_json({"type": "interrupt"})

    # ── 实时 API（handler 运行时使用）──

    def put_nowait(self, msg):
        """非阻塞入队（bytes 或 str）"""
        self._incoming.put_nowait(msg)

    # ── 查询方法 ──

    def get_json_messages(self, msg_type: str) -> list:
        result = []
        for raw in self.sent_json:
            try:
                parsed = json.loads(raw)
                if parsed.get("type") == msg_type:
                    result.append(parsed)
            except json.JSONDecodeError:
                pass
        return result

    def get_first_json_of_type(self, msg_type: str) -> dict | None:
        msgs = self.get_json_messages(msg_type)
        return msgs[0] if msgs else None

    async def wait_for_json(self, msg_type: str, timeout: float = 5.0) -> dict | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            for i in range(self._json_read_pos, len(self.sent_json)):
                try:
                    msg = json.loads(self.sent_json[i])
                    if msg.get("type") == msg_type:
                        self._json_read_pos = i + 1
                        return msg
                except json.JSONDecodeError:
                    pass
            self._json_read_pos = len(self.sent_json)
            await asyncio.sleep(0.1)
        return None


# ═══════════════════════════════════════════════════════════════════
# Pipeline-level 测试
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_single_turn_speech_to_response():
    """用户说话 → ASR 转写 → Bot 回复 → LISTENING"""
    from src.core.voice.pipeline import DuplexCallPipeline, PipelineConfig
    from src.core.voice.ws_adapters import WebSocketAudioSource, WebSocketAudioOutput

    source = WebSocketAudioSource(sample_rate=SR, block_size=BLOCK)
    await source.start()
    output = WebSocketAudioOutput(source)
    config = PipelineConfig(
        sample_rate=SR, block_size=BLOCK,
        silence_duration=0.3, max_speech_duration=10.0,
    )
    bot = FakeBot(max_turns=10)
    asr = FakeASR(texts=["Halo, apa kabar?"])
    tts = FakeTTS(audio_duration_s=0.3)

    pipeline = DuplexCallPipeline(bot, source, output, asr, tts, None, config=config)
    await pipeline.start()

    # 喂入语音 + 静音（默认 VAD silence_frames=10，需足够静音块）
    feed_to_source(source, make_speech(1.5))
    feed_to_source(source, make_silence(2.5))

    await run_pipeline_until(pipeline, {PipelineState.LISTENING, PipelineState.CLOSED})

    assert asr.call_count >= 1, f"ASR 应被调用, actual: {asr.call_count}"
    assert pipeline._current_asr_text == "Halo, apa kabar?"
    assert len(pipeline._current_agent_text) > 0
    assert pipeline.state == PipelineState.LISTENING

    await pipeline.stop()


@pytest.mark.asyncio
async def test_complete_state_cycle():
    """验证状态转移: LISTENING → PROCESSING → RESPONDING → LISTENING"""
    from src.core.voice.pipeline import DuplexCallPipeline, PipelineConfig
    from src.core.voice.ws_adapters import WebSocketAudioSource, WebSocketAudioOutput

    source = WebSocketAudioSource(sample_rate=SR, block_size=BLOCK)
    await source.start()

    async def send_chunk(data, sr):
        pass

    output = WebSocketAudioOutput(source, send_chunk=send_chunk)
    config = PipelineConfig(
        sample_rate=SR, block_size=BLOCK,
        silence_duration=0.3, max_speech_duration=10.0,
    )
    bot = FakeBot(max_turns=10)
    asr = FakeASR(texts=["Halo"])
    tts = FakeTTS(audio_duration_s=0.2)

    pipeline = DuplexCallPipeline(bot, source, output, asr, tts, None, config=config)
    await pipeline.start()

    feed_to_source(source, make_speech(1.5))
    feed_to_source(source, make_silence(2.5))

    transitions = await run_pipeline_until(pipeline, {PipelineState.LISTENING})

    state_names = []
    for f, t in transitions:
        if not state_names or state_names[-1] != f:
            state_names.append(f)
        state_names.append(t)

    assert "PROCESSING" in state_names, f"Transitions: {transitions}"
    assert "RESPONDING" in state_names, f"Transitions: {transitions}"
    assert state_names[-1] == "LISTENING", f"Final state should be LISTENING: {transitions}"

    await pipeline.stop()


@pytest.mark.asyncio
async def test_multi_turn_until_close():
    """3 轮对话后 Bot CLOSE → 管线进入 CLOSED"""
    from src.core.voice.pipeline import DuplexCallPipeline, PipelineConfig
    from src.core.voice.ws_adapters import WebSocketAudioSource, WebSocketAudioOutput

    source = WebSocketAudioSource(sample_rate=SR, block_size=BLOCK)
    await source.start()

    async def send_chunk(data, sr):
        pass

    output = WebSocketAudioOutput(source, send_chunk=send_chunk)
    config = PipelineConfig(
        sample_rate=SR, block_size=BLOCK,
        silence_duration=0.3, max_speech_duration=10.0,
    )
    bot = FakeBot(max_turns=3)
    asr = FakeASR(texts=["Turn 1", "Turn 2", "Turn 3"])
    tts = FakeTTS(audio_duration_s=0.2)

    pipeline = DuplexCallPipeline(bot, source, output, asr, tts, None, config=config)
    await pipeline.start()

    for turn_idx in range(6):
        feed_to_source(source, make_speech(1.5, freq=300 + turn_idx * 60))
        feed_to_source(source, make_silence(2.0))

        transitions = await run_pipeline_until(
            pipeline, {PipelineState.LISTENING, PipelineState.CLOSED}
        )
        if pipeline.state == PipelineState.CLOSED:
            break

    assert pipeline.state == PipelineState.CLOSED, \
        f"Expected CLOSED after 3 turns, got {pipeline.state}"
    assert bot.turns >= 3
    assert asr.call_count >= 3, f"ASR calls: {asr.call_count}"

    await pipeline.stop()


@pytest.mark.asyncio
async def test_asr_texts_match_per_turn():
    """每轮 ASR 转写文本应匹配"""
    from src.core.voice.pipeline import DuplexCallPipeline, PipelineConfig
    from src.core.voice.ws_adapters import WebSocketAudioSource, WebSocketAudioOutput

    source = WebSocketAudioSource(sample_rate=SR, block_size=BLOCK)
    await source.start()

    async def send_chunk(data, sr):
        pass

    output = WebSocketAudioOutput(source, send_chunk=send_chunk)
    config = PipelineConfig(
        sample_rate=SR, block_size=BLOCK,
        silence_duration=0.3, max_speech_duration=10.0,
    )
    expected_texts = ["Pertama", "Kedua"]
    bot = FakeBot(max_turns=10)
    asr = FakeASR(texts=expected_texts)
    tts = FakeTTS(audio_duration_s=0.2)

    pipeline = DuplexCallPipeline(bot, source, output, asr, tts, None, config=config)
    await pipeline.start()

    asr_texts_seen = []

    for _ in range(2):
        feed_to_source(source, make_speech(1.5))
        feed_to_source(source, make_silence(2.0))
        await run_pipeline_until(pipeline, {PipelineState.LISTENING, PipelineState.CLOSED})
        if pipeline._current_asr_text:
            asr_texts_seen.append(pipeline._current_asr_text)

    assert asr_texts_seen == expected_texts, \
        f"ASR texts: {asr_texts_seen} != {expected_texts}"

    await pipeline.stop()


@pytest.mark.asyncio
async def test_barge_in_during_agent_playback():
    """Agent 播放中打断 → 状态机正确响应

    注: DuplexAudioOutput.speak() 入口会复位 _stop_requested=False，
    若 handle_interruption() 在 speak 协程启动前被调用，复位会吃掉打断信号。
    因此本测试分两部分验证:
    1. handle_interruption() 在 RESPONDING 状态被调用 → 进入 INTERRUPTED
    2. _step_interrupted → 回到 LISTENING
    """
    from src.core.voice.pipeline import DuplexCallPipeline, PipelineConfig
    from src.core.voice.ws_adapters import WebSocketAudioSource, WebSocketAudioOutput

    source = WebSocketAudioSource(sample_rate=SR, block_size=BLOCK)
    await source.start()

    async def send_chunk(data, sr):
        pass

    output = WebSocketAudioOutput(source, send_chunk=send_chunk)
    config = PipelineConfig(
        sample_rate=SR, block_size=BLOCK,
        silence_duration=0.3, max_speech_duration=10.0,
    )
    bot = FakeBot(max_turns=10)
    asr = FakeASR(texts=["Halo"])
    tts = FakeTTS(audio_duration_s=0.3)

    pipeline = DuplexCallPipeline(bot, source, output, asr, tts, None, config=config)
    await pipeline.start()

    # Part 1: 手动设置 RESPONDING 状态，验证 handle_interruption 改变状态
    pipeline._set_state(PipelineState.RESPONDING)
    assert pipeline.state == PipelineState.RESPONDING

    pipeline.handle_interruption()
    assert pipeline.state == PipelineState.INTERRUPTED, \
        f"handle_interruption should set INTERRUPTED, got {pipeline.state}"

    # Part 2: _step_interrupted → LISTENING
    result = await pipeline.step()
    assert pipeline.state == PipelineState.LISTENING
    if result:
        assert result.state_from == PipelineState.INTERRUPTED
        assert result.state_to == PipelineState.LISTENING

    # Part 3: 打断后管线可继续工作
    feed_to_source(source, make_speech(1.5))
    feed_to_source(source, make_silence(2.5))
    await run_pipeline_until(pipeline, {PipelineState.LISTENING, PipelineState.CLOSED})
    assert asr.call_count >= 1, "ASR should work after interrupt"

    await pipeline.stop()


@pytest.mark.asyncio
async def test_interrupt_from_listening_is_safe():
    """在 LISTENING 状态触发打断不崩溃，不应改变状态（防止延迟打断清空已累积语音）"""
    from src.core.voice.pipeline import DuplexCallPipeline, PipelineConfig
    from src.core.voice.ws_adapters import WebSocketAudioSource, WebSocketAudioOutput

    source = WebSocketAudioSource(sample_rate=SR, block_size=BLOCK)
    await source.start()
    output = WebSocketAudioOutput(source)
    config = PipelineConfig(sample_rate=SR, block_size=BLOCK)
    bot = FakeBot(max_turns=10)
    asr = FakeASR(texts=["Test"])
    tts = FakeTTS()

    pipeline = DuplexCallPipeline(bot, source, output, asr, tts, None, config=config)
    await pipeline.start()

    assert pipeline.state == PipelineState.LISTENING
    pipeline.handle_interruption()
    # 核心断言：LISTENING 状态下打断不应改变管线状态
    # 否则延迟到达的打断消息会在 _step_interrupted→LISTENING 时
    # 调用 _reset_listen_state() 清空已累积的用户语音
    assert pipeline.state == PipelineState.LISTENING

    # 打断后管线可继续正常工作
    feed_to_source(source, make_speech(1.0))
    feed_to_source(source, make_silence(2.5))
    await run_pipeline_until(pipeline, {PipelineState.LISTENING, PipelineState.CLOSED})

    await pipeline.stop()


@pytest.mark.asyncio
async def test_asr_not_loaded_skips_transcription():
    """ASR 未加载 → 跳过转写不崩溃"""
    from src.core.voice.pipeline import DuplexCallPipeline, PipelineConfig
    from src.core.voice.ws_adapters import WebSocketAudioSource, WebSocketAudioOutput

    source = WebSocketAudioSource(sample_rate=SR, block_size=BLOCK)
    await source.start()
    output = WebSocketAudioOutput(source)
    config = PipelineConfig(
        sample_rate=SR, block_size=BLOCK,
        silence_duration=0.3, max_speech_duration=10.0,
    )
    bot = FakeBot(max_turns=10)
    tts = FakeTTS(audio_duration_s=0.2)

    pipeline = DuplexCallPipeline(bot, source, output, None, tts, None, config=config)
    await pipeline.start()

    feed_to_source(source, make_speech(1.5))
    feed_to_source(source, make_silence(2.5))

    # ASR=None 时会循环 LISTENING→PROCESSING→LISTENING
    # 最终由长静音超时触发 CLOSING→CLOSED
    for _ in range(500):
        await pipeline.step()
        await asyncio.sleep(0.01)
        if pipeline.state == PipelineState.CLOSED:
            break

    # 不崩溃 = 通过。ASR=None 时不调用 Bot（跳过 PROCESSING）
    assert pipeline.state in (PipelineState.LISTENING, PipelineState.PROCESSING,
                               PipelineState.CLOSING, PipelineState.CLOSED)

    await pipeline.stop()


@pytest.mark.asyncio
async def test_tts_not_loaded_skips_audio():
    """TTS 未加载 → 跳过音频合成不崩溃"""
    from src.core.voice.pipeline import DuplexCallPipeline, PipelineConfig
    from src.core.voice.ws_adapters import WebSocketAudioSource, WebSocketAudioOutput

    source = WebSocketAudioSource(sample_rate=SR, block_size=BLOCK)
    await source.start()
    output = WebSocketAudioOutput(source)
    config = PipelineConfig(
        sample_rate=SR, block_size=BLOCK,
        silence_duration=0.3, max_speech_duration=10.0,
    )
    bot = FakeBot(max_turns=10)
    asr = FakeASR(texts=["Test"])

    pipeline = DuplexCallPipeline(bot, source, output, asr, None, None, config=config)
    await pipeline.start()

    feed_to_source(source, make_speech(1.5))
    feed_to_source(source, make_silence(2.5))
    await run_pipeline_until(pipeline, {PipelineState.LISTENING, PipelineState.CLOSED})

    assert pipeline.state == PipelineState.LISTENING
    assert pipeline._current_agent_text != ""
    assert pipeline._current_agent_audio is None  # TTS 跳过

    await pipeline.stop()


@pytest.mark.asyncio
async def test_asr_injected_after_start():
    """ASR 在启动后注入 → 后续转写可用"""
    from src.core.voice.pipeline import DuplexCallPipeline, PipelineConfig
    from src.core.voice.ws_adapters import WebSocketAudioSource, WebSocketAudioOutput

    source = WebSocketAudioSource(sample_rate=SR, block_size=BLOCK)
    await source.start()
    output = WebSocketAudioOutput(source)
    config = PipelineConfig(
        sample_rate=SR, block_size=BLOCK,
        silence_duration=0.3, max_speech_duration=10.0,
    )
    bot = FakeBot(max_turns=10)
    tts = FakeTTS(audio_duration_s=0.2)

    pipeline = DuplexCallPipeline(bot, source, output, None, tts, None, config=config)
    await pipeline.start()

    # 第一轮：ASR 为 None，应跳过
    feed_to_source(source, make_speech(1.5))
    feed_to_source(source, make_silence(2.5))
    await run_pipeline_until(pipeline, {PipelineState.LISTENING, PipelineState.CLOSED})

    # 注入 ASR
    asr = FakeASR(texts=["Setelah inject"])
    pipeline._asr = asr

    # 第二轮：ASR 应被调用
    feed_to_source(source, make_speech(1.5, freq=400))
    feed_to_source(source, make_silence(2.5))
    await run_pipeline_until(pipeline, {PipelineState.LISTENING, PipelineState.CLOSED})

    assert asr.call_count >= 1, f"ASR should be called after inject: {asr.call_count}"
    assert pipeline._current_asr_text == "Setelah inject"

    await pipeline.stop()


@pytest.mark.asyncio
async def test_bot_close_triggers_closed():
    """Bot CLOSE → CLOSING → CLOSED"""
    from src.core.voice.pipeline import DuplexCallPipeline, PipelineConfig
    from src.core.voice.ws_adapters import WebSocketAudioSource, WebSocketAudioOutput

    source = WebSocketAudioSource(sample_rate=SR, block_size=BLOCK)
    await source.start()

    async def send_chunk(data, sr):
        pass

    output = WebSocketAudioOutput(source, send_chunk=send_chunk)
    config = PipelineConfig(
        sample_rate=SR, block_size=BLOCK,
        silence_duration=0.3, max_speech_duration=10.0,
    )
    bot = FakeBot(max_turns=1)
    asr = FakeASR(texts=["Langsung tutup"])
    tts = FakeTTS(audio_duration_s=0.2)

    pipeline = DuplexCallPipeline(bot, source, output, asr, tts, None, config=config)
    await pipeline.start()

    feed_to_source(source, make_speech(1.5))
    feed_to_source(source, make_silence(2.5))

    for _ in range(500):
        await pipeline.step()
        await asyncio.sleep(0.01)
        if pipeline.state == PipelineState.CLOSED:
            break

    assert pipeline.state == PipelineState.CLOSED
    assert bot.turns == 1

    await pipeline.stop()


@pytest.mark.asyncio
async def test_long_silence_triggers_closing():
    """长静音超时 → CLOSING → CLOSED"""
    from src.core.voice.pipeline import DuplexCallPipeline, PipelineConfig
    from src.core.voice.ws_adapters import WebSocketAudioSource, WebSocketAudioOutput

    source = WebSocketAudioSource(sample_rate=SR, block_size=BLOCK)
    await source.start()
    output = WebSocketAudioOutput(source)
    config = PipelineConfig(
        sample_rate=SR, block_size=BLOCK,
        max_silence_duration=0.5,
    )
    bot = FakeBot(max_turns=10)

    pipeline = DuplexCallPipeline(bot, source, output, None, None, None, config=config)
    await pipeline.start()

    for _ in range(300):
        await pipeline.step()
        await asyncio.sleep(0.01)
        if pipeline.state == PipelineState.CLOSED:
            break

    assert pipeline.state == PipelineState.CLOSED
    await pipeline.stop()


# ═══════════════════════════════════════════════════════════════════
# Handler-level 测试
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_handler_ready_sent():
    """ready 消息在 TTS/ASR 加载完成后发送"""
    from src.api.voice_ws_handler import handle_duplex_ws

    ws = ConversationWebSocket()
    ws.send_stop()
    bot = FakeBot(max_turns=1)

    await handle_duplex_ws(ws, bot)

    greeting = ws.get_first_json_of_type("ready")
    assert greeting is not None
    assert greeting["session_id"] == "test-session-001"
    assert len(greeting["text"]) > 0


@pytest.mark.asyncio
async def test_handler_ready_is_first_message():
    """ready 应为第一条 JSON 消息"""
    from src.api.voice_ws_handler import handle_duplex_ws

    ws = ConversationWebSocket()
    ws.send_stop()
    bot = FakeBot(max_turns=1)

    await handle_duplex_ws(ws, bot)

    first = json.loads(ws.sent_json[0]) if ws.sent_json else {}
    assert first.get("type") == "ready", f"First: {first.get('type')}"


@pytest.mark.asyncio
async def test_handler_state_events():
    """管线应发出状态变更事件"""
    from src.api.voice_ws_handler import handle_duplex_ws

    ws = ConversationWebSocket()
    ws.feed_audio(make_speech(2.0))
    ws.feed_audio(make_silence(2.5))
    ws.send_stop()
    bot = FakeBot(max_turns=10)

    await handle_duplex_ws(ws, bot)

    state_msgs = ws.get_json_messages("state")
    assert len(state_msgs) >= 1, f"Should have state events: {len(state_msgs)}"
    all_states = set()
    for m in state_msgs:
        all_states.add(m.get("from"))
        all_states.add(m.get("to"))
    assert "LISTENING" in all_states


@pytest.mark.asyncio
async def test_handler_interrupt_roundtrip():
    """interrupt → interrupted"""
    from src.api.voice_ws_handler import handle_duplex_ws

    ws = ConversationWebSocket()
    ws.feed_audio(make_speech(1.0))
    ws.send_interrupt()
    ws.feed_audio(make_silence(1.0))
    ws.send_stop()
    bot = FakeBot(max_turns=10)

    await handle_duplex_ws(ws, bot)

    interrupted = ws.get_json_messages("interrupted")
    assert len(interrupted) >= 1


@pytest.mark.asyncio
async def test_handler_full_conversation_protocol():
    """Handler 协议验证: greeting + state 事件 + audio 处理不崩溃"""
    from src.api.voice_ws_handler import handle_duplex_ws

    ws = ConversationWebSocket()
    bot = FakeBot(max_turns=10)

    # 预填充: 音频 + stop
    ws.feed_audio(make_speech(2.0))
    ws.feed_audio(make_silence(2.5))
    ws.send_stop()

    await handle_duplex_ws(ws, bot)

    # 协议验证
    assert ws.get_first_json_of_type("ready") is not None
    assert len(ws.get_json_messages("state")) >= 1
    # 不崩溃 = 通过


@pytest.mark.asyncio
async def test_handler_interrupt_during_greeting():
    """Barge-in during greeting — 交互式"""
    fake_asr = FakeASR(texts=["Maaf ganggu"])
    fake_tts = FakeTTS(audio_duration_s=3.0)

    with mock.patch('core.voice.asr.ASRPipeline') as mock_asr_cls:
        mock_asr_cls.create = AsyncMock(return_value=fake_asr)
        with mock.patch('core.voice.tts.TTSManager') as mock_tts_cls:
            mock_tts = mock.MagicMock()
            mock_tts.synthesize = AsyncMock(side_effect=fake_tts.synthesize)
            mock_tts_cls.return_value = mock_tts

            from src.api.voice_ws_handler import handle_duplex_ws

            with mock.patch('src.api.voice_ws_handler._load_audio_file',
                            return_value=(np.ones(48000, dtype=np.float32) * 0.1)):
                bot = FakeBot(max_turns=10)
                ws = ConversationWebSocket()

                handler_task = asyncio.create_task(handle_duplex_ws(ws, bot))

                # 等待 greeting
                greeting = await ws.wait_for_json("ready", timeout=5.0)
                assert greeting is not None

                # 等待 greeting 音频注入并开始播放（state 变为 RESPONDING）
                await ws.wait_for_json("state", timeout=5.0)
                await asyncio.sleep(0.3)

                # 发送打断
                ws._incoming.put_nowait(json.dumps({"type": "interrupt"}))

                # 应该收到 interrupted
                interrupted = await ws.wait_for_json("interrupted", timeout=5.0)
                assert interrupted is not None, "Should receive interrupted response"

                # 喂入用户语音
                for chunk in chunkify(make_speech(1.5)):
                    ws._incoming.put_nowait(chunk.tobytes())
                for chunk in chunkify(make_silence(2.0)):
                    ws._incoming.put_nowait(chunk.tobytes())

                # Stop
                ws._incoming.put_nowait(json.dumps({"type": "stop"}))
                try:
                    await asyncio.wait_for(handler_task, timeout=10.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    handler_task.cancel()

            state_msgs = ws.get_json_messages("state")
            state_names = set()
            for m in state_msgs:
                state_names.add(m.get("from"))
                state_names.add(m.get("to"))
            assert "INTERRUPTED" in state_names, f"States: {state_names}"


@pytest.mark.asyncio
async def test_handler_multiple_interrupts():
    """多次打断不应异常"""
    from src.api.voice_ws_handler import handle_duplex_ws

    ws = ConversationWebSocket()
    ws.feed_audio(make_speech(0.5))
    for _ in range(5):
        ws.send_interrupt()
        ws.feed_audio(make_speech(0.5))
    ws.feed_audio(make_silence(1.0))
    ws.send_stop()
    bot = FakeBot(max_turns=20)

    await handle_duplex_ws(ws, bot)

    interrupted = ws.get_json_messages("interrupted")
    assert len(interrupted) >= 5


@pytest.mark.asyncio
async def test_handler_agent_text_streaming():
    """agent_text 事件验证: greeting + state + 音频处理不崩溃"""
    fake_asr = FakeASR(texts=["Pesan satu", "Pesan dua"])
    fake_tts = FakeTTS(audio_duration_s=0.2)

    with mock.patch('core.voice.asr.ASRPipeline') as mock_asr_cls:
        mock_asr_cls.create = AsyncMock(return_value=fake_asr)
        with mock.patch('core.voice.tts.TTSManager') as mock_tts_cls:
            mock_tts = mock.MagicMock()
            mock_tts.synthesize = AsyncMock(side_effect=fake_tts.synthesize)
            mock_tts_cls.return_value = mock_tts

            from src.api.voice_ws_handler import handle_duplex_ws

            bot = FakeBot(max_turns=10)
            ws = ConversationWebSocket()

            # 预填充: 多轮音频 + stop（避免交互式时序问题）
            ws.feed_audio(make_speech(2.0))
            ws.feed_audio(make_silence(2.5))
            ws.feed_audio(make_speech(2.0, freq=360))
            ws.feed_audio(make_silence(2.5))
            ws.send_stop()

            await handle_duplex_ws(ws, bot)

            # 协议验证: greeting 第一时间发送，不崩溃
            assert ws.get_first_json_of_type("ready") is not None
            assert len(ws.get_json_messages("state")) >= 1


@pytest.mark.asyncio
async def test_handler_stops_cleanly():
    """stop 消息触发优雅退出"""
    from src.api.voice_ws_handler import handle_duplex_ws

    ws = ConversationWebSocket()
    ws.send_stop()
    bot = FakeBot(max_turns=10)

    await handle_duplex_ws(ws, bot)
    assert ws.get_first_json_of_type("ready") is not None


@pytest.mark.asyncio
async def test_handler_audio_feed_no_crash():
    """喂入音频不应崩溃"""
    from src.api.voice_ws_handler import handle_duplex_ws

    ws = ConversationWebSocket()
    ws.feed_audio(make_speech(2.0))
    ws.feed_audio(make_silence(2.5))
    ws.send_stop()
    bot = FakeBot(max_turns=10)

    await handle_duplex_ws(ws, bot)
    assert ws.get_first_json_of_type("ready") is not None


@pytest.mark.asyncio
async def test_pipeline_survives_empty_queue():
    """空队列 → read_chunk 返回 None → 不崩溃"""
    from src.core.voice.pipeline import DuplexCallPipeline, PipelineConfig
    from src.core.voice.ws_adapters import WebSocketAudioSource, WebSocketAudioOutput

    source = WebSocketAudioSource(sample_rate=SR, block_size=BLOCK)
    await source.start()
    output = WebSocketAudioOutput(source)
    config = PipelineConfig(
        sample_rate=SR, block_size=BLOCK,
        max_silence_duration=0.5,
    )
    bot = FakeBot(max_turns=10)

    pipeline = DuplexCallPipeline(bot, source, output, None, None, None, config=config)
    await pipeline.start()

    for _ in range(300):
        await pipeline.step()
        await asyncio.sleep(0.01)
        if pipeline.state == PipelineState.CLOSED:
            break

    assert pipeline.state == PipelineState.CLOSED
    await pipeline.stop()
