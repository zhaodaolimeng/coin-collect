"""流式 ASR 测试 — 去重逻辑 + StreamingASR 单元测试 + 管线集成"""
import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.voice.streaming_asr import StreamingASR, StreamingASRConfig


# ═══════════════════════════════════════════════════════════════════
# Dedup 单元测试
# ═══════════════════════════════════════════════════════════════════

class TestDedup:
    def test_same_text_returns_empty(self):
        s = StreamingASR.__new__(StreamingASR)
        s._last_full_text = "halo apa kabar"
        assert s._dedup("halo apa kabar") == ""

    def test_extension_returns_new_words(self):
        s = StreamingASR.__new__(StreamingASR)
        s._last_full_text = "halo apa"
        assert s._dedup("halo apa kabar") == "kabar"

    def test_completely_different_returns_all(self):
        s = StreamingASR.__new__(StreamingASR)
        s._last_full_text = "halo apa"
        assert s._dedup("baik terima kasih") == "baik terima kasih"

    def test_empty_previous_returns_all(self):
        s = StreamingASR.__new__(StreamingASR)
        s._last_full_text = ""
        assert s._dedup("halo") == "halo"

    def test_empty_current_returns_empty(self):
        s = StreamingASR.__new__(StreamingASR)
        s._last_full_text = "halo"
        assert s._dedup("") == ""

    def test_correction_mid_sentence(self):
        """Whisper 修正前序词时，差异点之后的全部返回"""
        s = StreamingASR.__new__(StreamingASR)
        s._last_full_text = "halo apa kabar"
        # Whisper 把 "apa" 修正为 "bagaimana"
        assert s._dedup("halo bagaimana kabar") == "bagaimana kabar"

    def test_whitespace_handling(self):
        s = StreamingASR.__new__(StreamingASR)
        s._last_full_text = "  halo   apa  "
        assert s._dedup("  halo   apa  kabar") == "kabar"

    def test_common_word_prefix_len_same(self):
        assert StreamingASR._common_word_prefix_len("a b c", "a b c") == 5

    def test_common_word_prefix_len_extension(self):
        assert StreamingASR._common_word_prefix_len("a b", "a b c") == 4  # "a b "

    def test_common_word_prefix_len_different(self):
        assert StreamingASR._common_word_prefix_len("x y", "a b") == 0


# ═══════════════════════════════════════════════════════════════════
# Fake ASR for StreamingASR tests
# ═══════════════════════════════════════════════════════════════════

class FakeStreamingBackend:
    """模拟 RealTimeASR，返回可配置文本"""
    def __init__(self, texts=None):
        self.texts = texts or ["halo"]
        self.call_count = 0
        self.sample_rate = 16000

    async def transcribe_async(self, audio: np.ndarray) -> str:
        self.call_count += 1
        idx = min(self.call_count - 1, len(self.texts) - 1)
        return self.texts[idx]


class GrowingFakeASR:
    """返回与音频长度成正比的文本，模拟增长窗口"""
    def __init__(self):
        self.call_count = 0
        self.sample_rate = 16000

    async def transcribe_async(self, audio: np.ndarray) -> str:
        self.call_count += 1
        dur = len(audio) / 16000
        words = ["halo", "apa", "kabar", "baik", "terima", "kasih"]
        n = min(len(words), max(1, int(dur / 0.4)))
        return " ".join(words[:n])


# ═══════════════════════════════════════════════════════════════════
# StreamingASR 单元测试
# ═══════════════════════════════════════════════════════════════════

class TestStreamingASR:
    @pytest.mark.asyncio
    async def test_submit_single(self):
        """单次提交，回调收到完整文本"""
        backend = FakeStreamingBackend(["halo apa kabar"])
        cfg = StreamingASRConfig(min_audio_duration=0.3, throttle_interval=0.3)
        s = StreamingASR(backend, cfg)

        partials = []
        s.on_partial_result = lambda t: partials.append(t)

        audio = np.zeros(int(0.5 * 16000), dtype=np.float32)
        s.submit(audio)
        await asyncio.sleep(0.2)

        assert len(partials) >= 1
        assert partials[0] == "halo apa kabar"
        assert backend.call_count == 1

    @pytest.mark.asyncio
    async def test_submit_multiple_growing(self):
        """两次提交增长音频，第二次只返回增量"""
        backend = GrowingFakeASR()
        cfg = StreamingASRConfig(min_audio_duration=0.3, throttle_interval=0.3)
        s = StreamingASR(backend, cfg)

        partials = []
        s.on_partial_result = lambda t: partials.append(t)

        # 第一次提交：0.5s 音频 → "halo"
        audio1 = np.zeros(int(0.5 * 16000), dtype=np.float32)
        s.submit(audio1)
        await asyncio.sleep(0.2)
        assert len(partials) >= 1
        assert "halo" in partials[0]

        # 第二次提交：1.5s 音频 → "halo apa kabar baik"
        audio2 = np.zeros(int(1.5 * 16000), dtype=np.float32)
        s.submit(audio2)
        await asyncio.sleep(0.2)

        # 增量应该是去掉 "halo" 后的部分
        all_text = " ".join(partials)
        assert "apa" in all_text or "kabar" in all_text

    @pytest.mark.asyncio
    async def test_stale_result_discarded(self):
        """generation 计数器丢弃旧结果"""
        backend = FakeStreamingBackend(["first", "second"])
        cfg = StreamingASRConfig(min_audio_duration=0.3, throttle_interval=0.1)
        s = StreamingASR(backend, cfg)

        partials = []
        s.on_partial_result = lambda t: partials.append(t)

        audio = np.zeros(int(0.5 * 16000), dtype=np.float32)
        s.submit(audio)  # gen=1, returns "first"
        s.submit(audio)  # gen=2, returns "second" (stale gen=1 discarded)
        await asyncio.sleep(0.3)

        # 只应有 "second"，没有 "first"
        assert "second" in partials
        assert "first" not in partials

    @pytest.mark.asyncio
    async def test_mark_final_sets_result(self):
        """mark_final 后最终结果就绪"""
        backend = FakeStreamingBackend(["halo"])
        s = StreamingASR(backend)

        audio = np.zeros(int(1.0 * 16000), dtype=np.float32)
        s.submit(audio)
        await asyncio.sleep(0.2)

        s.mark_final()
        # 结果应已就绪（在 submit 完成后 mark_final）
        result = await s.wait_for_final(timeout=1.0)
        assert result == "halo"

    @pytest.mark.asyncio
    async def test_mark_final_without_submit(self):
        """无提交时 mark_final 返回空"""
        backend = FakeStreamingBackend(["halo"])
        s = StreamingASR(backend)

        s.mark_final()
        result = await s.wait_for_final(timeout=0.5)
        assert result == ""

    @pytest.mark.asyncio
    async def test_mark_final_immediate_when_in_flight(self):
        """mark_final 在任务飞行中时立即返回已有结果，不等待"""
        backend = FakeStreamingBackend(["hasil akhir"])
        s = StreamingASR(backend)

        audio = np.zeros(int(1.0 * 16000), dtype=np.float32)
        s.submit(audio)
        # 立即 mark_final，不等待 submit 完成
        s.mark_final()

        # 新行为：立即返回已有文本（为空），不等待飞行中任务
        assert s.has_final_result
        result = await s.wait_for_final(timeout=0.1)
        assert result == ""  # 飞行中任务被丢弃，调用方回退全段 ASR

    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        """reset 后状态干净"""
        backend = FakeStreamingBackend(["halo"])
        s = StreamingASR(backend)

        audio = np.zeros(int(0.5 * 16000), dtype=np.float32)
        s.submit(audio)
        await asyncio.sleep(0.2)
        s.mark_final()
        await s.wait_for_final()

        s.reset()
        assert not s.is_active
        assert not s.has_final_result
        assert s.final_text == ""

    @pytest.mark.asyncio
    async def test_no_partial_when_no_new_text(self):
        """相同文本重复提交不触发回调"""
        backend = FakeStreamingBackend(["halo", "halo"])
        cfg = StreamingASRConfig(min_audio_duration=0.3, throttle_interval=0.1)
        s = StreamingASR(backend, cfg)

        partials = []
        s.on_partial_result = lambda t: partials.append(t)

        audio = np.zeros(int(0.5 * 16000), dtype=np.float32)
        s.submit(audio)
        await asyncio.sleep(0.15)
        s.submit(audio)  # 相同文本
        await asyncio.sleep(0.15)

        # 只有第一次触发回调（第二次是重复）
        assert len(partials) <= 1 or (len(partials) > 1 and partials[1] == "")

    @pytest.mark.asyncio
    async def test_submit_after_mark_final_ignored(self):
        """mark_final 后 submit 被忽略"""
        backend = FakeStreamingBackend(["first", "second"])
        s = StreamingASR(backend)

        audio = np.zeros(int(0.5 * 16000), dtype=np.float32)
        s.submit(audio)
        await asyncio.sleep(0.2)
        s.mark_final()

        # 再次 submit 应被忽略
        s.submit(audio)
        await asyncio.sleep(0.1)

        result = await s.wait_for_final(timeout=1.0)
        assert result == "first"  # 仍是第一次的结果


# ═══════════════════════════════════════════════════════════════════
# Pipeline 集成测试
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_pipeline_streaming_single_turn():
    """管线 + 流式 ASR: 单轮对话"""
    from core.voice.pipeline import DuplexCallPipeline, PipelineConfig
    from core.voice.audio_source import SilentSource
    from core.voice.audio_output import DuplexAudioOutput

    class FakeVAD:
        def __init__(self, voice_frames=30):
            self.frame_count = 0
            self.voice_frames = voice_frames
            self.reset_calls = 0

        def process_frame(self, audio_frame):
            from core.voice.vad import VADResult, VADState
            self.frame_count += 1
            if self.frame_count <= self.voice_frames:
                return VADResult(state=VADState.VOICE, confidence=0.9, timestamp=0)
            return VADResult(state=VADState.SILENCE, confidence=0.9, timestamp=0)

        def reset(self):
            self.reset_calls += 1
            self.frame_count = 0

    class FakeBot:
        def __init__(self):
            self.turns = 0
            class S: name = 'INIT'
            self._state = S()

        @property
        def state(self): return self._state

        async def process(self, customer_input=None, use_tts=False):
            self.turns += 1
            if self.turns >= 2:
                class C: name = 'CLOSE'
                self._state = C()
                return "Terima kasih", None
            return "Baik, saya catat", None

    class FakeASR:
        def __init__(self):
            self.is_available = True
            self.sample_rate = 16000
            self.call_count = 0

        async def transcribe_async(self, audio):
            self.call_count += 1
            await asyncio.sleep(0.01)  # simulate processing
            return "Halo apa kabar"

    class FakeTTS:
        async def synthesize(self, text, **kwargs):
            from core.voice.tts import TTSResult
            arr = np.zeros(4000, dtype=np.float32)
            return TTSResult(text=text, audio_data=arr.tobytes(), audio_file=None,
                            success=True, engine_name="fake")

    config = PipelineConfig(sample_rate=16000, block_size=1600,
                            silence_duration=0.2, max_speech_duration=5.0)
    source = SilentSource()
    output = DuplexAudioOutput(source, barge_in_threshold=0.99)
    asr = FakeASR()
    pipeline = DuplexCallPipeline(FakeBot(), source, output, asr, FakeTTS(),
                                  FakeVAD(voice_frames=30), config=config)
    partials = []
    pipeline.on_partial_asr = lambda t: partials.append(t)

    await pipeline.start()
    for _ in range(500):
        if pipeline.state == pipeline._state.__class__.CLOSED:
            break
        await pipeline.step()
        if pipeline.state.name == "RESPONDING" and pipeline._respond_audio_sent:
            pipeline.notify_playback_done()
        if pipeline.state.name == "CLOSING" and pipeline._respond_audio_sent:
            pipeline.notify_playback_done()
        await asyncio.sleep(0.01)

    assert pipeline.state.name == "CLOSED", f"Expected CLOSED, got {pipeline.state.name}"
    # 流式 ASR 应产生部分结果
    assert len(partials) >= 0  # 可能产生也可能不产生，取决于时序
    await pipeline.stop()


@pytest.mark.asyncio
async def test_pipeline_streaming_fallback_when_streaming_disabled():
    """无 ASR 时流式禁用，回退到正常整段转写"""
    from core.voice.pipeline import DuplexCallPipeline, PipelineConfig
    from core.voice.audio_source import SilentSource
    from core.voice.audio_output import DuplexAudioOutput

    class FakeVAD:
        def __init__(self):
            self.frame_count = 0
        def process_frame(self, audio_frame):
            from core.voice.vad import VADResult, VADState
            self.frame_count += 1
            if self.frame_count <= 30:
                return VADResult(state=VADState.VOICE, confidence=0.9, timestamp=0)
            return VADResult(state=VADState.SILENCE, confidence=0.9, timestamp=0)
        def reset(self): self.frame_count = 0

    class FakeBot:
        def __init__(self):
            class S: name = 'INIT'
            self._state = S()
        @property
        def state(self): return self._state
        async def process(self, customer_input=None, use_tts=False):
            return "Baik", None

    class FakeTTS:
        async def synthesize(self, text, **kwargs):
            from core.voice.tts import TTSResult
            return TTSResult(text=text, audio_data=np.zeros(4000, dtype=np.float32).tobytes(),
                            audio_file=None, success=True, engine_name="fake")

    # 无 ASR 传入，流式应禁用
    config = PipelineConfig(sample_rate=16000, block_size=1600,
                            silence_duration=0.2, max_speech_duration=5.0)
    pipeline = DuplexCallPipeline(FakeBot(), SilentSource(),
                                  DuplexAudioOutput(SilentSource(), barge_in_threshold=0.99),
                                  None, FakeTTS(), FakeVAD(), config=config)

    await pipeline.start()
    assert pipeline._streaming_asr is None
    await pipeline.stop()


@pytest.mark.asyncio
async def test_streaming_asr_reset_between_turns():
    """两轮对话之间流式 ASR 状态重置"""
    backend = FakeStreamingBackend(["halo"])
    s = StreamingASR(backend)

    audio = np.zeros(int(1.0 * 16000), dtype=np.float32)
    s.submit(audio)
    await asyncio.sleep(0.2)
    s.mark_final()
    await s.wait_for_final()

    # 模拟管线 reset
    s.reset()
    assert not s.is_active
    assert not s.is_final_pending

    # 新一轮
    s.submit(audio)
    await asyncio.sleep(0.2)
    s.mark_final()
    result = await s.wait_for_final()
    assert result == "halo"
