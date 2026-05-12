"""DuplexCallPipeline 单元测试 — 状态机 + 集成测试"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import tempfile
import numpy as np
import pytest
import asyncio
from core.voice.pipeline import DuplexCallPipeline, PipelineState, PipelineConfig, StepResult, InterruptionContext
from core.voice.audio_source import SilentSource, FileSource
from core.voice.audio_output import DuplexAudioOutput


class FakeASR:
    """测试用 ASR"""
    def __init__(self, fixed_text: str = "Ya"):
        self.fixed_text = fixed_text
        self.is_available = True
        self.transcribe_count = 0

    def transcribe(self, audio: np.ndarray) -> str:
        self.transcribe_count += 1
        return self.fixed_text

    async def transcribe_async(self, audio: np.ndarray) -> str:
        return self.transcribe(audio)


class FakeTTS:
    """测试用 TTS — 返回静音"""
    async def synthesize(self, text, output_file=None, voice=None, engine=None, **kwargs):
        from core.voice.tts import TTSResult
        arr = np.zeros(8000, dtype=np.float32)
        return TTSResult(text=text, audio_data=arr.tobytes(), audio_file=None, success=True, engine_name="fake")


class FakeVAD:
    """测试用 VAD — 固定返回语音活动"""
    def __init__(self, voice_duration_frames: int = 50):
        self.frame_count = 0
        self.voice_duration = voice_duration_frames
        self.reset_calls = 0

    def process_frame(self, audio_frame):
        from core.voice.vad import VADResult, VADState
        self.frame_count += 1
        if self.frame_count <= self.voice_duration:
            return VADResult(state=VADState.VOICE, confidence=0.9, timestamp=0)
        return VADResult(state=VADState.SILENCE, confidence=0.9, timestamp=0)

    def reset(self):
        self.reset_calls += 1
        self.frame_count = 0


class FakeBot:
    """测试用 Chatbot"""
    def __init__(self):
        self.turns_processed = 0
        self.commit_time = None
        self._state = _FakeInitState()

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    async def process(self, customer_input=None, use_tts=False):
        self.turns_processed += 1
        if self.turns_processed >= 3:
            self._state = _FakeCloseState()
            return "Terima kasih, selamat tinggal.", None
        return "Baik, saya catat ya.", None


class _FakeInitState:
    name = "INIT"

class _FakeCloseState:
    name = "CLOSE"


def make_pipeline(**kwargs):
    """工厂：创建最小可测 Pipeline"""
    config = PipelineConfig(sample_rate=16000, block_size=1600, silence_duration=0.2, max_speech_duration=5.0)
    source = SilentSource()
    output = DuplexAudioOutput(source, barge_in_threshold=0.99)
    asr = FakeASR()
    tts = FakeTTS()
    vad = FakeVAD(voice_duration_frames=30)
    bot = FakeBot()
    return DuplexCallPipeline(bot, source, output, asr, tts, vad, config=config)


@pytest.mark.asyncio
async def test_pipeline_starts_in_idle():
    pipeline = make_pipeline()
    assert pipeline.state == PipelineState.IDLE


@pytest.mark.asyncio
async def test_pipeline_start_transitions_to_listening():
    pipeline = make_pipeline()
    await pipeline.start()
    assert pipeline.state == PipelineState.LISTENING


@pytest.mark.asyncio
async def test_pipeline_stop_transitions_to_closed():
    pipeline = make_pipeline()
    await pipeline.start()
    await pipeline.stop()
    assert pipeline.state == PipelineState.CLOSED


@pytest.mark.asyncio
async def test_step_from_listening_to_processing():
    pipeline = make_pipeline()
    await pipeline.start()
    for _ in range(35):
        await pipeline.step()
    assert pipeline.state in (PipelineState.PROCESSING, PipelineState.RESPONDING, PipelineState.LISTENING)


@pytest.mark.asyncio
async def test_pipeline_full_cycle_to_close():
    """完整走完到 CLOSED"""
    pipeline = make_pipeline()
    await pipeline.start()
    for _ in range(200):
        if pipeline.state == PipelineState.CLOSED:
            break
        await pipeline.step()
        await asyncio.sleep(0.01)
    assert pipeline.state == PipelineState.CLOSED


@pytest.mark.asyncio
async def test_pipeline_state_callback_fires():
    """状态变化回调被调用"""
    pipeline = make_pipeline()
    states = []
    pipeline.on_state_change = lambda old, new: states.append((old, new))
    await pipeline.start()
    assert len(states) >= 1
    assert states[0] == (PipelineState.IDLE, PipelineState.LISTENING)


# ═══════════════════════════════════════════════════════════════════
# 集成测试 — FileSource + 全流程
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_pipeline_with_filesource():
    """Pipeline + FileSource 完整对话"""
    from scipy.io import wavfile
    from core.voice.vad import SimpleEnergyVAD
    sr = 16000
    data = (np.sin(2 * np.pi * 300 * np.arange(sr) / sr) * 0.3).astype(np.float32)
    path = tempfile.mktemp(suffix=".wav")
    wavfile.write(path, sr, data)

    try:
        source = FileSource(path, sample_rate=sr, block_size=1600, loop=True)
        output = DuplexAudioOutput(source, barge_in_threshold=0.99)
        vad = SimpleEnergyVAD(sample_rate=sr, energy_threshold=0.01, voice_frames=2, silence_frames=5)
        config = PipelineConfig(sample_rate=sr, block_size=1600, silence_duration=0.3, max_speech_duration=5.0)
        pipeline = DuplexCallPipeline(FakeBot(), source, output, FakeASR(), FakeTTS(), vad, config=config)

        await pipeline.start()
        for _ in range(200):
            if pipeline.state == PipelineState.CLOSED:
                break
            await pipeline.step()
            await asyncio.sleep(0.005)

        assert pipeline.state == PipelineState.CLOSED
        await pipeline.stop()
    finally:
        import os; os.unlink(path)


def test_step_result_fields():
    """StepResult 包含正确字段"""
    result = StepResult(
        state_from=PipelineState.LISTENING,
        state_to=PipelineState.PROCESSING,
        asr_text="Ya",
        agent_text="Baik",
        turn_id=1,
        elapsed_s=0.1,
    )
    assert result.turn_id == 1
    assert result.asr_text == "Ya"
    assert not result.interrupted


def test_interruption_context():
    """InterruptionContext 数据类"""
    ctx = InterruptionContext(
        agent_text_interrupted="Baik, jadi besok jam 5 ya?",
        agent_playback_position=0.6,
        customer_rms_peak=0.15,
    )
    assert ctx.agent_playback_position == 0.6
    assert "besok" in ctx.agent_text_interrupted
