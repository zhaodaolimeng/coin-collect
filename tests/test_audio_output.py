"""DuplexAudioOutput 单元测试"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pytest
import asyncio
from core.voice.audio_output import DuplexAudioOutput, PlaybackResult
from core.voice.audio_source import SilentSource


def make_silence(duration_s: float, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(duration_s * sr), dtype=np.float32)


def make_tone(duration_s: float, freq: float = 440.0, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(duration_s * sr)) / sr
    return (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)


@pytest.mark.asyncio
async def test_playback_completes_normally():
    """安静环境下播放完整"""
    source = SilentSource()
    await source.start()
    output = DuplexAudioOutput(source, barge_in_threshold=0.99)
    audio = make_silence(0.2)
    result = await output.speak(audio)
    assert result == PlaybackResult.COMPLETED
    await source.stop()


@pytest.mark.asyncio
async def test_stop_immediately():
    """stop() 立即停止播放"""
    source = SilentSource()
    await source.start()
    output = DuplexAudioOutput(source, barge_in_threshold=0.99)
    audio = make_silence(3.0)
    task = asyncio.create_task(output.speak(audio))
    await asyncio.sleep(0.15)  # 等待第一个 chunk 播放完成
    output.stop()
    result = await task
    assert result == PlaybackResult.INTERRUPTED
    await source.stop()


def test_playback_result_enum_values():
    """PlaybackResult 枚举值正确"""
    assert PlaybackResult.COMPLETED.value == "completed"
    assert PlaybackResult.INTERRUPTED.value == "interrupted"
    assert PlaybackResult.FAILED.value == "failed"


def test_duplex_output_initial_state():
    """初始状态正确"""
    source = SilentSource()
    output = DuplexAudioOutput(source)
    assert not output.is_speaking
    assert not output.is_ducking


@pytest.mark.asyncio
async def test_ducking_recovery_false_alarm():
    """短暂噪音后恢复正常播放（静音源不触发打断）"""
    source = SilentSource()
    await source.start()
    output = DuplexAudioOutput(
        source, barge_in_threshold=0.99,
        confirmation_duration=0.1,
    )
    audio = make_tone(0.3)
    result = await output.speak(audio)
    assert result == PlaybackResult.COMPLETED
    await source.stop()


@pytest.mark.asyncio
async def test_barge_in_threshold_stored():
    """barge_in_threshold 参数正确存储"""
    source = SilentSource()
    output = DuplexAudioOutput(source, barge_in_threshold=0.05)
    assert output._barge_in_threshold == 0.05
