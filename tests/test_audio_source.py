"""AudioSource 单元测试"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pytest
import asyncio
from core.voice.audio_source import AudioSource, MicrophoneSource, SilentSource


class DummySource(AudioSource):
    """最小实现用于测试 ABC"""
    def __init__(self):
        self._sr = 16000
        self._started = False
        self._stopped = False
        self._chunks = [np.zeros(1600, dtype=np.float32)]

    async def start(self):
        self._started = True

    async def stop(self):
        self._stopped = True

    async def read_chunk(self):
        if self._chunks:
            return self._chunks.pop(0)
        return None

    def current_rms(self):
        return 0.0

    @property
    def sample_rate(self):
        return self._sr


def test_audio_source_abc_enforces_interface():
    """ABC 不能直接实例化"""
    with pytest.raises(TypeError):
        AudioSource()


def test_audio_source_concrete_subclass_works():
    """具体实现可以实例化并调用"""
    src = DummySource()
    assert src.sample_rate == 16000
    assert src.is_real_time() is True


def test_microphone_source_requires_sounddevice():
    """MicrophoneSource 创建不抛异常，start 时检测 sounddevice"""
    src = MicrophoneSource()
    assert src.sample_rate == 16000
    asyncio.run(src.start())
    asyncio.run(src.stop())


def test_microphone_source_rms_defaults_zero():
    """未启动时 RMS 为 0"""
    src = MicrophoneSource()
    assert src.current_rms() == 0.0


def test_silent_source_produces_silence():
    """SilentSource 产生静音块"""
    src = SilentSource(sample_rate=16000, block_size=1600)
    asyncio.run(src.start())
    chunk = asyncio.run(src.read_chunk())
    assert chunk is not None
    assert len(chunk) == 1600
    assert np.max(np.abs(chunk)) == 0.0
    asyncio.run(src.stop())
