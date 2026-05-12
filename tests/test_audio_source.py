"""AudioSource 单元测试"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pytest
import asyncio
from core.voice.audio_source import AudioSource, MicrophoneSource, SilentSource, FileSource, SimulatedSource


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


# ═══════════════════════════════════════════════════════════════════
# FileSource 测试
# ═══════════════════════════════════════════════════════════════════

def test_file_source_loads_wav():
    """FileSource 读取 WAV 文件"""
    import tempfile
    from scipy.io import wavfile
    sr = 16000
    data = (np.sin(2 * np.pi * 440 * np.arange(sr) / sr) * 0.5).astype(np.float32)
    path = tempfile.mktemp(suffix=".wav")
    wavfile.write(path, sr, data)
    try:
        src = FileSource(path, sample_rate=sr, block_size=1600)
        asyncio.run(src.start())
        chunks = []
        while True:
            c = asyncio.run(src.read_chunk())
            if c is None:
                break
            chunks.append(c)
        asyncio.run(src.stop())
        assert len(chunks) > 0
        total = sum(len(c) for c in chunks)
        assert total >= len(data) - 1600
    finally:
        import os; os.unlink(path)


def test_file_source_reads_in_order():
    """FileSource 按顺序读取，分块正确"""
    import tempfile
    from scipy.io import wavfile
    sr = 16000
    data = np.arange(32000, dtype=np.float32) / 32000.0
    path = tempfile.mktemp(suffix=".wav")
    wavfile.write(path, sr, data)
    try:
        src = FileSource(path, sample_rate=sr, block_size=1600)
        asyncio.run(src.start())
        c1 = asyncio.run(src.read_chunk())
        c2 = asyncio.run(src.read_chunk())
        asyncio.run(src.stop())
        np.testing.assert_array_almost_equal(c1, data[:1600])
        np.testing.assert_array_almost_equal(c2, data[1600:3200])
    finally:
        import os; os.unlink(path)


def test_file_source_eof_returns_none():
    """读完文件后返回 None"""
    import tempfile
    from scipy.io import wavfile
    sr = 16000
    data = np.zeros(800, dtype=np.float32)
    path = tempfile.mktemp(suffix=".wav")
    wavfile.write(path, sr, data)
    try:
        src = FileSource(path, sample_rate=sr, block_size=1600)
        asyncio.run(src.start())
        c1 = asyncio.run(src.read_chunk())
        assert c1 is not None
        c2 = asyncio.run(src.read_chunk())
        assert c2 is None
        asyncio.run(src.stop())
    finally:
        import os; os.unlink(path)


def test_file_source_loop_mode():
    """loop=True 模式文件循环播放"""
    import tempfile
    from scipy.io import wavfile
    sr = 16000
    data = np.zeros(1600, dtype=np.float32)
    path = tempfile.mktemp(suffix=".wav")
    wavfile.write(path, sr, data)
    try:
        src = FileSource(path, sample_rate=sr, block_size=1600, loop=True)
        asyncio.run(src.start())
        for _ in range(5):
            c = asyncio.run(src.read_chunk())
            assert c is not None
        asyncio.run(src.stop())
    finally:
        import os; os.unlink(path)


def test_file_source_rms_reflects_data():
    """FileSource.current_rms() 反映文件数据能量"""
    import tempfile
    from scipy.io import wavfile
    sr = 16000
    data = np.ones(8000, dtype=np.float32) * 0.5
    path = tempfile.mktemp(suffix=".wav")
    wavfile.write(path, sr, data)
    try:
        src = FileSource(path, sample_rate=sr, block_size=1600)
        asyncio.run(src.start())
        c = asyncio.run(src.read_chunk())
        rms = src.current_rms()
        assert rms > 0.0
        asyncio.run(src.stop())
    finally:
        import os; os.unlink(path)


# ═══════════════════════════════════════════════════════════════════
# SimulatedSource 测试
# ═══════════════════════════════════════════════════════════════════

def test_simulated_source_feeds_chunks():
    """SimulatedSource 逐块注入音频"""
    sr = 16000
    audio = np.ones(8000, dtype=np.float32) * 0.3
    src = SimulatedSource(audio_data=audio, sample_rate=sr, block_size=1600)
    asyncio.run(src.start())
    chunks = []
    while True:
        c = asyncio.run(src.read_chunk())
        if c is None:
            break
        chunks.append(c)
    asyncio.run(src.stop())
    assert len(chunks) == 5  # 8000 / 1600


def test_simulated_source_rms():
    """SimulatedSource.current_rms() 反映当前写入位置附近能量"""
    sr = 16000
    audio = np.ones(16000, dtype=np.float32) * 0.5
    src = SimulatedSource(audio_data=audio, sample_rate=sr, block_size=1600)
    asyncio.run(src.start())
    c = asyncio.run(src.read_chunk())
    rms = src.current_rms()
    assert rms > 0.0
    asyncio.run(src.stop())


def test_simulated_source_real_time_flag():
    """SimulatedSource 实时标记为 True"""
    src = SimulatedSource(audio_data=np.zeros(1600, dtype=np.float32))
    assert src.is_real_time() is True


def test_simulated_source_append():
    """append() 追加音频数据"""
    sr = 16000
    src = SimulatedSource(audio_data=np.zeros(0, dtype=np.float32), sample_rate=sr, block_size=1600)
    src.append(np.ones(1600, dtype=np.float32) * 0.5)
    asyncio.run(src.start())
    c = asyncio.run(src.read_chunk())
    assert c is not None
    assert np.max(np.abs(c)) > 0.0


def test_simulated_source_stop_clears():
    """stop() 后清空缓冲区，重新 start 后无数据"""
    sr = 16000
    src = SimulatedSource(audio_data=np.ones(3200, dtype=np.float32), sample_rate=sr, block_size=1600)
    asyncio.run(src.start())
    c1 = asyncio.run(src.read_chunk())
    assert c1 is not None
    asyncio.run(src.stop())
    asyncio.run(src.start())
    c = asyncio.run(src.read_chunk())
    assert c is None  # stop 后缓冲区已清空
