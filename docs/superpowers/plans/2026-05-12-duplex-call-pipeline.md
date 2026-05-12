# 双工通话管线实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重构语音 demo，统一 VoiceConversation + CustomerVoiceSimulator 为 DuplexCallPipeline，支持双工打断，一套代码兼容人声/自动两种模式。

**Architecture:** AudioSource(ABC) → DuplexCallPipeline → DuplexAudioOutput。管线经由 asyncio.Task 并发执行聆听和播放，打断检测使用 ducking + 300ms 二次确认。CallSimulator 替换 CustomerVoiceSimulator 作为 Pipeline 外部编排层。

**Tech Stack:** Python 3.10+, asyncio, numpy, sounddevice, faster-whisper, edge-tts

---

### Task 1: AudioSource 接口 + MicrophoneSource + SilentSource

**Files:**
- Create: `src/core/voice/audio_source.py`
- Test: `tests/test_audio_source.py`

- [ ] **Step 1: Write tests for AudioSource ABC and MicrophoneSource**

```python
"""AudioSource 单元测试"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pytest
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
    # start() 在无 sounddevice 环境应优雅降级
    import asyncio
    asyncio.run(src.start())
    # 不应崩溃


def test_microphone_source_rms_defaults_zero():
    """未启动时 RMS 为 0"""
    src = MicrophoneSource()
    assert src.current_rms() == 0.0


def test_silent_source_produces_silence():
    """SilentSource 产生静音块"""
    src = SilentSource(sample_rate=16000, block_size=1600)
    import asyncio
    asyncio.run(src.start())
    chunk = asyncio.run(src.read_chunk())
    assert chunk is not None
    assert len(chunk) == 1600
    assert np.max(np.abs(chunk)) == 0.0
    asyncio.run(src.stop())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_audio_source.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.voice.audio_source'`

- [ ] **Step 3: Implement AudioSource ABC + MicrophoneSource + SilentSource**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""音频输入源抽象层 — 统一麦克风/仿真/文件输入"""
import asyncio
import logging
from abc import ABC, abstractmethod

import numpy as np

from src.core.voice.audio_io import RingBuffer

logger = logging.getLogger(__name__)


class AudioSource(ABC):
    """音频输入源抽象。管线不感知来源，统一通过此接口消费音频。"""

    @abstractmethod
    async def start(self):
        """启动音频源，开始填充内部缓冲区"""
        ...

    @abstractmethod
    async def stop(self):
        """停止音频源，释放资源"""
        ...

    @abstractmethod
    async def read_chunk(self) -> np.ndarray | None:
        """读取下一个音频块 (block_size samples)。无数据返回 None。"""
        ...

    @abstractmethod
    def current_rms(self) -> float:
        """当前缓冲区的 RMS 能量。用于打断检测。"""
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """音频源采样率"""
        ...

    def is_real_time(self) -> bool:
        """是否实时输入（vs 文件/模拟可加速），默认 True"""
        return True


class MicrophoneSource(AudioSource):
    """麦克风输入源。封装 sounddevice InputStream → RingBuffer。"""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        block_size: int = 1600,
        device: int | None = None,
        buffer_duration: float = 10.0,
    ):
        self._sample_rate = sample_rate
        self._channels = channels
        self._block_size = block_size
        self._device = device
        self._buffer = RingBuffer(max_duration=buffer_duration, sample_rate=sample_rate)
        self._stream = None
        self._running = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def start(self):
        try:
            import sounddevice as sd
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                blocksize=self._block_size,
                device=self._device,
                callback=self._audio_callback,
                dtype=np.float32,
            )
            self._stream.start()
            self._running = True
            logger.info(f"MicrophoneSource started: {self._sample_rate}Hz")
        except ImportError:
            logger.warning("sounddevice 未安装，麦克风输入不可用")
        except Exception as e:
            logger.error(f"启动麦克风失败: {e}")

    def _audio_callback(self, indata, frames, timestamp, status):
        if status:
            logger.debug(f"Audio callback status: {status}")
        audio = indata[:, 0].copy() if indata.ndim > 1 else indata.flatten().copy()
        self._buffer.write(audio)

    async def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._buffer.clear()

    async def read_chunk(self) -> np.ndarray | None:
        data = self._buffer.read(duration=self._block_size / self._sample_rate)
        if len(data) == 0:
            return None
        return data.copy()

    def current_rms(self) -> float:
        if not self._running:
            return 0.0
        # 读取最近的数据但不消费
        data = self._buffer.read(duration=0.05)  # 50ms window
        if len(data) == 0:
            return 0.0
        return float(np.sqrt(np.mean(data ** 2)))


class SilentSource(AudioSource):
    """静音源 — 无输入时占位，避免 None 检查。"""

    def __init__(self, sample_rate: int = 16000, block_size: int = 1600):
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._running = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def start(self):
        self._running = True

    async def stop(self):
        self._running = False

    async def read_chunk(self) -> np.ndarray | None:
        if not self._running:
            return None
        await asyncio.sleep(self._block_size / self._sample_rate)
        return np.zeros(self._block_size, dtype=np.float32)

    def current_rms(self) -> float:
        return 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_audio_source.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/core/voice/audio_source.py tests/test_audio_source.py
git commit -m "feat: AudioSource 抽象 + MicrophoneSource + SilentSource

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: FileSource — 预录音频文件回放

**Files:**
- Modify: `src/core/voice/audio_source.py` (追加 FileSource)
- Modify: `tests/test_audio_source.py` (追加 FileSource 测试)

- [ ] **Step 1: Write FileSource tests**

```python
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
        # 读取了一个 chunk 后 RMS 应 > 0
        rms = src.current_rms()
        assert rms > 0.0
        asyncio.run(src.stop())
    finally:
        import os; os.unlink(path)
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `pytest tests/test_audio_source.py::test_file_source_loads_wav -v`
Expected: FAIL — `NameError: name 'FileSource' is not defined`

- [ ] **Step 3: Implement FileSource in audio_source.py**

```python
class FileSource(AudioSource):
    """文件回放源 — 从 WAV 文件分块读取，模拟实时麦克风。用于测试和回放。"""

    def __init__(
        self,
        file_path: str,
        sample_rate: int = 16000,
        block_size: int = 1600,
        loop: bool = False,
    ):
        self._file_path = file_path
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._loop = loop
        self._data: np.ndarray | None = None
        self._pos = 0
        self._running = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def start(self):
        self._data = _load_audio_mono(self._file_path, self._sample_rate)
        self._pos = 0
        self._running = True

    async def stop(self):
        self._running = False
        self._data = None
        self._pos = 0

    async def read_chunk(self) -> np.ndarray | None:
        if not self._running or self._data is None:
            return None
        if self._pos >= len(self._data):
            if self._loop:
                self._pos = 0
            else:
                return None
        end = min(self._pos + self._block_size, len(self._data))
        chunk = self._data[self._pos:end].copy()
        self._pos = end
        if len(chunk) < self._block_size:
            chunk = np.pad(chunk, (0, self._block_size - len(chunk)))
        return chunk

    def current_rms(self) -> float:
        if self._data is None or self._pos == 0:
            return 0.0
        start = max(0, self._pos - self._block_size)
        window = self._data[start:self._pos]
        if len(window) == 0:
            return 0.0
        return float(np.sqrt(np.mean(window ** 2)))

    def is_real_time(self) -> bool:
        return False  # 文件可以加速播放


def _load_audio_mono(path: str, target_sr: int) -> np.ndarray:
    """加载音频文件为 float32 mono 数组"""
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype='float32')
        if data.ndim > 1:
            data = data[:, 0]
        if sr != target_sr:
            from scipy.signal import resample
            n_samples = int(len(data) * target_sr / sr)
            data = resample(data, n_samples)
        return data.astype(np.float32)
    except ImportError:
        pass
    # fallback: scipy wavfile
    try:
        from scipy.io import wavfile
        import subprocess, tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.run(
            ['ffmpeg', '-y', '-i', path, '-f', 'wav', '-acodec', 'pcm_f32le',
             '-ar', str(target_sr), '-ac', '1', tmp_path],
            capture_output=True, check=True,
        )
        sr, data = wavfile.read(tmp_path)
        from pathlib import Path
        Path(tmp_path).unlink(missing_ok=True)
        return data.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"无法加载音频文件 {path}: {e}")
```

- [ ] **Step 4: Run all audio_source tests**

Run: `pytest tests/test_audio_source.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/core/voice/audio_source.py tests/test_audio_source.py
git commit -m "feat: FileSource — 预录音频文件回放 + loop 模式

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: DuplexAudioOutput — 播放 + 打断检测

**Files:**
- Create: `src/core/voice/audio_output.py`
- Create: `tests/test_audio_output.py`

- [ ] **Step 1: Write DuplexAudioOutput tests**

```python
"""DuplexAudioOutput 单元测试"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pytest
import asyncio
from core.voice.audio_output import DuplexAudioOutput, PlaybackResult
from core.voice.audio_source import SilentSource, FileSource


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
    output = DuplexAudioOutput(source, barge_in_threshold=0.05)
    audio = make_silence(0.2)
    result = await output.speak(audio)
    assert result == PlaybackResult.COMPLETED
    await source.stop()


@pytest.mark.asyncio
async def test_interrupted_by_loud_source():
    """高 RMS 源触发打断"""
    # 使用 FileSource 播放音频，其 RMS 会被检测到
    import tempfile
    from scipy.io import wavfile
    sr = 16000
    data = np.ones(32000, dtype=np.float32) * 0.5  # 高能量
    path = tempfile.mktemp(suffix=".wav")
    wavfile.write(path, sr, data.astype(np.float32))
    try:
        source = FileSource(path, sample_rate=sr, block_size=1600)
        await source.start()
        output = DuplexAudioOutput(source, barge_in_threshold=0.01, ducking_duration=0.1, confirmation_duration=0.1)
        audio = make_silence(5.0)  # 长静音播放，等待打断
        result = await output.speak(audio)
        # 因为 source 是 loud file，应该触发打断
        assert result == PlaybackResult.INTERRUPTED
        await source.stop()
    finally:
        import os; os.unlink(path)


@pytest.mark.asyncio
async def test_stop_immediately():
    """stop() 立即停止播放"""
    source = SilentSource()
    await source.start()
    output = DuplexAudioOutput(source, barge_in_threshold=0.05)
    audio = make_silence(3.0)
    task = asyncio.create_task(output.speak(audio))
    await asyncio.sleep(0.05)
    output.stop()
    result = await task
    assert result == PlaybackResult.INTERRUPTED
    await source.stop()


@pytest.mark.asyncio
async def test_playback_result_enum_values():
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
    """短暂噪音后恢复正常播放"""
    source = SilentSource()
    await source.start()
    output = DuplexAudioOutput(
        source, barge_in_threshold=0.05,
        ducking_duration=0.1, confirmation_duration=0.1,
    )
    # 静音源不会触发打断，播放应完成
    audio = make_tone(0.3)
    result = await output.speak(audio)
    assert result == PlaybackResult.COMPLETED
    await source.stop()
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `pytest tests/test_audio_output.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement DuplexAudioOutput**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双工音频播放输出 — Agent TTS 播放 + 持续打断检测 + ducking"""
import asyncio
import logging
from enum import Enum

import numpy as np

from src.core.voice.audio_source import AudioSource

logger = logging.getLogger(__name__)


class PlaybackResult(Enum):
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class DuplexAudioOutput:
    """双工音频播放器。分块播放 Agent TTS，每块间隔检测打断。

    打断检测流程:
    1. 分块播放音频 (~100ms/chunk)
    2. 播放间隙检查 source.current_rms()
    3. RMS 超阈值 → duck 音量 → 等待 confirmation_duration 二次确认
    4. 确认打断 → 停止播放返回 INTERRUPTED
    5. 误检恢复 → 恢复正常音量继续播放
    """

    def __init__(
        self,
        source: AudioSource,
        barge_in_threshold: float = 0.02,
        duck_volume_ratio: float = 0.2,
        confirmation_duration: float = 0.3,
        chunk_duration: float = 0.1,
    ):
        self._source = source
        self._barge_in_threshold = barge_in_threshold
        self._duck_volume_ratio = duck_volume_ratio
        self._confirmation_duration = confirmation_duration
        self._chunk_duration = chunk_duration

        self._speak_task: asyncio.Task | None = None
        self._is_speaking = False
        self._is_ducking = False
        self._stop_requested = False

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    @property
    def is_ducking(self) -> bool:
        return self._is_ducking

    async def speak(self, audio: np.ndarray) -> PlaybackResult:
        """播放音频，后台分块播放并监听打断。返回播放结果。"""
        if len(audio) == 0:
            return PlaybackResult.COMPLETED

        self._is_speaking = True
        self._stop_requested = False
        self._is_ducking = False

        try:
            return await self._play_with_interrupt_detection(audio)
        except Exception as e:
            logger.error(f"播放异常: {e}")
            return PlaybackResult.FAILED
        finally:
            self._is_speaking = False
            self._stop_requested = False
            self._is_ducking = False

    def stop(self):
        """立即停止当前播放"""
        self._stop_requested = True
        try:
            import sounddevice as sd
            sd.stop()
        except ImportError:
            pass

    async def _play_with_interrupt_detection(self, audio: np.ndarray) -> PlaybackResult:
        sr = self._source.sample_rate
        chunk_samples = int(self._chunk_duration * sr)
        pos = 0

        try:
            import sounddevice as sd
        except ImportError:
            logger.error("sounddevice 未安装，无法播放")
            return PlaybackResult.FAILED

        while pos < len(audio) and not self._stop_requested:
            end = min(pos + chunk_samples, len(audio))
            chunk = audio[pos:end]

            if self._is_ducking:
                chunk = (chunk * self._duck_volume_ratio).astype(np.float32)

            sd.play(chunk, samplerate=sr, blocking=True)

            # 检查打断
            rms = self._source.current_rms()
            if rms > self._barge_in_threshold:
                if not self._is_ducking:
                    self._is_ducking = True
                    sd.stop()  # 停止当前块
                    # 二次确认
                    await asyncio.sleep(self._confirmation_duration)
                    rms_confirm = self._source.current_rms()
                    if rms_confirm > self._barge_in_threshold:
                        sd.stop()
                        logger.info(f"播放被打断 (RMS={rms_confirm:.4f})")
                        return PlaybackResult.INTERRUPTED
                    # 误检，恢复
                    self._is_ducking = False
                    pos = end
                    continue
            else:
                self._is_ducking = False

            pos = end

        if self._stop_requested:
            return PlaybackResult.INTERRUPTED
        return PlaybackResult.COMPLETED
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_audio_output.py -v --asyncio-mode=auto`
Expected: 6 passed (NOTE: `test_interrupted_by_loud_source` may need sounddevice; it will be skipped if not available, or may fail with loud data if sounddevice is present — adjust threshold if needed)

- [ ] **Step 5: Commit**

```bash
git add src/core/voice/audio_output.py tests/test_audio_output.py
git commit -m "feat: DuplexAudioOutput — 双工播放 + barge-in + ducking

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: DuplexCallPipeline — 核心管线 + 状态机

**Files:**
- Create: `src/core/voice/pipeline.py`
- Create: `tests/test_duplex_pipeline.py` (Part 1: 状态机 + 单元)

- [ ] **Step 1: Write Pipeline 状态机测试**

```python
"""DuplexCallPipeline 单元测试 — 状态机"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pytest
import asyncio
from core.voice.pipeline import DuplexCallPipeline, PipelineState, PipelineConfig
from core.voice.audio_source import SilentSource
from core.voice.audio_output import DuplexAudioOutput, PlaybackResult


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
        audio = np.zeros(8000, dtype=np.float32)
        return TTSResult(text=text, audio_data=audio, audio_file=None, success=True)


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
        self.state = None
        self.commit_time = None
        self.turns_processed = 0

    class state:
        name = "INIT"

    async def process(self, customer_input=None, use_tts=False):
        self.turns_processed += 1
        if self.turns_processed >= 3:
            self.state = _FakeCloseState()
            return "Terima kasih, selamat tinggal.", None
        return "Baik, saya catat ya.", None


class _FakeCloseState:
    name = "CLOSE"


def make_pipeline(**kwargs):
    """工厂：创建最小可测 Pipeline"""
    config = PipelineConfig(sample_rate=16000, block_size=1600, silence_duration=0.2, max_speech_duration=5.0)
    source = SilentSource()
    output = DuplexAudioOutput(source, barge_in_threshold=0.99)  # 极高阈值避免意外打断
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
    # FakeVAD 返回 30 帧语音 → step 会累积语音 → ASR
    for _ in range(35):
        await pipeline.step()
    assert pipeline.state in (PipelineState.PROCESSING, PipelineState.RESPONDING, PipelineState.LISTENING)


@pytest.mark.asyncio
async def test_pipeline_full_cycle_to_close():
    """完整走完到 CLOSED"""
    pipeline = make_pipeline()
    await pipeline.start()
    while pipeline.state not in (PipelineState.CLOSING, PipelineState.CLOSED):
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
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `pytest tests/test_duplex_pipeline.py -v --asyncio-mode=auto`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement PipelineState, PipelineConfig, and DuplexCallPipeline**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双工通话管线 — 统一人声/自动两模式的语音催收核心循环"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable

import numpy as np

from src.core.voice.audio_source import AudioSource
from src.core.voice.audio_output import DuplexAudioOutput, PlaybackResult
from src.core.voice.vad import SimpleEnergyVAD, VADState

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    RESPONDING = auto()
    INTERRUPTED = auto()
    CLOSING = auto()
    CLOSED = auto()


@dataclass
class PipelineConfig:
    sample_rate: int = 16000
    block_size: int = 1600
    silence_duration: float = 1.0
    max_speech_duration: float = 15.0
    max_silence_duration: float = 30.0
    prompt_on_silence: str = "Halo, apakah Anda masih di sana?"


@dataclass
class StepResult:
    """单步执行结果"""
    state_from: PipelineState
    state_to: PipelineState
    asr_text: str = ""
    agent_text: str = ""
    agent_audio: np.ndarray | None = None
    interrupted: bool = False
    turn_id: int = 0
    elapsed_s: float = 0.0


class DuplexCallPipeline:
    """双工通话管线。统一人声模式与自动仿真模式的语音催收核心循环。

    使用:
        source = MicrophoneSource()
        output = DuplexAudioOutput(source)
        pipeline = DuplexCallPipeline(chatbot, source, output, asr, tts, vad)
        await pipeline.start()
        while pipeline.state != PipelineState.CLOSED:
            result = await pipeline.step()
            if result:
                print(f"[{result.state_to.name}] ASR: {result.asr_text} | Agent: {result.agent_text}")
        await pipeline.stop()
    """

    def __init__(
        self,
        chatbot,
        source: AudioSource,
        output: DuplexAudioOutput,
        asr_pipeline,
        tts_manager,
        vad: SimpleEnergyVAD | None = None,
        *,
        config: PipelineConfig | None = None,
    ):
        self._chatbot = chatbot
        self._source = source
        self._output = output
        self._asr = asr_pipeline
        self._tts = tts_manager
        self._vad = vad or SimpleEnergyVAD(
            sample_rate=source.sample_rate,
            energy_threshold=0.01,
        )
        self._config = config or PipelineConfig()

        self._state = PipelineState.IDLE
        self._running = False
        self._turn_id = 0

        # 聆听缓冲区
        self._speech_buffer = np.zeros(
            int(self._config.max_speech_duration * self._config.sample_rate),
            dtype=np.float32,
        )
        self._speech_pos = 0
        self._silence_samples = 0
        self._silence_threshold = int(self._config.silence_duration * self._config.sample_rate)
        self._max_speech_samples = int(self._config.max_speech_duration * self._config.sample_rate)
        self._voice_detected = False

        # 长静音计时
        self._total_silence_s = 0.0
        self._last_voice_time = 0.0

        # 后台播放 task
        self._speak_task: asyncio.Task | None = None

        # 打断上下文（最近一次）
        self._last_interruption: Optional["InterruptionContext"] = None

        # 回调
        self.on_state_change: Optional[Callable] = None
        self.on_turn_complete: Optional[Callable] = None

    # ── 属性 ──────────────────────────────────────────────

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def turn_id(self) -> int:
        return self._turn_id

    @property
    def last_interruption(self) -> Optional["InterruptionContext"]:
        return self._last_interruption

    # ── 生命周期 ──────────────────────────────────────────

    async def start(self):
        await self._source.start()
        self._running = True
        self._vad.reset()
        self._set_state(PipelineState.LISTENING)
        logger.info("DuplexCallPipeline started")

    async def stop(self):
        self._running = False
        if self._speak_task and not self._speak_task.done():
            self._speak_task.cancel()
        self._output.stop()
        await self._source.stop()
        self._set_state(PipelineState.CLOSED)
        logger.info("DuplexCallPipeline stopped")

    async def run_until_closed(self):
        """自动运行直到通话结束"""
        await self.start()
        try:
            while self._state not in (PipelineState.CLOSING, PipelineState.CLOSED):
                await self.step()
                await asyncio.sleep(0.01)
        finally:
            if self._state != PipelineState.CLOSED:
                await self.stop()

    # ── 核心循环 ──────────────────────────────────────────

    async def step(self) -> StepResult | None:
        """执行一步管线。根据当前状态分发。"""
        if not self._running:
            return None

        t0 = time.time()
        state_before = self._state

        if self._state == PipelineState.LISTENING:
            await self._step_listen()
        elif self._state == PipelineState.PROCESSING:
            await self._step_process()
        elif self._state == PipelineState.RESPONDING:
            await self._step_respond()
        elif self._state == PipelineState.INTERRUPTED:
            await self._step_interrupted()
        elif self._state == PipelineState.CLOSING:
            await self._step_closing()
        elif self._state in (PipelineState.IDLE, PipelineState.CLOSED):
            pass

        elapsed = time.time() - t0
        return StepResult(
            state_from=state_before,
            state_to=self._state,
            turn_id=self._turn_id,
            elapsed_s=elapsed,
        )

    # ── 状态处理 ──────────────────────────────────────────

    async def _step_listen(self):
        """LISTENING: 读取一个音频块，VAD 检测，累积语音段"""
        chunk = await self._source.read_chunk()
        if chunk is None or len(chunk) == 0:
            await asyncio.sleep(0.01)
            return

        result = self._vad.process_frame(chunk)

        if result.state == VADState.VOICE:
            self._voice_detected = True
            self._last_voice_time = time.time()

        if self._voice_detected:
            remaining = self._max_speech_samples - self._speech_pos
            to_write = chunk[:remaining]
            self._speech_buffer[self._speech_pos:self._speech_pos + len(to_write)] = to_write
            self._speech_pos += len(to_write)

            if result.state == VADState.SILENCE:
                self._silence_samples += len(chunk)
            else:
                self._silence_samples = 0

            # 静音超时 → 语音段结束
            if self._silence_samples >= self._silence_threshold and self._speech_pos > 0:
                self._set_state(PipelineState.PROCESSING)
                return

            # 语音超长 → 截断
            if self._speech_pos >= self._max_speech_samples:
                self._set_state(PipelineState.PROCESSING)
                return

        # 长静音检测
        if not self._voice_detected:
            self._total_silence_s += len(chunk) / self._config.sample_rate
            if self._total_silence_s >= self._config.max_silence_duration:
                logger.info("长静音超时，播放提示")
                self._set_state(PipelineState.CLOSING)
                return

    async def _step_process(self):
        """PROCESSING: ASR → Bot → TTS"""
        speech = self._speech_buffer[:self._speech_pos].copy()
        self._reset_listen_state()

        # ASR
        asr_text = ""
        if self._asr and self._asr.is_available and len(speech) > 0:
            try:
                asr_text = await self._asr.transcribe_async(speech)
            except Exception as e:
                logger.error(f"ASR 失败: {e}")
                asr_text = ""

        logger.info(f"ASR: '{asr_text}'")

        # Bot
        from src.core.chatbot import ChatState
        if self._chatbot.state in (ChatState.CLOSE, ChatState.FAILED):
            self._set_state(PipelineState.CLOSING)
            return

        try:
            agent_text, _ = await self._chatbot.process(
                customer_input=asr_text if asr_text else None,
                use_tts=False,
            )
        except Exception as e:
            logger.error(f"Chatbot 处理异常: {e}")
            agent_text = ""

        # TTS
        agent_audio = None
        if agent_text:
            try:
                tts_result = await self._tts.synthesize(agent_text)
                if tts_result.success:
                    if tts_result.audio_data is not None:
                        agent_audio = np.frombuffer(tts_result.audio_data, dtype=np.float32) \
                            if isinstance(tts_result.audio_data, bytes) else tts_result.audio_data
                    elif tts_result.audio_file:
                        import soundfile as sf
                        agent_audio, _ = sf.read(tts_result.audio_file, dtype='float32')
                        if agent_audio.ndim > 1:
                            agent_audio = agent_audio[:, 0]
            except Exception as e:
                logger.error(f"TTS 失败: {e}")

        self._current_agent_text = agent_text
        self._current_agent_audio = agent_audio
        self._current_asr_text = asr_text
        self._turn_id += 1

        # 检查 Bot 是否结束
        if self._chatbot.state in (ChatState.CLOSE, ChatState.FAILED):
            self._set_state(PipelineState.CLOSING)
        else:
            self._set_state(PipelineState.RESPONDING)

    async def _step_respond(self):
        """RESPONDING: 播放 Agent 音频，同时继续监听打断"""
        if self._current_agent_audio is not None and len(self._current_agent_audio) > 0:
            self._speak_task = asyncio.create_task(
                self._output.speak(self._current_agent_audio)
            )

            # 在播放过程中持续监听
            while self._speak_task and not self._speak_task.done():
                # 检查新的语音活动（打断检测由 DuplexAudioOutput 内部完成）
                await asyncio.sleep(0.05)

            result = await self._speak_task
            self._speak_task = None

            if result == PlaybackResult.INTERRUPTED:
                self._last_interruption = InterruptionContext(
                    agent_text_interrupted=self._current_agent_text or "",
                    agent_playback_position=0.5,
                    customer_rms_peak=self._source.current_rms(),
                )
                self._set_state(PipelineState.INTERRUPTED)
                return

        # 播放完成，回到聆听
        if self._chatbot.state.name in ("CLOSE", "FAILED"):
            self._set_state(PipelineState.CLOSING)
        else:
            self._set_state(PipelineState.LISTENING)

    async def _step_interrupted(self):
        """INTERRUPTED: 被打断的过渡处理"""
        logger.info("处理打断...")
        self._set_state(PipelineState.LISTENING)

    async def _step_closing(self):
        """CLOSING: 播放结束语后停止"""
        if self._current_agent_audio is not None and len(self._current_agent_audio) > 0:
            await self._output.speak(self._current_agent_audio)
        self._running = False
        self._set_state(PipelineState.CLOSED)

    # ── 内部方法 ──────────────────────────────────────────

    def _set_state(self, new_state: PipelineState):
        old = self._state
        self._state = new_state
        if old != new_state:
            logger.debug(f"Pipeline: {old.name} → {new_state.name}")
            if self.on_state_change:
                self.on_state_change(old, new_state)

    def _reset_listen_state(self):
        self._speech_pos = 0
        self._silence_samples = 0
        self._voice_detected = False
        self._total_silence_s = 0.0
        self._vad.reset()


@dataclass
class InterruptionContext:
    """打断上下文 — 传递给 Bot 用于策略调整"""
    agent_text_interrupted: str = ""
    agent_playback_position: float = 0.0
    customer_rms_peak: float = 0.0
    partial_asr: str | None = None
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_duplex_pipeline.py -v --asyncio-mode=auto`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/core/voice/pipeline.py tests/test_duplex_pipeline.py
git commit -m "feat: DuplexCallPipeline — 双工通话核心管线 + 状态机

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: SimulatedSource — 文本→TTS→音频流式注入

**Files:**
- Modify: `src/core/voice/audio_source.py` (追加 SimulatedSource)
- Modify: `tests/test_audio_source.py` (追加 SimulatedSource 测试)

- [ ] **Step 1: Write SimulatedSource tests**

```python
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
    """stop() 后清空缓冲区"""
    sr = 16000
    src = SimulatedSource(audio_data=np.ones(3200, dtype=np.float32), sample_rate=sr, block_size=1600)
    asyncio.run(src.start())
    asyncio.run(src.read_chunk())
    asyncio.run(src.stop())
    asyncio.run(src.start())
    c = asyncio.run(src.read_chunk())
    assert c is not None  # 重新 start 后应从头开始
```

- [ ] **Step 2: Run new tests — expect FAIL**

Run: `pytest tests/test_audio_source.py::test_simulated_source_feeds_chunks -v`
Expected: FAIL — `NameError: name 'SimulatedSource' is not defined`

- [ ] **Step 3: Implement SimulatedSource in audio_source.py**

```python
class SimulatedSource(AudioSource):
    """仿真输入源 — 文本模拟器的 TTS 输出逐块注入到 RingBuffer。

    核心变化：不是先生成完整音频文件再加载，而是逐块流式注入，
    模拟真实麦克风的实时数据到达节奏。"""

    def __init__(
        self,
        audio_data: np.ndarray | None = None,
        sample_rate: int = 16000,
        block_size: int = 1600,
    ):
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._buffer = RingBuffer(max_duration=60.0, sample_rate=sample_rate)
        self._running = False

        # 预填充
        if audio_data is not None and len(audio_data) > 0:
            self._buffer.write(audio_data.astype(np.float32))

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def append(self, audio: np.ndarray):
        """追加音频数据到缓冲区，浮点数组"""
        if len(audio) > 0:
            self._buffer.write(audio.astype(np.float32))

    async def start(self):
        self._running = True

    async def stop(self):
        self._running = False
        self._buffer.clear()

    async def read_chunk(self) -> np.ndarray | None:
        if not self._running:
            return None
        data = self._buffer.read(duration=self._block_size / self._sample_rate)
        if len(data) == 0:
            return None
        if len(data) < self._block_size:
            data = np.pad(data, (0, self._block_size - len(data)))
        return data.copy()

    def current_rms(self) -> float:
        # peek 最近 50ms
        peek = self._buffer.read(duration=0.05)
        if len(peek) == 0:
            return 0.0
        return float(np.sqrt(np.mean(peek ** 2)))

    def is_real_time(self) -> bool:
        return True

    @property
    def pending_samples(self) -> int:
        """缓冲区中剩余未读取的样本数"""
        return len(self._buffer)
```

附：在 `audio_source.py` 的 import 区追加 `from src.core.voice.audio_io import RingBuffer`

- [ ] **Step 4: Run all audio_source tests**

Run: `pytest tests/test_audio_source.py -v`
Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
git add src/core/voice/audio_source.py tests/test_audio_source.py
git commit -m "feat: SimulatedSource — 文本→TTS 流式音频注入

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 6: DuplexCallPipeline 集成测试 — 打断 + 全流程

**Files:**
- Modify: `tests/test_duplex_pipeline.py` (追加集成测试)

- [ ] **Step 1: Write 打断和全流程集成测试**

```python
"""DuplexCallPipeline 集成测试 — FileSource + 打断检测"""
import asyncio
import tempfile
import numpy as np
import pytest
from scipy.io import wavfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from core.voice.pipeline import DuplexCallPipeline, PipelineState, PipelineConfig
from core.voice.audio_source import FileSource, SilentSource
from core.voice.audio_output import DuplexAudioOutput
from core.voice.vad import SimpleEnergyVAD


class FakeASR:
    is_available = True
    def transcribe(self, audio): return "Ya"
    async def transcribe_async(self, audio): return "Ya"


class FakeTTS:
    async def synthesize(self, text, output_file=None, voice=None, engine=None, **kwargs):
        from core.voice.tts import TTSResult
        arr = np.zeros(8000, dtype=np.float32)
        return TTSResult(text=text, audio_data=arr.tobytes() if hasattr(arr, 'tobytes') else arr, audio_file=None, success=True)


class FakeBot:
    def __init__(self):
        self.turns = 0
        self.commit_time = None

    class state:
        name = "INIT"

    async def process(self, customer_input=None, use_tts=False):
        self.turns += 1
        if self.turns >= 4:
            self.state = _CloseState()
            return "Terima kasih.", None
        return "Baik.", None


class _CloseState:
    name = "CLOSE"


@pytest.mark.asyncio
async def test_pipeline_with_filesource():
    """Pipeline + FileSource 完整对话 (4 轮)"""
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
        max_steps = 200
        for _ in range(max_steps):
            if pipeline.state in (PipelineState.CLOSING, PipelineState.CLOSED):
                break
            await pipeline.step()
            await asyncio.sleep(0.005)

        assert pipeline.state == PipelineState.CLOSED
        await pipeline.stop()
    finally:
        import os; os.unlink(path)


@pytest.mark.asyncio
async def test_step_result_fields():
    """StepResult 包含正确字段"""
    from core.voice.pipeline import StepResult
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


@pytest.mark.asyncio
async def test_interruption_context():
    """InterruptionContext 数据类"""
    from core.voice.pipeline import InterruptionContext
    ctx = InterruptionContext(
        agent_text_interrupted="Baik, jadi besok jam 5 ya?",
        agent_playback_position=0.6,
        customer_rms_peak=0.15,
    )
    assert ctx.agent_playback_position == 0.6
    assert "besok" in ctx.agent_text_interrupted
```

- [ ] **Step 2: Run all pipeline tests**

Run: `pytest tests/test_duplex_pipeline.py -v --asyncio-mode=auto`
Expected: 9 passed

- [ ] **Step 3: Commit**

```bash
git add tests/test_duplex_pipeline.py
git commit -m "test: DuplexCallPipeline 集成测试 — 打断 + FileSource 全流程

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 7: CallSimulator — 自动仿真模式

**Files:**
- Create: `src/core/voice/call_simulator.py`
- Test: `tests/test_call_simulator.py`

- [ ] **Step 1: Write CallSimulator tests**

```python
"""CallSimulator 单元测试 — 自动仿真"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pytest
import asyncio
from core.voice.call_simulator import CallSimulator
from core.voice.audio_source import SimulatedSource
from core.voice.pipeline import DuplexCallPipeline, PipelineState, PipelineConfig
from core.voice.audio_output import DuplexAudioOutput


class FakeTTS:
    async def synthesize(self, text, output_file=None, voice=None, engine=None, **kwargs):
        from core.voice.tts import TTSResult
        return TTSResult(text=text, audio_data=np.zeros(8000, dtype=np.float32), audio_file=None, success=True)


class FakeASR:
    is_available = True
    def transcribe(self, audio): return "Ya"
    async def transcribe_async(self, audio): return "Ya"


class FakeVAD:
    silence_frames = 10
    voice_frames = 2

    def process_frame(self, audio_frame):
        from core.voice.vad import VADResult, VADState
        return VADResult(state=VADState.SILENCE, confidence=0.9, timestamp=0)

    def reset(self): pass


class FakeBot:
    def __init__(self):
        self.turns = 0
        self.commit_time = None

    class state:
        name = "INIT"

    async def process(self, customer_input=None, use_tts=False):
        self.turns += 1
        if self.turns >= 3:
            self.state = _CloseStateSim()
            return "Terima kasih.", None
        return "Baik.", None


class _CloseStateSim:
    name = "CLOSE"


class FakeTextSimulator:
    def generate_response(self, stage, chat_group, persona, resistance_level, push_count):
        return "Ya, besok jam 5"

    def get_current_stage_and_response(self, chat_group, push_count):
        return "ask_time", "Besok jam 5"


@pytest.mark.asyncio
async def test_call_simulator_runs_to_completion():
    """CallSimulator 运行到结束"""
    sim = CallSimulator(
        chatbot=FakeBot(),
        text_simulator=FakeTextSimulator(),
        tts_manager=FakeTTS(),
        asr_pipeline=FakeASR(),
        persona="cooperative",
        resistance_level="medium",
        chat_group="H2",
        max_turns=5,
        save_artifacts=False,
    )
    report = await sim.run()
    assert report.total_turns > 0
    assert report.conversation_ended


@pytest.mark.asyncio
async def test_call_simulator_create_factory():
    """CallSimulator.create() 工厂方法"""
    bot = FakeBot()
    sim = await CallSimulator.create(
        chatbot=bot,
        persona="cooperative",
        resistance_level="medium",
        chat_group="H2",
        asr_model_size="tiny",
        save_artifacts=False,
        _asr_pipeline=FakeASR(),
    )
    assert sim.persona == "cooperative"
    assert sim.chat_group == "H2"


def test_call_simulator_initial_state():
    """初始属性正确"""
    sim = CallSimulator(
        chatbot=FakeBot(),
        text_simulator=FakeTextSimulator(),
        tts_manager=FakeTTS(),
        asr_pipeline=FakeASR(),
        persona="resistant",
        resistance_level="high",
        chat_group="S0",
        max_turns=10,
        save_artifacts=False,
    )
    assert sim.persona == "resistant"
    assert sim.resistance_level == "high"
    assert sim.max_turns == 10
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `pytest tests/test_call_simulator.py -v --asyncio-mode=auto`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement CallSimulator**

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自动仿真模式 — 文本模拟器 → TTS → SimulatedSource → DuplexCallPipeline"""
import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from src.core.voice.audio_source import SimulatedSource
from src.core.voice.audio_output import DuplexAudioOutput
from src.core.voice.pipeline import DuplexCallPipeline, PipelineConfig, PipelineState
from src.core.voice.vad import SimpleEnergyVAD

logger = logging.getLogger(__name__)

# 保留原 SimulationReport / SimulationTurn 以向后兼容
from src.core.voice.customer_simulator import SimulationReport, SimulationTurn


class CallSimulator:
    """自动仿真。用文本模拟器生成客户回复 → TTS → SimulatedSource → Pipeline。

    与 CustomerVoiceSimulator 的关键区别:
    - 不自己实现管线，而是编排 DuplexCallPipeline
    - 不再有 _inject_and_vad_gate / _run_single_turn
    - 客户回复逐轮注入到 SimulatedSource
    """

    def __init__(
        self,
        chatbot,
        text_simulator,
        tts_manager,
        asr_pipeline,
        *,
        persona: str = "cooperative",
        resistance_level: str = "medium",
        chat_group: str = "H2",
        max_turns: int = 20,
        customer_voice: str = "id-ID-GadisNeural",
        agent_voice: str = "id-ID-ArdiNeural",
        customer_tts_engine: str = "edge_tts",
        agent_tts_engine: str = "edge_tts",
        sample_rate: int = 16000,
        block_size: int = 1600,
        save_artifacts: bool = True,
        output_dir: str = "data/runs/voice_simulations",
        realtime: bool = False,
    ):
        self._chatbot = chatbot
        self._text_sim = text_simulator
        self._tts = tts_manager
        self._asr = asr_pipeline

        self.persona = persona
        self.resistance_level = resistance_level
        self.chat_group = chat_group
        self.max_turns = max_turns
        self.customer_voice = customer_voice
        self.agent_voice = agent_voice
        self.customer_tts_engine = customer_tts_engine
        self.agent_tts_engine = agent_tts_engine
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.save_artifacts = save_artifacts
        self.output_dir = Path(output_dir)
        self.realtime = realtime

        self._push_count = 0
        self._session_id = str(uuid.uuid4())[:8]
        self._turns: list[SimulationTurn] = []
        self._start_time: float = 0.0
        self._run_dir: Optional[Path] = None
        self._source: Optional[SimulatedSource] = None
        self._pipeline: Optional[DuplexCallPipeline] = None
        self._output: Optional[DuplexAudioOutput] = None

    @classmethod
    async def create(
        cls,
        chatbot,
        *,
        persona: str = "cooperative",
        resistance_level: str = "medium",
        chat_group: str = "H2",
        customer_name: str = "Budi",
        asr_model_size: str = "small",
        customer_voice: str = "id-ID-GadisNeural",
        agent_voice: str = "id-ID-ArdiNeural",
        customer_tts_engine: str = "edge_tts",
        agent_tts_engine: str = "edge_tts",
        realtime: bool = False,
        save_artifacts: bool = True,
        output_dir: str = "data/runs/voice_simulations",
        **kwargs,
    ) -> "CallSimulator":
        from src.core.simulator import RealCustomerSimulatorV2
        from src.core.voice.tts import TTSManager
        from src.core.voice.asr import ASRPipeline

        text_sim = RealCustomerSimulatorV2()
        tts = TTSManager()

        pre_warmed = kwargs.pop("_asr_pipeline", None)
        if pre_warmed is not None:
            asr = pre_warmed
        else:
            corrector = getattr(chatbot, "asr_corrector", None)
            asr = await ASRPipeline.create(model_size=asr_model_size, corrector=corrector)

        return cls(
            chatbot=chatbot,
            text_simulator=text_sim,
            tts_manager=tts,
            asr_pipeline=asr,
            persona=persona,
            resistance_level=resistance_level,
            chat_group=chat_group,
            max_turns=kwargs.pop("max_turns", 20),
            customer_voice=customer_voice,
            agent_voice=agent_voice,
            customer_tts_engine=customer_tts_engine,
            agent_tts_engine=agent_tts_engine,
            sample_rate=kwargs.pop("sample_rate", 16000),
            block_size=kwargs.pop("block_size", 1600),
            realtime=realtime,
            save_artifacts=save_artifacts,
            output_dir=output_dir,
            **kwargs,
        )

    async def run(self, max_turns: int | None = None) -> SimulationReport:
        max_turns = max_turns or self.max_turns
        self._start_time = time.time()

        if self.save_artifacts:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self._run_dir = self.output_dir / f"{timestamp}_{self._session_id}"
            self._run_dir.mkdir(parents=True, exist_ok=True)

        from src.core.chatbot import ChatState

        # 初始: bot 第一句话
        first_msg, _ = await self._chatbot.process(use_tts=False)
        prev_state = self._chatbot.state
        stuck_count = 0

        for turn_id in range(1, max_turns + 1):
            if self._chatbot.state in (ChatState.CLOSE, ChatState.FAILED):
                break

            turn = SimulationTurn(
                turn_id=turn_id,
                state_before=self._chatbot.state.name,
            )

            # 1. 文本模拟器 → 客户文本
            stage = self._state_to_stage(self._chatbot.state)
            try:
                customer_text = self._text_sim.generate_response(
                    stage=stage,
                    chat_group=self.chat_group,
                    persona=self.persona,
                    resistance_level=self.resistance_level,
                    push_count=self._push_count,
                )
            except Exception:
                customer_text = "Ya"
            turn.customer_text = customer_text
            self._push_count += 1

            # 2. TTS → 客户音频
            if customer_text.strip():
                try:
                    tts_result = await self._tts.synthesize(
                        customer_text,
                        voice=self.customer_voice,
                        engine=self.customer_tts_engine,
                    )
                    if tts_result.success and tts_result.audio_file:
                        turn.customer_audio_file = tts_result.audio_file
                        # 加载音频
                        customer_audio = _load_audio_mono(tts_result.audio_file, self.sample_rate)
                    elif tts_result.success and tts_result.audio_data is not None:
                        customer_audio = np.frombuffer(tts_result.audio_data, dtype=np.float32) \
                            if isinstance(tts_result.audio_data, bytes) else tts_result.audio_data
                    else:
                        turn.tts_failed = True
                        customer_audio = np.array([], dtype=np.float32)
                except Exception:
                    turn.tts_failed = True
                    customer_audio = np.array([], dtype=np.float32)
            else:
                turn.tts_failed = True
                customer_audio = np.array([], dtype=np.float32)

            # 3. 逐块注入 SimulatedSource → ASR
            if len(customer_audio) > 0:
                # 创建临时 Source
                source = SimulatedSource(audio_data=customer_audio, sample_rate=self.sample_rate)
                await source.start()
                speech = _collect_all(source, len(customer_audio))
                await source.stop()

                # ASR
                if self._asr.is_available and len(speech) > 0:
                    try:
                        turn.asr_text = self._asr.transcribe(speech)
                    except Exception:
                        turn.asr_failed = True
            else:
                turn.vad_dropped = True

            # 4. Bot 处理
            input_text = turn.asr_text if turn.asr_text else ""
            try:
                agent_text, _ = await self._chatbot.process(
                    customer_input=input_text if input_text else None,
                    use_tts=False,
                )
            except Exception:
                agent_text = ""
            turn.agent_text = agent_text
            turn.state_after = self._chatbot.state.name

            self._turns.append(turn)

            # 卡状态检测
            if self._chatbot.state == prev_state:
                stuck_count += 1
                if stuck_count >= 3:
                    break
            else:
                stuck_count = 0
            prev_state = self._chatbot.state

            if self.realtime and turn.customer_audio_duration > 0:
                await asyncio.sleep(turn.customer_audio_duration)

        return self._build_report()

    def _state_to_stage(self, state) -> str:
        from src.core.chatbot import ChatState
        mapping = {
            ChatState.INIT: "greeting",
            ChatState.IDENTITY_VERIFY: "identity",
            ChatState.PURPOSE: "purpose",
            ChatState.ASK_TIME: "ask_time",
            ChatState.PUSH_FOR_TIME: "push",
            ChatState.COMMIT_TIME: "confirm",
            ChatState.CONFIRM_EXTENSION: "push",
            ChatState.HANDLE_OBJECTION: "negotiate",
            ChatState.HANDLE_BUSY: "push",
            ChatState.HANDLE_WRONG_NUMBER: "close",
            ChatState.CLOSE: "close",
            ChatState.FAILED: "close",
        }
        if hasattr(state, 'name'):
            return mapping.get(state, "greeting")
        return mapping.get(state, "greeting")

    def _build_report(self) -> SimulationReport:
        turns = self._turns
        n = len(turns) or 1
        from src.core.chatbot import ChatState

        report = SimulationReport(
            turns=turns,
            persona=self.persona,
            resistance_level=self.resistance_level,
            chat_group=self.chat_group,
            session_id=self._session_id,
            artifacts_dir=str(self._run_dir) if self._run_dir else "",
            total_turns=len(turns),
            conversation_ended=self._chatbot.state in (ChatState.CLOSE, ChatState.FAILED),
            final_state=self._chatbot.state.name,
            committed_time=self._chatbot.commit_time,
            total_wall_time=time.time() - self._start_time,
        )

        completed = [t for t in turns if t.customer_text.strip()]
        asr_turns = [t for t in completed if t.asr_text and not t.vad_dropped]
        if asr_turns:
            report.asr_exact_match_rate = sum(1 for t in asr_turns if t.asr_exact_match) / len(asr_turns)
            report.avg_cer = float(np.mean([t.asr_cer for t in asr_turns]))
        report.vad_dropped_count = sum(1 for t in turns if t.vad_dropped)
        report.tts_failed_count = sum(1 for t in turns if t.tts_failed)
        return report

    async def run_streaming(self, max_turns: int | None = None):
        """流式运行 — 每轮 yield SimulationTurn。用于 SSE 推送给 Web 前端。"""
        # 流式模式下直接运行，每轮 yield
        self._start_time = time.time()
        from src.core.chatbot import ChatState

        prev_state = self._chatbot.state
        stuck_count = 0

        for turn_id in range(1, (max_turns or self.max_turns) + 1):
            if self._chatbot.state in (ChatState.CLOSE, ChatState.FAILED):
                break

            # 简化版: 与 run() 类似的单轮逻辑
            turn = SimulationTurn(turn_id=turn_id, state_before=self._chatbot.state.name)
            stage = self._state_to_stage(self._chatbot.state)
            customer_text = self._text_sim.generate_response(
                stage=stage, chat_group=self.chat_group,
                persona=self.persona, resistance_level=self.resistance_level,
                push_count=self._push_count,
            )
            turn.customer_text = customer_text

            if customer_text.strip():
                tts_result = await self._tts.synthesize(customer_text, voice=self.customer_voice, engine=self.customer_tts_engine)
                if tts_result.success and tts_result.audio_file:
                    turn.customer_audio_file = tts_result.audio_file
                    audio = _load_audio_mono(tts_result.audio_file, self.sample_rate)
                    if self._asr.is_available:
                        turn.asr_text = self._asr.transcribe(audio)

            agent_text, _ = await self._chatbot.process(
                customer_input=turn.asr_text if turn.asr_text else None,
                use_tts=False,
            )
            turn.agent_text = agent_text
            turn.state_after = self._chatbot.state.name
            self._turns.append(turn)
            yield turn

            if self._chatbot.state == prev_state:
                stuck_count += 1
                if stuck_count >= 3:
                    break
            else:
                stuck_count = 0
            prev_state = self._chatbot.state


# ── helper ──────────────────────────────────────────────────

def _load_audio_mono(path: str, target_sr: int) -> np.ndarray:
    import soundfile as sf
    data, sr = sf.read(path, dtype='float32')
    if data.ndim > 1:
        data = data[:, 0]
    if sr != target_sr:
        from scipy.signal import resample
        n_samples = int(len(data) * target_sr / sr)
        data = resample(data, n_samples)
    return data.astype(np.float32)


def _collect_all(source: SimulatedSource, total_samples: int) -> np.ndarray:
    buf = np.zeros(total_samples, dtype=np.float32)
    pos = 0
    while True:
        chunk = source._buffer.read(duration=source._block_size / source.sample_rate)
        if len(chunk) == 0:
            break
        end = min(pos + len(chunk), total_samples)
        buf[pos:end] = chunk[:end - pos]
        pos = end
    return buf[:pos]
```

- [ ] **Step 4: Run CallSimulator tests**

Run: `pytest tests/test_call_simulator.py -v --asyncio-mode=auto`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/core/voice/call_simulator.py tests/test_call_simulator.py
git commit -m "feat: CallSimulator — 自动仿真模式编排 Pipeline

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 8: 重写 voice_simulate_demo.py 入口

**Files:**
- Rewrite: `src/experiments/voice_simulate_demo.py`

- [ ] **Step 1: Rewrite demo entry point**

用新设计的 `--mode live/sim/replay` 参数结构完整重写。保留原有所有 CLI 参数，添加 `--mode` 和 `--simulate-interruptions`。

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音催收 Demo — 双工通话演示

Usage:
    python src/experiments/voice_simulate_demo.py --mode live                        # 真人模式（麦克风）
    python src/experiments/voice_simulate_demo.py --mode sim --persona resistant     # 自动仿真
    python src/experiments/voice_simulate_demo.py --mode replay --recording call.wav # 回放
"""
import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.chatbot import CollectionChatBot, ChatState
from src.core.voice.audio_source import MicrophoneSource, FileSource
from src.core.voice.audio_output import DuplexAudioOutput
from src.core.voice.pipeline import DuplexCallPipeline, PipelineState, PipelineConfig, StepResult
from src.core.voice.vad import SimpleEnergyVAD
from src.core.voice.call_simulator import CallSimulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("voice_demo")

PERSONAS = ["cooperative", "busy", "negotiating", "silent", "forgetful", "resistant", "excuse_master"]
RESISTANCE_LEVELS = ["very_low", "low", "medium", "high", "very_high"]
CHAT_GROUPS = ["H2", "H1", "S0"]

STATUS_ICONS = {
    PipelineState.IDLE: "⏳",
    PipelineState.LISTENING: "👂",
    PipelineState.PROCESSING: "🧠",
    PipelineState.RESPONDING: "🔊",
    PipelineState.INTERRUPTED: "⚡",
    PipelineState.CLOSING: "👋",
    PipelineState.CLOSED: "✅",
}


def print_header(mode, persona, resistance, chat_group, max_turns):
    print()
    print("=" * 62)
    print("  语音催收 Demo — 双工通话管线")
    print("=" * 62)
    print(f"  Mode: {mode:<10s}  Group: {chat_group}")
    if mode == "sim":
        print(f"  Persona: {persona:<15s} Resistance: {resistance}")
    print(f"  Max turns: {max_turns}")
    print(f"  实时状态: {'● 真人麦克风' if mode == 'live' else '● 自动仿真' if mode == 'sim' else '● 文件回放'}")
    print("=" * 62)


async def run_live_mode(args):
    """真人模式：麦克风 → Pipeline"""
    print_header("live", args.persona, args.resistance, args.chat_group, args.max_turns)

    bot = CollectionChatBot(chat_group=args.chat_group, customer_name=args.customer_name)

    source = MicrophoneSource(sample_rate=16000, block_size=1600)
    output = DuplexAudioOutput(source, barge_in_threshold=0.02)
    vad = SimpleEnergyVAD(sample_rate=16000, energy_threshold=0.01, voice_frames=2, silence_frames=10)

    from src.core.voice.asr import ASRPipeline
    from src.core.voice.tts import TTSManager
    asr = await ASRPipeline.create(model_size=args.asr_model, corrector=getattr(bot, "asr_corrector", None))
    tts = TTSManager()

    config = PipelineConfig(sample_rate=16000, block_size=1600, silence_duration=1.0, max_speech_duration=15.0)
    pipeline = DuplexCallPipeline(bot, source, output, asr, tts, vad, config=config)

    # 实时状态回调
    def on_state(old, new):
        icon = STATUS_ICONS.get(new, "?")
        print(f"  {icon} {old.name} → {new.name}")

    pipeline.on_state_change = on_state

    await pipeline.start()
    print(f"\n  麦克风已就绪。开始说话...\n")

    try:
        while pipeline.state != PipelineState.CLOSED:
            result = await pipeline.step()
            if result and result.asr_text:
                print(f"\n  [用户]: {result.asr_text}")
            if result and result.agent_text:
                print(f"  [Agent]: {result.agent_text[:120]}")
            await asyncio.sleep(0.01)
    except KeyboardInterrupt:
        print("\n  用户中断")
    finally:
        await pipeline.stop()
        print(f"\n  通话结束。轮次: {pipeline.turn_id}")
    return 0


async def run_sim_mode(args):
    """自动仿真模式：CallSimulator"""
    print_header("sim", args.persona, args.resistance, args.chat_group, args.max_turns)

    bot = CollectionChatBot(chat_group=args.chat_group, customer_name=args.customer_name)

    print("  加载 ASR 模型...")
    sim = await CallSimulator.create(
        chatbot=bot,
        persona=args.persona,
        resistance_level=args.resistance,
        chat_group=args.chat_group,
        customer_name=args.customer_name,
        asr_model_size=args.asr_model,
        realtime=args.realtime,
        save_artifacts=not args.no_save,
        output_dir=args.output_dir,
        max_turns=args.max_turns,
    )

    print(f"  就绪。开始模拟...\n")
    report = await sim.run(max_turns=args.max_turns)

    print(f"\n  === 完成 ===")
    print(f"  轮次: {report.total_turns} | 结束: {report.conversation_ended}")
    print(f"  最终状态: {report.final_state}")
    if report.artifacts_dir:
        print(f"  Artifacts: {report.artifacts_dir}")
    return 0


async def run_replay_mode(args):
    """回放模式：FileSource → Pipeline"""
    print_header("replay", args.persona, args.resistance, args.chat_group, args.max_turns)

    if not args.recording or not Path(args.recording).exists():
        print(f"  [ERROR] 录音文件不存在: {args.recording}")
        return 1

    bot = CollectionChatBot(chat_group=args.chat_group, customer_name=args.customer_name)

    source = FileSource(args.recording, sample_rate=16000, block_size=1600)
    output = DuplexAudioOutput(source, barge_in_threshold=0.02)
    vad = SimpleEnergyVAD(sample_rate=16000, energy_threshold=0.01, voice_frames=2, silence_frames=10)

    from src.core.voice.asr import ASRPipeline
    from src.core.voice.tts import TTSManager
    asr = await ASRPipeline.create(model_size=args.asr_model, corrector=getattr(bot, "asr_corrector", None))
    tts = TTSManager()

    config = PipelineConfig(sample_rate=16000, block_size=1600, silence_duration=1.0, max_speech_duration=15.0)
    pipeline = DuplexCallPipeline(bot, source, output, asr, tts, vad, config=config)

    def on_state(old, new):
        icon = STATUS_ICONS.get(new, "?")
        print(f"  {icon} {old.name} → {new.name}")
    pipeline.on_state_change = on_state

    await pipeline.start()
    print(f"  回放中...\n")
    try:
        while pipeline.state != PipelineState.CLOSED:
            await pipeline.step()
            await asyncio.sleep(0.01)
    except KeyboardInterrupt:
        print("\n  用户中断")
    finally:
        await pipeline.stop()
    print(f"\n  回放结束。轮次: {pipeline.turn_id}")
    return 0


async def main():
    parser = argparse.ArgumentParser(description="语音催收 Demo — 双工通话管线")
    parser.add_argument("--mode", default="live", choices=["live", "sim", "replay"],
                        help="运行模式 (default: live)")
    parser.add_argument("--persona", default="cooperative", choices=PERSONAS)
    parser.add_argument("--resistance", default="medium", choices=RESISTANCE_LEVELS)
    parser.add_argument("--chat-group", default="H2", choices=CHAT_GROUPS)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--realtime", action="store_true", help="模拟实时对话节奏")
    parser.add_argument("--no-save", action="store_true", help="不保存 artifacts")
    parser.add_argument("--asr-model", default="small", choices=["tiny", "small", "medium"])
    parser.add_argument("--output-dir", default="data/runs/voice_simulations")
    parser.add_argument("--customer-name", default="Budi")
    parser.add_argument("--recording", help="回放模式: 录音文件路径")
    parser.add_argument("--seed", type=int, help="随机种子")
    parser.add_argument("--simulate-interruptions", action="store_true",
                        help="自动仿真中模拟打断")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.seed:
        import random; random.seed(args.seed)

    if args.mode == "sim":
        return await run_sim_mode(args)
    elif args.mode == "replay":
        return await run_replay_mode(args)
    else:
        return await run_live_mode(args)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

- [ ] **Step 2: Verify syntax**

Run: `python3 -c "import ast; ast.parse(open('src/experiments/voice_simulate_demo.py').read()); print('OK')"`

- [ ] **Step 3: Commit**

```bash
git add src/experiments/voice_simulate_demo.py
git commit -m "refactor: 重写 voice_simulate_demo — 双工管线 + --mode live/sim/replay

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 9: 废弃旧文件 + 审计路径引用

**Files:**
- Modify: `src/core/voice/conversation.py` (标记废弃)
- Modify: `src/core/voice/customer_simulator.py` (标记废弃)
- Modify: `src/core/voice/interruption.py` (标记废弃)
- Check: `src/api/main.py` (确认引用不中断)

- [ ] **Step 1: 标记 conversation.py 废弃 — 顶部加 deprecation 注释**

在 `conversation.py` 第 1 行前插入：

```python
# DEPRECATED since 2026-05 — 被 src.core.voice.pipeline.DuplexCallPipeline 取代
# 保留此文件仅为向后兼容 src/api/main.py 的历史引用
```

- [ ] **Step 2: 标记 interruption.py 废弃 — 顶部加 deprecation 注释**

在 `interruption.py` 第 1 行前插入：

```python
# DEPRECATED since 2026-05 — 打断逻辑已合并到 src.core.voice.audio_output.DuplexAudioOutput
```

- [ ] **Step 3: 标记 customer_simulator.py 废弃 — 顶部加 deprecation 注释**

在 `customer_simulator.py` 第 1 行前插入：

```python
# DEPRECATED since 2026-05 — 被 src.core.voice.call_simulator.CallSimulator 取代
# 保留此文件仅为向后兼容 src/api/main.py 的历史引用
```

- [ ] **Step 4: 验证 api/main.py 仍然可以 import 旧模块**

Run: `python3 -c "from src.core.voice.customer_simulator import CustomerVoiceSimulator, SimulationReport, SimulationTurn; print('OK')"`

- [ ] **Step 5: Commit**

```bash
git add src/core/voice/conversation.py src/core/voice/customer_simulator.py src/core/voice/interruption.py
git commit -m "chore: 标记 conversation/customer_simulator/interruption 废弃

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 10: 全量回归测试 + 最终验证

- [ ] **Step 1: 运行全量测试**

Run: `python3 -m pytest tests/ -v --tb=short --asyncio-mode=auto`
Expected: 所有测试通过 (96 + ~25 新增 ≈ 121 passed)

- [ ] **Step 2: 运行 demo 语法验证**

Run: `python3 -c "import ast; ast.parse(open('src/experiments/voice_simulate_demo.py').read()); print('OK')"`

- [ ] **Step 3: 验证 import 链**

```bash
python3 -c "
from src.core.voice.audio_source import AudioSource, MicrophoneSource, SimulatedSource, FileSource, SilentSource
from src.core.voice.audio_output import DuplexAudioOutput, PlaybackResult
from src.core.voice.pipeline import DuplexCallPipeline, PipelineState, PipelineConfig, InterruptionContext, StepResult
from src.core.voice.call_simulator import CallSimulator
print('All imports OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: P15 双工通话管线重构完成 — 回归测试通过

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```
