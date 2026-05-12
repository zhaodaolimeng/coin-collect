#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""音频输入源抽象层 — 统一麦克风/仿真/文件输入"""
import asyncio
import logging
from abc import ABC, abstractmethod

import numpy as np

from core.voice.audio_io import RingBuffer

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
        data = self._buffer.read(duration=0.05)
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
