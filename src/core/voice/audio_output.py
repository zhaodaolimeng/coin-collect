#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双工音频播放输出 — Agent TTS 播放 + 持续打断检测 + ducking"""
import asyncio
import logging
from enum import Enum

import numpy as np

from core.voice.audio_source import AudioSource

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
        loop = asyncio.get_running_loop()

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

            # 在 executor 中播放，不阻塞事件循环
            await loop.run_in_executor(None, _play_chunk_sync, chunk, sr)

            # 检查打断
            rms = self._source.current_rms()
            if rms > self._barge_in_threshold:
                if not self._is_ducking:
                    self._is_ducking = True
                    # 二次确认
                    await asyncio.sleep(self._confirmation_duration)
                    rms_confirm = self._source.current_rms()
                    if rms_confirm > self._barge_in_threshold:
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


def _play_chunk_sync(chunk: np.ndarray, sample_rate: int):
    """同步播放音频块（在 executor 线程中运行）"""
    import sounddevice as sd
    sd.play(chunk, samplerate=sample_rate, blocking=True)
