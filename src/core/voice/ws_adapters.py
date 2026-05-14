#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WebSocket 适配器 — 将 AudioSource/AudioOutput 从 sounddevice 解耦"""
import asyncio
import logging
from collections import deque
from typing import Optional, Callable, Awaitable

import numpy as np

from core.voice.audio_source import AudioSource
from core.voice.audio_output import DuplexAudioOutput, PlaybackResult

logger = logging.getLogger(__name__)


class WebSocketAudioSource(AudioSource):
    """从 asyncio.Queue 读取浏览器发来的音频块"""

    def __init__(self, sample_rate: int = 16000, block_size: int = 1600):
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._recent_samples: deque = deque(maxlen=int(0.3 * sample_rate))
        self._running = False
        self.overflow_count = 0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def feed_chunk(self, chunk: bytes | np.ndarray) -> None:
        """浏览器音频块入队。bytes → float32。

        同步更新 _recent_samples，确保播放期间 current_rms() 仍能反映
        用户实时语音，使服务端打断检测不依赖 read_chunk() 的消费速度。
        """
        if isinstance(chunk, bytes):
            arr = np.frombuffer(chunk, dtype=np.float32)
        else:
            arr = chunk.astype(np.float32)
        if len(arr) != self._block_size:
            if self.overflow_count < 3:
                logger.warning(f"Audio chunk size mismatch: got {len(arr)}, expected {self._block_size}")
            if len(arr) < self._block_size:
                arr = np.pad(arr, (0, self._block_size - len(arr)))
            elif len(arr) > self._block_size:
                arr = arr[:self._block_size]
        self._recent_samples.extend(arr.tolist())
        try:
            self._queue.put_nowait(arr)
        except asyncio.QueueFull:
            self.overflow_count += 1

    async def start(self):
        self._running = True

    async def stop(self):
        self._running = False
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def read_chunk(self) -> np.ndarray | None:
        if not self._running:
            return None
        try:
            chunk = await asyncio.wait_for(self._queue.get(), timeout=0.05)
            self._recent_samples.extend(chunk.tolist())
            return chunk
        except asyncio.TimeoutError:
            return None

    def flush(self, keep_recent_s: float = 1.0):
        """清空音频队列，可选择保留最近 N 秒的音频。

        播放结束后调用，丢弃长时间积压的静音数据，
        同时保留最近几秒的用户语音（可能包含打断语句或播放末期的有效输入）。
        """
        keep_samples = int(keep_recent_s * self._sample_rate)
        kept_chunks = []
        kept_total = 0

        # 从队尾向前收集最近 keep_recent_s 秒的数据
        all_chunks = []
        while not self._queue.empty():
            try:
                all_chunks.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        for chunk in reversed(all_chunks):
            if kept_total >= keep_samples:
                break
            kept_chunks.append(chunk)
            kept_total += len(chunk)

        # 恢复保留的 chunk 到队列（保持原始顺序）
        for chunk in reversed(kept_chunks):
            self._queue.put_nowait(chunk)

        # 清除 _recent_samples 中的过期数据
        self._recent_samples.clear()
        for chunk in kept_chunks:
            self._recent_samples.extend(chunk.tolist())

        if len(all_chunks) > len(kept_chunks):
            logger.debug(f"Flushed {len(all_chunks) - len(kept_chunks)} stale chunks, "
                         f"kept {len(kept_chunks)} chunks ({kept_total / self._sample_rate:.1f}s)")

    def current_rms(self) -> float:
        if not self._recent_samples:
            return 0.0
        arr = np.array(list(self._recent_samples), dtype=np.float32)
        return float(np.sqrt(np.mean(arr ** 2)))


class WebSocketAudioOutput(DuplexAudioOutput):
    """将播放回调到浏览器（替代 sounddevice），打断检测保留服务端"""

    def __init__(
        self,
        source: AudioSource,
        *,
        send_chunk: Optional[Callable[[bytes, int], Awaitable[None]]] = None,
        barge_in_threshold: float = 0.02,
    ):
        super().__init__(source, barge_in_threshold=barge_in_threshold)
        self._send_chunk = send_chunk

    def set_send_chunk(self, send_chunk: Callable[[bytes, int], Awaitable[None]]):
        self._send_chunk = send_chunk

    async def _play_with_interrupt_detection(self, audio: np.ndarray) -> PlaybackResult:
        """覆盖父类：发送到浏览器而非 sounddevice。

        两阶段:
        1. 以最快速度发送所有 chunk（前端调度精确播放时间，消除间隙）
        2. 等待实际播放时长结束（保持 RESPONDING 状态，响应前端打断）
        """
        sr = self._source.sample_rate
        chunk_samples = int(self._chunk_duration * sr)
        pos = 0

        if self._send_chunk is None:
            logger.warning("send_chunk 未设置，无法播放")
            return PlaybackResult.FAILED

        # Phase 1: 快速发送所有 chunk
        while pos < len(audio) and not self._stop_requested:
            end = min(pos + chunk_samples, len(audio))
            chunk = audio[pos:end]

            int16_chunk = (chunk * 32767).astype(np.int16)
            try:
                await self._send_chunk(int16_chunk.tobytes(), sr)
            except Exception as e:
                logger.error(f"发送音频块失败: {e}")

            pos = end
            await asyncio.sleep(0)

        if self._stop_requested:
            return PlaybackResult.INTERRUPTED

        # Phase 2: 等待前端实际播放完成
        total_duration = len(audio) / sr
        elapsed = 0.0
        check_interval = 0.1
        while elapsed < total_duration and not self._stop_requested:
            await asyncio.sleep(check_interval)
            elapsed += check_interval

        if self._stop_requested:
            return PlaybackResult.INTERRUPTED
        return PlaybackResult.COMPLETED

    def stop(self):
        """立即停止当前播放"""
        self._stop_requested = True
