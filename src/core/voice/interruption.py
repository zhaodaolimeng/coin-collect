# DEPRECATED since 2026-05 — 打断逻辑已合并到 src.core.voice.audio_output.DuplexAudioOutput
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能打断处理模块
用于在TTS播放过程中检测用户打断并智能处理
"""
import asyncio
import numpy as np
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum
import time
from pathlib import Path

from core.voice.vad import SimpleEnergyVAD, VADState, VADResult


class InterruptionType(Enum):
    """打断类型"""
    NO_INTERRUPTION = "no_interruption"
    SHORT_INTERRUPTION = "short_interruption"  # 短打断（嗯、啊等）
    LONG_INTERRUPTION = "long_interruption"    # 长打断（用户说话）
    UNKNOWN_INTERRUPTION = "unknown_interruption"


@dataclass
class InterruptionEvent:
    """打断事件"""
    type: InterruptionType
    start_time: float
    end_time: float
    duration: float
    audio_chunk: Optional[np.ndarray] = None


class InterruptionHandler:
    """
    智能打断处理器
    """

    def __init__(
        self,
        vad: Optional[SimpleEnergyVAD] = None,
        short_interruption_threshold_ms: float = 500,
        min_silence_before_stop_ms: float = 300,
        grace_period_ms: float = 200
    ):
        """
        初始化打断处理器

        Args:
            vad: VAD检测器
            short_interruption_threshold_ms: 短打断阈值（毫秒）
            min_silence_before_stop_ms: 停止前的最小静音时间（毫秒）
            grace_period_ms: 开始播放后的宽限期（毫秒）
        """
        self.vad = vad or SimpleEnergyVAD()
        self.short_interruption_threshold_ms = short_interruption_threshold_ms
        self.min_silence_before_stop_ms = min_silence_before_stop_ms
        self.grace_period_ms = grace_period_ms

        self.is_monitoring = False
        self.playback_start_time = 0.0
        self.interruption_start_time = 0.0
        self.current_interruption: Optional[InterruptionEvent] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._audio_buffer: list = []

        # 回调函数
        self.on_interruption: Optional[Callable[[InterruptionEvent], Awaitable[None]]] = None
        self.on_playback_stop: Optional[Callable[[], Awaitable[None]]] = None

    async def start_monitoring(self):
        """开始监听打断"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.playback_start_time = time.time()
        self.vad.reset()
        self._audio_buffer = []
        self.current_interruption = None

        print("[打断处理] 开始监听")

    async def stop_monitoring(self):
        """停止监听打断"""
        self.is_monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None

        print("[打断处理] 停止监听")

    async def process_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[InterruptionEvent]:
        """
        处理音频块，检测打断

        Args:
            audio_chunk: 音频数据块

        Returns:
            打断事件（如果有）
        """
        if not self.is_monitoring:
            return None

        current_time = time.time()
        elapsed_ms = (current_time - self.playback_start_time) * 1000

        # 宽限期内不检测
        if elapsed_ms < self.grace_period_ms:
            return None

        # 缓存音频
        self._audio_buffer.append(audio_chunk.copy())

        # 处理VAD
        result = self.vad.process_frame(audio_chunk)

        if result.state == VADState.VOICE:
            if self.current_interruption is None:
                # 开始新的打断
                self.current_interruption = InterruptionEvent(
                    type=InterruptionType.UNKNOWN_INTERRUPTION,
                    start_time=current_time,
                    end_time=current_time,
                    duration=0.0
                )
                print(f"[打断处理] 检测到语音活动开始")
            else:
                # 更新当前打断
                self.current_interruption.end_time = current_time
                self.current_interruption.duration = (
                    self.current_interruption.end_time - self.current_interruption.start_time
                )

        elif result.state == VADState.SILENCE and self.current_interruption is not None:
            # 语音结束，判断打断类型
            duration_ms = self.current_interruption.duration * 1000

            if duration_ms < self.short_interruption_threshold_ms:
                self.current_interruption.type = InterruptionType.SHORT_INTERRUPTION
                print(f"[打断处理] 短打断 ({duration_ms:.0f}ms)")
            else:
                self.current_interruption.type = InterruptionType.LONG_INTERRUPTION
                print(f"[打断处理] 长打断 ({duration_ms:.0f}ms)")

            # 合并音频
            if self._audio_buffer:
                self.current_interruption.audio_chunk = np.concatenate(self._audio_buffer)

            event = self.current_interruption
            self.current_interruption = None
            self._audio_buffer = []

            # 触发回调
            if self.on_interruption:
                await self.on_interruption(event)

            return event

        return None

    def should_stop_playback(self, event: InterruptionEvent) -> bool:
        """
        判断是否应该停止播放

        Args:
            event: 打断事件

        Returns:
            是否应该停止
        """
        # 只有长打断才停止播放
        return event.type == InterruptionType.LONG_INTERRUPTION


class PlaybackController:
    """
    播放控制器 - 集成TTS播放和打断处理
    """

    def __init__(
        self,
        interruption_handler: Optional[InterruptionHandler] = None
    ):
        self.interruption_handler = interruption_handler or InterruptionHandler()
        self.is_playing = False
        self._should_stop = False

    async def play_with_interruption_detection(
        self,
        audio_file: str,
        audio_input_stream: Optional[Callable[[], Awaitable[Optional[np.ndarray]]]] = None
    ):
        """
        播放音频并检测打断

        Args:
            audio_file: 音频文件路径
            audio_input_stream: 音频输入流回调函数
        """
        self.is_playing = True
        self._should_stop = False

        print(f"[播放控制] 开始播放: {audio_file}")

        # 设置打断回调
        async def handle_interruption(event: InterruptionEvent):
            if self.interruption_handler.should_stop_playback(event):
                print(f"[播放控制] 检测到打断，停止播放")
                self._should_stop = True

        self.interruption_handler.on_interruption = handle_interruption
        await self.interruption_handler.start_monitoring()

        try:
            # 这里应该集成实际的音频播放逻辑
            # 模拟播放过程
            for i in range(50):  # 模拟50个播放块
                if self._should_stop:
                    break

                # 模拟播放进度
                await asyncio.sleep(0.1)

                # 如果有音频输入流，处理它
                if audio_input_stream:
                    audio_chunk = await audio_input_stream()
                    if audio_chunk is not None:
                        await self.interruption_handler.process_audio_chunk(audio_chunk)

            if self._should_stop:
                print("[播放控制] 播放被打断")
            else:
                print("[播放控制] 播放完成")

        finally:
            await self.interruption_handler.stop_monitoring()
            self.is_playing = False

    def stop(self):
        """停止播放"""
        self._should_stop = True


# 简单测试
if __name__ == "__main__":
    print("打断处理模块加载成功")

    # 创建组件
    vad = SimpleEnergyVAD()
    handler = InterruptionHandler(vad)
    controller = PlaybackController(handler)

    print("\n组件初始化完成")
    print(f"短打断阈值: {handler.short_interruption_threshold_ms}ms")
    print(f"宽限期: {handler.grace_period_ms}ms")
