#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音活动检测 (VAD) 模块
用于检测音频流中的语音活动
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)


class VADState(Enum):
    """VAD状态"""
    SILENCE = "silence"
    VOICE = "voice"
    UNKNOWN = "unknown"


@dataclass
class VADResult:
    """VAD检测结果"""
    state: VADState
    confidence: float
    timestamp: float
    duration: float = 0.0


class SimpleEnergyVAD:
    """
    基于能量的简单VAD检测器
    适用于实时语音活动检测
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        energy_threshold: float = 0.01,
        silence_frames: int = 10,
        voice_frames: int = 3
    ):
        """
        初始化VAD检测器

        Args:
            sample_rate: 采样率 (Hz)
            frame_duration_ms: 帧长度 (毫秒)
            energy_threshold: 能量阈值
            silence_frames: 连续静音帧数才判定为静音
            voice_frames: 连续语音帧数才判定为语音
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.energy_threshold = energy_threshold
        self.silence_frames = silence_frames
        self.voice_frames = voice_frames

        self.state = VADState.UNKNOWN
        self.silence_counter = 0
        self.voice_counter = 0
        self.start_time = None

    def _calculate_energy(self, audio_frame: np.ndarray) -> float:
        """
        计算音频帧的能量（RMS 幅值，未归一化）

        Args:
            audio_frame: 音频帧数据 (float32, 范围约为 [-1.0, 1.0])

        Returns:
            RMS 能量值 (0.0 = 纯静音, ~0.01 = 安静人声, ~0.1+ = 正常语音)
        """
        if len(audio_frame) == 0:
            return 0.0

        rms = np.sqrt(np.mean(audio_frame.astype(np.float64) ** 2))
        return float(rms)

    def process_frame(self, audio_frame: np.ndarray) -> VADResult:
        """
        处理单个音频帧

        Args:
            audio_frame: 音频帧数据

        Returns:
            VAD检测结果
        """
        energy = self._calculate_energy(audio_frame)
        timestamp = time.time()

        if energy > self.energy_threshold:
            self.voice_counter += 1
            self.silence_counter = 0

            if self.voice_counter >= self.voice_frames:
                if self.state != VADState.VOICE:
                    self.start_time = timestamp
                self.state = VADState.VOICE
        else:
            self.silence_counter += 1
            self.voice_counter = 0

            if self.silence_counter >= self.silence_frames:
                if self.state == VADState.VOICE and self.start_time:
                    duration = timestamp - self.start_time
                else:
                    duration = 0.0
                self.state = VADState.SILENCE
                self.start_time = None

        # 计算置信度
        if self.state == VADState.VOICE:
            confidence = min(1.0, energy / (self.energy_threshold * 2))
        elif self.state == VADState.SILENCE:
            confidence = min(1.0, (self.energy_threshold - energy) / self.energy_threshold)
        else:
            confidence = 0.5

        duration = timestamp - self.start_time if self.start_time else 0.0

        return VADResult(
            state=self.state,
            confidence=confidence,
            timestamp=timestamp,
            duration=duration
        )

    def reset(self):
        """重置VAD状态"""
        self.state = VADState.UNKNOWN
        self.silence_counter = 0
        self.voice_counter = 0
        self.start_time = None


class SileroVAD:
    """基于 Silero 神经网络的 VAD 检测器。

    将 pipeline 块内部分割为 Silero 要求的帧大小（16kHz=512, 8kHz=256），
    逐帧推理取最大语音概率。对低幅值音频自动增益后再送模型。
    """

    # 目标 RMS，低于此值的子帧会被放大
    _TARGET_RMS = 0.05

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 128,
        energy_threshold: float = 0.5,
        voice_frames: int = 1,
        silence_frames: int = 3,
    ):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.energy_threshold = energy_threshold
        self.voice_frames = voice_frames
        self.silence_frames = silence_frames

        self.state = VADState.UNKNOWN
        self._model = None
        self._vad_frame_size = 512 if sample_rate == 16000 else 256
        self._call_count = 0
        self._voice_counter = 0
        self._silence_counter = 0
        self._load_model()

    def _load_model(self):
        try:
            from silero_vad import load_silero_vad
            self._model = load_silero_vad()
            logger.info(f"[SileroVAD] 模型加载成功 sample_rate={self.sample_rate}")
        except Exception as e:
            logger.error(f"[SileroVAD] 模型加载失败: {e}")
            self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def process_frame(self, audio_frame: np.ndarray) -> VADResult:
        if self._model is None:
            return VADResult(
                state=VADState.SILENCE, confidence=0.0,
                timestamp=time.time(), duration=0.0,
            )

        import torch
        fs = self._vad_frame_size
        n_full = len(audio_frame) // fs
        probs = []
        for i in range(n_full):
            sub = audio_frame[i * fs:(i + 1) * fs].copy()
            # 自动增益：过低的幅值会让神经网络失效
            rms = float(np.sqrt(np.mean(sub.astype(np.float64) ** 2)))
            if 0 < rms < self._TARGET_RMS:
                sub *= (self._TARGET_RMS / rms)
            tensor = torch.from_numpy(sub.astype(np.float32))
            probs.append(self._model(tensor, self.sample_rate).item())

        remainder = len(audio_frame) % fs
        if remainder > 0:
            sub = np.zeros(fs, dtype=np.float32)
            sub[:remainder] = audio_frame[n_full * fs:]
            rms = float(np.sqrt(np.mean(sub.astype(np.float64) ** 2)))
            if 0 < rms < self._TARGET_RMS:
                sub *= (self._TARGET_RMS / rms)
            tensor = torch.from_numpy(sub)
            probs.append(self._model(tensor, self.sample_rate).item())

        if not probs:
            return VADResult(
                state=VADState.SILENCE, confidence=0.0,
                timestamp=time.time(), duration=0.0,
            )

        speech_prob = max(probs)
        timestamp = time.time()

        # 前 20 帧 + 每 50 帧输出概率，便于排查模型是否生效
        self._call_count += 1
        if self._call_count <= 20 or self._call_count % 50 == 0:
            logger.debug(f"[SileroVAD] frame#{self._call_count} "
                         f"speech_prob={speech_prob:.4f} threshold={self.energy_threshold} "
                         f"state={'VOICE' if speech_prob >= self.energy_threshold else 'SILENCE'}")

        if speech_prob >= self.energy_threshold:
            self._voice_counter += 1
            self._silence_counter = 0
            if self._voice_counter >= self.voice_frames:
                self.state = VADState.VOICE
            confidence = speech_prob
        else:
            self._silence_counter += 1
            self._voice_counter = 0
            if self._silence_counter >= self.silence_frames:
                self.state = VADState.SILENCE
            confidence = speech_prob

        return VADResult(
            state=self.state,
            confidence=float(confidence),
            timestamp=timestamp,
            duration=0.0,
        )

    def find_speech_segments(
        self, audio: np.ndarray,
        threshold: float | None = None,
        min_speech_ms: int = 400,
        max_gap_ms: int = 300,
    ) -> list[tuple[int, int]]:
        """精确定位语音段边界（离线分析），返回 [(start_sample, end_sample), ...]。

        逐帧推理 SileroVAD，获取 speech_prob 时间线后查找连续语音区。
        max_gap_ms 以内的停顿视为语流内停顿，不切分。
        min_speech_ms 以下的短段视为噪声碎片，丢弃。

        Args:
            audio: float32 音频数组
            threshold: 语音概率阈值，默认使用 self.energy_threshold
            min_speech_ms: 最短语音段（毫秒）
            max_gap_ms: 语流内最大间隙（毫秒）
        """
        if self._model is None or len(audio) == 0:
            return [(0, len(audio))] if len(audio) > 0 else []

        threshold = threshold if threshold is not None else self.energy_threshold
        fs = self._vad_frame_size
        n_full = len(audio) // fs
        if n_full == 0:
            return [(0, len(audio))]

        import torch
        probs = np.empty(n_full, dtype=np.float32)
        for i in range(n_full):
            sub = audio[i * fs:(i + 1) * fs].copy()
            rms = float(np.sqrt(np.mean(sub.astype(np.float64) ** 2)))
            if 0 < rms < self._TARGET_RMS:
                sub *= (self._TARGET_RMS / rms)
            tensor = torch.from_numpy(sub.astype(np.float32))
            probs[i] = self._model(tensor, self.sample_rate).item()

        is_speech = probs >= threshold
        min_frames = max(1, min_speech_ms * self.sample_rate // 1000 // fs)
        max_gap_frames = max_gap_ms * self.sample_rate // 1000 // fs

        segments: list[tuple[int, int]] = []
        i = 0
        while i < n_full:
            if is_speech[i]:
                start = i
                gap_start = -1
                while i < n_full:
                    if is_speech[i]:
                        gap_start = -1
                    else:
                        if gap_start < 0:
                            gap_start = i
                        if i - gap_start >= max_gap_frames:
                            break
                    i += 1
                end = i
                while end > start and not is_speech[end - 1]:
                    end -= 1
                if end - start >= min_frames:
                    segments.append((start * fs, min(end * fs, len(audio))))
            else:
                i += 1

        return segments if segments else []

    def reset(self):
        self.state = VADState.UNKNOWN
        self._call_count = 0
        self._voice_counter = 0
        self._silence_counter = 0


class VADAnalyzer:
    """
    VAD分析器 - 用于分析完整音频文件
    """

    def __init__(self, vad: Optional[SimpleEnergyVAD] = None):
        self.vad = vad or SimpleEnergyVAD()

    def analyze_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> List[Tuple[float, float, VADState]]:
        """
        分析完整音频数据

        Args:
            audio_data: 音频数据
            sample_rate: 采样率

        Returns:
            语音活动片段列表 [(start_time, end_time, state)]
        """
        frame_size = self.vad.frame_size
        segments = []
        current_state = VADState.UNKNOWN
        segment_start = 0.0

        num_frames = len(audio_data) // frame_size

        for i in range(num_frames):
            start_idx = i * frame_size
            end_idx = start_idx + frame_size
            frame = audio_data[start_idx:end_idx]

            result = self.vad.process_frame(frame)
            timestamp = i * self.vad.frame_duration_ms / 1000.0

            if result.state != current_state:
                if current_state != VADState.UNKNOWN:
                    segments.append((segment_start, timestamp, current_state))
                current_state = result.state
                segment_start = timestamp

        # 添加最后一个片段
        if current_state != VADState.UNKNOWN:
            end_time = num_frames * self.vad.frame_duration_ms / 1000.0
            segments.append((segment_start, end_time, current_state))

        return segments

    def get_voice_segments(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> List[Tuple[float, float]]:
        """
        获取语音活动片段

        Args:
            audio_data: 音频数据
            sample_rate: 采样率

        Returns:
            语音片段列表 [(start_time, end_time)]
        """
        segments = self.analyze_audio(audio_data, sample_rate)
        return [(s, e) for s, e, state in segments if state == VADState.VOICE]

    def calculate_speech_ratio(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> float:
        """
        计算语音占比

        Args:
            audio_data: 音频数据
            sample_rate: 采样率

        Returns:
            语音占比 (0.0 - 1.0)
        """
        segments = self.analyze_audio(audio_data, sample_rate)
        total_duration = len(audio_data) / sample_rate

        if total_duration == 0:
            return 0.0

        speech_duration = sum(e - s for s, e, state in segments if state == VADState.VOICE)
        return speech_duration / total_duration


# 简单测试
if __name__ == "__main__":
    print("VAD模块加载成功")

    # 创建VAD检测器
    vad = SimpleEnergyVAD()
    print(f"采样率: {vad.sample_rate}")
    print(f"帧大小: {vad.frame_size}")
    print(f"能量阈值: {vad.energy_threshold}")

    # 测试静音帧
    print("\n--- 测试静音帧 ---")
    silence_frame = np.zeros(vad.frame_size, dtype=np.float32)
    result = vad.process_frame(silence_frame)
    print(f"状态: {result.state}, 置信度: {result.confidence:.2f}")

    # 测试语音帧（模拟噪声）
    print("\n--- 测试语音帧 ---")
    voice_frame = np.random.randn(vad.frame_size).astype(np.float32) * 0.1
    result = vad.process_frame(voice_frame)
    print(f"状态: {result.state}, 置信度: {result.confidence:.2f}")

    print("\nVAD模块测试完成")
