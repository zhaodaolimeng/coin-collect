#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时ASR语音识别模块
封装Faster-Whisper，支持印尼语流式识别，与ASRCorrector串联
"""
import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ASRResult:
    """ASR识别结果"""
    text: str
    confidence: float = 1.0
    language: str = "id"
    duration: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


class RealTimeASR:
    """
    实时语音识别引擎
    基于Faster-Whisper，面向印尼语催收场景优化
    """

    _instance = None
    _lock = asyncio.Lock()

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "int8",
        language: str = "id",
        beam_size: int = 5,
        sample_rate: int = 16000,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.sample_rate = sample_rate
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=2)

    @classmethod
    async def get_instance(cls, **kwargs) -> "RealTimeASR":
        """获取单例实例"""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(**kwargs)
                cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        """加载Faster-Whisper模型"""
        try:
            from faster_whisper import WhisperModel
            device = self.device
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self._model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=self.compute_type,
            )
            logger.info(f"ASR model loaded: {self.model_size} on {device}")
        except ImportError:
            logger.error("faster-whisper not installed")
            self._model = None
        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")
            self._model = None

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def transcribe_file(self, audio_path: str) -> ASRResult:
        """
        转写整个音频文件（离线模式）

        Args:
            audio_path: 音频文件路径

        Returns:
            ASRResult with full transcription
        """
        if not self._model:
            return ASRResult(
                text="",
                success=False,
                error_message="ASR model not loaded",
            )

        start = time.time()
        try:
            segments, info = self._model.transcribe(
                audio_path,
                language=self.language,
                beam_size=self.beam_size,
                word_timestamps=False,
            )
            text_parts = []
            confidence_sum = 0.0
            segment_count = 0

            for segment in segments:
                text_parts.append(segment.text.strip())
                confidence_sum += segment.avg_logprob
                segment_count += 1

            text = " ".join(text_parts)
            avg_confidence = (
                np.exp(confidence_sum / segment_count) if segment_count > 0 else 0.0
            )
            duration = time.time() - start

            return ASRResult(
                text=text,
                confidence=float(avg_confidence),
                language=info.language if info else self.language,
                duration=duration,
            )
        except Exception as e:
            logger.error(f"ASR transcribe_file error: {e}")
            return ASRResult(
                text="",
                success=False,
                error_message=str(e),
            )

    def transcribe_array(self, audio: np.ndarray) -> ASRResult:
        """
        从numpy数组转写（流式模式）

        Args:
            audio: float32 array, shape (n_samples,), 16kHz

        Returns:
            ASRResult
        """
        if not self._model:
            return ASRResult(
                text="",
                success=False,
                error_message="ASR model not loaded",
            )

        if len(audio) == 0 or self._is_silence(audio):
            return ASRResult(text="", confidence=0.0)

        start = time.time()
        try:
            audio = audio.astype(np.float32)
            segments, info = self._model.transcribe(
                audio,
                language=self.language,
                beam_size=self.beam_size,
                word_timestamps=False,
                vad_filter=True,
            )
            text_parts = []
            confidence_sum = 0.0
            segment_count = 0

            for segment in segments:
                text_parts.append(segment.text.strip())
                confidence_sum += segment.avg_logprob
                segment_count += 1

            text = " ".join(text_parts)
            avg_confidence = (
                np.exp(confidence_sum / segment_count) if segment_count > 0 else 0.0
            )
            duration = time.time() - start

            return ASRResult(
                text=text,
                confidence=float(avg_confidence),
                language=info.language if info else self.language,
                duration=duration,
            )
        except Exception as e:
            logger.error(f"ASR transcribe_array error: {e}")
            return ASRResult(
                text="",
                success=False,
                error_message=str(e),
            )

    async def transcribe_async(self, audio: np.ndarray) -> ASRResult:
        """异步转写，避免阻塞事件循环"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.transcribe_array, audio)

    def _is_silence(self, audio: np.ndarray, threshold: float = 0.005) -> bool:
        """检查音频片段是否为静音"""
        rms = np.sqrt(np.mean(audio ** 2))
        return rms < threshold

    def reset(self):
        """重置状态（用于新会话）"""
        pass

    def shutdown(self):
        """释放资源"""
        self._executor.shutdown(wait=False)


class ASRPipeline:
    """
    ASR处理管线：识别 → 纠错 → 输出
    串联RealTimeASR与ASRCorrector
    """

    def __init__(self, model_size: str = "small", corrector=None):
        self.asr = RealTimeASR(model_size=model_size)
        self.corrector = corrector
        self._model_loaded = False

    @classmethod
    async def create(cls, model_size: str = "small", corrector=None) -> "ASRPipeline":
        """工厂方法：创建并加载模型"""
        pipeline = cls(model_size=model_size, corrector=corrector)
        pipeline.asr._load_model()
        pipeline._model_loaded = pipeline.asr.is_available
        return pipeline

    @property
    def is_available(self) -> bool:
        return self._model_loaded

    def transcribe(self, audio: np.ndarray) -> str:
        """
        识别音频并纠错

        Args:
            audio: float32 numpy array, 16kHz

        Returns:
            纠错后的文本
        """
        result = self.asr.transcribe_array(audio)
        if not result.success:
            return ""

        text = result.text
        if self.corrector:
            text = self.corrector.correct(text)
        return text.strip()

    async def transcribe_async(self, audio: np.ndarray) -> str:
        """异步版本（保留向后兼容，返回纯文本）"""
        text, _ = await self.transcribe_with_confidence(audio)
        return text

    async def transcribe_with_confidence(self, audio: np.ndarray) -> tuple[str, float]:
        """异步转写，返回 (文本, 置信度)，用于管线过滤"""
        result = await self.asr.transcribe_async(audio)
        if not result.success:
            return "", 0.0

        text = result.text
        if self.corrector:
            text = self.corrector.correct(text)
        return text.strip(), result.confidence

    def transcribe_file(self, audio_path: str) -> str:
        """离线文件转写"""
        result = self.asr.transcribe_file(audio_path)
        if not result.success:
            return ""

        text = result.text
        if self.corrector:
            text = self.corrector.correct(text)
        return text.strip()

    def shutdown(self):
        self.asr.shutdown()
