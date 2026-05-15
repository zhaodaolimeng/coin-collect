#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sherpa-onnx 流式 ASR 引擎 — 基于 streaming Zipformer 的真流式识别"""
import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).parent.parent.parent.parent / "data/models/sherpa-zipformer-id"
_DEFAULT_MODEL = _MODEL_DIR / "sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10"


@dataclass
class ASRResult:
    """ASR 识别结果"""
    text: str
    confidence: float = 1.0
    language: str = "id"
    duration: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


class SherpaRecognitionStream:
    """单个识别会话的流式状态。对应 sherpa-onnx OnlineStream 的生命周期。"""

    def __init__(self, recognizer, stream):
        self._rec = recognizer
        self._stream = stream
        self._finished = False

    def accept_waveform(self, samples: np.ndarray) -> None:
        """喂入音频 chunk (float32, 16kHz)。"""
        if self._finished:
            return
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        self._stream.accept_waveform(16000, samples)

    def decode(self) -> None:
        """解码所有就绪的帧。轻量操作，可同步调用。"""
        if self._finished:
            return
        while self._rec.is_ready(self._stream):
            self._rec.decode_stream(self._stream)

    def get_result(self) -> str:
        """获取当前累积的识别文本。"""
        return self._rec.get_result(self._stream)

    def is_endpoint(self) -> bool:
        """内置 endpoint 检测是否触发。"""
        return self._rec.is_endpoint(self._stream)

    def finish(self) -> None:
        """标记输入结束，解码剩余帧。"""
        if self._finished:
            return
        self._stream.input_finished()
        while self._rec.is_ready(self._stream):
            self._rec.decode_stream(self._stream)
        self._finished = True

    def reset(self) -> None:
        """重置 stream 状态（复用 stream 实例）。"""
        self._rec.reset(self._stream)
        self._finished = False


class SherpaASR:
    """sherpa-onnx 流式 ASR 引擎。

    基于 streaming Zipformer transducer 模型，支持真流式识别（非增长窗口），
    支持印尼语 (id) 及其他 8 种语言。

    Usage:
        asr = SherpaASR()
        stream = asr.create_stream()
        for chunk in audio_chunks:
            stream.accept_waveform(chunk)
            stream.decode()
            partial = stream.get_result()  # 实时增量文本
        stream.finish()
        final = stream.get_result()
    """

    def __init__(
        self,
        model_dir: str = None,
        num_threads: int = 4,
        enable_endpoint: bool = False,
        endpoint_rule1: float = 2.4,
        endpoint_rule2: float = 1.2,
        endpoint_rule3: float = 20.0,
    ):
        self._model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL
        self._recognizer = None
        self._num_threads = num_threads
        self._enable_endpoint = enable_endpoint
        self._endpoint_rule1 = endpoint_rule1
        self._endpoint_rule2 = endpoint_rule2
        self._endpoint_rule3 = endpoint_rule3
        self._load()

    def _load(self) -> None:
        try:
            import sherpa_onnx
        except ImportError:
            logger.error("sherpa-onnx 未安装: pip install sherpa-onnx")
            return

        tokens = str(self._model_dir / "tokens.txt")
        encoder = str(self._model_dir / "encoder-epoch-75-avg-11-chunk-16-left-128.int8.onnx")
        decoder = str(self._model_dir / "decoder-epoch-75-avg-11-chunk-16-left-128.onnx")
        joiner = str(self._model_dir / "joiner-epoch-75-avg-11-chunk-16-left-128.int8.onnx")
        bpe = str(self._model_dir / "bpe.model")

        missing = [f for f in [tokens, encoder, decoder, joiner, bpe] if not Path(f).exists()]
        if missing:
            logger.error(f"SherpaASR 模型文件缺失: {missing}")
            return

        try:
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=tokens,
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                modeling_unit="bpe",
                bpe_vocab=bpe,
                num_threads=self._num_threads,
                enable_endpoint_detection=self._enable_endpoint,
                rule1_min_trailing_silence=self._endpoint_rule1,
                rule2_min_trailing_silence=self._endpoint_rule2,
                rule3_min_utterance_length=self._endpoint_rule3,
            )
            logger.info(f"SherpaASR 已加载: {self._model_dir.name}")
        except Exception as e:
            logger.error(f"SherpaASR 加载失败: {e}")

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def is_available(self) -> bool:
        return self._recognizer is not None

    def create_stream(self) -> SherpaRecognitionStream:
        if not self._recognizer:
            raise RuntimeError("SherpaASR 模型未加载")
        return SherpaRecognitionStream(self._recognizer, self._recognizer.create_stream())

    # ── 离线转写 (兼容旧接口) ──────────────────────────────

    async def transcribe_async(self, audio: np.ndarray) -> str:
        """异步转写完整音频（离线模式，兼容 RealTimeASR 接口）。"""
        if not self._recognizer:
            return ""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        """同步转写完整音频。"""
        stream = self.create_stream()
        # 分块喂入（每 200ms 一块）
        chunk_size = 3200
        total = len(audio)
        pos = 0
        while pos < total:
            chunk = audio[pos:pos + chunk_size]
            pos += chunk_size
            if chunk.dtype != np.float32:
                chunk = chunk.astype(np.float32)
            stream.accept_waveform(chunk)
            stream.decode()
        stream.finish()
        return stream.get_result().strip()

    def shutdown(self) -> None:
        """释放资源。"""
        self._recognizer = None


class SherpaStreamingASR:
    """流式 ASR 适配器 — 与 StreamingASR 相同接口，使用 sherpa-onnx 原生流式。

    区别：
    - StreamingASR: 每次 submit 提交完整累积音频给 faster-whisper，去重提取增量
    - SherpaStreamingASR: 创建持久 sherpa stream，只喂入新样本，原生增量解码

    用法与 StreamingASR 完全一致：
        s = SherpaStreamingASR(asr)
        s.on_partial_result = lambda text: print(f"partial: {text}")
        s.submit(accumulated_audio)  # 每次提交累积音频（内部只喂新部分）
        s.mark_final()
        final = await s.wait_for_final()
    """

    def __init__(self, asr: SherpaASR):
        self._asr = asr
        self._stream: Optional[SherpaRecognitionStream] = None
        self._last_fed_samples: int = 0
        self._last_full_text: str = ""
        self._final_text: str = ""
        self._final_ready = asyncio.Event()
        self._active: bool = False
        self._is_final: bool = False
        self.on_partial_result = None

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def is_final_pending(self) -> bool:
        return self._is_final and not self._final_ready.is_set()

    @property
    def has_final_result(self) -> bool:
        return self._final_ready.is_set()

    @property
    def final_text(self) -> str:
        return self._final_text

    def submit(self, audio: np.ndarray) -> None:
        """提交完整累积音频快照。内部只喂入新样本到持久流中。"""
        if self._is_final:
            return
        self._active = True

        if self._stream is None:
            self._stream = self._asr.create_stream()
            self._last_fed_samples = 0

        new_audio = audio[self._last_fed_samples:]
        if len(new_audio) == 0:
            return

        self._stream.accept_waveform(new_audio)
        self._stream.decode()
        self._last_fed_samples = len(audio)

        cur_text = self._stream.get_result().strip()
        if cur_text and cur_text != self._last_full_text:
            incremental = cur_text[len(self._last_full_text):].strip()
            self._last_full_text = cur_text
            if incremental and self.on_partial_result:
                try:
                    self.on_partial_result(incremental)
                except Exception:
                    pass

        if self._is_final:
            self._final_text = cur_text or self._last_full_text
            self._final_ready.set()

    def mark_final(self) -> None:
        """标记最终提交。"""
        self._is_final = True
        if self._stream is not None:
            self._stream.finish()
            final = self._stream.get_result().strip()
            self._final_text = final or self._last_full_text
        else:
            self._final_text = self._last_full_text
        self._final_ready.set()

    async def wait_for_final(self, timeout: float = 5.0) -> str:
        try:
            await asyncio.wait_for(self._final_ready.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("SherpaStreamingASR: 等待最终结果超时")
        return self._final_text

    def reset(self) -> None:
        self._stream = None
        self._last_fed_samples = 0
        self._last_full_text = ""
        self._final_text = ""
        self._final_ready.clear()
        self._active = False
        self._is_final = False
