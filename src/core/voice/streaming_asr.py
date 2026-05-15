#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""流式 ASR — 增长窗口 + 结果去重，模拟实时转写"""
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StreamingASRConfig:
    """流式 ASR 配置"""
    min_audio_duration: float = 0.5   # 首次 ASR 提交前最少音频 (秒)
    throttle_interval: float = 0.5    # 连续 ASR 提交最小间隔 (秒)
    min_new_audio: float = 0.3        # 提交所需的最少新音频量 (秒)


class StreamingASR:
    """增长窗口流式 ASR 封装。

    在用户说话期间周期性提交增长中的音频快照给 faster_whisper，
    通过单词级公共前缀匹配去重提取增量文本。

    使用 generation 计数器取消旧结果：每次 submit() 递增，
    ASR 任务完成后检查 generation，不匹配则丢弃。
    """

    def __init__(self, asr, config: StreamingASRConfig = None):
        self._asr = asr                  # RealTimeASR 实例
        self._config = config or StreamingASRConfig()

        self._generation: int = 0        # 递增取消计数器
        self._in_flight: bool = False    # ASR 任务运行中
        self._last_full_text: str = ""   # 最近完整转写结果
        self._is_final: bool = False     # mark_final() 已调用
        self._final_text: str = ""       # 最终结果缓存
        self._final_ready = asyncio.Event()
        self._active: bool = False       # submit() 已调用但未完成
        self._sample_rate: int = getattr(asr, 'sample_rate', 16000)
        self._lock = asyncio.Lock()

        self.on_partial_result: Optional[Callable[[str], None]] = None

    # ── 属性 ──────────────────────────────────────────────

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

    # ── 公开 API ──────────────────────────────────────────

    def submit(self, audio: np.ndarray) -> None:
        """提交完整累积音频快照。不阻塞，旧飞行中结果自动弃用。"""
        if self._is_final:
            return
        self._active = True
        self._generation += 1
        gen = self._generation
        asyncio.create_task(self._run_transcribe(audio.copy(), gen))

    def mark_final(self) -> None:
        """标记最终提交，忽略后续 submit()。"""
        self._is_final = True
        if not self._in_flight and self._active:
            self._final_text = self._last_full_text
            self._final_ready.set()
        if not self._active:
            self._final_ready.set()

    async def wait_for_final(self, timeout: float = 5.0) -> str:
        """等待最终 ASR 结果。"""
        try:
            await asyncio.wait_for(self._final_ready.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("StreamingASR: 等待最终结果超时")
        return self._final_text

    def reset(self) -> None:
        """重置状态，准备新一轮对话。"""
        self._generation = 0
        self._in_flight = False
        self._last_full_text = ""
        self._is_final = False
        self._final_text = ""
        self._final_ready.clear()
        self._active = False

    # ── 内部 ──────────────────────────────────────────────

    async def _run_transcribe(self, audio: np.ndarray, gen: int) -> None:
        """在 executor 中运行 ASR，完成后检查 generation 并去重。"""
        self._in_flight = True
        try:
            full_text = await self._asr.transcribe_async(audio)
            if not isinstance(full_text, str):
                full_text = ""
        except Exception as e:
            logger.error(f"StreamingASR 转写失败: {e}")
            full_text = ""
        finally:
            self._in_flight = False

        async with self._lock:
            if gen != self._generation:
                return  # 陈旧结果，丢弃

            full_text = full_text.strip()
            incremental = self._dedup(full_text)
            self._last_full_text = full_text

            if incremental and self.on_partial_result:
                try:
                    self.on_partial_result(incremental)
                except Exception:
                    pass

            if self._is_final:
                self._final_text = full_text
                self._final_ready.set()

    def _dedup(self, full_text: str) -> str:
        """从完整文本中提取增量部分（单词级公共前缀匹配）。"""
        if not full_text:
            return ""
        if not self._last_full_text:
            return full_text

        prev_words = self._last_full_text.split()
        curr_words = full_text.split()

        match_count = 0
        for pw, cw in zip(prev_words, curr_words):
            if pw == cw:
                match_count += 1
            else:
                break

        if match_count == 0:
            return full_text
        return " ".join(curr_words[match_count:])

    @staticmethod
    def _common_word_prefix_len(a: str, b: str) -> int:
        """返回 a 和 b 的单词级公共前缀字符长度。"""
        a_words = a.split()
        b_words = b.split()
        match_count = 0
        for aw, bw in zip(a_words, b_words):
            if aw == bw:
                match_count += 1
            else:
                break
        if match_count == 0:
            return 0
        matched = " ".join(a_words[:match_count])
        if match_count < len(b_words):
            matched += " "
        return len(matched)
