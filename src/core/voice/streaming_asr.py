#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""流式 ASR — 增长窗口 + 结果去重，模拟实时转写"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
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

    取消机制 (generation counter):
        submit() → generation += 1 → 创建 task(generation=N)
        mark_final() → generation += 1 → 旧 task 完成后发现 gen != generation → 丢弃
        关键保证: mark_final() 立即设置 _final_ready，不等待飞行中 task

    transcribe_lightweight (beam_size=1, temperature=[0.0]):
        速度优先于精度。已知问题：beam_size=1 对所有短音频可能均不达标
        (log prob < -1.0)，导致增长窗口完全无 partial 产出。若需提升
        partial 命中率，可考虑 beam_size=3。

    去重算法:
        单词级公共前缀匹配。例：
        "halo apa" → "halo apa kabar" → 增量: "kabar"
        若前缀完全不匹配（上一轮纯 hallucination），返回完整新文本
    """

    def __init__(self, asr, config: StreamingASRConfig = None):
        self._asr = asr                  # ASRPipeline 实例
        self._config = config or StreamingASRConfig()

        self._generation: int = 0        # 递增取消计数器
        self._in_flight_count: int = 0   # 并发 ASR 任务计数
        self._last_full_text: str = ""   # 最近完整转写结果
        self._is_final: bool = False     # mark_final() 已调用
        self._final_text: str = ""       # 最终结果缓存
        self._final_ready = asyncio.Event()
        self._active: bool = False       # submit() 已调用但未完成
        self._sample_rate: int = getattr(asr, 'sample_rate', 16000)
        self._lock = asyncio.Lock()

        # 增长窗口 ASR 使用独立 executor，避免与完整 ASR 竞争
        # max_workers=1 确保最多一个增长窗口任务运行
        self._executor = ThreadPoolExecutor(max_workers=1)

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
        """标记最终提交，忽略后续 submit()。

        立即返回已有结果 (_last_full_text)，不等待飞行中 ASR 任务。
        这是延迟优化的关键：增长窗口 ASR 耗时 1-2s，若不立即返回
        管线会产生额外等待。

        副作用: 飞行中任务继续运行到完成（Python 无法可靠取消线程内
        C 扩展调用），但结果被 generation 检查丢弃。
        """
        self._is_final = True
        if not self._active:
            self._final_ready.set()
            return
        # 递增 generation 使所有飞行中增长窗口任务失效
        # 它们仍会完成但结果被丢弃
        self._generation += 1
        self._final_text = self._last_full_text
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
        self._in_flight_count = 0
        self._last_full_text = ""
        self._is_final = False
        self._final_text = ""
        self._final_ready.clear()
        self._active = False

    def shutdown(self) -> None:
        """释放 executor 资源。"""
        self._executor.shutdown(wait=False)

    # ── 内部 ──────────────────────────────────────────────

    async def _run_transcribe(self, audio: np.ndarray, gen: int) -> None:
        """在独立 executor 中运行轻量化 ASR，完成后检查 generation 并去重。

        beam_size=1 + temperature=[0.0]（无回退）。
        实测: 对 faster_whisper small + 印尼语，beam_size=1 的 log prob
        几乎总是低于 -1.0 阈值，导致所有增长窗口均无产出。增长窗口实际
        收益趋近于零，仅消耗 CPU。若需恢复 partial 能力，升至 beam_size=3。

        注意: _asr 必须是 ASRPipeline（有 transcribe_lightweight），
        非裸 RealTimeASR。传入 RealTimeASR 会导致 fallback 到无 temperature
        参数的 transcribe_async，触发完整 6 级温度回退。
        """
        self._in_flight_count += 1
        try:
            loop = asyncio.get_event_loop()
            # 优先使用轻量化同步方法（beam_size=1, 无temperature回退）
            light_fn = getattr(self._asr, 'transcribe_lightweight', None)
            if light_fn is not None:
                full_text = await loop.run_in_executor(
                    self._executor, light_fn, audio)
            else:
                # 回退：FakeStreamingBackend 等测试后端
                full_text = await self._asr.transcribe_async(audio)
            if not isinstance(full_text, str):
                full_text = ""
        except Exception as e:
            logger.error(f"StreamingASR 转写失败: {e}")
            full_text = ""
        finally:
            self._in_flight_count -= 1

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
