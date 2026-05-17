#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASR 置信度门 — 用 IndoBERT 伪困惑度过滤 ASR 幻觉

原理: 逐 token 掩码 → batch forward → 伪困惑度 (pseudo-perplexity)。
高困惑度（低 log prob）= 文本不自然 = 疑似幻觉。

能过滤的:
    - 多词非词/乱码 (如 "BISOK transfer" 等)
    - 严重偏离印尼语统计模式的输出

不能过滤的:
    - 与印尼语相似的马来西亚语文本 (如 "Terima kasih kerana menonton")
      这类文本在语言学上接近印尼语，语言模型无法有效区分
    - 单词级别错误 (如 "besok"→"bisok")，短文本上下文不足以支撑困惑度判断
      单词过滤受词频主导，"tidak" 的真实困惑度可能高于 "pisok"

已知局限 (单词级): 对 ≤2 个真实 token 的文本跳过困惑度检查，始终放行。
多词 hallucination 是主要威胁 (如 "Terima kasih kerana menonton")，
但无法通过困惑度过滤（文本本身符合印尼语统计规律）。

与 ChatGPT ASR 架构对齐: 专用 ASR (faster_whisper) → 置信度过滤 (IndoBERT) → LLM 理解。
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)

INDOBERT_MODEL = "cahya/bert-base-indonesian-1.5G"

# 短文本最少真实 token 数方可进行困惑度检查
# 单词级伪困惑度受词频主导，不可靠。只对 ≥3 token 的多词文本做检查。
MIN_REAL_TOKENS_FOR_CHECK = 3


@dataclass
class ASRQualityScore:
    """ASR 质量评分结果"""
    text: str
    pseudo_perplexity: float       # 伪困惑度，越低越好
    is_accepted: bool
    reason: str = ""               # 拒绝原因


class ASRConfidenceGate:
    """用 IndoBERT 对 ASR 输出做伪困惑度评分，过滤疑似幻觉。

    校准数据 (cahya/bert-base-indonesian-1.5G, MPS):
        正常多词印尼语:  PPL 9-1200   (如 "saya tidak tahu"=9, "saya mau bayar"=1064)
        正常单词:         PPL 7-189K  (词频主导, "iya"=7, "tidak"=189K)
        非词 hallucination: PPL 19K-14M ("pisok"=19K, "sok"=14M)

    用法:
        gate = ASRConfidenceGate(threshold=5000, device="mps")
        await gate.ensure_loaded()
        score = await gate.score("saya mau bayar")
        if score.is_accepted:
            ...

    加载失败时自动降级为直通模式 (is_available=False, score() 始终接受)。
    """

    def __init__(self, threshold: float = 5000, device: str = "mps"):
        self._threshold = threshold
        self._device = device if (device != "mps" or torch.backends.mps.is_available()) else "cpu"
        self._model = None
        self._tokenizer = None
        self._is_available = False
        self._load_error: Optional[str] = None

    @property
    def is_available(self) -> bool:
        return self._is_available

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    async def ensure_loaded(self) -> bool:
        """异步加载模型。失败返回 False，自动降级为直通模式。"""
        if self._is_available:
            return True
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._ensure_loaded_sync)

    def _ensure_loaded_sync(self) -> bool:
        """同步加载模型（在 executor 中运行）。"""
        if self._is_available:
            return True
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(INDOBERT_MODEL)
            self._model = AutoModelForMaskedLM.from_pretrained(INDOBERT_MODEL)
            self._model.to(self._device)
            self._model.eval()
            self._is_available = True
            logger.info(f"ASRConfidenceGate: IndoBERT 加载成功 (device={self._device}, "
                        f"threshold={self._threshold})")
        except Exception as e:
            self._load_error = str(e)
            logger.warning(f"ASRConfidenceGate: IndoBERT 加载失败，降级为直通模式: {e}")
            self._is_available = False
        return self._is_available

    async def score(self, text: str) -> ASRQualityScore:
        """计算伪困惑度并判断是否接受。降级模式下始终接受。"""
        if not text or not text.strip():
            return ASRQualityScore(
                text=text,
                pseudo_perplexity=float("inf"),
                is_accepted=False,
                reason="空文本",
            )

        if not self._is_available:
            return ASRQualityScore(
                text=text,
                pseudo_perplexity=0.0,
                is_accepted=True,
                reason="直通（模型未加载）",
            )

        ppl = await self._compute_ppl(text)
        is_accepted = ppl <= self._threshold
        reason = "" if is_accepted else f"困惑度 {ppl:.1f} > 阈值 {self._threshold}"
        return ASRQualityScore(
            text=text,
            pseudo_perplexity=ppl,
            is_accepted=is_accepted,
            reason=reason,
        )

    async def should_accept(self, text: str) -> bool:
        """便捷方法：是否接受此 ASR 结果"""
        result = await self.score(text)
        return result.is_accepted

    async def _compute_ppl(self, text: str) -> float:
        """逐 token 掩码 → batch forward → 伪困惑度。

        对每个真实 token（排除 [CLS]/[SEP]）分别掩码，一次 batch forward
        计算所有位置的 log prob。单次 forward (batch_size ≤ 32)，MPS 上 ~20-50ms。

        单词或双词文本 (真实 token ≤ 2) 跳过困惑度计算，返回 0（始终放行）。
        这是有意为之：短文本困惑度受词频主导，无法可靠区分正误。
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._compute_ppl_sync, text)

    def _compute_ppl_sync(self, text: str) -> float:
        """同步计算伪困惑度（在 executor 中运行）"""
        try:
            encoded = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )
            input_ids = encoded["input_ids"][0]  # (seq_len,)

            mask_token_id = self._tokenizer.mask_token_id
            real_positions = [
                i for i, tid in enumerate(input_ids.tolist())
                if tid not in (self._tokenizer.cls_token_id, self._tokenizer.sep_token_id)
            ]

            if not real_positions:
                return float("inf")

            # 短文本跳过困惑度检查
            if len(real_positions) < MIN_REAL_TOKENS_FOR_CHECK:
                return 0.0

            # 为每个真实 token 构造掩码版本，batch 一次 forward
            batch_input_ids = input_ids.unsqueeze(0).repeat(len(real_positions), 1)
            target_token_ids = []
            for batch_idx, pos in enumerate(real_positions):
                target_token_ids.append(input_ids[pos].item())
                batch_input_ids[batch_idx, pos] = mask_token_id

            attention_mask = encoded["attention_mask"].repeat(len(real_positions), 1)
            batch = {
                "input_ids": batch_input_ids.to(self._device),
                "attention_mask": attention_mask.to(self._device),
            }

            with torch.no_grad():
                output = self._model(**batch)
                logits = output.logits  # (batch, seq_len, vocab_size)

            log_probs = []
            for batch_idx, (pos, true_id) in enumerate(zip(real_positions, target_token_ids)):
                pos_logits = logits[batch_idx, pos]  # (vocab_size,)
                pos_log_prob = torch.log_softmax(pos_logits, dim=-1)
                log_probs.append(pos_log_prob[true_id].item())

            if not log_probs:
                return float("inf")

            mean_log_prob = sum(log_probs) / len(log_probs)
            return float(torch.exp(torch.tensor(-mean_log_prob)).item())

        except Exception as e:
            logger.error(f"ASRConfidenceGate: 困惑度计算失败: {e}")
            return float("inf")

    def shutdown(self):
        """释放模型资源"""
        if self._model is not None:
            del self._model
            self._model = None
        self._is_available = False
