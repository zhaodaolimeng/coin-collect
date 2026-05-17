"""ASR 置信度门测试 — IndoBERT 伪困惑度过滤"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.voice.asr_confidence import (
    ASRConfidenceGate,
    ASRQualityScore,
    INDOBERT_MODEL,
    MIN_REAL_TOKENS_FOR_CHECK,
)


# ═══════════════════════════════════════════════════════════════════
# ASRQualityScore dataclass
# ═══════════════════════════════════════════════════════════════════

class TestASRQualityScore:
    def test_defaults(self):
        s = ASRQualityScore(text="test", pseudo_perplexity=1.0, is_accepted=True)
        assert s.text == "test"
        assert s.pseudo_perplexity == 1.0
        assert s.is_accepted is True
        assert s.reason == ""

    def test_reason(self):
        s = ASRQualityScore(text="x", pseudo_perplexity=10.0, is_accepted=False, reason="too high")
        assert s.reason == "too high"


# ═══════════════════════════════════════════════════════════════════
# ASRConfidenceGate 初始化和配置
# ═══════════════════════════════════════════════════════════════════

class TestGateInit:
    def test_default_threshold(self):
        gate = ASRConfidenceGate()
        assert gate._threshold == 5000
        assert gate._is_available is False
        assert gate._model is None

    def test_custom_threshold(self):
        gate = ASRConfidenceGate(threshold=100.0)
        assert gate._threshold == 100.0

    def test_mps_fallback_to_cpu(self):
        with patch.object(torch.backends.mps, 'is_available', return_value=False):
            gate = ASRConfidenceGate(device="mps")
            assert gate._device == "cpu"


# ═══════════════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════════════

class TestGateLoading:
    def test_ensure_loaded_success(self):
        """模拟成功加载模型"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.cls_token_id = 3
        mock_tokenizer.sep_token_id = 2
        mock_tokenizer.mask_token_id = 4

        with patch(
            "transformers.AutoModelForMaskedLM.from_pretrained",
            return_value=mock_model,
        ), patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ):
            gate = ASRConfidenceGate()
            ok = gate._ensure_loaded_sync()
            assert ok is True
            assert gate.is_available is True
            assert gate._model is mock_model
            assert gate._tokenizer is mock_tokenizer

    def test_ensure_loaded_failure(self):
        """加载失败时降级"""
        with patch(
            "transformers.AutoModelForMaskedLM.from_pretrained",
            side_effect=OSError("No space left on device"),
        ):
            gate = ASRConfidenceGate()
            ok = gate._ensure_loaded_sync()
            assert ok is False
            assert gate.is_available is False
            assert gate.load_error is not None
            assert "No space left" in gate.load_error

    def test_ensure_loaded_idempotent(self):
        """重复加载不会重新创建模型"""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.cls_token_id = 3
        mock_tokenizer.sep_token_id = 2

        with patch(
            "transformers.AutoModelForMaskedLM.from_pretrained",
            return_value=mock_model,
        ), patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ):
            gate = ASRConfidenceGate()
            gate._ensure_loaded_sync()
            # 第二次调用应直接返回 True
            ok = gate._ensure_loaded_sync()
            assert ok is True


# ═══════════════════════════════════════════════════════════════════
# Score 方法 (需要 mock 模型)
# ═══════════════════════════════════════════════════════════════════

def _make_mock_gate(threshold=5000, log_probs=None):
    """构造一个已加载 mock 模型的 gate，用于测试 score 方法"""
    gate = ASRConfidenceGate(threshold=threshold)
    gate._is_available = True

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.cls_token_id = 3
    mock_tokenizer.sep_token_id = 2
    mock_tokenizer.mask_token_id = 4

    gate._model = mock_model
    gate._tokenizer = mock_tokenizer

    # 默认 tokenizer 行为：3 个真实 token
    token_ids = torch.tensor([3, 100, 200, 300, 2])  # CLS, w1, w2, w3, SEP
    mock_tokenizer.return_value = {
        "input_ids": token_ids.unsqueeze(0),
        "attention_mask": torch.ones(1, 5),
    }

    return gate, mock_model, mock_tokenizer, token_ids


class TestGateScore:
    """测试 score() 方法，使用 mock 模型"""

    def test_empty_text_rejected(self):
        gate = ASRConfidenceGate()
        result = asyncio.run(gate.score(""))
        assert result.is_accepted is False
        assert result.pseudo_perplexity == float("inf")
        assert "空文本" in result.reason

    def test_whitespace_only_rejected(self):
        gate = ASRConfidenceGate()
        result = asyncio.run(gate.score("   "))
        assert result.is_accepted is False

    def test_degraded_mode_always_accepts(self):
        """模型未加载时降级为直通模式"""
        gate = ASRConfidenceGate()
        result = asyncio.run(gate.score("anything"))
        assert result.is_accepted is True
        assert result.pseudo_perplexity == 0.0
        assert "直通" in result.reason

    def test_should_accept_convenience(self):
        gate = ASRConfidenceGate()
        result = asyncio.run(gate.should_accept("test"))
        assert result is True  # 降级模式

    def test_short_text_bypassed(self):
        """≤2 个真实 token 的短文本跳过检查"""
        gate, mock_model, mock_tokenizer, _ = _make_mock_gate()

        # 只有 1 个真实 token
        token_ids = torch.tensor([3, 100, 2])  # CLS, word, SEP
        mock_tokenizer.return_value = {
            "input_ids": token_ids.unsqueeze(0),
            "attention_mask": torch.ones(1, 3),
        }

        result = asyncio.run(gate.score("iya"))
        assert result.is_accepted is True
        assert result.pseudo_perplexity == 0.0
        # 确认模型没有被调用
        mock_model.assert_not_called()

    def test_multi_word_accepted_low_ppl(self):
        """多词正常文本 → 低困惑度 → 接受"""
        gate, mock_model, mock_tokenizer, token_ids = _make_mock_gate()

        # 模拟低困惑度 (高 log prob)
        logits = torch.zeros(3, 5, 32000)
        logits[:, :, :] = -100.0  # 其他 token 低概率
        for i in range(3):
            logits[i, i + 1, token_ids[i + 1]] = 10.0  # 正确 token 高概率

        mock_output = MagicMock()
        mock_output.logits = logits
        mock_model.return_value = mock_output

        result = asyncio.run(gate.score("saya mau bayar"))
        assert result.is_accepted is True
        assert result.pseudo_perplexity < 5000

    def test_multi_word_rejected_high_ppl(self):
        """多词幻觉 → 高困惑度 → 拒绝"""
        gate, mock_model, mock_tokenizer, token_ids = _make_mock_gate(threshold=100.0)

        # 模拟高困惑度 (低 log prob)
        logits = torch.zeros(3, 5, 32000)
        logits[:, :, :] = 10.0  # 均匀分布
        for i in range(3):
            logits[i, i + 1, token_ids[i + 1]] = 0.1

        mock_output = MagicMock()
        mock_output.logits = logits
        mock_model.return_value = mock_output

        result = asyncio.run(gate.score("BISOK transfer"))
        assert result.is_accepted is False
        assert "困惑度" in result.reason

    def test_zero_real_tokens_returns_inf(self):
        """只有特殊 token 时返回 inf"""
        gate, mock_model, mock_tokenizer, _ = _make_mock_gate()

        # 文本 tokenization 后无真实 token
        token_ids = torch.tensor([3, 2])
        mock_tokenizer.return_value = {
            "input_ids": token_ids.unsqueeze(0),
            "attention_mask": torch.ones(1, 2),
        }

        result = asyncio.run(gate.score(""))
        assert result.pseudo_perplexity == float("inf")


# ═══════════════════════════════════════════════════════════════════
# 阈值行为
# ═══════════════════════════════════════════════════════════════════

class TestThresholdBehavior:
    def test_threshold_below_rejects(self):
        """阈值设得非常低 → 正常文本也被拒绝"""
        gate, mock_model, mock_tokenizer, token_ids = _make_mock_gate(threshold=0.1)

        logits = torch.zeros(3, 5, 32000)
        logits[:, :, :] = -100.0
        for i in range(3):
            logits[i, i + 1, token_ids[i + 1]] = 5.0

        mock_output = MagicMock()
        mock_output.logits = logits
        mock_model.return_value = mock_output

        result = asyncio.run(gate.score("saya mau bayar"))
        assert result.is_accepted is False

    def test_threshold_high_accepts(self):
        """阈值设得很高 → 几乎所有文本都被接受"""
        gate, mock_model, mock_tokenizer, token_ids = _make_mock_gate(threshold=1e9)

        logits = torch.zeros(3, 5, 32000)
        logits[:, :, :] = 0.0

        mock_output = MagicMock()
        mock_output.logits = logits
        mock_model.return_value = mock_output

        result = asyncio.run(gate.score("saya mau bayar"))
        assert result.is_accepted is True


# ═══════════════════════════════════════════════════════════════════
# 资源清理
# ═══════════════════════════════════════════════════════════════════

class TestShutdown:
    def test_shutdown_clears_model(self):
        gate = ASRConfidenceGate()
        gate._model = MagicMock()
        gate._is_available = True
        gate.shutdown()
        assert gate._model is None
        assert gate._is_available is False

    def test_shutdown_idempotent(self):
        gate = ASRConfidenceGate()
        gate.shutdown()
        gate.shutdown()
        assert gate._model is None


# ═══════════════════════════════════════════════════════════════════
# 集成测试 — 需要真实模型
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestGateIntegration:
    """需要 cahya/bert-base-indonesian-1.5G 模型已下载"""

    @pytest.fixture(scope="class")
    def gate(self):
        g = ASRConfidenceGate(threshold=5000, device="cpu")
        ok = g._ensure_loaded_sync()
        if not ok:
            pytest.skip("IndoBERT 模型不可用")
        yield g
        g.shutdown()

    def test_normal_multi_word_accepted(self, gate):
        """正常多词印尼语应被接受"""
        texts = [
            "saya mau bayar",
            "besok saya transfer",
            "halo apa kabar",
            "saya tidak tahu",
            "nanti saya hubungi lagi",
        ]
        for text in texts:
            result = asyncio.run(gate.score(text))
            assert result.is_accepted, f"'{text}' should be accepted, got PPL={result.pseudo_perplexity:.1f}"

    def test_single_word_bypassed(self, gate):
        """短文本始终放行"""
        texts = ["besok", "iya", "tidak", "baik", "saya", "BISOK", "pisok", "sok"]
        for text in texts:
            result = asyncio.run(gate.score(text))
            assert result.is_accepted, f"'{text}' should be accepted (short text bypass)"

    def test_empty_rejected(self, gate):
        result = asyncio.run(gate.score(""))
        assert result.is_accepted is False

    def test_perplexity_ranking(self, gate):
        """正常文本困惑度应低于非词"""
        normal = asyncio.run(gate.score("saya tidak tahu"))
        weird = asyncio.run(gate.score("sok sok sok sok"))
        # "sok sok sok sok" 应该有更高的困惑度
        # (但如果被跳过检查则 PPL=0)
        if weird.pseudo_perplexity > 0 and normal.pseudo_perplexity > 0:
            assert weird.pseudo_perplexity > normal.pseudo_perplexity, \
                f"Expected weird PPL ({weird.pseudo_perplexity:.1f}) > normal PPL ({normal.pseudo_perplexity:.1f})"
