# 双工通话 ASR 转写方案调研与优化建议

> **日期**: 2026-05-14
> **状态**: 调研完成，待排期实施

---

## 当前设计

```
浏览器 → WebSocket → VAD(SimpleEnergyVAD, 能量检测) → 语音缓冲(固定15s) → 静音触发(0.3s) → ASR(faster-whisper small) → ASRCorrector → ChatBot
```

采用 **utterance-level batch** 方案：VAD 检测到声音后开始累积 buffer，检测到静音后将整段音频一次性送给 ASR 转写。

### 关键参数

| 参数 | 当前值 | 位置 |
|---|---|---|
| VAD 类型 | SimpleEnergyVAD (能量检测) | `voice_ws_handler.py:35` |
| VAD 阈值 | energy_threshold=0.0015 | `voice_ws_handler.py:35` |
| voice_frames / silence_frames | 1 / 3 | `voice_ws_handler.py:35` |
| ASR 模型 | faster-whisper `small` | `asr.py:31` |
| 量化 | int8 | `asr.py:31` |
| beam_size | 5 | `asr.py:31` |
| 静音超时 | 0.3s | `voice_ws_handler.py:53` |
| max_speech_duration | 15.0s | `voice_ws_handler.py:53` |
| 音频格式 | 16kHz float32 mono | `ws_adapters.py:20` |
| block_size | 2048 (128ms) | `voice_ws_handler.py:53` |

---

## 一、VAD：能量检测 vs 神经网络

### 现状

`SimpleEnergyVAD` 纯能量阈值判断（RMS > 0.0015），只有 `voice_frames=1`、`silence_frames=3` 两个计数器做滞回。

### 问题

- 对背景噪声、坐席语音回声几乎没有区分能力
- 无法处理交叠语音（agent 播报时用户同时说话）
- 软语音（轻声说话）和响噪声无法区分
- 实际通话环境（街道噪音、电视声、多人说话）误触发率高

### 推荐方案

替换为 **Silero VAD**（ONNX 推理的神经网络 VAD）：

```python
# 安装: pip install silero-vad onnxruntime
# 帧级语音概率输出 (0~1)，可做平滑阈值而非二元判断
```

**收益**：大幅降低误触发和漏检，尤其在真实通话噪声环境下。

---

## 二、端点检测：固定静音阈值 vs. 部分转写

### 现状

等 `silence_duration`（0.3s）静音 → 切段 → ASR。总延迟 = 静音等待（0.3s）+ ASR 推理（200-800ms）。

### 问题

- 用户说完一句话后，无论内容是否完整，都要等待固定静音窗口
- 说话中自然停顿超过 0.3s 会被错误切段（如 "Saya mau... bayar"）
- 实际上用户句末有语调下降等信号可以利用

### 推荐方案

1. **部分转写（partial ASR）提前端点判定**：

```
用户正在说话 → 每 500ms 跑一次 ASR 部分转写（beam_size=1）
  → 发现句末模式（。/?! / "saja" / "begitu"）→ 提前结束等待
  → 否则 → 继续累积
```

2. **动态静音阈值**：

- 短句子（< 2s 语音）→ 用较长阈值（1.0s），因为用户大概率还没说完
- 长句子（> 5s 语音）→ 用较短阈值（0.5s），因为大概率已说完

**收益**：感知延迟降低 30-50%，尤其对短句效果明显。

---

## 三、缓冲管理：固定 buffer vs 环形 buffer

### 现状

固定大小 `_speech_buffer`（15s × 16kHz = 240,000 采样），`_speech_pos` 指针递增。超过 15s 硬截断，剩余数据丢弃。

### 问题

- 15s 硬截断导致超长语音丢失末尾
- 语音起始被 VAD 裁剪（VAD 检测到 voice 之前的那段语音丢失）
- buffer 重置后丢失跨轮次上下文

### 推荐方案

1. **前置缓冲（pre-buffer）**：始终保留最近 500ms 音频，VAD 触发时把前置缓冲也包含进去。

2. **环形 buffer**：用 `RingBuffer` 替代 `_speech_pos` 指针：

```python
class RingBuffer:
    def __init__(self, capacity_samples: int):
        self._buf = np.zeros(capacity_samples, dtype=np.float32)
        self._head = 0          # 下一个写入位置
        self._total = 0         # 总写入量

    def write(self, chunk): ...
    def get_recent(self, n_samples) -> np.ndarray: ...
    def clear(self): ...
```

**收益**：消除边界 bug，语音起始 500ms 不被丢失。

---

## 四、ASR 推理优化

### 现状

faster-whisper `small` 模型，`int8` 量化，`beam_size=5`，ThreadPoolExecutor 异步执行。

### 可优化项

| 项目 | 现状 | 建议 | 预期影响 |
|---|---|---|---|
| beam_size | 5（全部） | partial 用 1，final 用 5 | 部分转写快 ~40% |
| 模型预热 | 无 | 首句 dummy 转写预热 CTranslate2 cache | 首句延迟 -100~300ms |
| VAD filter | vad_filter=True | 保留（已有外部 VAD 但无害） | 无变化 |
| 量化 | int8 | 可尝试 int8_float16（混合精度） | 部分硬件上有提升 |

**beam_size=1 vs 5**：印尼语用 greedy decoding（beam=1）时 WER 仅比 beam=5 差 1-2%，但速度快约 40%。对于部分转写可以用 beam=1 快速出结果，最终转写用 beam=5 保证质量。

---

## 五、打断时的 ASR 处理

### 现状

打断时 `flush(keep_recent_s=1.0)` 只保留最近 1 秒音频，打断后可能丢失用户实际说的话。

### 建议

- 保留打断确认窗口（300ms ducking）期间的所有音频，打断确认后立即送 ASR 转写
- `keep_recent_s` 参数从 1.0s 增加到 2-3s

---

## 六、优化优先级排序

| 优先级 | 优化项 | 改动量 | 预期收益 |
|---|---|---|---|
| **P0** | Silero VAD 替换能量检测 | 中 | 大幅降低误触发/漏检 |
| **P0** | 前置缓冲 500ms | 小 | 解决语音起始裁剪 |
| **P1** | 部分转写（每 500ms） | 中 | 感知延迟降低 30-50% |
| **P1** | beam_size=1 for partial | 极小 | 部分转写提速 40% |
| **P1** | 动态静音阈值 | 小 | 减少错误切段 |
| **P2** | 环形 buffer | 中 | 消除 15s 硬截断 |
| **P2** | 模型预热 | 极小 | 首句延迟降低 |
| **P2** | 打断缓冲增加至 2-3s | 极小 | 打断后不丢语音 |
| **P3** | 领域微调 Whisper | 大 | 印尼催收场景 WER 下降 |

---

## 七、关键文件索引

| 文件 | 作用 |
|---|---|
| `src/core/voice/asr.py` | RealTimeASR (faster-whisper 封装)、ASRPipeline (加 corrector) |
| `src/core/voice/vad.py` | SimpleEnergyVAD |
| `src/core/voice/pipeline.py` | DuplexCallPipeline 状态机，`_step_process()` 调用 ASR |
| `src/core/voice/ws_adapters.py` | WebSocketAudioSource (audio 缓冲), WebSocketAudioOutput |
| `src/api/voice_ws_handler.py` | WebSocket 处理器，初始化 VAD/ASR/TTS |
| `src/core/chatbot.py:298-371` | ASRCorrector（印尼催收领域规则修正） |
| `scripts/batch_asr_transcribe.py` | 离线批量转写脚本 |

---

## 八、总结

当前 ASR 方案的核心架构（faster-whisper + 状态机管线）是合理且成熟的。主要改进空间在：

1. **VAD 质量** — 能量检测 → Silero VAD（神经网络）— 对真实通话环境适应性的最大提升
2. **端点检测延迟** — 固定静音等待 → 部分转写辅助提前判定 — 感知延迟的直接改善
3. **缓冲管理** — 固定数组 → 环形 buffer + 前置缓冲 — 消除边界条件 bug

这三项改动不需要改变整体架构，可以在现有 pipeline 上渐进式实现。
