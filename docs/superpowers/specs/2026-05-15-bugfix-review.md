# 双工通话管线 — 问题回顾与状态机审查

> 2026-05-15 整理，覆盖 5月11日–15日发现并修复的所有问题

---

## 一、已修复问题清单

### 1.1 回声循环（3次出现）

| 时间 | 症状 | 根因 |
|------|------|------|
| ~16:20 | ASR 持续返回 "Terima kasih" 而非用户说的 "pukul dua" | TTS 回声被 VAD 识别为语音 |
| 17:23:40 | ASR 返回 "Terima kasih kerana menonton" | 同上，且10.9s TTS的playback_done仅1.3s后到达 |
| 17:55:31 | ASR 识别完全错误的文本 | 同上，25秒内3次TTS（三个piper文件），典型回声循环 |

**修复方案（4项联动）：**

1. **冷却机制** (`cooldown_duration = 0.3s`): 播放/打断结束后，基于音频样本数丢弃0.3秒的音频，而非墙钟时间。避免steps以10ms间隔运行时墙钟冷却丢弃过多音频（30块 ≈ 3.8s）

2. **ASR/Agent 文本去重** (`voice_ws_handler.py: _run_pipeline`): 仅在新值与上次发送值不同时才推送到前端，消除死循环刷屏

3. **PROCESSING 入口缓存清除** (`pipeline.py: _step_process`): 每轮进入时清空 `_current_asr_text` / `_current_agent_text`，避免等待ASR就绪期间反复发送旧文本

4. **前端 barge-in 保护窗口** (`app.js`): 每次响应播放的前500ms禁用 barge-in 检测，防止问候语被环境回声截断

### 1.2 死循环刷屏

**症状:** 同一条 ASR/Agent 文本每 ~12ms 重复发送数百次

**根因:** `StepResult` 包含 `_current_asr_text` / `_current_agent_text`（当 `state_before == PROCESSING`），这些值不会被清除。`_step_process` 等待 ASR 就绪时每 10ms 返回一个 StepResult，每次都携带相同的旧文本。

**修复:** 两个层面 — `_step_process` 入口清空缓存值；`_run_pipeline` 去重。

### 1.3 问候语音频截断（"Halo"后半个音消失）

**症状:** 第一句话"Halo"的尾音被截断，随后恢复

**根因:** 前端 barge-in 检测（RMS > 0.02）过于敏感。问候语播放 ~100-200ms 后，第一个 ScriptProcessor 窗口检测到环境回声/噪声，触发 interrupt → 停止播放。

**修复:** 500ms barge-in 保护窗口（见 1.1 修复4）

### 1.4 TTS 数字格式朗读错误

**症状:** Piper TTS 将 "Rp 500,000" 读成印尼语小数点格式（逗号=小数点），将 "jt" 拼读为 "j-e-t-e"

**修复:** `_normalize_tts_text()` — 千位分隔逗号 → 点号，缩写展开（jt → juta, rb → ribu）

### 1.5 TTS 引擎优先级

**修复:** 优先级从 Edge-TTS（网络延迟高）改为 Piper-TTS（本地、亚实时）优先

### 1.6 ASR 线程池并发不足

**修复:** `ThreadPoolExecutor(max_workers=1)` → `max_workers=2`，允许流式 ASR 的新旧任务并发

### 1.7 调试消息跨会话残留

**症状:** 切换到不同会话后，前一会话的调试消息仍然显示

**修复:** `_debugMessages` Map 按 sessionId 存储，`viewSession` 时重新渲染

---

## 二、状态机完整图

```
                    ┌──────────────────────────────────────────┐
                    │                                          │
                    ▼                                          │
  IDLE ──start()──► LISTENING ──静音/VAD触发──► PROCESSING ──► RESPONDING
                     ▲  ▲                          │  ▲           │
                     │  │                          │  │           │
                     │  └─── 冷却后 ────┐           │  │           │
                     │                  │           │  │           │
                     │  INTERRUPTED ────┘           │  │           │
                     │    (打断过渡)                 │  │           │
                     │                              │  │           │
                     │  ASR超时回退                  │  │           │
                     │  (5s超时，丢语音)              │  │           │
                     │                              │  │           │
                     │                              ▼  │           ▼
                     └────────────────────────────────┘  │       CLOSING
                                                          │         │
                                                          │    stop()
                                                          │         ▼
                                                          │      CLOSED
                                                          │
                                                   Bot已结束
```

### 状态转移条件详表

| 从 | 到 | 条件 | 冷却 |
|----|----|------|------|
| IDLE | LISTENING | `start()` | - |
| LISTENING | PROCESSING | VAD检测语音 → 静音超时 (1s) 或 语音超长 (15s) | - |
| LISTENING | CLOSING | 长静音超时 (30s，未检测到语音) | - |
| PROCESSING | RESPONDING | ASR → Bot → TTS 完成，Bot 未结束 | - |
| PROCESSING | CLOSING | Bot 状态为 CLOSE/FAILED | - |
| PROCESSING | LISTENING | ASR 加载超时 (5s) | **无** |
| RESPONDING | LISTENING | 前端 playback_done | **0.3s** |
| RESPONDING | CLOSING | playback_done + Bot 已结束 | - |
| RESPONDING | INTERRUPTED | 前端 interrupt 或 barge-in | - |
| INTERRUPTED | LISTENING | 立即（一帧过渡） | **0.3s** |
| CLOSING | CLOSED | 播放结束语后 | - |
| LISTENING | CLOSED | `stop()` 被调用 | - |

---

## 三、待修复问题

### 3.1 [严重] `_step_closing` 不等待音频播放完成

**位置:** `pipeline.py:645-650`

```python
async def _step_closing(self):
    if self._current_agent_audio is not None and len(self._current_agent_audio) > 0:
        await self._output.speak(self._current_agent_audio)
    self._running = False          # ← 立即停止
    self._set_state(PipelineState.CLOSED)  # ← 立即关闭
```

**问题:** WebSocket 模式下 `speak()` 仅负责发送所有 chunk（快速返回），不等待实际播放。管线立即进入 CLOSED，handler 的 finally 块调用 `pipeline.stop()` 关闭 WebSocket。**用户可能听不到结束语**（"Terima kasih, selamat tinggal"）。

**建议修复:** 仿照 `_step_respond` 两阶段模式 — 发送结束后等待 `playback_done`，再进入 CLOSED。

### 3.2 [高] `_step_respond` 排空循环丢失用户语音

**位置:** `pipeline.py:610-617`

```python
if not self._frontend_playback_done:
    # 消费积压音频，防止队列溢出
    if hasattr(self._source, '_queue'):
        while not self._source._queue.empty():
            try:
                self._source._queue.get_nowait()
            except Exception:
                break
    return
```

**问题:** 等待 playback_done 期间，**每 10ms 排空整个音频队列**。如果用户在 Agent 播放期间开始说话（但未触发前端 barge-in，RMS < 0.02），其语音数据会被丢弃。后续 `flush(keep_recent_s=0.2)` 只能保留0.2秒，远不足以捕获完整语音。

**建议修复:** 改为限速消费 — 只丢弃超过一定队列深度（如 maxsize*0.8）的数据，保留最近的音频块。或者使用计时器限制排空频率。

### 3.3 [中] PROCESSING → LISTENING 回退路径缺少冷却

**位置:** `pipeline.py:495-498`

```python
self._debug(f"[PROCESS] ASR等待超时 → 回退LISTENING")
self._asr_wait_retries = 0
self._set_state(PipelineState.LISTENING)  # ← 无冷却
```

**问题:** ASR 模型加载超时（5s）时回退到 LISTENING，未设置 `_listen_cooldown_samples`。虽然概率极低（ASR 通常在1-2s内加载完成），但一旦触发，累积的语音缓冲区会被 `_reset_listen_state()` 清空，**已缓存的语音数据丢失**。

**建议修复:** 回退前保留语音数据，或设置冷却。

### 3.4 [中] `_asr_wait_retries` 未在 `__init__` 中初始化

**位置:** `pipeline.py:488`

```python
retries = getattr(self, '_asr_wait_retries', 0)
```

使用 `getattr` 兜底，功能正确但属代码异味。应在 `__init__` 中显式初始化为 0。

### 3.5 [低] 排空循环期间 `_recent_samples` 不反映实际音频

**位置:** `pipeline.py:610-617` + `ws_adapters.py`

`feed_chunk` 将 chunk 添加到 `_recent_samples`（入队时），但排空循环使用 `get_nowait()` 而非 `read_chunk()`，不更新 `_recent_samples`。虽然 deque 有 maxlen (0.3s) 限制不会无限增长，但 RMS 读数在排空期间不准确。

### 3.6 [低] StreamingASR `mark_final()` 竞态条件

**位置:** `streaming_asr.py:77-84`

```python
def mark_final(self):
    self._is_final = True
    if not self._in_flight and self._active:
        self._final_text = self._last_full_text
        self._final_ready.set()
    if not self._active:
        self._final_ready.set()  # 可能设置空文本
```

**问题:** 如果 `submit()` 从未被调用（`_active=False`），`mark_final()` 会立即设置 `_final_ready`，最终文本为空字符串。`submit()` 和 `mark_final()` 之间没有锁保护，可能在 `_active` 检查和 `_final_ready.set()` 之间被修改。

**建议:** 添加 `asyncio.Lock` 保护 `mark_final()` 的关键区段。

---

## 四、VAD 双门限机制

```
语音检测 = SileroVAD.speech_prob > 0.25  AND  RMS energy > 0.001
              (模型置信度)                    (声学能量)
```

| 场景 | speech_prob | energy | 结果 |
|------|-------------|--------|------|
| 正常说话 | 0.5-0.9 | 0.005-0.5 | 识别为语音 |
| 背景噪声 | 0.1-0.4 | 0.0001-0.0008 | 滤除 |
| TTS 回声 | 0.3-0.6 | 0.001-0.01 | 可能误识别 |

**注意:** 回声的 energy 范围与轻声说话重叠。仅靠双门限无法完全区分回声和真实语音，需要冷却机制配合。

---

## 五、前端调度时序

```
服务端: 快速发送所有 chunk (asyncio.sleep(0) 间隔)
   ↓
前端:   接收 chunk → 解码 PCM → AudioBuffer → source.start(startTime)
   │                                                    ↓
   │                            _nextPlayTime += buffer.duration (每个 chunk 0.1s)
   │                                                    ↓
   └── 所有 source.onended 触发 → _activeSources.length === 0
                                                      ↓
                                         发送 playback_done
```

**已知问题:** 前端在某些情况下提前发送 playback_done（5.1s 音频仅 0.5s 后，10.9s 音频仅 1.3s 后）。前端调度代码逻辑上看是正确的 — 可能原因：AudioContext 挂起/恢复导致时间跳跃，或浏览器回声消除干扰 `onended` 事件。

---

## 六、调试与诊断

### ASR 音频保存

每次 ASR 识别结果自动保存到 `data/runs/debug/asr_HHMMSS_文本前20字符.wav`，轮转保留最近 50 个文件。可用于回放任一次识别的输入音频。

### 关键日志标签

| 标签 | 含义 |
|------|------|
| `[LISTEN]` | VAD 逐帧状态（energy, speech_prob, voice_det） |
| `[VAD]` | VAD 状态变更（语音检测/静音超时） |
| `[VD-TRACE]` | `_voice_detected` 变更轨迹 |
| `[PROCESS]` | 进入处理阶段 |
| `[ASR]` | ASR 转写结果 |
| `[ASR-Partial]` | 流式 ASR 增量结果 |
| `[Bot]` | Chatbot 回复 |
| `[TTS]` | TTS 合成结果 |
| `[PLAY]` | 音频发送/播放状态 |
| `[FLUSH]` | 音频队列清理 |
| `[打断]` | 打断处理 |

---

## 七、测试状态

- 流式 ASR 测试: `tests/test_streaming_asr.py`
- 管线集成测试: `tests/test_duplex_integration.py` (38/38 通过)
- 管线单元测试: `tests/test_duplex_pipeline.py` (14/15, 1个已知时序不稳定)
- WebSocket 适配器测试: `tests/test_ws_adapters.py`
- WebSocket Handler 测试: `tests/test_voice_ws_handler.py`
