# Demo 演示重构：双工通话管线

> **Goal:** 重构语音 demo，使演示尽可能贴近真实双工电话通话场景（打断、重叠语音），同时消除现有两套管线代码的重复、完善测试覆盖。

> **Stakeholder:** 业务方（体验逼真度优先）

---

## 1. 现有问题诊断

### 1.1 架构重复

```
VoiceConversation (conversation.py)          CustomerVoiceSimulator (customer_simulator.py)
─────────────────────────────────            ──────────────────────────────────────────
麦克风 → VAD → ASR → Bot → TTS → 扬声器      文本模拟器 → TTS → 文件 → VAD → ASR → Bot → 报告
         ↑ 人声输入                                           ↑ 自动仿真
```

两条路径各自实现了一遍 VAD→ASR→Bot→TTS 流程，但互不共享。`InterruptionHandler` (interruption.py) 独立存在但未集成。

### 1.2 半双工限制

`VoiceConversation.run_once()`: 听 → 处理 → 说 → 听 → ...（严格串行）
`_play_with_interrupt()`: 仅播放间隙用 RMS 检测打断，不持续监听

真实通话是**全双工**：催收员说话时客户可随时插话，催收员听到打断后应停止说话并响应。

### 1.3 测试薄弱

`voice_simulation_test.py`: MockASRClient 返回硬编码字符串，MockTTSClient 不生成音频，不测真实管线。

---

## 2. 目标架构

```
                    ┌─────────────────────────────────┐
                    │     DuplexCallPipeline           │
                    │                                  │
  AudioSource       │  ┌──────────┐    ┌──────────┐   │   AudioOutput
  (input)           │  │ Listen   │───→│ Process  │   │   (output)
  ┌──────────┐     │  │ (VAD)    │    │ (ASR→Bot)│   │   ┌──────────────┐
  │ Mic      │────→│  └────┬─────┘    └────┬─────┘   │──→│ TTS → Speak  │
  │ Sim      │     │       │               │         │   │ + Barge-in   │
  └──────────┘     │       │  ┌────────────┘         │   │ + Ducking    │
                    │       │  │                      │   └──────────────┘
                    │       ▼  ▼                      │
                    │  ┌──────────────┐               │
                    │  │ Interrupt    │←──────────────│── barge-in event
                    │  │ Handler      │               │
                    │  └──────────────┘               │
                    └─────────────────────────────────┘
```

**核心原则：一套管线，两种输入源，双工并发。**

---

## 3. 组件设计

### 3.1 AudioSource（输入源抽象）

```python
class AudioSource(ABC):
    """音频输入源抽象。所有输入源统一接口，管线不感知来源。"""

    @abstractmethod
    async def start(self): ...

    @abstractmethod
    async def stop(self): ...

    @abstractmethod
    async def read_chunk(self) -> np.ndarray | None:
        """读取下一个音频块 (block_size samples)"""

    @abstractmethod
    def current_rms(self) -> float:
        """当前缓冲区的 RMS 能量值，用于打断检测"""

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    def is_real_time(self) -> bool:
        """是否实时输入 (vs 文件/模拟可加速播放)"""
        return True
```

**三个实现：**

| 类 | 用途 | 数据来源 |
|---|---|---|
| `MicrophoneSource` | 真人说话（演示模式） | sounddevice InputStream → RingBuffer |
| `SimulatedSource` | 自动仿真 | 文本模拟器 → TTS → 音频数组分批发送 |
| `FileSource` | 回放录音 | 音频文件分块读取 |

`MicrophoneSource` 实质上是现有 `AudioInput` 的瘦封装。`SimulatedSource` 核心变化：不是先生成完整音频再注入，而是**逐块流式注入**到 RingBuffer，模拟真实麦克风的实时数据到达节奏。

### 3.2 DuplexAudioOutput（双工播放输出）

```python
class DuplexAudioOutput:
    """双工音频播放。后台播放 Agent TTS，持续监听打断。"""

    def __init__(self, source: AudioSource, barge_in_threshold: float, ...):
        self._source = source         # 用于打断检测 (读取 mic RMS)
        self._speak_task: asyncio.Task | None = None
        self._ducking = False
        self._interrupted = False

    async def speak(self, audio: np.ndarray) -> PlaybackResult:
        """
        播放音频，返回 PLAYBACK_COMPLETED / INTERRUPTED / FAILED。

        内部启动 asyncio.Task 分块播放，每块播放后检查 source.current_rms()，
        超过阈值则 duck → 短暂确认 → 停止或恢复。
        """

    async def wait_done(self):
        """等待当前播放任务结束"""

    def stop(self):
        """立即停止播放"""
```

**打断处理逻辑：**
1. 分块播放（每块 ~100ms），每块之间检查 RMS
2. RMS > `barge_in_threshold`: 音量降至 20%（ducking），等待 300ms 二次确认
3. 300ms 后仍超阈值 → 确认打断，停止播放，返回 `INTERRUPTED`
4. 300ms 内回落 → 误检（咳嗽、背景音），恢复正常音量继续播放

### 3.3 InterruptionContext（打断上下文）

```python
@dataclass
class InterruptionContext:
    """打断时的上下文，传递给 Bot 用于策略调整"""
    agent_text_interrupted: str     # 被打断时正在说的内容
    agent_playback_position: float  # 播放进度 (0.0–1.0)
    customer_rms_peak: float        # 打断时的音量峰值
    partial_asr: str | None = None  # 打断后立即做的快速 ASR（可选）
```

打断上下文让 Bot 知道"我刚才说到一半被打断了"，可以选择重复、缩短回复、或直接响应客户的插话。

### 3.4 DuplexCallPipeline（核心管线）

```python
class PipelineState(Enum):
    IDLE = auto()
    LISTENING = auto()       # 正在听用户说话
    PROCESSING = auto()      # ASR + Bot 处理中
    RESPONDING = auto()      # Agent TTS 播放中（可与 LISTENING 重叠）
    INTERRUPTED = auto()     # 被打断，短暂过渡
    CLOSING = auto()         # 结束流程
    CLOSED = auto()

class DuplexCallPipeline:
    """
    双工通话管线。

    使用示例:
        source = MicrophoneSource()
        output = DuplexAudioOutput(source)
        pipeline = DuplexCallPipeline(chatbot, source, output, asr, tts)

        # 手动逐步控制（演示用）
        await pipeline.start()
        while pipeline.state != PipelineState.CLOSED:
            await pipeline.step()

        # 或自动运行
        await pipeline.run_until_closed()
    """

    def __init__(self, chatbot, source, output, asr_pipeline, tts_manager, vad, *, config: PipelineConfig):
        ...

    async def start(self): ...
    async def step(self) -> StepResult | None: ...
    async def run_until_closed(self): ...
    async def stop(self): ...
```

**step() 核心循环：**

```
                    ┌─────────────────────────────────────┐
                    │               step()                 │
                    │                                      │
  IDLE ──→ LISTENING ──→ PROCESSING ──→ RESPONDING ──→ LISTENING
                    │       │              │      │        │
                    │       │              │      ├─ INTERRUPTED → LISTENING
                    │       │              │      └─ CALL_CLOSING → CLOSED
                    │       │              │
                    │       │              └── (启动播放 task，同时继续监听)
                    │       │
                    │       └── ASR transcribe → Bot process → TTS synthesize
                    │
                    └── VAD 检测语音活动 → 累积语音段 → 静音超时触发结束
```

**关键：RESPONDING 期间不停止监听。** Agent 播放启动后，下一个 step() 同时做两件事：继续喂音频块给播放器、检查 VAD 是否有新语音。检测到新语音且确认打断，立即停止播放并切到 LISTENING 状态。

### 3.5 CallSimulator（自动仿真，替代 CustomerVoiceSimulator）

```python
class CallSimulator:
    """
    自动仿真模式：用文本模拟器生成客户回复 → TTS → SimulatedSource → Pipeline。

    与 CustomerVoiceSimulator 的区别：
    - 不是自己实现管线，而是向 DuplexCallPipeline 注入 SimulatedSource
    - 不再有独立的 _inject_and_vad_gate / _run_single_turn
    - 报告生成仍在外部
    """

    @classmethod
    async def create(cls, chatbot, persona, ...) -> "CallSimulator":
        """工厂：创建 SimulatedSource + DuplexCallPipeline"""

    async def run(self, max_turns: int = 20) -> SimulationReport:
        """
        运行自动仿真。

        每轮：生成客户文本 → TTS → 馈入 SimulatedSource → pipeline.step() → agent 回复
        """

    async def run_streaming(self, max_turns: int = 20):
        """流式 yield，用于 Web 前端 SSE"""
```

`SimulationReport` 和 `SimulationTurn` 保留现有结构（customer_simulator.py 中的 dataclass），向后兼容。

---

## 4. 双工并发模型

```
时间线 →

真人模式:
  Mic:   ▁▁▁▁████████▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁██████▁▁▁▁▁▁▁▁
  Agent: ▁▁▁▁▁▁▁▁▁▁▁▁▁████████████▁▁▁▁▁▁▁▁▁▁▁██████
               ↑ 用户说话        ↑ 打断        ↑ 用户继续
               (agent 静默)      (agent 中断)   (agent 响应)

自动模式:
  Sim:   ▁▁▁▁████████▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁██████▁▁▁▁▁▁▁▁
  Agent: ▁▁▁▁▁▁▁▁▁▁▁▁▁████████████▁▁▁▁▁▁▁▁▁▁▁██████
               ↑ TTS 注入音频   ↑ 模拟打断    ↑
               (SimulatedSource)
```

两个模式的时间线完全一致，区别仅在于音频来源。

---

## 5. 状态机细节

```
                        start()
                          │
                          ▼
    ┌────────────────── IDLE ──────────────────┐
    │                     │                     │
    │              step() │                     │ stop()
    │                     ▼                     │
    │    ┌────────── LISTENING ◄───────────┐    │
    │    │           │         │            │    │
    │    │  语音结束  │         │ 打断确认   │    │
    │    │           ▼         │            │    │
    │    │      PROCESSING     │            │    │
    │    │  (ASR → Bot → TTS)  │            │    │
    │    │           │         │            │    │
    │    │           ▼         │            │    │
    │    │      RESPONDING ────┼── INTERRUPTED  │
    │    │  (播放+监听)        │            │    │
    │    │      │    │         │            │    │
    │    │ 播放完成│ 打断检测───┘            │    │
    │    │      │                           │    │
    │    │  (Bot 说 CLOSE/FAILED)           │    │
    │    │      │                           │    │
    │    │      ▼                           │    │
    │    │   CLOSING                        │    │
    │    │      │                           │    │
    │    └──────┴───────────────────────────┘    │
    │                     │                      │
    │                     ▼                      ▼
    └────────────────── CLOSED ──────────────────┘
```

`RESPONDING → INTERRUPTED` 是双工的关键转换：Agent 正在播放时检测到打断 → 短暂过渡状态 → 立即切回 LISTENING。

---

## 6. 错误处理

| 场景 | 处理 |
|---|---|
| ASR 模型未加载 | 回退到文字输入模式，提示"语音识别不可用" |
| TTS 连续失败 3 次 | 降级为纯文本输出，记录 warning |
| 麦克风权限被拒 | 启动时检测，给出明确中文提示 |
| 打断误检（背景噪音） | ducking 后 300ms 二次确认，误检恢复播放 |
| VAD 持续静音超过 30s | 播放提示"我还在，请说话"，避免尴尬沉默 |
| Bot 进入 CLOSE/FAILED | 正常结束管线，播放结束语后停止 |

---

## 7. 测试策略

### 7.1 单元测试（~20 项）

| 模块 | 测试内容 |
|---|---|
| `AudioSource` | MicrophoneSource 启停、SimulatedSource 分块注入、FileSource EOF |
| `DuplexAudioOutput` | 正常播放完成、打断检测、ducking 逻辑、误检恢复、300ms 二次确认 |
| `InterruptionContext` | 打断位置记录、agent_text 截断 |
| `Pipeline` | 状态转换合法性、step() 循环退出、打断状态机 |
| `VAD` | 现有测试保留 |

### 7.2 集成测试（~10 项）

| 场景 | 验证 |
|---|---|
| 人声模式完整对话 | MicrophoneSource → Pipeline → 5 轮对话 → CLOSE |
| 自动模式完整对话 | SimulatedSource → Pipeline → 10 轮 → 报告正确 |
| 打断场景 | Agent 播放中注入音频 → 打断确认 → 状态切换到 LISTENING |
| 无打断场景 | 完整播放到结束 → 正常进入下一轮聆听 |
| ASR 误差链 | 模拟 ASR 错误 → Bot 仍能处理 → 不崩溃 |
| TTS 失败恢复 | TTS 连续失败 2 次 → 第 3 次成功 → 继续对话 |
| 长静音 | 30s 无语音 → 提示播放 → 正常恢复 |

### 7.3 端到端演示测试（~3 项）

| 场景 |
|---|
| 真人说 5 句话，完整走通 H2 催收流程 |
| 自动仿真 3 种 persona 各跑 5 轮，不崩 |
| 打断 3 次后 bot 仍能正常结束通话 |

### 7.4 测试基础设施

- `VoiceTestHarness`: 测试夹具，组装 Pipeline + FileSource（预录音频），不依赖麦克风
- `RecordingSource`: 用预录音频文件模拟实时麦克风输入（FileSource + 时间戳模拟）
- 注入点：`Pipeline.on_state_change`、`Pipeline.on_turn_complete` 回调用于断言

---

## 8. 文件变更

| 文件 | 操作 | 职责 |
|---|---|---|
| `src/core/voice/audio_source.py` | **新建** | AudioSource 抽象 + MicrophoneSource / SimulatedSource / FileSource |
| `src/core/voice/audio_output.py` | **新建** | DuplexAudioOutput + barge-in + ducking |
| `src/core/voice/pipeline.py` | **新建** | DuplexCallPipeline + 状态机 + 并发控制 |
| `src/core/voice/call_simulator.py` | **新建** | CallSimulator (替代 CustomerVoiceSimulator) |
| `src/core/voice/audio_io.py` | **修改** | AudioInput 精简为 MicrophoneSource 内部使用；RingBuffer 保留；AudioOutput 移除（被 DuplexAudioOutput 替代） |
| `src/core/voice/conversation.py` | **废弃** | VoiceConversation 不再使用，保留文件加 deprecation 注释 |
| `src/core/voice/customer_simulator.py` | **废弃** | 逻辑迁移到 call_simulator.py |
| `src/core/voice/interruption.py` | **废弃** | 打断逻辑合并到 audio_output.py |
| `src/experiments/voice_simulate_demo.py` | **重写** | 入口改为 DuplexCallPipeline + --mode live/sim |
| `tests/test_duplex_pipeline.py` | **新建** | ~30 项测试（单元+集成） |
| `tests/test_audio_output.py` | **新建** | ~10 项 DuplexAudioOutput 测试 |

### 不变文件
- `src/core/voice/vad.py` — SimpleEnergyVAD 保持不变
- `src/core/voice/asr.py` — ASRPipeline 保持不变
- `src/core/voice/tts.py` — TTSManager 保持不变
- `src/core/chatbot.py` — CollectionChatBot 不变
- `src/core/simulator.py` — 文本模拟器不变

---

## 9. 迁移路径

1. 新建 4 个文件（audio_source, audio_output, pipeline, call_simulator）
2. 写测试，确保新管线独立跑通
3. 重写 `voice_simulate_demo.py` 入口
4. 标记旧文件废弃（conversation.py, customer_simulator.py, interruption.py）
5. 跑全量回归测试确认无破坏

不删除旧文件，避免影响 `src/api/main.py` 中可能引用 conversation.py 的代码。API 层后续单独迁移。

---

## 10. 演示运行方式

```bash
# 真人模式（麦克风输入，默认）
python src/experiments/voice_simulate_demo.py --mode live

# 自动仿真模式
python src/experiments/voice_simulate_demo.py --mode sim --persona resistant --resistance high

# 自动仿真 + 模拟打断（SimulatedSource 在 agent 播放期间注入音频触发打断）
python src/experiments/voice_simulate_demo.py --mode sim --simulate-interruptions

# 回放模式（用于回归测试）
python src/experiments/voice_simulate_demo.py --mode replay --recording data/test/fixtures/sample_call.wav
```

---

## 11. 验收标准

1. 双工打断：demo 中 agent 说话时用户插话，agent 在 500ms 内停止播放并响应
2. 一键切换：`--mode live` 和 `--mode sim` 共用一个 Pipeline 实例
3. 测试：30+ 测试项，覆盖管线、输出、打断三个模块
4. 回归：现有 96 项测试全部通过
5. 可视化反馈：终端输出清晰展示当前状态（LISTENING / SPEAKING / INTERRUPTED / PROCESSING）
6. 降级可用：任一组件（ASR/TTS/麦克风）不可用时给出明确提示，不崩溃
