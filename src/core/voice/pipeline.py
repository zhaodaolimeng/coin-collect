#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双工通话管线 — 统一人声/自动两模式的语音催收核心循环"""
import asyncio
import collections
import logging
import subprocess
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Callable

import numpy as np

from core.voice.audio_source import AudioSource
from core.voice.audio_output import DuplexAudioOutput, PlaybackResult
from core.voice.streaming_asr import StreamingASR, StreamingASRConfig
from core.voice.vad import SimpleEnergyVAD, VADState

logger = logging.getLogger(__name__)


def _load_audio_ffmpeg(file_path: str, target_sr: int = 16000) -> np.ndarray:
    """用 ffmpeg 加载音频文件（支持 MP3/WAV），返回 float32 单声道"""
    result = subprocess.run(
        ['ffmpeg', '-y', '-i', file_path, '-f', 'f32le', '-acodec', 'pcm_f32le',
         '-ar', str(target_sr), '-ac', '1', '-'],
        capture_output=True,
        timeout=60,
    )
    if result.returncode != 0 or len(result.stdout) == 0:
        raise RuntimeError(f"ffmpeg 加载音频失败: {result.stderr.decode()}")
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def _normalize_tts_text(text: str) -> str:
    """归一化印尼语 TTS 文本中的数字格式，确保 Piper TTS 正确朗读。

    转换规则：
    - 千位分隔逗号 → 点号: "Rp 500,000" → "Rp 500.000"
      印尼语中逗号是小数点，点号才是千位分隔；逗号后跟恰好3位数字时转换
    - 缩写展开: "jt" → "juta", "rb" → "ribu"
      Piper 会把缩写读成字母拼写 "j-e-t-e" 而非 "juta"
    """
    import re
    # 千位分隔符: 数字 + 逗号 + 恰好3位数字(后面不是数字) → 点号
    # 循环处理 "2,500,000" 这种多层分隔
    prev = None
    while text != prev:
        prev = text
        text = re.sub(r'(\d),(\d{3})(?!\d)', r'\1.\2', text)
    # 缩写展开 (注意边界: 前后不是字母)
    text = re.sub(r'\bjt\b', 'juta', text)
    text = re.sub(r'\brb\b', 'ribu', text)
    return text


class PipelineState(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    RESPONDING = auto()
    INTERRUPTED = auto()
    CLOSING = auto()
    CLOSED = auto()


@dataclass
class PipelineConfig:
    sample_rate: int = 16000
    block_size: int = 1600
    silence_duration: float = 1.0
    max_speech_duration: float = 15.0
    max_silence_duration: float = 30.0
    cooldown_duration: float = 0.3  # 播放结束后丢弃音频的冷却时间，避免喇叭回声被识别
    prompt_on_silence: str = "Halo, apakah Anda masih di sana?"


@dataclass
class StepResult:
    """单步执行结果"""
    state_from: PipelineState
    state_to: PipelineState
    asr_text: str = ""
    agent_text: str = ""
    agent_audio: np.ndarray | None = None
    interrupted: bool = False
    turn_id: int = 0
    elapsed_s: float = 0.0


@dataclass
class InterruptionContext:
    """打断上下文 — 传递给 Bot 用于策略调整"""
    agent_text_interrupted: str = ""
    agent_playback_position: float = 0.0
    customer_rms_peak: float = 0.0
    partial_asr: str | None = None


class DuplexCallPipeline:
    """双工通话管线。统一人声模式与自动仿真模式的语音催收核心循环。

    使用:
        source = MicrophoneSource()
        output = DuplexAudioOutput(source)
        pipeline = DuplexCallPipeline(chatbot, source, output, asr, tts, vad)
        await pipeline.start()
        while pipeline.state != PipelineState.CLOSED:
            result = await pipeline.step()
            if result:
                print(f"[{result.state_to.name}] ASR: {result.asr_text}")
        await pipeline.stop()
    """

    def __init__(
        self,
        chatbot,
        source: AudioSource,
        output: DuplexAudioOutput,
        asr_pipeline,
        tts_manager,
        vad: SimpleEnergyVAD | None = None,
        *,
        config: PipelineConfig | None = None,
    ):
        self._chatbot = chatbot
        self._source = source
        self._output = output
        self._asr = asr_pipeline
        self._tts = tts_manager
        self._vad = vad or SimpleEnergyVAD(
            sample_rate=source.sample_rate,
            energy_threshold=0.01,
        )
        self._config = config or PipelineConfig()

        self._state = PipelineState.IDLE
        self._running = False
        self._turn_id = 0

        # 聆听缓冲区
        self._speech_buffer = np.zeros(
            int(self._config.max_speech_duration * self._config.sample_rate),
            dtype=np.float32,
        )
        self._speech_pos = 0
        self._max_speech_samples = int(self._config.max_speech_duration * self._config.sample_rate)
        self._voice_detected = False

        # 前置缓冲 — VAD 触发前保留最近 500ms 音频，避免语音开头截断
        pre_buffer_chunks = max(1, int(0.5 * self._config.sample_rate / self._config.block_size))
        self._pre_buffer: collections.deque = collections.deque(maxlen=pre_buffer_chunks)

        # 长静音计时 — 使用 wall-clock 时间，避免积压音频快速消费导致计时失真
        self._silence_since: float | None = None

        # 冷却计时 — 播放结束后短时间丢弃音频，避免喇叭回声被 VAD 误识别
        self._listen_cooldown_samples: int = 0  # 还需丢弃的样本数

        # 后台播放 task
        self._speak_task: asyncio.Task | None = None

        # 打断上下文（最近一次）
        self._last_interruption: Optional[InterruptionContext] = None

        # 前端播放状态 — 发送音频后等待前端 playback_done
        self._frontend_playback_done: bool = False
        self._respond_audio_sent: bool = False

        # 流式 ASR — 增长窗口 + 去重，模拟实时转写
        self._streaming_asr: StreamingASR | None = None
        self._streaming_config = StreamingASRConfig()
        self._streaming_final_text: str = ""
        self._last_asr_submit_time: float = 0.0
        self._last_asr_submit_pos: int = 0

        # 当前轮次的 Agent 输出
        self._current_agent_text = ""
        self._current_agent_audio: np.ndarray | None = None
        self._current_asr_text = ""

        # 回调
        self.on_state_change: Optional[Callable] = None
        self.on_turn_complete: Optional[Callable] = None
        self.on_debug: Optional[Callable[[str], None]] = None

    def _debug(self, msg: str):
        """发送调试信息到回调 + 后端日志"""
        logger.debug(msg)
        if self.on_debug:
            try:
                self.on_debug(msg)
            except Exception:
                pass

    def _save_asr_audio(self, speech: np.ndarray, asr_text: str):
        """保存 ASR 输入音频用于诊断回声/误识别问题，轮转保留最近 50 个文件"""
        import wave
        import os
        import re
        from pathlib import Path
        dump_dir = Path(__file__).parent.parent.parent.parent / "data/runs/debug"
        dump_dir.mkdir(parents=True, exist_ok=True)
        # 轮转: 保留最近 50 个文件
        existing = sorted(dump_dir.glob("asr_*.wav"), key=lambda p: p.stat().st_mtime)
        for old in existing[:-49]:
            try:
                old.unlink()
            except OSError:
                pass
        # 文件名: asr_HHMMSS_文本前20字符.wav
        safe_text = re.sub(r'[^\w]', '_', asr_text[:20].strip())
        fname = f"asr_{time.strftime('%H%M%S')}_{safe_text}.wav"
        dump_path = dump_dir / fname
        int16_audio = (speech * 32767).astype(np.int16)
        with wave.open(str(dump_path), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(int16_audio.tobytes())
        logger.info(f"[DEBUG] ASR音频已保存: {dump_path.name} ({len(speech) / 16000:.1f}s, text='{asr_text[:60]}')")

    def _set_voice_detected(self, value: bool):
        """设置 _voice_detected 并记录变更轨迹，排查隐性重置"""
        old = self._voice_detected
        if old != value:
            import traceback
            tb = traceback.extract_stack()
            caller = tb[-2] if len(tb) >= 2 else tb[-1]
            self._debug(f"[VD-TRACE] _voice_detected {old} → {value} "
                        f"caller={caller.name}:{caller.lineno}")
        self._voice_detected = value

    # ── 属性 ──────────────────────────────────────────────

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def turn_id(self) -> int:
        return self._turn_id

    @property
    def last_interruption(self) -> Optional[InterruptionContext]:
        return self._last_interruption

    def handle_interruption(self):
        """外部打断：停止当前播放，状态切换至 INTERRUPTED

        仅在 RESPONDING 状态时切换，避免延迟到达的打断消息在管线已回到
        LISTENING 后触发不必要的 _reset_listen_state() 清空已累积的用户语音。
        """
        self._debug(f"[打断] handle_interruption 当前状态={self._state.name}")
        self._output.stop()
        if self._state == PipelineState.RESPONDING:
            self._debug("[打断] RESPONDING → INTERRUPTED")
            self._reset_respond_state()
            self._set_state(PipelineState.INTERRUPTED)
        else:
            self._debug(f"[打断] 状态={self._state.name} 非RESPONDING，忽略状态切换")

    def notify_playback_done(self):
        """前端通知：Agent 音频播放完成"""
        self._debug("[PLAY] 前端通知 playback_done")
        self._frontend_playback_done = True

    def _reset_respond_state(self):
        """重置 RESPONDING 阶段的状态标记"""
        self._respond_audio_sent = False
        self._frontend_playback_done = False
        self._speak_task = None

    def inject_greeting_audio(self, audio: np.ndarray):
        """注入问候音频，交由管线播放（确保打断检测生效）"""
        self._current_agent_audio = audio.astype(np.float32)
        self._set_state(PipelineState.RESPONDING)

    # ── 生命周期 ──────────────────────────────────────────

    async def start(self):
        await self._source.start()
        self._running = True
        self._vad.reset()
        if self._asr is not None and self._asr.is_available:
            asr_engine = getattr(self._asr, 'asr', self._asr)
            if hasattr(asr_engine, 'sample_rate'):
                # 检测 sherpa-onnx 引擎 → 原生流式（无需增长窗口）
                try:
                    from core.voice.sherpa_asr import SherpaASR, SherpaStreamingASR
                except ImportError:
                    SherpaASR = None
                    SherpaStreamingASR = None
                if SherpaASR is not None and isinstance(asr_engine, SherpaASR):
                    self._streaming_asr = SherpaStreamingASR(asr_engine)
                    self._streaming_asr.on_partial_result = self._on_streaming_partial
                    self._debug("[流式ASR] 已启用 sherpa-onnx 原生流式转写")
                else:
                    self._streaming_asr = StreamingASR(asr_engine, self._streaming_config)
                    self._streaming_asr.on_partial_result = self._on_streaming_partial
                    self._debug("[流式ASR] 已启用增长窗口流式转写")
            else:
                self._streaming_asr = None
                self._debug("[流式ASR] ASR引擎不支持流式，回退整段转写")
        else:
            self._streaming_asr = None
        self._set_state(PipelineState.LISTENING)
        logger.info("DuplexCallPipeline started")

    async def stop(self):
        self._running = False
        if self._speak_task and not self._speak_task.done():
            self._speak_task.cancel()
        self._output.stop()
        await self._source.stop()
        self._set_state(PipelineState.CLOSED)
        logger.info("DuplexCallPipeline stopped")

    async def run_until_closed(self):
        """自动运行直到通话结束"""
        await self.start()
        try:
            while self._state not in (PipelineState.CLOSING, PipelineState.CLOSED):
                await self.step()
                await asyncio.sleep(0.01)
        finally:
            if self._state != PipelineState.CLOSED:
                await self.stop()

    # ── 核心循环 ──────────────────────────────────────────

    async def step(self) -> StepResult | None:
        """执行一步管线。根据当前状态分发。"""
        if not self._running:
            return None

        t0 = time.time()
        state_before = self._state

        if self._state == PipelineState.LISTENING:
            await self._step_listen()
        elif self._state == PipelineState.PROCESSING:
            await self._step_process()
        elif self._state == PipelineState.RESPONDING:
            await self._step_respond()
        elif self._state == PipelineState.INTERRUPTED:
            await self._step_interrupted()
        elif self._state == PipelineState.CLOSING:
            await self._step_closing()
        elif self._state in (PipelineState.IDLE, PipelineState.CLOSED):
            pass

        elapsed = time.time() - t0
        return StepResult(
            state_from=state_before,
            state_to=self._state,
            asr_text=self._current_asr_text if state_before == PipelineState.PROCESSING else "",
            agent_text=self._current_agent_text if state_before == PipelineState.PROCESSING else "",
            turn_id=self._turn_id,
            elapsed_s=elapsed,
        )

    # ── 状态处理 ──────────────────────────────────────────

    async def _step_listen(self):
        """LISTENING: 读取一个音频块，VAD 检测，累积语音段"""
        chunk = await self._source.read_chunk()

        # None = 暂无数据（WebSocket queue 为空），不是真正的静音
        if chunk is None:
            now = time.time()
            if self._voice_detected:
                # 使用墙钟时间检测静音，避免空轮询错误累积
                if self._silence_since is None:
                    self._silence_since = now
                elif now - self._silence_since >= self._config.silence_duration and self._speech_pos > 0:
                    dur = self._speech_pos / self._config.sample_rate
                    self._debug(f"[VAD] 静音超时: wall_clock={now - self._silence_since:.1f}s 语音={dur:.1f}s → PROCESSING")
                    self._set_state(PipelineState.PROCESSING)
            elif self._running:
                if self._silence_since is None:
                    self._silence_since = now
                elif now - self._silence_since >= self._config.max_silence_duration:
                    self._debug(f"[VAD] 长静音超时({now - self._silence_since:.1f}s) → CLOSING")
                    self._set_state(PipelineState.CLOSING)
            return

        if len(chunk) == 0:
            return

        # 样本级冷却：播放/打断结束后丢弃指定时长的音频，避免喇叭回声被 VAD 误识别
        if self._listen_cooldown_samples > 0:
            self._listen_cooldown_samples -= len(chunk)
            return

        result = await asyncio.to_thread(self._vad.process_frame, chunk)
        energy = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))

        # 逐帧调试：记录高能量帧及每 20 帧汇总，排查 VAD 不触发原因
        self._listen_step_count = getattr(self, '_listen_step_count', 0) + 1
        if energy > 0.0008 or self._listen_step_count <= 5 or self._listen_step_count % 20 == 0:
            self._debug(f"[LISTEN] step#{self._listen_step_count} energy={energy:.5f} "
                        f"vad_state={result.state.value} speech_prob={result.confidence:.3f} "
                        f"voice_det={self._voice_detected} speech_pos={self._speech_pos}")

        # 未检测到语音时，保存到前置缓冲（避免语音开头被 VAD 迟滞截断）
        if not self._voice_detected:
            self._pre_buffer.append(chunk.copy())

        # 双门 VAD：speech_prob > threshold 且 RMS energy > floor
        # 背景噪声 speech_prob 可达 0.1-0.4 但 energy < 0.001，需滤除
        is_true_voice = result.state == VADState.VOICE and energy > 0.001

        if is_true_voice:
            if not self._voice_detected:
                self._debug(f"[VAD] 检测到语音 energy={energy:.4f} threshold={getattr(self._vad, 'energy_threshold', '?')}")
                # 将前置缓冲内容复制到语音缓冲区开头
                prepended = 0
                for old_chunk in self._pre_buffer:
                    remaining = self._max_speech_samples - self._speech_pos
                    to_write = old_chunk[:remaining]
                    end_pos = self._speech_pos + len(to_write)
                    if end_pos <= len(self._speech_buffer):
                        self._speech_buffer[self._speech_pos:end_pos] = to_write
                    self._speech_pos = min(end_pos, len(self._speech_buffer))
                    prepended += len(to_write)
                if prepended > 0:
                    self._debug(f"[VAD] 前置缓冲: {prepended}样本({prepended / self._config.sample_rate:.1f}s)已追加")
                self._pre_buffer.clear()
            self._set_voice_detected(True)
            self._silence_since = None

        if self._voice_detected:
            remaining = self._max_speech_samples - self._speech_pos
            to_write = chunk[:remaining]
            end_pos = self._speech_pos + len(to_write)
            if end_pos <= len(self._speech_buffer):
                self._speech_buffer[self._speech_pos:end_pos] = to_write
            self._speech_pos = min(end_pos, len(self._speech_buffer))

            if not is_true_voice:
                if self._silence_since is None:
                    self._silence_since = time.time()
            else:
                self._silence_since = None

            # 流式 ASR: 语音累积足够后周期性提交增长中的音频快照
            self._maybe_submit_streaming_asr()

            # 静音超时（墙钟时间）→ 语音段结束
            if self._silence_since is not None:
                now = time.time()
                if now - self._silence_since >= self._config.silence_duration and self._speech_pos > 0:
                    dur = self._speech_pos / self._config.sample_rate
                    self._debug(f"[VAD] 静音超时: wall_clock={now - self._silence_since:.1f}s 语音={dur:.1f}s → PROCESSING")
                    self._finalize_streaming_asr()
                    self._set_state(PipelineState.PROCESSING)
                    return

            # 语音超长 → 截断
            if self._speech_pos >= self._max_speech_samples:
                self._debug(f"[VAD] 语音超长({self._speech_pos}样本) → PROCESSING")
                self._finalize_streaming_asr()
                self._set_state(PipelineState.PROCESSING)
                return

        # 长静音检测（未检测到语音时）
        if not self._voice_detected:
            now = time.time()
            if self._silence_since is None:
                self._silence_since = now
            elif now - self._silence_since >= self._config.max_silence_duration:
                self._debug(f"[VAD] 长静音超时({now - self._silence_since:.1f}s) → CLOSING")
                self._set_state(PipelineState.CLOSING)
                return

    async def _step_process(self):
        """PROCESSING: ASR → Bot → TTS"""

        # 每次进入 PROCESSING 时清空上一轮的缓存结果，
        # 避免等待 ASR 就绪期间反复发送相同的旧文本
        self._current_asr_text = ""
        self._current_agent_text = ""

        # ASR 未就绪时等待（不丢弃已累积的语音），超时后回退 LISTENING
        if self._asr is None or not self._asr.is_available:
            retries = getattr(self, '_asr_wait_retries', 0)
            if retries == 0:
                self._debug(f"[PROCESS] ASR未就绪，等待加载（最多5s）...")
            if retries < 50:  # 50 * 100ms = 5s
                self._asr_wait_retries = retries + 1
                await asyncio.sleep(0.1)
                return  # 保持 PROCESSING 状态，下次 step 重试
            self._debug(f"[PROCESS] ASR等待超时 → 回退LISTENING")
            self._asr_wait_retries = 0
            self._set_state(PipelineState.LISTENING)
            return
        self._asr_wait_retries = 0

        # 等待流式 ASR 最终结果（如有），必须在 _reset_listen_state 之前保存
        streaming_text: str = ""
        if self._streaming_asr is not None and self._streaming_asr.is_active:
            if self._streaming_asr.is_final_pending:
                if not self._streaming_asr.has_final_result:
                    return  # 等待流式 ASR 完成，下次 step 重试
            streaming_text = self._streaming_asr.final_text or ""

        speech = self._speech_buffer[:self._speech_pos].copy()
        speech_dur = len(speech) / self._config.sample_rate
        self._debug(f"[PROCESS] 开始处理语音段: {len(speech)}样本({speech_dur:.1f}s)")
        self._reset_listen_state()

        # ASR — 优先使用流式结果，否则全段转写
        asr_text = ""
        if streaming_text:
            # 安全校验：非 ASCII 字符说明 ASR 出错 → 回退全段转写
            if streaming_text.isascii():
                asr_text = streaming_text
                self._debug(f"[ASR] 使用流式结果: '{asr_text}'")
                # 保存音频用于诊断（保存所有结果，不仅仅是短结果）
                if len(speech) > 0:
                    self._save_asr_audio(speech, asr_text)
            else:
                self._debug(f"[ASR] 流式结果含非拉丁字符'{streaming_text}'，回退 faster-whisper")
        elif len(speech) > 0:
            try:
                asr_text = await self._asr.transcribe_async(speech)
            except Exception as e:
                logger.error(f"ASR 失败: {e}")
                asr_text = ""
            if asr_text:
                self._save_asr_audio(speech, asr_text)
                self._debug(f"[DEBUG] 已保存ASR音频: '{asr_text[:60]}'")

        self._debug(f"[ASR] 转写结果: '{asr_text}'" if asr_text else "[ASR] 转写结果: (空)")

        # Bot
        if self._bot_is_finished():
            self._set_state(PipelineState.CLOSING)
            return

        try:
            agent_text, _ = await self._chatbot.process(
                customer_input=asr_text if asr_text else None,
                use_tts=False,
            )
        except Exception as e:
            logger.error(f"Chatbot 处理异常: {e}")
            agent_text = ""

        self._debug(f"[Bot] 回复: '{agent_text[:80]}'" if agent_text else "[Bot] 回复: (空)")

        # TTS
        agent_audio = None
        if agent_text and self._tts is not None:
            try:
                tts_text = _normalize_tts_text(agent_text)
                if tts_text != agent_text:
                    self._debug(f"[TTS] 数字归一化: '{agent_text[:60]}' -> '{tts_text[:60]}'")
                tts_result = await self._tts.synthesize(tts_text)
                if tts_result.success:
                    if tts_result.audio_data is not None:
                        if isinstance(tts_result.audio_data, bytes):
                            agent_audio = np.frombuffer(tts_result.audio_data, dtype=np.float32)
                        else:
                            agent_audio = tts_result.audio_data
                    elif tts_result.audio_file:
                        agent_audio = _load_audio_ffmpeg(tts_result.audio_file)
            except Exception as e:
                logger.error(f"TTS 失败: {e}")

        self._debug(f"[TTS] 合成{'成功' if agent_audio is not None else '失败/跳过'}: {len(agent_audio) if agent_audio is not None else 0}样本")

        self._current_agent_text = agent_text
        self._current_agent_audio = agent_audio
        self._current_asr_text = asr_text
        self._turn_id += 1

        # 检查 Bot 是否结束
        if self._bot_is_finished():
            self._debug("[PROCESS] Bot已结束 → CLOSING")
            self._set_state(PipelineState.CLOSING)
        else:
            self._set_state(PipelineState.RESPONDING)

    async def _step_respond(self):
        """RESPONDING: 发送 Agent 音频后等待前端 playback_done 或 interrupt

        分两阶段:
        - 阶段 A (首次): 快速发送所有音频 chunk，设置 _respond_audio_sent 标记
        - 阶段 B (后续): 非阻塞轮询 _frontend_playback_done，消费积压音频防止队列溢出
        """
        if not self._respond_audio_sent:
            # 阶段 A: 发送音频
            if self._current_agent_audio is not None and len(self._current_agent_audio) > 0:
                dur = len(self._current_agent_audio) / self._config.sample_rate
                self._debug(f"[PLAY] 开始发送Agent音频: {len(self._current_agent_audio)}样本({dur:.1f}s)")
                self._speak_task = asyncio.create_task(
                    self._output.speak(self._current_agent_audio)
                )
            self._respond_audio_sent = True
            return

        # 阶段 B: 等待前端 playback_done
        if self._speak_task and not self._speak_task.done():
            return  # 发送尚未完成，继续等待

        if not self._frontend_playback_done:
            # 消费积压音频，防止队列溢出 (maxsize=200)
            if hasattr(self._source, '_queue'):
                while not self._source._queue.empty():
                    try:
                        self._source._queue.get_nowait()
                    except Exception:
                        break
            return

        # 前端确认播放完成
        self._debug("[PLAY] 收到 playback_done，播放完成")
        if hasattr(self._source, 'flush'):
            self._source.flush(keep_recent_s=0.2)
            self._debug("[FLUSH] 播放结束，已清理过期音频队列（保留最近0.2秒）")
        self._respond_audio_sent = False
        self._frontend_playback_done = False
        if self._bot_is_finished():
            self._set_state(PipelineState.CLOSING)
        else:
            self._listen_cooldown_samples = int(self._config.cooldown_duration * self._config.sample_rate)
            self._set_state(PipelineState.LISTENING)

    async def _step_interrupted(self):
        """INTERRUPTED: 被打断的过渡处理"""
        logger.info("处理打断...")
        self._debug("[打断] _step_interrupted → LISTENING")
        self._reset_respond_state()
        if hasattr(self._source, 'flush'):
            # 打断时不保留近期音频：队列中的残响是 TTS 回放的音频混入麦克风，
            # 保留它会导致 ASR 识别出错误的文本（如将 "halo" 识别为 "Selamat tinggal"）
            self._source.flush(keep_recent_s=0.0)
            self._debug("[FLUSH] 打断后，已清空音频队列（保留0秒）")
        self._listen_cooldown_samples = int(self._config.cooldown_duration * self._config.sample_rate)
        self._set_state(PipelineState.LISTENING)

    async def _step_closing(self):
        """CLOSING: 播放结束语后停止"""
        if self._current_agent_audio is not None and len(self._current_agent_audio) > 0:
            await self._output.speak(self._current_agent_audio)
        self._running = False
        self._set_state(PipelineState.CLOSED)

    # ── 流式 ASR 方法 ─────────────────────────────────────

    def _maybe_submit_streaming_asr(self):
        """语音活跃时周期性提交增长窗口音频快照给流式 ASR"""
        if self._streaming_asr is None:
            return
        min_samples = int(self._streaming_config.min_audio_duration * self._config.sample_rate)
        if self._speech_pos < min_samples:
            return
        now = time.time()
        time_since = now - self._last_asr_submit_time
        samples_since = self._speech_pos - self._last_asr_submit_pos
        min_new_samples = int(self._streaming_config.min_new_audio * self._config.sample_rate)
        if time_since >= self._streaming_config.throttle_interval and samples_since >= min_new_samples:
            self._last_asr_submit_time = now
            self._last_asr_submit_pos = self._speech_pos
            snapshot = self._speech_buffer[:self._speech_pos].copy()
            self._streaming_asr.submit(snapshot)

    def _finalize_streaming_asr(self):
        """标记流式 ASR 最终提交，语音段结束"""
        if self._streaming_asr is not None and self._streaming_asr.is_active:
            self._streaming_asr.mark_final()

    def _on_streaming_partial(self, partial_text: str):
        """流式 ASR 增量结果回调"""
        self._debug(f"[ASR-Partial] '{partial_text}'")

    # ── 内部方法 ──────────────────────────────────────────

    def _set_state(self, new_state: PipelineState):
        old = self._state
        self._state = new_state
        if old != new_state:
            self._debug(f"[状态] {old.name} → {new_state.name}")
            if new_state == PipelineState.LISTENING:
                self._reset_listen_state()
            if self.on_state_change:
                self.on_state_change(old, new_state)

    def _bot_is_finished(self) -> bool:
        """检查 Bot 状态是否为结束状态"""
        state_name = getattr(self._chatbot.state, 'name', str(self._chatbot.state))
        return state_name in ("CLOSE", "FAILED")

    def _reset_listen_state(self):
        self._speech_pos = 0
        self._set_voice_detected(False)
        self._silence_since = None
        self._listen_step_count = 0
        self._pre_buffer.clear()
        self._vad.reset()
        self._streaming_final_text = ""
        self._last_asr_submit_time = 0.0
        self._last_asr_submit_pos = 0
        if self._streaming_asr is not None:
            self._streaming_asr.reset()
