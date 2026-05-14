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
        self._silence_samples = 0
        self._silence_threshold = int(self._config.silence_duration * self._config.sample_rate)
        self._max_speech_samples = int(self._config.max_speech_duration * self._config.sample_rate)
        self._voice_detected = False

        # 前置缓冲 — VAD 触发前保留最近 500ms 音频，避免语音开头截断
        pre_buffer_chunks = max(1, int(0.5 * self._config.sample_rate / self._config.block_size))
        self._pre_buffer: collections.deque = collections.deque(maxlen=pre_buffer_chunks)

        # 长静音计时 — 使用 wall-clock 时间，避免积压音频快速消费导致计时失真
        self._silence_since: float | None = None

        # 后台播放 task
        self._speak_task: asyncio.Task | None = None

        # 打断上下文（最近一次）
        self._last_interruption: Optional[InterruptionContext] = None

        # 当前轮次的 Agent 输出
        self._current_agent_text = ""
        self._current_agent_audio: np.ndarray | None = None
        self._current_asr_text = ""

        # 回调
        self.on_state_change: Optional[Callable] = None
        self.on_turn_complete: Optional[Callable] = None
        self.on_debug: Optional[Callable[[str], None]] = None

    def _debug(self, msg: str):
        """发送调试信息到回调"""
        if self.on_debug:
            try:
                self.on_debug(msg)
            except Exception:
                pass

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
            self._set_state(PipelineState.INTERRUPTED)
        else:
            self._debug(f"[打断] 状态={self._state.name} 非RESPONDING，忽略状态切换")

    def inject_greeting_audio(self, audio: np.ndarray):
        """注入问候音频，交由管线播放（确保打断检测生效）"""
        self._current_agent_audio = audio.astype(np.float32)
        self._set_state(PipelineState.RESPONDING)

    # ── 生命周期 ──────────────────────────────────────────

    async def start(self):
        await self._source.start()
        self._running = True
        self._vad.reset()
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
            if self._voice_detected:
                # 已检测到语音，暂无新数据视为静音间隙
                self._silence_samples += self._config.block_size
                if self._silence_samples >= self._silence_threshold and self._speech_pos > 0:
                    dur = self._speech_pos / self._config.sample_rate
                    self._debug(f"[VAD] 语音段结束: {self._speech_pos}样本({dur:.1f}s) → PROCESSING")
                    self._set_state(PipelineState.PROCESSING)
            elif self._running:
                now = time.time()
                if self._silence_since is None:
                    self._silence_since = now
                elif now - self._silence_since >= self._config.max_silence_duration:
                    self._debug(f"[VAD] 长静音超时({now - self._silence_since:.1f}s) → CLOSING")
                    self._set_state(PipelineState.CLOSING)
            return

        if len(chunk) == 0:
            return

        result = self._vad.process_frame(chunk)
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

        if result.state == VADState.VOICE:
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
            self._silence_since = None  # 检测到语音，重置长静音计时

        if self._voice_detected:
            remaining = self._max_speech_samples - self._speech_pos
            to_write = chunk[:remaining]
            end_pos = self._speech_pos + len(to_write)
            if end_pos <= len(self._speech_buffer):
                self._speech_buffer[self._speech_pos:end_pos] = to_write
            self._speech_pos = min(end_pos, len(self._speech_buffer))

            if result.state == VADState.SILENCE:
                self._silence_samples += len(chunk)
            else:
                self._silence_samples = 0

            # 静音超时 → 语音段结束
            if self._silence_samples >= self._silence_threshold and self._speech_pos > 0:
                dur = self._speech_pos / self._config.sample_rate
                self._debug(f"[VAD] 静音超时: silence={self._silence_samples}样本 语音={self._speech_pos}样本({dur:.1f}s) → PROCESSING")
                self._set_state(PipelineState.PROCESSING)
                return

            # 语音超长 → 截断
            if self._speech_pos >= self._max_speech_samples:
                self._debug(f"[VAD] 语音超长({self._speech_pos}样本) → PROCESSING")
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

        speech = self._speech_buffer[:self._speech_pos].copy()
        speech_dur = len(speech) / self._config.sample_rate
        self._debug(f"[PROCESS] 开始处理语音段: {len(speech)}样本({speech_dur:.1f}s)")
        self._reset_listen_state()

        # ASR
        asr_text = ""
        if len(speech) > 0:
            try:
                asr_text = await self._asr.transcribe_async(speech)
            except Exception as e:
                logger.error(f"ASR 失败: {e}")
                asr_text = ""

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
                tts_result = await self._tts.synthesize(agent_text)
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
        """RESPONDING: 播放 Agent 音频，同时继续监听打断"""
        if self._current_agent_audio is not None and len(self._current_agent_audio) > 0:
            dur = len(self._current_agent_audio) / self._config.sample_rate
            self._debug(f"[PLAY] 开始播放Agent音频: {len(self._current_agent_audio)}样本({dur:.1f}s)")
            self._speak_task = asyncio.create_task(
                self._output.speak(self._current_agent_audio)
            )

            while self._speak_task and not self._speak_task.done():
                await asyncio.sleep(0.05)

            result = await self._speak_task
            self._speak_task = None

            self._debug(f"[PLAY] 播放结果: {result.name}")

            if result == PlaybackResult.INTERRUPTED:
                self._last_interruption = InterruptionContext(
                    agent_text_interrupted=self._current_agent_text or "",
                    agent_playback_position=0.5,
                    customer_rms_peak=self._source.current_rms(),
                )
                self._set_state(PipelineState.INTERRUPTED)
                return

        # 播放完成，回到聆听；清空积压的过期音频（保留最近1秒），避免消费播放期间的静音
        if hasattr(self._source, 'flush'):
            self._source.flush(keep_recent_s=1.0)
            self._debug("[FLUSH] 播放结束，已清理过期音频队列（保留最近1秒）")
        if self._bot_is_finished():
            self._set_state(PipelineState.CLOSING)
        else:
            self._set_state(PipelineState.LISTENING)

    async def _step_interrupted(self):
        """INTERRUPTED: 被打断的过渡处理"""
        logger.info("处理打断...")
        self._debug("[打断] _step_interrupted → LISTENING")
        if hasattr(self._source, 'flush'):
            self._source.flush(keep_recent_s=1.0)
            self._debug("[FLUSH] 打断后，已清理过期音频队列（保留最近1秒）")
        self._set_state(PipelineState.LISTENING)

    async def _step_closing(self):
        """CLOSING: 播放结束语后停止"""
        if self._current_agent_audio is not None and len(self._current_agent_audio) > 0:
            await self._output.speak(self._current_agent_audio)
        self._running = False
        self._set_state(PipelineState.CLOSED)

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
        self._silence_samples = 0
        self._set_voice_detected(False)
        self._silence_since = None
        self._listen_step_count = 0
        self._pre_buffer.clear()
        self._vad.reset()
