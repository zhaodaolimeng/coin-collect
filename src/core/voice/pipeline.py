#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双工通话管线 — 统一人声/自动两模式的语音催收核心循环"""
import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Callable

import numpy as np

from core.voice.audio_source import AudioSource
from core.voice.audio_output import DuplexAudioOutput, PlaybackResult
from core.voice.vad import SimpleEnergyVAD, VADState

logger = logging.getLogger(__name__)


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

        # 长静音计时
        self._total_silence_s = 0.0

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
        if chunk is None or len(chunk) == 0:
            await asyncio.sleep(0.01)
            return

        result = self._vad.process_frame(chunk)

        if result.state == VADState.VOICE:
            self._voice_detected = True

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
                self._set_state(PipelineState.PROCESSING)
                return

            # 语音超长 → 截断
            if self._speech_pos >= self._max_speech_samples:
                self._set_state(PipelineState.PROCESSING)
                return

        # 长静音检测
        if not self._voice_detected:
            self._total_silence_s += len(chunk) / self._config.sample_rate
            if self._total_silence_s >= self._config.max_silence_duration:
                logger.info("长静音超时，进入关闭流程")
                self._set_state(PipelineState.CLOSING)
                return

    async def _step_process(self):
        """PROCESSING: ASR → Bot → TTS"""
        speech = self._speech_buffer[:self._speech_pos].copy()
        self._reset_listen_state()

        # ASR
        asr_text = ""
        if self._asr and self._asr.is_available and len(speech) > 0:
            try:
                asr_text = await self._asr.transcribe_async(speech)
            except Exception as e:
                logger.error(f"ASR 失败: {e}")
                asr_text = ""

        logger.info(f"ASR: '{asr_text}'")

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

        # TTS
        agent_audio = None
        if agent_text:
            try:
                tts_result = await self._tts.synthesize(agent_text)
                if tts_result.success:
                    if tts_result.audio_data is not None:
                        if isinstance(tts_result.audio_data, bytes):
                            agent_audio = np.frombuffer(tts_result.audio_data, dtype=np.float32)
                        else:
                            agent_audio = tts_result.audio_data
                    elif tts_result.audio_file:
                        import soundfile as sf
                        agent_audio, _ = sf.read(tts_result.audio_file, dtype='float32')
                        if agent_audio.ndim > 1:
                            agent_audio = agent_audio[:, 0]
            except Exception as e:
                logger.error(f"TTS 失败: {e}")

        self._current_agent_text = agent_text
        self._current_agent_audio = agent_audio
        self._current_asr_text = asr_text
        self._turn_id += 1

        # 检查 Bot 是否结束
        if self._bot_is_finished():
            self._set_state(PipelineState.CLOSING)
        else:
            self._set_state(PipelineState.RESPONDING)

    async def _step_respond(self):
        """RESPONDING: 播放 Agent 音频，同时继续监听打断"""
        if self._current_agent_audio is not None and len(self._current_agent_audio) > 0:
            self._speak_task = asyncio.create_task(
                self._output.speak(self._current_agent_audio)
            )

            while self._speak_task and not self._speak_task.done():
                await asyncio.sleep(0.05)

            result = await self._speak_task
            self._speak_task = None

            if result == PlaybackResult.INTERRUPTED:
                self._last_interruption = InterruptionContext(
                    agent_text_interrupted=self._current_agent_text or "",
                    agent_playback_position=0.5,
                    customer_rms_peak=self._source.current_rms(),
                )
                self._set_state(PipelineState.INTERRUPTED)
                return

        # 播放完成，回到聆听
        if self._bot_is_finished():
            self._set_state(PipelineState.CLOSING)
        else:
            self._set_state(PipelineState.LISTENING)

    async def _step_interrupted(self):
        """INTERRUPTED: 被打断的过渡处理"""
        logger.info("处理打断...")
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
            logger.debug(f"Pipeline: {old.name} → {new_state.name}")
            if self.on_state_change:
                self.on_state_change(old, new_state)

    def _bot_is_finished(self) -> bool:
        """检查 Bot 状态是否为结束状态"""
        state_name = getattr(self._chatbot.state, 'name', str(self._chatbot.state))
        return state_name in ("CLOSE", "FAILED")

    def _reset_listen_state(self):
        self._speech_pos = 0
        self._silence_samples = 0
        self._voice_detected = False
        self._total_silence_s = 0.0
        self._vad.reset()
