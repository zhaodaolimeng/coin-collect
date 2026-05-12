# DEPRECATED since 2026-05 — 被 src.core.voice.pipeline.DuplexCallPipeline 取代
# 保留此文件仅为向后兼容 src/api/main.py 的历史引用
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音对话管线模块
串联：麦克风 → VAD → ASR → 纠错 → Chatbot → TTS → 扬声器
支持打断（barge-in）、流式交互
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Callable

import numpy as np

from src.core.voice.audio_io import AudioInput, AudioOutput, RingBuffer
from src.core.voice.vad import SimpleEnergyVAD, VADState
from src.core.voice.asr import ASRPipeline
from src.core.voice.tts import TTSManager

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """语音对话状态"""
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()
    CLOSED = auto()


@dataclass
class VoiceTurn:
    """语音对话轮次记录"""
    turn_id: int = 0
    user_audio: Optional[np.ndarray] = None
    asr_text: str = ""
    corrected_text: str = ""
    agent_text: str = ""
    tts_audio: Optional[np.ndarray] = None
    state: str = ""
    timestamp: float = field(default_factory=time.time)


class VoiceConversation:
    """
    语音对话管线

    使用示例::

        conv = VoiceConversation(chatbot=bot)
        conv.on_agent_response = lambda text: print(f"Agent: {text}")
        await conv.start()
        await conv.run_forever()
    """

    def __init__(
        self,
        chatbot,
        *,
        sample_rate: int = 16000,
        block_size: int = 1600,
        silence_duration: float = 1.0,
        max_speech_duration: float = 15.0,
        energy_threshold: float = 0.01,
        asr_model_size: str = "small",
        device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.silence_duration = silence_duration
        self.max_speech_duration = max_speech_duration
        self.energy_threshold = energy_threshold

        self._chatbot = chatbot
        self._vad = SimpleEnergyVAD(
            sample_rate=sample_rate,
            frame_duration_ms=int(block_size / sample_rate * 1000),
            energy_threshold=energy_threshold,
        )

        self._asr: Optional[ASRPipeline] = None
        self._asr_model_size = asr_model_size
        self._tts = TTSManager()
        self._audio_in = AudioInput(sample_rate=sample_rate, block_size=block_size, device=device)
        self._audio_out = AudioOutput(sample_rate=sample_rate, device=device)

        self._state = ConversationState.IDLE
        self._turns: list[VoiceTurn] = []
        self._turn_counter = 0
        self._running = False

        # callbacks
        self.on_state_change: Optional[Callable] = None
        self.on_asr_result: Optional[Callable] = None
        self.on_agent_response: Optional[Callable] = None
        self.on_interrupted: Optional[Callable] = None

    @property
    def state(self) -> ConversationState:
        return self._state

    @property
    def turns(self) -> list:
        return self._turns

    async def _load_asr(self) -> bool:
        if self._asr is not None:
            return self._asr.is_available

        corrector = getattr(self._chatbot, "asr_corrector", None)
        self._asr = await ASRPipeline.create(
            model_size=self._asr_model_size,
            corrector=corrector,
        )
        return self._asr.is_available

    def _set_state(self, new_state: ConversationState):
        old = self._state
        self._state = new_state
        if old != new_state and self.on_state_change:
            self.on_state_change(old, new_state)

    async def start(self):
        """启动语音管线"""
        asr_ok = await self._load_asr()
        if not asr_ok:
            logger.warning("ASR model not available, voice conversation will not work")

        self._audio_in.start()
        self._running = True
        self._set_state(ConversationState.LISTENING)
        logger.info("Voice conversation started")

    async def stop(self):
        """停止语音管线"""
        self._running = False
        self._audio_in.stop()
        self._audio_out.stop()
        if self._asr:
            self._asr.shutdown()
        self._set_state(ConversationState.CLOSED)
        logger.info("Voice conversation stopped")

    async def run_once(self) -> Optional[VoiceTurn]:
        """
        执行一轮对话：听 → 识别 → 回复 → 播放

        Returns:
            本轮对话记录，如果没检测到语音则返回 None
        """
        self._set_state(ConversationState.LISTENING)

        # 1. 监听用户语音
        user_audio = await self._listen_for_speech()
        if user_audio is None or len(user_audio) == 0:
            return None

        self._set_state(ConversationState.PROCESSING)

        turn = VoiceTurn(
            turn_id=self._turn_counter,
            user_audio=user_audio,
            state=self._chatbot.state.name,
        )
        self._turn_counter += 1

        # 2. ASR 识别
        if self._asr and self._asr.is_available:
            asr_text = await self._asr.transcribe_async(user_audio)
        else:
            asr_text = ""

        turn.asr_text = asr_text
        turn.corrected_text = asr_text  # ASRPipeline already corrects internally

        if self.on_asr_result:
            self.on_asr_result(asr_text)

        logger.info(f"ASR: '{asr_text}'")

        # 3. Chatbot 处理
        agent_text, tts_file = await self._chatbot.process(
            customer_input=asr_text if asr_text else None,
            use_tts=False,  # TTS handled separately here
        )
        turn.agent_text = agent_text

        if self.on_agent_response:
            self.on_agent_response(agent_text)

        # 4. TTS 合成
        if agent_text:
            tts_result = await self._tts.synthesize(agent_text)
            if tts_result.success and tts_result.audio_data is not None:
                turn.tts_audio = tts_result.audio_data
            elif tts_result.success and tts_result.audio_file:
                try:
                    import soundfile as sf
                    audio_data, _ = sf.read(tts_result.audio_file, dtype="float32")
                    if audio_data.ndim > 1:
                        audio_data = audio_data[:, 0]
                    turn.tts_audio = audio_data
                except Exception:
                    pass

        # 5. 播放（支持打断）
        if turn.tts_audio is not None and len(turn.tts_audio) > 0:
            self._set_state(ConversationState.SPEAKING)
            interrupted = await self._play_with_interrupt(turn.tts_audio)
            if interrupted and self.on_interrupted:
                self.on_interrupted()

        self._turns.append(turn)
        return turn

    async def run_forever(self):
        """持续运行对话，直到会话关闭或手动停止"""
        try:
            while self._running and self._state != ConversationState.CLOSED:
                # 检查 chatbot 是否已结束
                from src.core.chatbot import ChatState
                if self._chatbot.state in (ChatState.CLOSE, ChatState.FAILED):
                    logger.info("Conversation ended by chatbot")
                    break

                await self.run_once()

                # 短暂间隔防止 CPU 空转
                await asyncio.sleep(0.1)
        finally:
            await self.stop()

    async def _listen_for_speech(self) -> Optional[np.ndarray]:
        """监听麦克风，检测语音活动，返回完整语音段"""
        total_samples = int(self.max_speech_duration * self.sample_rate)
        speech_buffer = np.zeros(total_samples, dtype=np.float32)
        speech_pos = 0
        silence_samples = 0
        silence_threshold = int(self.silence_duration * self.sample_rate)
        voice_detected = False

        chunk_duration = self.block_size / self.sample_rate
        poll_interval = max(chunk_duration * 0.5, 0.01)

        while self._running:
            # 从麦克风缓冲区读取
            audio_chunk = self._audio_in.read(duration=chunk_duration)
            if len(audio_chunk) == 0:
                await asyncio.sleep(poll_interval)
                continue

            # VAD 检测
            result = self._vad.process_frame(audio_chunk)

            if result.state == VADState.VOICE:
                voice_detected = True

            if voice_detected:
                # 写入语音缓冲区
                remaining = total_samples - speech_pos
                to_write = audio_chunk[:remaining]
                speech_buffer[speech_pos:speech_pos + len(to_write)] = to_write
                speech_pos += len(to_write)

                if result.state == VADState.SILENCE:
                    silence_samples += len(audio_chunk)
                else:
                    silence_samples = 0

                # 静音超时 → 语音结束
                if silence_samples >= silence_threshold:
                    speech = speech_buffer[:speech_pos].copy()
                    self._vad.reset()
                    return speech

                # 语音超时 → 截断
                if speech_pos >= total_samples:
                    self._vad.reset()
                    return speech_buffer[:speech_pos].copy()

            await asyncio.sleep(poll_interval)

        return None

    async def _play_with_interrupt(self, audio: np.ndarray) -> bool:
        """
        播放音频，同时监听打断

        Returns:
            True if interrupted, False if completed
        """
        chunk_size = 1024
        pos = 0
        interrupted = False

        try:
            import sounddevice as sd

            poll_interval = chunk_size / self.sample_rate * 0.5

            while pos < len(audio) and not interrupted:
                end = min(pos + chunk_size, len(audio))
                sd.play(audio[pos:end], samplerate=self.sample_rate, blocking=True)

                # 检测打断：检查是否有新的语音输入
                rms = self._audio_in.current_rms()
                if rms > self.energy_threshold * 1.5:
                    sd.stop()
                    interrupted = True

                pos = end
                if not interrupted:
                    await asyncio.sleep(poll_interval)

        except ImportError:
            logger.error("sounddevice not installed")
        except Exception as e:
            logger.error(f"Playback error: {e}")

        return interrupted

    def reset(self):
        """重置会话状态"""
        self._turns.clear()
        self._turn_counter = 0
        self._vad.reset()
        if self._asr:
            self._asr.asr.reset()
