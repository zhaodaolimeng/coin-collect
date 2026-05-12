# DEPRECATED since 2026-05 — 被 src.core.voice.call_simulator.CallSimulator 取代
# 保留此文件仅为向后兼容 src/api/main.py 的历史引用
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
客户语音模拟器
生成印尼语客户语音，注入ASR管线，形成端到端测试闭环

Pipeline: 文本模拟器 → TTS(女声) → VAD → ASR → Chatbot → Agent TTS(男声) → 循环
"""
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Union

import numpy as np

from core.voice.audio_io import RingBuffer
from core.voice.vad import SimpleEnergyVAD, VADState
from core.voice.asr import ASRPipeline
from core.voice.tts import TTSManager

logger = logging.getLogger(__name__)


@dataclass
class SimulationTurn:
    """单轮模拟记录"""
    turn_id: int
    state_before: str = ""
    state_after: str = ""

    # 客户侧
    customer_text: str = ""
    customer_audio_file: str = ""
    customer_audio_duration: float = 0.0

    # ASR
    asr_text: str = ""
    asr_confidence: float = 0.0
    asr_time: float = 0.0
    asr_exact_match: bool = False
    asr_cer: float = 0.0

    # Agent侧
    agent_text: str = ""
    agent_audio_file: str = ""

    # 时序
    tts_time: float = 0.0
    chatbot_time: float = 0.0
    total_time: float = 0.0

    # 异常标记
    vad_dropped: bool = False
    tts_failed: bool = False
    asr_failed: bool = False


@dataclass
class SimulationReport:
    """汇总报告"""
    turns: List[SimulationTurn] = field(default_factory=list)
    persona: str = "cooperative"
    resistance_level: str = "medium"
    chat_group: str = "H2"
    session_id: str = ""
    artifacts_dir: str = ""

    total_turns: int = 0
    conversation_ended: bool = False
    final_state: str = ""
    committed_time: Optional[str] = None

    # ASR 指标
    asr_exact_match_rate: float = 0.0
    avg_cer: float = 0.0
    avg_asr_confidence: float = 0.0

    # 时延指标
    avg_tts_time: float = 0.0
    avg_asr_time: float = 0.0
    avg_chatbot_time: float = 0.0
    avg_round_trip_time: float = 0.0
    total_wall_time: float = 0.0

    # VAD 问题
    vad_dropped_count: int = 0
    tts_failed_count: int = 0


def _char_error_rate(reference: str, hypothesis: str) -> float:
    """字符错误率 (CER)"""
    ref = reference.lower().strip()
    hyp = hypothesis.lower().strip()
    if not ref:
        return 1.0 if hyp else 0.0

    # Levenshtein distance
    m, n = len(ref), len(hyp)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[m][n] / max(m, 1)


def _normalize_text(text: str) -> str:
    """标准化文本用于精确匹配"""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """
    加载音频文件为 float32 mono 16kHz 数组

    支持 WAV, MP3, FLAC 等格式。
    多声道自动转单声道，非16kHz自动重采样。
    """
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype='float32')
        if data.ndim > 1:
            data = data[:, 0]  # stereo → mono
        if sr != target_sr:
            from scipy.signal import resample
            n_samples = int(len(data) * target_sr / sr)
            data = resample(data, n_samples)
        return data.astype(np.float32)
    except ImportError:
        pass

    # fallback: scipy + ffmpeg
    try:
        from scipy.io import wavfile
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.run(
            ['ffmpeg', '-y', '-i', path, '-f', 'wav', '-acodec', 'pcm_f32le',
             '-ar', str(target_sr), '-ac', '1', tmp_path],
            capture_output=True, check=True,
        )
        sr, data = wavfile.read(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)
        return data.astype(np.float32)
    except Exception:
        pass

    raise RuntimeError(f"无法加载音频文件: {path}，请安装 soundfile 或 ffmpeg")


class CustomerVoiceSimulator:
    """
    客户语音模拟器

    使用文本模拟器生成印尼语回复，TTS合成为语音，注入ASR管线。

    Usage::

        sim = await CustomerVoiceSimulator.create(
            chatbot=bot,
            persona="resistant",
            resistance_level="high",
            chat_group="H2",
        )
        report = await sim.run(max_turns=15)
        print(f"ASR exact match: {report.asr_exact_match_rate:.1%}")
    """

    def __init__(
        self,
        chatbot,
        text_simulator,
        tts_manager: TTSManager,
        asr_pipeline: ASRPipeline,
        *,
        customer_voice: str = "id-ID-GadisNeural",
        agent_voice: str = "id-ID-ArdiNeural",
        customer_tts_engine: str = "edge_tts",
        agent_tts_engine: str = "edge_tts",
        persona: str = "cooperative",
        resistance_level: str = "medium",
        chat_group: str = "H2",
        sample_rate: int = 16000,
        block_size: int = 1600,
        energy_threshold: float = 0.005,
        realtime: bool = False,
        save_artifacts: bool = True,
        output_dir: str = "data/runs/voice_simulations",
    ):
        self._chatbot = chatbot
        self._simulator = text_simulator
        self._tts = tts_manager
        self._asr = asr_pipeline

        self.customer_voice = customer_voice
        self.agent_voice = agent_voice
        self.customer_tts_engine = customer_tts_engine
        self.agent_tts_engine = agent_tts_engine
        self.persona = persona
        self.resistance_level = resistance_level
        self.chat_group = chat_group
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.realtime = realtime
        self.save_artifacts = save_artifacts
        self.output_dir = Path(output_dir)

        self._vad = SimpleEnergyVAD(
            sample_rate=sample_rate,
            frame_duration_ms=int(block_size / sample_rate * 1000),
            energy_threshold=energy_threshold,
        )
        self._turns: List[SimulationTurn] = []
        self._push_count = 0
        self._session_id = str(uuid.uuid4())[:8]
        self._run_dir: Optional[Path] = None
        self._start_time: float = 0.0

    @classmethod
    async def create(
        cls,
        chatbot,
        *,
        persona: str = "cooperative",
        resistance_level: str = "medium",
        chat_group: str = "H2",
        customer_name: str = "Budi",
        asr_model_size: str = "small",
        customer_voice: str = "id-ID-GadisNeural",
        agent_voice: str = "id-ID-ArdiNeural",
        customer_tts_engine: str = "edge_tts",
        agent_tts_engine: str = "edge_tts",
        realtime: bool = False,
        save_artifacts: bool = True,
        output_dir: str = "data/runs/voice_simulations",
        **kwargs,
    ) -> "CustomerVoiceSimulator":
        """工厂方法：自动装配所有组件"""
        from core.simulator import RealCustomerSimulatorV2

        text_sim = RealCustomerSimulatorV2()
        tts = TTSManager()
        corrector = getattr(chatbot, "asr_corrector", None)

        # Use pre-warmed ASR pipeline if provided
        pre_warmed_asr = kwargs.pop("_asr_pipeline", None)
        if pre_warmed_asr is not None:
            asr = pre_warmed_asr
        else:
            asr = await ASRPipeline.create(model_size=asr_model_size, corrector=corrector)

        return cls(
            chatbot=chatbot,
            text_simulator=text_sim,
            tts_manager=tts,
            asr_pipeline=asr,
            persona=persona,
            resistance_level=resistance_level,
            chat_group=chat_group,
            customer_voice=customer_voice,
            agent_voice=agent_voice,
            customer_tts_engine=customer_tts_engine,
            agent_tts_engine=agent_tts_engine,
            realtime=realtime,
            save_artifacts=save_artifacts,
            output_dir=output_dir,
            **kwargs,
        )

    # ---- 状态映射 ----

    def _state_to_stage(self, state) -> str:
        """ChatState → simulator stage"""
        from core.chatbot import ChatState

        mapping = {
            ChatState.INIT: "greeting",
            ChatState.IDENTITY_VERIFY: "identity",
            ChatState.PURPOSE: "purpose",
            ChatState.ASK_TIME: "ask_time",
            ChatState.PUSH_FOR_TIME: "push",
            ChatState.COMMIT_TIME: "confirm",
            ChatState.CONFIRM_EXTENSION: "push",
            ChatState.HANDLE_OBJECTION: "negotiate",
            ChatState.HANDLE_BUSY: "push",
            ChatState.HANDLE_WRONG_NUMBER: "close",
            ChatState.CLOSE: "close",
            ChatState.FAILED: "close",
        }
        if hasattr(state, 'name'):
            return mapping.get(state, "greeting")
        return mapping.get(state, "greeting")

    # ---- 注入与VAD ----

    def _inject_and_vad_gate(self, audio: np.ndarray, silence_duration: float = 1.0) -> np.ndarray:
        """
        将语音逐帧注入RingBuffer，经VAD提取语音段

        模拟真实的 _listen_for_speech() 流程，但不依赖麦克风。
        """
        if len(audio) == 0:
            return np.array([], dtype=np.float32)

        # 前后加静音垫片
        pad_samples = int(0.3 * self.sample_rate)
        padded = np.concatenate([
            np.zeros(pad_samples, dtype=np.float32),
            audio,
            np.zeros(pad_samples, dtype=np.float32),
        ])

        silence_threshold = int(silence_duration * self.sample_rate)
        max_samples = int(60 * self.sample_rate)
        speech_buffer = np.zeros(max_samples, dtype=np.float32)
        speech_pos = 0
        silence_count = 0
        voice_detected = False

        self._vad.reset()

        for offset in range(0, len(padded), self.block_size):
            chunk = padded[offset:offset + self.block_size]
            if len(chunk) < self.block_size:
                break

            result = self._vad.process_frame(chunk)

            if result.state == VADState.VOICE:
                voice_detected = True

            if voice_detected:
                remaining = max_samples - speech_pos
                to_write = chunk[:remaining]
                speech_buffer[speech_pos:speech_pos + len(to_write)] = to_write
                speech_pos += len(to_write)

                if result.state == VADState.SILENCE:
                    silence_count += len(chunk)
                else:
                    silence_count = 0

                if silence_count >= silence_threshold:
                    break

                if speech_pos >= max_samples:
                    break

        if not voice_detected or speech_pos == 0:
            return np.array([], dtype=np.float32)

        return speech_buffer[:speech_pos].copy()

    # ---- 单轮 ----

    async def _run_single_turn(self, turn_id: int) -> Optional[SimulationTurn]:
        """执行单轮：客户语音 → ASR → Chatbot"""
        from core.chatbot import ChatState

        turn_start = time.time()
        state_before = self._chatbot.state

        turn = SimulationTurn(
            turn_id=turn_id,
            state_before=state_before.name if hasattr(state_before, 'name') else str(state_before),
        )

        # 1. 文本模拟器 → 客户文本
        stage = self._state_to_stage(self._chatbot.state)
        try:
            customer_text = self._simulator.generate_response(
                stage=stage,
                chat_group=self.chat_group,
                persona=self.persona,
                resistance_level=self.resistance_level,
                push_count=self._push_count,
            )
        except Exception as e:
            logger.warning(f"文本模拟器失败: {e}")
            customer_text = "Ya"

        turn.customer_text = customer_text
        self._push_count += 1

        # 2. TTS → 客户语音
        if customer_text.strip():
            tts_start = time.time()
            try:
                tts_result = await self._tts.synthesize(
                    customer_text,
                    voice=self.customer_voice,
                    engine=self.customer_tts_engine,
                )
                turn.tts_time = time.time() - tts_start

                if tts_result.success and tts_result.audio_file:
                    turn.customer_audio_file = tts_result.audio_file
                else:
                    turn.tts_failed = True
                    logger.warning(f"TTS失败: {tts_result.error_message}")
            except Exception as e:
                turn.tts_failed = True
                logger.warning(f"TTS异常: {e}")
        else:
            turn.tts_failed = True

        # 3. 加载音频 → VAD → ASR
        if turn.customer_audio_file and not turn.tts_failed:
            try:
                audio_data = _load_audio(turn.customer_audio_file, self.sample_rate)
                turn.customer_audio_duration = len(audio_data) / self.sample_rate
            except Exception as e:
                logger.warning(f"音频加载失败: {e}")
                audio_data = np.array([], dtype=np.float32)

            speech = self._inject_and_vad_gate(audio_data)
            if len(speech) == 0:
                turn.vad_dropped = True
                logger.info(f"VAD丢弃短语音: '{customer_text}' (时长={turn.customer_audio_duration:.2f}s)")

            if not turn.vad_dropped:
                asr_start = time.time()
                try:
                    asr_result = self._asr.transcribe(speech)
                    turn.asr_text = asr_result or ""
                except Exception as e:
                    turn.asr_failed = True
                    logger.warning(f"ASR失败: {e}")
                turn.asr_time = time.time() - asr_start

        # 4. ASR准确性
        if turn.asr_text and turn.customer_text:
            turn.asr_exact_match = _normalize_text(turn.customer_text) == _normalize_text(turn.asr_text)
            turn.asr_cer = _char_error_rate(turn.customer_text, turn.asr_text)

        # 5. Chatbot处理
        chatbot_start = time.time()
        input_text = turn.asr_text if turn.asr_text else ""
        try:
            agent_text, _ = await self._chatbot.process(
                customer_input=input_text if input_text else None,
                use_tts=False,
            )
        except Exception as e:
            logger.error(f"Chatbot处理异常: {e}")
            agent_text = ""
        turn.chatbot_time = time.time() - chatbot_start
        turn.agent_text = agent_text

        state_after = self._chatbot.state
        turn.state_after = state_after.name if hasattr(state_after, 'name') else str(state_after)
        turn.total_time = time.time() - turn_start

        # 实时模式：等待客户语音实际播放时长
        if self.realtime and turn.customer_audio_duration > 0:
            await asyncio.sleep(turn.customer_audio_duration)

        return turn

    # ---- 完整对话循环 ----

    async def run(self, max_turns: int = 20) -> SimulationReport:
        """运行完整模拟对话"""
        from core.chatbot import ChatState

        self._start_time = time.time()

        if self.save_artifacts:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self._run_dir = self.output_dir / f"{timestamp}_{self._session_id}"
            self._run_dir.mkdir(parents=True, exist_ok=True)

        # 初始: 让机器人先说第一句话
        first_msg, _ = await self._chatbot.process(use_tts=False)
        if first_msg and self.save_artifacts and self._run_dir:
            tts_result = await self._tts.synthesize(first_msg, voice=self.agent_voice, engine=self.agent_tts_engine)
            if tts_result.success and tts_result.audio_file:
                _copy_artifact(tts_result.audio_file, self._run_dir / "turn_00_agent_greeting.mp3")

        prev_state = self._chatbot.state
        stuck_count = 0
        tts_fail_streak = 0

        for turn_id in range(1, max_turns + 1):
            if self._chatbot.state in (ChatState.CLOSE, ChatState.FAILED):
                break

            turn = await self._run_single_turn(turn_id)
            if turn is None:
                break

            self._turns.append(turn)

            # 检测状态卡住
            if self._chatbot.state == prev_state:
                stuck_count += 1
                if stuck_count >= 3:
                    logger.warning(f"Chatbot状态卡在 {self._chatbot.state} 已 {stuck_count} 轮，强制退出")
                    break
            else:
                stuck_count = 0
            prev_state = self._chatbot.state

            # 检测TTS连续失败
            if turn.tts_failed:
                tts_fail_streak += 1
                if tts_fail_streak >= 3:
                    logger.warning("TTS连续失败3次，中止模拟")
                    break
            else:
                tts_fail_streak = 0

            # 保存artifacts
            if self.save_artifacts and self._run_dir:
                await self._save_turn_artifacts(turn)

        # 生成报告
        report = self.get_report()
        if self.save_artifacts and self._run_dir:
            self._write_report(report)
            self._write_transcript()

        return report

    async def run_streaming(self, max_turns: int = 20):
        """
        流式运行模拟，每轮 yield SimulationTurn

        用于 SSE 推送给 Web 前端。与 run() 不同：
        - 不在内部 collect turns
        - 每轮立即合成 agent TTS 并填充 agent_audio_file
        - 不写报告文件

        Usage::

            async for turn in sim.run_streaming(max_turns=10):
                send_to_frontend(turn)
        """
        from core.chatbot import ChatState

        self._start_time = time.time()

        prev_state = self._chatbot.state
        stuck_count = 0
        tts_fail_streak = 0

        for turn_id in range(1, max_turns + 1):
            if self._chatbot.state in (ChatState.CLOSE, ChatState.FAILED):
                break

            turn = await self._run_single_turn(turn_id)
            if turn is None:
                break

            self._turns.append(turn)

            # 合成 agent TTS（流式需要立即生成音频URL）
            if turn.agent_text:
                tts_result = await self._tts.synthesize(turn.agent_text, voice=self.agent_voice, engine=self.agent_tts_engine)
                if tts_result.success and tts_result.audio_file:
                    turn.agent_audio_file = str(tts_result.audio_file)

            yield turn

            # 检测卡状态
            if self._chatbot.state == prev_state:
                stuck_count += 1
                if stuck_count >= 3:
                    break
            else:
                stuck_count = 0
            prev_state = self._chatbot.state

            # 检测TTS连续失败
            if turn.tts_failed:
                tts_fail_streak += 1
                if tts_fail_streak >= 3:
                    break
            else:
                tts_fail_streak = 0

    async def _save_turn_artifacts(self, turn: SimulationTurn):
        """保存单轮音频和转录"""
        run_dir = self._run_dir
        tid = f"{turn.turn_id:02d}"

        # 客户音频
        if turn.customer_audio_file:
            _copy_artifact(turn.customer_audio_file, run_dir / f"turn_{tid}_customer.mp3")

        # Agent TTS
        if turn.agent_text:
            tts_result = await self._tts.synthesize(turn.agent_text, voice=self.agent_voice, engine=self.agent_tts_engine)
            if tts_result.success and tts_result.audio_file:
                turn.agent_audio_file = tts_result.audio_file
                _copy_artifact(tts_result.audio_file, run_dir / f"turn_{tid}_agent.mp3")

    def get_report(self) -> SimulationReport:
        """汇总指标"""
        turns = self._turns
        n = len(turns) or 1

        from core.chatbot import ChatState

        completed = [t for t in turns if t.customer_text.strip()]
        asr_turns = [t for t in completed if t.asr_text and not t.vad_dropped]

        report = SimulationReport(
            turns=turns,
            persona=self.persona,
            resistance_level=self.resistance_level,
            chat_group=self.chat_group,
            session_id=self._session_id,
            artifacts_dir=str(self._run_dir) if self._run_dir else "",
            total_turns=len(turns),
            conversation_ended=self._chatbot.state in (ChatState.CLOSE, ChatState.FAILED),
            final_state=self._chatbot.state.name if hasattr(self._chatbot.state, 'name') else str(self._chatbot.state),
            committed_time=self._chatbot.commit_time,
            total_wall_time=time.time() - self._start_time,
        )

        if asr_turns:
            report.asr_exact_match_rate = sum(1 for t in asr_turns if t.asr_exact_match) / len(asr_turns)
            report.avg_cer = float(np.mean([t.asr_cer for t in asr_turns]))
            confidences = [t.asr_confidence for t in asr_turns if t.asr_confidence > 0]
            report.avg_asr_confidence = float(np.mean(confidences)) if confidences else 0.0

        if completed:
            tts_times = [t.tts_time for t in completed if t.tts_time > 0]
            asr_times = [t.asr_time for t in completed if t.asr_time > 0]
            bot_times = [t.chatbot_time for t in completed if t.chatbot_time > 0]
            total_times = [t.total_time for t in completed if t.total_time > 0]
            report.avg_tts_time = float(np.mean(tts_times)) if tts_times else 0.0
            report.avg_asr_time = float(np.mean(asr_times)) if asr_times else 0.0
            report.avg_chatbot_time = float(np.mean(bot_times)) if bot_times else 0.0
            report.avg_round_trip_time = float(np.mean(total_times)) if total_times else 0.0

        report.vad_dropped_count = sum(1 for t in turns if t.vad_dropped)
        report.tts_failed_count = sum(1 for t in turns if t.tts_failed)

        return report

    def _write_report(self, report: SimulationReport):
        """写出汇总报告 JSON + MD"""
        run_dir = self._run_dir

        report_data = {
            "session_id": report.session_id,
            "persona": report.persona,
            "resistance_level": report.resistance_level,
            "chat_group": report.chat_group,
            "total_turns": report.total_turns,
            "conversation_ended": report.conversation_ended,
            "final_state": report.final_state,
            "committed_time": report.committed_time,
            "asr": {
                "exact_match_rate": round(report.asr_exact_match_rate, 4),
                "avg_cer": round(report.avg_cer, 4),
                "avg_confidence": round(report.avg_asr_confidence, 4),
            },
            "timing": {
                "avg_tts_s": round(report.avg_tts_time, 3),
                "avg_asr_s": round(report.avg_asr_time, 3),
                "avg_chatbot_s": round(report.avg_chatbot_time, 3),
                "avg_round_trip_s": round(report.avg_round_trip_time, 3),
                "total_wall_clock_s": round(report.total_wall_time, 1),
            },
            "issues": {
                "vad_dropped_count": report.vad_dropped_count,
                "tts_failed_count": report.tts_failed_count,
            },
        }
        with open(run_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        # Markdown报告
        md = _format_markdown_report(report)
        with open(run_dir / "report.md", "w", encoding="utf-8") as f:
            f.write(md)

    def _write_transcript(self):
        """写出完整转录 JSON"""
        run_dir = self._run_dir
        transcript = []
        for t in self._turns:
            transcript.append({
                "turn_id": t.turn_id,
                "state_before": t.state_before,
                "state_after": t.state_after,
                "customer_text": t.customer_text,
                "asr_text": t.asr_text,
                "asr_exact_match": t.asr_exact_match,
                "asr_cer": round(t.asr_cer, 4),
                "agent_text": t.agent_text,
                "timing": {
                    "tts_s": round(t.tts_time, 3),
                    "asr_s": round(t.asr_time, 3),
                    "chatbot_s": round(t.chatbot_time, 3),
                    "total_s": round(t.total_time, 3),
                },
                "issues": {
                    "vad_dropped": t.vad_dropped,
                    "tts_failed": t.tts_failed,
                    "asr_failed": t.asr_failed,
                },
            })
        with open(run_dir / "transcript.json", "w", encoding="utf-8") as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)


def _copy_artifact(src: str, dst: Path):
    """复制音频artifact"""
    import shutil
    src_path = Path(src)
    if src_path.exists():
        shutil.copy2(src_path, dst)


def _format_markdown_report(report: SimulationReport) -> str:
    """格式化 Markdown 报告"""
    lines = [
        "# Voice Simulation Report",
        "",
        f"**Session:** `{report.session_id}` | **Group:** {report.chat_group}",
        f"**Persona:** {report.persona} | **Resistance:** {report.resistance_level}",
        "",
        "## Conversation Outcome",
        f"- Turns: {report.total_turns}",
        f"- Ended: {'Yes' if report.conversation_ended else 'No'}",
        f"- Final state: `{report.final_state}`",
        f"- Committed time: {report.committed_time or 'N/A'}",
        "",
        "## ASR Accuracy",
        f"| Exact Match | Avg CER | Avg Confidence |",
        f"|---|---|---|",
        f"| {report.asr_exact_match_rate:.1%} | {report.avg_cer:.3f} | {report.avg_asr_confidence:.3f} |",
        "",
        "## Timing",
        f"| TTS | ASR | Chatbot | Round Trip | Wall Clock |",
        f"|---|---|---|---|---|",
        f"| {report.avg_tts_time:.2f}s | {report.avg_asr_time:.2f}s | {report.avg_chatbot_time:.2f}s | {report.avg_round_trip_time:.2f}s | {report.total_wall_time:.0f}s |",
        "",
        "## Issues",
        f"- VAD dropped: {report.vad_dropped_count}",
        f"- TTS failed: {report.tts_failed_count}",
        "",
        "## Turns",
    ]

    for t in report.turns:
        match_icon = "✓" if t.asr_exact_match else "✗"
        issues = []
        if t.vad_dropped:
            issues.append("VAD-DROP")
        if t.tts_failed:
            issues.append("TTS-FAIL")
        issue_str = f" [{', '.join(issues)}]" if issues else ""

        lines.append(f"### Turn {t.turn_id} ({t.state_before} → {t.state_after}){issue_str}")
        lines.append(f"- Customer: *{t.customer_text}*")
        lines.append(f"- ASR {match_icon}: {t.asr_text or '(empty)'} (CER: {t.asr_cer:.3f})")
        lines.append(f"- Agent: {t.agent_text}")
        lines.append(f"- Time: TTS={t.tts_time:.2f}s ASR={t.asr_time:.2f}s Chatbot={t.chatbot_time:.2f}s")
        lines.append("")

    return "\n".join(lines)
