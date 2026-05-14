#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自动仿真模式 — 文本模拟器 → TTS → ASR → Bot，替代 CustomerVoiceSimulator"""
import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from core.voice.customer_simulator import SimulationReport, SimulationTurn

logger = logging.getLogger(__name__)


class CallSimulator:
    """自动仿真。用文本模拟器生成客户回复 → TTS → ASR → Bot。

    与 CustomerVoiceSimulator 的关键区别:
    - 不自己实现管线，而是编排独立的 TTS/ASR/Bot 组件
    - 不再有 _inject_and_vad_gate / _run_single_turn
    - 客户回复逐轮生成和评估
    """

    def __init__(
        self,
        chatbot,
        text_simulator,
        tts_manager,
        asr_pipeline,
        *,
        persona: str = "cooperative",
        resistance_level: str = "medium",
        chat_group: str = "H2",
        max_turns: int = 20,
        customer_voice: str = "id-ID-GadisNeural",
        agent_voice: str = "id-ID-ArdiNeural",
        customer_tts_engine: str = "edge_tts",
        agent_tts_engine: str = "edge_tts",
        sample_rate: int = 16000,
        block_size: int = 1600,
        save_artifacts: bool = True,
        output_dir: str = "data/runs/voice_simulations",
        realtime: bool = False,
    ):
        self._chatbot = chatbot
        self._text_sim = text_simulator
        self._tts = tts_manager
        self._asr = asr_pipeline

        self.persona = persona
        self.resistance_level = resistance_level
        self.chat_group = chat_group
        self.max_turns = max_turns
        self.customer_voice = customer_voice
        self.agent_voice = agent_voice
        self.customer_tts_engine = customer_tts_engine
        self.agent_tts_engine = agent_tts_engine
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.save_artifacts = save_artifacts
        self.output_dir = Path(output_dir)
        self.realtime = realtime

        self._push_count = 0
        self._session_id = str(uuid.uuid4())[:8]
        self._turns: list[SimulationTurn] = []
        self._start_time: float = 0.0
        self._run_dir: Optional[Path] = None

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
    ) -> "CallSimulator":
        from core.simulator import RealCustomerSimulatorV2
        from core.voice.tts import TTSManager

        text_sim = RealCustomerSimulatorV2()
        tts = TTSManager()

        pre_warmed = kwargs.pop("_asr_pipeline", None)
        if pre_warmed is not None:
            asr = pre_warmed
        else:
            from core.voice.asr import ASRPipeline
            corrector = getattr(chatbot, "asr_corrector", None)
            asr = await ASRPipeline.create(model_size=asr_model_size, corrector=corrector)

        return cls(
            chatbot=chatbot,
            text_simulator=text_sim,
            tts_manager=tts,
            asr_pipeline=asr,
            persona=persona,
            resistance_level=resistance_level,
            chat_group=chat_group,
            max_turns=kwargs.pop("max_turns", 20),
            customer_voice=customer_voice,
            agent_voice=agent_voice,
            customer_tts_engine=customer_tts_engine,
            agent_tts_engine=agent_tts_engine,
            sample_rate=kwargs.pop("sample_rate", 16000),
            block_size=kwargs.pop("block_size", 1600),
            realtime=realtime,
            save_artifacts=save_artifacts,
            output_dir=output_dir,
            **kwargs,
        )

    async def run(self, max_turns: int | None = None) -> SimulationReport:
        max_turns = max_turns or self.max_turns
        self._start_time = time.time()

        if self.save_artifacts:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self._run_dir = self.output_dir / f"{timestamp}_{self._session_id}"
            self._run_dir.mkdir(parents=True, exist_ok=True)

        from core.chatbot import ChatState

        self._chatbot.state = ChatState.INIT

        # 初始: bot 第一句话
        first_msg, _ = await self._chatbot.process(use_tts=False)
        prev_state = self._chatbot.state
        stuck_count = 0

        for turn_id in range(1, max_turns + 1):
            if self._bot_is_finished():
                break

            turn = SimulationTurn(
                turn_id=turn_id,
                state_before=getattr(self._chatbot.state, 'name', str(self._chatbot.state)),
            )

            # 1. 文本模拟器 → 客户文本
            stage = self._state_to_stage(self._chatbot.state)
            try:
                customer_text = self._text_sim.generate_response(
                    stage=stage,
                    chat_group=self.chat_group,
                    persona=self.persona,
                    resistance_level=self.resistance_level,
                    push_count=self._push_count,
                )
            except Exception:
                customer_text = "Ya"
            turn.customer_text = customer_text
            self._push_count += 1

            # 2. TTS → 客户音频
            customer_audio = np.array([], dtype=np.float32)
            if customer_text.strip():
                try:
                    tts_result = await self._tts.synthesize(
                        customer_text,
                        voice=self.customer_voice,
                        engine=self.customer_tts_engine,
                    )
                    if tts_result.success:
                        if tts_result.audio_file:
                            turn.customer_audio_file = tts_result.audio_file
                            customer_audio = _load_audio_mono(tts_result.audio_file, self.sample_rate)
                        elif tts_result.audio_data is not None:
                            if isinstance(tts_result.audio_data, bytes):
                                customer_audio = np.frombuffer(tts_result.audio_data, dtype=np.float32)
                            else:
                                customer_audio = tts_result.audio_data
                    else:
                        turn.tts_failed = True
                except Exception:
                    turn.tts_failed = True
            else:
                turn.tts_failed = True

            # 3. ASR
            if self._asr and self._asr.is_available and len(customer_audio) > 0:
                try:
                    turn.asr_text = self._asr.transcribe(customer_audio)
                except Exception:
                    turn.asr_failed = True
            else:
                turn.vad_dropped = True

            # 4. Bot 处理
            input_text = turn.asr_text if turn.asr_text else ""
            try:
                agent_text, _ = await self._chatbot.process(
                    customer_input=input_text if input_text else None,
                    use_tts=False,
                )
            except Exception:
                agent_text = ""
            turn.agent_text = agent_text
            turn.state_after = getattr(self._chatbot.state, 'name', str(self._chatbot.state))

            self._turns.append(turn)

            # 卡状态检测
            if self._chatbot.state == prev_state:
                stuck_count += 1
                if stuck_count >= 3:
                    break
            else:
                stuck_count = 0
            prev_state = self._chatbot.state

        return self._build_report()

    def _bot_is_finished(self) -> bool:
        state_name = getattr(self._chatbot.state, 'name', str(self._chatbot.state))
        return state_name in ("CLOSE", "FAILED")

    def _state_to_stage(self, state) -> str:
        mapping = {
            "INIT": "greeting",
            "GREETING": "greeting",
            "IDENTITY_VERIFY": "identity",
            "PURPOSE": "purpose",
            "ASK_TIME": "ask_time",
            "PUSH_FOR_TIME": "push",
            "COMMIT_TIME": "confirm",
            "CONFIRM_EXTENSION": "push",
            "HANDLE_OBJECTION": "negotiate",
            "HANDLE_BUSY": "push",
            "HANDLE_WRONG_NUMBER": "close",
            "CLOSE": "close",
            "FAILED": "close",
        }
        state_name = getattr(state, 'name', str(state))
        return mapping.get(state_name, "greeting")

    def _build_report(self) -> SimulationReport:
        turns = self._turns
        asr_turns = [t for t in turns if t.asr_text and not t.vad_dropped]
        report = SimulationReport(
            turns=turns,
            persona=self.persona,
            resistance_level=self.resistance_level,
            chat_group=self.chat_group,
            session_id=self._session_id,
            artifacts_dir=str(self._run_dir) if self._run_dir else "",
            total_turns=len(turns),
            conversation_ended=self._bot_is_finished(),
            final_state=getattr(self._chatbot.state, 'name', str(self._chatbot.state)),
            committed_time=self._chatbot.commit_time,
            total_wall_time=time.time() - self._start_time,
        )
        if asr_turns:
            report.asr_exact_match_rate = sum(1 for t in asr_turns if t.asr_exact_match) / len(asr_turns)
            report.avg_cer = float(np.mean([t.asr_cer for t in asr_turns]))
        report.vad_dropped_count = sum(1 for t in turns if t.vad_dropped)
        report.tts_failed_count = sum(1 for t in turns if t.tts_failed)
        return report

    def get_report(self) -> SimulationReport:
        """返回当前仿真报告（公开接口）"""
        return self._build_report()

    async def run_streaming(self, max_turns: int | None = None):
        """流式运行 — 每轮 yield SimulationTurn。用于 SSE 推送给 Web 前端。"""
        self._start_time = time.time()
        from core.chatbot import ChatState

        self._chatbot.state = ChatState.INIT
        prev_state = self._chatbot.state
        stuck_count = 0

        for turn_id in range(1, (max_turns or self.max_turns) + 1):
            if self._bot_is_finished():
                break

            turn = SimulationTurn(
                turn_id=turn_id,
                state_before=getattr(self._chatbot.state, 'name', str(self._chatbot.state)),
            )
            stage = self._state_to_stage(self._chatbot.state)
            customer_text = self._text_sim.generate_response(
                stage=stage, chat_group=self.chat_group,
                persona=self.persona, resistance_level=self.resistance_level,
                push_count=self._push_count,
            )
            turn.customer_text = customer_text
            self._push_count += 1

            if customer_text.strip():
                tts_result = await self._tts.synthesize(
                    customer_text, voice=self.customer_voice, engine=self.customer_tts_engine,
                )
                if tts_result.success and tts_result.audio_file:
                    turn.customer_audio_file = tts_result.audio_file
                    audio = _load_audio_mono(tts_result.audio_file, self.sample_rate)
                    if self._asr.is_available:
                        turn.asr_text = self._asr.transcribe(audio)

            agent_text, _ = await self._chatbot.process(
                customer_input=turn.asr_text if turn.asr_text else None,
                use_tts=False,
            )
            turn.agent_text = agent_text
            turn.state_after = getattr(self._chatbot.state, 'name', str(self._chatbot.state))
            self._turns.append(turn)
            yield turn

            if self._chatbot.state == prev_state:
                stuck_count += 1
                if stuck_count >= 3:
                    break
            else:
                stuck_count = 0
            prev_state = self._chatbot.state


def _load_audio_mono(path: str, target_sr: int) -> np.ndarray:
    """加载音频文件为 float32 mono 数组"""
    import soundfile as sf
    data, sr = sf.read(path, dtype='float32')
    if data.ndim > 1:
        data = data[:, 0]
    if sr != target_sr:
        from scipy.signal import resample
        n_samples = int(len(data) * target_sr / sr)
        data = resample(data, n_samples)
    return data.astype(np.float32)
