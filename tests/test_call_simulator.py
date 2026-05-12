"""CallSimulator 单元测试 — 自动仿真"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pytest
import asyncio
from core.voice.call_simulator import CallSimulator


class FakeTTS:
    async def synthesize(self, text, output_file=None, voice=None, engine=None, **kwargs):
        from core.voice.tts import TTSResult
        return TTSResult(text=text, audio_data=np.zeros(8000, dtype=np.float32), audio_file=None, success=True, engine_name="fake")


class FakeASR:
    is_available = True
    def transcribe(self, audio): return "Ya"
    async def transcribe_async(self, audio): return "Ya"


class FakeBot:
    def __init__(self):
        self.turns = 0
        self.commit_time = None
        self._state = _FakeInitState()

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    async def process(self, customer_input=None, use_tts=False):
        self.turns += 1
        if self.turns >= 3:
            self._state = _FakeCloseState()
            return "Terima kasih.", None
        return "Baik.", None


class _FakeInitState:
    name = "INIT"

class _FakeCloseState:
    name = "CLOSE"


class FakeTextSimulator:
    def generate_response(self, stage, chat_group, persona, resistance_level, push_count):
        return "Ya, besok jam 5"

    def get_current_stage_and_response(self, chat_group, push_count):
        return "ask_time", "Besok jam 5"


@pytest.mark.asyncio
async def test_call_simulator_runs_to_completion():
    """CallSimulator 运行到结束"""
    sim = CallSimulator(
        chatbot=FakeBot(),
        text_simulator=FakeTextSimulator(),
        tts_manager=FakeTTS(),
        asr_pipeline=FakeASR(),
        persona="cooperative",
        resistance_level="medium",
        chat_group="H2",
        max_turns=5,
        save_artifacts=False,
    )
    report = await sim.run()
    assert report.total_turns > 0
    assert report.conversation_ended


@pytest.mark.asyncio
async def test_call_simulator_create_factory():
    """CallSimulator.create() 工厂方法"""
    bot = FakeBot()
    sim = await CallSimulator.create(
        chatbot=bot,
        persona="cooperative",
        resistance_level="medium",
        chat_group="H2",
        asr_model_size="tiny",
        save_artifacts=False,
        _asr_pipeline=FakeASR(),
    )
    assert sim.persona == "cooperative"
    assert sim.chat_group == "H2"


def test_call_simulator_initial_state():
    """初始属性正确"""
    sim = CallSimulator(
        chatbot=FakeBot(),
        text_simulator=FakeTextSimulator(),
        tts_manager=FakeTTS(),
        asr_pipeline=FakeASR(),
        persona="resistant",
        resistance_level="high",
        chat_group="S0",
        max_turns=10,
        save_artifacts=False,
    )
    assert sim.persona == "resistant"
    assert sim.resistance_level == "high"
    assert sim.max_turns == 10
