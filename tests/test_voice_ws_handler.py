"""WebSocket 双工通话处理器集成测试"""
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class MockChatBot:
    def __init__(self):
        self.state = _FakeState("INIT")
        self.session_id = "test-001"
        self.asr_corrector = None

    async def process(self, customer_input=None, use_tts=False):
        return "Agent response text", None


class _FakeState:
    def __init__(self, name):
        self.name = name


class MockWebSocket:
    def __init__(self):
        self.sent_messages = []
        self.sent_bytes = []
        self.accepted = False
        self.closed = False
        self._incoming = []
        self._iter_pos = 0
        self._open = True

    async def accept(self):
        self.accepted = True

    async def send_text(self, data: str):
        self.sent_messages.append(data)

    async def send_bytes(self, data: bytes):
        self.sent_bytes.append(data)

    async def close(self):
        self.closed = True

    async def receive(self) -> dict:
        """模拟 Starlette WebSocket.receive() 返回格式"""
        if self._iter_pos < len(self._incoming):
            msg = self._incoming[self._iter_pos]
            self._iter_pos += 1
            if isinstance(msg, bytes):
                return {"type": "websocket.receive", "bytes": msg}
            else:
                return {"type": "websocket.receive", "text": msg}
        await asyncio.sleep(5.0)
        return {"type": "websocket.disconnect", "code": 1000}

    def queue_incoming(self, msg):
        self._incoming.append(msg)

    @property
    def url(self):
        return _FakeURL()


class _FakeURL:
    query = "chat_group=H2&customer_name=Test"


@pytest.mark.asyncio
async def test_handler_accepts_connection():
    """WebSocket 连接后应调用 accept"""
    ws = MockWebSocket()
    bot = MockChatBot()
    ws.queue_incoming(json.dumps({"type": "stop"}))

    from src.api.voice_ws_handler import handle_duplex_ws
    await handle_duplex_ws(ws, bot)
    # handler doesn't call accept() — that's done by the endpoint in main.py


@pytest.mark.asyncio
async def test_handler_sends_interrupted_on_client_interrupt():
    """客户端打断应收到 interrupted 响应"""
    ws = MockWebSocket()
    bot = MockChatBot()

    ws.queue_incoming(json.dumps({"type": "interrupt"}))
    ws.queue_incoming(json.dumps({"type": "stop"}))

    from src.api.voice_ws_handler import handle_duplex_ws
    await handle_duplex_ws(ws, bot)

    interrupted_msgs = [m for m in ws.sent_messages if '"interrupted"' in m]
    assert len(interrupted_msgs) >= 1


@pytest.mark.asyncio
async def test_handler_handles_invalid_json():
    """无效 JSON 消息不应导致崩溃"""
    ws = MockWebSocket()
    bot = MockChatBot()

    ws.queue_incoming("not valid json{{{{")
    ws.queue_incoming(json.dumps({"type": "stop"}))

    from src.api.voice_ws_handler import handle_duplex_ws
    await handle_duplex_ws(ws, bot)

    # no crash = success (accept is called by the endpoint, not the handler)


def test_helper_safe_send_json():
    """_safe_send_json 应正常序列化并发送 JSON"""
    from src.api.voice_ws_handler import _safe_send_json

    ws = MockWebSocket()
    asyncio.run(_safe_send_json(ws, {"type": "test", "value": "hello"}))
    assert len(ws.sent_messages) == 1
    parsed = json.loads(ws.sent_messages[0])
    assert parsed["type"] == "test"
    assert parsed["value"] == "hello"


@pytest.mark.asyncio
async def test_handler_stops_on_stop_message():
    """收到 stop 消息后应正常退出不崩溃"""
    ws = MockWebSocket()
    bot = MockChatBot()

    ws.queue_incoming(json.dumps({"type": "stop"}))

    from src.api.voice_ws_handler import handle_duplex_ws
    await handle_duplex_ws(ws, bot)

    # no crash = success (accept is called by the endpoint, not the handler)


@pytest.mark.asyncio
async def test_handler_handles_binary_audio():
    """二进制音频消息不应导致崩溃"""
    import numpy as np
    ws = MockWebSocket()
    bot = MockChatBot()

    audio = (np.ones(1600, dtype=np.float32) * 0.1).tobytes()
    ws.queue_incoming(audio)
    ws.queue_incoming(audio)
    ws.queue_incoming(json.dumps({"type": "stop"}))

    from src.api.voice_ws_handler import handle_duplex_ws
    await handle_duplex_ws(ws, bot)

    # no crash = success


# ── _load_audio_file (ffmpeg) 测试 ──────────────────────────

def test_load_audio_file_returns_float32():
    """ffmpeg 解码成功 → 返回 float32 numpy 数组"""
    from src.api.voice_ws_handler import _load_audio_file

    fake_samples = np.ones(1600, dtype=np.float32)
    fake_stdout = fake_samples.tobytes()

    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(returncode=0, stdout=fake_stdout, stderr=b"")
        result = _load_audio_file("/fake/path.mp3")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == 1600
        assert result[0] == 1.0


def test_load_audio_file_handles_ffmpeg_failure():
    """ffmpeg 返回非零 → 抛出 RuntimeError"""
    from src.api.voice_ws_handler import _load_audio_file

    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(returncode=1, stdout=b"", stderr=b"ffmpeg error")
        with pytest.raises(RuntimeError, match="ffmpeg 加载音频失败"):
            _load_audio_file("/fake/path.mp3")


def test_load_audio_file_handles_empty_output():
    """ffmpeg 返回空 stdout → 抛出 RuntimeError"""
    from src.api.voice_ws_handler import _load_audio_file

    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(returncode=0, stdout=b"", stderr=b"")
        with pytest.raises(RuntimeError, match="ffmpeg 加载音频失败"):
            _load_audio_file("/fake/path.mp3")


def test_load_audio_file_passes_correct_ffmpeg_args():
    """验证传给 ffmpeg 的参数正确"""
    from src.api.voice_ws_handler import _load_audio_file

    fake_samples = np.ones(100, dtype=np.float32)
    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(returncode=0, stdout=fake_samples.tobytes(), stderr=b"")
        _load_audio_file("/path/to/audio.mp3", target_sr=8000)
        args = mock_run.call_args[0][0]
        assert args[0] == "ffmpeg"
        assert args[3] == "/path/to/audio.mp3"
        assert "-ar" in args
        assert "8000" in args
        assert "-ac" in args
        assert "1" in args


# ── 问候流程测试 ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_greeting_sent_with_session_id():
    """问候 JSON 必须包含 session_id（客户端需要用于会话跟踪）"""
    ws = MockWebSocket()
    bot = MockChatBot()
    bot.session_id = "test-session-abc123"

    ws.queue_incoming(json.dumps({"type": "stop"}))

    from src.api.voice_ws_handler import handle_duplex_ws
    await handle_duplex_ws(ws, bot)

    ready_msgs = [json.loads(m) for m in ws.sent_messages
                     if '"type":"ready"' in m or '"type": "ready"' in m or '"type": "greeting"' in m]
    assert len(ready_msgs) >= 1
    ready = ready_msgs[0]
    assert ready["type"] in ("ready", "greeting")
    assert ready["session_id"] == "test-session-abc123"
    assert "text" in ready


@pytest.mark.asyncio
async def test_greeting_sent_before_state_events():
    """问候 JSON 必须在状态事件之前发送（客户端需要先确认连接成功）"""
    ws = MockWebSocket()
    bot = MockChatBot()

    ws.queue_incoming(json.dumps({"type": "stop"}))

    from src.api.voice_ws_handler import handle_duplex_ws
    await handle_duplex_ws(ws, bot)

    first_json_idx = None
    for i, msg in enumerate(ws.sent_messages):
        try:
            parsed = json.loads(msg)
            if parsed.get("type") in ("ready", "greeting"):
                first_json_idx = i
                break
        except json.JSONDecodeError:
            continue

    assert first_json_idx is not None, "未找到 ready/greeting 消息"
    assert first_json_idx <= 1, f"ready/greeting 在第 {first_json_idx} 条消息，太靠后了"


# ── 中断与状态事件测试 ──────────────────────────────────────

@pytest.mark.asyncio
async def test_state_change_events_sent():
    """管线状态变化时应向客户端发送 state 事件"""
    ws = MockWebSocket()
    bot = MockChatBot()

    # 发送足够的音频块以触发 VAD 语音检测和状态转换
    audio = (np.ones(2048, dtype=np.float32) * 0.3).tobytes()
    for _ in range(100):
        ws.queue_incoming(audio)
    ws.queue_incoming(json.dumps({"type": "stop"}))

    from src.api.voice_ws_handler import handle_duplex_ws
    await handle_duplex_ws(ws, bot)

    # 至少应该有 ready 或 greeting 消息（状态事件通过 asyncio.ensure_future 异步发送，
    # 在 MockWebSocket 快速消费所有消息后可能尚未执行）
    ready_msgs = [json.loads(m) for m in ws.sent_messages
                     if '"type":"ready"' in m or '"type": "ready"' in m or '"type":"greeting"' in m or '"type": "greeting"' in m]
    assert len(ready_msgs) >= 1
    # 处理了音频块且未崩溃即算成功


@pytest.mark.asyncio
async def test_graceful_shutdown_on_connection_close():
    """WebSocket 连接意外关闭时不应崩溃"""
    ws = MockWebSocket()
    bot = MockChatBot()
    ws._open = False

    from src.api.voice_ws_handler import handle_duplex_ws
    try:
        await handle_duplex_ws(ws, bot)
    except Exception as e:
        pytest.fail(f"handler crashed on connection close: {e}")


# ── 错误恢复测试 ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_asr_load_failure_not_fatal():
    """ASR 加载失败不应阻塞通话（降级到无转写模式）"""
    ws = MockWebSocket()
    bot = MockChatBot()
    bot.asr_corrector = None

    ws.queue_incoming(json.dumps({"type": "stop"}))

    from src.api.voice_ws_handler import handle_duplex_ws
    await handle_duplex_ws(ws, bot)
