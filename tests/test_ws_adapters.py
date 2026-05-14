"""WebSocket 适配器测试"""
import asyncio
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def _make_source(**kw):
    from src.core.voice.ws_adapters import WebSocketAudioSource
    return WebSocketAudioSource(**kw)


class TestWebSocketAudioSource:
    @pytest.mark.asyncio
    async def test_feed_and_read_chunk(self):
        source = await _make_source(sample_rate=16000, block_size=1600)
        chunk = np.ones(1600, dtype=np.float32) * 0.5
        source.feed_chunk(chunk)
        await source.start()
        result = await source.read_chunk()
        assert result is not None
        assert len(result) == 1600
        assert result.dtype == np.float32

    def test_queue_overflow_handled(self):
        source = asyncio.run(_make_source(sample_rate=16000, block_size=1600))
        for _ in range(250):
            source.feed_chunk(np.zeros(1600, dtype=np.float32))
        assert source.overflow_count > 0

    @pytest.mark.asyncio
    async def test_current_rms_calculation(self):
        source = await _make_source(sample_rate=16000, block_size=1600)
        await source.start()
        chunk = np.ones(1600, dtype=np.float32)
        source.feed_chunk(chunk)
        await source.read_chunk()
        rms = source.current_rms()
        assert rms > 0.9

    @pytest.mark.asyncio
    async def test_read_returns_none_when_stopped(self):
        source = await _make_source()
        await source.start()
        await source.stop()
        result = await source.read_chunk()
        assert result is None

    @pytest.mark.asyncio
    async def test_bytes_input_converted_to_float32(self):
        source = await _make_source(sample_rate=16000, block_size=800)
        source._block_size = 800
        data = np.ones(800, dtype=np.float32)
        source.feed_chunk(data.tobytes())
        await source.start()
        result = await source.read_chunk()
        assert result is not None
        assert result.dtype == np.float32
        assert len(result) == 800

    @pytest.mark.asyncio
    async def test_stop_clears_queue(self):
        source = await _make_source(sample_rate=16000, block_size=1600)
        await source.start()
        source.feed_chunk(np.ones(1600, dtype=np.float32))
        await source.stop()
        assert source._queue.empty()

    @pytest.mark.asyncio
    async def test_short_chunk_padded_to_block_size(self):
        source = await _make_source(sample_rate=16000, block_size=1600)
        data = np.ones(800, dtype=np.float32)
        source.feed_chunk(data)
        await source.start()
        result = await source.read_chunk()
        assert result is not None
        assert len(result) == 1600

    @pytest.mark.asyncio
    async def test_long_chunk_truncated_to_block_size(self):
        source = await _make_source(sample_rate=16000, block_size=1600)
        data = np.ones(3200, dtype=np.float32)
        source.feed_chunk(data)
        await source.start()
        result = await source.read_chunk()
        assert result is not None
        assert len(result) == 1600

    @pytest.mark.asyncio
    async def test_read_timeout_returns_none(self):
        """超时无数据时返回 None（而非伪静音），避免污染 VAD 状态"""
        source = await _make_source(sample_rate=16000, block_size=1600)
        await source.start()
        result = await source.read_chunk()
        assert result is None

    @pytest.mark.asyncio
    async def test_recent_samples_limited(self):
        source = await _make_source(sample_rate=16000, block_size=1600)
        await source.start()
        for _ in range(50):
            source.feed_chunk(np.ones(1600, dtype=np.float32))
            await source.read_chunk()
        assert len(source._recent_samples) <= int(0.3 * 16000)


class TestWebSocketAudioOutput:
    def test_init_with_source(self):
        from src.core.voice.ws_adapters import WebSocketAudioSource, WebSocketAudioOutput
        source = asyncio.run(_make_source())
        output = WebSocketAudioOutput(source)
        assert output._barge_in_threshold == 0.02

    def test_set_send_chunk(self):
        from src.core.voice.ws_adapters import WebSocketAudioSource, WebSocketAudioOutput
        source = asyncio.run(_make_source())

        sent = []
        async def cb(data, sr):
            sent.append((len(data), sr))

        output = WebSocketAudioOutput(source, send_chunk=cb)
        assert output._send_chunk is cb

    @pytest.mark.asyncio
    async def test_play_sends_chunks_to_callback(self):
        from src.core.voice.ws_adapters import WebSocketAudioSource, WebSocketAudioOutput
        source = await _make_source()
        await source.start()

        sent_chunks = []
        async def cb(data, sr):
            sent_chunks.append(len(data))
            source.feed_chunk(np.zeros(1600, dtype=np.float32))

        output = WebSocketAudioOutput(source, send_chunk=cb, barge_in_threshold=0.99)
        output._chunk_duration = 0.01

        audio = np.ones(4800, dtype=np.float32) * 0.1
        result = await output.speak(audio)
        assert len(sent_chunks) > 0

    @pytest.mark.asyncio
    async def test_play_fails_when_no_send_chunk(self):
        from src.core.voice.ws_adapters import WebSocketAudioSource, WebSocketAudioOutput, PlaybackResult
        source = await _make_source()
        output = WebSocketAudioOutput(source)
        result = await output.speak(np.ones(1600, dtype=np.float32))
        assert result == PlaybackResult.FAILED

    def test_stop_sets_flag(self):
        from src.core.voice.ws_adapters import WebSocketAudioSource, WebSocketAudioOutput
        source = asyncio.run(_make_source())
        output = WebSocketAudioOutput(source)
        output.stop()
        assert output._stop_requested is True
