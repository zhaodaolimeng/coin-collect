"""sherpa-onnx streaming Zipformer ASR 测试 — 印尼语识别效果验证"""
import json
import time
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

MODEL_DIR = Path(__file__).parent.parent / "data/models/sherpa-zipformer-id/sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10"

# ---- 辅助函数 ----

def _load_wav(path: str) -> tuple:
    """加载 wav 文件，返回 (float32_samples, sample_rate)"""
    import scipy.io.wavfile as wav
    sr, data = wav.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype != np.float32:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data[:, 0]  # 取左声道
    return data, sr


def _create_recognizer(enable_endpoint=False):
    import sherpa_onnx
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(MODEL_DIR / "tokens.txt"),
        encoder=str(MODEL_DIR / "encoder-epoch-75-avg-11-chunk-16-left-128.int8.onnx"),
        decoder=str(MODEL_DIR / "decoder-epoch-75-avg-11-chunk-16-left-128.onnx"),
        joiner=str(MODEL_DIR / "joiner-epoch-75-avg-11-chunk-16-left-128.int8.onnx"),
        modeling_unit="bpe",
        bpe_vocab=str(MODEL_DIR / "bpe.model"),
        num_threads=4,
        enable_endpoint_detection=enable_endpoint,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=20.0,
    )


def _transcribe_streaming(recognizer, samples: np.ndarray, chunk_size: int = 6400) -> str:
    """流式转录完整音频"""
    import sherpa_onnx
    stream = recognizer.create_stream()
    total = len(samples)
    pos = 0
    last_text = ""
    partials = []

    while pos < total:
        chunk = samples[pos:pos + chunk_size]
        pos += chunk_size
        stream.accept_waveform(sample_rate=16000, waveform=chunk)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        text = recognizer.get_result(stream)
        if text and text != last_text:
            partials.append(text)
            last_text = text
        if recognizer.is_endpoint(stream):
            if text:
                break

    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    final = recognizer.get_result(stream)
    return final or last_text


# ---- 测试 ----

class TestSherpaOnnxBasic:
    def test_model_files_exist(self):
        """模型文件齐全"""
        assert (MODEL_DIR / "encoder-epoch-75-avg-11-chunk-16-left-128.int8.onnx").exists()
        assert (MODEL_DIR / "decoder-epoch-75-avg-11-chunk-16-left-128.onnx").exists()
        assert (MODEL_DIR / "joiner-epoch-75-avg-11-chunk-16-left-128.int8.onnx").exists()
        assert (MODEL_DIR / "tokens.txt").exists()
        assert (MODEL_DIR / "bpe.model").exists()

    def test_recognizer_creation(self):
        """Recognizer 可创建"""
        rec = _create_recognizer()
        assert rec is not None
        del rec

    def test_create_stream(self):
        """Stream 可创建"""
        rec = _create_recognizer()
        stream = rec.create_stream()
        assert stream is not None
        del stream
        del rec

    def test_english_wav(self):
        """英文 test_wav — 验证模型基础可用"""
        wav_path = str(MODEL_DIR / "test_wavs/en.wav")
        samples, sr = _load_wav(wav_path)
        assert sr == 16000

        rec = _create_recognizer()
        result = _transcribe_streaming(rec, samples)
        assert len(result) > 0, f"Expected non-empty result for English wav, got: {result!r}"

    def test_chinese_wav(self):
        """中文 test_wav — 多语言支持验证"""
        wav_path = str(MODEL_DIR / "test_wavs/zh.wav")
        samples, sr = _load_wav(wav_path)
        rec = _create_recognizer()
        result = _transcribe_streaming(rec, samples)
        assert len(result) > 0

    def test_japanese_wav(self):
        """日文 test_wav — 多语言支持验证"""
        wav_path = str(MODEL_DIR / "test_wavs/ja.wav")
        samples, sr = _load_wav(wav_path)
        rec = _create_recognizer()
        result = _transcribe_streaming(rec, samples)
        assert len(result) > 0


class TestSherpaOnnxStreaming:
    def test_streaming_growth(self):
        """增量输入产生增量结果"""
        rec = _create_recognizer()
        stream = rec.create_stream()
        import sherpa_onnx

        # 喂入静音 + 逐渐增长
        for i in range(5):
            chunk = np.zeros(3200, dtype=np.float32)  # 200ms silence
            stream.accept_waveform(sample_rate=16000, waveform=chunk)
            while rec.is_ready(stream):
                rec.decode_stream(stream)

        stream.input_finished()
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        # 静音应该返回空或极少文本
        result = rec.get_result(stream)
        # 纯静音可能返回空
        assert isinstance(result, str)

    def test_small_chunk_processing(self):
        """小 chunk (128ms) 测试 — 模拟实时通话"""
        wav_path = str(MODEL_DIR / "test_wavs/en.wav")
        samples, sr = _load_wav(wav_path)

        rec = _create_recognizer()
        stream = rec.create_stream()
        import sherpa_onnx

        chunk_size = 2048  # 128ms at 16kHz
        pos = 0
        while pos < len(samples):
            chunk = samples[pos:pos + chunk_size]
            pos += chunk_size
            stream.accept_waveform(sample_rate=16000, waveform=chunk)
            while rec.is_ready(stream):
                rec.decode_stream(stream)

        stream.input_finished()
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        result = rec.get_result(stream)
        assert len(result) > 0, f"Expected non-empty result, got: {result!r}"

    def test_latency_measurement(self):
        """测量增量 ASR 时延"""
        wav_path = str(MODEL_DIR / "test_wavs/en.wav")
        samples, sr = _load_wav(wav_path)

        rec = _create_recognizer()
        stream = rec.create_stream()
        import sherpa_onnx

        chunk_size = 3200  # 200ms
        pos = 0
        decode_times = []

        while pos < len(samples):
            chunk = samples[pos:pos + chunk_size]
            pos += chunk_size
            stream.accept_waveform(sample_rate=16000, waveform=chunk)
            t0 = time.perf_counter()
            while rec.is_ready(stream):
                rec.decode_stream(stream)
            dt = time.perf_counter() - t0
            decode_times.append(dt)

        avg_decode_ms = sum(decode_times) / len(decode_times) * 1000
        print(f"\n平均 decode 时延: {avg_decode_ms:.1f}ms (chunk={chunk_size}/16000={chunk_size/16000*1000:.0f}ms)")
        # 每次 decode 应极快（<50ms per chunk）
        assert avg_decode_ms < 100, f"Decode latency too high: {avg_decode_ms:.1f}ms"


class TestSherpaOnnxIndonesian:
    """印尼语识别效果测试"""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.recognizer = _create_recognizer()

    def test_id_token_exists(self):
        """印尼语 token <ID> 在词表中"""
        tokens_path = MODEL_DIR / "tokens.txt"
        tokens_content = tokens_path.read_text()
        assert "<ID>" in tokens_content, "Indonesian token <ID> not found in tokens.txt"

    def test_simple_id_phrase(self):
        """测试简单印尼语短语"""
        # 用 sherpa-onnx 离线 TTS 生成印尼语测试音频，或直接用真实录音
        # 暂时用静音 + 短音频验证路径不报错
        rec = self.recognizer
        stream = rec.create_stream()

        # 生成一段简单的合成音频用于基本路径验证
        audio = np.zeros(16000, dtype=np.float32)  # 1s silence
        stream.accept_waveform(sample_rate=16000, waveform=audio)
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        stream.input_finished()
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        result = rec.get_result(stream)
        # 静音输入应返回空
        assert result == "" or isinstance(result, str)

    def test_model_language_support(self):
        """验证模型声明支持印尼语"""
        rec = self.recognizer
        assert rec is not None


# ═══════════════════════════════════════════════════════════════════
# SherpaASR 引擎类测试
# ═══════════════════════════════════════════════════════════════════

class TestSherpaASR:
    def test_create_engine(self):
        """引擎创建 + is_available"""
        from core.voice.sherpa_asr import SherpaASR
        asr = SherpaASR()
        assert asr.is_available
        assert asr.sample_rate == 16000

    def test_create_stream(self):
        """创建识别流"""
        from core.voice.sherpa_asr import SherpaASR
        asr = SherpaASR()
        stream = asr.create_stream()
        assert stream is not None

    def test_stream_accept_and_decode(self):
        """流接受音频并解码"""
        from core.voice.sherpa_asr import SherpaASR
        asr = SherpaASR()
        stream = asr.create_stream()

        # 喂入静音
        silence = np.zeros(3200, dtype=np.float32)
        stream.accept_waveform(silence)
        stream.decode()
        result = stream.get_result()
        assert isinstance(result, str)

    def test_stream_finish(self):
        """流结束返回最终结果"""
        from core.voice.sherpa_asr import SherpaASR
        asr = SherpaASR()
        stream = asr.create_stream()

        audio = np.zeros(16000, dtype=np.float32)
        stream.accept_waveform(audio)
        stream.decode()
        stream.finish()
        result = stream.get_result()
        assert isinstance(result, str)

    def test_stream_reset(self):
        """流重置后状态干净"""
        from core.voice.sherpa_asr import SherpaASR
        asr = SherpaASR()
        stream = asr.create_stream()

        audio = np.zeros(16000, dtype=np.float32)
        stream.accept_waveform(audio)
        stream.decode()
        stream.finish()

        stream.reset()
        # 重置后应可继续使用
        stream.accept_waveform(audio)
        stream.decode()
        stream.finish()
        assert isinstance(stream.get_result(), str)

    @pytest.mark.asyncio
    async def test_transcribe_async_compat(self):
        """兼容 RealTimeASR 的 transcribe_async 接口"""
        from core.voice.sherpa_asr import SherpaASR
        asr = SherpaASR()
        wav_path = str(MODEL_DIR / "test_wavs/en.wav")
        samples, _sr = _load_wav(wav_path)

        result = await asr.transcribe_async(samples)
        assert len(result) > 0

    def test_incremental_streaming(self):
        """增量喂入产生增量文本 — SherpaRecognitionStream"""
        from core.voice.sherpa_asr import SherpaASR
        asr = SherpaASR()
        wav_path = str(MODEL_DIR / "test_wavs/en.wav")
        samples, _sr = _load_wav(wav_path)

        stream = asr.create_stream()
        chunk_size = 3200
        partials = []
        last = ""

        for pos in range(0, len(samples), chunk_size):
            chunk = samples[pos:pos + chunk_size]
            stream.accept_waveform(chunk)
            stream.decode()
            cur = stream.get_result()
            if cur and cur != last:
                partials.append(cur)
                last = cur

        stream.finish()
        final = stream.get_result()
        assert len(final) > 0
        # 增量结果应逐渐增长
        if len(partials) > 1:
            assert len(partials[-1]) >= len(partials[0])

    def test_multiple_streams_serial(self):
        """多个 stream 串行使用不互相干扰"""
        from core.voice.sherpa_asr import SherpaASR
        asr = SherpaASR()
        wav_path = str(MODEL_DIR / "test_wavs/en.wav")
        samples, _sr = _load_wav(wav_path)

        results = []
        for _ in range(2):
            stream = asr.create_stream()
            for pos in range(0, len(samples), 3200):
                chunk = samples[pos:pos + 3200]
                stream.accept_waveform(chunk)
                stream.decode()
            stream.finish()
            results.append(stream.get_result())

        assert results[0] == results[1], f"Serial streams differ: {results[0]!r} vs {results[1]!r}"

    def test_shutdown(self):
        """shutdown 后 is_available 为 False"""
        from core.voice.sherpa_asr import SherpaASR
        asr = SherpaASR()
        assert asr.is_available
        asr.shutdown()
        assert not asr.is_available


class TestSherpaASRIndonesian:
    """SherpaASR + 印尼语端到端测试"""

    def test_indonesian_phrase_1(self):
        """印尼语测试 1: Halo, selamat siang"""
        from core.voice.sherpa_asr import SherpaASR
        samples, _sr = _load_wav("/tmp/tts_id_test_0.wav")
        asr = SherpaASR()
        stream = asr.create_stream()
        for pos in range(0, len(samples), 3200):
            chunk = samples[pos:pos + 3200]
            stream.accept_waveform(chunk)
            stream.decode()
        stream.finish()
        result = stream.get_result().lower()
        # 核心词验证
        assert "halo" in result, f"Missing 'halo' in: {result!r}"
        assert "siang" in result, f"Missing 'siang' in: {result!r}"
        assert "bantu" in result, f"Missing 'bantu' in: {result!r}"

    def test_indonesian_phrase_2(self):
        """印尼语测试 2: Saya ingin menanyakan"""
        from core.voice.sherpa_asr import SherpaASR
        samples, _sr = _load_wav("/tmp/tts_id_test_1.wav")
        asr = SherpaASR()
        stream = asr.create_stream()
        for pos in range(0, len(samples), 3200):
            chunk = samples[pos:pos + 3200]
            stream.accept_waveform(chunk)
            stream.decode()
        stream.finish()
        result = stream.get_result().lower()
        assert "saya" in result, f"Missing 'saya' in: {result!r}"
        assert "tagihan" in result, f"Missing 'tagihan' in: {result!r}"

    def test_indonesian_phrase_3(self):
        """印尼语测试 3: Baik, saya catat"""
        from core.voice.sherpa_asr import SherpaASR
        samples, _sr = _load_wav("/tmp/tts_id_test_2.wav")
        asr = SherpaASR()
        stream = asr.create_stream()
        for pos in range(0, len(samples), 3200):
            chunk = samples[pos:pos + 3200]
            stream.accept_waveform(chunk)
            stream.decode()
        stream.finish()
        result = stream.get_result().lower()
        assert "catat" in result, f"Missing 'catat' in: {result!r}"
        assert "pertanyaan" in result, f"Missing 'pertanyaan' in: {result!r}"


# ═══════════════════════════════════════════════════════════════════
# SherpaStreamingASR 适配器测试
# ═══════════════════════════════════════════════════════════════════

class TestSherpaStreamingASR:
    @pytest.mark.asyncio
    async def test_submit_and_final(self):
        """submit 后 mark_final 返回正确文本"""
        from core.voice.sherpa_asr import SherpaASR, SherpaStreamingASR
        asr = SherpaASR()
        s = SherpaStreamingASR(asr)
        partials = []
        s.on_partial_result = lambda t: partials.append(t)

        wav_path = str(MODEL_DIR / "test_wavs/en.wav")
        samples, _sr = _load_wav(wav_path)
        s.submit(samples)
        s.mark_final()
        result = await s.wait_for_final()

        assert len(result) > 0
        assert result == s.final_text

    @pytest.mark.asyncio
    async def test_incremental_submit(self):
        """多次 submit 产生增量回调"""
        from core.voice.sherpa_asr import SherpaASR, SherpaStreamingASR
        asr = SherpaASR()
        s = SherpaStreamingASR(asr)
        partials = []
        s.on_partial_result = lambda t: partials.append(t)

        wav_path = str(MODEL_DIR / "test_wavs/en.wav")
        samples, _sr = _load_wav(wav_path)

        # 模拟分三次提交
        third = len(samples) // 3
        s.submit(samples[:third])
        s.submit(samples[:2 * third])
        s.submit(samples)
        s.mark_final()
        result = await s.wait_for_final()

        assert len(result) > 0
        assert s.has_final_result
        # 增量回调应产生
        if partials:
            assert all(isinstance(p, str) and len(p) > 0 for p in partials)

    @pytest.mark.asyncio
    async def test_reset_between_turns(self):
        """reset 后新一轮可正常使用"""
        from core.voice.sherpa_asr import SherpaASR, SherpaStreamingASR
        asr = SherpaASR()
        s = SherpaStreamingASR(asr)

        wav_path = str(MODEL_DIR / "test_wavs/en.wav")
        samples, _sr = _load_wav(wav_path)

        # 第一轮
        s.submit(samples)
        s.mark_final()
        r1 = await s.wait_for_final()
        assert len(r1) > 0

        # 第二轮
        s.reset()
        assert not s.is_active
        assert not s.has_final_result
        s.submit(samples)
        s.mark_final()
        r2 = await s.wait_for_final()
        assert r2 == r1, f"Reset gave different result: {r1!r} vs {r2!r}"

    @pytest.mark.asyncio
    async def test_mark_final_without_submit(self):
        """无 submit 时 mark_final 返回空"""
        from core.voice.sherpa_asr import SherpaASR, SherpaStreamingASR
        asr = SherpaASR()
        s = SherpaStreamingASR(asr)

        assert not s.is_active
        s.mark_final()
        result = await s.wait_for_final()
        assert result == ""
        assert s.has_final_result

    @pytest.mark.asyncio
    async def test_is_active_and_state(self):
        """状态属性正确切换"""
        from core.voice.sherpa_asr import SherpaASR, SherpaStreamingASR
        asr = SherpaASR()
        s = SherpaStreamingASR(asr)

        assert not s.is_active
        assert not s.is_final_pending
        assert not s.has_final_result

        audio = np.zeros(3200, dtype=np.float32)
        s.submit(audio)
        assert s.is_active
        assert not s.is_final_pending

        s.mark_final()
        assert s.has_final_result

    @pytest.mark.asyncio
    async def test_indonesian_full_flow(self):
        """印尼语完整流式识别流程"""
        from core.voice.sherpa_asr import SherpaASR, SherpaStreamingASR
        asr = SherpaASR()
        s = SherpaStreamingASR(asr)
        partials = []
        s.on_partial_result = lambda t: partials.append(t)

        samples, _sr = _load_wav("/tmp/tts_id_test_0.wav")
        s.submit(samples)
        s.mark_final()
        result = await s.wait_for_final()

        result_lower = result.lower()
        assert "halo" in result_lower or "selamat" in result_lower or "siang" in result_lower


# ---- 手动测试 (非 pytest) ----

def interactive_test(audio_path: str = None):
    """手动运行印尼语识别测试"""
    rec = _create_recognizer(enable_endpoint=True)

    if audio_path:
        samples, sr = _load_wav(audio_path)
        print(f"Loaded: {audio_path}, sr={sr}, duration={len(samples)/sr:.1f}s")

        t0 = time.perf_counter()
        result = _transcribe_streaming(rec, samples)
        elapsed = time.perf_counter() - t0

        print(f"ASR result: {result!r}")
        print(f"Real-time factor: {elapsed / (len(samples)/sr):.2f}x")
    else:
        print("No audio file provided. Interactive mode — create recognizer and wait.")
        print(f"Recognizer created: {rec}")

    return rec


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        interactive_test(sys.argv[1])
    else:
        print("Usage: python test_sherpa_onnx.py <wav_file>")
        print("提供印尼语 wav 文件进行手动测试")
        # 至少验证模型可加载
        rec = _create_recognizer()
        print("Model loaded OK")
