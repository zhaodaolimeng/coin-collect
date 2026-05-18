#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS引擎抽象层
支持多种TTS后端：Piper-TTS（本地）、Edge-TTS、Coqui-TTS等
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import time
import os

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


@dataclass
class TTSResult:
    """TTS合成结果"""
    text: str
    audio_file: Optional[str] = None
    audio_data: Optional[bytes] = None
    duration: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    engine_name: str = ""


class TTSEngine(ABC):
    """TTS引擎抽象基类"""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        output_file: Optional[str] = None,
        voice: Optional[str] = None,
        **kwargs
    ) -> TTSResult:
        """
        合成语音

        Args:
            text: 要合成的文本
            output_file: 输出文件路径
            voice: 语音类型
            **kwargs: 其他参数

        Returns:
            TTS合成结果
        """
        pass

    @abstractmethod
    async def list_voices(self, locale: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        列出可用语音

        Args:
            locale: 语言区域

        Returns:
            语音列表
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查引擎是否可用"""
        pass

    @abstractmethod
    def get_engine_name(self) -> str:
        """获取引擎名称"""
        pass


class EdgeTTSEngine(TTSEngine):
    """
    Edge-TTS引擎实现
    """

    def __init__(self, default_voice: str = "id-ID-ArdiNeural"):
        self.default_voice = default_voice
        self._available = False
        self._edge_tts = None

        try:
            import edge_tts
            self._edge_tts = edge_tts
            self._available = True
        except ImportError:
            logger.warning("edge-tts未安装，TTS将不可用")

    async def synthesize(
        self,
        text: str,
        output_file: Optional[str] = None,
        voice: Optional[str] = None,
        **kwargs
    ) -> TTSResult:
        if not self._available:
            return TTSResult(
                text=text,
                success=False,
                error_message="Edge-TTS不可用",
                engine_name=self.get_engine_name()
            )

        start_time = time.time()
        voice = voice or self.default_voice

        try:
            if output_file is None:
                output_dir = _PROJECT_ROOT / "data/runs/tts_output"
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S_%f")
                output_file = str(output_dir / f"tts_{timestamp}.mp3")

            communicate = self._edge_tts.Communicate(text, voice)
            await communicate.save(output_file)

            duration = time.time() - start_time

            return TTSResult(
                text=text,
                audio_file=output_file,
                duration=duration,
                success=True,
                engine_name=self.get_engine_name()
            )

        except Exception as e:
            return TTSResult(
                text=text,
                success=False,
                error_message=str(e),
                engine_name=self.get_engine_name()
            )

    async def list_voices(self, locale: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self._available:
            return []

        try:
            voices = await self._edge_tts.list_voices()
            if locale:
                voices = [v for v in voices if locale in v.get("Locale", "")]
            return voices
        except:
            return []

    def is_available(self) -> bool:
        return self._available

    def get_engine_name(self) -> str:
        return "edge_tts"


class CoquiTTSEngine(TTSEngine):
    """
    Coqui-TTS引擎实现
    """

    def __init__(self, model_name: str = "tts_models/id/css10/vits"):
        self.model_name = model_name
        self._available = False
        self._tts = None
        self._model = None

        try:
            import TTS
            self._tts = TTS
            self._available = True
            logger.info("Coqui-TTS已加载")
        except ImportError:
            logger.debug("Coqui-TTS未安装（可选引擎）")

    async def synthesize(
        self,
        text: str,
        output_file: Optional[str] = None,
        voice: Optional[str] = None,
        **kwargs
    ) -> TTSResult:
        if not self._available:
            return TTSResult(
                text=text,
                success=False,
                error_message="Coqui-TTS不可用",
                engine_name=self.get_engine_name()
            )

        start_time = time.time()

        try:
            # 延迟加载模型
            if self._model is None:
                from TTS.api import TTS as CoquiTTS
                self._model = CoquiTTS(model_name=self.model_name, progress_bar=False)

            if output_file is None:
                output_dir = _PROJECT_ROOT / "data/runs/tts_output"
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S_%f")
                output_file = str(output_dir / f"coqui_{timestamp}.wav")

            # 使用线程池执行同步操作
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._model.tts_to_file(text=text, file_path=output_file)
            )

            duration = time.time() - start_time

            return TTSResult(
                text=text,
                audio_file=output_file,
                duration=duration,
                success=True,
                engine_name=self.get_engine_name()
            )

        except Exception as e:
            return TTSResult(
                text=text,
                success=False,
                error_message=str(e),
                engine_name=self.get_engine_name()
            )

    async def list_voices(self, locale: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self._available:
            return []

        try:
            models = self._tts.list_models()
            voices = []
            for model in models:
                if locale and locale not in model:
                    continue
                voices.append({
                    "Name": model,
                    "Locale": model.split("/")[1] if len(model.split("/")) > 1 else "",
                    "Model": model
                })
            return voices
        except:
            return []

    def is_available(self) -> bool:
        return self._available

    def get_engine_name(self) -> str:
        return "coqui_tts"


class PiperTTSEngine(TTSEngine):
    """
    Piper-TTS引擎实现 - 纯本地TTS，无需网络
    支持多种语言的ONNX语音模型
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        default_voice: str = "id_ID-news_tts-medium",
    ):
        self.default_voice = default_voice
        self._available = False
        self._piper = None
        self._voice = None
        self._voice_loaded_for = None

        try:
            import piper
            self._piper = piper
            self._available = True
        except ImportError:
            logger.debug("piper-tts未安装（可选引擎）")

        # 默认模型路径
        if model_path is None:
            self._model_base = Path.home() / ".piper-voices"
        else:
            self._model_base = Path(model_path)

    def _find_model(self, voice_name: str) -> tuple:
        """查找语音模型文件 (onnx_path, json_path)"""
        # 1. 在子目录中查找: model_base/xx_XX/voice_name.onnx
        if self._model_base.exists():
            for onnx_file in self._model_base.rglob("*.onnx"):
                if onnx_file.stem == voice_name or voice_name in str(onnx_file):
                    onnx_path = str(onnx_file)
                    json_path = onnx_path + ".json"
                    return onnx_path, json_path

        # 2. 直接路径: model_base/voice_name.onnx
        direct_onnx = self._model_base / f"{voice_name}.onnx"
        if direct_onnx.exists():
            return str(direct_onnx), str(direct_onnx) + ".json"

        # 3. 作为绝对路径
        direct_path = Path(voice_name)
        if direct_path.exists() and direct_path.suffix == ".onnx":
            return str(direct_path), str(direct_path) + ".json"

        return None, None

    def _load_voice(self, voice_name: str):
        """加载或切换语音"""
        if self._voice is not None and self._voice_loaded_for == voice_name:
            return self._voice

        onnx_path, _ = self._find_model(voice_name)
        if onnx_path is None:
            raise FileNotFoundError(f"Piper voice model not found: {voice_name}")
        self._voice = self._piper.PiperVoice.load(onnx_path)
        self._voice_loaded_for = voice_name
        return self._voice

    async def synthesize(
        self,
        text: str,
        output_file: Optional[str] = None,
        voice: Optional[str] = None,
        **kwargs
    ) -> TTSResult:
        if not self._available:
            return TTSResult(
                text=text,
                success=False,
                error_message="Piper-TTS不可用",
                engine_name=self.get_engine_name()
            )

        start_time = time.time()
        voice_name = voice or self.default_voice

        try:
            # 如果指定语音不存在，回退到默认语音
            onnx_path, _ = self._find_model(voice_name)
            if onnx_path is None:
                voice_name = self.default_voice
            voice_obj = self._load_voice(voice_name)

            if output_file is None:
                output_dir = _PROJECT_ROOT / "data/runs/tts_output"
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S_%f")
                output_file = str(output_dir / f"piper_{timestamp}.wav")

            # 在线程池中执行同步TTS合成
            loop = asyncio.get_event_loop()
            raw_pcm, sample_rate = await loop.run_in_executor(
                None,
                self._synth_sync,
                voice_obj,
                text,
            )

            # 写入WAV文件
            import wave
            with wave.open(output_file, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(raw_pcm)

            duration = time.time() - start_time

            return TTSResult(
                text=text,
                audio_file=output_file,
                duration=duration,
                success=True,
                engine_name=self.get_engine_name()
            )

        except Exception as e:
            return TTSResult(
                text=text,
                success=False,
                error_message=str(e),
                engine_name=self.get_engine_name()
            )

    def _synth_sync(self, voice_obj, text: str):
        """同步TTS合成（在线程池中运行）"""
        raw_pcm = b""
        sample_rate = None
        for chunk in voice_obj.synthesize(text):
            if sample_rate is None:
                sample_rate = chunk.sample_rate
            raw_pcm += chunk.audio_int16_bytes
        return raw_pcm, sample_rate

    async def list_voices(self, locale: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self._available:
            return []

        voices = []
        if self._model_base.exists():
            for onnx_file in self._model_base.rglob("*.onnx"):
                name = onnx_file.stem
                voices.append({
                    "Name": name,
                    "Locale": name.split("-")[0] if "-" in name else "",
                    "Model": str(onnx_file),
                })
        return voices

    def is_available(self) -> bool:
        if not self._available:
            return False
        try:
            onnx_path, _ = self._find_model(self.default_voice)
            return onnx_path is not None
        except Exception:
            return False

    def get_engine_name(self) -> str:
        return "piper_tts"


class TTSManager:
    """
    TTS管理器 - 支持多引擎切换
    """

    def __init__(self):
        self.engines: Dict[str, TTSEngine] = {}
        self.default_engine: Optional[str] = None

        # 注册默认引擎
        self._register_default_engines()

    def _register_default_engines(self):
        """注册默认引擎（本地 Piper-TTS 优先，低延迟）"""
        # Piper-TTS（优先 - 纯本地，无需网络，亚实时合成）
        piper_engine = PiperTTSEngine()
        if piper_engine.is_available():
            self.engines[piper_engine.get_engine_name()] = piper_engine
            if self.default_engine is None:
                self.default_engine = piper_engine.get_engine_name()

        # Edge-TTS（备选 - 网络TTS，质量更好但延迟高）
        edge_engine = EdgeTTSEngine()
        if edge_engine.is_available():
            self.engines[edge_engine.get_engine_name()] = edge_engine
            if self.default_engine is None:
                self.default_engine = edge_engine.get_engine_name()

        # Coqui-TTS（备选）
        coqui_engine = CoquiTTSEngine()
        if coqui_engine.is_available():
            self.engines[coqui_engine.get_engine_name()] = coqui_engine
            if self.default_engine is None:
                self.default_engine = coqui_engine.get_engine_name()

    def register_engine(self, engine: TTSEngine, set_default: bool = False):
        """注册TTS引擎"""
        self.engines[engine.get_engine_name()] = engine
        if set_default or self.default_engine is None:
            self.default_engine = engine.get_engine_name()

    def get_engine(self, engine_name: Optional[str] = None) -> Optional[TTSEngine]:
        """获取TTS引擎"""
        if engine_name is None:
            engine_name = self.default_engine

        return self.engines.get(engine_name)

    async def synthesize(
        self,
        text: str,
        output_file: Optional[str] = None,
        voice: Optional[str] = None,
        engine: Optional[str] = None,
        **kwargs
    ) -> TTSResult:
        """合成语音（自动选择可用引擎）"""
        tts_engine = self.get_engine(engine)

        if tts_engine is None:
            return TTSResult(
                text=text,
                success=False,
                error_message="没有可用的TTS引擎"
            )

        return await tts_engine.synthesize(text, output_file, voice, **kwargs)

    def get_available_engines(self) -> List[str]:
        """获取可用引擎列表"""
        return [name for name, engine in self.engines.items() if engine.is_available()]


# 简单测试
if __name__ == "__main__":
    print("TTS引擎模块加载成功")

    manager = TTSManager()
    print(f"可用引擎: {manager.get_available_engines()}")
    print(f"默认引擎: {manager.default_engine}")
