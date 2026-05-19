#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能催收对话机器人 - 语音版本 (v3)
集成TTS功能，完善状态机，支持变量替换
基于246条对话分析
"""
import copy
import random
import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Dict, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import sys
import io

_PROJECT_ROOT = Path(__file__).parent.parent.parent

# 可选：导入简易ML分类器
try:
    from core.simple_classifier import SimpleIntentClassifier
    ML_CLASSIFIER_AVAILABLE = True
except ImportError:
    ML_CLASSIFIER_AVAILABLE = False

# 可选：导入 LLM Fallback 模块
try:
    from core.llm_config import LLMConfig
    from core.llm_provider import LLMProvider, LLMUnavailableError
    from core.fallback_detector import FallbackDetector
    LLM_FALLBACK_AVAILABLE = True
except ImportError:
    LLM_FALLBACK_AVAILABLE = False

# 确保输出编码正确
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    import edge_tts
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("警告: edge-tts未安装，TTS功能不可用")


class ChatState(Enum):
    """对话状态枚举"""
    INIT = auto()
    GREETING = auto()
    IDENTITY_VERIFY = auto()  # 身份确认
    PURPOSE = auto()  # 说明来意
    HANDLE_OBJECTION = auto()  # 处理用户异议
    ASK_TIME = auto()  # 询问还款时间
    PUSH_FOR_TIME = auto()  # 催促确认时间
    COMMIT_TIME = auto()  # 确认用户还款时间
    CONFIRM_EXTENSION = auto()  # 确认展期
    HANDLE_BUSY = auto()  # 处理用户忙碌情况
    HANDLE_WRONG_NUMBER = auto()  # 处理错号情况
    CLOSE = auto()
    FAILED = auto()
    LLM_FALLBACK = auto()  # LLM 兜底处理


@dataclass
class ChatTurn:
    """对话回合"""
    agent: str
    customer: Optional[str] = None
    state: Optional[Any] = None  # ChatState at the time this turn was created
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    latency_ms: Optional[float] = None


@dataclass
class ConversationLog:
    """对话日志"""
    session_id: str
    chat_group: str
    customer_info: Dict
    turns: List[ChatTurn] = field(default_factory=list)
    success: bool = False
    commit_time: Optional[str] = None
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None


class TextToSpeech:
    """TTS封装类 - 优先本地Piper，备选Edge TTS"""

    def __init__(self, voice: str = "id-ID-ArdiNeural"):
        self.voice = voice
        self._local_available = False
        self._piper = None

        # 尝试加载本地Piper引擎
        try:
            import piper
            model_path = Path.home() / ".piper-voices" / "id_ID" / "id_ID-news_tts-medium.onnx"
            if model_path.exists():
                self._piper = piper
                self._local_available = True
        except Exception:
            pass

        self.available = self._local_available or TTS_AVAILABLE

    async def synthesize(self, text: str, output_file: Optional[str] = None) -> Optional[str]:
        """合成语音"""
        if not text:
            return None

        if output_file is None:
            output_dir = _PROJECT_ROOT / "data/runs/tts_output"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_file = str(output_dir / f"tts_{timestamp}.wav")

        # 优先使用本地Piper
        if self._local_available and self._piper:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, self._piper_synth, text, str(output_file)
                )
                if result:
                    return result
            except Exception:
                pass

        # 备选Edge TTS
        if not self._local_available and TTS_AVAILABLE:
            try:
                communicate = edge_tts.Communicate(text, self.voice)
                await communicate.save(output_file)
                return output_file
            except Exception as e:
                print(f"TTS错误: {e}")
                return None

        return None

    def _piper_synth(self, text: str, output_file: str) -> Optional[str]:
        """同步Piper合成"""
        import wave
        model_path = str(Path.home() / ".piper-voices" / "id_ID" / "id_ID-news_tts-medium.onnx")
        voice = self._piper.PiperVoice.load(model_path)
        raw_pcm = b""
        sample_rate = 22050
        for chunk in voice.synthesize(text):
            raw_pcm += chunk.audio_int16_bytes
            sample_rate = chunk.sample_rate
        with wave.open(output_file, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(raw_pcm)
        return output_file

    async def list_voices(self, locale: str = "id-ID") -> List[Dict]:
        """列出可用语音"""
        if not self.available:
            return []
        voices = await edge_tts.list_voices()
        return [v for v in voices if locale in v["Locale"]]


class VariableReplacer:
    """话术模板变量替换器"""

    def __init__(self):
        self.default_vars = {
            "time": "nanti",
            "name": "Pak/Bu",
            "amount": "pinjaman",
            "date": "hari ini"
        }

    def replace(self, text: str, **kwargs) -> str:
        """替换变量"""
        vars = {**self.default_vars, **kwargs}
        try:
            return text.format(**vars)
        except KeyError as e:
            print(f"警告: 变量缺失 {e}")
            for key in e.args:
                text = text.replace(f"{{{key}}}", vars.get(key, f"{{{key}}}"))
            return text


class TimeDetector:
    """时间检测器 - 增强版，支持复杂口语化时间表达"""

    # 时间模式，按优先级排序：具体时间 > 相对时间 > 模糊时间
    TIME_PATTERNS = [
        # 具体点钟
        ("jam 12", ["jam 12", "12 siang", "jam dua belas", "siang jam 12", "tengah hari"]),
        ("jam 11", ["jam 11", "jam sebelas", "11 siang", "11 pagi"]),
        ("jam 10", ["jam 10", "jam sepuluh", "10 pagi", "10 siang"]),
        ("jam 9", ["jam 9", "jam sembilan", "9 pagi", "9 siang"]),
        ("jam 8", ["jam 8", "jam delapan", "8 pagi", "8 siang"]),
        ("jam 7", ["jam 7", "jam tujuh", "7 pagi"]),
        ("jam 6", ["jam 6", "jam enam", "6 pagi"]),
        ("jam 5", ["jam 5", "jam lima", "5 sore", "5 pagi"]),
        ("jam 4", ["jam 4", "jam empat", "4 sore", "4 pagi"]),
        ("jam 3", ["jam 3", "jam tiga", "3 sore", "3 pagi"]),
        ("jam 2", ["jam 2", "jam dua", "2 sore", "2 siang"]),
        ("jam 1", ["jam 1", "jam satu", "1 siang", "1 pagi"]),
        # 半点和刻钟表达
        ("jam 2.30", ["setengah 3", "setengah tiga", "jam 2 kurang 30", "3 kurang setengah"]),
        ("jam 1.30", ["setengah 2", "setengah dua", "jam 1 kurang 30", "2 kurang setengah"]),
        ("jam 3.30", ["setengah 4", "setengah empat", "jam 3 kurang 30", "4 kurang setengah"]),
        ("jam 2.45", ["jam 3 kurang 15", "seperempat jam 3"]),
        ("jam 2.15", ["seperempat lewat 2", "jam 2 lebih 15"]),
        # 时间段
        ("pagi hari", ["pagi", "pagi hari", "nanti pagi", "pagi ini", "besok pagi"]),
        ("siang hari", ["siang", "siang hari", "nanti siang", "siang ini", "tengah hari"]),
        ("sore hari", ["sore", "sore hari", "nanti sore", "sore ini", "petang"]),
        ("malam hari", ["malam", "malam hari", "nanti malam", "malam ini", "besok malam"]),
        # 具体日期
        ("hari ini", ["hari ini", "sekarang", "hari ini siang", "hari ini sore", "hari ini pagi", "sekarang juga", "nanti hari ini", "hari ini malam"]),
        ("besok", ["besok", "besok pagi", "besok siang", "besok sore", "besok malam", "hari besok", "esok"]),
        ("lusa", ["lusa", "hari lusa", "besok lusa"]),
        # 星期
        ("hari senin", ["senin", "hari senin", "minggu ini senin"]),
        ("hari selasa", ["selasa", "hari selasa", "minggu ini selasa"]),
        ("hari rabu", ["rabu", "hari rabu", "minggu ini rabu"]),
        ("hari kamis", ["kamis", "hari kamis", "minggu ini kamis"]),
        ("hari jumat", ["jumat", "hari jumat", "jum'at", "minggu ini jumat"]),
        ("hari sabtu", ["sabtu", "hari sabtu", "minggu ini sabtu"]),
        ("hari minggu", ["minggu", "hari minggu", "minggu ini minggu"]),
        # 相对时间
        ("minggu ini", ["minggu ini", "pekan ini"]),
        ("minggu depan", ["minggu depan", "pekan depan"]),
        ("akhir minggu", ["akhir minggu", "weekend", "akhir pekan"]),
        ("awal bulan", ["awal bulan", "awal bulan depan"]),
        ("akhir bulan", ["akhir bulan", "akhir bulan depan", "menjelang gaji", "pas gaji"]),
        ("nanti", ["nanti", "sebentar lagi", "beberapa saat lagi", "satu jam lagi", "dua jam lagi", "nanti ya", "nanti dulu"]),
        ("beberapa hari lagi", ["beberapa hari lagi", "dua hari lagi", "tiga hari lagi", "beberapa hari lagi ya"]),
    ]

    @classmethod
    def detect(cls, text: str) -> Optional[str]:
        """检测时间，优先返回更具体的时间表达，支持组合时间提取"""
        if not text:
            return None

        text_lower = text.lower()
        detected_times = []

        # 先检测所有匹配的时间
        for time_value, patterns in cls.TIME_PATTERNS:
            for pattern in patterns:
                if pattern in text_lower:
                    # 权重：匹配长度越大概率越具体
                    detected_times.append((len(pattern), time_value))

        # 检测通用jam + 数字模式
        import re
        jam_matches = re.findall(r'jam\s+(\d+)', text_lower)
        if jam_matches:
            for jam in jam_matches:
                detected_times.append((4, f"jam {jam}"))  # 长度4作为权重

        # 检测具体日期模式：tanggal + 数字
        tgl_matches = re.findall(r'(tanggal|tgl)\s+(\d+)', text_lower)
        if tgl_matches:
            for _, tgl in tgl_matches:
                detected_times.append((7, f"tanggal {tgl}"))

        # 如果检测到多个时间，尝试组合日期+时间
        if len(detected_times) >= 2:
            date_words = ["besok", "lusa", "hari ini", "senin", "selasa", "rabu", "kamis", "jumat", "sabtu", "minggu"]
            dates = []
            times = []
            for _, val in detected_times:
                if val in date_words or val.startswith("tanggal") or val.startswith("hari"):
                    dates.append(val)
                elif val.startswith("jam") or val in ["pagi hari", "siang hari", "sore hari", "malam hari"]:
                    times.append(val)

            # 如果同时有日期和时间，组合起来返回
            if dates and times:
                return f"{dates[0]} {times[0]}"

        # 否则返回最长匹配（最具体）的那个
        if detected_times:
            # 按匹配长度降序排序，取最长的
            detected_times.sort(reverse=True, key=lambda x: x[0])
            return detected_times[0][1]

        return None


class ASRCorrector:
    """印尼语ASR错误纠正器 - 基于常见错误映射"""

    # 印尼语ASR常见错误修正映射，从batch_annotate.py迁移并扩展
    ASR_CORRECTIONS = {
        "nasian": "lunas",
        "puluh nasian": "penuh lunas",
        "tempat": "tempo",
        "Hajianya": "tagihan Anda",
        "Ufah Nau": "Uang",
        "Kuala": "Nanti lah",
        "kemaren": "hari ini",
        "Jalan waktu itu": "Baik, jam 10 ya",
        "Extra Uang": "Uang Extra",
        "Uang extra": "Uang Extra",
        "uang extra": "Uang Extra",
        "tagihan nya": "tagihan Anda",
        "tagihan mu": "tagihan Anda",
        "bapak nya": "bapak Anda",
        "ibu nya": "ibu Anda",
        "saya nya": "saya",
        "dia nya": "dia",
        "kita nya": "kita",
        "mereka nya": "mereka",
        "nggak": "tidak",
        "gak": "tidak",
        "ga": "tidak",
        "tdk": "tidak",
        "ya": "ya",
        "iya": "iya",
        "iiya": "iya",
        "iyya": "iya",
        "yaa": "ya",
        "yaa": "ya",
        "ngga": "tidak",
        "gaga": "tidak",
        "saya": "saya",
        "sayaa": "saya",
        "sya": "saya",
        "sy": "saya",
        "kamu": "kamu",
        "km": "kamu",
        "anda": "Anda",
        "Anda": "Anda",
        "kd": "Anda",
        "jp": "jam",
        "jp.": "jam",
        "rp": "Rp",
        "rupiah": "rupiah",
        "rb": "ribu",
        "juta": "juta",
        "jt": "juta",
        "wang": "uang",
        "wangnya": "uangnya",
        "buktifnya": "buktinya",
        # faster_whisper small 模型对 "besok" 的常见误识别
        "bisok": "besok",
        "pisok": "besok",
        "disok": "besok",
    }

    @classmethod
    def correct(cls, text: str) -> str:
        """纠正ASR识别错误"""
        import re
        if not text:
            return text

        corrected_text = text.strip()
        # 按错误字符串长度从长到短排序，避免短的错误先被替换掉
        sorted_corrections = sorted(cls.ASR_CORRECTIONS.items(), key=lambda x: len(x[0]), reverse=True)
        for error, correct in sorted_corrections:
            # 全字匹配替换，使用单词边界符，避免替换单词内部的子字符串
            # 对包含特殊正则字符的错误字符串进行转义
            error_escaped = re.escape(error)
            corrected_text = re.sub(rf'\b{error_escaped}\b', correct, corrected_text, flags=re.IGNORECASE)

        return corrected_text


class IntentDetector:
    """用户意图识别器 - 混合规则式+轻量级ML分类器
    优先使用规则系统匹配，未匹配到则使用ML分类器作为fallback
    """

    # ML分类器缓存（类变量，全局共享）
    _ml_classifier = None
    _ml_threshold = 0.3  # ML分类结果置信度阈值，低于这个值还是返回unknown
    _use_ml_fallback = True  # 是否启用ML fallback

    # 注意顺序：越具体的意图越靠前，避免被更通用的意图匹配覆盖
    INTENT_PATTERNS = [
        ("deny_identity", [r"\bbukan\b", r"\bsalah nomor\b", r"\banda salah orang\b", r"\bsaya tidak kenal\b", r"\bini bukan nomornya\b", r"\bsalah orang\b", r"\bbukan orang yang anda cari\b"]),
        ("busy_later", [r"\bsibuk\b", r"\bnanti ya\b(?!.*bayar|.*transfer)", r"\bnanti dulu ya\b", r"\bnanti.*ya\b(?!.*bayar|.*transfer)", r"\bsaya lagi diluar\b", r"\bnanti saya hubungi balik\b", r"\bsebentar lagi\b", r"\bsaya lagi mengemudi\b", r"\bsaya sedang rapat\b", r"\bnanti saya telepon kembali\b", r"\bsaya tidak bisa bicara sekarang\b", r"\bsalat dulu\b", r"\bberhenti dulu\b", r"\bnanti saya wasap\b"]),
        ("user_abuse", [r"\banjing\b", r"\bbangsat\b", r"\bgoblok\b", r"\btolol\b", r"\bbego\b", r"\bkampret\b", r"\bbrengsek\b", r"\bsetan\b", r"\biblis\b"]),
        ("threaten", [r"\bsaya akan laporkan ke ojk\b", r"\bsaya akan lapor polisi\b", r"\banda ancam saya\b", r"\bsaya akan lapor ke pihak berwenang\b", r"\bsaya akan komplain\b", r"\bsaya rekam\b", r"\brekam ini\b", r"\bsaya adukan\b", r"\bmesti aduk\b", r"\bakan saya adukan\b", r"\bpengawasnya\b", r"\bada pengawas\b", r"\bancam\b", r"\blapor\b"]),
        ("ask_extension", [r"\bperpanjang\b", r"\bdiperpanjang\b", r"\bperpanjangan\b", r"\bbisa tidak diperpanjang\b", r"\bbisa nggak diperpanjang\b", r"\bbisa gak diperpanjang\b", r"\bextension\b", r"\btunda bayar\b", r"\bditunda\b", r"\bbisa ditunda ya\b", r"\bsaya mau perpanjang\b", r"\bberapa hari bisa ditunda\b", r"\bnanti minggu depan baru bisa bayar\b", r"\bkasih waktu\b", r"\bbisa kasih waktu\b", r"\btunda dulu\b", r"\bperpanjang waktu\b", r"\bbutuh waktu lagi\b", r"\bbisa tunggu sampai\b", r"\bmau tunda bayar\b"]),
        ("ask_amount", [r"\bjumlahnya berapa\b", r"\btagihan berapa\b", r"\bbesarnya berapa\b", r"\bberapa nominalnya\b", r"\bbesar tagihan\b", r"\bberapa bayarnya\b"]),
        ("question_identity", [r"\bsiapa kamu\b", r"\banda dari mana\b", r"\bmana buktinya\b", r"\bsaya tidak percaya\b", r"\bpenipuan\b", r"\bapakah ini penipuan\b", r"\banda siapa\b", r"\bsaya tidak pinjam\b", r"\btidak pernah pinjam\b"]),
        ("request_identity_verification", [r"\bbuktikan kamu dari extra uang\b", r"\bmana kartu identitas kamu\b", r"\bsaya mau lihat surat kuasa\b", r"\bapakah kamu benar dari extra uang\b", r"\bbuktikan kamu petugas resmi\b"]),
        ("request_interest_reduction", [r"\bbisa kurangi bunga\b", r"\bbisa kurangin bunga", r"\bbisa hapus denda\b", r"\bpotongan biaya\b", r"\bkurangi denda keterlambatan\b", r"\bkurangin denda keterlambatan\b", r"\bbisa tidak denda dihilangkan\b", r"\bturunin bunga\b", r"\bbisa turunin bunga\b", r"\bkurangin dong bunganya\b", r"\bbunganya terlalu tinggi\b", r"\bbunga terlalu tinggi\b", r"\bbiaya terlalu mahal\b"]),
        ("request_short_extension", [r"\bbisa tunggu 3 hari lagi\b", r"\bberi waktu 2 hari ya\b", r"\btunda dulu 5 hari\b", r"\bsaya bayar minggu depan ya\b", r"\bbisa kasih waktu beberapa hari\b", r"\bkasih 2 hari\b", r"\bkasih 3 hari ya\b", r"\btunda 3 hari\b"]),
        ("complain_high_interest", [r"\bbunganya terlalu tinggi\b", r"\bbiaya terlalu mahal\b", r"\bkenapa bunga begitu besar\b", r"\bwaktu pinjam tidak dijelaskan biaya tinggi\b", r"\bbunganya selangit\b"]),
        ("app_uninstalled", [r"\bsaya sudah hapus aplikasinya\b", r"\bapk sudah dihapus\b", r"\btidak ada aplikasi nya\b", r"\bsaya tidak instal aplikasinya\b", r"\blupa password aplikasi\b"]),
        ("request_payment_reminder", [r"\bkirim pengingat ke whatsapp saya\b", r"\bsms kan detail tagihan\b", r"\bkirim pesan tagihan ya\b", r"\bingetin saya nanti ya\b", r"\bkirim bukti tagihan ke hp saya\b"]),
        ("request_settlement_proof", [r"\bsaya mau surat keterangan lunas\b", r"\bberikan bukti sudah lunas\b", r"\bbagaimana cara dapat surat lunas\b", r"\bsaya perlu bukti pembayaran\b"]),
        ("inquire_consequences", [r"\bjika tidak bayar bagaimana\b", r"\bakibatnya apa kalau tidak bayar\b", r"\bapa yang terjadi jika saya tidak bayar\b", r"\bkalau tidak bayar ada apa ya\b"]),
        ("borrowing_money", [r"\bsaya sedang pinjam uang dulu\b", r"\bsaya sedang kumpulkan uang\b", r"\bsedang cari uang untuk bayar\b", r"\bsaya sedang minta tolong temen\b", r"\bsedang pinjam ke saudara\b"]),
        ("transfer_in_process", [r"\bsaya sedang transfer sekarang\b", r"\bsedang proses transfer\b", r"\bsudah mau transfer\b", r"\bsedang isi rekening dulu\b", r"\bsedang bayar sekarang\b"]),
        ("no_money", [r"\btidak ada duit\b", r"\bbelum ada duit\b", r"\bsaya tidak punya uang\b", r"\bsaya belum punya uang\b", r"\bsaya tidak punya duit\b", r"\bsaya belum punya duit\b", r"\blagi susah\b", r"\belum ada uang\b", r"\btidak ada uang\b", r"\bsaya tidak sanggup\b", r"\bbenar-benar tidak sanggup\b", r"\bsaya sedang kesulitan keuangan\b", r"\buang saya belum masuk\b", r"\bgaji belum cair\b", r"\bsulit\b", r"\bkesulitan\b", r"\bkeberatan\b", r"\btidak mampu\b", r"\bbelum mampu\b", r"\btidak bisa bayar\b", r"\bnggak bisa bayar\b", r"\bgak bisa bayar\b", r"\btidak sanggup bayar\b"]),
        ("agree_to_pay", [r"\bsiap bayar\b", r"\bolehan\b", r"\bsetuju\b", r"\bsaya akan bayar\b", r"\bsaya bayar nanti\b", r"\bnanti saya transfer\b", r"\bsaya bayar besok\b", r"\bsaya proses sekarang\b", r"\bsaya bayar hari ini\b", r"\baiya, saya bayar\b", r"\bbaik, saya bayar\b", r"\bsaya bayar\b", r"\btransfer segera\b", r"\bsaya akan transfer\b", r"\bnanti saya bayar\b", r"\bsaya akan bayar segera\b", r"\bsaya akan proses pembayaran\b", r"\bsaya bayar nanti ya\b", r"\bya saya bayar\b", r"\biya saya bayar\b", r"\bobayarnya saya transfer nanti\b", r"\bayar nanti ya\b", r"\bsaya akan bayar ya\b", r"\bok saya bayar\b", r"\boke saya bayar\b", r"\bok\b.*\bbayar", r"\bok\b.*\btransfer", r"\bok\b.*\buang", r"\bok\b.*\btagihan", r"\bok\b.*\bpembayaran", r"\boke\b.*\bbayar", r"\boke\b.*\btransfer", r"\boke\b.*\buang", r"\boke\b.*\btagihan", r"\boke\b.*\bpembayaran", r"\bya\b.*\bbayar", r"\bya\b.*\btransfer", r"\bya\b.*\buang", r"\bya\b.*\btagihan", r"\bya\b.*\bpembayaran", r"\biya\b.*\bbayar", r"\biya\b.*\btransfer", r"\biya\b.*\buang", r"\biya\b.*\btagihan", r"\biya\b.*\bpembayaran", r"\bbetul\b.*\bbayar", r"\bbetul\b.*\btransfer", r"\bbetul\b.*\buang", r"\bbetul\b.*\btagihan", r"\bbetul\b.*\bpembayaran", r"\bbaik\b.*\bbayar", r"\bbaik\b.*\btransfer", r"\bbaik\b.*\buang", r"\bbaik\b.*\btagihan", r"\bbaik\b.*\bpembayaran", r"\bsaya tunggu ya\b", r"\btunggu.*bayar", r"\btunggu.*pembayaran", r"\bnanti saya bayar ya\b", r"\bada pembayaran masuk\b", r"\bakan dibayarkan\b", r"\bsudah dibayarkan\b", r"\bsaya sudah transfer\b", r"\bsaya kirim nanti\b", r"\bnanti saya kirim\b", r"\bsaya proses dulu\b", r"\bproses pembayaran\b", r"\bsaya akan bayarnya nanti\b", r"\bobayarnya nanti ya\b", r"\bbayarnya nanti saja\b", r"\bsaya transfer nanti\b", r"\bbukti transfer\b", r"\bbukti pembayaran\b", r"\bkirim.*bukti\b", r"\bkirim.*buktinya\b", r"\bterima kasih saya bayar\b", r"\bok saya transfer\b", r"\boke saya transfer\b", r"\biya saya transfer\b", r"\bya saya transfer\b", r"\bsaya akan kirim\b", r"\bsaya bayar jam [0-9]+", r"\bsaya transfer jam [0-9]+", r"\bnanti jam [0-9]+ saya bayar\b", r"\bjam [0-9]+ saya bayar\b", r"\bjam [0-9]+ ya saya bayar\b", r"\bsaya bayar besok\b", r"\bsaya bayar hari ini\b", r"\bsaya bayar nanti sore\b", r"\bsaya bayar nanti pagi\b", r"\bsaya bayar siang ini\b", r"\bpukul [0-9]+ saya bayar\b", r"\bjam [0-9]+ saya transfer\b", r"\bbesok saya bayar\b", r"\bhari ini saya bayar\b", r"\bnanti sore saya bayar\b", r"\bpasti di jam [0-9]+ sudah dibayarkan\b", r"\bsudah dibayarkan jam [0-9]+\b", r"\bakan dibayarkan jam [0-9]+\b", r"\bsaya bayar tanggal [0-9]+\b", r"\bsaya bayar minggu depan\b", r"\bsaya bayar lusa\b", r"\bbayar\b", r"\bdibayar\b", r"\btransfer\b", r"\bproses\b", r"\bpembayaran\b"]),
        ("confirm_time", [r"\bjam [0-9]+ ya\s*$", r"\bbesok jam [0-9]+\b", r"\bhari ini jam [0-9]+\b", r"\bnanti sore jam [0-9]+\b", r"\bjuga jam [0-9]+\b", r"\bjam [0-9]+\b", r"\bpukul [0-9]+\b", r"\btanggal [0-9]+\b", r"\bbesok\b", r"\bhari ini\b", r"\bnanti sore\b", r"\bnanti pagi\b", r"\bsiang ini\b", r"\bminggu depan\b", r"\blusa\b"]),
        ("greeting", [r"^\s*halo\b", r"^\s*hai\b", r"^\s*pagi\b", r"^\s*siang\b", r"^\s*sore\b", r"^\s*selamat pagi\b", r"^\s*selamat siang\b", r"^\s*selamat sore\b", r"^\s*selamat malam\b", r"^\s*apa kabar\b", r"^\s*hi\b", r"^\s*hello\b", r"^\s*maaf\b", r"^\s*terima kasih\b", r"^\s*selamat datang\b", r"^\s*(ya|iya|oke|baik|betul)\s*[,]?\s*selamat\b", r"^\s*(ya|iya|oke|baik|betul)\s*[,]?\s*pagi\b", r"^\s*(ya|iya|oke|baik|betul)\s*[,]?\s*siang\b", r"^\s*(ya|iya|oke|baik|betul)\s*[,]?\s*sore\b", r"^\s*(ya|iya|oke|baik|betul)\s*[,]?\s*malam\b", r"^\s*(ya|iya|oke|baik|betul)\s*[,]?\s*halo\b", r"^\s*(ya|iya|oke|baik|betul)\s*[,]?\s*hai\b", r"^\s*(ya|iya|oke|baik|betul)\s*[,]?\s*apa kabar\b", r"^\s*(ya|iya|oke|baik|betul)\s*[,]?\s*terima kasih\b", r"^\s*(ya|iya|oke|baik|betul)\s*[,]?\s*maaf\b", r".*\bselamat pagi\b", r".*\bselamat siang\b", r".*\bselamat sore\b", r".*\bselamat malam\b", r".*\bapa kabar\b", r".*\bmaaf\b", r".*\bterima kasih\b", r".*\bhalo\b", r".*\bhai\b"]),
        ("confirm_identity", [
            # 明确的身份确认表达，去掉纯确认词的规则，交给上下文逻辑和ML处理
            r"\bsaya adalah\b",
            r"\baiya ini\b",
            r"\bbetul saya\b",
            r"\biya, ini saya\b",
            r"\benar, saya yang\b",
            r"\biya betul\b(?!.*bayar|.*transfer|.*pembayaran|.*bayarnya|.*uang|.*rp|.*rupiah|.*tagihan|.*bukti|.*buktinya|.*proses|.*lunas|.*cicil|.*angsuran|.*selamat|.*pagi|.*siang|.*sore|.*malam|.*halo|.*hai|.*apa kabar|.*terima kasih|.*maaf|.*jam|.*tanggal|.*besok|.*nanti|.*rekening|.*bunga|.*denda|.*potong|.*keringanan|.*sakit|.*kehilangan|.*rumah|.*keluarga)",
            r"\bada apa\b",
            r"\bya ada apa\b",
            r"\biya ada apa\b",
            r"\bsaya iya\b(?!.*bayar|.*transfer|.*pembayaran|.*bayarnya|.*uang|.*rp|.*rupiah|.*tagihan|.*bukti|.*buktinya|.*proses|.*lunas|.*cicil|.*angsuran|.*selamat|.*pagi|.*siang|.*sore|.*malam|.*halo|.*hai|.*apa kabar|.*terima kasih|.*maaf|.*jam|.*tanggal|.*besok|.*nanti|.*rekening|.*bunga|.*denda|.*potong|.*keringanan|.*sakit|.*kehilangan|.*rumah|.*keluarga)",
            r"\bya saya\b(?!.*bayar|.*transfer|.*pembayaran|.*bayarnya|.*uang|.*rp|.*rupiah|.*tagihan|.*bukti|.*buktinya|.*proses|.*lunas|.*cicil|.*angsuran|.*selamat|.*pagi|.*siang|.*sore|.*malam|.*halo|.*hai|.*apa kabar|.*terima kasih|.*maaf|.*jam|.*tanggal|.*besok|.*nanti|.*rekening|.*bunga|.*denda|.*potong|.*keringanan|.*sakit|.*kehilangan|.*rumah|.*keluarga)",
            r"\biya saya\b(?!.*bayar|.*transfer|.*pembayaran|.*bayarnya|.*uang|.*rp|.*rupiah|.*tagihan|.*bukti|.*buktinya|.*proses|.*lunas|.*cicil|.*angsuran|.*selamat|.*pagi|.*siang|.*sore|.*malam|.*halo|.*hai|.*apa kabar|.*terima kasih|.*maaf|.*jam|.*tanggal|.*besok|.*nanti|.*rekening|.*bunga|.*denda|.*potong|.*keringanan|.*sakit|.*kehilangan|.*rumah|.*keluarga)",
            r"\bbetul ini\b",
            r"\biya betul ini\b",
            r"\bini saya\b",
            r"\bsaya yang\b",
            r"\bbiarkan saya\b",
            r"\bmau, saya\b",
            r"\bsaya mau\b(?!.*bayar|.*transfer|.*pembayaran|.*bayarnya|.*uang|.*rp|.*rupiah|.*tagihan|.*bukti|.*buktinya|.*proses|.*lunas|.*cicil|.*angsuran|.*selamat|.*pagi|.*siang|.*sore|.*malam|.*halo|.*hai|.*apa kabar|.*terima kasih|.*maaf|.*jam|.*tanggal|.*besok|.*nanti|.*rekening|.*bunga|.*denda|.*potong|.*keringanan|.*sakit|.*kehilangan|.*rumah|.*keluarga)",
            r"\bya betul\b(?!.*bayar|.*transfer|.*pembayaran|.*bayarnya|.*uang|.*rp|.*rupiah|.*tagihan|.*bukti|.*buktinya|.*proses|.*lunas|.*cicil|.*angsuran|.*selamat|.*pagi|.*siang|.*sore|.*malam|.*halo|.*hai|.*apa kabar|.*terima kasih|.*maaf|.*jam|.*tanggal|.*besok|.*nanti|.*rekening|.*bunga|.*denda|.*potong|.*keringanan|.*sakit|.*kehilangan|.*rumah|.*keluarga)",
            r"\biya bener\b(?!.*bayar|.*transfer|.*pembayaran|.*bayarnya|.*uang|.*rp|.*rupiah|.*tagihan|.*bukti|.*buktinya|.*proses|.*lunas|.*cicil|.*angsuran|.*selamat|.*pagi|.*siang|.*sore|.*malam|.*halo|.*hai|.*apa kabar|.*terima kasih|.*maaf|.*jam|.*tanggal|.*besok|.*nanti|.*rekening|.*bunga|.*denda|.*potong|.*keringanan|.*sakit|.*kehilangan|.*rumah|.*keluarga)",
            r"\bya bener\b(?!.*bayar|.*transfer|.*pembayaran|.*bayarnya|.*uang|.*rp|.*rupiah|.*tagihan|.*bukti|.*buktinya|.*proses|.*lunas|.*cicil|.*angsuran|.*selamat|.*pagi|.*siang|.*sore|.*malam|.*halo|.*hai|.*apa kabar|.*terima kasih|.*maaf|.*jam|.*tanggal|.*besok|.*nanti|.*rekening|.*bunga|.*denda|.*potong|.*keringanan|.*sakit|.*kehilangan|.*rumah|.*keluarga)",
            # 扩展：印尼口语常见身份确认表达
            r"\bya,? bisa\b",
            r"\biya,? bisa\b",
            r"\bini apa ya\b",
            r"\bapa ini ya\b",
            r"\bsaya cari,? pak\b",
            r"\bya saya cari\b",
            r"\biya saya cari\b",
            r"\bada yang bisa\b",
            r"\bya,? ada\b",
            r"\biya,? ada\b",
            r"\bsaya ada\b",
            r"\bada saya\b",
            r"\bentar ya\b(?!.*bayar|.*transfer)",
            r"\bsebentar ya\b(?!.*bayar|.*transfer)",
            r"\bsebentar ya\b(?!.*bayar|.*transfer)",
            r"\btunggu ya\b(?!.*bayar|.*transfer|.*uang)",
            r"\bsaya tunggu\b(?!.*bayar|.*uang)",
            r"\bmenunggu ya\b",
            r"\bok,? pak\b",
            r"\boke,? pak\b",
            r"\bbaik,? pak\b",
            r"\bsiapa\b",
            r"\bmana\b",
            r"\bada apa,? pak\b",
            r"\benar saya\b(?!.*bayar|.*transfer|.*pembayaran|.*bayarnya|.*uang|.*rp|.*rupiah|.*tagihan|.*bukti|.*buktinya|.*proses|.*lunas|.*cicil|.*angsuran|.*selamat|.*pagi|.*siang|.*sore|.*malam|.*halo|.*hai|.*apa kabar|.*terima kasih|.*maaf|.*jam|.*tanggal|.*besok|.*nanti|.*rekening|.*bunga|.*denda|.*potong|.*keringanan|.*sakit|.*kehilangan|.*rumah|.*keluarga)",
            r"\biya benar saya\b(?!.*bayar|.*transfer|.*pembayaran|.*bayarnya|.*uang|.*rp|.*rupiah|.*tagihan|.*bukti|.*buktinya|.*proses|.*lunas|.*cicil|.*angsuran|.*selamat|.*pagi|.*siang|.*sore|.*malam|.*halo|.*hai|.*apa kabar|.*terima kasih|.*maaf|.*jam|.*tanggal|.*besok|.*nanti|.*rekening|.*bunga|.*denda|.*potong|.*keringanan|.*sakit|.*kehilangan|.*rumah|.*keluarga)",
            r"\bya saya itu\b(?!.*bayar|.*transfer|.*pembayaran|.*bayarnya|.*uang|.*rp|.*rupiah|.*tagihan|.*bukti|.*buktinya|.*proses|.*lunas|.*cicil|.*angsuran|.*selamat|.*pagi|.*siang|.*sore|.*malam|.*halo|.*hai|.*apa kabar|.*terima kasih|.*maaf|.*jam|.*tanggal|.*besok|.*nanti|.*rekening|.*bunga|.*denda|.*potong|.*keringanan|.*sakit|.*kehilangan|.*rumah|.*keluarga)",
            r"\biya saya itu\b(?!.*bayar|.*transfer|.*pembayaran|.*bayarnya|.*uang|.*rp|.*rupiah|.*tagihan|.*bukti|.*buktinya|.*proses|.*lunas|.*cicil|.*angsuran|.*selamat|.*pagi|.*siang|.*sore|.*malam|.*halo|.*hai|.*apa kabar|.*terima kasih|.*maaf|.*jam|.*tanggal|.*besok|.*nanti|.*rekening|.*bunga|.*denda|.*potong|.*keringanan|.*sakit|.*kehilangan|.*rumah|.*keluarga)",
            r"\betul saya itu\b(?!.*bayar|.*transfer|.*pembayaran|.*bayarnya|.*uang|.*rp|.*rupiah|.*tagihan|.*bukti|.*buktinya|.*proses|.*lunas|.*cicil|.*angsuran|.*selamat|.*pagi|.*siang|.*sore|.*malam|.*halo|.*hai|.*apa kabar|.*terima kasih|.*maaf|.*jam|.*tanggal|.*besok|.*nanti|.*rekening|.*bunga|.*denda|.*potong|.*keringanan|.*sakit|.*kehilangan|.*rumah|.*keluarga)",
            r"\benar saya itu\b(?!.*bayar|.*transfer|.*pembayaran|.*bayarnya|.*uang|.*rp|.*rupiah|.*tagihan|.*bukti|.*buktinya|.*proses|.*lunas|.*cicil|.*angsuran|.*selamat|.*pagi|.*siang|.*sore|.*malam|.*halo|.*hai|.*apa kabar|.*terima kasih|.*maaf|.*jam|.*tanggal|.*besok|.*nanti|.*rekening|.*bunga|.*denda|.*potong|.*keringanan|.*sakit|.*kehilangan|.*rumah|.*keluarga)",
            r"\saya sendiri\b",
            r"\biya saya sendiri\b(?!.*bayar|.*transfer|.*pembayaran|.*bayarnya|.*uang|.*rp|.*rupiah|.*tagihan|.*bukti|.*buktinya|.*proses|.*lunas|.*cicil|.*angsuran|.*selamat|.*pagi|.*siang|.*sore|.*malam|.*halo|.*hai|.*apa kabar|.*terima kasih|.*maaf|.*jam|.*tanggal|.*besok|.*nanti|.*rekening|.*bunga|.*denda|.*potong|.*keringanan|.*sakit|.*kehilangan|.*rumah|.*keluarga)",
            r"\bya saya sendiri\b(?!.*bayar|.*transfer|.*pembayaran|.*bayarnya|.*uang|.*rp|.*rupiah|.*tagihan|.*bukti|.*buktinya|.*proses|.*lunas|.*cicil|.*angsuran|.*selamat|.*pagi|.*siang|.*sore|.*malam|.*halo|.*hai|.*apa kabar|.*terima kasih|.*maaf|.*jam|.*tanggal|.*besok|.*nanti|.*rekening|.*bunga|.*denda|.*potong|.*keringanan|.*sakit|.*kehilangan|.*rumah|.*keluarga)",
            r"\betul saya sendiri\b(?!.*bayar|.*transfer|.*pembayaran|.*bayarnya|.*uang|.*rp|.*rupiah|.*tagihan|.*bukti|.*buktinya|.*proses|.*lunas|.*cicil|.*angsuran|.*selamat|.*pagi|.*siang|.*sore|.*malam|.*halo|.*hai|.*apa kabar|.*terima kasih|.*maaf|.*jam|.*tanggal|.*besok|.*nanti|.*rekening|.*bunga|.*denda|.*potong|.*keringanan|.*sakit|.*kehilangan|.*rumah|.*keluarga)"
        ]),
        ("refuse_to_pay", [r"\btidak mau bayar\b", r"\bgak bayar\b", r"\btak bayar\b", r"\bsaya tidak akan bayar\b", r"\bsaya tidak mau membayar\b", r"\btidak usah ditagih\b", r"\bsaya tidak bayar\b", r"\btidak bayar\b", r"\bgak bisa bayar\b", r"\btidak bisa bayar\b", r"\bbelum bisa bayar\b", r"\bsaya belum bisa bayar\b", r"\bsaya gak bisa bayar\b", r"\bsaya tidak bisa bayar\b"]),
        ("ask_fee", [r"\bbunga berapa\b", r"\bdenda berapa\b", r"\bbiaya admin berapa\b", r"\bkenapa begitu besar\b", r"\bbiaya berapa\b"]),
        ("ask_payment_method", [r"\btransfer kemana\b", r"\brekening mana\b", r"\bnomor rekening\b", r"\bbayar kemana\b", r"\bagaimana cara bayar\b"]),
        ("already_paid", [r"\bsudah bayar\b", r"\bsudah transfer\b", r"\bsaya sudah bayar\b", r"\btadi sudah bayar\b", r"\bsudah dibayar\b"]),
        ("partial_payment", [r"\bmau bayar berapa\b", r"\bbisa bayar setengah dulu\b", r"\bbayar sebagian\b", r"\bcicil\b", r"\bayar sedikit dulu\b"]),
        ("third_party", [r"\bkeluarga dia\b", r"\borang tua dia\b", r"\banak dia\b", r"\bsaudara dia\b", r"\bdia tidak ada\b", r"\bsaya bukan orang yang anda cari\b", r"\bdia sedang keluar\b", r"\borangnya ga ada\b", r"\borangnya tidak ada\b"]),
        ("dont_know", [r"\btidak tahu\b", r"\bsaya tidak tahu\b", r"\btidak mengerti\b", r"\btidak paham\b", r"\bsaya tidak paham\b"]),
    ]

    # 付款请求相关的关键词，用于上下文判断
    PAYMENT_REQUEST_KEYWORDS = [
        r"\bbayar\b", r"\btagihan\b", r"\btransfer\b", r"\bpembayaran\b", r"\blunas\b",
        r"\bjam berapa\b", r"\bkapan\b", r"\bwaktu\b", r"\btanggal\b",
        r"\bbisa bayar\b", r"\bmau bayar\b", r"\bakan bayar\b",
    ]

    # 身份验证相关的关键词，用于上下文判断
    IDENTITY_REQUEST_KEYWORDS = [
        r"\bbisa bicara\b", r"\bapakah benar\b", r"\bini bapak\b", r"\bini ibu\b",
        r"\bsaya dari extra uang\b", r"\bpetugas extra uang\b",
        r"\bverifikasi identitas\b", r"\bkonfirmasi identitas\b",
    ]

    # 独立的确认词
    STANDALONE_CONFIRMATION_PATTERNS = [
        r"^\s*ya\s*[.,!?]?\s*$",
        r"^\s*iya\s*[.,!?]?\s*$",
        r"^\s*oke\s*[.,!?]?\s*$",
        r"^\s*ok\s*[.,!?]?\s*$",
        r"^\s*baik\s*[.,!?]?\s*$",
        r"^\s*betul\s*[.,!?]?\s*$",
    ]

    @classmethod
    def detect(cls, text: str, context: str = None) -> str:
        """识别用户意图，按照优先级顺序匹配
        :param context: 上下文，即上一条机器人的消息，用于歧义判断
        """
        import re
        if not text:
            return "unknown"

        text_lower = text.lower()

        # ========== 纯确认词特殊处理逻辑（最优先，消除歧义） ==========
        # 前置检查：如果文本包含支付相关关键词，跳过纯确认逻辑，避免误判
        payment_words = {"bayar", "transfer", "tagihan", "lunas", "waktu", "jam", "tanggal", "rekening", "bukti"}
        has_payment_word = any(word in text_lower for word in payment_words)

        # 判断是否是纯确认词（只有ya/oke/baik/betul等，支持重复如"oke oke"、"ya ya"）
        pure_confirmation_pattern = r"^\s*(ya|iya|oke|ok|baik|betul|benar)(\s*[,]?\s*(ya|iya|oke|ok|baik|betul|benar|pak|bu|tunggu))*\s*[.,!?]?\s*$"
        is_pure_confirmation = bool(re.search(pure_confirmation_pattern, text_lower, re.IGNORECASE)) and not has_payment_word

        if is_pure_confirmation and context is not None:
            context_lower = context.lower()
            # 优先判断上下文是否是身份验证相关（避免身份问题带支付关键词时误判）
            identity_keywords = [
                r"\bbapak\b", r"\bpak\b", r"\bibu\b", r"\bbu\b", r"\bsiapa\b", r"\bapakah\b",
                r"\bbenar\b", r"\bini anda\b", r"\bini dengan\b", r"\bverifikasi\b",
                r"\bkenalan\b", r"\bnama\b", r"\bbisa bicara\b", r"\byang punya pinjaman\b"
            ]
            for pattern in identity_keywords:
                if re.search(pattern, context_lower, re.IGNORECASE):
                    return "confirm_identity"

            # 再判断上下文是否是支付/还款相关请求
            payment_keywords = [
                r"\bbayar\b", r"\btagihan\b", r"\btransfer\b", r"\bpembayaran\b", r"\blunas\b",
                r"\brekening\b", r"\bbukti transfer\b", r"\bbayar kapan\b", r"\bbisa bayar\b",
                r"\bmau bayar\b", r"\bjumlah tagihan\b"
            ]
            for pattern in payment_keywords:
                if re.search(pattern, context_lower, re.IGNORECASE):
                    return "agree_to_pay"

            # 没有匹配到任何上下文关键词时，默认短确认是身份确认（大部分场景下是对身份问题的回答）
            # 注：短确认回复问候时通常也是身份确认，如"Halo Pak" → "Ya?"，所以不需要单独处理问候场景
            return "confirm_identity"

        # ========== 正常的模式匹配 ==========
        for intent, patterns in cls.INTENT_PATTERNS:
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return intent

        # ========== ML fallback：规则未匹配到时，尝试用ML分类器预测 ==========
        if cls._use_ml_fallback and cls._ml_classifier is not None:
            try:
                predictions = cls._ml_classifier.predict(text, top_k=1)
                if predictions:
                    intent, confidence = predictions[0]
                    if confidence >= cls._ml_threshold:
                        return intent
            except Exception as e:
                # ML预测失败时不报错，还是返回unknown
                pass

        return "unknown"

    @classmethod
    def load_ml_classifier(cls, model_path: str = 'models/simple_intent_classifier.pkl') -> bool:
        """加载ML分类器模型
        :return: 加载成功返回True，失败返回False
        """
        if not ML_CLASSIFIER_AVAILABLE:
            print("警告: scikit-learn未安装，无法使用ML分类器功能")
            return False

        try:
            cls._ml_classifier = SimpleIntentClassifier.load_model(model_path)
            print(f"ML意图分类器加载成功，支持 {len(cls._ml_classifier.intent_labels)} 种意图")
            return True
        except Exception as e:
            print(f"ML分类器加载失败: {e}")
            return False

    @classmethod
    def set_ml_threshold(cls, threshold: float):
        """设置ML分类结果的置信度阈值"""
        cls._ml_threshold = max(0.0, min(1.0, threshold))

    @classmethod
    def enable_ml_fallback(cls, enable: bool = True):
        """启用或禁用ML fallback功能"""
        cls._use_ml_fallback = enable


class CollectionChatBot:
    """催收对话机器人 - 增强版"""

    def __init__(self, chat_group: str = "H2", customer_name: Optional[str] = None,
                 overdue_amount: int = 500000, overdue_days: int = 5,
                 new_flag: int = 0, strategy_profile=None,
                 user_memory=None):  # P15-D01: 跨会话用户记忆
        # chat_group 催收阶段: H2=宽限期前2天(温和引导), H1=宽限期前1天(引导+暗示后果), S0=实质性逾期(高压催收)
        self.chat_group = chat_group
        self.customer_name = customer_name or "Pak/Bu"
        self.overdue_amount = overdue_amount  # 欠款金额，默认500k
        self.overdue_days = overdue_days  # 逾期天数，默认5天
        # P15-B01: 分客群策略配置
        from core.strategy_profile import get_strategy_profile, StrategyProfile
        if strategy_profile is not None:
            self.strategy = copy.copy(strategy_profile)
        else:
            self.strategy = copy.copy(get_strategy_profile(new_flag, chat_group))
        self.extension_fee = int(overdue_amount * self.strategy.extension_fee_ratio)
        self.state: ChatState = ChatState.INIT
        self.conversation: List[ChatTurn] = []
        self.commit_time: Optional[str] = None
        self.extension_agreed: bool = False
        self.objection_count: int = 0
        self.max_objections: int = self.strategy.max_objections
        # P15-A01: 分类型异议递进计数器
        self.no_money_count: int = 0
        self.busy_count: int = 0
        self.dont_know_count: int = 0
        self.push_round: int = 0  # P15-B02: PUSH_FOR_TIME 中已执行的 push 次数
        self.user_intent: str = ""
        # 对话记忆
        self.user_history_intents: List[str] = []  # 用户历史意图列表
        self.user_asked_amount: bool = False  # 用户是否已经询问过金额
        self.user_asked_fee: bool = False  # 用户是否已经询问过费用
        self.user_asked_payment_method: bool = False  # 用户是否已经询问过还款方式
        self.user_mentioned_no_money: bool = False  # 用户是否已经提到过没钱
        self.user_mentioned_busy: bool = False  # 用户是否已经提到过忙碌
        self.partial_payment_discussed: bool = False  # 是否已经讨论过分部分还款
        self.extension_discussed: bool = False  # 是否已经讨论过展期

        # 组件
        self.tts = TextToSpeech()
        self.var_replacer = VariableReplacer()
        self.time_detector = TimeDetector()
        self.intent_detector = IntentDetector()
        self.asr_corrector = ASRCorrector()

        # 会话ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 回复去重：记录最近2次回复，避免连续重复
        self.last_responses: List[str] = []

        # LLM Fallback 相关
        self.llm_enabled: bool = False
        self.llm_config = None
        self.llm_provider = None
        self.fallback_detector = None
        self.in_llm_fallback: bool = False
        self.llm_turn_count: int = 0
        self.llm_used_this_turn: bool = False  # 标记当前轮次是否使用了 LLM

        # 沉默处理
        self.silence_count: int = 0  # 连续沉默轮次计数

        # P15-D01: 跨会话用户记忆
        self.user_memory = user_memory
        self.customer_phone = user_memory.phone if user_memory else None
        self._is_returning_customer = False

        if user_memory and user_memory.has_previous_sessions:
            self._is_returning_customer = True
            if user_memory.previously_promised_but_failed:
                self.strategy.push_intensity = min(5, self.strategy.push_intensity + 1)
                self.strategy.consequence_emphasis = min(5, self.strategy.consequence_emphasis + 1)
            if user_memory.is_low_trust:
                self.strategy.max_push_rounds = min(5, self.strategy.max_push_rounds + 1)

            # P15-H05: T3 跨通话轨迹演化策略调整
            if user_memory.trajectory_adjustments:
                adj = user_memory.trajectory_adjustments
                if adj.get("approach"):
                    self.strategy.approach = adj["approach"]
                if adj.get("tone"):
                    self.strategy.tone = adj["tone"]
                self.strategy.push_intensity = max(1, min(5,
                    self.strategy.push_intensity + adj.get("push_intensity_delta", 0)))
                self.strategy.consequence_emphasis = max(1, min(5,
                    self.strategy.consequence_emphasis + adj.get("consequence_emphasis_delta", 0)))
                self.strategy.max_push_rounds = max(1, min(5,
                    self.strategy.max_push_rounds + adj.get("max_push_rounds_delta", 0)))
                if adj.get("extension_priority") is not None:
                    self.strategy.extension_priority = adj["extension_priority"]
                if adj.get("education_emphasis") is not None:
                    self.strategy.education_emphasis = adj["education_emphasis"]
                if adj.get("relationship_emphasis") is not None:
                    self.strategy.relationship_emphasis = adj["relationship_emphasis"]
                self.strategy.extension_fee_ratio = max(0.10, min(0.40,
                    self.strategy.extension_fee_ratio + adj.get("extension_fee_ratio_delta", 0.0)))

        # 话术库 - 扩展版
        self._init_script_lib()

    def enable_ml_intent_classification(self, model_path: str = 'models/simple_intent_classifier.pkl', threshold: float = 0.6) -> bool:
        """启用ML意图分类作为规则系统的fallback
        :param model_path: 模型文件路径
        :param threshold: 置信度阈值
        :return: 启用成功返回True
        """
        success = IntentDetector.load_ml_classifier(model_path)
        if success:
            IntentDetector.set_ml_threshold(threshold)
            IntentDetector.enable_ml_fallback(True)
        return success

    def enable_llm_fallback(self, config=None) -> bool:
        """启用 LLM Fallback 功能
        :param config: LLMConfig 实例，为 None 时从环境变量读取
        :return: 启用成功返回 True
        """
        if not LLM_FALLBACK_AVAILABLE:
            print("[LLM] LLM Fallback 模块不可用")
            return False

        if config is None:
            config = LLMConfig.from_env()

        self.llm_config = config
        self.llm_provider = LLMProvider(config)
        self.fallback_detector = FallbackDetector()
        self.llm_enabled = True

        if config.is_mock:
            print(f"[LLM] LLM Fallback 已启用 (mock 模式)")
        else:
            print(f"[LLM] LLM Fallback 已启用 (provider={config.provider}, model={config.model})")
        return True

    def _init_script_lib(self):
        """初始化话术库"""
        self.script_lib = {
            "greeting": {
                "H2": ["Halo?", "Halo.", "Halo, selamat pagi.", "Halo, apa kabar?"],
                "H1": ["Halo?", "Halo, selamat pagi.", "Halo, selamat siang.", "Halo, apa kabar?"],
                "S0": ["Halo?", "Halo.", "Halo, selamat sore.", "Halo, apa kabar?"]
            },
            "identity_verify": {
                "H2": [
                    "Halo, selamat pagi {name}. Saya dari aplikasi Extra Uang, bisa bicara dengan {name} sendiri?",
                    "Halo {name}, saya dari Extra Uang. Apakah saya berbicara dengan {name} yang punya pinjaman di aplikasi kami ya?",
                    "Halo, selamat pagi. Saya petugas dari Extra Uang, bisa bicara dengan Bapak/Ibu {name}?"
                ],
                "H1": [
                    "Halo, selamat siang {name}. Saya dari aplikasi Extra Uang, apakah ini benar dengan {name}?",
                    "Halo {name}, saya dari Extra Uang. Saya menelpon tentang tagihan pinjaman Anda yang sudah jatuh tempo ya.",
                    "Selamat siang, saya petugas dari Extra Uang. Apakah ini Bapak/Ibu {name}?"
                ],
                "S0": [
                    "Halo, selamat sore {name}. Saya dari aplikasi Extra Uang, apakah saya berbicara dengan {name}?",
                    "Halo {name}, saya dari Extra Uang. Saya menelpon tentang tagihan pinjaman Anda yang sudah lama jatuh tempo ya.",
                    "Selamat sore, saya petugas dari Extra Uang. Bisakah saya bicara dengan Bapak/Ibu {name}?"
                ]
            },
            "educate_intro": {
                "*": [
                    "Saya akan bantu jelaskan informasi tagihan Anda dan solusi yang tersedia ya. Tujuan kami membantu Anda menjaga riwayat kredit tetap baik.",
                    "Ibu/Bapak baru pertama kali mendapat telepon dari kami, mohon maaf mengganggu. Saya akan jelaskan status tagihan dan cara pembayarannya ya.",
                    "Kami menghubungi untuk membantu Anda menyelesaikan tagihan tepat waktu demi menjaga skor kredit. Bisa saya jelaskan status tagihannya?"
                ]
            },
            "purpose": {
                "H2": [
                    "Saya menelpon untuk memberitahu bahwa tagihan pinjaman {name} sebesar Rp {amount} sudah jatuh tempo selama {days} hari ya.",
                    "Tentang pinjaman Anda di Extra Uang, sekarang sudah jatuh tempo {days} hari dengan total tagihan Rp {amount} ya.",
                    "Saya ingin memberitahu bahwa tagihan pinjaman Anda sebesar Rp {amount} sudah harus dibayar sekarang ya."
                ],
                "H1": [
                    "Saya dari Extra Uang, ingin memberitahu bahwa tagihan {name} sebesar Rp {amount} sudah jatuh tempo lebih dari 30 hari ya.",
                    "Tentang pinjaman Anda, sudah lebih dari 30 hari jatuh tempo dengan total tagihan Rp {amount}. Kami perlu segera menyelesaikan ini ya.",
                    "Saya menelpon tentang tagihan pinjaman Anda yang sudah lewat jatuh tempo lebih dari sebulan, totalnya Rp {amount} ya."
                ],
                "S0": [
                    "Kita bicara tentang pinjaman {name} yang sudah jatuh tempo lebih dari 90 hari ya, jumlahnya Rp {amount}.",
                    "Pinjaman Anda di Extra Uang sudah jatuh tempo lebih dari 3 bulan dengan total tagihan Rp {amount}. Kita harus segera cari solusi ya.",
                    "Saya menelpon tentang tagihan pinjaman Anda yang sudah lama tidak dibayar, totalnya Rp {amount} ya."
                ]
            },
            "ask_time": {
                "H2": [
                    "Kira-kira kapan {name} bisa melunasi tagihan ini ya?",
                    "Ada rencana bayar jam berapa ya?",
                    "Kapan Bapak/Ibu bisa melakukan pembayaran untuk tagihan ini ya?",
                    "Untuk tagihan Rp {amount} ini, kira-kira bisa dibayar kapan ya?"
                ],
                "H1": [
                    "Kapan {name} bisa melakukan pembayaran tagihan ini?",
                    "Jam berapa tepatnya bisa bayar?",
                    "Kira-kira hari apa Bapak/Ibu bisa membayar tagihan ini ya?",
                    "Untuk tagihan yang sudah jatuh tempo ini, ada rencana bayar kapan ya?"
                ],
                "S0": [
                    "Bagaimana rencana pembayaran {name} untuk tagihan ini ya?",
                    "Kapan bisa bayar ya?",
                    "Kira-kira kapan Bapak/Ibu bisa menyelesaikan tagihan ini ya?",
                    "Untuk tagihan yang sudah lama ini, kita harus segera selesaikan. Kira-kira bisa bayar kapan ya?"
                ]
            },
            "push": {
                "H2": [
                    "Jam berapa tepatnya ya {name}?",
                    "Hari ini jam berapa bisa bayar?",
                    "Bisa kasih tahu jam berapa pasti bisa bayar ya?",
                    "Untuk hari ini, kira-kira jam berapa ya Bapak/Ibu bisa transfer?"
                ],
                "H1": [
                    "Jam berapa tepatnya {name} bisa bayar?",
                    "Besok jam berapa ya?",
                    "Bisa kasih kepastian jam berapa ya?",
                    "Kita butuh kepastian waktu pembayaran ya, kira-kira jam berapa?"
                ],
                "S0": [
                    "Jam berapa tepatnya ya?",
                    "Hari apa bisa bayar ya?",
                    "Bisa kasih tahu pasti hari apa dan jam berapa ya?",
                    "Kita harus segera selesaikan ini, bisa kasih tahu kapan pastinya bisa bayar?"
                ]
            },
            "push_hard": {
                "*": [
                    "Bapak/Ibu, jika tidak dibayar hari ini denda akan terus bertambah dan bisa berdampak pada skor kredit Anda di OJK. Kapan pastinya bisa bayar?",
                    "Saya perlu kepastian sekarang. Tagihan ini sudah sangat terlambat. Jam berapa hari ini Bapak/Ibu bisa transfer?",
                    "Ini panggilan terakhir untuk tagihan Anda. Jika tidak ada pembayaran dalam 24 jam, kami harus melanjutkan ke proses selanjutnya. Bisa bayar jam berapa?"
                ]
            },
            "push_final": {
                "*": [
                    "Bapak/Ibu, ini adalah peringatan terakhir. Jika tagihan tidak diselesaikan hari ini, kami terpaksa melanjutkan ke proses penagihan selanjutnya sesuai prosedur. Kapan pastinya bisa bayar?",
                    "Saya sudah memberikan beberapa kesempatan. Jika tidak ada pembayaran dalam waktu dekat, akun Anda akan masuk ke proses penagihan lanjutan. Ini kesempatan terakhir, kapan bisa bayar?",
                    "Baik, saya catat bahwa Bapak/Ibu belum bisa memberikan kepastian pembayaran. Perlu diketahui ini akan mempengaruhi status kredit Anda. Ada yang bisa saya bantu untuk menyelesaikan ini hari ini?"
                ]
            },
            "escalate_to_supervisor": {
                "*": [
                    "Baik, karena kita belum mencapai kesepakatan hari ini, kasus ini akan saya teruskan ke supervisor untuk penanganan lebih lanjut. Anda akan dihubungi kembali dalam 1-2 hari kerja. Terima kasih.",
                    "Saya akan mencatat bahwa hari ini belum ada kesepakatan pembayaran. Kasus Anda akan ditinjau oleh tim penagihan senior dan mereka akan menghubungi Anda kembali. Terima kasih atas waktunya."
                ]
            },
            "commit_time": {
                "H2": [
                    "Oke, saya catat ya {name} akan bayar {time}.",
                    "Baik, saya catat bahwa Anda akan membayar pada {time} ya.",
                    "Oke, saya tunggu pembayaran Anda pada {time} ya."
                ],
                "H1": [
                    "Ya, ya. Oke, {time} ya {name}, saya tunggu pembayarannya.",
                    "Baik, saya catat jadwal pembayaran Anda pada {time} ya.",
                    "Oke, saya akan tunggu pembayaran Anda sampai {time} ya."
                ],
                "S0": [
                    "Ya, ya, ya. Oke, {time} ya {name}.",
                    "Baik, saya catat bahwa Anda akan membayar pada {time} ya.",
                    "Oke, saya harap Anda benar-benar akan membayar pada {time} ya."
                ]
            },
            "confirm_commit": {
                "H2": [
                    "Jadi {name} setuju akan membayar tagihan sebesar Rp {amount} pada {time} ya?",
                    "Untuk konfirmasi, Bapak/Ibu akan membayar tagihan Rp {amount} pada {time} ya?",
                    "Jadi kesepakatannya Anda akan membayar pada {time} ya?"
                ],
                "H1": [
                    "Apakah benar {name} akan membayar tagihan ini pada {time} ya?",
                    "Untuk memastikan, Bapak/Ibu benar-benar akan membayar pada {time} ya?",
                    "Jadi Anda setuju untuk membayar tagihan ini pada {time} ya?"
                ],
                "S0": [
                    "Jadi kesepakatannya {name} akan membayar tagihan ini pada {time} ya?",
                    "Untuk konfirmasi terakhir, Anda akan membayar tagihan ini pada {time} ya?",
                    "Jadi Anda benar-benar akan membayar pada {time} ya?"
                ]
            },
            "wait": {
                "H2": [
                    "Saya tunggu pembayarannya ya {name}.",
                    "Saya tunggu sampai {time} ya.",
                    "Saya harap Anda benar-benar akan membayar pada {time} ya.",
                    "Terima kasih atas kerjasamanya, saya tunggu pembayarannya ya."
                ],
                "H1": [
                    "Saya tunggu pembayarannya ya {name}.",
                    "Saya akan menunggu sampai {time} ya.",
                    "Saya harap Anda memenuhi janji untuk membayar pada {time} ya."
                ],
                "S0": [
                    "Saya tunggu ya {name}.",
                    "Saya harap Anda benar-benar akan membayar pada {time} ya.",
                    "Terima kasih atas kesediaannya untuk menyelesaikan tagihan ini."
                ]
            },
            "closing": {
                "H2": [
                    "Terima kasih atas kerjasamanya {name}. Selamat pagi.",
                    "Terima kasih {name}, sampai jumpa.",
                    "Sukses selalu untuk Anda ya, terima kasih.",
                    "Selamat tinggal, semoga hari Anda menyenangkan."
                ],
                "H1": [
                    "Terima kasih {name}. Selamat siang.",
                    "Terima kasih atas kerjasamanya.",
                    "Sukses selalu untuk Anda ya, terima kasih.",
                    "Selamat tinggal, semoga hari Anda menyenangkan."
                ],
                "S0": [
                    "Terima kasih {name}. Selamat sore.",
                    "Terima kasih atas perhatiannya.",
                    "Sukses selalu untuk Anda ya, terima kasih.",
                    "Selamat tinggal, semoga masalah keuangan Anda segera selesai."
                ]
            },
            "closing_warm": {
                "*": [
                    "Terima kasih banyak {name}, kami hargai kerjasama Anda selama ini. Sampai jumpa dan sehat selalu ya.",
                    "Terima kasih {name}, senang bisa membantu. Kalau ada pertanyaan, silakan hubungi kami kapan saja ya.",
                    "Baik {name}, terima kasih sudah menjadi nasabah setia kami. Semoga bisnis Anda lancar terus ya."
                ]
            },
            "closing_wrong_number": {
                "*": [
                    "Mohon maaf atas ketidaknyamanannya ya, sepertinya saya salah nomor. Terima kasih.",
                    "Maaf ya, sepertinya saya menghubungi nomor yang salah. Mohon maaf atas gangguannya.",
                    "Saya minta maaf, sepertinya saya salah nomor. Terima kasih atas waktunya."
                ]
            },
            "closing_busy": {
                "*": [
                    "Baik {name}, saya akan hubungi kembali nanti ya. Terima kasih.",
                    "Oke, kalau sedang sibuk saya akan telepon lagi nanti ya. Terima kasih.",
                    "Saya mengerti Anda sedang sibuk, saya akan hubungi kembali besok ya. Terima kasih."
                ]
            },
            # 异议处理话术
            "answer_amount": {
                "*": [
                    "Tagihan {name} sebesar Rp {amount} ya, itu termasuk pokok pinjaman dan biaya administrasi.",
                    "Total tagihan Anda adalah Rp {amount}, termasuk pokok pinjaman dan biaya layanan ya.",
                    "Jumlah yang harus Anda bayar adalah Rp {amount} ya, itu sudah termasuk semua biaya."
                ]
            },
            "explain_extension": {
                "*": [
                    "Jika {name} mengalami kesulitan untuk membayar penuh, kami menyediakan opsi perpanjangan dengan biaya administrasi sebesar Rp {extension_fee} saja ya. Dengan itu, tanggal jatuh tempo akan diundur 30 hari lagi.",
                    "Kalau Anda tidak bisa membayar penuh sekarang, kami punya opsi perpanjangan dengan biaya tambahan Rp {extension_fee} saja. Jadi Anda punya waktu 30 hari lagi untuk membayar ya.",
                    "Untuk mengurangi beban Anda, kami menawarkan opsi perpanjangan dengan biaya administrasi Rp {extension_fee}. Dengan itu, Anda bisa membayar nanti 30 hari lagi ya."
                ]
            },
            "confirm_extension": {
                "*": [
                    "Apakah {name} setuju untuk mengambil opsi perpanjangan ini ya? Jika setuju, saya akan proses sekarang.",
                    "Jadi Anda memilih opsi perpanjangan ya? Kalau setuju saya akan segera proses untuk Anda.",
                    "Anda setuju untuk mengambil opsi perpanjangan dengan biaya Rp {extension_fee} ya? Kalau ya saya akan proses sekarang."
                ]
            },
            "answer_identity": {
                "*": [
                    "Saya adalah petugas penagihan dari aplikasi Extra Uang ya {name}. Saya menelpon tentang tagihan pinjaman {name} yang sudah jatuh tempo.",
                    "Saya dari tim penagihan Extra Uang ya, menelpon tentang tagihan pinjaman Anda yang sudah jatuh tempo.",
                    "Perkenalkan saya Budi dari Extra Uang, saya menelpon tentang tagihan pinjaman Anda yang sudah lewat jatuh tempo ya."
                ]
            },
            "handle_no_money": {
                "*": [
                    "Saya mengerti {name} sedang mengalami kesulitan keuangan. Apakah {name} bisa membayar sebagian dulu, atau mengambil opsi perpanjangan yang biayanya lebih ringan?",
                    "Saya paham sedang sulit uang ya. Apakah Anda bisa membayar sebagian dulu, atau mau ambil opsi perpanjangan yang cicilannya lebih kecil?",
                    "Saya mengerti kondisinya. Tapi kita harus cari solusi ya. Apakah Anda bisa membayar sedikit dulu, atau mau ambil opsi perpanjangan?"
                ]
            },
            "handle_no_money_level2": {
                "*": [
                    "Baik {name}, kalau belum bisa bayar penuh, bagaimana kalau bayar sebagian dulu? Rp100.000 atau Rp200.000 tidak apa-apa, yang penting ada itikad baik dulu ya.",
                    "Saya kasih saran ya {name}, daripada didenda terus, mending bayar sebagian kecil dulu. Berapa yang {name} sanggup hari ini?",
                    "Gini saja {name}, coba lihat dulu saldo sekarang. Kalau ada Rp50.000 atau Rp100.000, bayar itu dulu sebagai tanda itikad baik. Nanti sisanya bisa menyusul."
                ]
            },
            "handle_no_money_level3": {
                "*": [
                    "Saya sudah coba bantu {name} dengan opsi perpanjangan dan cicilan. Tapi kalau tidak ada pembayaran sama sekali, konsekuensinya tagihan ini akan terus bertambah dendanya dan riwayat kredit {name} di OJK bisa terpengaruh. Ini serius ya, Pak/Bu.",
                    "Baik {name}, saya sudah tawarkan berbagai solusi. Perlu {name} tahu, kalau tagihan ini tidak diurus, dendanya bertambah setiap hari dan nanti bisa masuk daftar hitam BI Checking. Masih ada waktu sebelum itu terjadi.",
                    "{name}, ini peringatan dari saya dengan itikad baik. Tanpa pembayaran atau perpanjangan, sistem otomatis akan melaporkan status kredit {name} ke OJK. Saya tidak ingin itu terjadi. Jadi bagaimana baiknya?"
                ]
            },
            "handle_busy_level2": {
                "*": [
                    "Baik {name}, saya catat ya. Saya telepon lagi nanti jam {time_suggestion}. Tolong diangkat ya, hanya perlu 2 menit untuk urus pembayarannya.",
                    "Saya paham {name} sibuk. Nanti saya hubungi lagi sekitar jam 3 sore, atau {name} bisa tentukan sendiri jam yang enak. Yang penting hari ini kita dapat kepastian ya.",
                    "Tidak apa-apa {name}, kita jadwal ulang. Hari ini jam berapa {name} ada waktu 2 menit? Saya sesuaikan dengan jadwal {name}."
                ]
            },
            "handle_dont_know_level2": {
                "*": [
                    "Baik {name}, coba sekarang buka aplikasi Extra Uang ya. Di halaman utama ada detail tagihan yang sedang berjalan. Saya tunggu sebentar.",
                    "{name}, untuk memastikan, coba cek aplikasi Extra Uang sekarang. Di menu 'Tagihan Saya' akan muncul detail pinjaman yang belum lunas. Nanti kita lanjutkan setelah {name} cek.",
                    "Saran saya {name}, buka dulu aplikasinya. Kalau sudah ketemu, kita bisa langsung urus pembayarannya sekarang juga. Gimana?"
                ]
            },
            "handle_threat": {
                "*": [
                    "Mohon maaf jika ada yang tidak berkenan ya {name}. Saya hanya ingin membantu menyelesaikan masalah tagihan ini dengan baik.",
                    "Maaf jika Anda merasa terganggu ya. Tujuan saya hanya ingin membantu mencari solusi terbaik untuk masalah tagihan Anda.",
                    "Saya minta maaf jika ada perkataan yang tidak berkenan. Mari kita bicara baik-baik untuk mencari solusi ya."
                ]
            },
            "handle_user_abuse": {
                "*": [
                    "Mohon bicara yang baik ya Bapak/Ibu. Saya di sini hanya ingin membantu menyelesaikan masalah tagihan Anda.",
                    "Saya mengerti Anda emosi, tapi mari kita bicara dengan baik ya. Tujuan saya hanya ingin mencari solusi terbaik untuk Anda.",
                    "Maaf jika Anda merasa kesal, tapi tolong bicara yang sopan ya. Saya di sini untuk membantu Anda menyelesaikan masalah tagihan."
                ]
            },
            "handle_wrong_number": {
                "*": [
                    "Mohon maaf ya, sepertinya saya salah nomor. Terima kasih atas waktunya.",
                    "Maaf ya, saya mencari Bapak/Ibu {name}. Sepertinya ini nomor yang salah. Mohon maaf atas gangguannya.",
                    "Saya minta maaf, sepertinya saya menghubungi nomor yang salah. Terima kasih."
                ]
            },
            "objection_general": {
                "H2": [
                    "Saya mengerti {name} keberatan, tapi kita harus selesaikan tagihan ini ya. Kira-kira kapan bisa bayar?",
                    "Paham, tapi bagaimana rencana pembayarannya ya?",
                    "Saya mengerti kondisinya, tapi tagihan ini harus segera diselesaikan ya. Kira-kira ada rencana bayar kapan?",
                    "Paham, mari kita cari solusi terbaik ya. Kira-kira kapan Anda bisa membayar tagihan ini?"
                ],
                "H1": [
                    "Saya mengerti kondisinya {name}, tapi tagihan ini harus segera diselesaikan ya. Kapan bisa bayar?",
                    "Saya paham Anda punya alasan, tapi kita harus selesaikan tagihan ini ya. Kira-kira bisa bayar kapan?",
                    "Saya mengerti, tapi tagihan ini sudah lama jatuh tempo ya. Kita harus segera cari solusi. Kira-kira kapan bisa bayar?"
                ],
                "S0": [
                    "Paham, tapi kita harus cari solusi untuk tagihan ini ya. Bagaimana rencananya?",
                    "Saya mengerti, tapi tagihan ini sudah lebih dari 3 bulan jatuh tempo ya. Kita harus segera selesaikan. Bagaimana rencana pembayarannya?",
                    "Paham, tapi ini sudah terlalu lama ya. Mari kita cari solusi yang terbaik untuk kedua pihak ya. Bagaimana rencana Anda?"
                ]
            },
            "answer_fee": {
                "*": [
                    "Biaya tersebut termasuk biaya administrasi dan biaya keterlambatan sesuai dengan perjanjian pinjaman yang Anda setujui sebelumnya.",
                    "Total biaya ini termasuk pokok pinjaman, bunga, dan biaya keterlambatan ya, sesuai dengan kesepakatan awal pinjaman Anda.",
                    "Biaya ini sudah sesuai dengan perjanjian yang Anda tandatangani saat mengambil pinjaman ya, termasuk biaya administrasi dan denda keterlambatan."
                ]
            },
            "answer_payment_method": {
                "*": [
                    "Anda bisa membayar melalui rekening resmi kami: BCA 1234567890 a.n. PT Extra Uang Indonesia. Pastikan nama penerima sesuai ya.",
                    "Pembayaran bisa dilakukan melalui transfer ke rekening BCA 1234567890 atas nama PT Extra Uang Indonesia ya.",
                    "Untuk membayar, silakan transfer ke rekening resmi kami: BCA 1234567890 a.n. PT Extra Uang Indonesia. Jangan lupa konfirmasi setelah transfer ya."
                ]
            },
            "handle_already_paid": {
                "*": [
                    "Terima kasih sudah membayar, kami akan segera memverifikasi pembayaran Anda. Mohon maaf atas gangguannya.",
                    "Oh, terima kasih ya. Kami akan cek segera pembayaran Anda. Mohon maaf sudah mengganggu waktu Anda.",
                    "Baik, terima kasih sudah membayar. Tim kami akan memverifikasi pembayaran Anda secepatnya. Terima kasih atas kerjasamanya."
                ]
            },
            "handle_partial_payment": {
                "*": [
                    "Jika Anda ingin membayar sebagian, minimal pembayaran adalah 30% dari jumlah tagihan. Berapa jumlah yang ingin Anda bayar sekarang, dan kapan waktunya?",
                    "Untuk pembayaran sebagian, minimal Anda harus membayar 30% dari total tagihan ya. Berapa yang ingin Anda bayar sekarang, dan kapan waktunya?",
                    "Kami mendukung pembayaran sebagian dengan minimal 30% dari total tagihan. Berapa jumlah yang ingin Anda bayar, dan kapan Anda bisa membayar nya?"
                ]
            },
            "handle_third_party": {
                "*": [
                    "Mohon maaf mengganggu, terima kasih.",
                    "Maaf sudah mengganggu ya, terima kasih atas waktunya.",
                    "Mohon maaf atas gangguannya, terima kasih."
                ]
            },
            "handle_dont_know": {
                "*": [
                    "Mohon maaf, saya ulangi ya: Kami dari Extra Uang, menelpon tentang tagihan pinjaman Bapak/Ibu {name} yang sudah jatuh tempo.",
                    "Maaf jika penjelasan saya kurang jelas ya. Saya ulangi: Saya dari Extra Uang, menelpon tentang tagihan pinjaman {name} yang sudah jatuh tempo.",
                    "Mohon maaf, saya akan ulangi lagi: Kami dari Extra Uang, menghubungi Anda tentang tagihan pinjaman yang sudah jatuh tempo ya."
                ]
            },
            "handle_unknown": {
                "*": [
                    "Mohon maaf, saya tidak mengerti maksud Anda. Bisa diulangi lagi ya?",
                    "Maaf, saya kurang paham apa yang Anda maksud. Bisa diulangi sekali lagi ya?",
                    "Mohon maaf, saya tidak menangkap maksud Anda. Bisakah Anda ulangi sekali lagi?"
                ]
            },
            "silence_level_1": {
                "*": [
                    "Halo Bapak/Ibu {name}, bisa dengar suara saya? Bilang 'ya' atau 'bisa' saja ya.",
                    "Halo {name}, saya ada informasi penting soal tagihan Anda. Bisa dengarkan sebentar?",
                    "Selamat pagi {name}, saya dari Extra Uang. Bapak/Ibu ada di telepon?"
                ]
            },
            "silence_level_2": {
                "*": [
                    "Baik, sambil Bapak/Ibu mendengarkan, izinkan saya informasikan bahwa tagihan Anda saat ini sebesar Rp {amount} dan sudah lewat jatuh tempo {days} hari ya. Kalau sudah paham, bilang 'ya' saja ya.",
                    "Saya informasikan dulu ya: pinjaman Bapak/Ibu {name} sebesar Rp {amount} sudah jatuh tempo sejak {days} hari yang lalu. Kami ingin membantu mencari solusi terbaik untuk Anda. Apakah Bapak/Ibu bersedia bicara sebentar?",
                    "Untuk informasi, tagihan Anda saat ini Rp {amount} dengan keterlambatan {days} hari. Kami memahami mungkin ada kendala, dan kami siap membantu. Bisa lanjutkan?"
                ]
            },
            "silence_level_3": {
                "*": [
                    "Saya kasih tiga pilihan ya {name}, tidak perlu jawab panjang: (1) bayar lunas, (2) bayar sebagian, atau (3) perpanjang waktu. Bapak/Ibu pilih yang nomor berapa?",
                    "Kami punya solusi mudah: (1) lunas hari ini, (2) cicil separuh dulu, (3) perpanjang dengan biaya Rp {extension_fee}. Tinggal sebut nomornya saja ya.",
                    "Begini saja {name} — saya sebutkan pilihannya: satu, bayar penuh; dua, bayar setengah; tiga, perpanjangan. Pilih yang mana? Cukup sebut angkanya."
                ]
            },
            "silence_level_4": {
                "*": [
                    "Karena kita belum bisa mencapai kesepakatan, kami informasikan bahwa keterlambatan ini akan tercatat dan denda akan terus bertambah ya. Jika nanti Bapak/Ibu sudah siap, silakan hubungi customer service kami di aplikasi Extra Uang. Terima kasih atas waktunya, selamat siang.",
                    "Baik, karena belum ada konfirmasi, kami sampaikan bahwa tagihan ini akan terus berjalan dengan denda keterlambatan ya. Anda bisa menghubungi kami kembali kapan saja melalui aplikasi jika sudah siap membayar. Terima kasih, selamat sore.",
                    "Terakhir dari saya: pembayaran yang tertunda akan menambah biaya denda dan mempengaruhi catatan kredit Anda. Jika ada pertanyaan nanti, silakan hubungi layanan pelanggan kami. Kami tunggu kabar baik dari Anda ya. Terima kasih."
                ]
            },
            "silence_engage": {
                "*": [
                    "Saya bicara dengan Bapak/Ibu {name} ya? Jawab 'ya' saja.",
                    "{name}, cukup jawab 'ya' kalau Anda dengar saya.",
                    "Sebelum lanjut — Bapak/Ibu {name} masih di telepon? Bilang 'ya' saja.",
                    "Maaf {name}, tolong konfirmasi sebentar — Anda dengar suara saya?",
                    "Satu pertanyaan singkat ya {name}: Anda sudah terima SMS tagihan dari kami?"
                ]
            },
            "push_time_unknown": {
                "*": [
                    "Mohon maaf, bisakah Anda memberitahu jam berapa pasti bisa bayar ya?",
                    "Maaf, saya kurang mengerti. Bisa kasih tahu jam berapa Anda bisa transfer ya?",
                    "Mohon maaf, bisa jelaskan lebih jelas ya? Kira-kira jam berapa Anda bisa membayar tagihan ini?"
                ]
            },
            "handle_no_money_repeat": {
                "*": [
                    "Saya sangat mengerti kondisinya Pak/Bu. Apakah ada yang bisa kami bantu untuk meringankan beban Anda?",
                    "Saya paham Anda sedang kesulitan. Apakah Anda ingin opsi perpanjangan waktu atau pembayaran sebagian?",
                    "Mengerti kesulitan Anda. Kami bisa bicarakan solusi yang sesuai untuk situasi Anda, apakah Anda mau?"
                ]
            },
            "close_agree_pay": {
                "*": [
                    "Baik, kami tunggu pembayaran Anda ya. Terima kasih.",
                    "Oke, kami akan menunggu dana masuk ya. Terima kasih atas kerjasamanya.",
                    "Siap, kami tunggu transfer Anda. Terima kasih banyak."
                ]
            },
            "close_general": {
                "*": [
                    "Terima kasih atas waktunya Pak/Bu. Jika ada pertanyaan lain silakan hubungi customer service kami ya.",
                    "Terima kasih untuk waktunya. Jika butuh bantuan lain, silakan hubungi tim layanan pelanggan kami ya.",
                    "Saya mengerti, terima kasih atas waktu Anda. Kalau ada pertanyaan lain bisa hubungi customer service kami kapan saja."
                ]
            },
            "close_firm": {
                "*": [
                    "Baik, karena belum ada kesepakatan, tagihan akan terus berjalan dengan denda harian. Silakan hubungi CS kami melalui aplikasi jika sudah siap membayar. Terima kasih.",
                    "Kami catat bahwa pembayaran belum bisa dilakukan hari ini. Perlu diingat denda akan terus bertambah. Silakan hubungi kami jika sudah siap. Terima kasih.",
                    "Karena tidak ada kesepakatan, kami akan melanjutkan proses sesuai prosedur. Anda bisa menghubungi kami kapan saja jika sudah siap menyelesaikan tagihan. Terima kasih."
                ]
            },
            "consequence_warning": {
                "*": [
                    "Perlu diingat, jika tagihan tidak segera diselesaikan, denda harian akan terus bertambah dan dapat mempengaruhi skor kredit Anda di OJK.",
                    "Keterlambatan pembayaran akan dikenakan denda setiap hari dan bisa berdampak pada riwayat kredit Anda. Lebih baik segera diselesaikan ya.",
                    "Tagihan yang tidak dibayar akan terus bertambah dendanya dan bisa mempengaruhi kemampuan Anda mendapatkan pinjaman di masa depan."
                ]
            },
            "confirm_extension_repeat": {
                "*": [
                    "Apakah Anda masih setuju dengan opsi perpanjangan dengan biaya administrasi Rp {extension_fee} ya?",
                    "Jadi Anda masih setuju untuk mengambil opsi perpanjangan dengan biaya Rp {extension_fee} ya?",
                    "Kembali ke opsi perpanjangan ya, apakah Anda setuju dengan biaya administrasi Rp {extension_fee} ini?"
                ]
            },
            "partial_payment_repeat": {
                "*": [
                    "Berapa jumlah yang ingin Anda bayar sekarang, dan kapan waktunya ya?",
                    "Bisa kasih tahu berapa yang ingin Anda bayar sekarang dan kapan waktunya ya?",
                    "Anda ingin bayar berapa sekarang, dan kapan bisa transfer ya?"
                ]
            },
            "unknown_too_many": {
                "*": [
                    "Mohon maaf, saya tidak bisa memahami maksud Anda. Kami akan hubungi kembali nanti ya. Terima kasih.",
                    "Maaf, saya kurang mengerti apa yang Anda maksud. Kami akan telepon kembali nanti ya. Terima kasih.",
                    "Mohon maaf, saya tidak menangkap maksud Anda. Kami akan hubungi lagi besok ya. Terima kasih."
                ]
            },
            "handle_identity_verification_request": {
                "*": [
                    "Saya adalah petugas penagihan resmi dari Extra Uang ya. Jika Anda memerlukan verifikasi, Anda bisa menghubungi customer service kami melalui aplikasi ya.",
                    "Kami adalah tim penagihan resmi dari PT Extra Uang Indonesia. Anda bisa memverifikasi melalui call center resmi kami ya.",
                    "Perkenalkan saya dari tim penagihan Extra Uang. Jika Anda ragu, silakan hubungi layanan pelanggan kami di aplikasi untuk konfirmasi ya."
                ]
            },
            "handle_interest_reduction_request": {
                "*": [
                    "Mohon maaf ya, biaya keterlambatan sudah sesuai dengan perjanjian pinjaman yang Anda setujui. Jika Anda kesulitan, kami bisa menawarkan opsi perpanjangan ya.",
                    "Saya mengerti keluhan Anda, namun denda keterlambatan tidak bisa dikurangi karena sudah sesuai dengan ketentuan perjanjian. Apakah Anda ingin opsi perpanjangan?",
                    "Maaf ya, biaya administrasi dan denda sudah sesuai dengan kesepakatan awal. Kami hanya bisa menawarkan opsi perpanjangan untuk meringankan beban Anda ya."
                ]
            },
            "handle_short_extension_request": {
                "*": [
                    "Untuk keterlambatan beberapa hari, kami masih bisa menunggu ya. Tapi tolong pastikan Anda membayar maksimal {max_days} hari lagi ya.",
                    "Kami bisa memberikan toleransi keterlambatan sampai {max_days} hari ya. Tapi tolong pastikan Anda membayar sebelum batas waktu tersebut ya.",
                    "Untuk saat ini kami masih bisa menunggu sampai {max_days} hari lagi. Silakan bayar sebelum batas waktu untuk menghindari denda tambahan ya."
                ]
            },
            "handle_high_interest_complaint": {
                "*": [
                    "Mohon maaf jika Anda merasa biaya tinggi ya. Semua biaya sudah dijelaskan saat proses pengajuan pinjaman dan sesuai dengan perjanjian yang Anda tandatangani.",
                    "Saya paham keluhan Anda. Namun ketentuan biaya sudah diinformasikan secara jelas sebelum Anda menyetujui pinjaman ya.",
                    "Maaf atas ketidaknyamanannya. Semua biaya pinjaman sudah dijelaskan pada saat pengajuan dan sesuai dengan peraturan yang berlaku ya."
                ]
            },
            "handle_app_uninstalled_problem": {
                "*": [
                    "Jika Anda sudah menghapus aplikasi, Anda masih bisa membayar melalui transfer rekening resmi kami ya: BCA 1234567890 a.n. PT Extra Uang Indonesia.",
                    "Tidak masalah jika aplikasi sudah dihapus. Anda bisa langsung transfer ke rekening resmi kami ya. Setelah transfer, silakan simpan bukti transfer ya.",
                    "Anda tidak perlu menginstal ulang aplikasi untuk membayar. Cukup transfer ke rekening resmi kami dan konfirmasi jika sudah bayar ya."
                ]
            },
            "handle_payment_reminder_request": {
                "*": [
                    "Baik, saya akan catat untuk mengirimkan pengingat pembayaran melalui WhatsApp ke nomor Anda ya.",
                    "Oke, kami akan kirimkan pesan pengingat beserta detail tagihan ke nomor ponsel Anda ya.",
                    "Baik, nanti tim kami akan mengirimkan pengingat pembayaran melalui SMS ke nomor Anda ya."
                ]
            },
            "handle_settlement_proof_request": {
                "*": [
                    "Setelah pembayaran Anda terverifikasi, Anda bisa meminta surat keterangan lunas melalui customer service di aplikasi ya.",
                    "Bukti pembayaran lunas akan tersedia di aplikasi setelah pembayaran Anda terverifikasi oleh sistem ya.",
                    "Setelah pembayaran Anda kami terima, Anda bisa mengajukan surat keterangan lunas melalui layanan pelanggan kami ya."
                ]
            },
            "handle_consequence_inquiry": {
                "*": [
                    "Jika tagihan tidak dibayar tepat waktu, akan ada denda keterlambatan sesuai perjanjian dan bisa mempengaruhi skor kredit Anda di OJK ya.",
                    "Keterlambatan pembayaran akan menambah biaya denda dan bisa berdampak negatif pada riwayat kredit Anda ya.",
                    "Jika tidak dibayar, biaya keterlambatan akan terus bertambah dan nama Anda bisa terdaftar di daftar hitam lembaga keuangan ya."
                ]
            },
            "handle_borrowing_money_response": {
                "*": [
                    "Baik, saya tunggu proses pembayaran Anda ya. Mohon konfirmasi jika sudah selesai transfer ya.",
                    "Oke, semoga proses peminjaman uang Anda lancar. Jangan lupa untuk segera melakukan pembayaran ya.",
                    "Baik, saya mengerti. Silakan hubungi kami jika sudah selesai melakukan pembayaran ya."
                ]
            },
            "handle_transfer_in_process_response": {
                "*": [
                    "Baik, saya tunggu konfirmasi pembayaran Anda ya. Terima kasih atas kerjasamanya.",
                    "Oke, jika sudah selesai transfer, mohon simpan bukti transfer ya. Terima kasih.",
                    "Baik, kami akan menunggu pembayaran masuk ke rekening kami ya. Terima kasih."
                ]
            }
        }

    def _get_closing(self) -> str:
        """获取闭幕语，关系型客户用温暖版本"""
        if self.strategy.relationship_emphasis:
            warm = self.script_lib.get("closing_warm", {}).get("*", [])
            if warm:
                script = random.choice(warm)
                self.last_responses.append(script)
                if len(self.last_responses) > 2:
                    self.last_responses.pop(0)
                return self.var_replacer.replace(script, name=self.customer_name)
        return self._get_script("closing")

    def _get_script(self, category: str, **kwargs) -> str:
        """获取话术并替换变量，避免连续重复回复"""
        # 先尝试获取对应催收阶段的话术，如果没有则用通配符"*"的话术
        category_scripts = self.script_lib.get(category, {})
        scripts = category_scripts.get(self.chat_group, [])
        if not scripts:
            scripts = category_scripts.get("*", [])

        # 如果只有1条话术，直接返回，无法避免重复
        if len(scripts) == 1:
            script = scripts[0]
        else:
            # 随机取，避免和最近2次重复
            max_tries = 10
            for _ in range(max_tries):
                script = random.choice(scripts)
                if script not in self.last_responses:
                    break
            # 如果试了10次都重复，就随便返回一条

        # 更新最近回复列表，最多保留2条
        self.last_responses.append(script)
        if len(self.last_responses) > 2:
            self.last_responses.pop(0)

        # 合并变量，包括用户信息、欠款信息等
        vars = {
            "name": self.customer_name,
            "amount": f"{self.overdue_amount:,}",
            "days": str(self.overdue_days),
            "extension_fee": f"{self.extension_fee:,}"
        }
        vars.update(kwargs)

        return self.var_replacer.replace(script, **vars)

    async def process(
        self,
        customer_input: Optional[str] = None,
        use_tts: bool = False
    ) -> Tuple[str, Optional[str]]:
        """
        处理用户输入，返回回复
        返回: (文本回复, 音频文件路径)
        """
        start_time = datetime.now()

        # 初始状态，机器人先说话
        if self.state == ChatState.INIT:
            self.state = ChatState.IDENTITY_VERIFY
            identity_verify = self._get_script("identity_verify")
            # P15-D01: 回头客追加关系维护前缀
            if self._is_returning_customer and self.user_memory:
                if self.user_memory.previously_promised_but_failed:
                    identity_verify = (
                        f"Halo {self.customer_name}, kita pernah bicara sebelumnya. "
                        f"Terakhir kali Bapak/Ibu bilang akan bayar tapi sampai sekarang belum ya. "
                        + identity_verify
                    )
                else:
                    identity_verify = (
                        f"Halo {self.customer_name}, terima kasih sudah bicara lagi dengan kami. "
                        + identity_verify
                    )
            # 教育型策略：身份核实后追加简短合同说明
            if self.strategy.education_emphasis:
                identity_verify += " " + self._get_script("educate_intro")
            self.conversation.append(ChatTurn(agent=identity_verify, state=self.state))
            audio_file = await self._tts_speak(identity_verify, use_tts)
            return identity_verify, audio_file

        # 记录用户输入
        corrected_input = ""
        # 判断是否为有效输入：排除空字符串、纯空格、省略号等纯标点输入
        is_silent_input = (
            not customer_input or
            not customer_input.strip() or
            customer_input.strip().replace(".", "").replace(",", "").replace("?", "").replace("!", "").strip() == ""
        )
        if not is_silent_input and customer_input.strip():
            # 先纠正ASR错误
            corrected_input = self.asr_corrector.correct(customer_input)
            self.conversation[-1].customer = corrected_input
            # 识别用户意图，传入上一条机器人消息作为上下文
            self.user_intent = self.intent_detector.detect(corrected_input, context=self.conversation[-1].agent)
            # 更新对话记忆
            self.user_history_intents.append(self.user_intent)
            # 有效的用户输入 → 重置沉默计数
            self.silence_count = 0
            if self.user_intent == "ask_amount":
                self.user_asked_amount = True
            elif self.user_intent == "ask_fee":
                self.user_asked_fee = True
            elif self.user_intent == "ask_payment_method":
                self.user_asked_payment_method = True
            elif self.user_intent == "no_money":
                self.user_mentioned_no_money = True
            elif self.user_intent == "busy_later":
                self.user_mentioned_busy = True
            elif self.user_intent == "partial_payment":
                self.partial_payment_discussed = True
            elif self.user_intent == "ask_extension" or self.user_intent == "request_short_extension":
                self.extension_discussed = True
            # P15-A01: 用户展现合作意图时，重置异议递进计数器
            if self.user_intent in ("agree_to_pay", "confirm_time", "greeting",
                                     "ask_extension", "request_short_extension",
                                     "partial_payment", "confirm_identity",
                                     "respond_to_greeting", "propose_repayment_time"):
                self.no_money_count = 0
                self.busy_count = 0
                self.dont_know_count = 0
        else:
            # 空输入、纯空格、省略号 → 视为沉默
            self.conversation[-1].customer = customer_input or ""
            self.user_intent = "silence"
            self.user_history_intents.append("silence")

        response = ""
        next_state = self.state

        # 检测用户提到的时间（提前检测，方便公共意图处理）
        detected_time = self.time_detector.detect(corrected_input or "")

        # ========== LLM Fallback 检查 ==========
        self.llm_used_this_turn = False

        if self.in_llm_fallback:
            # 已在 LLM 兜底状态，继续用 LLM 处理
            response = await self._process_llm_fallback(corrected_input, detected_time)
            if response:
                self.conversation.append(ChatTurn(agent=response, state=self.state))
            audio_file = await self._tts_speak(response, use_tts)
            return response, audio_file

        if self.llm_enabled and not self.in_llm_fallback:
            triggered, trigger = self._check_llm_fallback()
            if triggered:
                print(f"[LLM] 触发 LLM 兜底: {trigger.description}")
                self.in_llm_fallback = True
                self.llm_turn_count = 0
                self.state = ChatState.LLM_FALLBACK
                response = await self._process_llm_fallback(corrected_input, detected_time)
                if response:
                    self.conversation.append(ChatTurn(agent=response, state=self.state))
                audio_file = await self._tts_speak(response, use_tts)
                return response, audio_file

        # ========== 沉默处理（优先于状态机） ==========
        if self.user_intent == "silence":
            response, silence_next_state = self._handle_silence()
            if silence_next_state is not None:
                self.state = silence_next_state
            self.conversation.append(ChatTurn(agent=response, state=self.state))
            audio_file = await self._tts_speak(response, use_tts)
            return response, audio_file

        # 状态机逻辑
        if self.state == ChatState.IDENTITY_VERIFY:
            # 先检查用户是否已经确认身份（除非明确否认，否则默认确认）
            identity_confirmed = True
            if self.user_intent == "deny_identity" or self.user_intent == "third_party":
                identity_confirmed = False

            # 先处理公共意图
            common_response, common_next_state = self._handle_common_intents(detected_time)
            if common_response is not None:
                response = common_response
                if common_next_state is not None:
                    next_state = common_next_state
                    # 如果处理完异议且身份已确认，继续询问还款时间
                    if identity_confirmed and next_state == ChatState.HANDLE_OBJECTION:
                        response += " " + self._get_script("ask_time")
                        next_state = ChatState.ASK_TIME
                else:
                    # 公共意图没有指定下一个状态，保持当前状态或根据身份确认情况处理
                    if identity_confirmed:
                        # 用户已经确认身份，回答后进入来意说明并询问还款时间
                        response += " " + self._get_script("purpose") + " " + self._get_script("ask_time")
                        next_state = ChatState.ASK_TIME
                    else:
                        # 还没确认身份，回答后继续确认身份
                        next_state = ChatState.IDENTITY_VERIFY
            elif self.user_intent == "dont_know":
                # 用户说不知道，重复说明来意
                response = self._get_script("handle_dont_know")
                next_state = ChatState.PURPOSE
            elif identity_confirmed:
                # 用户确认了身份，处理剩余的特殊意图
                if self.user_intent == "confirm_time":
                    # 用户直接给出了还款时间
                    if detected_time:
                        self.commit_time = detected_time
                        # 直接生成确认和结束语
                        commit_resp = self._get_script("commit_time", time=detected_time)
                        wait_script = self._get_script("wait", time=detected_time)
                        closing = self._get_closing()
                        response = f"{commit_resp} {wait_script} {closing}"
                        next_state = ChatState.CLOSE
                    else:
                        # 进入询问时间阶段
                        next_state = ChatState.ASK_TIME
                        response = self._get_script("ask_time")
                elif self.user_intent == "borrowing_money":
                    # 用户说正在借钱
                    response = self._get_script("handle_borrowing_money_response")
                    next_state = ChatState.PUSH_FOR_TIME
                elif self.user_intent == "confirm_identity" or self.user_intent == "agree_to_pay" or self.user_intent == "greeting":
                    # 用户只是确认身份或问候，进入来意说明并询问还款时间
                    next_state = ChatState.ASK_TIME
                    response = self._get_script("purpose") + " " + self._get_script("ask_time")
                else:
                    # 其他意图，先进入来意说明并询问还款时间
                    next_state = ChatState.ASK_TIME
                    response = self._get_script("purpose") + " " + self._get_script("ask_time")
            else:
                # 没有确认身份，再次确认身份
                response = self._get_script("identity_verify")
                next_state = ChatState.IDENTITY_VERIFY


        elif self.state == ChatState.PURPOSE:
            # 说明来意后，处理用户的回复
            # 先处理公共意图
            common_response, common_next_state = self._handle_common_intents(detected_time)
            if common_response is not None:
                response = common_response
                if common_next_state is not None:
                    next_state = common_next_state
                else:
                    # 公共意图没有指定下一个状态，回答后进入询问时间阶段
                    response += " " + self._get_script("ask_time")
                    next_state = ChatState.ASK_TIME
            elif self.user_intent == "dont_know":
                # 用户说不知道，再次说明来意
                response = self._get_script("handle_dont_know")
                next_state = ChatState.PURPOSE
            elif self.user_intent == "confirm_time":
                # 用户直接给出了还款时间
                detected_time = self.time_detector.detect(corrected_input or "")
                if detected_time:
                    self.commit_time = detected_time
                    # 直接生成确认和结束语
                    commit_resp = self._get_script("commit_time", time=detected_time)
                    wait_script = self._get_script("wait", time=detected_time)
                    closing = self._get_closing()
                    response = f"{commit_resp} {wait_script} {closing}"
                    next_state = ChatState.CLOSE
                else:
                    next_state = ChatState.ASK_TIME
                    response = self._get_script("ask_time")
            elif self.user_intent == "borrowing_money":
                # 用户说正在借钱
                response = self._get_script("handle_borrowing_money_response")
                next_state = ChatState.PUSH_FOR_TIME
            else:
                # 其他情况，进入询问还款时间环节
                next_state = ChatState.ASK_TIME
                response = self._get_script("ask_time")

        elif self.state == ChatState.CONFIRM_EXTENSION:
            # 确认用户是否同意展期
            if self.user_intent == "agree_to_pay" or "ya" in (corrected_input or "").lower():
                # 用户同意展期
                self.extension_agreed = True
                response = self._get_script("ask_time")
                next_state = ChatState.ASK_TIME
            else:
                # 先处理公共意图
                common_response, common_next_state = self._handle_common_intents()
                if common_response is not None:
                    response = common_response
                    if common_next_state is not None:
                        next_state = common_next_state
                    else:
                        # 公共意图没有指定下一个状态，保持当前状态继续确认展期
                        next_state = ChatState.CONFIRM_EXTENSION
                elif self.user_intent == "dont_know":
                    # 用户说不知道，重复展期说明
                    response = self._get_script("handle_dont_know") + " " + self._get_script("explain_extension")
                    next_state = ChatState.CONFIRM_EXTENSION
                elif self.user_intent == "partial_payment":
                    # 用户询问部分还款，引导给出金额和时间
                    response = self._get_script("handle_partial_payment")
                    next_state = ChatState.PUSH_FOR_TIME
                else:
                    # 用户不同意，继续询问还款时间
                    response = self._get_script("ask_time")
                    next_state = ChatState.ASK_TIME

        elif self.state == ChatState.ASK_TIME:
            # 询问还款时间后，处理用户回复
            detected_time = self.time_detector.detect(corrected_input or "")
            # 排除模糊时间，只有具体时间才接受
            fuzzy_times = ["nanti", "sekarang", "nanti aja"]
            if detected_time and detected_time not in fuzzy_times:
                # 用户给出了具体时间，直接生成确认和结束语，结束对话
                self.commit_time = detected_time
                commit_resp = self._get_script("commit_time", time=detected_time)
                wait_script = self._get_script("wait", time=detected_time)
                closing = self._get_closing()
                response = f"{commit_resp} {wait_script} {closing}"
                next_state = ChatState.CLOSE
            else:
                # 先处理公共意图
                common_response, common_next_state = self._handle_common_intents()
                if common_response is not None:
                    response = common_response
                    if common_next_state is not None:
                        next_state = common_next_state
                    else:
                        # 公共意图没有指定下一个状态，保持当前状态继续询问时间
                        next_state = ChatState.ASK_TIME
                elif self.user_intent == "dont_know":
                    # 用户说不知道，再次询问时间
                    response = self._get_script("handle_dont_know") + " " + self._get_script("ask_time")
                    next_state = ChatState.ASK_TIME
                elif self.user_intent == "partial_payment":
                    # 用户询问部分还款，引导给出金额和时间
                    response = self._get_script("handle_partial_payment")
                    next_state = ChatState.PUSH_FOR_TIME
                else:
                    # 没有检测到时间，催促用户
                    # P15-H04: T2 策略再评估（即将放弃前）
                    if self.objection_count >= self.max_objections - 1:
                        self._re_evaluate_strategy("ask_time")
                    if self.objection_count < self.max_objections:
                        self.objection_count += 1
                        # P15-B01: 展期优先策略 — 首次 push 先推展期而非全额还款
                        if self.strategy.extension_priority and self.objection_count == 1:
                            response = self._get_script("explain_extension")
                            next_state = ChatState.CONFIRM_EXTENSION
                        else:
                            next_state = ChatState.PUSH_FOR_TIME
                            response = self._get_script("push")
                    else:
                        next_state = ChatState.FAILED
                        response = ""

        elif self.state == ChatState.PUSH_FOR_TIME:
            # 催促后，处理用户回复
            detected_time = self.time_detector.detect(corrected_input or "")
            if detected_time:
                # 用户给出了时间，直接生成确认和结束语，结束对话
                self.commit_time = detected_time
                commit_resp = self._get_script("commit_time", time=detected_time)
                wait_script = self._get_script("wait", time=detected_time)
                closing = self._get_closing()
                response = f"{commit_resp} {wait_script} {closing}"
                next_state = ChatState.CLOSE
            elif self.user_intent == "ask_extension":
                # 用户询问展期
                response = self._get_script("explain_extension")
                next_state = ChatState.CONFIRM_EXTENSION
            elif self.user_intent == "ask_amount":
                # 用户询问金额
                response = self._get_script("answer_amount")
                next_state = ChatState.PUSH_FOR_TIME
            elif self.user_intent == "ask_fee":
                # 用户询问费用
                response = self._get_script("answer_fee")
                next_state = ChatState.PUSH_FOR_TIME
            elif self.user_intent == "ask_payment_method":
                # 用户询问支付方式
                response = self._get_script("answer_payment_method")
                next_state = ChatState.PUSH_FOR_TIME
            elif self.user_intent == "already_paid":
                # 用户说已经付款
                response = self._get_script("handle_already_paid")
                next_state = ChatState.CLOSE
            elif self.user_intent == "partial_payment":
                # 用户询问部分还款，引导给出金额和时间
                response = self._get_script("handle_partial_payment")
                next_state = ChatState.PUSH_FOR_TIME
            elif self.user_intent == "third_party":
                # 第三方接听
                response = self._get_script("handle_third_party")
                next_state = ChatState.CLOSE
            elif self.user_intent == "dont_know":
                # 用户说不知道，再次催促
                response = self._get_script("handle_dont_know") + " " + self._get_script("push")
                next_state = ChatState.PUSH_FOR_TIME
            elif self.user_intent == "busy_later":
                # 用户现在忙 —— 递进式反驳 P15-A01
                self.busy_count += 1
                if self.busy_count == 1:
                    response = self._get_script("handle_busy_level2", time_suggestion="jam 3 sore")
                    next_state = ChatState.PUSH_FOR_TIME
                elif self.busy_count == 2:
                    response = self._get_script("closing_busy")
                    next_state = ChatState.CLOSE
                else:
                    response = self._get_script("push")
                    next_state = ChatState.PUSH_FOR_TIME
            elif self.user_intent == "question_identity":
                # 用户质疑身份
                response = self._get_script("answer_identity")
                next_state = ChatState.PUSH_FOR_TIME
            elif self.user_intent == "request_identity_verification":
                # 用户要求验证身份合法性
                response = self._get_script("handle_identity_verification_request")
                next_state = ChatState.PUSH_FOR_TIME
            elif self.user_intent == "request_interest_reduction":
                # 用户要求减免利息
                response = self._get_script("handle_interest_reduction_request")
                next_state = ChatState.HANDLE_OBJECTION
            elif self.user_intent == "request_short_extension":
                # 用户要求短期延期
                response = self._get_script("handle_short_extension_request", max_days="3")
                next_state = ChatState.CLOSE
            elif self.user_intent == "complain_high_interest":
                # 用户抱怨利率太高
                response = self._get_script("handle_high_interest_complaint")
                next_state = ChatState.HANDLE_OBJECTION
            elif self.user_intent == "app_uninstalled":
                # 用户说已经卸载了APP
                response = self._get_script("handle_app_uninstalled_problem")
                next_state = ChatState.PUSH_FOR_TIME
            elif self.user_intent == "request_payment_reminder":
                # 用户要求发送还款提醒
                response = self._get_script("handle_payment_reminder_request")
                next_state = ChatState.PUSH_FOR_TIME
            elif self.user_intent == "request_settlement_proof":
                # 用户要求开具结清证明
                response = self._get_script("handle_settlement_proof_request")
                next_state = ChatState.CLOSE
            elif self.user_intent == "inquire_consequences":
                # 用户询问逾期后果
                response = self._get_script("handle_consequence_inquiry")
                next_state = ChatState.PUSH_FOR_TIME
            elif self.user_intent == "borrowing_money":
                # 用户说正在借钱
                response = self._get_script("handle_borrowing_money_response")
                next_state = ChatState.PUSH_FOR_TIME
            elif self.user_intent == "transfer_in_process":
                # 用户说正在转账
                response = self._get_script("handle_transfer_in_process_response")
                next_state = ChatState.CLOSE
            elif self.user_intent == "no_money":
                # 用户说没钱 —— 递进式反驳链 P15-A01
                self.no_money_count += 1
                if self.no_money_count == 1:
                    response = self._get_script("handle_no_money")
                elif self.no_money_count == 2:
                    response = self._get_script("handle_no_money_level2")
                else:
                    response = self._get_script("handle_no_money_level3")
                next_state = ChatState.HANDLE_OBJECTION
            elif self.user_intent == "threaten":
                # 用户威胁
                response = self._get_script("handle_threat")
                next_state = ChatState.HANDLE_OBJECTION
            elif self.user_intent == "refuse_to_pay":
                # 用户拒绝还款
                response = self._get_script("objection_general")
                next_state = ChatState.HANDLE_OBJECTION
            elif self.user_intent == "unknown":
                # 无法识别用户意图，请求重复
                response = self._get_script("handle_unknown")
                next_state = ChatState.PUSH_FOR_TIME
            else:
                # P15-B02: 递进式升级链 — objection_count 为最终安全阀
                if self.objection_count >= self.max_objections:
                    next_state = ChatState.FAILED
                    response = ""
                elif self.push_round < self.strategy.max_push_rounds:
                    # P15-H04: T2 策略再评估（即将触发 fallback 前, 至少一次 push 后）
                    if self.push_round > 0 and self.push_round >= self.strategy.max_push_rounds - 1:
                        self._re_evaluate_strategy("push_for_time")
                    if self.push_round < self.strategy.max_push_rounds:
                        self.objection_count += 1
                        self.push_round += 1
                        if self.push_round == 1:
                            response = self._get_script("push")
                        elif self.push_round == 2:
                            response = self._get_script("push_hard")
                        else:
                            response = self._get_script("push_final")
                        if self.strategy.consequence_emphasis >= 3:
                            response += " " + self._get_script("consequence_warning")
                        next_state = ChatState.PUSH_FOR_TIME
                else:
                    # 主策略耗尽，激活 fallback_approach
                    fallback = self.strategy.fallback_approach
                    if fallback == "offer_extension" and not self.extension_discussed:
                        self.extension_discussed = True
                        response = self._get_script("explain_extension")
                        next_state = ChatState.CONFIRM_EXTENSION
                    elif fallback == "partial_payment" and not self.partial_payment_discussed:
                        self.partial_payment_discussed = True
                        response = self._get_script("handle_partial_payment")
                        next_state = ChatState.PUSH_FOR_TIME
                    elif fallback == "callback_later":
                        response = self._get_script("closing_busy")
                        next_state = ChatState.CLOSE
                    elif fallback == "accept_promise":
                        response = self._get_script("closing")
                        next_state = ChatState.CLOSE
                    elif fallback == "escalate":
                        response = self._get_script("escalate_to_supervisor")
                        next_state = ChatState.FAILED
                    else:
                        next_state = ChatState.FAILED
                        response = ""

        elif self.state == ChatState.HANDLE_OBJECTION:
            # 处理一般异议
            if self.user_intent == "ask_extension":
                response = self._get_script("explain_extension")
                next_state = ChatState.CONFIRM_EXTENSION
            elif self.user_intent == "partial_payment":
                # 用户询问部分还款，引导给出金额和时间
                response = self._get_script("handle_partial_payment")
                next_state = ChatState.PUSH_FOR_TIME
            elif self.user_intent == "ask_amount":
                # 用户询问金额
                response = self._get_script("answer_amount")
                next_state = ChatState.ASK_TIME
            elif self.user_intent == "ask_fee":
                # 用户询问费用
                response = self._get_script("answer_fee")
                next_state = ChatState.ASK_TIME
            elif self.user_intent == "ask_payment_method":
                # 用户询问支付方式
                response = self._get_script("answer_payment_method")
                next_state = ChatState.ASK_TIME
            elif self.user_intent == "already_paid":
                # 用户说已经付款
                response = self._get_script("handle_already_paid")
                next_state = ChatState.CLOSE
            elif self.user_intent == "third_party":
                # 第三方接听
                response = self._get_script("handle_third_party")
                next_state = ChatState.CLOSE
            elif self.user_intent == "dont_know":
                # 用户说不知道 —— 递进式反驳
                self.dont_know_count += 1
                if self.dont_know_count == 1:
                    response = self._get_script("handle_dont_know")
                elif self.dont_know_count == 2:
                    response = self._get_script("handle_dont_know_level2")
                else:
                    response = self._get_script("handle_dont_know") + " " + self._get_script("objection_general")
                next_state = ChatState.ASK_TIME
            elif self.user_intent == "busy_later":
                # 用户现在忙 —— 递进式反驳，不直接挂断
                self.busy_count += 1
                if self.busy_count == 1:
                    response = self._get_script("closing_busy")
                elif self.busy_count == 2:
                    response = self._get_script("handle_busy_level2", time_suggestion="jam 3 sore")
                else:
                    response = self._get_script("push")
                next_state = ChatState.PUSH_FOR_TIME if self.busy_count >= 2 else ChatState.CLOSE
            elif self.user_intent == "no_money":
                # 用户再次说没钱 —— 递进式反驳链
                self.no_money_count += 1
                if self.no_money_count == 1:
                    response = self._get_script("handle_no_money")
                elif self.no_money_count == 2:
                    response = self._get_script("handle_no_money_level2")
                else:
                    response = self._get_script("handle_no_money_level3")
                next_state = ChatState.HANDLE_OBJECTION
            elif self.user_intent == "question_identity":
                # 用户质疑身份
                response = self._get_script("answer_identity")
                next_state = ChatState.HANDLE_OBJECTION
            elif self.user_intent == "request_identity_verification":
                # 用户要求验证身份合法性
                response = self._get_script("handle_identity_verification_request")
                next_state = ChatState.HANDLE_OBJECTION
            elif self.user_intent == "request_interest_reduction":
                # 用户要求减免利息
                response = self._get_script("handle_interest_reduction_request")
                next_state = ChatState.HANDLE_OBJECTION
            elif self.user_intent == "request_short_extension":
                # 用户要求短期延期
                response = self._get_script("handle_short_extension_request", max_days="3")
                next_state = ChatState.CLOSE
            elif self.user_intent == "complain_high_interest":
                # 用户抱怨利率太高
                response = self._get_script("handle_high_interest_complaint")
                next_state = ChatState.HANDLE_OBJECTION
            elif self.user_intent == "app_uninstalled":
                # 用户说已经卸载了APP
                response = self._get_script("handle_app_uninstalled_problem")
                next_state = ChatState.ASK_TIME
            elif self.user_intent == "request_payment_reminder":
                # 用户要求发送还款提醒
                response = self._get_script("handle_payment_reminder_request")
                next_state = ChatState.ASK_TIME
            elif self.user_intent == "request_settlement_proof":
                # 用户要求开具结清证明
                response = self._get_script("handle_settlement_proof_request")
                next_state = ChatState.CLOSE
            elif self.user_intent == "inquire_consequences":
                # 用户询问逾期后果
                response = self._get_script("handle_consequence_inquiry")
                next_state = ChatState.ASK_TIME
            elif self.user_intent == "borrowing_money":
                # 用户说正在借钱
                response = self._get_script("handle_borrowing_money_response")
                next_state = ChatState.PUSH_FOR_TIME
            elif self.user_intent == "transfer_in_process":
                # 用户说正在转账
                response = self._get_script("handle_transfer_in_process_response")
                next_state = ChatState.CLOSE
            elif self.user_intent == "threaten":
                # 用户再次威胁
                response = self._get_script("handle_threat")
                next_state = ChatState.HANDLE_OBJECTION
            elif self.user_intent == "refuse_to_pay":
                # 用户拒绝还款，继续处理异议
                response = self._get_script("objection_general")
                next_state = ChatState.HANDLE_OBJECTION
            elif self.user_intent == "unknown":
                # 无法识别用户意图，请求重复
                response = self._get_script("handle_unknown")
                next_state = ChatState.HANDLE_OBJECTION
            else:
                # 其他异议，返回通用回复，继续询问还款时间
                response = self._get_script("objection_general")
                next_state = ChatState.ASK_TIME

        elif self.state == ChatState.CLOSE or self.state == ChatState.FAILED:
            # 对话结束后仍然处理用户的后续输入，回答常见问题
            common_response, common_next_state = self._handle_common_intents(detected_time)
            if common_response is not None:
                response = common_response
                # 如果公共意图返回了非CLOSE状态，还是保持CLOSE状态，避免重新进入流程
                next_state = ChatState.CLOSE
            elif self.user_intent == "agree_to_pay" or self.user_intent == "confirm_time":
                # 用户后续又确认还款，回复确认
                response = self._get_script("close_agree_pay")
                next_state = ChatState.CLOSE
            else:
                # P15-B01: 强硬/紧急语气用 firm closing
                if self.strategy.tone in ("firm", "urgent"):
                    response = self._get_script("close_firm")
                else:
                    response = self._get_script("close_general")
                next_state = ChatState.CLOSE

        # 记录机器人回复
        if response:
            self.conversation.append(ChatTurn(agent=response, state=self.state))

        self.state = next_state
        audio_file = await self._tts_speak(response, use_tts)
        return response, audio_file

    def _re_evaluate_strategy(self, at_boundary: str = "ask_time"):
        """P15-H04: T2 会话中策略再评估

        在即将放弃或即将触发fallback时，根据对话信号调整策略参数。
        纯规则驱动，无LLM调用延迟。
        """
        recent_intents = self.user_history_intents[-3:] if self.user_history_intents else []
        cooperative_signals = {"agree_to_pay", "confirm_time", "give_time", "confirm_identity",
                               "propose_repayment_time", "ask_extension"}
        hostile_signals = {"refuse_to_pay", "threaten"}
        hardship_signals = {"no_money", "complain_high_interest", "ask_fee"}

        has_cooperation = any(i in cooperative_signals for i in recent_intents)
        has_hostility = any(i in hostile_signals for i in recent_intents)
        has_hardship = any(i in hardship_signals for i in recent_intents)
        repeated_no_money = self.no_money_count >= 2
        user_is_engaging = len(self.conversation) >= 5 and not has_hostility

        if at_boundary == "ask_time":
            if has_cooperation and not has_hostility:
                if self.strategy.max_objections <= self.objection_count + 1:
                    self.max_objections += 1
                    self.strategy.max_objections += 1
            if repeated_no_money:
                self.strategy.extension_priority = True

        elif at_boundary == "push_for_time":
            if self.partial_payment_discussed or (has_hardship and not has_hostility):
                if self.strategy.fallback_approach in ("", "escalate"):
                    self.strategy.fallback_approach = "partial_payment"
            elif not self.extension_discussed and not has_hostility:
                if self.strategy.fallback_approach in ("", "escalate"):
                    self.strategy.fallback_approach = "offer_extension"
            if user_is_engaging and has_cooperation:
                if self.strategy.max_push_rounds <= self.push_round + 1:
                    self.strategy.max_push_rounds += 1

    def _check_llm_fallback(self) -> tuple:
        """检查是否需要触发 LLM Fallback"""
        if not self.llm_enabled or self.fallback_detector is None:
            return False, None
        return self.fallback_detector.check(self)

    async def _process_llm_fallback(self, customer_input: str, detected_time: Optional[str] = None) -> str:
        """LLM 兜底处理，含四级降级链"""
        from core.compliance_checker import get_compliance_checker

        self.llm_used_this_turn = True

        # 构建对话历史
        history = []
        for turn in self.conversation[-10:]:
            if turn.customer:
                history.append({"role": "user", "content": turn.customer})
            history.append({"role": "agent", "content": turn.agent})
        if customer_input:
            history.append({"role": "user", "content": customer_input})

        # L2: 尝试 LLM
        llm_response = None
        if self.llm_provider is not None:
            try:
                llm_response = await self.llm_provider.generate(
                    conversation_history=history,
                    context={
                        "chat_group": self.chat_group,
                        "customer_name": self.customer_name,
                        "objection_count": self.objection_count,
                    }
                )
            except LLMUnavailableError as e:
                print(f"[LLM] LLM 不可用: {e}，降级到 ML 分类器")

        # 合规后置过滤
        if llm_response:
            compliance_checker = get_compliance_checker()
            filtered, violations = compliance_checker.post_filter(llm_response)
            if filtered is None:
                print(f"[LLM] 合规过滤拦截: {[v['rule_id'] for v in violations if v['severity'] == 'high']}")
                # 重试一次
                try:
                    llm_response = await self.llm_provider.generate(
                        conversation_history=history,
                        context={
                            "chat_group": self.chat_group,
                            "customer_name": self.customer_name,
                            "objection_count": self.objection_count,
                        }
                    )
                    filtered, violations = compliance_checker.post_filter(llm_response)
                    if filtered is None:
                        print("[LLM] 重试后仍不合规，降级到 ML 分类器")
                        llm_response = None
                    else:
                        llm_response = filtered
                except Exception:
                    llm_response = None
            else:
                llm_response = filtered

        # L3: ML 分类器降级
        if llm_response is None:
            ml_response = self._try_ml_fallback(customer_input)
            if ml_response:
                return ml_response

        # L4: 默认话术
        if llm_response is None:
            llm_response = self._get_script("handle_unknown")

        # 检测 LLM 回复中的时间
        if llm_response and not detected_time:
            detected_time = self.time_detector.detect(llm_response)
        if not detected_time and customer_input:
            detected_time = self.time_detector.detect(customer_input)

        if detected_time:
            self.commit_time = detected_time
            self.in_llm_fallback = False
            self.state = ChatState.CLOSE
            commit_resp = self._get_script("commit_time", time=detected_time)
            wait_script = self._get_script("wait", time=detected_time) if self.commit_time else "Saya tunggu ya."
            closing = self._get_closing()
            return f"{commit_resp} {wait_script} {closing}"

        # LLM 轮数限制
        self.llm_turn_count += 1
        max_turns = self.llm_config.max_llm_turns if self.llm_config else 3
        if self.llm_turn_count >= max_turns:
            print(f"[LLM] 达到 LLM 轮数上限 ({max_turns})，切回规则机")
            self.in_llm_fallback = False
            self.state = ChatState.PUSH_FOR_TIME
            return self._get_script("push")

        return llm_response or self._get_script("handle_unknown")

    def _try_ml_fallback(self, customer_input: str) -> Optional[str]:
        """尝试 ML 分类器作为降级方案"""
        if not ML_CLASSIFIER_AVAILABLE:
            return None
        if IntentDetector._ml_classifier is None:
            return None
        try:
            predictions = IntentDetector._ml_classifier.predict(customer_input or "", top_k=1)
            if predictions:
                intent, confidence = predictions[0]
                if confidence >= IntentDetector._ml_threshold:
                    self.user_intent = intent
                    self.user_history_intents.append(intent)
                    # 返回 None 让调用方继续用其他降级方案
        except Exception:
            pass
        return None

    def _handle_silence(self) -> Tuple[str, Optional[ChatState]]:
        """处理用户沉默：5级递进式主动话术（低门槛破冰→主动介绍→给选项→告知后果）"""
        self.silence_count += 1
        level = min(self.silence_count, 5)

        if level == 1:
            # 第1次沉默：超低门槛破冰 → 只需要回答"ya"或确认收到短信
            return self._get_script("silence_engage"), None
        elif level == 2:
            # 第2次沉默：确认通话质量，用是非题降低回应门槛
            return self._get_script("silence_level_1"), None
        elif level == 3:
            # 第3次沉默：主动介绍账单信息，降低信息不对称
            return self._get_script("silence_level_2"), None
        elif level == 4:
            # 第4次沉默：给三选一选项，锚定选择框架
            return self._get_script("silence_level_3"), ChatState.ASK_TIME
        else:  # level >= 5
            # 第5次沉默：告知后果 + 留联系方式 + 礼貌挂断
            return self._get_script("silence_level_4"), ChatState.CLOSE

    async def _tts_speak(self, text: str, use_tts: bool) -> Optional[str]:
        """TTS说话"""
        if not use_tts or not text:
            return None
        return await self.tts.synthesize(text)

    def _handle_common_intents(self, detected_time: Optional[str] = None) -> Tuple[Optional[str], Optional[ChatState]]:
        """
        处理各个状态下都可能出现的公共意图
        返回: (回复内容, 下一个状态) 如果没有匹配到公共意图返回 (None, None)
        """
        # 结束类意图
        if self.user_intent == "user_abuse":
            # 用户辱骂/人身攻击，立即礼貌结束对话
            return self._get_script("handle_user_abuse"), ChatState.CLOSE
        elif self.user_intent == "deny_identity":
            # 用户否认身份/打错电话，直接回复错号结束语，结束对话
            return self._get_script("closing_wrong_number"), ChatState.CLOSE
        elif self.user_intent == "busy_later":
            # 在询问还款时间阶段，用户说"nanti"是指还款时间，不是说现在忙，不结束对话
            if self.state not in [ChatState.ASK_TIME, ChatState.PUSH_FOR_TIME]:
                # 用户现在忙，回复忙的结束语，结束对话
                return self._get_script("closing_busy"), ChatState.CLOSE
            # 在询问时间阶段，按未知意图处理，催促用户给出具体时间
            return None, None
        elif self.user_intent == "already_paid":
            # 用户说已经付款，确认后结束对话
            return self._get_script("handle_already_paid"), ChatState.CLOSE
        elif self.user_intent == "third_party":
            # 第三方接听，结束对话
            return self._get_script("handle_third_party"), ChatState.CLOSE
        elif self.user_intent == "request_settlement_proof":
            # 用户要求开具结清证明
            return self._get_script("handle_settlement_proof_request"), ChatState.CLOSE
        elif self.user_intent == "transfer_in_process":
            # 用户说正在转账
            return self._get_script("handle_transfer_in_process_response"), ChatState.CLOSE
        elif self.user_intent == "request_short_extension":
            # 用户要求短期延期
            return self._get_script("handle_short_extension_request", max_days="3"), ChatState.CLOSE

        # 信息查询类意图
        elif self.user_intent == "ask_amount":
            # 用户询问金额
            response = self._get_script("answer_amount")
            if not self.user_asked_amount:
                # 第一次询问，补充说明
                response += " " + "Jika ada pertanyaan lain silakan bertanya ya."
            return response, None  # 返回None表示保持当前状态
        elif self.user_intent == "ask_fee":
            # 用户询问费用
            response = self._get_script("answer_fee")
            if not self.user_asked_fee:
                # 第一次询问，补充说明
                response += " " + "Ini sudah sesuai perjanjian awal ya."
            return response, None
        elif self.user_intent == "ask_payment_method":
            # 用户询问支付方式
            response = self._get_script("answer_payment_method")
            if not self.user_asked_payment_method:
                # 第一次询问，补充说明
                response += " " + "Pastikan transfer atas nama PT Extra Uang Indonesia ya."
            return response, None
        elif self.user_intent == "question_identity":
            # 用户质疑身份
            return self._get_script("answer_identity"), None
        elif self.user_intent == "request_identity_verification":
            # 用户要求验证身份合法性
            return self._get_script("handle_identity_verification_request"), None
        elif self.user_intent == "inquire_consequences":
            # 用户询问逾期后果
            response = self._get_script("handle_consequence_inquiry")
            return response, None
        elif self.user_intent == "app_uninstalled":
            # 用户说已经卸载了APP
            response = self._get_script("handle_app_uninstalled_problem")
            return response, None
        elif self.user_intent == "request_payment_reminder":
            # 用户要求发送还款提醒
            response = self._get_script("handle_payment_reminder_request")
            return response, None

        # 异议类意图
        elif self.user_intent == "no_money":
            # 用户说没钱 —— 递进式反驳链 P15-A01
            self.no_money_count += 1
            if self.no_money_count == 1:
                return self._get_script("handle_no_money"), ChatState.HANDLE_OBJECTION
            elif self.no_money_count == 2:
                return self._get_script("handle_no_money_level2"), ChatState.HANDLE_OBJECTION
            else:
                return self._get_script("handle_no_money_level3"), ChatState.HANDLE_OBJECTION
        elif self.user_intent == "request_interest_reduction":
            # 用户要求减免利息
            return self._get_script("handle_interest_reduction_request"), ChatState.HANDLE_OBJECTION
        elif self.user_intent == "complain_high_interest":
            # 用户抱怨利率太高
            return self._get_script("handle_high_interest_complaint"), ChatState.HANDLE_OBJECTION
        elif self.user_intent == "threaten":
            # 用户威胁
            return self._get_script("handle_threat"), ChatState.HANDLE_OBJECTION
        elif self.user_intent == "refuse_to_pay":
            # 用户拒绝还款
            return self._get_script("objection_general"), ChatState.HANDLE_OBJECTION

        # 协商类意图
        elif self.user_intent == "ask_extension":
            # 用户询问展期
            if not self.extension_discussed:
                # 第一次询问展期，详细说明
                return self._get_script("explain_extension"), ChatState.CONFIRM_EXTENSION
            else:
                # 已经讨论过展期，再次确认
                return self._get_script("confirm_extension_repeat"), ChatState.CONFIRM_EXTENSION
        elif self.user_intent == "partial_payment":
            # 用户询问部分还款
            if not self.partial_payment_discussed:
                # 第一次询问，详细说明
                return self._get_script("handle_partial_payment"), ChatState.PUSH_FOR_TIME
            else:
                # 已经讨论过，直接询问金额和时间
                return self._get_script("partial_payment_repeat"), ChatState.PUSH_FOR_TIME

        # 无法识别意图
        elif self.user_intent == "unknown":
            # 统计unknown出现的次数
            unknown_count = self.user_history_intents.count("unknown")
            if unknown_count >= 3:
                # 多次无法识别，结束对话
                return self._get_script("unknown_too_many"), ChatState.CLOSE
            else:
                # 请求用户重复
                return self._get_script("handle_unknown"), None

        # 没有匹配到公共意图
        return None, None

    def is_finished(self) -> bool:
        """对话是否结束"""
        return self.state in [ChatState.CLOSE, ChatState.FAILED]

    def is_successful(self) -> bool:
        """对话是否成功（获取到还款时间，或者用户同意展期并给出时间）"""
        return self.state == ChatState.CLOSE and self.commit_time is not None

    def get_log(self) -> ConversationLog:
        """获取对话日志"""
        return ConversationLog(
            session_id=self.session_id,
            chat_group=self.chat_group,
            customer_info={"name": self.customer_name},
            turns=self.conversation.copy(),
            success=self.is_successful(),
            commit_time=self.commit_time,
            end_time=datetime.now().isoformat()
        )

    def reset(self, chat_group: str = "H2", customer_name: Optional[str] = None):
        """重置状态"""
        self.chat_group = chat_group
        self.customer_name = customer_name or "Pak/Bu"
        self.state = ChatState.INIT
        self.conversation = []
        self.commit_time = None
        self.objection_count = 0
        self.max_objections = self.strategy.max_objections  # P15-B01: 恢复策略值
        self.in_llm_fallback = False
        self.llm_turn_count = 0
        self.llm_used_this_turn = False
        self.silence_count = 0
        self.no_money_count = 0
        self.busy_count = 0
        self.dont_know_count = 0
        self.push_round = 0  # P15-B02
        self.extension_discussed = False
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")


class CustomerSimulator:
    """模拟客户回应 - 扩展版"""

    def __init__(self, persona: str = "cooperative"):
        self.persona = persona
        self._init_responses()

    def _init_responses(self):
        """初始化客户回应库"""
        self.customer_responses = {
            "cooperative": {
                "greeting": ["Halo.", "Pagi.", "Siang.", "Sore.", "Iya?"],
                "identity": ["Iya.", "Ya.", "Ya, betul."],
                "purpose": ["Oh, ingatnya.", "Ya.", "Oh ya."],
                "ask_time": ["Jam 5 ya.", "Jam 4.", "Jam 3.", "Besok jam 2."],
                "push": ["Hari ini jam 5.", "Besok jam 3."],
                "commit": ["Iya.", "Ya.", "Oke."],
                "confirm": ["Iya.", "Ya.", "Oke."],
                "close": ["Terima kasih.", "Terima kasih kembali."]
            },
            "busy": {
                "greeting": ["Sibuk.", "Ada apa?", "Sebentar ya."],
                "identity": ["Sibuk nih.", "Nanti ya."],
                "purpose": ["Saya lagi sibuk.", "Nanti saya hubungi balik."],
                "ask_time": ["Saya lagi luar.", "Nanti ya."],
                "push": ["Jam 5 deh."],
                "commit": ["Iya deh.", "Oke."],
                "confirm": ["Ya."],
                "close": ["Iya."]
            },
            "negotiating": {
                "greeting": ["Halo.", "Ada apa?"],
                "identity": ["Ya."],
                "purpose": ["Oh, bisa nggak diperpanjang?"],
                "ask_time": ["Minggu ini bisa?", "Besok bisa?"],
                "push": ["Besok jam 3."],
                "commit": ["Oke, besok jam 3."],
                "confirm": ["Iya."],
                "close": ["Terima kasih."]
            },
            "resistant": {
                "greeting": ["Halo?", "Apaan sih?"],
                "identity": ["Ya, apa?"],
                "purpose": ["Aduh, saya lagi susah.", "Nanti dulu ya."],
                "ask_time": ["Saya belum punya duit.", "Gak bisa."],
                "push": ["Saya benar-benar belum bisa."],
                "commit": [],
                "confirm": [],
                "close": []
            },
            "silent": {
                "greeting": ["...", "", "Iya?"],
                "identity": ["...", "Ya."],
                "purpose": ["...", "Oh."],
                "ask_time": ["...", "Jam 5."],
                "push": ["Jam 5."],
                "commit": ["Iya."],
                "confirm": ["Iya."],
                "close": ["..."]
            },
            "forgetful": {
                "greeting": ["Halo?", "Oh iya."],
                "identity": ["Ya."],
                "purpose": ["Oh ya, saya lupa."],
                "ask_time": ["Nanti ya.", "Sebentar lagi."],
                "push": ["Jam 4 deh."],
                "commit": ["Oke."],
                "confirm": ["Iya."],
                "close": ["Terima kasih."]
            }
        }

    def respond(self, stage: str, agent_said: str) -> str:
        """生成客户回应"""
        if self.persona not in self.customer_responses:
            self.persona = "cooperative"

        responses = self.customer_responses[self.persona].get(stage, [])
        if not responses:
            responses = self.customer_responses["cooperative"].get(stage, ["Iya."])

        return random.choice(responses)


def get_stage_from_state(state: ChatState) -> str:
    """从状态获取阶段名称"""
    stage_map = {
        ChatState.INIT: "greeting",
        ChatState.GREETING: "greeting",
        ChatState.IDENTITY_VERIFY: "identity",
        ChatState.PURPOSE: "purpose",
        ChatState.HANDLE_OBJECTION: "negotiate",
        ChatState.CONFIRM_EXTENSION: "negotiate",
        ChatState.ASK_TIME: "ask_time",
        ChatState.PUSH_FOR_TIME: "push",
        ChatState.COMMIT_TIME: "commit",
        ChatState.HANDLE_BUSY: "close",
        ChatState.HANDLE_WRONG_NUMBER: "close",
        ChatState.CLOSE: "close",
        ChatState.FAILED: "close",
    }
    return stage_map.get(state, "greeting")


async def run_conversation_test(
    chat_group: str = "H2",
    customer_persona: str = "cooperative",
    max_turns: int = 15,
    verbose: bool = True,
    use_tts: bool = False
) -> Dict:
    """运行对话测试"""
    bot = CollectionChatBot(chat_group)
    customer = CustomerSimulator(customer_persona)

    if verbose:
        print(f"\n{'='*70}")
        print(f"场景: {chat_group}环节, 客户类型: {customer_persona}")
        print(f"{'='*70}")

    agent_says, audio_file = await bot.process(use_tts=use_tts)
    if verbose:
        print(f"AGENT: {agent_says}")
        if audio_file:
            print(f"       [音频: {audio_file}]")

    for turn in range(max_turns):
        if bot.is_finished():
            break

        current_stage = get_stage_from_state(bot.state)
        customer_says = customer.respond(current_stage, agent_says)

        if verbose:
            print(f"CUSTOMER: {customer_says}")

        agent_says, audio_file = await bot.process(customer_says, use_tts=use_tts)

        if agent_says:
            if verbose:
                print(f"AGENT: {agent_says}")
                if audio_file:
                    print(f"       [音频: {audio_file}]")
        else:
            if verbose:
                print("AGENT: [对话结束]")
            break

    success = bot.is_successful()
    log = bot.get_log()

    if verbose:
        print(f"\n{'='*70}")
        status_msg = "SUCCESS" if success else "FAILED"
        print(f"对话结束: {status_msg}")
        if bot.commit_time:
            print(f"约定时间: {bot.commit_time}")
        print(f"{'='*70}")

    return {
        "session_id": bot.session_id,
        "chat_group": chat_group,
        "customer_persona": customer_persona,
        "success": success,
        "commit_time": bot.commit_time,
        "log": log
    }


async def run_test_suite(use_tts: bool = False):
    """运行完整测试套件"""
    test_scenarios = [
        ("H2", "cooperative", "H2早期 + 合作客户"),
        ("H2", "busy", "H2早期 + 忙碌客户"),
        ("H2", "negotiating", "H2早期 + 协商客户"),
        ("H2", "silent", "H2早期 + 沉默客户"),
        ("H2", "forgetful", "H2早期 + 健忘客户"),
        ("H1", "cooperative", "H1中期 + 合作客户"),
        ("H1", "negotiating", "H1中期 + 协商客户"),
        ("H1", "busy", "H1中期 + 忙碌客户"),
        ("H1", "forgetful", "H1中期 + 健忘客户"),
        ("S0", "cooperative", "S0晚期 + 合作客户"),
        ("S0", "negotiating", "S0晚期 + 协商客户"),
        ("S0", "resistant", "S0晚期 + 抗拒客户"),
        ("S0", "silent", "S0晚期 + 沉默客户"),
        ("S0", "forgetful", "S0晚期 + 健忘客户"),
    ]

    print(f"\n{'='*80}")
    print(f"开始测试 {len(test_scenarios)} 个场景")
    print(f"{'='*80}")

    results = []
    all_logs = []

    for i, (chat_group, persona, desc) in enumerate(test_scenarios, 1):
        print(f"\n\n--- 场景 {i}: {desc} ---")
        result = await run_conversation_test(
            chat_group, persona, verbose=True, use_tts=use_tts
        )
        results.append(result)
        all_logs.append(result["log"])

    # 汇总结果
    print(f"\n\n{'='*80}")
    print(f"测试结果汇总")
    print(f"{'='*80}")

    for i, result in enumerate(results, 1):
        status = "SUCCESS" if result["success"] else "FAILED"
        time_info = f" (时间: {result['commit_time']})" if result["commit_time"] else ""
        print(f"{i:2d}. {result['chat_group']:2s} + {result['customer_persona']:12s}: {status}{time_info}")

    success_count = sum(1 for r in results if r["success"])
    print(f"\n总体成功率: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")

    # 保存结果
    output_dir = _PROJECT_ROOT / "data/runs/chatbot_tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump([{
            "session_id": r["session_id"],
            "chat_group": r["chat_group"],
            "customer_persona": r["customer_persona"],
            "success": r["success"],
            "commit_time": r["commit_time"]
        } for r in results], f, ensure_ascii=False, indent=2)

    # 保存详细日志
    logs_file = output_dir / f"test_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(logs_file, "w", encoding="utf-8") as f:
        json.dump([{
            "session_id": log.session_id,
            "chat_group": log.chat_group,
            "customer_info": log.customer_info,
            "success": log.success,
            "commit_time": log.commit_time,
            "start_time": log.start_time,
            "end_time": log.end_time,
            "turns": [{"agent": t.agent, "customer": t.customer, "timestamp": t.timestamp} for t in log.turns]
        } for log in all_logs], f, ensure_ascii=False, indent=2)

    print(f"\n完整结果已保存到:")
    print(f"  - {results_file}")
    print(f"  - {logs_file}")

    return results


async def interactive_chat(chat_group: str = "H2", use_tts: bool = False):
    """交互式对话模式"""
    print(f"\n{'='*70}")
    print(f"交互式对话模式 - {chat_group}环节")
    print(f"{'='*70}")
    print("输入 'quit' 或 'exit' 退出\n")

    bot = CollectionChatBot(chat_group)

    agent_says, audio_file = await bot.process(use_tts=use_tts)
    print(f"AGENT: {agent_says}")
    if audio_file:
        print(f"       [音频: {audio_file}]")

    while not bot.is_finished():
        try:
            customer_input = input("CUSTOMER: ").strip()

            if customer_input.lower() in ["quit", "exit", "q"]:
                print("\n结束对话")
                break

            agent_says, audio_file = await bot.process(customer_input, use_tts=use_tts)

            if agent_says:
                print(f"AGENT: {agent_says}")
                if audio_file:
                    print(f"       [音频: {audio_file}]")

        except KeyboardInterrupt:
            print("\n\n结束对话")
            break

    print(f"\n结果: {'成功' if bot.is_successful() else '失败'}")
    if bot.commit_time:
        print(f"约定时间: {bot.commit_time}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="智能催收对话机器人 v3")
    parser.add_argument("--mode", choices=["test", "interactive"], default="test", help="运行模式")
    parser.add_argument("--chat-group", choices=["H2", "H1", "S0"], default="H2", help="催收环节")
    parser.add_argument("--use-tts", action="store_true", help="启用TTS语音合成")
    parser.add_argument("--persona", default="cooperative", help="客户类型 (测试模式)")

    args = parser.parse_args()

    print("="*70)
    print("智能催收对话机器人 v3")
    print("  - 集成TTS语音合成")
    print("  - 完善状态机逻辑")
    print("  - 支持变量替换")
    print("  - 对话日志记录")
    print("="*70)

    if args.mode == "interactive":
        asyncio.run(interactive_chat(args.chat_group, use_tts=args.use_tts))
    else:
        asyncio.run(run_test_suite(use_tts=args.use_tts))


if __name__ == "__main__":
    main()
