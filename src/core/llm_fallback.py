#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能催收对话机器人 - 带LLM Fallback版本 (v4)
当规则引擎无法处理时，自动切换到LLM处理
"""
import random
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
from pathlib import Path
import json
from datetime import datetime
import sys
import io


from core.chatbot import ChatState  # noqa: E402 — 统一使用 chatbot 的权威定义


@dataclass
class ChatTurn:
    """对话回合"""
    agent: str
    customer: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    is_llm_fallback: bool = False  # 新增：标记是否LLM生成


@dataclass
class FallbackTrigger:
    """LLM Fallback触发条件"""
    name: str
    condition: Callable[['CollectionChatBotV4'], bool]
    description: str


class LLMInterface:
    """
    LLM接口抽象
    可以接入 OpenAI/Anthropic/本地模型
    """

    def __init__(self, provider: str = "mock"):
        self.provider = provider

    async def generate(
        self,
        conversation_history: List[Dict[str, str]],
        system_prompt: str,
        context: Dict[str, Any]
    ) -> str:
        """
        调用LLM生成回复

        Args:
            conversation_history: 对话历史 [{"role": "user", "content": "..."}]
            system_prompt: 系统提示
            context: 上下文信息
        """
        # 目前用Mock实现
        return await self._mock_generate(conversation_history, context)

    async def _mock_generate(
        self,
        conversation_history: List[Dict[str, str]],
        context: Dict[str, Any]
    ) -> str:
        """Mock LLM 回复 - 演示用"""
        chat_group = context.get("chat_group", "H2")
        name = context.get("name", "Pak/Bu")

        # 根据场景返回不同的回复
        if chat_group == "H2":
            mock_responses = [
                f"Saya mengerti {name}, apakah bisa bayar besok jam 5?",
                f"Baik {name}, kita tentukan waktu yang cocok untuk Anda.",
                f"Terima kasih {name}, kapan kira-kira Anda bisa melakukan pembayaran?"
            ]
        elif chat_group == "H1":
            mock_responses = [
                f"{name}, kita harus selesaikan ini, apakah jam 3 besok bisa?",
                f"Saya paham situasinya, bisa kita tentukan waktu yang jelas?"
            ]
        else:
            mock_responses = [
                f"{name}, ini sudah cukup lama, bagaimana kalau besok jam 2?",
                f"Kita butuh kepastian, bisa bayar hari ini jam 5?"
            ]

        return random.choice(mock_responses)


class FallbackDetector:
    """
    检测是否需要LLM Fallback
    """

    def __init__(self):
        self.triggers: List[FallbackTrigger] = []
        self._register_default_triggers()

    def _register_default_triggers(self):
        """注册默认触发条件"""

        # 1. 连续多次追问失败
        def too_many_pushes(bot) -> bool:
            return bot.objection_count >= bot.max_objections - 1
        self.triggers.append(FallbackTrigger(
            name="too_many_pushes",
            condition=too_many_pushes,
            description="连续多次追问失败"
        ))

        # 2. 客户回复完全不相关
        def irrelevant_response(bot) -> bool:
            if not bot.conversation or len(bot.conversation) < 2:
                return False
            last_customer = bot.conversation[-1].customer
            if not last_customer:
                return False
            # 检测是否有转移话题的关键词
            divert_keywords = [
                "cuaca", "makan", "lagu", "film", "olahraga",
                "bicara nanti", "tidak ingin bicara"
            ]
            return any(kw in last_customer.lower() for kw in divert_keywords)
        self.triggers.append(FallbackTrigger(
            name="irrelevant_response",
            condition=irrelevant_response,
            description="客户回复不相关/转移话题"
        ))

        # 3. 检测到复杂的抗拒需要特殊处理
        def complex_resistance(bot) -> bool:
            if not bot.conversation or len(bot.conversation) < 2:
                return False
            last_customer = bot.conversation[-1].customer or ""
            # 多种理由组合
            keywords_combo = ["tidak punya uang", "sakit", "kehilangan pekerjaan"]
            count = sum(1 for kw in keywords_combo if kw in last_customer.lower())
            return count >= 2
        self.triggers.append(FallbackTrigger(
            name="complex_resistance",
            condition=complex_resistance,
            description="客户提出多种抗拒理由"
        ))

        # 4. 沉默太久/回复太少
        def too_silent(bot) -> bool:
            if len(bot.conversation) < 3:
                return False
            # 检查最近3轮客户回复
            silent_count = 0
            for turn in reversed(bot.conversation[-3:]):
                customer_text = turn.customer or ""
                if len(customer_text.strip()) < 3 or customer_text.strip() in ["...", "", "iya", "ya"]:
                    silent_count += 1
            return silent_count >= 2
        self.triggers.append(FallbackTrigger(
            name="too_silent",
            condition=too_silent,
            description="客户沉默/回复过于简短"
        ))

    def check(self, bot: 'CollectionChatBotV4') -> Tuple[bool, Optional[FallbackTrigger]]:
        """
        检查是否触发LLM Fallback
        返回 (是否触发, 触发的trigger)
        """
        for trigger in self.triggers:
            if trigger.condition(bot):
                return True, trigger
        return False, None


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
            for key in e.args:
                text = text.replace(f"{{{key}}}", vars.get(key, f"{{{key}}}"))
            return text


class TimeDetector:
    """时间检测器"""

    TIME_PATTERNS = [
        ("jam 12", ["jam 12", "12 siang"]),
        ("jam 11", ["jam 11"]),
        ("jam 10", ["jam 10"]),
        ("jam 9", ["jam 9"]),
        ("jam 8", ["jam 8"]),
        ("jam 7", ["jam 7"]),
        ("jam 6", ["jam 6"]),
        ("jam 5", ["jam 5"]),
        ("jam 4", ["jam 4"]),
        ("jam 3", ["jam 3"]),
        ("jam 2", ["jam 2"]),
        ("jam 1", ["jam 1"]),
        ("hari ini", ["hari ini", "sekarang"]),
        ("besok", ["besok"]),
        ("minggu ini", ["minggu ini"]),
        ("nanti", ["nanti"]),
    ]

    @classmethod
    def detect(cls, text: str) -> Optional[str]:
        """检测时间"""
        if not text:
            return None
        text_lower = text.lower()

        for time_value, patterns in cls.TIME_PATTERNS:
            for pattern in patterns:
                if pattern in text_lower:
                    return time_value

        if "jam" in text_lower:
            words = text_lower.split()
            for i, word in enumerate(words):
                if word == "jam" and i < len(words) - 1:
                    return f"jam {words[i+1]}"
        return None


class CollectionChatBotV4:
    """
    催收对话机器人 - LLM Fallback版本
    """

    def __init__(self, chat_group: str = "H2", customer_name: Optional[str] = None):
        self.chat_group = chat_group
        self.customer_name = customer_name or "Pak/Bu"
        self.state: ChatState = ChatState.INIT
        self.conversation: List[ChatTurn] = []
        self.commit_time: Optional[str] = None
        self.objection_count: int = 0
        self.max_objections: int = 3

        # 核心组件
        self.var_replacer = VariableReplacer()
        self.time_detector = TimeDetector()
        self.fallback_detector = FallbackDetector()
        self.llm = LLMInterface()

        # LLM Fallback相关
        self.in_llm_fallback: bool = False
        self.llm_conversation_count: int = 0
        self.max_llm_turns: int = 3  # 最多LLM回复3轮后回到规则

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._init_script_lib()

    def _init_script_lib(self):
        """初始化话术库"""
        self.script_lib = {
            "greeting": {
                "H2": ["Halo?", "Halo.", "Hello?"],
                "H1": ["Halo?", "Halo.", "Halo, selamat pagi."],
                "S0": ["Halo?", "Halo."]
            },
            "greeting_response": {
                "H2": ["Halo, selamat pagi {name}.", "Halo, selamat siang {name}."],
                "H1": ["Halo, selamat pagi {name}.", "Halo, selamat siang {name}."],
                "S0": ["Halo, selamat sore {name}."]
            },
            "identify": {
                "H2": ["Saya dari aplikasi Extra."],
                "H1": ["Saya dari aplikasi Extra."],
                "S0": ["Saya dari aplikasi Extra."]
            },
            "purpose": {
                "H2": ["Untuk pinjaman ya {name}."],
                "H1": ["Untuk pinjaman yang sudah jatuh tempo."],
                "S0": ["Kita bicara tentang pinjaman yang sudah agak lama ya {name}."]
            },
            "ask_time": {
                "H2": ["Kapan bisa bayar {name}?", "Jam berapa ya?"],
                "H1": ["Kapan bisa melakukan pembayaran?", "Jam berapa ya?"],
                "S0": ["Bagaimana rencana pembayaran {name}?", "Kapan bisa bayar ya?"]
            },
            "push": {
                "H2": ["Jam berapa tepatnya?", "Hari ini jam berapa ya?"],
                "H1": ["Jam berapa tepatnya?", "Besok jam berapa ya?"],
                "S0": ["Jam berapa tepatnya?", "Hari apa ya?", "Jam berapa ya?"]
            },
            "commit_time": {
                "H2": ["Oke, {time} ya {name}.", "Ya, ya, ya. {time} ya {name}."],
                "H1": ["Ya, ya. Oke, {time} ya {name}.", "Saya tunggu {time}."],
                "S0": ["Ya, ya, ya. Oke, {time} ya {name}."]
            },
            "confirm": {
                "H2": ["Ya, ya, ya.", "Iya.", "Baik."],
                "H1": ["Ya, ya.", "Iya.", "Baik."],
                "S0": ["Ya, ya, ya.", "Baik."]
            },
            "wait": {
                "H2": ["Saya tunggu ya.", "Saya tunggu {time}."],
                "H1": ["Saya tunggu ya."],
                "S0": ["Saya tunggu ya."]
            },
            "closing": {
                "H2": ["Terima kasih.", "Terima kasih. Selamat pagi."],
                "H1": ["Terima kasih.", "Terima kasih. Selamat siang."],
                "S0": ["Terima kasih.", "Terima kasih. Selamat sore."]
            },
            "llm_handoff_notice": {
                "H2": ["Saya akan bantu lebih lanjut."],
                "H1": ["Mari kita bicarakan lebih detil."],
                "S0": ["Mari kita cari solusi bersama."]
            }
        }

    def _get_script(self, category: str, **kwargs) -> str:
        """获取话术并替换变量"""
        scripts = self.script_lib.get(category, {}).get(self.chat_group, [])
        script = random.choice(scripts) if scripts else ""
        vars = {"name": self.customer_name}
        vars.update(kwargs)
        return self.var_replacer.replace(script, **vars)

    async def process(
        self,
        customer_input: Optional[str] = None
    ) -> str:
        """
        处理用户输入，返回回复
        - 先尝试规则引擎
        - 检测到需要fallback时，切换到LLM
        """
        if self.state == ChatState.INIT:
            self.state = ChatState.GREETING
            greeting = self._get_script("greeting")
            self.conversation.append(ChatTurn(agent=greeting))
            return greeting

        if customer_input:
            self.conversation[-1].customer = customer_input

        # 检查是否需要LLM Fallback
        need_fallback, trigger = self.fallback_detector.check(self)
        if need_fallback and not self.in_llm_fallback:
            print(f"[Fallback] 触发LLM兜底: {trigger.description}")
            self.in_llm_fallback = True
            self.llm_conversation_count = 0
            self.state = ChatState.LLM_FALLBACK

        if self.in_llm_fallback:
            return await self._process_with_llm(customer_input)

        return await self._process_with_rules(customer_input)

    async def _process_with_rules(self, customer_input: Optional[str]) -> str:
        """规则引擎处理"""
        response = ""
        next_state = self.state

        if self.state == ChatState.GREETING:
            next_state = ChatState.IDENTITY_VERIFY
            greeting_resp = self._get_script("greeting_response")
            identify = self._get_script("identify")
            response = f"{greeting_resp} {identify}"

        elif self.state == ChatState.IDENTITY_VERIFY:
            next_state = ChatState.PURPOSE
            response = self._get_script("purpose")

        elif self.state == ChatState.PURPOSE:
            next_state = ChatState.ASK_TIME
            response = self._get_script("ask_time")

        elif self.state == ChatState.ASK_TIME:
            detected_time = self.time_detector.detect(customer_input or "")
            if detected_time:
                self.commit_time = detected_time
                next_state = ChatState.COMMIT_TIME
                response = self._get_script("commit_time", time=detected_time)
            else:
                if self.objection_count < self.max_objections:
                    self.objection_count += 1
                    next_state = ChatState.PUSH_FOR_TIME
                    response = self._get_script("push")
                else:
                    next_state = ChatState.FAILED

        elif self.state == ChatState.PUSH_FOR_TIME:
            detected_time = self.time_detector.detect(customer_input or "")
            if detected_time:
                self.commit_time = detected_time
                next_state = ChatState.COMMIT_TIME
                response = self._get_script("commit_time", time=detected_time)
            else:
                if self.objection_count < self.max_objections:
                    self.objection_count += 1
                    response = self._get_script("push")
                else:
                    next_state = ChatState.FAILED

        elif self.state == ChatState.COMMIT_TIME:
            next_state = ChatState.CONFIRM_EXTENSION
            response = self._get_script("confirm")

        elif self.state == ChatState.CONFIRM_EXTENSION:
            next_state = ChatState.CLOSE
            wait = self._get_script("wait", time=self.commit_time) if self.commit_time else "Saya tunggu ya."
            closing = self._get_script("closing")
            response = f"{wait} {closing}"

        if response:
            self.conversation.append(ChatTurn(agent=response))
        self.state = next_state
        return response

    async def _process_with_llm(self, customer_input: Optional[str]) -> str:
        """LLM处理"""
        # 构建对话历史
        history = []
        for turn in self.conversation:
            if turn.customer:
                history.append({"role": "user", "content": turn.customer})
            history.append({"role": "assistant", "content": turn.agent})

        # LLM生成
        llm_response = await self.llm.generate(
            conversation_history=history,
            system_prompt=f"You are a debt collection agent. Be polite but firm.",
            context={
                "chat_group": self.chat_group,
                "name": self.customer_name,
                "objection_count": self.objection_count
            }
        )

        # 记录LLM生成的回复
        self.conversation.append(ChatTurn(
            agent=llm_response,
            is_llm_fallback=True
        ))

        # 检测是否从LLM回复中获取到时间
        detected_time = self.time_detector.detect(llm_response)
        if not detected_time and customer_input:
            detected_time = self.time_detector.detect(customer_input)

        if detected_time:
            # 获取到时间了，切回规则引擎进行确认和关闭
            self.commit_time = detected_time
            self.in_llm_fallback = False
            self.state = ChatState.COMMIT_TIME
            response = self._get_script("commit_time", time=detected_time)
            self.conversation.append(ChatTurn(agent=response))
            return response

        # LLM轮数限制
        self.llm_conversation_count += 1
        if self.llm_conversation_count >= self.max_llm_turns:
            print("[Fallback] 达到LLM轮数上限，切回规则引擎")
            self.in_llm_fallback = False
            self.state = ChatState.PUSH_FOR_TIME
            response = self._get_script("push")
            self.conversation.append(ChatTurn(agent=response))
            return response

        return llm_response

    def is_finished(self) -> bool:
        return self.state in [ChatState.CLOSE, ChatState.FAILED]

    def is_successful(self) -> bool:
        return self.state == ChatState.CLOSE and self.commit_time is not None

    def get_stats(self) -> Dict[str, Any]:
        """获取运行统计"""
        llm_turns = sum(1 for turn in self.conversation if turn.is_llm_fallback)
        return {
            "total_turns": len(self.conversation),
            "llm_turns": llm_turns,
            "used_fallback": llm_turns > 0,
            "objection_count": self.objection_count,
            "success": self.is_successful(),
            "commit_time": self.commit_time
        }


# ========== 演示代码 ==========

class ScenarioTester:
    """场景测试"""

    def __init__(self):
        pass

    def run_demo_scenarios(self):
        """运行演示场景"""
        scenarios = [
            self._demo_basic_rule_flow,
            self._demo_fallback_too_many_pushes,
            self._demo_fallback_silent_customer,
        ]

        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{'='*70}")
            print(f"场景 {i}: {scenario.__name__}")
            print('='*70)
            asyncio.run(scenario())

    async def _demo_basic_rule_flow(self):
        """演示：正常规则流程"""
        print("\n[正常规则流程 - 不需要Fallback]")
        bot = CollectionChatBotV4("H2", "Pak Budi")

        # 正常合作客户
        responses = ["Halo", "Iya", "Oh ya", "Jam 5", "Iya"]
        resp_idx = 0

        agent_says = await bot.process()
        print(f"AGENT: {agent_says}")

        while not bot.is_finished() and resp_idx < len(responses):
            customer = responses[resp_idx]
            resp_idx += 1
            print(f"CUSTOMER: {customer}")
            agent_says = await bot.process(customer)
            if agent_says:
                print(f"AGENT: {agent_says}")

        print(f"\n统计: {bot.get_stats()}")

    async def _demo_fallback_too_many_pushes(self):
        """演示：多次追问触发Fallback"""
        print("\n[多次追问 - 触发LLM Fallback]")
        bot = CollectionChatBotV4("H2", "Pak Andi")

        # 客户不断推托
        responses = ["Halo", "Iya", "Oh ya", "Nanti ya", "Saya lagi sibuk", "Besok deh"]
        resp_idx = 0

        agent_says = await bot.process()
        print(f"AGENT: {agent_says}")

        while not bot.is_finished() and resp_idx < len(responses):
            customer = responses[resp_idx]
            resp_idx += 1
            print(f"CUSTOMER: {customer}")
            agent_says = await bot.process(customer)
            if agent_says:
                llm_flag = " [LLM]" if bot.conversation[-1].is_llm_fallback else ""
                print(f"AGENT: {agent_says}{llm_flag}")

        print(f"\n统计: {bot.get_stats()}")

    async def _demo_fallback_silent_customer(self):
        """演示：沉默客户触发Fallback"""
        print("\n[沉默客户 - 触发LLM Fallback]")
        bot = CollectionChatBotV4("S0", "Pak Candra")

        # 客户几乎不说话
        responses = ["...", "Ya", "...", "Iya", "Jam 4"]
        resp_idx = 0

        agent_says = await bot.process()
        print(f"AGENT: {agent_says}")

        while not bot.is_finished() and resp_idx < len(responses):
            customer = responses[resp_idx]
            resp_idx += 1
            print(f"CUSTOMER: {customer}")
            agent_says = await bot.process(customer)
            if agent_says:
                llm_flag = " [LLM]" if bot.conversation[-1].is_llm_fallback else ""
                print(f"AGENT: {agent_says}{llm_flag}")

        print(f"\n统计: {bot.get_stats()}")


if __name__ == "__main__":
    print("="*70)
    print("Collection Chatbot v4 - LLM Fallback")
    print("="*70)

    tester = ScenarioTester()
    tester.run_demo_scenarios()
