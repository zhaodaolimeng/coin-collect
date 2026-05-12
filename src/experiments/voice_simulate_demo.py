#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音催收 Demo — 双工通话演示

Usage:
    python src/experiments/voice_simulate_demo.py --mode live                        # 真人模式（麦克风）
    python src/experiments/voice_simulate_demo.py --mode sim --persona resistant     # 自动仿真
    python src/experiments/voice_simulate_demo.py --mode replay --recording call.wav # 回放

Backward-compatible (defaults to live mode):
    python src/experiments/voice_simulate_demo.py                                    # 同 --mode live
    python src/experiments/voice_simulate_demo.py --persona resistant --resistance high  # 同 --mode sim
"""
import argparse
import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.chatbot import CollectionChatBot, ChatState
from src.core.voice.audio_source import MicrophoneSource, FileSource
from src.core.voice.audio_output import DuplexAudioOutput
from src.core.voice.pipeline import DuplexCallPipeline, PipelineState, PipelineConfig, StepResult
from src.core.voice.vad import SimpleEnergyVAD
from src.core.voice.call_simulator import CallSimulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("voice_demo")

PERSONAS = ["cooperative", "busy", "negotiating", "silent", "forgetful", "resistant", "excuse_master"]
RESISTANCE_LEVELS = ["very_low", "low", "medium", "high", "very_high"]
CHAT_GROUPS = ["H2", "H1", "S0"]

STATUS_ICONS = {
    PipelineState.IDLE: "...",
    PipelineState.LISTENING: "[聆听]",
    PipelineState.PROCESSING: "[处理]",
    PipelineState.RESPONDING: "[播放]",
    PipelineState.INTERRUPTED: "[打断]",
    PipelineState.CLOSING: "[结束]",
    PipelineState.CLOSED: "[完成]",
}


def print_header(mode, persona, resistance, chat_group, max_turns):
    print()
    print("=" * 62)
    print("  语音催收 Demo — 双工通话管线")
    print("=" * 62)
    print(f"  Mode: {mode:<10s}  Group: {chat_group}")
    if mode == "sim":
        print(f"  Persona: {persona:<15s} Resistance: {resistance}")
    print(f"  Max turns: {max_turns}")
    realtime_label = {
        "live": "真人麦克风 (sounddevice)",
        "sim": "自动仿真 (TTS → ASR → Bot)",
        "replay": "文件回放",
    }
    print(f"  模式: {realtime_label.get(mode, mode)}")
    print("=" * 62)


async def run_live_mode(args):
    """真人模式：麦克风 → Pipeline"""
    print_header("live", args.persona, args.resistance, args.chat_group, args.max_turns)

    bot = CollectionChatBot(chat_group=args.chat_group, customer_name=args.customer_name)

    source = MicrophoneSource(sample_rate=16000, block_size=1600)
    output = DuplexAudioOutput(source, barge_in_threshold=0.02)
    vad = SimpleEnergyVAD(sample_rate=16000, energy_threshold=0.01, voice_frames=2, silence_frames=10)

    from src.core.voice.asr import ASRPipeline
    from src.core.voice.tts import TTSManager
    print("  加载 ASR 模型...")
    asr = await ASRPipeline.create(model_size=args.asr_model, corrector=getattr(bot, "asr_corrector", None))
    tts = TTSManager()

    config = PipelineConfig(sample_rate=16000, block_size=1600, silence_duration=1.0, max_speech_duration=15.0)
    pipeline = DuplexCallPipeline(bot, source, output, asr, tts, vad, config=config)

    def on_state(old, new):
        icon = STATUS_ICONS.get(new, "?")
        print(f"  {icon} {old.name} → {new.name}")

    pipeline.on_state_change = on_state

    await pipeline.start()
    print(f"\n  麦克风已就绪。开始说话...\n")

    try:
        while pipeline.state != PipelineState.CLOSED:
            result = await pipeline.step()
            if result and result.asr_text:
                print(f"\n  [用户]: {result.asr_text}")
            if result and result.agent_text:
                print(f"  [Agent]: {result.agent_text[:120]}")
            await asyncio.sleep(0.01)
    except KeyboardInterrupt:
        print("\n  用户中断")
    finally:
        await pipeline.stop()
        print(f"\n  通话结束。轮次: {pipeline.turn_id}")
    return 0


async def run_sim_mode(args):
    """自动仿真模式：CallSimulator"""
    print_header("sim", args.persona, args.resistance, args.chat_group, args.max_turns)

    bot = CollectionChatBot(chat_group=args.chat_group, customer_name=args.customer_name)

    print("  加载 ASR 模型...")
    sim = await CallSimulator.create(
        chatbot=bot,
        persona=args.persona,
        resistance_level=args.resistance,
        chat_group=args.chat_group,
        customer_name=args.customer_name,
        asr_model_size=args.asr_model,
        realtime=args.realtime,
        save_artifacts=not args.no_save,
        output_dir=args.output_dir,
        max_turns=args.max_turns,
    )

    print(f"  就绪。开始模拟...\n")

    try:
        report = await sim.run(max_turns=args.max_turns)

        # 打印每轮
        for turn in report.turns:
            print(f"\n--- Turn {turn.turn_id} ({turn.state_before} → {turn.state_after}) ---")
            if turn.tts_failed:
                print(f"  [TTS FAILED]")
            else:
                print(f"  Customer text : \"{turn.customer_text}\"")
            if turn.vad_dropped:
                print(f"  [VAD DROPPED]")
            elif turn.asr_text:
                print(f"  ASR          : \"{turn.asr_text}\"")
            if turn.agent_text:
                print(f"  Agent        : {turn.agent_text[:120]}")

        print(f"\n  === 完成 ===")
        print(f"  轮次: {report.total_turns} | 结束: {report.conversation_ended}")
        print(f"  最终状态: {report.final_state}")
        if report.artifacts_dir:
            print(f"  Artifacts: {report.artifacts_dir}")
    except KeyboardInterrupt:
        print("\n  用户中断")
    return 0


async def run_replay_mode(args):
    """回放模式：FileSource → Pipeline"""
    print_header("replay", args.persona, args.resistance, args.chat_group, args.max_turns)

    if not args.recording or not Path(args.recording).exists():
        print(f"  [ERROR] 录音文件不存在: {args.recording}")
        return 1

    bot = CollectionChatBot(chat_group=args.chat_group, customer_name=args.customer_name)

    source = FileSource(args.recording, sample_rate=16000, block_size=1600)
    output = DuplexAudioOutput(source, barge_in_threshold=0.02)
    vad = SimpleEnergyVAD(sample_rate=16000, energy_threshold=0.01, voice_frames=2, silence_frames=10)

    from src.core.voice.asr import ASRPipeline
    from src.core.voice.tts import TTSManager
    asr = await ASRPipeline.create(model_size=args.asr_model, corrector=getattr(bot, "asr_corrector", None))
    tts = TTSManager()

    config = PipelineConfig(sample_rate=16000, block_size=1600, silence_duration=1.0, max_speech_duration=15.0)
    pipeline = DuplexCallPipeline(bot, source, output, asr, tts, vad, config=config)

    def on_state(old, new):
        icon = STATUS_ICONS.get(new, "?")
        print(f"  {icon} {old.name} → {new.name}")
    pipeline.on_state_change = on_state

    await pipeline.start()
    print(f"  回放中...\n")
    try:
        while pipeline.state != PipelineState.CLOSED:
            await pipeline.step()
            await asyncio.sleep(0.01)
    except KeyboardInterrupt:
        print("\n  用户中断")
    finally:
        await pipeline.stop()
    print(f"\n  回放结束。轮次: {pipeline.turn_id}")
    return 0


async def main():
    parser = argparse.ArgumentParser(description="语音催收 Demo — 双工通话管线")
    parser.add_argument("--mode", default="sim", choices=["live", "sim", "replay"],
                        help="运行模式: live=真人麦克风, sim=自动仿真, replay=文件回放 (default: sim)")
    parser.add_argument("--persona", default="cooperative", choices=PERSONAS)
    parser.add_argument("--resistance", default="medium", choices=RESISTANCE_LEVELS)
    parser.add_argument("--chat-group", default="H2", choices=CHAT_GROUPS)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--realtime", action="store_true", help="模拟实时对话节奏")
    parser.add_argument("--no-save", action="store_true", help="不保存 artifacts")
    parser.add_argument("--asr-model", default="small", choices=["tiny", "small", "medium"])
    parser.add_argument("--output-dir", default="data/runs/voice_simulations")
    parser.add_argument("--customer-name", default="Budi")
    parser.add_argument("--recording", help="回放模式: 录音文件路径")
    parser.add_argument("--seed", type=int, help="随机种子")
    parser.add_argument("--simulate-interruptions", action="store_true",
                        help="自动仿真中模拟打断")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.seed:
        import random; random.seed(args.seed)

    if args.mode == "sim":
        return await run_sim_mode(args)
    elif args.mode == "replay":
        return await run_replay_mode(args)
    else:
        return await run_live_mode(args)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
