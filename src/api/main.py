from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import sys
from pathlib import Path
from datetime import datetime
import uuid
from typing import Optional, Dict, List
import asyncio
import json

_PROJECT_ROOT = Path(__file__).parent.parent.parent

sys.path.append(str(Path(__file__).parent.parent))

from core.logger import setup_logging, get_logger

logger = get_logger(__name__)

from api.schemas import (
    ChatTurnRequest,
    ChatTurnResponse,
    ChatSessionResponse,
    ChatLogEntry,
    ChatState,
    ChatGroup,
    CustomerPersona,
    TestScenarioRequest,
    TestResultResponse,
    MessageResponse,
    HealthResponse,
    StatsResponse,
    ScriptResponse,
    ScriptUpdateRequest,
    TranslateRequest,
    TranslateResponse,
    SimulateCustomerRequest,
    SimulateCustomerResponse,
    VoiceStartRequest,
    VoiceTurnRequest,
    VoiceSessionResponse,
    ASRResponse,
    SessionSummary,
    SessionListResponse,
)
from api.database import (
    get_db,
    init_db,
    ChatSession as DBChatSession,
    ChatTurn as DBChatTurn,
)
from core.chatbot import (
    CollectionChatBot,
    get_stage_from_state,
)
from core.simulator import (
    RealCustomerSimulatorV2,
)
from core.metrics import (
    collector,
    ConversationMetrics,
    PerformanceMetrics,
    get_system_metrics,
)

# 翻译服务
from core.translator import translate_text


def convert_bot_state_to_schema(bot_state):
    """将chatbot的状态转换为schema兼容的状态"""
    state_map = {
        'INIT': 'init',
        'GREETING': 'greeting',
        'IDENTITY_VERIFY': 'identify',
        'PURPOSE': 'purpose',
        'ASK_TIME': 'ask_time',
        'PUSH_FOR_TIME': 'push_for_time',
        'COMMIT_TIME': 'commit_time',
        'CONFIRM_EXTENSION': 'negotiate',
        'HANDLE_OBJECTION': 'negotiate',
        'HANDLE_BUSY': 'close',
        'HANDLE_WRONG_NUMBER': 'close',
        'CLOSE': 'close',
        'FAILED': 'failed',
        'LLM_FALLBACK': 'push_for_time'
    }

    # 获取状态名称
    if hasattr(bot_state, 'name'):
        state_name = bot_state.name
    else:
        state_name = str(bot_state).split('.')[-1]

    return state_map.get(state_name, 'init')


app = FastAPI(
    title="智能催收对话系统 API",
    description="基于状态机的印尼语催收对话系统",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

active_sessions: Dict[str, CollectionChatBot] = {}
simulator = RealCustomerSimulatorV2()


@app.on_event("startup")
async def startup_event():
    setup_logging()
    init_db()
    logger.info("Database initialized!")

    # 后台预加载翻译模型，避免首次翻译请求等待
    import threading
    def warmup_translator():
        try:
            from core.translator import get_translator
            get_translator()
            logger.info("Translation model preloaded!")
        except Exception as e:
            logger.warning(f"Translation preload skipped: {e}")
    threading.Thread(target=warmup_translator, daemon=True).start()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
    )


def save_session_to_db(db: Session, bot: CollectionChatBot, chat_group: str):
    db_session = DBChatSession(
        session_id=bot.session_id,
        chat_group=chat_group,
        customer_name=bot.customer_name,
        customer_phone=getattr(bot, 'customer_phone', None),
        is_finished=bot.is_finished(),
        is_successful=bot.is_successful(),
        commit_time=bot.commit_time,
        conversation_length=len(bot.conversation),
    )
    db.add(db_session)
    db.flush()

    for turn_num, turn in enumerate(bot.conversation, 1):
        turn_state = turn.state if turn.state is not None else bot.state
        db_turn = DBChatTurn(
            session_id=db_session.id,
            turn_number=turn_num,
            agent_text=turn.agent,
            customer_text=turn.customer,
            state=get_stage_from_state(turn_state),
            timestamp=turn.timestamp,
        )
        db.add(db_turn)

    db.commit()
    db.refresh(db_session)
    return db_session


@app.post("/chat/start", response_model=ChatTurnResponse)
async def start_chat(request: ChatTurnRequest, db: Session = Depends(get_db)):
    session_id = str(uuid.uuid4())

    # P15-D01: 查询用户历史记忆
    user_memory = None
    if request.customer_phone:
        from core.user_memory import UserMemoryStore
        store = UserMemoryStore(db)
        user_memory = store.load(request.customer_phone)

    bot = CollectionChatBot(
        chat_group=request.chat_group.value,
        customer_name=request.customer_name,
        user_memory=user_memory,
    )
    bot.session_id = session_id

    active_sessions[session_id] = bot

    start_time = datetime.now()
    agent_response, audio_file = await bot.process(use_tts=False)
    latency_ms = (datetime.now() - start_time).total_seconds() * 1000

    save_session_to_db(db, bot, request.chat_group.value)

    return ChatTurnResponse(
        session_id=session_id,
        agent_response=agent_response,
        current_state=ChatState(convert_bot_state_to_schema(bot.state)),
        commit_time=bot.commit_time,
        conversation_length=len(bot.conversation),
        is_finished=bot.is_finished(),
        is_successful=bot.is_successful(),
        audio_file=audio_file,
        latency_ms=round(latency_ms, 2),
        llm_used=False,
    )


@app.post("/chat/turn", response_model=ChatTurnResponse)
async def chat_turn(request: ChatTurnRequest, db: Session = Depends(get_db)):
    session_id = request.session_id
    if not session_id or session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="会话不存在")

    bot = active_sessions[session_id]

    if bot.is_finished():
        del active_sessions[session_id]
        raise HTTPException(status_code=400, detail="会话已结束")

    start_time = datetime.now()
    agent_response, audio_file = await bot.process(
        customer_input=request.customer_input,
        use_tts=False,
    )
    latency_ms = (datetime.now() - start_time).total_seconds() * 1000

    db_session = db.query(DBChatSession).filter(
        DBChatSession.session_id == session_id
    ).first()

    if db_session:
        db_session.is_finished = bot.is_finished()
        db_session.is_successful = bot.is_successful()
        db_session.commit_time = bot.commit_time
        db_session.conversation_length = len(bot.conversation)

        if bot.is_finished():
            db_session.end_time = datetime.now().isoformat()

        # 全量重写对话轮到数据库 (避免 customer_text 更新丢失)
        if bot.conversation:
            db.query(DBChatTurn).filter(
                DBChatTurn.session_id == db_session.id
            ).delete()
            for i, turn in enumerate(bot.conversation):
                turn_state = turn.state if turn.state is not None else bot.state
                db_turn = DBChatTurn(
                    session_id=db_session.id,
                    turn_number=i + 1,
                    agent_text=turn.agent,
                    customer_text=turn.customer,
                    state=get_stage_from_state(turn_state),
                    timestamp=turn.timestamp,
                )
                db.add(db_turn)

        db.commit()

    # 对话结束后自动清理内存中的会话
    if bot.is_finished():
        active_sessions.pop(session_id, None)

    return ChatTurnResponse(
        session_id=session_id,
        agent_response=agent_response,
        current_state=ChatState(convert_bot_state_to_schema(bot.state)),
        commit_time=bot.commit_time,
        conversation_length=len(bot.conversation),
        is_finished=bot.is_finished(),
        is_successful=bot.is_successful(),
        audio_file=audio_file,
        latency_ms=round(latency_ms, 2),
        llm_used=getattr(bot, "llm_used_this_turn", False),
    )


@app.get("/chat/session/{session_id}", response_model=ChatSessionResponse)
async def get_session(session_id: str, db: Session = Depends(get_db)):
    if session_id in active_sessions:
        bot = active_sessions[session_id]
        log = bot.get_log()

        conversation_log = []
        for turn in bot.conversation:
            if turn.agent:
                conversation_log.append(ChatLogEntry(
                    role="agent",
                    text=turn.agent,
                    timestamp=turn.timestamp,
                ))
            if turn.customer:
                conversation_log.append(ChatLogEntry(
                    role="customer",
                    text=turn.customer,
                    timestamp=turn.timestamp,
                ))

        return ChatSessionResponse(
            session_id=log.session_id,
            chat_group=ChatGroup(log.chat_group),
            customer_name=bot.customer_name,
            is_finished=bot.is_finished(),
            is_successful=bot.is_successful(),
            commit_time=log.commit_time,
            conversation_length=len(conversation_log),
            conversation_log=conversation_log,
            start_time=log.start_time,
            end_time=log.end_time,
            created_at=log.start_time,
        )

    db_session = db.query(DBChatSession).filter(DBChatSession.session_id == session_id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="会话不存在")

    db_turns = db.query(DBChatTurn).filter(DBChatTurn.session_id == db_session.id).order_by(DBChatTurn.turn_number).all()

    conversation_log = []
    for turn in db_turns:
        if turn.agent_text:
            conversation_log.append(ChatLogEntry(
                role="agent",
                text=turn.agent_text,
                timestamp=turn.timestamp,
            ))
        if turn.customer_text:
            conversation_log.append(ChatLogEntry(
                role="customer",
                text=turn.customer_text,
                timestamp=turn.timestamp,
            ))

    return ChatSessionResponse(
        session_id=db_session.session_id,
        chat_group=ChatGroup(db_session.chat_group),
        customer_name=db_session.customer_name,
        is_finished=db_session.is_finished,
        is_successful=db_session.is_successful,
        commit_time=db_session.commit_time,
        conversation_length=db_session.conversation_length,
        conversation_log=conversation_log,
        start_time=db_session.start_time,
        end_time=db_session.end_time,
        created_at=db_session.created_at,
    )


@app.post("/chat/session/{session_id}/close", response_model=MessageResponse)
async def close_session(session_id: str, db: Session = Depends(get_db)):
    db_session = db.query(DBChatSession).filter(DBChatSession.session_id == session_id).first()

    if session_id in active_sessions:
        bot = active_sessions[session_id]
        if db_session:
            db_session.is_finished = bot.is_finished()
            db_session.is_successful = bot.is_successful()
            db_session.end_time = datetime.now().isoformat()
            db.commit()
        del active_sessions[session_id]
        return MessageResponse(message="会话已关闭")

    if db_session:
        db_session.is_finished = True
        db_session.end_time = datetime.now().isoformat()
        db.commit()
        return MessageResponse(message="会话已关闭")

    raise HTTPException(status_code=404, detail="会话不存在")


@app.get("/chat/sessions", response_model=List[ChatSessionResponse])
async def list_sessions(skip: int = 0, limit: int = 100, chat_group: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(DBChatSession)
    if chat_group:
        query = query.filter(DBChatSession.chat_group == chat_group)
    db_sessions = query.order_by(DBChatSession.created_at.desc()).offset(skip).limit(limit).all()

    results = []
    for db_session in db_sessions:
        db_turns = db.query(DBChatTurn).filter(DBChatTurn.session_id == db_session.id).order_by(DBChatTurn.turn_number).all()

        conversation_log = []
        for turn in db_turns:
            if turn.agent_text:
                conversation_log.append(ChatLogEntry(
                    role="agent",
                    text=turn.agent_text,
                    timestamp=turn.timestamp,
                ))
            if turn.customer_text:
                conversation_log.append(ChatLogEntry(
                    role="customer",
                    text=turn.customer_text,
                    timestamp=turn.timestamp,
                ))

        results.append(ChatSessionResponse(
            session_id=db_session.session_id,
            chat_group=ChatGroup(db_session.chat_group),
            customer_name=db_session.customer_name,
            is_finished=db_session.is_finished,
            is_successful=db_session.is_successful,
            commit_time=db_session.commit_time,
            conversation_length=db_session.conversation_length,
            conversation_log=conversation_log,
            start_time=db_session.start_time,
            end_time=db_session.end_time,
            created_at=db_session.created_at,
        ))

    return results


@app.get("/chat/sessions/active", response_model=SessionListResponse)
async def list_active_sessions(db: Session = Depends(get_db)):
    """返回轻量会话列表：进行中 + 已完成"""
    active_summaries = []
    for sid, bot in active_sessions.items():
        state_name = None
        if hasattr(bot.state, 'name'):
            state_name = bot.state.name
        active_summaries.append(SessionSummary(
            session_id=sid,
            chat_group=ChatGroup(bot.chat_group) if bot.chat_group in ["H2","H1","S0"] else ChatGroup.H2,
            customer_name=bot.customer_name,
            is_finished=bot.is_finished(),
            is_successful=False,
            state=state_name,
            conversation_length=len(bot.conversation),
            start_time=datetime.now().isoformat(),
            end_time=None,
        ))

    completed_summaries = []
    db_sessions = db.query(DBChatSession).filter(
        DBChatSession.is_finished == True
    ).order_by(DBChatSession.created_at.desc()).limit(50).all()

    for s in db_sessions:
        if s.session_id in active_sessions:
            continue
        completed_summaries.append(SessionSummary(
            session_id=s.session_id,
            chat_group=ChatGroup(s.chat_group),
            customer_name=s.customer_name,
            is_finished=True,
            is_successful=s.is_successful,
            state=None,
            conversation_length=s.conversation_length,
            start_time=s.start_time,
            end_time=s.end_time,
        ))

    return SessionListResponse(
        active=active_summaries,
        completed=completed_summaries,
    )


@app.delete("/chat/session/{session_id}", response_model=MessageResponse)
async def delete_session(session_id: str, db: Session = Depends(get_db)):
    """删除会话（内存中 + 数据库）"""
    deleted_memory = False
    if session_id in active_sessions:
        del active_sessions[session_id]
        deleted_memory = True

    db_session = db.query(DBChatSession).filter(
        DBChatSession.session_id == session_id
    ).first()
    if db_session:
        db.query(DBChatTurn).filter(DBChatTurn.session_id == db_session.id).delete()
        db.delete(db_session)
        db.commit()
        return MessageResponse(message=f"会话 {session_id} 已删除")

    if deleted_memory:
        return MessageResponse(message=f"会话 {session_id} (仅内存) 已删除")

    raise HTTPException(status_code=404, detail="会话不存在")


@app.post("/test/scenario", response_model=TestResultResponse)
async def run_test_scenario(request: TestScenarioRequest):
    results = []
    success_count = 0

    for i in range(request.num_tests):
        session_id = str(uuid.uuid4())
        bot = CollectionChatBot(
            chat_group=request.chat_group.value,
        )
        bot.session_id = session_id

        agent_text, _ = await bot.process(use_tts=False)

        push_count = 0
        max_turns = 20

        for turn in range(max_turns):
            if bot.is_finished():
                break

            if "jam berapa" in agent_text.lower() or "kapan" in agent_text.lower():
                push_count += 1

            customer_text = simulator.generate_response(
                stage=get_stage_from_state(bot.state),
                chat_group=request.chat_group.value,
                persona=request.persona.value,
                push_count=push_count,
            )

            agent_text, _ = await bot.process(customer_text, use_tts=False)

        is_success = bot.is_successful()

        if is_success:
            success_count += 1

        results.append({
            "session_id": session_id,
            "success": is_success,
            "commit_time": bot.commit_time,
            "conversation_length": len(bot.conversation),
        })

    success_rate = success_count / request.num_tests if request.num_tests > 0 else 0

    return TestResultResponse(
        total_tests=request.num_tests,
        success_count=success_count,
        failed_count=request.num_tests - success_count,
        success_rate=round(success_rate * 100, 2),
        results=results,
    )


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    audio_path = _PROJECT_ROOT / "data/runs/tts_output" / filename
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="音频文件不存在")
    suffix = audio_path.suffix.lower()
    mime_map = {
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.ogg': 'audio/ogg',
        '.opus': 'audio/ogg',
        '.webm': 'audio/webm',
    }
    media_type = mime_map.get(suffix, 'audio/mpeg')
    return FileResponse(audio_path, media_type=media_type)


@app.get("/")
async def root():
    index_path = Path(__file__).parent.parent / "static" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return MessageResponse(message="欢迎使用智能催收对话系统 API")


# ============ 管理API ============

@app.get("/admin/stats", response_model=StatsResponse)
async def get_stats(db: Session = Depends(get_db)):
    """获取系统统计数据"""
    from sqlalchemy import func

    # 会话统计
    total_sessions = db.query(func.count(DBChatSession.id)).scalar() or 0
    successful_sessions = db.query(func.count(DBChatSession.id)).filter(
        DBChatSession.is_successful == True
    ).scalar() or 0
    success_rate = (successful_sessions / total_sessions * 100) if total_sessions > 0 else 0.0

    # 回合统计
    total_turns = db.query(func.count(DBChatTurn.id)).scalar() or 0
    avg_turns = (total_turns / total_sessions) if total_sessions > 0 else 0.0

    # 按组别统计
    chat_group_stats = {}
    groups = db.query(DBChatSession.chat_group, func.count(DBChatSession.id)).group_by(
        DBChatSession.chat_group
    ).all()

    for group, count in groups:
        successful = db.query(func.count(DBChatSession.id)).filter(
            DBChatSession.chat_group == group,
            DBChatSession.is_successful == True
        ).scalar() or 0
        chat_group_stats[group] = {
            "total": count,
            "successful": successful,
            "success_rate": round(successful / count * 100, 1) if count > 0 else 0.0
        }

    return StatsResponse(
        total_sessions=total_sessions,
        successful_sessions=successful_sessions,
        success_rate=round(success_rate, 1),
        total_turns=total_turns,
        avg_turns_per_session=round(avg_turns, 1),
        active_sessions=len(active_sessions),
        chat_group_stats=chat_group_stats
    )


@app.get("/admin/scripts", response_model=List[ScriptResponse])
async def list_scripts(
    chat_group: Optional[str] = None,
    category: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """列出脚本库"""
    from api.database import ScriptLibrary

    query = db.query(ScriptLibrary)

    if chat_group:
        query = query.filter(ScriptLibrary.chat_group == chat_group)
    if category:
        query = query.filter(ScriptLibrary.category == category)

    scripts = query.order_by(ScriptLibrary.category, ScriptLibrary.script_key).all()

    return [
        ScriptResponse(
            id=s.id,
            category=s.category,
            chat_group=s.chat_group,
            script_key=s.script_key,
            script_text=s.script_text,
            variables=s.variables,
            is_active=s.is_active
        )
        for s in scripts
    ]


@app.get("/admin/scripts/{script_id}", response_model=ScriptResponse)
async def get_script(script_id: int, db: Session = Depends(get_db)):
    """获取单个脚本"""
    from api.database import ScriptLibrary

    script = db.query(ScriptLibrary).filter(ScriptLibrary.id == script_id).first()
    if not script:
        raise HTTPException(status_code=404, detail="脚本不存在")

    return ScriptResponse(
        id=script.id,
        category=script.category,
        chat_group=script.chat_group,
        script_key=script.script_key,
        script_text=script.script_text,
        variables=script.variables,
        is_active=script.is_active
    )


@app.put("/admin/scripts/{script_id}", response_model=ScriptResponse)
async def update_script(
    script_id: int,
    request: ScriptUpdateRequest,
    db: Session = Depends(get_db)
):
    """更新脚本"""
    from api.database import ScriptLibrary

    script = db.query(ScriptLibrary).filter(ScriptLibrary.id == script_id).first()
    if not script:
        raise HTTPException(status_code=404, detail="脚本不存在")

    if request.script_text is not None:
        script.script_text = request.script_text
    if request.is_active is not None:
        script.is_active = request.is_active
    if request.variables is not None:
        script.variables = request.variables

    script.updated_at = datetime.now().isoformat()
    db.commit()
    db.refresh(script)

    return ScriptResponse(
        id=script.id,
        category=script.category,
        chat_group=script.chat_group,
        script_key=script.script_key,
        script_text=script.script_text,
        variables=script.variables,
        is_active=script.is_active
    )


@app.get("/admin/metrics")
async def get_metrics():
    """获取系统指标"""
    return get_system_metrics()


@app.post("/admin/metrics/reset")
async def reset_metrics():
    """重置指标"""
    collector.reset()
    return MessageResponse(message="指标已重置")


# ============ 翻译API ============

@app.post("/api/translate", response_model=TranslateResponse)
async def translate_endpoint(request: TranslateRequest):
    """翻译文本 - 印尼文<->英文"""
    text = request.text.strip()
    source = request.source
    target = request.target

    try:
        result = translate_text(text, source, target)
        return TranslateResponse(
            original_text=result.original_text,
            translated_text=result.translated_text,
            source=result.source_lang,
            target=result.target_lang,
            success=result.success
        )
    except Exception as e:
        logger.debug(f"Translation endpoint error: {e}")
        return TranslateResponse(
            original_text=text,
            translated_text=text,
            source=source,
            target=target,
            success=False
        )


# ============ 仿真客户API ============

@app.post("/api/simulate-customer", response_model=SimulateCustomerResponse)
async def simulate_customer(request: SimulateCustomerRequest, db: Session = Depends(get_db)):
    """仿真客户回复"""
    try:
        session_id = request.session_id
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="会话不存在")

        bot = active_sessions[session_id]

        # 获取当前状态对应的阶段
        current_stage = get_stage_from_state(bot.state)

        # 使用模拟器生成回复
        persona = request.persona.value

        # 计算push_count（简单计算）
        push_count = 0
        for turn in bot.conversation:
            if turn.customer:
                push_count += 1

        customer_response = simulator.generate_response(
            stage=current_stage,
            chat_group=bot.chat_group,
            persona=persona,
            push_count=push_count,
            resistance_level=request.resistance_level
        )

        return SimulateCustomerResponse(
            customer_response=customer_response,
            persona=persona,
            resistance_level=request.resistance_level,
            success=True
        )

    except Exception as e:
        return SimulateCustomerResponse(
            customer_response="",
            persona=request.persona.value,
            resistance_level=request.resistance_level,
            success=False
        )


# ============ 语音模式 API ============

@app.post("/voice/start", response_model=VoiceSessionResponse)
async def voice_start(request: VoiceStartRequest, db: Session = Depends(get_db)):
    """启动语音会话，返回TTS音频"""
    from core.voice.tts import TTSManager
    import base64
    import io

    session_id = str(uuid.uuid4())

    bot = CollectionChatBot(
        chat_group=request.chat_group.value,
        customer_name=request.customer_name,
    )
    bot.session_id = session_id
    active_sessions[session_id] = bot

    agent_text, _ = await bot.process(use_tts=False)

    # TTS 合成
    tts = TTSManager()
    tts_result = await tts.synthesize(agent_text, voice="id-ID-ArdiNeural", engine="edge_tts")

    audio_base64 = None
    audio_file_url = None

    if tts_result.success and tts_result.audio_file:
        audio_file_url = f"/audio/{Path(tts_result.audio_file).name}"
    if tts_result.success and tts_result.audio_data is not None:
        try:
            import soundfile as sf
            with io.BytesIO() as buf:
                sf.write(buf, tts_result.audio_data, tts_result.sample_rate or 16000, format="WAV")
                buf.seek(0)
                audio_base64 = base64.b64encode(buf.read()).decode()
        except ImportError:
            pass

    save_session_to_db(db, bot, request.chat_group.value)

    return VoiceSessionResponse(
        session_id=session_id,
        agent_text=agent_text,
        audio_data_base64=audio_base64,
        audio_file=audio_file_url,
        state=convert_bot_state_to_schema(bot.state),
        is_finished=bot.is_finished(),
        is_successful=bot.is_successful(),
    )


@app.post("/voice/turn", response_model=VoiceSessionResponse)
async def voice_turn(request: VoiceTurnRequest, db: Session = Depends(get_db)):
    """语音会话轮次，返回TTS音频"""
    import base64
    import io

    session_id = request.session_id
    if not session_id or session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="会话不存在")

    bot = active_sessions[session_id]
    if bot.is_finished():
        del active_sessions[session_id]
        raise HTTPException(status_code=400, detail="会话已结束")

    agent_text, _ = await bot.process(
        customer_input=request.customer_input,
        use_tts=False,
    )

    # TTS 合成
    from core.voice.tts import TTSManager
    tts = TTSManager()
    tts_result = await tts.synthesize(agent_text, voice="id-ID-ArdiNeural", engine="edge_tts")

    audio_base64 = None
    audio_file_url = None

    if tts_result.success and tts_result.audio_file:
        audio_file_url = f"/audio/{Path(tts_result.audio_file).name}"
    if tts_result.success and tts_result.audio_data is not None:
        try:
            import soundfile as sf
            with io.BytesIO() as buf:
                sf.write(buf, tts_result.audio_data, tts_result.sample_rate or 16000, format="WAV")
                buf.seek(0)
                audio_base64 = base64.b64encode(buf.read()).decode()
        except ImportError:
            pass

    # 更新数据库
    db_session = db.query(DBChatSession).filter(
        DBChatSession.session_id == session_id
    ).first()
    if db_session:
        db_session.is_finished = bot.is_finished()
        db_session.is_successful = bot.is_successful()
        db_session.commit_time = bot.commit_time
        db_session.conversation_length = len(bot.conversation)

        if bot.is_finished():
            db_session.end_time = datetime.now().isoformat()

        # 全量重写对话轮到数据库 (避免 customer_text 更新丢失)
        if bot.conversation:
            db.query(DBChatTurn).filter(
                DBChatTurn.session_id == db_session.id
            ).delete()
            for i, turn in enumerate(bot.conversation):
                turn_state = turn.state if turn.state is not None else bot.state
                db_turn = DBChatTurn(
                    session_id=db_session.id,
                    turn_number=i + 1,
                    agent_text=turn.agent,
                    customer_text=turn.customer,
                    state=get_stage_from_state(turn_state),
                    timestamp=turn.timestamp,
                )
                db.add(db_turn)

        db.commit()

    # 对话结束后自动清理内存中的会话
    if bot.is_finished():
        active_sessions.pop(session_id, None)

    return VoiceSessionResponse(
        session_id=session_id,
        agent_text=agent_text,
        audio_data_base64=audio_base64,
        audio_file=audio_file_url,
        state=convert_bot_state_to_schema(bot.state),
        is_finished=bot.is_finished(),
        is_successful=bot.is_successful(),
    )


@app.post("/voice/end")
async def voice_end(request: VoiceTurnRequest, db: Session = Depends(get_db)):
    """结束语音会话 — 挂断时调用，标记会话已完成"""
    session_id = request.session_id
    if not session_id or session_id not in active_sessions:
        return {"status": "not_found", "message": "会话不存在或已过期"}

    bot = active_sessions[session_id]
    from core.chatbot import ChatState
    bot.state = ChatState.CLOSE

    # Update existing DB record (don't INSERT duplicate)
    db_session = db.query(DBChatSession).filter(
        DBChatSession.session_id == session_id
    ).first()
    if db_session:
        db_session.is_finished = True
        db_session.end_time = datetime.now().isoformat()
        db.commit()

    active_sessions.pop(session_id, None)

    return {"status": "ok", "message": "会话已结束", "session_id": session_id}


@app.get("/voice/voices")
async def list_tts_voices(locale: Optional[str] = "id"):
    """列出可用的TTS语音"""
    from core.voice.tts import TTSManager
    tts = TTSManager()
    engine = tts.get_engine()
    if engine:
        return {"voices": await engine.list_voices(locale=locale)}
    return {"voices": [], "error": "No TTS engine available"}


# ============ 语音仿真流式 API ============

# Global ASR warmup cache
_asr_pipeline_cache = None


@app.post("/voice/warmup")
async def voice_warmup(asr_model: str = "tiny"):
    """预热 ASR 模型，减少首次自动仿真等待时间"""
    global _asr_pipeline_cache
    try:
        from core.voice.asr import ASRPipeline
        if _asr_pipeline_cache is None or not _asr_pipeline_cache.is_available:
            _asr_pipeline_cache = await ASRPipeline.create(model_size=asr_model)
        return {"status": "ok", "asr_available": _asr_pipeline_cache.is_available, "model": asr_model}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/voice/asr", response_model=ASRResponse)
async def voice_asr(audio: UploadFile = File(...)):
    """语音转文字 - 支持 webm/opus/wav 格式"""
    import tempfile
    import subprocess
    import numpy as np

    global _asr_pipeline_cache

    try:
        # Save uploaded audio to temp file
        suffix = Path(audio.filename).suffix if audio.filename else '.webm'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            content = await audio.read()
            tmp_in.write(content)
            input_path = tmp_in.name

        # Convert to WAV using ffmpeg
        wav_path = input_path + '.wav'
        try:
            result = subprocess.run(
                ['ffmpeg', '-y', '-i', input_path,
                 '-ar', '16000', '-ac', '1', '-f', 'wav',
                 '-acodec', 'pcm_s16le', wav_path],
                capture_output=True, timeout=15,
            )
            if result.returncode != 0:
                return ASRResponse(text='', success=False,
                                   error=f'Audio conversion failed: {result.stderr.decode()[:200]}')
        except subprocess.TimeoutExpired:
            return ASRResponse(text='', success=False, error='Audio conversion timed out')
        except FileNotFoundError:
            return ASRResponse(text='', success=False, error='ffmpeg not installed on server')

        # Ensure ASR pipeline is loaded
        if _asr_pipeline_cache is None or not _asr_pipeline_cache.is_available:
            from core.voice.asr import ASRPipeline
            _asr_pipeline_cache = await ASRPipeline.create(model_size='tiny')

        # Transcribe
        text = _asr_pipeline_cache.transcribe_file(wav_path)

        # Cleanup
        Path(input_path).unlink(missing_ok=True)
        Path(wav_path).unlink(missing_ok=True)

        return ASRResponse(text=text.strip(), success=bool(text.strip()))

    except Exception as e:
        logger.error(f"ASR endpoint error: {e}")
        return ASRResponse(text='', success=False, error=str(e))


@app.websocket("/voice/duplex/ws")
async def voice_duplex_websocket(websocket: WebSocket):
    """WebSocket 双工通话端点。浏览器流式推送麦克风音频，服务端实时返回 Agent 音频。

    Query params:
        chat_group: H2|H1|S0 (default H2)
        customer_name: 客户名 (default "User")
    """
    await websocket.accept()

    try:
        from urllib.parse import parse_qs
        qs = parse_qs(str(websocket.url.query))
        chat_group = qs.get("chat_group", ["H2"])[0]
        customer_name = qs.get("customer_name", ["User"])[0]

        import uuid
        from src.core.chatbot import CollectionChatBot

        session_id = str(uuid.uuid4())
        bot = CollectionChatBot(chat_group=chat_group, customer_name=customer_name)
        bot.session_id = session_id
        active_sessions[session_id] = bot

        from src.api.voice_ws_handler import handle_duplex_ws
        await handle_duplex_ws(websocket, bot)
    except Exception as e:
        logger.error(f"WS handler error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/voice/simulate/stream")
async def voice_simulate_stream(
    persona: str = "cooperative",
    resistance: str = "medium",
    chat_group: str = "H2",
    max_turns: int = 20,
    asr_model: str = "tiny",
    customer_name: str = "Budi",
):
    """
    SSE流式语音仿真。

    每轮推送一个JSON事件，包含客户/Agent文本和音频URL。
    前端可按顺序播放音频。

    Query params:
        persona: cooperative|busy|negotiating|silent|forgetful|resistant|excuse_master
        resistance: very_low|low|medium|high|very_high
        chat_group: H2|H1|S0
        max_turns: 最大轮数 (default 15)
        asr_model: tiny|small|medium (default small)
    """
    from core.chatbot import CollectionChatBot
    from core.voice.call_simulator import CallSimulator
    from starlette.responses import StreamingResponse
    import asyncio as aio

    async def event_generator():
        global _asr_pipeline_cache
        import uuid as _uuid

        conn_id = str(_uuid.uuid4())[:8]
        logger.info(f"[SSE:{conn_id}] Connection opened — persona={persona} resistance={resistance} group={chat_group} customer={customer_name}")

        bot = None
        try:
            bot = CollectionChatBot(
                chat_group=chat_group,
                customer_name=customer_name,
            )
            bot.session_id = str(_uuid.uuid4())
            active_sessions[bot.session_id] = bot
            logger.debug(f"[SSE:{conn_id}] Bot created session_id={bot.session_id}")

            # 加载ASR模型时发送心跳
            yield f": loading_asr\n\n"
            logger.debug(f"[SSE:{conn_id}] Loading CallSimulator...")

            sim_start = asyncio.get_event_loop().time()
            sim = await CallSimulator.create(
                chatbot=bot,
                persona=persona,
                resistance_level=resistance,
                chat_group=chat_group,
                customer_name=customer_name,
                asr_model_size=asr_model,
                realtime=False,
                save_artifacts=False,
                _asr_pipeline=_asr_pipeline_cache,
            )
            sim_elapsed = asyncio.get_event_loop().time() - sim_start
            logger.info(f"[SSE:{conn_id}] Simulator ready in {sim_elapsed:.1f}s, ASR available={sim._asr.is_available if sim._asr else False}")

            if not sim._asr.is_available:
                yield f"data: {{\"error\": \"ASR model not loaded\"}}\n\n"
                active_sessions.pop(bot.session_id, None)
                logger.error(f"[SSE:{conn_id}] ASR not available, closing")
                return

            # 发送初始问候 (不等待TTS)
            first_msg, _ = await bot.process(use_tts=False)
            logger.debug(f"[SSE:{conn_id}] Greeting text: {first_msg[:80]}...")


            init_data = {
                "type": "greeting",
                "session_id": bot.session_id,
                "agent_text": first_msg,
                "state": sim._state_to_stage(bot.state),
            }
            yield f"data: {json.dumps(init_data, ensure_ascii=False)}\n\n"

            # 后台合成问候音频（心跳循环防止长时间等待时断开）
            if first_msg:
                try:
                    tts_task = aio.create_task(
                        sim._tts.synthesize(first_msg, voice=sim.agent_voice, engine=sim.agent_tts_engine)
                    )
                    logger.debug(f"[SSE:{conn_id}] Greeting TTS started, waiting with heartbeat...")
                    tts_start = asyncio.get_event_loop().time()
                    heartbeat_count = 0
                    while not tts_task.done():
                        try:
                            await aio.wait_for(aio.shield(tts_task), timeout=4.0)
                        except aio.TimeoutError:
                            heartbeat_count += 1
                            logger.debug(f"[SSE:{conn_id}] Greeting TTS heartbeat #{heartbeat_count} (elapsed {asyncio.get_event_loop().time() - tts_start:.1f}s)")
                            yield f": heartbeat\n\n"
                    tts_elapsed = asyncio.get_event_loop().time() - tts_start
                    tts_result = tts_task.result()
                    if tts_result.success and tts_result.audio_file:
                        logger.info(f"[SSE:{conn_id}] Greeting TTS done in {tts_elapsed:.1f}s audio={Path(tts_result.audio_file).name}")
                        audio_data = {"type": "greeting_audio", "agent_audio_url": f"/audio/{Path(tts_result.audio_file).name}"}
                        yield f"data: {json.dumps(audio_data, ensure_ascii=False)}\n\n"
                    else:
                        logger.warning(f"[SSE:{conn_id}] Greeting TTS failed: success={tts_result.success}")
                except Exception as e:
                    logger.warning(f"[SSE:{conn_id}] Greeting TTS exception: {e}")

            # 流式仿真循环 — 生产者-消费者模式，在长时间等待时插入心跳注释
            logger.info(f"[SSE:{conn_id}] Starting simulation loop, max_turns={max_turns}")
            turn_queue: aio.Queue = aio.Queue()
            producer_done = False
            producer_error = None

            async def produce_turns():
                nonlocal producer_error
                try:
                    async for turn in sim.run_streaming(max_turns=max_turns):
                        await turn_queue.put(('turn', turn))
                    await turn_queue.put(('done', None))
                except Exception as e:
                    producer_error = str(e)
                    await turn_queue.put(('error', e))

            producer = aio.create_task(produce_turns())
            turn_count = 0

            try:
                while True:
                    try:
                        msg_type, payload = await aio.wait_for(
                            turn_queue.get(), timeout=5.0
                        )
                    except aio.TimeoutError:
                        yield f": heartbeat\n\n"
                        continue

                    if msg_type == 'done':
                        logger.info(f"[SSE:{conn_id}] Simulation loop done, {turn_count} turns")
                        break
                    if msg_type == 'error':
                        logger.error(f"[SSE:{conn_id}] Producer error: {payload}")
                        yield f"data: {{\"type\": \"error\", \"message\": \"{str(payload)}\"}}\n\n"
                        return

                    turn = payload
                    turn_count += 1
                    logger.debug(f"[SSE:{conn_id}] Turn {turn_count}: {turn.state_before} -> {turn.state_after} (finished={bot.is_finished()})")
                    turn_data = {
                        "type": "turn",
                        "session_id": bot.session_id,
                        "turn_id": turn.turn_id,
                        "state_before": turn.state_before,
                        "state_after": turn.state_after,
                        "customer_text": turn.customer_text,
                        "customer_audio_url": (
                            f"/audio/{Path(turn.customer_audio_file).name}"
                            if turn.customer_audio_file else None
                        ),
                        "asr_text": turn.asr_text,
                        "asr_exact_match": turn.asr_exact_match,
                        "asr_cer": round(turn.asr_cer, 4),
                        "agent_text": turn.agent_text,
                        "agent_audio_url": (
                            f"/audio/{Path(turn.agent_audio_file).name}"
                            if turn.agent_audio_file else None
                        ),
                        "is_finished": bot.is_finished(),
                    }
                    import shutil
                    # 复制音频文件到 tts_output 以便 /audio/ 端点可访问
                    for key in ("customer_audio_file", "agent_audio_file"):
                        audio_path = turn_data.get(
                            "customer_audio_url" if key == "customer_audio_file" else "agent_audio_url"
                        )
                        if audio_path:
                            src = getattr(turn, key)
                            if src and Path(src).exists():
                                dst = _PROJECT_ROOT / "data/runs/tts_output" / Path(src).name
                                if not dst.exists():
                                    shutil.copy2(src, dst)

                    yield f"data: {json.dumps(turn_data, ensure_ascii=False)}\n\n"
            finally:
                producer.cancel()
                try:
                    await producer
                except aio.CancelledError:
                    pass

            # 发送完成事件
            from core.chatbot import ChatState
            report = sim.get_report()
            logger.info(f"[SSE:{conn_id}] Simulation complete — turns={report.total_turns} state={report.final_state} committed_time={report.committed_time}")
            done_data = {
                "type": "done",
                "session_id": bot.session_id,
                "total_turns": report.total_turns,
                "final_state": report.final_state,
                "committed_time": report.committed_time,
                "asr_exact_match_rate": round(report.asr_exact_match_rate, 4),
                "avg_cer": round(report.avg_cer, 4),
                "avg_tts_time": round(report.avg_tts_time, 3),
                "avg_asr_time": round(report.avg_asr_time, 3),
                "conversation_ended": report.conversation_ended,
            }
            yield f"data: {json.dumps(done_data, ensure_ascii=False)}\n\n"

            # 保存自动仿真会话到数据库
            try:
                from api.database import SessionLocal
                db = SessionLocal()
                save_session_to_db(db, bot, chat_group)
                db.close()
                logger.debug(f"[SSE:{conn_id}] Session saved to database")
            except Exception as e:
                logger.warning(f"[SSE:{conn_id}] DB save failed: {e}")

        except Exception as e:
            logger.error(f"[SSE:{conn_id}] Fatal error: {e}", exc_info=True)
            yield f"data: {{\"type\": \"error\", \"message\": \"{str(e)}\"}}\n\n"
        finally:
            if bot is not None:
                active_sessions.pop(bot.session_id, None)
                logger.debug(f"[SSE:{conn_id}] Bot removed from active_sessions, connection closed")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
