from pydantic import BaseModel
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime


class ChatState(str, Enum):
    INIT = "init"
    GREETING = "greeting"
    IDENTIFY = "identify"
    PURPOSE = "purpose"
    ASK_TIME = "ask_time"
    PUSH_FOR_TIME = "push_for_time"
    COMMIT_TIME = "commit_time"
    CONFIRM = "confirm"
    NEGOTIATE = "negotiate"
    CLOSE = "close"
    FAILED = "failed"


class ChatGroup(str, Enum):
    H2 = "H2"
    H1 = "H1"
    S0 = "S0"


class CustomerPersona(str, Enum):
    COOPERATIVE = "cooperative"
    BUSY = "busy"
    NEGOTIATING = "negotiating"
    RESISTANT = "resistant"
    SILENT = "silent"
    FORGETFUL = "forgetful"
    EXCUSE_MASTER = "excuse_master"


class ChatTurnRequest(BaseModel):
    session_id: Optional[str] = None
    chat_group: ChatGroup = ChatGroup.H2
    customer_name: Optional[str] = "Pak/Bu"
    customer_input: Optional[str] = None
    customer_phone: Optional[str] = None  # P15-D01: 跨会话用户标识


class ChatTurnResponse(BaseModel):
    session_id: str
    agent_response: str
    current_state: ChatState
    commit_time: Optional[str] = None
    conversation_length: int
    is_finished: bool
    is_successful: bool
    audio_file: Optional[str] = None
    latency_ms: Optional[float] = None
    llm_used: bool = False


class ChatLogEntry(BaseModel):
    role: str
    text: str
    timestamp: str


class ChatSessionResponse(BaseModel):
    session_id: str
    chat_group: ChatGroup
    customer_name: Optional[str]
    is_finished: bool
    is_successful: bool
    commit_time: Optional[str]
    conversation_length: int
    conversation_log: List[ChatLogEntry]
    start_time: str
    end_time: Optional[str] = None
    created_at: str


class TestScenarioRequest(BaseModel):
    chat_group: ChatGroup
    persona: CustomerPersona
    num_tests: int = 10


class TestResultResponse(BaseModel):
    total_tests: int
    success_count: int
    failed_count: int
    success_rate: float
    results: List[Dict[str, Any]]


class MessageResponse(BaseModel):
    message: str
    success: bool = True


class HealthResponse(BaseModel):
    status: str       # healthy | degraded | unhealthy
    version: str
    timestamp: str
    checks: Dict[str, str] = {}  # component_name → status


class StatsResponse(BaseModel):
    """统计数据响应"""
    total_sessions: int
    successful_sessions: int
    success_rate: float
    total_turns: int
    avg_turns_per_session: float
    active_sessions: int
    chat_group_stats: Dict[str, Dict[str, int]]


class ScriptResponse(BaseModel):
    """脚本库响应"""
    id: int
    category: str
    chat_group: str
    script_key: str
    script_text: str
    variables: Optional[List[str]] = None
    is_active: bool


class ScriptUpdateRequest(BaseModel):
    """脚本更新请求"""
    script_text: Optional[str] = None
    is_active: Optional[bool] = None
    variables: Optional[List[str]] = None


class TranslateRequest(BaseModel):
    """翻译请求"""
    text: str
    source: Optional[str] = "id"
    target: Optional[str] = "en"


class TranslateResponse(BaseModel):
    """翻译响应"""
    original_text: str
    translated_text: str
    source: str
    target: str
    success: bool


class SimulateCustomerRequest(BaseModel):
    """仿真客户请求"""
    session_id: str
    persona: CustomerPersona = CustomerPersona.COOPERATIVE
    resistance_level: Optional[str] = "medium"


class SimulateCustomerResponse(BaseModel):
    """仿真客户响应"""
    customer_response: str
    persona: str
    resistance_level: str
    success: bool


class SessionSummary(BaseModel):
    """会话摘要（用于左侧列表）"""
    session_id: str
    chat_group: ChatGroup
    customer_name: Optional[str]
    is_finished: bool
    is_successful: bool
    state: Optional[str] = None
    conversation_length: int
    start_time: str
    end_time: Optional[str] = None


class SessionListResponse(BaseModel):
    """会话列表响应"""
    active: List[SessionSummary]
    completed: List[SessionSummary]


class VoiceStartRequest(BaseModel):
    """语音会话启动请求"""
    chat_group: ChatGroup = ChatGroup.H1
    customer_name: str = "Budi"


class VoiceTurnRequest(BaseModel):
    """语音会话轮次请求"""
    session_id: str
    customer_input: Optional[str] = None


class VoiceSessionResponse(BaseModel):
    """语音会话响应"""
    session_id: str
    agent_text: str
    audio_data_base64: Optional[str] = None
    audio_file: Optional[str] = None
    state: str
    is_finished: bool
    is_successful: bool


class ASRResponse(BaseModel):
    """ASR 转写响应"""
    text: str
    success: bool
    error: Optional[str] = None
