from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from pathlib import Path
import os


_DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "collection.db"
_DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
DATABASE_URL = os.getenv("DB_URI", f"sqlite:///{_DEFAULT_DB_PATH}")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    chat_group = Column(String)
    customer_name = Column(String, nullable=True)
    customer_phone = Column(String, nullable=True)
    is_finished = Column(Boolean, default=False)
    is_successful = Column(Boolean, default=False)
    commit_time = Column(String, nullable=True)
    conversation_length = Column(Integer, default=0)
    start_time = Column(String, default=lambda: datetime.now().isoformat())
    end_time = Column(String, nullable=True)
    created_at = Column(String, default=lambda: datetime.now().isoformat())

    turns = relationship("ChatTurn", back_populates="session", cascade="all, delete-orphan")


class ChatTurn(Base):
    __tablename__ = "chat_turns"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"))
    turn_number = Column(Integer)
    agent_text = Column(Text)
    customer_text = Column(Text, nullable=True)
    state = Column(String)
    timestamp = Column(String, default=lambda: datetime.now().isoformat())
    latency_ms = Column(Integer, nullable=True)

    session = relationship("ChatSession", back_populates="turns")


class ScriptLibrary(Base):
    __tablename__ = "script_library"

    id = Column(Integer, primary_key=True, index=True)
    category = Column(String, index=True)
    chat_group = Column(String, index=True)
    script_key = Column(String)
    script_text = Column(Text)
    variables = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(String, default=lambda: datetime.now().isoformat())
    updated_at = Column(String, default=lambda: datetime.now().isoformat())


class TestScenario(Base):
    __tablename__ = "test_scenarios"

    id = Column(Integer, primary_key=True, index=True)
    scenario_name = Column(String)
    chat_group = Column(String)
    persona = Column(String)
    description = Column(Text, nullable=True)
    expected_success = Column(Boolean, default=True)
    num_runs = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    created_at = Column(String, default=lambda: datetime.now().isoformat())


class TestResult(Base):
    __tablename__ = "test_results"

    id = Column(Integer, primary_key=True, index=True)
    scenario_id = Column(Integer, ForeignKey("test_scenarios.id"))
    session_id = Column(String)
    is_successful = Column(Boolean)
    commit_time = Column(String, nullable=True)
    conversation_length = Column(Integer)
    conversation_log = Column(JSON, nullable=True)
    created_at = Column(String, default=lambda: datetime.now().isoformat())


class MetricLog(Base):
    __tablename__ = "metric_logs"

    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String, index=True)
    metric_value = Column(String)
    metric_type = Column(String)
    chat_group = Column(String, nullable=True)
    session_id = Column(String, nullable=True)
    created_at = Column(String, default=lambda: datetime.now().isoformat())


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_script_library(db):
    scripts = [
        {
            "category": "greeting",
            "chat_group": "H2",
            "script_key": "greeting_1",
            "script_text": "Halo?",
        },
        {
            "category": "greeting",
            "chat_group": "H2",
            "script_key": "greeting_2",
            "script_text": "Halo.",
        },
        {
            "category": "greeting_response",
            "chat_group": "H2",
            "script_key": "greeting_response_1",
            "script_text": "Halo, selamat pagi {name}.",
            "variables": ["name"],
        },
        {
            "category": "greeting_response",
            "chat_group": "H2",
            "script_key": "greeting_response_2",
            "script_text": "Halo, selamat siang {name}.",
            "variables": ["name"],
        },
        {
            "category": "identify",
            "chat_group": "H2",
            "script_key": "identify_1",
            "script_text": "Saya dari aplikasi Extra.",
        },
        {
            "category": "purpose",
            "chat_group": "H2",
            "script_key": "purpose_1",
            "script_text": "Untuk pinjaman ya {name}.",
            "variables": ["name"],
        },
        {
            "category": "ask_time",
            "chat_group": "H2",
            "script_key": "ask_time_1",
            "script_text": "Kapan bisa bayar {name}?",
            "variables": ["name"],
        },
        {
            "category": "ask_time",
            "chat_group": "H2",
            "script_key": "ask_time_2",
            "script_text": "Jam berapa ya?",
        },
        {
            "category": "push",
            "chat_group": "H2",
            "script_key": "push_1",
            "script_text": "Jam berapa tepatnya?",
        },
        {
            "category": "push",
            "chat_group": "H2",
            "script_key": "push_2",
            "script_text": "Hari ini jam berapa ya?",
        },
        {
            "category": "commit_time",
            "chat_group": "H2",
            "script_key": "commit_time_1",
            "script_text": "Oke, {time} ya {name}.",
            "variables": ["time", "name"],
        },
        {
            "category": "commit_time",
            "chat_group": "H2",
            "script_key": "commit_time_2",
            "script_text": "Ya, ya, ya. {time} ya {name}.",
            "variables": ["time", "name"],
        },
        {
            "category": "confirm",
            "chat_group": "H2",
            "script_key": "confirm_1",
            "script_text": "Ya, ya, ya.",
        },
        {
            "category": "confirm",
            "chat_group": "H2",
            "script_key": "confirm_2",
            "script_text": "Iya.",
        },
        {
            "category": "confirm",
            "chat_group": "H2",
            "script_key": "confirm_3",
            "script_text": "Baik.",
        },
        {
            "category": "wait",
            "chat_group": "H2",
            "script_key": "wait_1",
            "script_text": "Saya tunggu ya.",
        },
        {
            "category": "wait",
            "chat_group": "H2",
            "script_key": "wait_2",
            "script_text": "Saya tunggu {time}.",
            "variables": ["time"],
        },
        {
            "category": "closing",
            "chat_group": "H2",
            "script_key": "closing_1",
            "script_text": "Terima kasih.",
        },
        {
            "category": "closing",
            "chat_group": "H2",
            "script_key": "closing_2",
            "script_text": "Terima kasih. Selamat pagi.",
        },
    ]

    for script in scripts:
        existing = db.query(ScriptLibrary).filter(
            ScriptLibrary.category == script["category"],
            ScriptLibrary.chat_group == script["chat_group"],
            ScriptLibrary.script_key == script["script_key"],
        ).first()

        if not existing:
            db_script = ScriptLibrary(
                category=script["category"],
                chat_group=script["chat_group"],
                script_key=script["script_key"],
                script_text=script["script_text"],
                variables=script.get("variables"),
                is_active=True,
            )
            db.add(db_script)

    db.commit()


if __name__ == "__main__":
    init_db()
    print("Database initialized!")

    db = next(get_db())
    init_script_library(db)
    print("Script library initialized!")
