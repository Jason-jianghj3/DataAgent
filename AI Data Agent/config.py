import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# 加载 .env 文件到环境变量
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv 未安装时静默跳过，环境变量需手动设置


@dataclass
class FineReportConfig:
    base_url: str = os.getenv("FINEREPORT_BASE_URL", "http://localhost:8080/webroot/decision")
    username: str = os.getenv("FINEREPORT_USERNAME", "admin")
    password: str = os.getenv("FINEREPORT_PASSWORD", "")
    report_path: str = os.getenv("FINEREPORT_REPORT_PATH", "/检验数据报表.cpt")
    timeout: int = 30


@dataclass
class LLMConfig:
    provider: str = os.getenv("LLM_PROVIDER", "openai")
    api_key: str = os.getenv("LLM_API_KEY", "")
    api_base: str = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
    model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1000"))
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    fallback_api_key: str = os.getenv("LLM_FALLBACK_API_KEY", "")
    fallback_api_base: str = os.getenv("LLM_FALLBACK_API_BASE", "")
    fallback_model: str = os.getenv("LLM_FALLBACK_MODEL", "")


@dataclass
class TTSConfig:
    provider: str = os.getenv("TTS_PROVIDER", "azure")
    api_key: str = os.getenv("TTS_API_KEY", "")
    region: str = os.getenv("TTS_REGION", "eastasia")
    voice_name: str = os.getenv("TTS_VOICE_NAME", "zh-CN-XiaoxiaoNeural")
    output_dir: str = os.getenv("TTS_OUTPUT_DIR", "./audio_output")


@dataclass
class NotificationConfig:
    enable_wechat: bool = os.getenv("ENABLE_WECHAT", "false").lower() == "true"
    enable_email: bool = os.getenv("ENABLE_EMAIL", "false").lower() == "true"
    enable_webhook: bool = os.getenv("ENABLE_WEBHOOK", "false").lower() == "true"
    webhook_url: str = os.getenv("WEBHOOK_URL", "")
    smtp_server: str = os.getenv("SMTP_SERVER", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_username: str = os.getenv("SMTP_USERNAME", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    email_recipients: list = None

    def __post_init__(self):
        if self.email_recipients is None:
            self.email_recipients = []


@dataclass
class FlaskConfig:
    host: str = os.getenv("FLASK_HOST", "0.0.0.0")
    port: int = int(os.getenv("FLASK_PORT", "5000"))
    debug: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"


@dataclass
class AppConfig:
    fine_report: FineReportConfig = None
    llm: LLMConfig = None
    tts: TTSConfig = None
    notification: NotificationConfig = None
    flask: FlaskConfig = None

    def __post_init__(self):
        if self.fine_report is None:
            self.fine_report = FineReportConfig()
        if self.llm is None:
            self.llm = LLMConfig()
        if self.tts is None:
            self.tts = TTSConfig()
        if self.notification is None:
            self.notification = NotificationConfig()
        if self.flask is None:
            self.flask = FlaskConfig()


config = AppConfig()
