import os
from dataclasses import dataclass
from typing import Dict, Optional

try:
    from dotenv import load_dotenv
    from pathlib import Path
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=True)
except ImportError:
    pass


@dataclass
class DatabaseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    charset: str = 'utf8'
    timeout: int = 30
    login_timeout: int = 15

    def to_dict(self) -> Dict:
        return {
            'server': self.host,
            'port': self.port,
            'user': self.user,
            'password': self.password,
            'database': self.database,
            'charset': self.charset,
            'timeout': self.timeout,
            'login_timeout': self.login_timeout,
        }

    def to_pymssql_kwargs(self) -> Dict:
        return self.to_dict()


def get_db_config(connection_name: str) -> Optional[DatabaseConfig]:
    prefix = f"DB_{connection_name.upper()}_"

    host = os.getenv(f"{prefix}HOST")
    if not host:
        return None

    return DatabaseConfig(
        host=host,
        port=int(os.getenv(f"{prefix}PORT", "1433")),
        user=os.getenv(f"{prefix}USER", ""),
        password=os.getenv(f"{prefix}PASSWORD", ""),
        database=os.getenv(f"{prefix}DATABASE", connection_name.upper()),
    )


def get_all_db_configs() -> Dict[str, DatabaseConfig]:
    configs = {}
    for name in ['EAM', 'WMS_PROD', 'EKP', 'HISTORIAN', 'DW']:
        cfg = get_db_config(name)
        if cfg and cfg.user and cfg.password:
            configs[name] = cfg
    return configs


EAM_CONFIG = get_db_config('EAM')
WMS_CONFIG = get_db_config('WMS_PROD')
EKP_CONFIG = get_db_config('EKP')
HISTORIAN_CONFIG = get_db_config('HISTORIAN')
DW_CONFIG = get_db_config('DW')
