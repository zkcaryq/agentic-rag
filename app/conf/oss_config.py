"""阿里云 OSS 配置（替换原 MinIO）。

bucket 保持私有读，正文里只存 storage_key，取图时再换签名 URL。
"""
from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class OSSConfig:
    endpoint: str
    bucket: str
    access_key_id: str
    access_key_secret: str
    enabled: bool
    signed_url_ttl_seconds: int
    upload_prefix: str


oss_config = OSSConfig(
    endpoint=os.getenv("OSS_ENDPOINT", ""),
    bucket=os.getenv("OSS_BUCKET", ""),
    access_key_id=os.getenv("OSS_ACCESS_KEY_ID", ""),
    access_key_secret=os.getenv("OSS_ACCESS_KEY_SECRET", ""),
    enabled=_env_bool("OSS_ENABLED", False),
    signed_url_ttl_seconds=int(os.getenv("OSS_SIGNED_URL_TTL_SECONDS", "3600")),
    upload_prefix=os.getenv("OSS_UPLOAD_PREFIX", "rag"),
)
