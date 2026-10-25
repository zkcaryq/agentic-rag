"""阿里云 OSS 客户端（替换原 MinIO）。

bucket 保持私有读：正文里只存 storage_key，前端取图走 /media/{storage_key}，
由服务端在访问时生成带有效期的签名 URL，避免把内部文档图片暴露成公共读。
"""
import hashlib
import re
from pathlib import Path
from typing import Optional

from app.conf.oss_config import oss_config
from app.core.logger import logger

_bucket = None


def is_enabled() -> bool:
    return bool(
        oss_config.enabled
        and oss_config.endpoint
        and oss_config.bucket
        and oss_config.access_key_id
        and oss_config.access_key_secret
    )


def get_bucket():
    """OSS Bucket 单例；未启用或依赖缺失时返回 None，由调用方降级。"""
    global _bucket
    if not is_enabled():
        return None
    if _bucket is None:
        try:
            import oss2

            auth = oss2.Auth(oss_config.access_key_id, oss_config.access_key_secret)
            _bucket = oss2.Bucket(auth, oss_config.endpoint, oss_config.bucket)
            logger.info(f"OSS 客户端初始化成功 bucket={oss_config.bucket}")
        except Exception as e:
            logger.error(f"OSS 客户端初始化失败: {e}")
            return None
    return _bucket


def build_storage_key(file_title: str, filename: str, sub_dir: str = "images") -> str:
    """生成稳定且不冲突的 storage_key。

    用文件名的哈希做前缀，避免同一文档里同名图片互相覆盖。
    """
    safe_title = re.sub(r"[^\w\-.]+", "_", file_title or "unknown")[:60]
    digest = hashlib.md5(filename.encode("utf-8")).hexdigest()[:10]
    safe_name = re.sub(r"[^\w\-.]+", "_", Path(filename).name)[:80]
    return f"{oss_config.upload_prefix}/{safe_title}/{sub_dir}/{digest}_{safe_name}"


def upload_bytes(data: bytes, storage_key: str) -> Optional[str]:
    """上传字节内容，成功返回 storage_key。"""
    bucket = get_bucket()
    if bucket is None:
        logger.warning(f"OSS 未启用，跳过上传: {storage_key}")
        return None
    try:
        bucket.put_object(storage_key, data)
        return storage_key
    except Exception as e:
        logger.error(f"OSS 上传失败 {storage_key}: {e}")
        return None


def upload_file(local_path: str, storage_key: str) -> Optional[str]:
    """上传本地文件，成功返回 storage_key。"""
    bucket = get_bucket()
    if bucket is None:
        logger.warning(f"OSS 未启用，跳过上传: {storage_key}")
        return None
    try:
        bucket.put_object_from_file(storage_key, local_path)
        return storage_key
    except Exception as e:
        logger.error(f"OSS 上传失败 {storage_key}: {e}")
        return None


def sign_url(storage_key: str, ttl_seconds: Optional[int] = None) -> Optional[str]:
    """生成临时可访问的签名 URL；对象不存在或无权限时返回 None。"""
    bucket = get_bucket()
    if bucket is None:
        return None
    try:
        if not bucket.object_exists(storage_key):
            logger.warning(f"OSS 对象不存在: {storage_key}")
            return None
        ttl = ttl_seconds or oss_config.signed_url_ttl_seconds
        return bucket.sign_url("GET", storage_key, ttl, slash_safe=True)
    except Exception as e:
        logger.error(f"OSS 签名失败 {storage_key}: {e}")
        return None


def object_exists(storage_key: str) -> bool:
    bucket = get_bucket()
    if bucket is None:
        return False
    try:
        return bool(bucket.object_exists(storage_key))
    except Exception:
        return False
