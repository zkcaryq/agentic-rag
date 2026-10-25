"""供应商 HTTP 调用的共用小工具（带重试、鉴权失败不重试、错误信息不泄露凭据）。"""
import time

import requests


def post_json(
    url: str,
    api_key: str,
    payload: dict,
    *,
    vendor: str,
    timeout: int = 60,
    retries: int = 2,
) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error = ""
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            last_error = f"网络异常 {exc.__class__.__name__}"
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(f"{vendor} 请求失败：{last_error}") from exc

        if resp.status_code == 200:
            return resp.json()

        # 响应体截断后再带进异常，避免日志里出现超长内容
        last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
        if resp.status_code in (401, 403):
            raise RuntimeError(f"{vendor} 鉴权失败：{last_error}")
        if (resp.status_code == 429 or resp.status_code >= 500) and attempt < retries:
            time.sleep(1.5 * (attempt + 1))
            continue
        raise RuntimeError(f"{vendor} 请求失败：{last_error}")

    raise RuntimeError(f"{vendor} 请求失败：{last_error}")
