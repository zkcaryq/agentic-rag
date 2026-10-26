#!/usr/bin/env python
"""并发压测：流式问答的客户端口径 TTFT / 端到端延迟。

用法：
    # 先起服务：.venv/Scripts/python.exe scripts/demo_local.py --serve
    .venv/Scripts/python.exe scripts/bench_concurrency.py --concurrency 4 --total 20
    .venv/Scripts/python.exe scripts/bench_concurrency.py \
        --base-url http://127.0.0.1:8001 --concurrency 16 --duration 120 \
        --question "H3C 路由器如何配置 VLAN" --json output/bench/$(date +%s).json

指标口径：
  - ttft_ms   = 发出 POST /query → 收到第一个 delta 事件（含排队、检索、重排、生成）
  - total_ms  = 发出 POST /query → 收到 final 事件
  客户端计时是"用户感知的延迟"，和服务端 /metrics 里的 answer_generate（纯模型首 Token）
  不是同一件事，两个都要看才知道慢在排队还是在模型。

依赖只有 requests 和线程，不需要装 locust；并发上限受服务端线程池
（query_service 的 max_workers=4）和向量库连接限制。
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:  # Windows 控制台默认 GBK，中文输出会花屏
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app.core.metrics import _percentile  # 与服务端 /metrics 用同一套插值算法

DEFAULT_QUESTION = "打印机如何更换硒鼓？"

_stop = threading.Event()
_lock = threading.Lock()
_samples: list[dict] = []
_failures: dict[str, int] = {}


def _fail(reason: str) -> bool:
    """记一次失败并返回 False，方便调用方 `return _fail(...)`。"""
    with _lock:
        _failures[reason] = _failures.get(reason, 0) + 1
    return False


def run_one_round(base_url: str, question: str, timeout: float) -> bool:
    """一轮完整问答：提交 → 连流 → 读到 final。返回是否成功。"""
    session_id = str(uuid.uuid4())
    t0 = time.perf_counter()
    try:
        submit = requests.post(
            f"{base_url}/query",
            json={"query": question, "session_id": session_id, "is_stream": True},
            timeout=timeout,
        )
        if submit.status_code != 200:
            return _fail(f"POST /query -> {submit.status_code}")
    except requests.RequestException as e:
        return _fail(f"提交失败: {type(e).__name__}")

    ttft_ms: float | None = None
    deltas = 0
    got_final = False
    answer_len = 0
    try:
        with requests.get(
            f"{base_url}/stream/{session_id}", stream=True, timeout=timeout
        ) as stream:
            if stream.status_code != 200:
                return _fail(f"GET /stream -> {stream.status_code}")
            event = None
            for line in stream.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("event:"):
                    event = line[6:].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                try:
                    payload = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    return _fail("SSE data 解析失败")
                elapsed = (time.perf_counter() - t0) * 1000
                if event == "delta":
                    if ttft_ms is None:
                        ttft_ms = elapsed
                    deltas += 1
                elif event == "final":
                    got_final = True
                    answer_len = len(payload.get("answer") or "")
                    total_ms = elapsed
                    break
                elif event == "error":
                    return _fail("服务端返回 error 事件")
                elif event == "__close__":
                    break
                if time.perf_counter() - t0 > timeout:
                    return _fail("读流超时")
    except requests.RequestException as e:
        return _fail(f"读流中断: {type(e).__name__}")

    if not got_final:
        return _fail("流结束但没有 final 事件")

    with _lock:
        _samples.append(
            {
                "session_id": session_id,
                "ttft_ms": round(ttft_ms if ttft_ms is not None else total_ms, 1),
                "total_ms": round(total_ms, 1),
                "deltas": deltas,
                "answer_len": answer_len,
            }
        )
    return True


def worker_loop(base_url: str, question: str, timeout: float, counter: "Counter") -> None:
    while not _stop.is_set() and counter.take():
        run_one_round(base_url, question, timeout)
        if _stop.is_set():
            break


class Counter:
    """轮数配额：给了 --total 就限次，否则无限（由 --duration 到点叫停）。"""

    def __init__(self, limit: int | None):
        self._limit = limit
        self._taken = 0
        self._lock = threading.Lock()

    def take(self) -> bool:
        if self._limit is None:
            return True
        with self._lock:
            if self._taken >= self._limit:
                return False
            self._taken += 1
            return True


def report(base_url: str, elapsed: float, timeout: float) -> int:
    total = len(_samples)
    attempts = total + sum(_failures.values())
    print("\n" + "=" * 70)
    print(f"并发压测结果（{elapsed:.1f}s，请求 {attempts} 轮，成功 {total} 轮）")
    print("=" * 70)
    if not total:
        print("没有成功样本，先看服务是否在跑：scripts/demo_local.py --serve")
        for reason, count in sorted(_failures.items(), key=lambda kv: -kv[1]):
            print(f"  {count:>4} x {reason}")
        return 1

    ttfts = sorted(s["ttft_ms"] for s in _samples)
    totals = sorted(s["total_ms"] for s in _samples)
    print(f"  成功率: {total / attempts * 100:.1f}%  吞吐: {total / elapsed:.2f} 轮/秒")
    print(f"  TTFT   P50/P95/P99: {_percentile(ttfts, .5):.0f} / "
          f"{_percentile(ttfts, .95):.0f} / {_percentile(ttfts, .99):.0f} ms")
    print(f"  端到端 P50/P95/P99: {_percentile(totals, .5):.0f} / "
          f"{_percentile(totals, .95):.0f} / {_percentile(totals, .99):.0f} ms")
    print(f"  平均增量包: {sum(s['deltas'] for s in _samples) / total:.1f}  "
          f"平均答案长度: {sum(s['answer_len'] for s in _samples) / total:.0f} 字")
    if _failures:
        print("  失败分布:")
        for reason, count in sorted(_failures.items(), key=lambda kv: -kv[1]):
            print(f"    {count:>4} x {reason}")

    try:
        snap = requests.get(f"{base_url}/metrics", timeout=timeout).json()
        print(f"\n  服务端 /metrics（dispatch={snap.get('query_dispatch')}）:")
        for ep, values in (snap.get("endpoints") or {}).items():
            print(
                f"    {ep}: count={values['count']} "
                f"ttft(n={values['ttft_count']}) p50={values['ttft_p50']} "
                f"p95={values['ttft_p95']} total_p95={values['total_p95']}"
            )
        if not (snap.get("endpoints") or {}):
            print("    （窗口为空：本轮请求都派发到了 worker 进程，或还没跑过查询）")
    except Exception as e:
        print(f"\n  读取 /metrics 失败: {e}")

    print("=" * 70)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="流式问答并发压测")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--total", type=int, help="总轮数，不给则按时长跑")
    parser.add_argument("--duration", type=float, default=30.0, help="秒，--total 未指定时生效")
    parser.add_argument("--timeout", type=float, default=60.0, help="单轮读流上限（秒）")
    parser.add_argument("--json", help="把样本写到该 JSON 文件")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    counter = Counter(args.total)

    threads = [
        threading.Thread(
            target=worker_loop,
            args=(base_url, args.question, args.timeout, counter),
            daemon=True,
        )
        for _ in range(max(1, args.concurrency))
    ]
    started = time.perf_counter()
    print(
        f"压测 {base_url}  并发={len(threads)}  "
        f"{'轮数=' + str(args.total) if args.total else '时长=' + str(args.duration) + 's'}"
    )
    print(f"问题：{args.question}")
    for t in threads:
        t.start()

    try:
        if args.total:
            while any(t.is_alive() for t in threads):
                time.sleep(0.5)
        else:
            time.sleep(max(1.0, args.duration))
    except KeyboardInterrupt:
        print("\n收到中断，正在收尾…")
    finally:
        _stop.set()
        for t in threads:
            t.join(timeout=args.timeout + 5)

    elapsed = time.perf_counter() - started

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "base_url": base_url,
                    "question": args.question,
                    "concurrency": len(threads),
                    "elapsed_seconds": round(elapsed, 2),
                    "samples": _samples,
                    "failures": _failures,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"样本已写入 {out}")

    return report(base_url, elapsed, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
