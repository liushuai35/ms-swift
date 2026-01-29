#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Qwen3 client (text-only) calling an OpenAI-compatible vLLM endpoint.

Features:
- Pass system prompts and user query from CLI
- Call /v1/chat/completions with optional API key
- Configurable model, temperature, max_tokens, timeout

Example:
  python qwen3_client.py \
    --base-url http://127.0.0.1:8000 \
    --model Qwen3-Seg-4B \
    --system "你是严格且专业的评审助手。" \
    --query "请总结以下段落的要点：......" \
    --temperature 0.2 \
    --max-tokens 512

python ms-swift/qwen3_client.py \
  --base-url http://127.0.0.1:8000 \
  --model Qwen3 \
  --system "你是严格且专业的评审助手。" \
  --query "请用三句话总结以下段落的要点：......" \
  --temperature 0.2 \
  --max-tokens 1024

Environment:
  Optionally set OPENAI_API_KEY for protected endpoints, or pass --api-key
"""

import argparse
import os
import sys
import time
from typing import List, Optional

import requests


def list_models(base_url: str, api_key: str = "", timeout: int = 15) -> list:
    url = base_url.rstrip("/") + "/v1/models"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # OpenAI-compatible: { data: [ {id: ...}, ... ] } or { object: "list", data: [...] }
        items = data.get("data", []) if isinstance(data, dict) else []
        return [str(x.get("id", "")) for x in items if isinstance(x, dict)]
    except Exception:
        return []


def build_messages(system_prompts: List[str], query: str) -> list:
    messages = []
    for sp in system_prompts:
        sp = (sp or "").strip()
        if sp:
            messages.append({"role": "system", "content": sp})
    messages.append({"role": "user", "content": query})
    return messages


def chat_once(
    base_url: str,
    model: str,
    query: str,
    system_prompts: Optional[List[str]] = None,
    api_key: str = "",
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: int = 60,
    max_retries: int = 3,
) -> str:
    system_prompts = system_prompts or []
    url = base_url.rstrip("/") + "/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    key = api_key or os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"

    payload = {
        "model": model,
        "messages": build_messages(system_prompts, query),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    last_err = None
    for attempt in range(max(1, int(max_retries))):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=int(timeout))
            resp.raise_for_status()
            data = resp.json()
            # OpenAI-style response
            content = data["choices"][0]["message"]["content"]
            return (content or "").strip()
        except requests.exceptions.HTTPError as e:
            last_err = e
            # If 404 Not Found, try to list available models to assist correction
            if e.response is not None and e.response.status_code == 404:
                available = list_models(base_url, key, timeout=10)
                hint = (
                    "[Hint] The server returned 404 for the model name. "
                    + (f"Available models: {available}" if available else "No models listed from /v1/models.")
                )
                print(hint, file=sys.stderr)
                break
            if attempt >= int(max_retries) - 1:
                break
            time.sleep(2 ** attempt)
        except Exception as e:
            last_err = e
            if attempt >= int(max_retries) - 1:
                break
            # simple backoff
            time.sleep(2 ** attempt)
    if last_err:
        raise last_err
    return ""


def main():
    parser = argparse.ArgumentParser(description="Qwen3 text client via vLLM OpenAI API")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default="", help="Optional API key (or set OPENAI_API_KEY)")
    parser.add_argument("--model", default="Qwen3", help="Model name served by vLLM")
    parser.add_argument("--system", action="append", default=["""您正在为嵌入/RAG 对文档进行分段。在连续块之间放置一行，内容为：<|segment_flag_token|> 不要添加其他内容。"""], help="System prompt (can repeat)")
    parser.add_argument("--query", default=None, help="User query text; if omitted, read from --query-file")
    parser.add_argument("--query-file", default="qwen3_query.txt", help="Path to a text file containing the query (default: qwen3_query.txt)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    # Resolve query text: prefer --query, fallback to --query-file
    query_text = args.query
    if not query_text:
        try:
            with open(args.query_file, "r", encoding="utf-8") as f:
                query_text = f.read().strip()
        except Exception as e:
            print(f"[Error] Failed to read query file: {args.query_file} ({e})", file=sys.stderr)
            sys.exit(1)

    try:
        output = chat_once(
            base_url=args.base_url,
            model=args.model,
            query=query_text,
            system_prompts=args.system,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
    except Exception as e:
        print(f"[Error] {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        sys.exit(1)

    print(output)


if __name__ == "__main__":
    main()


    """
    python qwen3_client.py \
  --base-url http://127.0.0.1:8000 \
  --model /home/shuai.liu01/merged_qwen3_4b_with_special_token \
  --system "按语义分段：请在内容独立处插入换行及“===SEGMENT===”标记。保持原文完全一致，禁止删改。" \
  --query "请用三句话总结以下段落的要点：......" \
  --temperature 0.2 \
  --max-tokens 4096
    """