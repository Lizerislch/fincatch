#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import asyncio
import random
import re
import time
from typing import Dict, List, Optional, Tuple
from llm_summarizer import summarize_and_log as llm_summary


import aiohttp
from bs4 import BeautifulSoup, Tag

# ========= 行为开关 =========
USE_REAL_HEADERS = True           # 使用带 sec-ch-ua 的“真实”浏览器头
USE_RJINA_FALLBACK = False        # 是否启用 r.jina.ai 兜底（可能 451）
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 20
TOTAL_TIMEOUT = 30
MAX_RETRIES = 4
RETRY_STATUS = {403, 429, 500, 502, 503, 504}
BACKOFF_BASE = 0.6
JITTER = (0.0, 0.3)
REQUEST_JITTER = (0.0, 0.15)

# ========= 选择器/清洗 =========
CONTENT_SELECTOR_NORMAL = "div.comp.mntl-sc-page.mntl-block.article-body-content"
CONTENT_SELECTORS_AMP = ["article", "div.mntl-sc-page", "div.mntl-block"]
CITATION_BRACKETS = re.compile(r"\s*\[(\d+|citation needed|note \d+|[a-f])\]\s*", re.I)
WHITESPACE_CLEAN = re.compile(r"\s+")

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = CITATION_BRACKETS.sub(" ", s)
    s = WHITESPACE_CLEAN.sub(" ", s).strip()
    return s

# ========= 请求头 =========
REAL_HEADERS: Dict[str, str] = {
    # 这一组来自你成功的“真实浏览器”抓包
    "Sec-Ch-Ua": '"Not.A/Brand";v="99", "Google Chrome";v="139", "Chromium";v="139"',
    "Sec-Ch-Ua-Arch": '"arm"',
    "Sec-Ch-Ua-Bitness": '"64"',
    "Sec-Ch-Ua-Full-Version": '"139.0.7258.139"',
    "Sec-Ch-Ua-Full-Version-List": '"Not.A/Brand";v="99.0.0.0", "Google Chrome";v="139.0.7258.139", "Chromium";v="139.0.7258.139"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Model": '""',
    "Sec-Ch-Ua-Platform": '"macOS"',
    "Sec-Ch-Ua-Platform-Version": '"15.2.0"',
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/139.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Connection": "keep-alive",
    "Referer": "https://www.google.com/",
}

UA_POOL = [
    # 备用的简化 UA 池（当你不想用 sec-ch-ua 时）
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/16.5 Safari/605.1.15",
]

def pick_headers() -> Dict[str, str]:
    if USE_REAL_HEADERS:
        return REAL_HEADERS
    # 简化头
    return {
        "User-Agent": random.choice(UA_POOL),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Upgrade-Insecure-Requests": "1",
        "Connection": "keep-alive",
        "Referer": "https://www.google.com/",
    }

# ========= 抓取 =========
def to_amp_url(url: str) -> Tuple[str, str]:
    amp1 = url
    if "://www.investopedia.com/" in url:
        amp1 = url.replace("://www.investopedia.com/", "://www.investopedia.com/amp/", 1)
    sep = "&" if "?" in url else "?"
    amp2 = url + f"{sep}output=amp"
    return amp1, amp2

async def fetch_once(session: aiohttp.ClientSession, url: str) -> Tuple[int, str]:
    async with session.get(url, allow_redirects=True) as resp:
        status = resp.status
        text = await resp.text(errors="ignore")
    return status, text

async def fetch_html_with_retries(url: str) -> Tuple[str, str]:
    """
    返回: (html, err_msg)
      - 成功: (html, "")
      - 失败: ("", "错误字符串")
    """
    attempt = 0
    err = ""
    while attempt <= MAX_RETRIES:
        await asyncio.sleep(random.uniform(*REQUEST_JITTER))  # 微抖动
        timeout = aiohttp.ClientTimeout(total=TOTAL_TIMEOUT, connect=CONNECT_TIMEOUT, sock_read=READ_TIMEOUT)
        connector = aiohttp.TCPConnector(limit=32, limit_per_host=8, ssl=True)
        try:
            async with aiohttp.ClientSession(headers=pick_headers(), timeout=timeout, connector=connector) as session:
                status, html = await fetch_once(session, url)
                if status == 200 and html:
                    return html, ""
                if status == 403:
                    for amp in to_amp_url(url):
                        s2, h2 = await fetch_once(session, amp)
                        if s2 == 200 and h2:
                            return h2, ""
                        if s2 in RETRY_STATUS:
                            pass
                        else:
                            return "", f"HTTP {status} & AMP {s2}"
                    err = "HTTP 403"
                elif status in RETRY_STATUS:
                    err = f"HTTP {status}"
                else:
                    return "", f"HTTP {status}"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

        attempt += 1
        if attempt > MAX_RETRIES:
            break
        backoff = BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(*JITTER)
        time.sleep(backoff)

    if USE_RJINA_FALLBACK:
        try:
            fallback_url = f"https://r.jina.ai/http://{url.replace('https://', '').replace('http://', '')}"
            timeout = aiohttp.ClientTimeout(total=TOTAL_TIMEOUT, connect=CONNECT_TIMEOUT, sock_read=READ_TIMEOUT)
            async with aiohttp.ClientSession(headers=pick_headers(), timeout=timeout) as session:
                s3, h3 = await fetch_once(session, fallback_url)
                if s3 == 200 and h3 and len(h3.strip()) > 0:
                    return h3, ""
                return "", f"fallback HTTP {s3}"
        except Exception as e:
            return "", f"fallback {type(e).__name__}: {e}"

    return "", err or "fetch failed"

# ========= 解析 =========
def get_title(page: BeautifulSoup) -> str:
    h1 = page.find("h1")
    return clean_text(h1.get_text(" ", strip=True)) if h1 else ""

def get_content_container(page: BeautifulSoup) -> Optional[Tag]:
    c = page.select_one(CONTENT_SELECTOR_NORMAL)
    if c:
        return c
    for sel in CONTENT_SELECTORS_AMP:
        c = page.select_one(sel)
        if c:
            return c
    return None

def collect_intro_text(container: Tag) -> str:
    chunks: List[str] = []
    for node in container.children:
        if isinstance(node, Tag):
            if node.name in ("h2", "h3"):
                break
            if node.name == "p":
                chunks.append(node.get_text(" ", strip=True))
    return clean_text(" ".join(chunks))

def flatten_sections(container: Tag, max_level: int = 3) -> List[Dict]:
    out: List[Dict] = []
    headers: List[Tag] = []
    for lvl in range(2, max_level + 1):
        headers.extend(container.find_all(f"h{lvl}"))

    if not headers:
        paragraphs = container.find_all("p")
        text = clean_text(" ".join(p.get_text(" ", strip=True) for p in paragraphs))
        return [{"heading": "", "level": 1, "text": text}]

    def hlevel(tag: Tag) -> int:
        try:
            return int(tag.name[1])
        except Exception:
            return 6

    def sort_key(h: Tag):
        return ((getattr(h, "sourceline", None) or 10**9), (getattr(h, "sourcepos", None) or 0))
    headers = sorted(headers, key=sort_key)

    for i, h in enumerate(headers):
        lvl = hlevel(h)
        next_cut = None
        for j in range(i + 1, len(headers)):
            if hlevel(headers[j]) <= lvl:
                next_cut = headers[j]
                break
        chunks: List[str] = []
        node = h.next_sibling
        while node and node is not next_cut:
            if isinstance(node, Tag) and node.name == "p":
                chunks.append(node.get_text(" ", strip=True))
            node = node.next_sibling
        text = clean_text(" ".join(chunks))
        if text:
            out.append({"heading": clean_text(h.get_text(" ", strip=True)), "level": lvl, "text": text})
    return out

def simple_summary(text: str, max_sentences: int = 3) -> str:
    sents = re.split(r"(?<=[。!?\.])\s+", text)
    sents = [s.strip() for s in sents if s.strip()]
    return " ".join(sents[:max_sentences])

# ========= 对外同步接口 =========
def fetch_and_parse(url: str) -> Dict:
    """
    同步函数，供 extractor.py 调用。
    内部开事件循环执行异步抓取。
    """
    try:
        html, err = asyncio.run(fetch_html_with_retries(url))
    except RuntimeError:
        # 若在已有事件循环内（某些环境），开一个新 loop
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            html, err = loop.run_until_complete(fetch_html_with_retries(url))
        finally:
            loop.close()

    if not html:
        return {"title": "", "sections": [], "full_text": "", "summary": f"[ERROR] {err or 'fetch failed'}"}

    # r.jina.ai 可能直接返回纯文本/markdown，不含 <html>
    if html.lstrip().startswith("{") or "<html" not in html.lower():
        full_text = clean_text(html)
        return {
            "title": "",
            "sections": [],
            "full_text": full_text,
            "summary": simple_summary(full_text)
        }

    page = BeautifulSoup(html, "html.parser")
    title = get_title(page)
    container = get_content_container(page)
    if container is None:
        return {"title": title, "sections": [], "full_text": "", "summary": "[ERROR] body not found"}

    intro = collect_intro_text(container)
    sections = flatten_sections(container, max_level=3)
    parts = [intro] if intro else []
    parts.extend([s["text"] for s in sections])
    full_text = clean_text(" ".join(parts))
    return {
        "title": title,
        "sections": sections,
        "full_text": full_text,
        "summary": llm_summary(full_text, title=title)
    }

# quick test
if __name__ == "__main__":
    print(fetch_and_parse("https://www.investopedia.com/terms/r/retirement-planning.asp"))
