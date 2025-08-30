#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
wikipedia.py
提供 fetch_and_parse(url: str) -> dict
- 使用 wikipedia-api 库抓取
- 提取 title / sections / full_text / summary
"""

from __future__ import annotations
from typing import Dict, Tuple
import wikipediaapi
from urllib.parse import urlparse, unquote
from llm_summarizer import summarize_and_log as llm_summary


def _url_to_title_lang(url: str) -> Tuple[str, str]:
    # e.g. https://en.wikipedia.org/wiki/Currency -> ("Currency", "en")
    p = urlparse(url)
    lang = p.netloc.split(".")[0] if p.netloc.endswith("wikipedia.org") else "en"
    title = unquote(p.path.split("/wiki/")[-1]) if "/wiki/" in p.path else ""
    return title, (lang or "en")

def fetch_and_parse(url: str) -> Dict:
    title, lang = _url_to_title_lang(url)
    if not title:
        return {"title": "", "sections": [], "full_text": "", "summary": "[ERROR] invalid wiki url"}

    wiki = wikipediaapi.Wikipedia(
        language=lang,
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent="WikiBatchExtractor/1.0 (+https://example.org/bot)"
    )
    page = wiki.page(title)
    if not page.exists():
        return {"title": "", "sections": [], "full_text": "", "summary": "[ERROR] page not found"}

    def _flatten(secs, level=2, max_level=4):
        out = []
        for s in secs:
            if level <= max_level and s.text:
                out.append({"heading": s.title or "", "level": level, "text": s.text})
            if s.sections:
                out.extend(_flatten(s.sections, level=level+1, max_level=max_level))
        return out

    sections = _flatten(page.sections, level=2, max_level=4)
    lead = page.summary or ""
    full_text = " ".join([lead] + [s["text"] for s in sections if s.get("text")]).strip()
    return {
        "title": page.title or title,
        "sections": sections,
        "full_text": full_text,
        "summary": llm_summary(full_text, title=page.title or title)
    }


if __name__ == "__main__":
    u = "https://en.wikipedia.org/wiki/Currency"
    result = fetch_and_parse(u)

    print("\n=== Title ===")
    print(result["title"])

    print("\n=== Summary ===")
    print(result["summary"])
    #
    # print("\n=== Sections (前3个) ===")
    # for sec in result["sections"][:3]:
    #     print(f"- {sec['heading']} ({len(sec['text'])} chars)")
    #
    # print("\n=== Full text (前500字) ===")
    # print(result["full_text"][:500], "...")
