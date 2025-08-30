#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extractor.py
------------
- 读取 Read FinCatch_Sources_Medium.csv (source, URL)
- wiki -> wikipedia.fetch_and_parse
- investopedia -> investopedia.fetch_and_parse
- 并发执行，输出:
Output:
    output.json  (成功)(Success)
    errors.json  (失败)(Fail)
"""

import os, json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict
from wikipedia import fetch_and_parse as fetch_wiki
from investopedia import fetch_and_parse as fetch_invest


# 定义一下，Similar to config
CSV_NAME = "FinCatch_Sources_Medium.csv"
OUT_NAME = "output.json"
ERR_NAME = "errors.json"
MAX_WORKERS = 8


# Determine Wiki Or Investopedia
def _route(source: str, url: str) -> Dict:
    s = (source or "").strip().lower()
    if s == "wiki" or s == "wikipedia":
        return fetch_wiki(url)
    if s == "investopedia":
        return fetch_invest(url)
    return {"title": "", "sections": [], "full_text": "", "summary": f"[ERROR] unknown source {source}"}

def main():
    if not os.path.exists(CSV_NAME):
        print(f"找不到 {CSV_NAME}")
        return
    df = pd.read_csv(CSV_NAME)
    if not {"source","URL"}.issubset(df.columns):
        raise ValueError("CSV 必须包含 'source' 和 'URL'")

    pairs = [(row["source"], row["URL"]) for _, row in df.iterrows() if pd.notna(row["URL"])]
    print(f"共 {len(pairs)} 条链接，开始并发处理…")

    results: Dict[str, Dict] = {}
    errors: Dict[str, Dict] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futmap = {ex.submit(_route, s, u): u for s, u in pairs}
        done = 0
        for fut in as_completed(futmap):
            u = futmap[fut]
            try:
                data = fut.result()
            except Exception as e:
                data = {"title":"", "sections":[], "full_text":"", "summary": f"[ERROR] {type(e).__name__}: {e}"}
            if "[ERROR]" in (data.get("summary") or ""):
                errors[u] = data
            else:
                results[u] = data
            done += 1
            if done % 5 == 0 or done == len(pairs):
                print(f"  progress: {done}/{len(pairs)}")

    with open(OUT_NAME, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(ERR_NAME, "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    print(f"完成：成功 {len(results)} 条写入 {OUT_NAME}，失败 {len(errors)} 条写入 {ERR_NAME}")

if __name__ == "__main__":
    main()
