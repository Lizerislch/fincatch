# q2.py
# ------------------------------------------------------------
# Q2: Visualize causal relationships of financial knowledge in Neo4j
# 输入：../Extractor/output.json  (与本文件同级的上级目录)
# 输出：把概念/因果/共现写入 Neo4j，可在 Browser 中可视化
# 依赖：pip install neo4j spacy python-dotenv
#       python -m spacy download en_core_web_sm
# ------------------------------------------------------------

import os
import re
import json
from typing import List, Tuple, Set, Dict

from neo4j import GraphDatabase
from dotenv import load_dotenv

# ============ Neo4j 连接 ============
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test1234")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


# ============ 可调参数（一处改全局生效） ============
DF_MIN = 1                 # 文档频次阈值（第一遍保留概念，先放宽）
KEEP_TOP_N = 400           # 保留概念上限（None 表示不限）
JACCARD_MAX = 0.98         # 概念对过相似则丢弃的阈值（越大越宽松）
MAX_PHRASE_WORDS = 8       # 概念最长保留词数
REQUIRE_FIN_CONCEPTS = False   # 是否强制必须像“金融名词短语”
TITLE_ALWAYS_KEEP = True       # 标题里的概念不管 df 都保留为种子

# 可视化建议阈值（仅用于查询展示，不影响入库）
COOCCUR_MIN_TO_VIEW = 1    # 共现权重最低展示
CAUSES_MIN_COUNT_TO_VIEW = 1  # 因果证据条数最低展示

# ============ 规则/词表 ============
FORWARD_CUES = {"lead to", "leads to", "result in", "results in", "resulted in"}
BECAUSE_CUES = {"because of", "due to", "because"}
CAUSE_LEMMAS = {"cause", "lead", "result", "trigger", "drive"}

UP_WORDS = {"increase", "rise", "rises", "higher", "raise", "boost", "strengthen", "grow", "growth"}
DOWN_WORDS = {"decrease", "fall", "falls", "lower", "drop", "reduce", "weaken", "decline"}

BAD_TOKS = {
    "it", "this", "that", "they", "them", "which", "who", "whom", "those",
    "these", "there", "here"
}

# 金融词根（命中其一，更像金融概念）
FIN_TERMS = {
    "interest", "rate", "margin", "turnover", "leverage", "equity", "asset", "liability", "debt",
    "roe", "roa", "dividend", "yield", "valuation", "discount", "terminal", "profit", "income",
    "sales", "revenue", "cost", "expense", "inventory", "cash", "liquidity", "hedge", "futures",
    "forward", "spot", "premium", "basis", "volatility", "beta", "alpha", "return", "growth",
    "capital", "structure", "risk", "risk-free", "price", "pricing", "contract", "maturity",
    "convenience", "storage", "maintenance", "initial", "margin", "index", "bond", "stock"
}

# 过于泛化的概念黑名单（可按数据加词）
GENERIC_BAD = {
    "it", "this", "that", "they", "them", "which", "who", "whom", "those", "these",
    "there", "here", "value", "thing", "something", "example", "contract", "market",
    "price", "prices"
}

# 同义/规范化（可扩展）
CANON = {
    "roe": "return on equity",
    "roa": "return on assets",
    "risk free rate": "risk-free rate",
    "futures contract": "futures",
    "forward contracts": "forward contract",
}


# ============ spaCy ============
try:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
except Exception as e:
    raise SystemExit(
        f"[ERROR] spaCy 加载失败：{e}\n"
        "请先执行：\n"
        "  pip install spacy\n"
        "  python -m spacy download en_core_web_sm\n"
    )


# ============ Neo4j 基础方法 ============
def ensure_constraints():
    with driver.session() as s:
        s.run("CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE;")
        s.run("CREATE CONSTRAINT article_url IF NOT EXISTS FOR (a:Article) REQUIRE a.url IS UNIQUE;")


def merge_article(url: str, title: str, source: str = ""):
    with driver.session() as s:
        s.run("""
        MERGE (a:Article {url:$url})
        ON CREATE SET a.title=$title, a.source=$source
        """, url=url, title=title or url.split("/")[-1], source=source or "")


def merge_concept(name: str):
    with driver.session() as s:
        s.run("MERGE (c:Concept {name:$name})", name=name)


def merge_mentions(url: str, concept: str, count: int = 1):
    with driver.session() as s:
        s.run("""
        MATCH (a:Article {url:$url}), (c:Concept {name:$concept})
        MERGE (a)-[m:MENTIONS]->(c)
        ON CREATE SET m.count=$count
        ON MATCH SET  m.count = coalesce(m.count,0) + $count
        """, url=url, concept=concept, count=count)


def merge_cooccur(c1: str, c2: str, level: str):
    # 共现权重：句级较高
    w = 2 if level == "sent" else 1
    with driver.session() as s:
        s.run("""
        MATCH (x:Concept {name:$c1}), (y:Concept {name:$c2})
        MERGE (x)-[r:CO_OCCUR]->(y)
        ON CREATE SET r.weight = $w
        ON MATCH SET  r.weight = coalesce(r.weight,0) + $w
        """, c1=c1, c2=c2, w=w)


def merge_causes_agg(c1: str, c2: str, url: str, sentence: str, pol: int, cue: str, conf: float):
    # 因果边做聚合：count/conf_sum/conf
    with driver.session() as s:
        s.run("""
        MATCH (x:Concept {name:$c1}), (y:Concept {name:$c2})
        MERGE (x)-[r:CAUSES]->(y)
        ON CREATE SET r.count=1, r.conf_sum=$conf, r.conf=$conf,
                      r.polarity=$pol, r.cue=$cue, r.url=$url, r.sentence=$sentence
        ON MATCH  SET r.count   = coalesce(r.count,0)+1,
                      r.conf_sum= coalesce(r.conf_sum,0)+$conf,
                      r.conf    = r.conf_sum / r.count,
                      r.polarity= $pol, r.cue=$cue, r.url=$url, r.sentence=$sentence
        """, c1=c1, c2=c2, url=url, sentence=sentence[:480],
           pol=int(pol), cue=cue, conf=float(conf))


def graph_stats() -> Tuple[int, int, int]:
    with driver.session() as s:
        nodes = s.run("MATCH (c:Concept) RETURN count(c) AS n").single()["n"]
        causes = s.run("MATCH ()-[r:CAUSES]->() RETURN count(r) AS e").single()["e"]
        cooc = s.run("MATCH ()-[r:CO_OCCUR]->() RETURN count(r) AS e").single()["e"]
    return nodes, causes, cooc


# ============ 文本/概念处理 ============
def canonize(s: str) -> str:
    return CANON.get(s, s)


def normalize(text: str) -> str:
    # 温和归一化：小写 + 去噪 + 压空格（不词干）
    s = re.sub(r"[^A-Za-z0-9\s\-%]", " ", text or "")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def looks_bad_phrase(s: str) -> bool:
    toks = s.split()
    if len("".join(toks)) < 3:
        return True
    return all(t in BAD_TOKS for t in toks)


def is_financial_concept(s: str) -> bool:
    toks = set(s.split())
    if toks & FIN_TERMS:
        return True
    # 介词结构（常见金融名词短语）
    if " of " in s or " on " in s or " in " in s:
        return True
    return False


def looks_good_concept(s: str) -> bool:
    if not s or len(s) < 3:
        return False
    if s in GENERIC_BAD:
        return False
    if looks_bad_phrase(s):
        return False
    if REQUIRE_FIN_CONCEPTS:
        return is_financial_concept(s)
    return True


def jaccard_sim(a: str, b: str) -> float:
    A, B = set(a.split()), set(b.split())
    if not A or not B:
        return 1.0
    return len(A & B) / len(A | B)


def polarity_from_sentence(s: str) -> int:
    s = s.lower()
    score = 0
    if any(w in s for w in UP_WORDS):
        score += 1
    if any(w in s for w in DOWN_WORDS):
        score -= 1
    return 1 if score > 0 else (-1 if score < 0 else 0)


def confidence(s: str) -> float:
    hedges = {"may", "might", "could", "can", "tend", "likely", "suggests", "appears", "seems"}
    base = 0.8
    if any((" " + h + " ") in (" " + s.lower() + " ") for h in hedges):
        base -= 0.25
    return max(0.1, min(1.0, base))


def last_nouny(chunk: str) -> str:
    m = re.findall(r"[A-Za-z][A-Za-z ]{2,}", chunk)
    return m[-1].strip() if m else chunk[-60:].strip()


def first_nouny(chunk: str) -> str:
    m = re.findall(r"[A-Za-z][A-Za-z ]{2,}", chunk)
    return m[0].strip() if m else chunk[:60].strip()


def pick_nouny_left(chunk: str) -> str:
    m = re.findall(r"[A-Za-z][A-Za-z ]{2,}\s(?:of|on|in)\s[A-Za-z][A-Za-z ]{2,}", chunk)
    if m:
        return m[-1].strip()
    return last_nouny(chunk)


def pick_nouny_right(chunk: str) -> str:
    m = re.findall(r"(?:of|on|in)\s[A-Za-z][A-Za-z ]{2,}", chunk)
    if m:
        return m[0].split(" ", 1)[1].strip()
    return first_nouny(chunk)


def clip_words(s: str, k: int = MAX_PHRASE_WORDS) -> str:
    ws = s.split()
    return " ".join(ws[-k:]) if len(ws) > k else s


def extract_edges(sent: str) -> List[Tuple[str, str, str, float, int]]:
    """返回 [(cause, effect, cue, conf, polarity)]，带多重过滤以防自环/噪声"""
    out = []
    s = sent.strip()
    low = s.lower()

    # 1) X leads to Y / results in Y
    for cue in FORWARD_CUES:
        if cue in low:
            left, right = low.split(cue, 1)
            c1 = canonize(normalize(clip_words(pick_nouny_left(left))))
            c2 = canonize(normalize(clip_words(pick_nouny_right(right))))
            if c1 and c2:
                out.append((c1, c2, cue, confidence(s), polarity_from_sentence(s)))
            break

    # 2) Y because (of) X / due to X → X causes Y
    for cue in BECAUSE_CUES:
        if cue in low:
            left, right = low.split(cue, 1)
            eff = canonize(normalize(clip_words(pick_nouny_left(left))))
            cau = canonize(normalize(clip_words(pick_nouny_right(right))))
            if cau and eff:
                out.append((cau, eff, cue, confidence(s), polarity_from_sentence(s)))
            break

    # 3) 动词 lemma 触发（近似左右名词片段）
    doc = nlp(s)
    for i, tok in enumerate(doc):
        if tok.lemma_.lower() in CAUSE_LEMMAS and tok.pos_ == "VERB":
            left_text = doc[:i].text
            right_text = doc[i + 1 :].text
            c1 = canonize(normalize(clip_words(pick_nouny_left(left_text))))
            c2 = canonize(normalize(clip_words(pick_nouny_right(right_text))))
            if c1 and c2:
                out.append((c1, c2, tok.lemma_.lower(), confidence(s), polarity_from_sentence(s)))

    # 过滤
    cleaned = []
    for c1, c2, cue, conf, pol in out:
        if not c1 or not c2:
            continue
        if c1 == c2:
            continue
        if looks_bad_phrase(c1) or looks_bad_phrase(c2):
            continue
        if jaccard_sim(c1, c2) >= JACCARD_MAX:
            continue
        if not (looks_good_concept(c1) and looks_good_concept(c2)):
            continue
        if c1 in c2 or c2 in c1:
            # 互为子串（极易成为“伪自环”）直接跳过
            continue
        cleaned.append((c1, c2, cue, conf, pol))
    return cleaned


def extract_concepts_from_sentence(sent_text: str) -> List[str]:
    """句级概念候选：左右 nouny + 所有 of/on/in 结构"""
    low = sent_text.lower()
    cands: Set[str] = set()

    for chunk in (pick_nouny_left(low), pick_nouny_right(low)):
        cand = canonize(normalize(clip_words(chunk)))
        if looks_good_concept(cand):
            cands.add(cand)

    for m in re.findall(r"[A-Za-z][A-Za-z ]{2,}\s(?:of|on|in)\s[A-Za-z][A-Za-z ]{2,}", low):
        cand = canonize(normalize(clip_words(m)))
        if looks_good_concept(cand):
            cands.add(cand)

    return list(cands)


# ============ 针对站点的轻清洗 ============
def clean_wiki_text(t: str) -> str:
    lines = [ln for ln in t.splitlines() if ln.strip()]
    drop_heads = ("see also", "notes", "references", "external links", "==")
    keep = []
    for ln in lines:
        low = ln.strip().lower()
        if any(low.startswith(h) for h in drop_heads):
            break
        keep.append(ln)
    return "\n".join(keep)


def clean_investopedia_text(t: str) -> str:
    # 去掉“Bottom Line”等尾注段的冗余
    return re.sub(r"\bthe bottom line\b.*$", "", t, flags=re.IGNORECASE | re.DOTALL)


def site_clean(url: str, text: str) -> str:
    if "wikipedia.org" in url:
        return clean_wiki_text(text)
    if "investopedia.com" in url:
        return clean_investopedia_text(text)
    return text


# ============ 主流程：两遍扫描 ============
def main():
    ensure_constraints()

    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "Extractor", "output.json")
    if not os.path.exists(path):
        raise SystemExit(f"[ERROR] 没找到 output.json：{path}")

    data: Dict[str, dict] = json.load(open(path, "r", encoding="utf-8"))
    print(f"[INFO] articles={len(data)}")

    # ---------- 第一遍：统计文档频次 DF ----------
    df: Dict[str, int] = {}
    title_concepts_seen: Set[str] = set()

    for url, rec in data.items():
        text = (rec.get("full_text") or "").strip()
        if not text:
            continue
        text = site_clean(url, text)

        doc_concepts: Set[str] = set()

        # 标题概念
        title = (rec.get("title") or "").strip()
        title_terms = re.findall(r"[A-Za-z][A-Za-z ]{2,}", title)
        if title_terms:
            tconcept = canonize(normalize(max(title_terms, key=len)))
            if looks_good_concept(tconcept):
                doc_concepts.add(tconcept)
                title_concepts_seen.add(tconcept)

        # 句子概念
        for sent in nlp(text).sents:
            for c in extract_concepts_from_sentence(sent.text):
                doc_concepts.add(c)

        for c in doc_concepts:
            df[c] = df.get(c, 0) + 1

    KEEP: Set[str] = {c for c, v in df.items() if v >= DF_MIN}
    if TITLE_ALWAYS_KEEP:
        KEEP |= title_concepts_seen
    if KEEP_TOP_N and len(KEEP) > KEEP_TOP_N:
        KEEP = set(sorted(KEEP, key=lambda c: df.get(c, 0), reverse=True)[:KEEP_TOP_N])

    print(f"[INFO] concepts_all={len(df)}; kept={len(KEEP)}; df_min={DF_MIN}; topN={KEEP_TOP_N}")

    # ---------- 第二遍：入库（仅 KEEP 为中心，但允许“至少一端在 KEEP”来连边） ----------
    for url, rec in data.items():
        title = (rec.get("title") or "").strip()
        raw = (rec.get("full_text") or "").strip()
        if not raw:
            continue
        text = site_clean(url, raw)

        merge_article(url, title)

        # 标题 mentions
        title_terms = re.findall(r"[A-Za-z][A-Za-z ]{2,}", title)
        if title_terms:
            concept = canonize(normalize(max(title_terms, key=len)))
            if concept in KEEP and looks_good_concept(concept):
                merge_concept(concept)
                merge_mentions(url, concept, 1)

        # 逐句：抽概念→共现；抽因果→聚合
        for sent in nlp(text).sents:
            cands = [c for c in extract_concepts_from_sentence(sent.text)]
            # 共现（至少一端在 KEEP）
            for i in range(len(cands)):
                for j in range(i + 1, len(cands)):
                    c1, c2 = cands[i], cands[j]
                    if c1 == c2:
                        continue
                    if (c1 in KEEP) or (c2 in KEEP):
                        merge_concept(c1)
                        merge_concept(c2)
                        merge_cooccur(c1, c2, level="sent")

            # 因果（至少一端在 KEEP）
            for c1, c2, cue, conf, pol in extract_edges(sent.text):
                if c1 != c2 and ((c1 in KEEP) or (c2 in KEEP)):
                    merge_concept(c1)
                    merge_concept(c2)
                    merge_causes_agg(c1, c2, url, sent.text, pol, cue, conf)

    n, e1, e2 = graph_stats()
    print(f"[INFO] graph nodes={n}, CAUSES={e1}, CO_OCCUR={e2}")
    print("\n[HINT] 推荐在 Neo4j Browser 里先用这条查看：\n"
          f"  MATCH (c1:Concept)-[r]->(c2:Concept)\n"
          f"  WHERE (type(r)='CAUSES'  AND coalesce(r.count,1)  >= {CAUSES_MIN_COUNT_TO_VIEW}) OR\n"
          f"        (type(r)='CO_OCCUR' AND coalesce(r.weight,1) >= {COOCCUR_MIN_TO_VIEW})\n"
          f"  RETURN c1,r,c2;")


if __name__ == "__main__":
    main()
