# q3.py
# ------------------------------------------------------------
# Q3: Cluster financial knowledge (concepts) using content + graph relationships
# 输出：
#  1) clusters.csv     概念与聚类结果（图社区+嵌入KMeans）与度数/PR等
#  2) 回写 Neo4j       给 Concept 节点打 cluster_graph / cluster_embed 属性
#  3) 可选：umap_*.csv 2D降维坐标，可用于前端或Notebook可视化
# ------------------------------------------------------------
import os, json, math
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from neo4j import GraphDatabase

import networkx as nx
from node2vec import Node2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import community as community_louvain  # python-louvain
    HAS_LOUVAIN = True
except Exception:
    HAS_LOUVAIN = False

# ============ 配置 ============
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "test1234")  # 按你刚才设置

# 图边权重参数（聚合 Q2 的 CAUSES / CO_OCCUR）
ALPHA_CAUSES = 1.0    # 对 CAUSES.count 的权重
BETA_CO = 0.5         # 对 CO_OCCUR.weight 的权重
GAMMA_CONF = 0.3      # 对 CAUSES.conf 的加成
MIN_EDGE_SCORE = 0.5  # 边的最小分数阈值（太弱的边丢弃）

# 嵌入与聚类参数
N2V_DIM = 64
N2V_WALKS = 20
N2V_WALKLEN = 40
N2V_WINDOW = 5
N2V_P = 1.0
N2V_Q = 1.0

TFIDF_NGRAM = (1,2)
SVD_DIM = 128              # 文本降维维度（TF-IDF -> SVD）
K_RANGE = list(range(4, 13))  # KMeans 搜索的 K 值范围
UMAP_DIM = 2               # 需要 umap-learn 时可以生成 2D 坐标（可选）

OUT_CSV = "clusters.csv"
UMAP_CSV = "umap_coords.csv"

# 可选：从 Q1 的 output.json 抽取 “概念上下文” 作为文本增强（找不到不报错）
HERE = os.path.dirname(__file__)
OUTPUT_JSON = os.path.join(HERE, "..", "Extractor", "output.json")


# ============ Neo4j 取图 ============
def fetch_graph_and_text():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    nodes = []
    edges = []
    sentences = []  # (c1, c2, sentence, url)
    mentions = defaultdict(list)  # concept -> [article title]

    with driver.session() as s:
        # 概念节点
        res = s.run("MATCH (c:Concept) RETURN c.name AS name")
        nodes = [r["name"] for r in res]

        # 关系（CAUSES / CO_OCCUR）
        res = s.run("""
        MATCH (c1:Concept)-[r]->(c2:Concept)
        WHERE type(r) IN ['CAUSES','CO_OCCUR']
        RETURN c1.name AS s, c2.name AS t, type(r) AS rel,
               coalesce(r.count, 1) AS count,
               coalesce(r.weight, 1) AS weight,
               coalesce(r.conf, 0.5) AS conf,
               r.sentence AS sentence, r.url AS url
        """)
        for r in res:
            edges.append((r["s"], r["t"], r["rel"], float(r["count"]), float(r["weight"]), float(r["conf"])))
            if r["rel"] == "CAUSES" and r["sentence"]:
                sentences.append((r["s"], r["t"], r["sentence"], r.get("url")))

        # 文章标题作为概念的轻量文本
        res = s.run("""
        MATCH (a:Article)-[:MENTIONS]->(c:Concept)
        RETURN c.name AS name, collect(distinct a.title)[..5] AS titles
        """)
        for r in res:
            mentions[r["name"]] = list(r["titles"] or [])

    driver.close()
    return nodes, edges, sentences, mentions


def edge_score(rel, count, weight, conf):
    if rel == "CAUSES":
        return ALPHA_CAUSES * count + GAMMA_CONF * conf
    else:  # CO_OCCUR
        return BETA_CO * weight


def build_graph(nodes, edges):
    """构建带权无向图（社区检测/Node2Vec 更稳）；箭头信息用于可视化，不影响聚类"""
    G = nx.Graph()
    for n in nodes:
        G.add_node(n)
    for s, t, rel, cnt, w, conf in edges:
        sc = edge_score(rel, cnt, w, conf)
        if sc < MIN_EDGE_SCORE:
            continue
        if G.has_edge(s, t):
            G[s][t]["weight"] += sc
        else:
            G.add_edge(s, t, weight=sc, rels=Counter({rel:1}))
    return G


# ============ 文本特征 ============
def build_text_corpus(nodes, sentences, mentions):
    """为每个概念汇总一句话证据 + 标题碎片；找不到就用概念名本身"""
    by_concept = defaultdict(list)
    for s, t, sent, url in sentences:
        by_concept[s].append(sent)
        by_concept[t].append(sent)
    for c, titles in mentions.items():
        by_concept[c].extend(titles)

    # optional: 加一点 Q1 的内容（如能读到 output.json）
    if os.path.exists(OUTPUT_JSON):
        try:
            data = json.load(open(OUTPUT_JSON, "r", encoding="utf-8"))
            # 只取 title + summary，避免过长
            for url, rec in data.items():
                title = (rec.get("title") or "").strip()
                summ  = (rec.get("summary") or "").strip()
                # 简单地把标题包含的概念“指派”这段文本
                for c in nodes:
                    if c in title.lower():
                        by_concept[c].extend([title, summ])
        except Exception:
            pass

    corpus = []
    for c in nodes:
        txt = " ".join(by_concept.get(c, [c]))  # 保底用概念名
        corpus.append(txt)
    return corpus


def text_embeddings(corpus):
    vec = TfidfVectorizer(ngram_range=TFIDF_NGRAM, min_df=1, max_df=0.9)
    X = vec.fit_transform(corpus)  # 稀疏
    # 降维到 SVD_DIM
    k = min(SVD_DIM, max(2, X.shape[1]-1))
    svd = TruncatedSVD(n_components=k, random_state=42)
    Xr = svd.fit_transform(X)
    Xr = StandardScaler().fit_transform(Xr)
    return Xr


# ============ 图嵌入 ============
def graph_embeddings(G):
    # Node2Vec（无向、带权）
    n2v = Node2Vec(G, dimensions=N2V_DIM, walk_length=N2V_WALKLEN,
                   num_walks=N2V_WALKS, workers=1, p=N2V_P, q=N2V_Q,
                   weight_key="weight")
    model = n2v.fit(window=N2V_WINDOW, min_count=1, batch_words=128)
    nodes = list(G.nodes())
    emb = np.vstack([model.wv[n] for n in nodes])
    return nodes, emb


# ============ 聚类 ============
def louvain_cluster(G):
    if HAS_LOUVAIN:
        part = community_louvain.best_partition(G, weight="weight", random_state=42)
        modularity = community_louvain.modularity(part, G, weight="weight")
        return part, modularity
    else:
        # 退化：使用贪婪模块度社区
        comms = nx.algorithms.community.greedy_modularity_communities(G, weight="weight")
        part = {}
        for i, com in enumerate(comms):
            for n in com:
                part[n] = i
        mod = nx.algorithms.community.quality.modularity(G, comms, weight="weight")
        return part, mod


def kmeans_best(X, node_names):
    best_k, best_s, best_labels = None, -1, None
    n = X.shape[0]
    ks = [k for k in K_RANGE if k < n] or [min(4, n)]
    for k in ks:
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = km.fit_predict(X)
        if len(set(labels)) > 1:
            s = silhouette_score(X, labels, metric="euclidean")
        else:
            s = -1
        if s > best_s:
            best_k, best_s, best_labels = k, s, labels
    return best_k, best_s, best_labels


# ============ 主流程 ============
def main():
    nodes, edges, sentences, mentions = fetch_graph_and_text()
    print(f"[INFO] nodes={len(nodes)}, edges(raw)={len(edges)}")

    G = build_graph(nodes, edges)
    nodes = list(G.nodes())
    print(f"[INFO] nodes(after filter)={len(nodes)}, edges={G.number_of_edges()}")

    # 图指标（可用于评估 & 导出）
    deg = dict(G.degree(weight=None))
    pr = nx.pagerank(G, weight="weight") if G.number_of_edges() > 0 else {n:1/len(nodes) for n in nodes}

    # 嵌入（图 + 文本）
    n2v_nodes, Zg = graph_embeddings(G)
    name2idx = {n:i for i,n in enumerate(n2v_nodes)}

    corpus = build_text_corpus(nodes, sentences, mentions)
    Zt = text_embeddings(corpus)

    # 对齐顺序
    Zt = Zt[[nodes.index(n) for n in n2v_nodes], :]
    # 拼接
    Z = np.hstack([Zg, Zt])

    # Louvain（结构社区）
    part, modularity = louvain_cluster(G)
    cluster_graph = [part[n] for n in n2v_nodes]

    # KMeans（内容+结构嵌入）
    best_k, best_s, labels = kmeans_best(Z, n2v_nodes)
    cluster_embed = labels.tolist()

    print(f"[INFO] Louvain modularity={modularity:.4f}")
    print(f"[INFO] KMeans k={best_k} silhouette={best_s:.4f}")

    # 导出 CSV
    df = pd.DataFrame({
        "concept": n2v_nodes,
        "cluster_graph": cluster_graph,
        "cluster_embed": cluster_embed,
        "degree": [deg.get(n,0) for n in n2v_nodes],
        "pagerank": [pr.get(n,0.0) for n in n2v_nodes]
    }).sort_values(["cluster_graph","pagerank"], ascending=[True, False])
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"[INFO] wrote {OUT_CSV} ({len(df)} rows)")

    # 回写 Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session() as s:
        for _, row in df.iterrows():
            s.run("""
            MATCH (c:Concept {name:$name})
            SET c.cluster_graph=$cg, c.cluster_embed=$ce
            """, name=row["concept"], cg=int(row["cluster_graph"]), ce=int(row["cluster_embed"]))
    driver.close()
    print("[INFO] clusters written back to Neo4j: c.cluster_graph / c.cluster_embed")

    # 可选：UMAP 2D 降维，方便前端/Notebook展示
    try:
        import umap
        reducer = umap.UMAP(n_components=UMAP_DIM, random_state=42)
        coords = reducer.fit_transform(Z)
        umap_df = pd.DataFrame(coords, columns=["x","y"])
        umap_df["concept"] = n2v_nodes
        umap_df["cluster_embed"] = cluster_embed
        umap_df["cluster_graph"] = cluster_graph
        umap_df.to_csv(UMAP_CSV, index=False, encoding="utf-8")
        print(f"[INFO] wrote {UMAP_CSV}")
    except Exception as e:
        print(f"[WARN] UMAP skipped: {e}")


if __name__ == "__main__":
    main()
