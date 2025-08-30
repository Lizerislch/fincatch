# -*- coding: utf-8 -*-
"""
Financial Causal Graph Explorer (Streamlit)
-------------------------------------------

This app connects to a Neo4j database that contains a causal/co-occurrence graph
built from Q1/Q2, and renders an interactive visualization in the browser.

Key features:
- Sidebar filters for edge thresholds (CAUSES confidence/count, CO_OCCUR weight)
- Optional center concept to focus on a small neighborhood (1..2 hops)
- Cluster-aware coloring (cluster_graph / cluster_embed) and simple cluster filter
- Click on a CAUSES edge to open its source URL in a new tab

Requirements:
- A running Neo4j 5.x with the graph already ingested by your Q2 pipeline
- .env file placed next to this script with NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD
- Streamlit to run: `streamlit run app.py`

Note:
- We embed a small HTML page using vis-network (client-side) to avoid pyvis template issues.
- A small `make_net` function with pyvis is kept for reference but is not used by default.
"""

import os
import json
import tempfile
from typing import Dict, List, Any, Optional, Set

import streamlit as st
from neo4j import GraphDatabase
from pyvis.network import Network  # kept for reference (see make_net), not used in default path
from dotenv import load_dotenv

# -----------------------------
# Environment & Neo4j driver
# -----------------------------

load_dotenv()  # load .env into process env

# Fallbacks are provided for convenience; prefer setting them via .env
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "test1234")


@st.cache_resource
def get_driver(uri: str, user: str, pwd: str):
    """
    Create and cache a Neo4j driver (Streamlit resource cache).
    The driver is a heavy object; caching avoids rebuilding it on every rerun.

    Args:
        uri: Bolt URI, e.g. bolt://localhost:7687
        user: Neo4j username
        pwd:  Neo4j password

    Returns:
        neo4j.Driver
    """
    # You could add a small connectivity check here if desired:
    # with GraphDatabase.driver(uri, auth=(user, pwd)) as drv:
    #     with drv.session() as s:
    #         s.run("RETURN 1")
    return GraphDatabase.driver(uri, auth=(user, pwd))


driver = get_driver(NEO4J_URI, NEO4J_USER, NEO4J_PASS)

# -----------------------------
# Streamlit page setup
# -----------------------------

st.set_page_config(page_title="Financial Causal Graph", layout="wide")
st.title("Financial Causal Graph")


# -----------------------------
# Sidebar filters & controls
# -----------------------------
with st.sidebar:
    st.subheader("Filters")

    # Optional "center" concept to focus the query on a 1..2 hop neighborhood
    query_concept: str = st.text_input("Center concept (optional)", "")

    # Thresholds for CAUSES edges
    min_conf: float = st.slider("Min CAUSES confidence", 0.0, 1.0, 0.6, 0.05)
    min_count: int = st.slider("Min CAUSES evidence count", 1, 5, 1, 1)

    # Whether to include co-occurrence edges and their threshold
    include_co: bool = st.checkbox("Include CO_OCCUR edges", True)
    min_co: int = st.slider(
        "Min CO_OCCUR weight", 1, 5, 1, 1, disabled=not include_co
    )

    # Hard limit to avoid rendering super-dense graphs in the browser
    limit_edges: int = st.slider("Edge limit", 50, 2000, 500, 50)

    # Cluster coloring options: none / graph community / embedding KMeans
    color_mode: str = st.selectbox(
        "Color nodes by",
        ["none", "cluster_graph", "cluster_embed"],
        index=1,
        help="Pick which cluster label to color nodes by. "
             "`cluster_graph`: community from graph structure; "
             "`cluster_embed`: KMeans over (Node2Vec ⊕ text embedding).",
    )

    # Allow users to show only a subset of clusters (e.g. "0,1,4")
    cluster_filter: str = st.text_input(
        "Show only clusters (comma-separated, e.g. 0,1)", value=""
    )
    selected_clusters: Set[int] = set(
        int(x.strip()) for x in cluster_filter.split(",") if x.strip().isdigit()
    )


# -----------------------------
# Cypher query helpers
# -----------------------------
def fetch_edges_global(
    session,
    min_conf: float,
    min_count: int,
    include_co: bool,
    min_co: int,
    limit_edges: int,
) -> List[Dict[str, Any]]:
    """
    Fetch a global sample of edges (CAUSES and optionally CO_OCCUR) with thresholds.
    We UNION two MATCH queries to get both edge types and return normalized columns.

    Returns:
        List of dict rows with keys:
        - rel: 'CAUSES' or 'CO_OCCUR'
        - s, t: source/target concept names
        - conf, cnt, cue, sentence, url: edge metadata (CAUSES only for conf/cue/sentence/url)
        - cg1, cg2, ce1, ce2: cluster labels for coloring
    """
    if include_co:
        q = """
        MATCH (c1:Concept)-[r:CAUSES]->(c2:Concept)
        WHERE coalesce(r.conf,0) >= $min_conf AND coalesce(r.count,1) >= $min_count
        RETURN 'CAUSES' AS rel, c1.name AS s, c2.name AS t,
               r.conf AS conf, r.count AS cnt, r.cue AS cue, r.sentence AS sentence, r.url AS url,
               c1.cluster_graph AS cg1, c2.cluster_graph AS cg2, c1.cluster_embed AS ce1, c2.cluster_embed AS ce2
        LIMIT $limit_edges
        UNION ALL
        MATCH (c1:Concept)-[r:CO_OCCUR]->(c2:Concept)
        WHERE coalesce(r.weight,1) >= $min_co
        RETURN 'CO_OCCUR' AS rel, c1.name AS s, c2.name AS t,
               null AS conf, r.weight AS cnt, null AS cue, null AS sentence, null AS url,
               c1.cluster_graph AS cg1, c2.cluster_graph AS cg2, c1.cluster_embed AS ce1, c2.cluster_embed AS ce2
        LIMIT $limit_edges
        """
        params = dict(
            min_conf=min_conf,
            min_count=min_count,
            min_co=min_co,
            limit_edges=limit_edges,
        )
    else:
        q = """
        MATCH (c1:Concept)-[r:CAUSES]->(c2:Concept)
        WHERE coalesce(r.conf,0) >= $min_conf AND coalesce(r.count,1) >= $min_count
        RETURN 'CAUSES' AS rel, c1.name AS s, c2.name AS t,
               r.conf AS conf, r.count AS cnt, r.cue AS cue, r.sentence AS sentence, r.url AS url,
               c1.cluster_graph AS cg1, c2.cluster_graph AS cg2, c1.cluster_embed AS ce1, c2.cluster_embed AS ce2
        LIMIT $limit_edges
        """
        params = dict(min_conf=min_conf, min_count=min_count, limit_edges=limit_edges)

    return session.run(q, **params).data()


def fetch_edges_center(
    session,
    center: str,
    min_conf: float,
    min_count: int,
    include_co: bool,
    min_co: int,
    limit_edges: int,
) -> List[Dict[str, Any]]:
    """
    Focused query around a center node within 1..2 hops.
    UNWIND is used to normalize variable-length path expansions into individual edges.

    Args:
        center: Concept name to focus on (must match node property `:Concept {name:center}`)

    Returns:
        Same column schema as fetch_edges_global().
    """
    q = """
    MATCH p=(c:Concept {name:$center})-[r*1..2]-(n:Concept)
    UNWIND r AS rl
    UNWIND rl AS r
    WITH DISTINCT r, type(r) AS typ, startNode(r) AS c1, endNode(r) AS c2
    WHERE (typ='CAUSES'  AND coalesce(r.conf,0)  >= $min_conf AND coalesce(r.count,1)  >= $min_count)
       OR ($include_co AND typ='CO_OCCUR' AND coalesce(r.weight,1) >= $min_co)
    RETURN typ AS rel, c1.name AS s, c2.name AS t,
           (CASE WHEN typ='CAUSES' THEN r.conf   ELSE null END) AS conf,
           (CASE WHEN typ='CAUSES' THEN r.count  ELSE r.weight END) AS cnt,
           (CASE WHEN typ='CAUSES' THEN r.cue    ELSE null END) AS cue,
           (CASE WHEN typ='CAUSES' THEN r.sentence ELSE null END) AS sentence,
           (CASE WHEN typ='CAUSES' THEN r.url    ELSE null END) AS url,
           c1.cluster_graph AS cg1, c2.cluster_graph AS cg2, c1.cluster_embed AS ce1, c2.cluster_embed AS ce2
    LIMIT $limit_edges
    """
    return session.run(
        q,
        center=center,
        min_conf=min_conf,
        min_count=min_count,
        include_co=include_co,
        min_co=min_co,
        limit_edges=limit_edges,
    ).data()


# -----------------------------
# (Optional) PyVis helper — not used by default
# -----------------------------
def make_net(rows: List[Dict[str, Any]]) -> Network:
    """
    Build a PyVis Network from the rows.
    Note: Not used in the default render path (we use pure vis-network HTML),
    but kept here as a reference example.

    Caution:
    - set_options must receive **pure JSON**, not "var options = {...}" JavaScript.
    """
    net = Network(height="750px", width="100%", directed=True)

    # IMPORTANT: pass *pure JSON* here (no JS variable assignment).
    net.set_options(
        """
    {
      "nodes": { "shape": "dot", "size": 12, "font": { "size": 14 } },
      "edges": { "arrows": { "to": { "enabled": true, "scaleFactor": 0.7 } }, "smooth": false },
      "physics": { "stabilization": true, "barnesHut": { "springLength": 150 } },
      "interaction": { "hover": true }
    }
    """
    )

    nodes_seen: Set[str] = set()
    for r in rows:
        s, t, rel = r["s"], r["t"], r["rel"]

        for n in (s, t):
            if n not in nodes_seen:
                nodes_seen.add(n)
                net.add_node(n, label=n, title=n)

        if rel == "CAUSES":
            title = f"CAUSES (conf={r['conf']:.2f}, count={r['cnt']})"
            if r.get("cue"):
                title += f"\\ncue={r['cue']}"
            if r.get("sentence"):
                title += f"\\n{r['sentence']}"
            if r.get("url"):
                title += f"\\n{r['url']}"
            net.add_edge(s, t, title=title, arrows="to")  # directed CAUSES edge
        else:
            title = f"CO_OCCUR (weight={r['cnt']})"
            net.add_edge(s, t, title=title, arrows="", dashes=True)  # undirected-ish style

    return net


# -----------------------------
# vis-network (HTML) renderer
# -----------------------------
def render_vis_network(
    rows: List[Dict[str, Any]],
    color_mode: str = "cluster_graph",
    selected_clusters: Optional[Set[int]] = None,
) -> str:
    """
    Build a self-contained HTML string that renders a vis-network graph.

    Args:
        rows: list of edges produced by fetch_edges_*()
        color_mode: 'none' | 'cluster_graph' | 'cluster_embed'
        selected_clusters: if provided, only nodes/edges whose nodes are in the selected
                           clusters (w.r.t. color_mode) will be included

    Returns:
        HTML string to be passed to streamlit.components.v1.html(...)
    """
    nodes: Dict[str, Dict[str, Any]] = {}  # concept -> info (cg, ce, label/title)
    edges: List[Dict[str, Any]] = []

    def palette(idx: int) -> str:
        """A simple 20-color palette; cycles if idx is large/negative."""
        colors = [
            "#2563eb", "#16a34a", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4",
            "#f97316", "#22c55e", "#3b82f6", "#a855f7", "#10b981", "#eab308",
            "#dc2626", "#14b8a6", "#fb7185", "#60a5fa", "#34d399", "#fbbf24",
            "#c084fc", "#2dd4bf"
        ]
        try:
            return colors[idx % len(colors)]
        except Exception:
            return "#6b7280"  # gray

    def node_color(info: Dict[str, Any]) -> str:
        """Pick a color based on chosen cluster label; 'none' -> gray."""
        if color_mode == "cluster_graph":
            cid = info.get("cg", None)
            return palette(cid) if cid is not None else "#6b7280"
        if color_mode == "cluster_embed":
            cid = info.get("ce", None)
            return palette(cid) if cid is not None else "#6b7280"
        return "#6b7280"

    # Collect node cluster data from edge rows
    for r in rows:
        s, t = r["s"], r["t"]
        s_info = nodes.get(s, {"id": s, "label": s, "title": s})
        t_info = nodes.get(t, {"id": t, "label": t, "title": t})
        s_info["cg"], s_info["ce"] = r.get("cg1"), r.get("ce1")
        t_info["cg"], t_info["ce"] = r.get("cg2"), r.get("ce2")
        nodes[s], nodes[t] = s_info, t_info

    # Optional cluster-based node filtering
    if selected_clusters:
        def in_sel(nfo: Dict[str, Any]) -> bool:
            cid = (
                nfo.get("cg")
                if color_mode == "cluster_graph"
                else (nfo.get("ce") if color_mode == "cluster_embed" else None)
            )
            return cid in selected_clusters

        nodes = {k: v for k, v in nodes.items() if in_sel(v)}

    # Build edges (respect filtering if active)
    for r in rows:
        if selected_clusters:
            s_ok = (
                r.get("cg1") in selected_clusters
                if color_mode == "cluster_graph"
                else r.get("ce1") in selected_clusters
            )
            t_ok = (
                r.get("cg2") in selected_clusters
                if color_mode == "cluster_graph"
                else r.get("ce2") in selected_clusters
            )
            if not (s_ok and t_ok):
                continue

        rel = r["rel"]
        if rel == "CAUSES":
            title = "CAUSES"
            if r.get("conf") is not None:
                title += f" (conf={r['conf']:.2f}, count={r['cnt']})"
            if r.get("cue"):
                title += f"\\ncue={r['cue']}"
            if r.get("sentence"):
                title += f"\\n{r['sentence']}"
            if r.get("url"):
                title += f"\\n{r['url']}"
            edges.append({"from": r["s"], "to": r["t"], "arrows": "to", "title": title})
        else:
            edges.append({
                "from": r["s"], "to": r["t"], "dashes": True,
                "title": f"CO_OCCUR (weight={r['cnt']})"
            })

    # Compute simple degree for sizing
    degree: Dict[str, int] = {}
    for e in edges:
        degree[e["from"]] = degree.get(e["from"], 0) + 1
        degree[e["to"]] = degree.get(e["to"], 0) + 1

    # Assemble vis-network nodes
    vis_nodes = []
    for n, info in nodes.items():
        size = 8 + int((degree.get(n, 1) - 1) * 1.5)  # scale size slightly by degree
        vis_nodes.append({
            "id": n, "label": n, "title": n, "size": size,
            "color": {"background": node_color(info), "border": "#111827"}
        })

    # Self-contained HTML with CDN-loaded vis-network
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    #mynetwork {{ height: 750px; width: 100%; border: 1px solid #e5e7eb; }}
    .legend {{ font: 13px/1.2 sans-serif; margin: 8px 0 12px; color: #334155; }}
  </style>
</head>
<body>
<div class="legend">Color by: <b>{color_mode}</b>{(" — showing clusters: "+",".join(map(str,sorted(selected_clusters)))) if selected_clusters else ""}</div>
<div id="mynetwork"></div>
<script>
  const nodes = new vis.DataSet({json.dumps(vis_nodes)});
  const edges = new vis.DataSet({json.dumps(edges)});
  const container = document.getElementById('mynetwork');
  const data = {{ nodes, edges }};
  const options = {{
    nodes: {{ shape: "dot", font: {{ size: 14 }} }},
    edges: {{ smooth: false, arrows: {{ to: {{ enabled: true, scaleFactor: 0.7 }} }} }},
    physics: {{ stabilization: true, barnesHut: {{ springLength: 150 }} }},
    interaction: {{ hover: true }}
  }};
  const network = new vis.Network(container, data, options);

  // Open source URL if present in the edge tooltip (CAUSES edges embed the URL in title)
  network.on("click", function (params) {{
    if (params.edges && params.edges.length > 0) {{
      const eid = params.edges[0]; const e = edges.get(eid);
      if (e && e.title && e.title.includes("http")) {{
        const m = e.title.match(/https?:\\/\\/[^\\s]+/);
        if (m) window.open(m[0], "_blank");
      }}
    }}
  }});
</script>
</body>
</html>
"""
    return html


# -----------------------------
# Main query & render
# -----------------------------
with driver.session() as session:
    if query_concept.strip():
        # Centered view around a chosen concept (1..2 hops)
        rows = fetch_edges_center(
            session,
            query_concept.strip(),
            min_conf,
            min_count,
            include_co,
            min_co,
            limit_edges,
        )
    else:
        # Global sample view with thresholds
        rows = fetch_edges_global(
            session, min_conf, min_count, include_co, min_co, limit_edges
        )

# Empty result guard
if not rows:
    st.info(
        "No edges match current filters. Try lowering thresholds or searching another concept."
    )
else:
    # Build HTML and embed in Streamlit
    html = render_vis_network(
        rows, color_mode=color_mode, selected_clusters=selected_clusters
    )
    st.components.v1.html(html, height=780, scrolling=True)
