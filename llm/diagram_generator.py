from __future__ import annotations

import os
import base64
import json
import textwrap
import time
import hashlib
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

import requests
load_dotenv()
OUTPUT_DIR = Path("outputs/diagrams")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
KROKI_URL   = "https://kroki.io/mermaid/svg"
HF_IMG_URL  = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"


def _slug(text: str, length: int = 32) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:length]


def _save(content: bytes, suffix: str, label: str) -> str:
    fname = OUTPUT_DIR / f"{label}_{_slug(label)}{suffix}"
    fname.write_bytes(content)
    return str(fname)



def _render_mermaid_svg(mermaid_definition: str, label: str) -> Optional[str]:
   
    try:
       
        resp = requests.post(
            KROKI_URL,
            data=mermaid_definition.encode("utf-8"),
            headers={"Content-Type": "text/plain"},
            timeout=15,
        )
        if resp.status_code == 200 and b"<svg" in resp.content:
            path = _save(resp.content, ".svg", label)
            print(f"      [Kroki] SVG saved → {path}")
            return path
        else:
            print(f"      [Kroki] HTTP {resp.status_code}: {resp.text[:120]}")
    except Exception as e:
        print(f"      [Kroki] Request failed: {e}")
    return None


def _build_image_prompt(query: str, level: str) -> str:

    positive = (
        f"educational diagram explaining {query}, "
        "clean infographic style, labeled arrows, "
        "white background, academic illustration, "
        f"suitable for {level} students, "
        "high detail, vector art look, minimal color palette"
    )
    return positive

# Adding Negative prompts for refined results
_HF_NEGATIVE = (
    "photorealistic, dark background, blurry, watermark, "
    "signature, text errors, cluttered, people, faces"
)


def _render_hf_image(query: str, level: str, label: str) -> Optional[str]:
  
    if not HF_TOKEN:
        print("      [HF] HF_TOKEN not set — skipping image fallback")
        return None

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": _build_image_prompt(query, level),
        "parameters": {
            "negative_prompt": _HF_NEGATIVE,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "width": 768,
            "height": 512,
        },
    }
    # Facing errors thus implementing retry strategy
    for attempt in range(3):
        try:
            resp = requests.post(HF_IMG_URL, headers=headers,
                                 json=payload, timeout=90)
            if resp.status_code == 200:
                path = _save(resp.content, ".png", label)
                print(f"      [HF] Image saved → {path}")
                return path
            elif resp.status_code == 503:
                wait = json.loads(resp.text).get("estimated_time", 20)
                print(f"      [HF] Model loading, waiting {wait:.0f}s…")
                time.sleep(min(wait, 30))
            else:
                print(f"      [HF] HTTP {resp.status_code}: {resp.text[:120]}")
                break
        except Exception as e:
            print(f"      [HF] Error: {e}")
            break
    return None

#Fallback function
def _render_matplotlib(query: str, mermaid_definition: str, label: str) -> str:
    """
    Parse Mermaid node/edge lines and draw a simple NetworkX graph
    using matplotlib.  Works 100 % offline.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import re

        nodes: dict[str, str] = {}
        edges: list[tuple[str, str, str]] = []

        for line in mermaid_definition.splitlines():
            line = line.strip()
         
            for m in re.finditer(r'(\w+)\[([^\]]+)\]', line):
                nodes[m.group(1)] = m.group(2)
           
            em = re.search(r'(\w+)\s*--[->]+\|?([^|]*)\|?\s*(\w+)', line)
            if em:
                edges.append((em.group(1), em.group(3), em.group(2).strip()))

        # Fall back to ID as label
        all_ids = {n for e in edges for n in (e[0], e[1])}
        for nid in all_ids:
            nodes.setdefault(nid, nid)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("#ffffff")
        ax.axis("off")
        ax.set_title(f"Concept Map: {query}", fontsize=14, fontweight="bold", pad=12)

        if not nodes:
            ax.text(0.5, 0.5, f"Topic: {query}\n\n(Install networkx for full graph)",
                    ha="center", va="center", fontsize=12, wrap=True,
                    bbox=dict(boxstyle="round", facecolor="#e3f2fd"))
        else:
            try:
                import networkx as nx
                G = nx.DiGraph()
                G.add_nodes_from(nodes.keys())
                G.add_edges_from([(e[0], e[1]) for e in edges])
                pos = nx.spring_layout(G, seed=42)
                labels = {k: textwrap.fill(v, 12) for k, v in nodes.items()}
                nx.draw_networkx_nodes(G, pos, ax=ax, node_size=2000,
                                       node_color="#bbdefb", alpha=0.9)
                nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)
                nx.draw_networkx_edges(G, pos, ax=ax, arrows=True,
                                       arrowsize=20, edge_color="#1565c0",
                                       connectionstyle="arc3,rad=0.1")
                edge_labels = {(e[0], e[1]): e[2] for e in edges if e[2]}
                nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=7)
            except ImportError:
                ax.text(0.5, 0.5, "\n".join(f"{nodes.get(a,a)} → {nodes.get(b,b)}"
                                             for a, b, _ in edges),
                        ha="center", va="center", fontsize=10)

        path = str(OUTPUT_DIR / f"{label}_{_slug(label)}_fallback.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"      [matplotlib] Fallback diagram saved → {path}")
        return path

    except Exception as e:
        print(f"      [matplotlib] Failed: {e}")
        
        placeholder = str(OUTPUT_DIR / f"{label}_placeholder.txt")
        Path(placeholder).write_text(f"Diagram generation failed for: {query}\n{e}")
        return placeholder


def generate_diagram(
    query: str,
    mermaid_definition: str = "",
    context: str = "",
    level: str = "beginner",
) -> str:
    """
    Generate an educational diagram for `query`.

    Parameters
    ----------
    query               : Topic or question string.
    mermaid_definition  : Mermaid code from tutor_llm.generate_diagram_prompt().
                          If empty, a generic definition is constructed.
    context             : RAG context (used only for HF image fallback).
    level               : 'beginner' | 'intermediate' | 'advanced'

    Returns
    -------
    File path of the saved diagram (SVG or PNG).
    """
    label = query[:40].replace(" ", "_").lower()

    # Build a minimal Mermaid definition if none provided
    if not mermaid_definition.strip():
        mermaid_definition = textwrap.dedent(f"""\
            flowchart TD
                TOPIC["{query}"]
                TOPIC --> C1["Key Concept 1"]
                TOPIC --> C2["Key Concept 2"]
                TOPIC --> C3["Key Concept 3"]
                C1 --> EX1["Example"]
                C2 --> EX2["Example"]
        """)

 
    path = _render_mermaid_svg(mermaid_definition, label)
    if path:
        return path

    path = _render_hf_image(query, level, label)
    if path:
        return path

    return _render_matplotlib(query, mermaid_definition, label)