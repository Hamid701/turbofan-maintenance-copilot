"""Render the README architecture diagram in a light and a dark variant.

Run from the repository root:  python docs/build_architecture.py

Both files are identical except for the palette in the <style> block, so the two
themes cannot drift apart.
"""

from pathlib import Path

Palette = dict[str, str]

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'

PALETTES = {
    "light": dict(bg="#f6f8fa", bg_line="#d1d9e0", box="#ffffff", line="#d1d9e0",
                  frame="#8c959f", text="#1f2328", muted="#59636e",
                  accent="#0b6e8f", accent_fill="#e6f3f7",
                  warn="#9a6700", warn_fill="#fff8c5", warn_text="#7d4e00"),
    "dark": dict(bg="#161b22", bg_line="#30363d", box="#0d1117", line="#3d444d",
                 frame="#6e7681", text="#e6edf3", muted="#9198a1",
                 accent="#4cb3d4", accent_fill="#0c2d38",
                 warn="#d29922", warn_fill="#2e2100", warn_text="#e3b341"),
}

W, H = 960, 600


def style(p: Palette) -> str:
    return f"""
    .bg {{ fill: {p['bg']}; stroke: {p['bg_line']}; }}
    .box {{ fill: {p['box']}; stroke: {p['line']}; stroke-width: 1; }}
    .frame {{ fill: none; stroke: {p['frame']}; stroke-width: 1.2; stroke-dasharray: 6 5; }}
    .llm {{ fill: {p['accent_fill']}; stroke: {p['accent']}; stroke-width: 1.6; }}
    .pill {{ fill: {p['warn_fill']}; stroke: {p['warn']}; stroke-width: 1.2; stroke-dasharray: 4 3; }}
    text {{ font-family: {FONT}; }}
    .t {{ fill: {p['text']}; font-size: 13px; font-weight: 600; }}
    .s {{ fill: {p['muted']}; font-size: 11px; }}
    .band {{ fill: {p['muted']}; font-size: 11px; font-weight: 600; letter-spacing: 0.08em; }}
    .lbl {{ fill: {p['muted']}; font-size: 11px; }}
    .t-accent {{ fill: {p['accent']}; font-size: 13px; font-weight: 600; }}
    .s-accent {{ fill: {p['accent']}; font-size: 11px; }}
    .t-warn {{ fill: {p['warn_text']}; font-size: 12px; font-weight: 600; }}
    .s-warn {{ fill: {p['warn_text']}; font-size: 11px; }}
    .lbl-accent {{ fill: {p['accent']}; font-size: 11px; }}
    .edge {{ stroke: {p['muted']}; stroke-width: 1.4; fill: none; }}
    .edge-accent {{ stroke: {p['accent']}; stroke-width: 1.6; fill: none; }}
    .edge-warn {{ stroke: {p['warn']}; stroke-width: 1.4; stroke-dasharray: 4 3; fill: none; }}
    .head {{ fill: {p['muted']}; }}
    .head-accent {{ fill: {p['accent']}; }}
    .head-warn {{ fill: {p['warn']}; }}
    """


def marker(name: str, cls: str) -> str:
    return (
        f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" class="{cls}"/></marker>'
    )


def box(
    x: float, y: float, w: float, h: float, title: str,
    subs: tuple[str, ...] | list[str] = (), cls: str = "box", tcls: str = "t", scls: str = "s",
) -> str:
    cx = x + w / 2
    lines = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="{cls}"/>']
    count = 1 + len(subs)
    first = y + h / 2 - (count - 1) * 8 + 4
    lines.append(f'<text x="{cx}" y="{first}" text-anchor="middle" class="{tcls}">{title}</text>')
    for i, sub in enumerate(subs, start=1):
        lines.append(
            f'<text x="{cx}" y="{first + 16 * i}" text-anchor="middle" class="{scls}">{sub}</text>'
        )
    return "\n".join(lines)


def edge(x1: float, y1: float, x2: float, y2: float, cls: str = "edge", head: str = "arrow") -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{cls}" marker-end="url(#{head})"/>'


def text(x: float, y: float, value: str, cls: str = "lbl", anchor: str = "start") -> str:
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="{cls}">{value}</text>'


def render(p: Palette) -> str:
    row_y, row_h = 208, 70
    mid = row_y + row_h / 2
    parts = [f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="14" class="bg"/>']

    # Online: one question
    parts.append(text(24, 40, "EVERY QUESTION, ONLINE", "band"))
    parts.append('<rect x="150" y="168" width="800" height="132" rx="12" class="frame"/>')
    parts.append(text(166, 190, "API CONTAINER (FASTAPI)", "band"))

    parts.append(box(20, row_y, 80, row_h, "Browser", ["chat page"]))
    parts.append(box(160, row_y, 100, row_h, "FastAPI", ["routes + SSE"]))
    parts.append(box(300, row_y, 130, row_h, "Retrieve", ["lexical + BGE, RRF"]))
    parts.append(box(490, row_y, 130, row_h, "Engine tools", ["trend + RUL model", "if engine named"]))
    parts.append(box(680, row_y, 110, row_h, "Generate", ["structured reply"]))
    parts.append(box(840, row_y, 100, row_h, "Cite", ["numbers to pages"]))

    parts.append(edge(100, mid - 11, 160, mid - 11))
    parts.append(text(125, mid - 19, "ask", anchor="middle"))
    parts.append(edge(160, mid + 11, 100, mid + 11))
    parts.append(text(125, mid + 27, "answer", anchor="middle"))

    parts.append(edge(260, mid, 300, mid))
    parts.append(edge(430, mid, 490, mid))
    parts.append(text(460, mid + 19, "passages", anchor="middle"))
    parts.append(edge(620, mid, 680, mid))
    parts.append(text(650, mid + 19, "+ report", anchor="middle"))
    parts.append(edge(790, mid, 840, mid))
    parts.append(text(815, mid + 19, "numbers", anchor="middle"))

    # The one non-deterministic hop
    parts.append(box(660, 64, 110, 60, "OpenAI API", ["gpt-4o-mini"],
                     cls="llm", tcls="t-accent", scls="s-accent"))
    parts.append(edge(700, row_y, 700, 124, cls="edge-accent", head="arrow-accent"))
    parts.append(text(694, 152, "prompt", "lbl-accent", "end"))
    parts.append(edge(730, 124, 730, row_y, cls="edge-accent", head="arrow-accent"))
    parts.append(text(736, 152, "reply", "lbl-accent"))

    # Refusal exits
    parts.append(box(390, 92, 140, 40, "Refuse", ["nothing retrieved"],
                     cls="pill", tcls="t-warn", scls="s-warn"))
    parts.append(edge(460, mid, 460, 132, cls="edge-warn", head="arrow-warn"))
    parts.append(box(800, 92, 140, 40, "Refuse", ["model declines"],
                     cls="pill", tcls="t-warn", scls="s-warn"))
    parts.append(edge(815, mid, 815, 132, cls="edge-warn", head="arrow-warn"))

    # Database
    parts.append('<rect x="150" y="360" width="800" height="76" rx="10" class="box"/>')
    parts.append(text(166, 384, "PostgreSQL 17 + pgvector", "t"))
    parts.append(text(166, 416, "query runs + feedback", "s"))
    parts.append(text(365, 416, "chunks + vectors (HNSW)", "s", "middle"))
    parts.append(text(555, 416, "sensor readings, RUL targets", "s", "middle"))
    parts.append(text(934, 416, "one database, 36 MB", "s", "end"))

    parts.append(edge(210, 278, 210, 360))
    parts.append(text(216, 334, "logs each query"))
    parts.append(edge(365, 360, 365, 278))
    parts.append(text(371, 334, "loaded at startup"))
    parts.append(edge(555, 360, 555, 278))
    parts.append(text(561, 334, "sensor rows"))

    # Offline: build once. Each writer sits directly beneath the reader of the same
    # table, so every database section is one straight column: in from below, out above.
    parts.append(text(24, 482, "BUILD ONCE, OFFLINE", "band"))
    parts.append(box(20, 500, 120, 60, "FAA handbook", ["3 PDF chapters"]))
    parts.append(box(170, 500, 100, 60, "Chunks", ["700 chars", "one page each"]))
    parts.append(box(300, 500, 130, 60, "BGE-small", ["384-d vectors"]))
    parts.append(box(490, 500, 130, 60, "NASA C-MAPSS", ["FD001 trajectories"]))
    parts.append(edge(140, 530, 170, 530))
    parts.append(edge(270, 530, 300, 530))
    parts.append(edge(365, 500, 365, 436))
    parts.append(text(371, 472, "upsert"))
    parts.append(edge(555, 500, 555, 436))
    parts.append(text(561, 472, "upsert"))
    parts.append(text(934, 534, "scripts, run from the host", "s", "end"))

    claim = (
        "One question's path through the copilot. Every step is deterministic code except the "
        "highlighted OpenAI call, which writes the prose and picks passage numbers. Citations "
        "and engine figures come from code and data. Dashed exits are the two refusal points."
    )
    defs = "".join([
        marker("arrow", "head"),
        marker("arrow-accent", "head-accent"),
        marker("arrow-warn", "head-warn"),
    ])
    body = "\n".join(parts)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-labelledby="title desc">\n'
        "<!-- Generated. The light and dark variants differ only in the palette below. -->\n"
        '<title id="title">Turbofan Maintenance Copilot architecture</title>\n'
        f'<desc id="desc">{claim}</desc>\n'
        f"<style>{style(p)}</style>\n"
        f"<defs>{defs}</defs>\n"
        f"{body}\n"
        "</svg>\n"
    )


out = Path(__file__).resolve().parent
for name, palette in PALETTES.items():
    path = out / f"architecture-{name}.svg"
    path.write_text(render(palette), encoding="utf-8")
    print("wrote", path)
