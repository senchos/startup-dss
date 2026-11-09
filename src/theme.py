"""Shared visual language for the Streamlit UI (app.py): CSS injection, status
pills/chips, component bars and an inline-SVG radar chart. Kept separate from
app.py so page functions stay focused on data flow, not markup."""
import math

CATEGORY_STATUS = {"низька": "critical", "помірна": "warning", "висока": "good", "дуже висока": "good"}

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
:root{
  --bg:#f9f9f7; --surface:#fcfcfb;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781;
  --border:rgba(11,11,11,0.10); --gridline:#e1e0d9;
  --accent:#4a3aa7; --accent-soft:rgba(74,58,167,0.10);
  --ml:#2a78d6; --fuzzy:#eb6834; --deal:#1baf7a;
  --good:#0ca30c; --warning:#fab219; --critical:#d03b3b;
  --good-soft:rgba(12,163,12,0.12); --warning-soft:rgba(250,178,25,0.16); --critical-soft:rgba(208,59,59,0.12);
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#0d0d0d; --surface:#1a1a19;
    --text-primary:#ffffff; --text-secondary:#c3c2b7; --muted:#898781;
    --border:rgba(255,255,255,0.10); --gridline:#2c2c2a;
    --accent:#9085e9; --accent-soft:rgba(144,133,233,0.14);
    --ml:#3987e5; --fuzzy:#d95926; --deal:#199e70;
    --good-soft:rgba(12,163,12,0.16); --warning-soft:rgba(250,178,25,0.14); --critical-soft:rgba(208,59,59,0.16);
  }
}

[data-testid="stAppViewContainer"], [data-testid="stSidebar"]{
  font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
}
.mono{font-family:"IBM Plex Mono","SFMono-Regular",Consolas,monospace}
.tab{font-variant-numeric:tabular-nums}

/* ---- sidebar nav ---- */
[data-testid="stSidebar"] [role="radiogroup"]{display:flex; flex-direction:column; gap:4px}
[data-testid="stSidebar"] [role="radiogroup"] label{border-radius:8px; padding:6px 10px; transition:background .12s ease}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:var(--accent-soft)}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:var(--accent-soft)}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p{color:var(--accent); font-weight:600}
[data-testid="stSidebar"] input[type="radio"]{accent-color:var(--accent)}

/* ---- cards ---- */
.card{background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:20px; min-width:0; margin-bottom:12px}
.eyebrow{font-size:11px; letter-spacing:.6px; text-transform:uppercase; color:var(--muted); font-weight:600; margin:0 0 10px}
.company-name{font-size:22px; font-weight:700; letter-spacing:-.2px; color:var(--text-primary)}

.tags{display:flex; flex-wrap:wrap; gap:6px; margin:10px 0 18px}
.tag{font-size:11px; color:var(--text-secondary); background:var(--bg); border:1px solid var(--border); padding:3px 9px; border-radius:100px}

.stat-grid{display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px 16px; margin-bottom:16px}
.stat{min-width:0}
.stat-label{font-size:10.5px; color:var(--muted); text-transform:uppercase; letter-spacing:.3px; font-family:"IBM Plex Sans",sans-serif}
.stat-value{font-size:15px; font-weight:600; margin-top:2px; color:var(--text-primary); font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums}

.deal-strip{display:flex; gap:18px; flex-wrap:wrap; padding:12px 14px; background:var(--bg); border:1px dashed var(--border); border-radius:10px; margin-bottom:18px}
.deal-strip .stat-value{font-size:14px}

/* ---- radar ---- */
.radar-svg text{fill:var(--text-secondary); font-family:"IBM Plex Sans",sans-serif; font-size:11px}
.radar-svg .grid-ring{fill:none; stroke:var(--gridline); stroke-width:1}
.radar-svg .spoke{stroke:var(--gridline); stroke-width:1}
.radar-svg .radar-fill{fill:var(--accent); fill-opacity:.16; stroke:var(--accent); stroke-width:2; stroke-linejoin:round}
.radar-svg .radar-dot{fill:var(--accent); stroke:var(--surface); stroke-width:2}

/* ---- score card ---- */
.score-row{display:flex; align-items:center; flex-wrap:wrap; gap:18px; margin-bottom:6px}
.score-value{font-size:48px; font-weight:700; line-height:1; color:var(--text-primary)}
.score-max{font-size:16px; color:var(--muted); font-weight:500}

.pill{display:inline-flex; align-items:center; gap:7px; padding:6px 13px; border-radius:100px; font-size:13px; font-weight:600; color:var(--text-primary)}
.pill .dot{width:8px; height:8px; border-radius:50%}
.pill.good{background:var(--good-soft)} .pill.good .dot{background:var(--good)}
.pill.warning{background:var(--warning-soft)} .pill.warning .dot{background:var(--warning)}
.pill.critical{background:var(--critical-soft)} .pill.critical .dot{background:var(--critical)}

.recommendation{font-size:13.5px; color:var(--text-secondary); line-height:1.55; margin:14px 0 22px; padding-left:14px; border-left:3px solid var(--good)}

.component{margin-bottom:14px}
.component-head{display:flex; align-items:baseline; justify-content:space-between; margin-bottom:6px}
.component-name{font-size:13px; font-weight:600; display:flex; align-items:center; gap:7px; color:var(--text-primary)}
.component-name .sw{width:10px; height:10px; border-radius:3px; flex-shrink:0}
.component-weight{font-size:11px; color:var(--muted); font-family:"IBM Plex Mono",monospace}
.component-value{font-size:13px; font-weight:600; color:var(--text-primary); font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums}
.track{height:10px; background:var(--gridline); border-radius:5px; overflow:hidden}
.fill{height:100%; border-radius:5px}

.formula{margin-top:14px; padding:10px 14px; background:var(--bg); border:1px solid var(--border); border-radius:8px; font-size:12px; color:var(--text-secondary); overflow-x:auto; white-space:nowrap}
.formula b{color:var(--text-primary)}

/* ---- history table ---- */
.table-wrap{overflow-x:auto; border:1px solid var(--border); border-radius:12px}
.table-wrap table{width:100%; border-collapse:collapse; font-size:13px; min-width:480px}
.table-wrap thead th{text-align:left; font-size:10.5px; text-transform:uppercase; letter-spacing:.3px; color:var(--muted); font-weight:600; padding:10px 14px; background:var(--surface); border-bottom:1px solid var(--border)}
.table-wrap tbody td{padding:11px 14px; border-bottom:1px solid var(--border); color:var(--text-secondary)}
.table-wrap tbody tr:last-child td{border-bottom:none}
.table-wrap tbody tr:hover{background:var(--accent-soft)}
.table-wrap td.name{color:var(--text-primary); font-weight:600}
.table-wrap td.score{font-family:"IBM Plex Mono",monospace; color:var(--text-primary); font-weight:600; font-variant-numeric:tabular-nums}

.chip{display:inline-flex; align-items:center; gap:6px; font-size:11.5px; font-weight:600; padding:3px 10px; border-radius:100px; color:var(--text-primary)}
.chip .dot{width:6px; height:6px; border-radius:50%}
.chip.good{background:var(--good-soft)} .chip.good .dot{background:var(--good)}
.chip.warning{background:var(--warning-soft)} .chip.warning .dot{background:var(--warning)}
.chip.critical{background:var(--critical-soft)} .chip.critical .dot{background:var(--critical)}
</style>
"""


def inject(st):
    st.markdown(CSS, unsafe_allow_html=True)


def tag(text):
    return f'<span class="tag">{text}</span>'


def stat(label, value):
    return f'<div class="stat"><div class="stat-label">{label}</div><div class="stat-value">{value}</div></div>'


def _status_badge(category, css_class):
    status = CATEGORY_STATUS.get(category, "warning")
    label = category.capitalize() if category else "—"
    return f'<span class="{css_class} {status}"><span class="dot"></span>{label}</span>'


def status_pill(category):
    return _status_badge(category, "pill")


def chip(category):
    return _status_badge(category, "chip")


def component_bar(label, color_var, value, weight=None):
    # Built as one unbroken string (no template line left to collapse to pure
    # whitespace when `weight` is None) -- see the note on render_score_card in
    # app.py: a blank/whitespace-only line inside an injected HTML block makes
    # Streamlit's markdown parser stop treating the rest of it as HTML.
    width = max(0.0, min(100.0, value))
    weight_html = f'<span class="component-weight">вага {weight:.2f}</span>' if weight is not None else ""
    return (
        '<div class="component">'
        '<div class="component-head">'
        f'<span class="component-name"><span class="sw" style="background:var({color_var})"></span>{label}</span>'
        f"{weight_html}"
        "</div>"
        f'<div class="track"><div class="fill" style="width:{width}%; background:var({color_var})"></div></div>'
        f'<div style="text-align:right; margin-top:3px"><span class="component-value">{value:.1f}</span></div>'
        "</div>"
    )


def _polar(cx, cy, r, angle_deg):
    rad = math.radians(angle_deg)
    return cx + r * math.cos(rad), cy + r * math.sin(rad)


def radar_svg(team_quality, traction, market_size, competition):
    """Inline SVG radar of the four qualitative inputs (0-10 scale)."""
    cx, cy, radius = 165, 140, 75
    angles = [-90, 0, 90, 180]
    values = [team_quality, traction, market_size, competition]
    labels = ["Команда", "Traction", "Ринок", "Конкуренція"]
    anchors = ["middle", "start", "middle", "end"]

    def scaled(angle, value):
        return _polar(cx, cy, radius * (value / 10), angle)

    rings = "".join(
        '<polygon class="grid-ring" points="{}"></polygon>'.format(
            " ".join(f"{x:.1f},{y:.1f}" for x, y in (scaled(a, v) for a in angles))
        )
        for v in (2, 4, 6, 8, 10)
    )
    spokes = "".join(
        '<line class="spoke" x1="{}" y1="{}" x2="{:.1f}" y2="{:.1f}"></line>'.format(cx, cy, *scaled(a, 10))
        for a in angles
    )
    data_pts = [scaled(a, v) for a, v in zip(angles, values)]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in data_pts)
    dots = "".join(f'<circle class="radar-dot" cx="{x:.1f}" cy="{y:.1f}" r="3.5"></circle>' for x, y in data_pts)

    label_pts = [_polar(cx, cy, radius + 28, a) for a in angles]
    label_tags = "".join(
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">{label} · {value:g}</text>'
        for (x, y), anchor, label, value in zip(label_pts, anchors, labels, values)
    )

    return (
        '<svg class="radar-svg" viewBox="-30 0 370 300" '
        'style="width:100%; height:auto; max-width:320px; display:block; margin:0 auto">'
        f"{rings}{spokes}"
        f'<polygon class="radar-fill" points="{poly}"></polygon>'
        f"{dots}{label_tags}"
        "</svg>"
    )
