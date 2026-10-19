# Copyright (C) 2026 Jean-François Brisson, Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Build the static GitHub Pages view of the SYNTHETIC replay.

Usage (from the repository root, with the pinned pilottrace checkout in .deps/pilottrace):

    python scripts/build_site.py _site

Every scenario is replayed at build time through the same code as the CLI, including the
unmodified, hash-checked pilottrace scoring. The page shows, per scenario, the navigation error
over the route, the labelled event windows and the frames that pilottrace latched. The output is
one self-contained index.html: no JavaScript, no external requests, no fonts or trackers.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pilottrace_nav import __version__  # noqa: E402
from pilottrace_nav._deps import PILOTTRACE_COMMIT  # noqa: E402
from pilottrace_nav.integrations.oes_adapter import frames_from_residuals  # noqa: E402
from pilottrace_nav.provenance.evidence_report import BOUNDARIES  # noqa: E402
from pilottrace_nav.replay import run_scenario  # noqa: E402
from pilottrace_nav.scenarios import SCENARIOS, SimConfig, simulate  # noqa: E402

TITLE = "PilotTrace Navigation Replay Pack"
W, H, PAD_L, PAD_R, PAD_T, STRIP = 760, 170, 46, 12, 14, 18


def scenario_data(name: str, config: SimConfig) -> dict:
    """Replay one scenario and return what the page needs (all SYNTHETIC)."""
    spec = SCENARIOS[name]
    records = simulate(spec, config)
    frames = frames_from_residuals([r.residuals for r in records])
    result = run_scenario(spec, config)
    report = json.loads(result["report_bytes"])
    flagged = [t for (t, _), fr in zip(frames, report["frames"], strict=True) if fr["flagged"]]
    return {
        "name": name,
        "spec": spec,
        "errors": [r.nav_error_m for r in records],
        "flagged": flagged,
        "summary": result["summary"],
    }


def _chart(data: dict, epochs: int) -> str:
    errors = data["errors"]
    top = max(errors)
    ceiling = max(5.0, top * 1.1)
    plot_w, plot_h = W - PAD_L - PAD_R, H - PAD_T - STRIP - 22

    def x(t: float) -> float:
        return round(PAD_L + t / max(1, epochs - 1) * plot_w, 2)

    def y(v: float) -> float:
        return round(PAD_T + plot_h - v / ceiling * plot_h, 2)

    parts = []
    for event in data["spec"].events:
        end = min(event.end, epochs) - 1
        parts.append(
            f'<rect class="event" x="{x(event.start)}" y="{PAD_T}" width="{round(x(end) - x(event.start), 2)}" '
            f'height="{plot_h}"><title>{html.escape(event.kind)}: epochs {event.start} to {event.end}</title></rect>'
        )
    for frac in (0.0, 0.5, 1.0):
        value = ceiling * frac
        parts.append(f'<line class="grid" x1="{PAD_L}" x2="{W - PAD_R}" y1="{y(value)}" y2="{y(value)}"/>')
        parts.append(f'<text class="tick" x="{PAD_L - 6}" y="{y(value) + 4}">{value:.0f} m</text>')
    points = " ".join(f"{x(t)},{y(v)}" for t, v in enumerate(errors))
    parts.append(f'<polyline class="err" points="{points}"/>')
    strip_y = PAD_T + plot_h + 6
    parts.append(f'<rect class="strip" x="{PAD_L}" y="{strip_y}" width="{plot_w}" height="{STRIP - 6}"/>')
    for t in data["flagged"]:
        parts.append(f'<rect class="latch" x="{x(t)}" y="{strip_y}" width="1.6" height="{STRIP - 6}"/>')
    parts.append(f'<text class="tick" x="{PAD_L - 6}" y="{strip_y + 10}">latch</text>')
    parts.append(f'<text class="axis" x="{PAD_L}" y="{H - 4}">epoch 0</text>')
    parts.append(f'<text class="axis end" x="{W - PAD_R}" y="{H - 4}">epoch {epochs - 1} (1 Hz)</text>')
    summary = data["summary"]
    desc = (
        f"Navigation error peaks at {top:.1f} m. {len(data['spec'].events)} labelled event window(s). "
        f"{summary['frames_latched']} of {summary['frames']} frames latched; "
        f"{summary['false_alarm_frames']} latched frames fall outside labelled windows."
    )
    cid = f"c-{data['name']}"
    return (
        f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-labelledby="{cid}-t {cid}-d">'
        f'<title id="{cid}-t">{html.escape(summary["title"])}, SYNTHETIC</title>'
        f'<desc id="{cid}-d">{html.escape(desc)}</desc>{"".join(parts)}</svg>'
    )


CSS = """
:root{--navy:#0d1b2a;--panel:#13263a;--ink:#e8eef4;--muted:#a9b8c6;--gold:#c9a84c;--cyan:#4cc9e0;--latch:#f08a7e;--line:#23405c}
*{box-sizing:border-box}
body{margin:0;background:var(--navy);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
a{color:var(--cyan)}a:focus-visible,summary:focus-visible{outline:2px solid var(--gold);outline-offset:2px}
.skip{position:absolute;left:-999px}.skip:focus{left:1rem;top:1rem;background:var(--gold);color:var(--navy);padding:.4rem .7rem}
header,main,footer{max-width:980px;margin:0 auto;padding:1.2rem}
header{padding-top:2.4rem}
h1{font-size:2rem;margin:.25rem 0;color:var(--gold)}
h2{font-size:1.3rem;margin:2.2rem 0 .6rem;color:var(--gold)}
h3{font-size:1.05rem;margin:0}
.tag{display:inline-block;font-size:.75rem;font-weight:700;letter-spacing:.08em;border:1px solid var(--cyan);color:var(--cyan);border-radius:4px;padding:.05rem .45rem}
.lede{color:var(--muted);max-width:48rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:1rem 1.1rem;margin:.9rem 0}
.meta{color:var(--muted);font-size:.9rem;margin:.25rem 0 .5rem}
.chart{width:100%;height:auto;display:block}
.chart .event{fill:var(--gold);opacity:.18}
.chart .grid{stroke:var(--line)}
.chart .tick{fill:var(--muted);font-size:11px;text-anchor:end}
.chart .axis{fill:var(--muted);font-size:11px}.chart .end{text-anchor:end}
.chart .err{fill:none;stroke:var(--cyan);stroke-width:1.8;stroke-linejoin:round}
.chart .strip{fill:#0a1522;stroke:var(--line)}
.chart .latch{fill:var(--latch)}
.legend{display:flex;flex-wrap:wrap;gap:1.2rem;color:var(--muted);font-size:.9rem;margin:.4rem 0 0}
.sw{display:inline-block;width:.9rem;height:.9rem;border-radius:2px;vertical-align:-2px;margin-right:.35rem}
.miss{color:var(--latch);font-weight:700}
table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{text-align:left;border-bottom:1px solid var(--line);padding:.4rem}
th{color:var(--muted);font-weight:600}
ul.b{margin:.3rem 0;padding-left:1.1rem}
footer{color:var(--muted);font-size:.85rem;border-top:1px solid var(--line);margin-top:2.5rem}
"""


def _event_text(summary: dict) -> str:
    if not summary["events"]:
        return "No labelled event."
    bits = []
    for event in summary["events"]:
        kind = html.escape(str(event.get("kind", "event")))
        if event.get("detected_during_event"):
            latency = event.get("detection_latency_s")
            verdict = "latched during event" + (f" after {latency:.0f} s" if latency else " at once")
        else:
            verdict = '<span class="miss">NOT latched during event</span>'
        bits.append(f"{kind}: {verdict}")
    return "; ".join(bits) + "."


def render(config: SimConfig | None = None, names: list[str] | None = None) -> str:
    config = config or SimConfig()
    names = names or list(SCENARIOS)
    cards = []
    rows = []
    for name in names:
        data = scenario_data(name, config)
        s = data["summary"]
        cards.append(
            f'<article class="card" id="{html.escape(name)}"><h3>{html.escape(s["title"])} '
            f'<code>{html.escape(name)}</code></h3><p class="meta">{html.escape(s["description"])}</p>'
            f'{_chart(data, config.epochs)}<p class="meta">{_event_text(s)} '
            f"Max navigation error {s['nav_error_m']['max']:.1f} m. "
            f"Cross-check: {s['cross_check']['latch_disagreements']} latch disagreements between pilottrace "
            f"and the fail-closed adapter.</p></article>"
        )
        rows.append(
            f'<tr><td><a href="#{html.escape(name)}">{html.escape(name)}</a></td>'
            f"<td>{s['frames_latched']} / {s['frames']}</td><td>{s['false_alarm_frames']}</td>"
            f"<td>{s['nav_error_m']['max']:.1f} m</td></tr>"
        )
    boundaries = "".join(f"<li>{html.escape(b)}</li>" for b in BOUNDARIES)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src 'self' data:">
<meta name="referrer" content="no-referrer">
<title>{TITLE}: SYNTHETIC replay</title>
<meta name="description" content="Deterministic SYNTHETIC GNSS-denied maritime navigation replay scored through pilottrace. Research prototype by Jean-François Brisson, Spark AI NLP.">
<style>{CSS}</style>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header>
<span class="tag">SYNTHETIC</span> <span class="tag">RESEARCH PROTOTYPE</span>
<h1>{TITLE}</h1>
<p class="lede">A deterministic software replay of GNSS-denied maritime navigation (seed {config.seed},
{config.epochs} epochs at 1 Hz). Sixteen navigation residual channels become 16 x 32 = 512-value frames, scored by the
unmodified pilottrace package (commit <code>{PILOTTRACE_COMMIT[:12]}</code>) with the latch
S = 0.45 Peak + 0.35 RMS + 0.20 MeanAbs &ge; {config.tau}. This page was generated at build time by the same code as
the CLI (version {__version__}).</p>
</header>
<main id="main">
<h2>Boundaries</h2>
<div class="card"><ul class="b">{boundaries}</ul></div>

<h2>Summary</h2>
<div class="card"><table><caption class="meta">All values SYNTHETIC</caption>
<thead><tr><th scope="col">Scenario</th><th scope="col">Frames latched</th><th scope="col">Latched outside labelled windows</th><th scope="col">Max navigation error</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>

<h2>Scenarios</h2>
<p class="legend"><span><span class="sw" style="background:#4cc9e0"></span>navigation error (m)</span>
<span><span class="sw" style="background:#c9a84c;opacity:.5"></span>labelled event window</span>
<span><span class="sw" style="background:#f08a7e"></span>frame latched by pilottrace</span></p>
{"".join(cards)}
<p class="lede">Read the misses as carefully as the hits: the slow, self-consistent spoof is not caught while it is
active. These are properties of invented scenarios, declared scales and a toy estimator; they say nothing about real
receivers, vessels or navigation systems.</p>
</main>
<footer>
<p>{TITLE} {__version__}. Copyright (c) 2026 Jean-François Brisson, Spark AI NLP. AGPL-3.0-only.
Source: <a href="https://github.com/sparkainlp-x/pilottrace-nav-replay">github.com/sparkainlp-x/pilottrace-nav-replay</a>.
This page makes no network requests and uses no JavaScript.</p>
</footer>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    out_dir = Path(args[0]) if args else ROOT / "_site"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(render(), encoding="utf-8")
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")
    print(f"wrote {out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
