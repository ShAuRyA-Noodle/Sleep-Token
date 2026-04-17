"""Build a styled HTML version of PITCH_DECK.md (can be printed to PDF via browser).

Avoids pandoc/LaTeX dependency. Output at demo/SupplyMind_pitch.html — open in any
browser and "Print → Save as PDF" to produce the final slide deck.
"""
from __future__ import annotations

from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "demo" / "PITCH_DECK.md"
OUT = ROOT / "demo" / "SupplyMind_pitch.html"

STYLE = """
<style>
  @page { size: A4 landscape; margin: 1.5cm; }
  body { font-family: "SF Pro Display", "Segoe UI", -apple-system, sans-serif;
         max-width: 1100px; margin: 2em auto; padding: 2em;
         color: #1a1a1a; line-height: 1.55; }
  h1 { color: #1a5490; font-size: 2.2em; border-bottom: 4px solid #1a5490;
       padding-bottom: 0.2em; page-break-after: avoid; }
  h2 { color: #2b6cb0; font-size: 1.5em; margin-top: 2em;
       page-break-before: always; page-break-after: avoid; }
  h3 { color: #333; page-break-after: avoid; }
  hr { border: 0; border-top: 2px dashed #ccc; margin: 3em 0; page-break-after: always; }
  table { border-collapse: collapse; margin: 1em 0; width: 100%; font-size: 0.9em; }
  th { background: #1a5490; color: white; padding: 0.6em; text-align: left; }
  td { border-bottom: 1px solid #ddd; padding: 0.5em; }
  tr:nth-child(even) { background: #f8f8f8; }
  code { background: #f0f0f0; padding: 0.2em 0.4em; border-radius: 3px;
         font-family: "SF Mono", Monaco, Consolas, monospace; font-size: 0.9em; }
  pre { background: #1a1a1a; color: #eee; padding: 1em; border-radius: 6px;
        overflow-x: auto; page-break-inside: avoid; }
  blockquote { border-left: 4px solid #1a5490; padding-left: 1em;
               color: #555; font-style: italic; }
  strong { color: #1a5490; }
  a { color: #2b6cb0; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px;
           font-size: 0.8em; margin-right: 4px; }
  .badge-s { background: #1a5490; color: white; }
  .badge-a { background: #38a169; color: white; }
  ul, ol { padding-left: 1.5em; }
  li { margin: 0.3em 0; }
</style>
"""

HEADER = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>SupplyMind v3.0-arcadia — Pitch Deck</title>"""

FOOTER = """
<hr><p style="text-align:center; color:#888; font-size:0.85em;">
  SupplyMind v3.0-arcadia · Meta PyTorch OpenEnv Hackathon 2026 · MIT License ·
  <a href="https://github.com/ShAuRyA-Noodle/Sleep-Token">github.com/ShAuRyA-Noodle/Sleep-Token</a>
</p></body></html>"""


def main():
    md_text = MD.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
    )
    full = HEADER + STYLE + "</head><body>" + html_body + FOOTER
    OUT.write_text(full, encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"Size: {OUT.stat().st_size / 1024:.1f} KB")
    print(f"\nTo generate PDF:")
    print(f"  1. Open {OUT} in Chrome/Edge/Firefox")
    print(f"  2. Cmd/Ctrl+P → Destination: Save as PDF → landscape, margins: default")
    print(f"  3. Save as demo/SupplyMind_pitch.pdf")


if __name__ == "__main__":
    main()
