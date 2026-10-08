#!/usr/bin/env python3
"""Render the actual synthetic --demo output as a dependency-free README SVG."""
import contextlib
import html
import io
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ty

class Terminal(io.StringIO):
    def isatty(self):
        return True

stream = Terminal()
ui = ty.UI('yes', stream)
ui.color = True
with contextlib.redirect_stderr(stream):
    ty.demo(SimpleNamespace(**ty.DEFAULTS), ui)
captured = re.sub(r'\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)', '', stream.getvalue())
lines = captured.strip('\n').splitlines()
colors = {'0': '#e2e9e6', '32': '#a2d7ac', '36': '#8bd5ca', '90': '#869c97', '31': '#f09791', '33': '#ead29a'}
width, height = 940, 76 + len(lines) * 23
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="ty terminal interface, illustrative demo">',
       '<title>ty — tiny, at your terminal</title>',
       '<desc>Actual synthetic demo output showing compact tools, formatted answers, and command hints. Timings are illustrative.</desc>',
       f'<rect width="{width}" height="{height}" rx="16" fill="#101b1d"/>',
       f'<path d="M16 0H{width - 16}Q{width} 0 {width} 16V44H0V16Q0 0 16 0" fill="#1d2a2b"/>',
       '<circle cx="24" cy="23" r="5" fill="#ef8c83"/><circle cx="42" cy="23" r="5" fill="#e6c789"/><circle cx="60" cy="23" r="5" fill="#9ccba5"/>',
       '<text x="470" y="28" fill="#a2b5ae" text-anchor="middle" font-family="monospace" font-size="12">ty · local agent</text>']
for i, line in enumerate(lines):
    color, bold = colors['0'], False
    x = 24
    for chunk in re.split(r'(\x1b\[[0-9;]*m)', line):
        if chunk.startswith('\x1b['):
            for code in chunk[2:-1].split(';'):
                if code in ('', '0'):
                    color, bold = colors['0'], False
                elif code == '1':
                    bold = True
                elif code in colors:
                    color = colors[code]
        elif chunk:
            weight = '700' if bold else '400'
            svg.append(f'<text x="{x:.1f}" y="{72 + i * 23}" fill="{color}" font-weight="{weight}" font-family="DejaVu Sans Mono, monospace" font-size="15" xml:space="preserve">{html.escape(chunk)}</text>')
            x += len(chunk) * 9.03
svg.append('</svg>')
(ROOT / 'docs' / 'terminal.svg').write_text('\n'.join(svg) + '\n')
print('docs/terminal.svg')
