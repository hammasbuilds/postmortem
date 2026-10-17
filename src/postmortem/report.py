"""Incident report for one agent run: Markdown, HTML and PDF, no dependencies.

Why this product emits a PDF. A postmortem is an artefact that outlives the
terminal it came from: it gets attached to an incident ticket, mailed to whoever
owns the agent, and read by someone who was not there. Fixed pagination and a
format that cannot reflow is the point.

Why it can be hand-written. The content is Latin-script, so it uses the standard
PDF fonts every reader already has - no font file to embed, no shaping. The
writer itself is the same small one Clause uses; both products keep their own
copy rather than sharing a library across repos.

Product rule: the report names the product and the behaviour, never the
internals of how the attribution was produced.
"""

from __future__ import annotations

import datetime as _dt
import re
import zlib

PAGE_W, PAGE_H = 595, 842  # A4 in points
MARGIN = 54
LEADING = 14
FONT_BODY, FONT_BOLD = "F1", "F2"


def _esc(text: str) -> str:
    out = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    # The standard fonts are Latin-1; anything outside it becomes a plain marker
    # rather than a mojibake byte.
    return "".join(ch if ord(ch) < 256 else "?" for ch in out)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = f"{cur} {w}".strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


class _Pdf:
    """A deliberately small PDF writer: pages of text lines in two fonts."""

    def __init__(self) -> None:
        self.pages: list[list[tuple[str, int, str]]] = [[]]  # (font, size, text)

    def line(self, text: str = "", *, size: int = 10, bold: bool = False) -> None:
        width = int((PAGE_W - 2 * MARGIN) / (size * 0.52))
        for part in _wrap(text, width) if text else [""]:
            if len(self.pages[-1]) >= int((PAGE_H - 2 * MARGIN) / LEADING):
                self.pages.append([])
            self.pages[-1].append((FONT_BOLD if bold else FONT_BODY, size, part))

    def bytes(self) -> bytes:
        objects: list[bytes] = []

        def add(payload: bytes) -> int:
            objects.append(payload)
            return len(objects)

        font_body = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
        font_bold = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")

        page_ids: list[int] = []
        content_ids: list[int] = []
        for page in self.pages:
            chunks = ["BT"]
            y = PAGE_H - MARGIN
            for font, size, text in page:
                chunks.append(f"/{font} {size} Tf 1 0 0 1 {MARGIN} {y} Tm ({_esc(text)}) Tj")
                y -= LEADING
            chunks.append("ET")
            stream = zlib.compress("\n".join(chunks).encode("latin-1", "replace"))
            content_ids.append(
                add(b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(stream) + stream + b"\nendstream")
            )
            page_ids.append(0)  # filled once the Pages object exists

        pages_id = len(objects) + len(self.pages) + 1
        for i, content_id in enumerate(content_ids):
            page_ids[i] = add(
                f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
                f"/Resources << /Font << /{FONT_BODY} {font_body} 0 R /{FONT_BOLD} {font_bold} 0 R >> >> "
                f"/Contents {content_id} 0 R >>".encode("latin-1")
            )
        kids = " ".join(f"{pid} 0 R" for pid in page_ids)
        pages_obj = add(f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>".encode("latin-1"))
        catalog = add(f"<< /Type /Catalog /Pages {pages_obj} 0 R >>".encode("latin-1"))

        out = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for i, payload in enumerate(objects, 1):
            offsets.append(len(out))
            out += f"{i} 0 obj\n".encode("latin-1") + payload + b"\nendobj\n"
        xref_at = len(out)
        out += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
        out += b"0000000000 65535 f \n"
        for off in offsets[1:]:
            out += f"{off:010d} 00000 n \n".encode("latin-1")
        out += (
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
        ).encode("latin-1")
        return bytes(out)


def _title_for(data: dict, name: str) -> str:
    return name or f"{data.get('task', 'agent run')} / {data.get('policy', 'run')}"


def _summary_rows(data: dict) -> list[tuple[str, str]]:
    a = data.get("attribution", {})
    step = a.get("breaking_step")
    return [
        ("Outcome", data.get("outcome", "unknown")),
        ("Diagnosis", (a.get("label_kind") or "").replace("_", " ")),
        ("Fault class", (a.get("fault_class") or "none").replace("_", " ")),
        ("Point of no return", "none - nothing was damaged" if step is None else f"step {step}"),
        ("First wrong move",
         "none" if a.get("first_wrong_step") is None else f"step {a.get('first_wrong_step')}"),
        ("Confidence", f"{round(100 * (a.get('confidence') or 0))}%"),
        ("Evidence used", a.get("layer", "")),
        ("Steps in the run", str(len(data.get("steps", [])))),
    ]


def markdown(data: dict, *, name: str = "", min_agreement: int = 0) -> str:
    a = data.get("attribution", {})
    lines = [f"# Postmortem - {_title_for(data, name)}", "", f"Prepared by Postmortem · {_dt.date.today().isoformat()}", ""]
    for k, v in _summary_rows(data):
        lines.append(f"- **{k}:** {v}")
    lines += ["", "## What to change", "", a.get("advice", "(no advice)"), "", "## Why", ""]
    for e in a.get("evidence", []):
        bits = ", ".join(f"{k}: {v}" for k, v in e.items() if k != "rule")
        lines.append(f"- {e.get('rule', '')}" + (f" ({bits})" if bits else ""))
    lines += ["", "## The run, step by step", ""]
    for n in a.get("step_notes", []):
        mark = "FAILED" if not n["ok"] else n["effect"]
        extra = f" — {n['error']}" if n.get("error") else ""
        repeat = f" (repeat {n['repeat']})" if n.get("repeat", 1) > 1 else ""
        lines.append(f"{n['index']}. `{n['tool']}` — {mark}{repeat}{extra}")
    if a.get("neighbours"):
        lines += ["", "## Similar runs seen before", ""]
        for nb in a["neighbours"][:5]:
            lines.append(
                f"- {nb['task']} ({nb.get('fault_from_replay') or 'unlabelled'}), "
                f"diagnosis {nb.get('label_kind', '')}, similarity {nb.get('bm25', 0)}"
            )
    return "\n".join(lines) + "\n"


def pdf(data: dict, *, name: str = "", min_agreement: int = 0) -> bytes:
    a = data.get("attribution", {})
    doc = _Pdf()
    doc.line(f"Postmortem - {_title_for(data, name)}", size=16, bold=True)
    doc.line(f"Prepared by Postmortem  ·  {_dt.date.today().isoformat()}", size=9)
    doc.line()
    for k, v in _summary_rows(data):
        doc.line(f"  {k}: {v}", size=10)
    doc.line()
    doc.line("What to change", size=13, bold=True)
    doc.line(f"  {a.get('advice', '')}", size=10)
    doc.line()
    doc.line("Why", size=13, bold=True)
    for e in a.get("evidence", []):
        doc.line(f"  -  {e.get('rule', '')}", size=10)
        bits = ", ".join(f"{k}: {v}" for k, v in e.items() if k != "rule")
        if bits:
            doc.line(f"     {bits}", size=9)
    doc.line()
    doc.line("The run, step by step", size=13, bold=True)
    for n in a.get("step_notes", []):
        mark = "FAILED" if not n["ok"] else n["effect"]
        repeat = f" (repeat {n['repeat']})" if n.get("repeat", 1) > 1 else ""
        doc.line(f"  {n['index']:>2}. {n['tool']} - {mark}{repeat}", size=10)
        if n.get("error"):
            doc.line(f"      {n['error']}", size=9)
    if a.get("neighbours"):
        doc.line()
        doc.line("Similar runs seen before", size=13, bold=True)
        for nb in a["neighbours"][:5]:
            doc.line(f"  -  {nb['task']}: {nb.get('fault_from_replay') or 'unlabelled'} "
                     f"(similarity {nb.get('bm25', 0)})", size=9)
    return doc.bytes()


_TAG = re.compile(r"<[^>]+>")


def html(data: dict, *, name: str = "", min_agreement: int = 0) -> str:
    md = markdown(data, name=name)
    body = []
    for line in md.splitlines():
        if line.startswith("## "):
            body.append(f"<h2>{_TAG.sub('', line[3:])}</h2>")
        elif line.startswith("# "):
            body.append(f"<h1>{_TAG.sub('', line[2:])}</h1>")
        elif line.startswith("- ") or line[:2].rstrip().isdigit():
            body.append(f"<li>{line.lstrip('-0123456789. ')}</li>")
        elif line.strip():
            body.append(f"<p>{line}</p>")
    return (
        "<!doctype html><meta charset='utf-8'><title>Postmortem</title>"
        "<style>body{font:14px/1.6 ui-sans-serif,system-ui,sans-serif;max-width:46em;margin:40px auto;padding:0 16px}"
        "h1{font-size:22px}h2{font-size:16px;margin-top:26px;border-bottom:1px solid #ddd}"
        "li{margin:3px 0}code{background:#f3f4f6;padding:1px 4px;border-radius:4px}"
        "@media print{body{margin:0}}</style>" + "\n".join(body)
    )
