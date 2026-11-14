"""Generate LaTeX tables from result JSONs.

Run from the repository root:
    python scripts/generate_tables.py

Outputs:
    generated/table_main_results.tex
    generated/table_deviations.tex

Paste them into the Overleaf project's generated/ folder.
"""
import json
from pathlib import Path

OUT = Path("generated")
OUT.mkdir(exist_ok=True)

ARMS = [
    ("A", "A (real only)"),
    ("C", "C (photometric)"),
    ("D", "D (276 synth.)"),
    ("D-138", "D-138 (138)"),
    ("D-bal", "D-bal (276 mix)"),
]

summary = json.loads(Path("outputs/week5_arms/summary.json").read_text())

def fmt(x):
    return f"{x:.4f}"

rows = []
# find max for bold
values = {arm: [
    summary[arm]["42"]["night_residential_recall"],
    summary[arm]["42"]["night_macro_f1"],
    summary[arm]["42"]["day_macro_f1"],
    summary[arm]["42"]["test_macro_f1"],
] for arm, _ in ARMS}

maxes = [max(v[i] for v in values.values()) for i in range(4)]

for arm, label in ARMS:
    v = values[arm]
    cells = []
    for i, x in enumerate(v):
        s = fmt(x)
        if abs(x - maxes[i]) < 1e-9:
            s = f"\\textbf{{{s}}}"
        cells.append(s)
    rows.append(f"{label} & " + " & ".join(cells) + r" \\")

table = r"""\begin{table}[H]
\centering
\caption{Classifier arms, seed~42. Night set $n{=}1{,}931$ (residential $n{=}147$); day set $n{=}2{,}447$; overall test $n{=}4{,}745$. Best per column in bold.}
\label{tab:main-results}
\begin{tabular}{lcccc}
\toprule
Arm & Night Res.\ Recall & Night Macro-F1 & Day Macro-F1 & Test Macro-F1 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

(OUT / "table_main_results.tex").write_text(table, encoding="utf-8")
print(f"wrote {OUT / 'table_main_results.tex'}")