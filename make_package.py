"""Build the award package zip: solution code (README, requirements, single entry script run_all.py, Task 2-only
run_task2.py, specs) + the PDF report. Excludes caches and local tooling (Kaggle upload helpers keep no secrets: the
Kaggle CLI reads its own credentials file, which is never part of the package).

  python make_package.py   ->  TrafficFlowBench_Abderahmane_tb_package.zip
"""
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "TrafficFlowBench_Abderahmane_tb_package.zip")
SKIP_DIRS = {"__pycache__", "catboost_info"}
SKIP_FILES = {"submit_batch.sh"}

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(os.path.join(HERE, "solution")):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in sorted(files):
            if f in SKIP_FILES or f.endswith((".pyc", ".log")):
                continue
            p = os.path.join(root, f)
            zf.write(p, os.path.join("TrafficFlowBench_Abderahmane_tb", os.path.relpath(p, HERE)))
    zf.write(os.path.join(HERE, "report", "TECHNICAL_REPORT.pdf"),
             "TrafficFlowBench_Abderahmane_tb/TECHNICAL_REPORT.pdf")
with zipfile.ZipFile(OUT) as zf:
    names = zf.namelist()
print(OUT, f"{os.path.getsize(OUT) / 1e6:.1f} MB,", len(names), "files")
for n in names:
    if not n.endswith(".py"):
        print("  ", n)
