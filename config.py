"""Where the data, the results and the outputs live.

Defaults, all next to this file: the data set under ./data/Wang2023 (README.md), the result files in ./results, and the
LaTeX tables and Fig. 2 written by paper_tables.py and figure2.py in ./output. Environment variables put them elsewhere:
GAIT_FG_WANG, GAIT_FG_RESULTS, GAIT_FG_OUTPUT.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Wang et al. (2023): Processed_data/ (extracted from Processed_data.rar) and cache/ (written on first use)
WANG = os.environ.get("GAIT_FG_WANG", os.path.join(HERE, "data", "Wang2023"))
RESULTS = os.environ.get("GAIT_FG_RESULTS", os.path.join(HERE, "results"))
OUTPUT = os.environ.get("GAIT_FG_OUTPUT", os.path.join(HERE, "output"))

WANG_PROCESSED = os.path.join(WANG, "Processed_data")
WANG_CACHE = os.path.join(WANG, "cache")
