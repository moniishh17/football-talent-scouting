"""Central configuration for the ScoutIQ research pipeline (paths, seeds, pre-specified thresholds)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "research"
CACHE = DATA / "cache"           # per-match cached downloads (+ manifest with provenance)
PROCESSED = DATA / "processed"   # derived tables
AUDIT = DATA / "audit"
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"
TRANSFERMARKT = ROOT.parent / "transfermkt"

OPEN_DATA_BASE = "https://raw.githubusercontent.com/hudl/open-data/master/data"
USER_AGENT = "scoutiq-research/1.0 (academic research; StatsBomb Open Data)"
SEED = 20260101
MIN_MINUTES_GRID = (450, 900, 1350)   # exposure thresholds examined in sensitivity analysis
PRIMARY_MIN_MINUTES = 450
