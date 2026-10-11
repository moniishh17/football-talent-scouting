#!/usr/bin/env bash
# Resumable ingestion of every competition-season used by the study (safe to re-run; cached matches are skipped).
set -e
cd "$(dirname "$0")/../.."
python -m research.ingest.events --competition 11 --seasons 37 38 39 40 41 21 22 23 24 25 26 27 2 1 4 42 90
python -m research.ingest.events --competition 2  --seasons 27      # Premier League 2015/16 (full)
python -m research.ingest.events --competition 12 --seasons 27      # Serie A 2015/16 (full)
python -m research.ingest.events --competition 7  --seasons 27 108 235  # Ligue 1 2015/16 (full), 2021/22 & 2022/23 (PSG matches)
python -m research.ingest.events --competition 9  --seasons 27 281  # Bundesliga 2015/16, 2023/24 (Leverkusen matches)
python -m research.ingest.events --competition 44 --seasons 107     # MLS 2023 (6 matches)
