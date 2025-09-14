# Dota 2 Ward Analysis (TI 2025 Swiss)

This repo analyzes ward placement for two Dota 2 teams (Team Falcons and PVISION/Pari) using OpenDota data. It produces per‑match reports and aggregate “time‑window” maps split by side (Radiant/Dire), with optional stitched views.

## Features
- Fetches all TI 2025 Swiss (league_id) matches per team
- Per‑match time‑window maps (5‑minute windows), with optional score and advantage annotations
- Aggregate maps across all TI matches:
  - Radiant‑only and Dire‑only time‑window maps
  - Combined 4‑column layout (Night/Dire × Night/Radiant × Day/Dire × Day/Radiant)
- High‑res minimap overlay (pixel‑space mapping; no flip)
- Observer/Sentry markers easily distinguishable
- Day/Night icon per window (correct −5–0 as night)

## Requirements
- Python 3.9+
- Packages: `requests, pandas, matplotlib, seaborn, numpy, Pillow`

Install:
- `python -m pip install -r requirements.txt`

## Optional Assets
- Latest minimap image (recommended): place as `assets/minimap_latest2.png` (or use `--map-image <path>`). PNG/JPG supported.
- Optional tower positions: `assets/tower_positions.json`
  - Format (world coordinates, not pixels):
    - `[ {"team":"radiant","lane":"top|mid|bot","tier":1..4,"x":<float> ,"y":<float> }, ... ]`
  - If present, single‑match reports can show alive/destroyed towers per window.

## Quick Start (Aggregate Maps Only)
Generate per‑side aggregate maps for both teams into `./output2`:
- `python ti_swiss_reporter.py --teams "Team Falcons,PVISION" --league-ids 18324 --limit 200 --max-matches 0 --only-time-windows --aggregate --map-image assets/minimap_latest2.png --output-dir output2`

Outputs (per team):
- `output2/<Team>/aggregate_TI_radiant/<Team>-天辉-眼位分析.png`
- `output2/<Team>/aggregate_TI_dire/<Team>-夜魇-眼位分析.png`
- Stitched n×4 view: `output2/<Team>/aggregate_TI_combined/<Team>-天辉夜魇-拼接-眼位分析.png`
- Also generated: 4‑column recomposed figure `aggregate_TI_combined/time_windows_combined.png`

## Per‑Match Time‑Window Maps
Generate per‑match “time_windows_map_analysis.png” (two columns, one subplot per 5‑minute window):
- `python ti_swiss_reporter.py --teams "Team Falcons" --league-ids 18324 --limit 80 --max-matches 1 --only-time-windows --map-image assets/minimap_latest2.png --output-dir output2`

This produces:
- `output2/Team_Falcons/<RadiantTeam>-<DireTeam> <match_id>/time_windows_map_analysis.png`

## CLI Options (ti_swiss_reporter.py)
- `--teams` Comma‑separated team names (supported: `Team Falcons, PVISION, Paris`)
- `--league-ids` Comma‑separated OpenDota league IDs (e.g. TI 2025 Swiss: `18324`)
- `--limit` How many recent matches to fetch per team (fetches then filters by league)
- `--max-matches` Per‑match renders cap
  - `0` means “skip per‑match images”; useful with `--aggregate`
- `--only-time-windows` Only generate the time‑window map(s); skip other analyses
- `--aggregate` Build aggregate maps across all matching TI games (Radiant and Dire). Also stitches the two into one n×4 image
- `--map-image` Path to a minimap image; if omitted, loader searches `assets/minimap_latest2.png`, then other fallbacks
- `--output-dir` Base output folder (default `output`); we use `output2` for “current” outputs
- `--show-vision` true/false to draw vision/truesight circles (disabled by default)
- `--adv-threshold` Only used by extended single‑match analyses (not used when `--only-time-windows`)

## Marker & Layout Conventions
- Observer: yellow circle with orange edge (smaller size)
- Sentry: light‑blue circle (#87CEFA) with blue edge (slightly smaller)
- Time windows: two columns (aggregate and per‑match), five‑minute bins sorted ascending
- Day/Night icon (subtitle): interval `[5n, 5(n+1))` is night when `n` is odd (e.g. −5–0, 5–10 is night)
- Team tag (our team):
  - Radiant: placed at 15%/15% (bottom‑left)
  - Dire: placed at 85%/85% (top‑right)

## Coordinate Mapping Notes
- OpenDota ward coords are on a 0..255 grid, origin at top‑left
- We map wards directly into minimap pixel space:
  - `wx = (x/255) * width`, `wy = (y/255) * height`
  - Matplotlib renders the map with `origin='upper'` to match image space
- This avoids the “central clustering” caused by mapping into world coordinates on a non‑cropped minimap

## Example Workflows
- Both teams, aggregate only (Radiant/Dire) with stitched n×4:
  - `python ti_swiss_reporter.py --teams "Team Falcons,PVISION" --league-ids 18324 --limit 200 --max-matches 0 --only-time-windows --aggregate --map-image assets/minimap_latest2.png --output-dir output2`
- One team, single per‑match quick check (1 match):
  - `python ti_swiss_reporter.py --teams "Team Falcons" --league-ids 18324 --limit 50 --max-matches 1 --only-time-windows --map-image assets/minimap_latest2.png --output-dir output2`

## Troubleshooting
- “I don’t see outputs”: ensure `--league-ids` is set (e.g. `18324`) and `--output-dir` points to a writable folder
- “Map looks flipped”: we use `origin='upper'` (no vertical flip). If your custom minimap has unusual orientation, pass a suitable replacement with `--map-image`
- “Wards look central”: verify you’re using the latest code (pixel mapping) and a full‑frame minimap (no heavy borders). If your minimap has margins, consider trimming it externally

## Repo Layout
- Main scripts
  - `ti_swiss_reporter.py` – CLI entry to fetch, aggregate and render
  - `complete_map_overlay_generator.py` – all plotting and map logic
  - `fetch_latest_minimap.py` – best‑effort fetch of a minimap (may be blocked; prefer supplying your own)
- Outputs
  - `output2/<Team>/...` – current outputs (per‑match, per‑side aggregates, stitched combined)
- Archive
  - `__archive__/legacy_scripts` – older scripts no longer used
  - `__archive__/previous_outputs` – earlier outputs and data moved out of the way

## Notes
- OpenDota free API is rate‑limited; large aggregate runs take a few minutes
- League IDs: we auto‑discovered TI 2025 Swiss as `18324`; if you need a different event, pass its `leagueid`

## License
- Internal analysis tooling intended for research/reporting; no license specified

