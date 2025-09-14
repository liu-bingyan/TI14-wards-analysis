#!/usr/bin/env python3
"""
TI 2025 Swiss Round Reporter

Fetches matches for specified teams, splits TI 2025 Swiss Round (by league_id)
vs. historical matches, and generates:
 - Real-map per-5-minute ward reports
 - Offensive vs Defensive ward maps (objectives context)
 - Advantage/Even/Disadvantage ward maps (net worth advantage)

Usage examples:

  python ti_swiss_reporter.py \
    --teams "Team Falcons,PVISION" \
    --league-ids 16220 \
    --limit 150 \
    --map-image assets/minimap_latest.png \
    --show-vision true

Notes:
 - league_ids must correspond to TI 2025 Swiss round leagues in OpenDota data.
 - You can pass multiple league IDs separated by commas if needed.
 - Place an updated minimap under assets/minimap_latest.png (or provide --map-image).
"""

import argparse
import json
import time
from typing import Dict, List, Optional

import requests
from pathlib import Path

from complete_map_overlay_generator import CompleteMapOverlayGenerator

# Optional tqdm for progress bars
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable


TEAM_NAME_TO_ID = {
    # Keep consistent with existing codebase
    "Team Falcons": 9247354,
    "PVISION": 9572001,
    # Alias for user's wording
    "Paris": 9572001,
}


class OpenDotaClient:
    def __init__(self, api_key: Optional[str] = None, request_delay: float = 1.0):
        self.base_url = "https://api.opendota.com/api"
        self.api_key = api_key
        self.s = requests.Session()
        self.delay = request_delay if not api_key else 0.1

    def _get(self, endpoint: str, params: Dict = None):
        params = params or {}
        if self.api_key:
            params['api_key'] = self.api_key
        time.sleep(self.delay)
        r = self.s.get(f"{self.base_url}/{endpoint}", params=params)
        r.raise_for_status()
        return r.json()

    def team_matches(self, team_id: int, limit: int = 100) -> List[Dict]:
        return self._get(f"teams/{team_id}/matches", {"limit": limit})

    def match_details(self, match_id: int) -> Dict:
        return self._get(f"matches/{match_id}")


def split_matches_by_league(matches: List[Dict], league_ids: List[int]):
    ti = []
    other = []
    for m in matches:
        lid = m.get('leagueid') or m.get('league_id')
        if lid in league_ids:
            ti.append(m)
        else:
            other.append(m)
    return ti, other


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def run_reports_for_match(gen: CompleteMapOverlayGenerator, team_name: str, team_id: int, match_id: int, only_time_windows: bool = False):
    """Generate all requested map-based reports for a match."""
    match_data = gen.get_match_details(match_id)
    if not match_data:
        return False
    ward_data, progression = gen.extract_enhanced_ward_data(match_data, team_id)
    if not ward_data:
        return False

    # Compute custom folder name: Radiant-Dire matchid
    r_name = None
    d_name = None
    try:
        rt = match_data.get('radiant_team') or {}
        dt = match_data.get('dire_team') or {}
        r_name = rt.get('name') or match_data.get('radiant_name') or 'Radiant'
        d_name = dt.get('name') or match_data.get('dire_name') or 'Dire'
    except Exception:
        r_name, d_name = 'Radiant', 'Dire'
    safe = lambda s: str(s).replace('/', '_').replace('\\', '_')
    folder_name = f"{safe(r_name)}-{safe(d_name)} {match_id}"

    # Determine side and tag
    is_radiant_team = match_data.get('radiant_team_id') == team_id
    tn = team_name.lower()
    if 'falcon' in tn:
        team_tag = 'FALCON'
    elif 'pvision' in tn or 'paris' in tn:
        team_tag = 'PARI'
    else:
        team_tag = team_name.split()[0].upper()

    # Core outputs
    print(f"    [*] {team_name} match {match_id}: time windows...")
    gen.create_time_windows_on_map(ward_data, match_id, team_name,
                                   folder_name=folder_name,
                                   is_radiant_team=is_radiant_team,
                                   team_tag=team_tag,
                                   match_data=match_data,
                                   team_id=team_id)
    if not only_time_windows:
        gen.create_time_window_reports(ward_data, match_id, team_name)
        print(f"    [*] {team_name} match {match_id}: game stages & momentum...")
        gen.create_game_stages_on_map(ward_data, match_id, team_name)
        gen.create_momentum_on_map(ward_data, match_id, team_name)
        print(f"    [*] {team_name} match {match_id}: offense/defense & advantage...")
        gen.create_offense_defense_on_map(ward_data, progression, match_id, team_name, match_data, team_id)
        gen.create_advantage_state_on_map(ward_data, match_id, team_name, match_data, team_id)
        print(f"    [*] {team_name} match {match_id}: comprehensive panel...")
        gen.create_ultimate_comprehensive_map_analysis(ward_data, match_id, team_name)

    # Save enhanced raw data for downstream analysis
    out_dir = Path("output") / team_name.replace(' ', '_') / f"match_{match_id}"
    ensure_dir(out_dir)
    (out_dir / "enhanced_ward_data.json").write_text(json.dumps(ward_data, indent=2))
    (out_dir / "match_info.json").write_text(json.dumps({
        "match_id": match_id,
        "leagueid": match_data.get('leagueid'),
        "radiant_team_id": match_data.get('radiant_team_id'),
        "dire_team_id": match_data.get('dire_team_id'),
        "duration": match_data.get('duration'),
    }, indent=2))
    return True


def main():
    parser = argparse.ArgumentParser(description="TI 2025 Swiss Round vision reporter")
    parser.add_argument('--teams', type=str, default="Team Falcons,PVISION",
                        help='Comma-separated team names (must match internal mapping)')
    parser.add_argument('--league-ids', type=str, required=True,
                        help='Comma-separated OpenDota league IDs for TI 2025 Swiss round')
    parser.add_argument('--limit', type=int, default=150, help='How many recent matches to fetch per team')
    parser.add_argument('--api-key', type=str, default=None, help='OpenDota API key')
    parser.add_argument('--map-image', type=str, default=None, help='Path to latest minimap image')
    parser.add_argument('--show-vision', type=str, default='false', help='Draw vision/truesight circles (true/false)')
    parser.add_argument('--adv-threshold', type=int, default=2000, help='Net worth advantage threshold for Advantage/Disadvantage')
    parser.add_argument('--max-matches', type=int, default=999, help='Max TI Swiss matches to render per team')
    parser.add_argument('--output-dir', type=str, default='output', help='Base output directory (default: output)')
    parser.add_argument('--only-time-windows', action='store_true', help='Only generate time_windows_map_analysis')
    parser.add_argument('--aggregate', action='store_true', help='Also generate aggregated time_windows per side (Radiant/Dire) across TI matches')
    args = parser.parse_args()

    team_names = [t.strip() for t in args.teams.split(',') if t.strip()]
    league_ids = [int(x.strip()) for x in args.league_ids.split(',') if x.strip()]
    show_vision = str(args.show_vision).lower() in ['1', 'true', 'yes', 'y']

    client = OpenDotaClient(api_key=args.api_key)
    gen = CompleteMapOverlayGenerator(api_key=args.api_key,
                                      map_image_path=args.map_image,
                                      show_vision=show_vision,
                                      advantage_threshold=args.adv_threshold,
                                      output_dir=args.output_dir)

    for team_name in team_names:
        if team_name not in TEAM_NAME_TO_ID:
            print(f"Unknown team: {team_name}. Available: {list(TEAM_NAME_TO_ID)}")
            continue
        team_id = TEAM_NAME_TO_ID[team_name]
        print(f"\n=== Processing {team_name} (ID {team_id}) ===")
        matches = client.team_matches(team_id, limit=args.limit)
        print(f"Fetched {len(matches)} matches")

        ti_matches, historical_matches = split_matches_by_league(matches, league_ids)
        print(f"  TI Swiss matches: {len(ti_matches)} | Historical: {len(historical_matches)}")

        # Save metadata split
        base_dir = Path(args.output_dir) / team_name.replace(' ', '_')
        ensure_dir(base_dir / "ti2025_swiss")
        ensure_dir(base_dir / "historical")
        (base_dir / "ti2025_swiss" / "matches.json").write_text(json.dumps(ti_matches, indent=2))
        (base_dir / "historical" / "matches.json").write_text(json.dumps(historical_matches, indent=2))

        # Generate reports for TI Swiss matches
        generated = 0
        ti_subset = ti_matches[:args.max_matches]
        for m in tqdm(ti_subset, desc=f"Generating {team_name} TI Swiss reports", total=len(ti_subset)):
            mid = m.get('match_id') or m.get('matchId')
            if not mid:
                continue
            ok = run_reports_for_match(gen, team_name, team_id, int(mid), only_time_windows=args.only_time_windows)
            if ok:
                generated += 1
        print(f"  ✅ Generated reports for {generated} TI Swiss matches")

        # Aggregated per side (Radiant/Dire)
        if args.aggregate:
            def team_tag_for(name: str):
                n = name.lower()
                if 'falcon' in n:
                    return 'FALCON'
                if 'pvision' in n or 'paris' in n:
                    return 'PARI'
                return name.split()[0].upper()

            # Collect entries for both sides
            radiant_entries = []
            dire_entries = []
            print(f"  📊 Aggregating {team_name} across TI matches (Radiant/Dire)...")
            for m in tqdm(ti_matches, desc=f"Collect {team_name} all sides"):
                mid = m.get('match_id') or m.get('matchId')
                if not mid:
                    continue
                md = gen.get_match_details(int(mid))
                if not md:
                    continue
                is_radiant = (md.get('radiant_team_id') == team_id)
                wd, _ = gen.extract_enhanced_ward_data(md, team_id)
                if not wd:
                    continue
                if is_radiant:
                    radiant_entries.extend(wd)
                else:
                    dire_entries.extend(wd)

            # Individual per-side outputs (retain)
            if radiant_entries:
                gen.create_time_windows_on_map(
                    radiant_entries, 0, team_name,
                    folder_name="aggregate_TI_radiant",
                    is_radiant_team=True,
                    team_tag=team_tag_for(team_name),
                    match_data=None, team_id=team_id,
                    custom_title=f"{team_name} - TI Aggregate (Radiant)\nWard Placement by Time Windows",
                    output_filename=f"{team_name}-天辉-眼位分析.png"
                )
            if dire_entries:
                gen.create_time_windows_on_map(
                    dire_entries, 0, team_name,
                    folder_name="aggregate_TI_dire",
                    is_radiant_team=False,
                    team_tag=team_tag_for(team_name),
                    match_data=None, team_id=team_id,
                    custom_title=f"{team_name} - TI Aggregate (Dire)\nWard Placement by Time Windows",
                    output_filename=f"{team_name}-夜魇-眼位分析.png"
                )

            # Combined 4-column aggregate
            if radiant_entries or dire_entries:
                gen.create_combined_side_daynight_aggregate(radiant_entries, dire_entries, team_name)
                # Simple horizontal stitching of the two per-side images
                base_dir = Path(args.output_dir) / team_name.replace(' ', '_')
                left_img = base_dir / 'aggregate_TI_radiant' / f"{team_name}-天辉-眼位分析.png"
                right_img = base_dir / 'aggregate_TI_dire' / f"{team_name}-夜魇-眼位分析.png"
                out_img = base_dir / 'aggregate_TI_combined' / f"{team_name}-天辉夜魇-拼接-眼位分析.png"
                if left_img.exists() and right_img.exists():
                    gen.stitch_side_by_side(left_img, right_img, out_img)


if __name__ == "__main__":
    main()
