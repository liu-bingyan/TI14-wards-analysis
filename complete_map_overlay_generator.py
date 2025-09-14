#!/usr/bin/env python3
"""
Complete Map Overlay Generator
Creates ALL types of ward analysis using the real Dota 2 map as background
- Time window analysis on real map
- Game stage analysis on real map  
- Momentum analysis on real map
- Comprehensive multi-dimensional analysis on real map
"""

import requests
import re
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import os

from matplotlib import patches


try:
    from tqdm import tqdm  # optional progress bar
except Exception:
    def tqdm(x, **kwargs):
        return x


class CompleteMapOverlayGenerator:
    def __init__(self, api_key=None, map_image_path: str = None, show_vision: bool = False,
                 observer_vision_radius: int = 1400, sentry_truesight_radius: int = 1000,
                 advantage_threshold: int = 2000, show_labels: bool = False,
                 output_dir: str = "output"):
        self.base_url = "https://api.opendota.com/api"
        self.api_key = api_key
        self.session = requests.Session()
        self.output_dir = Path(output_dir)
        self.heroes = {}
        self.request_delay = 1.0 if not api_key else 0.1
        
        # Dota 2 map parameters
        self.map_bounds = {
            'left': -8000,
            'right': 8000, 
            'bottom': -8000,
            'top': 8000
        }
        
        # Visualization options
        self.show_vision = show_vision
        self.observer_vision_radius = observer_vision_radius
        self.sentry_truesight_radius = sentry_truesight_radius
        self.advantage_threshold = advantage_threshold
        self.show_labels = show_labels

        # Load the map and hero data
        self.map_image = self._load_dota_map(map_image_path)
        # Cache map pixel size
        try:
            self.map_height, self.map_width = self.map_image.shape[0], self.map_image.shape[1]
        except Exception:
            self.map_height, self.map_width = 1024, 1024
        self._load_hero_data()
    
    def _load_dota_map(self, custom_path: str = None):
        """Load the Dota 2 map image"""
        # Priority: user provided -> assets/minimap_latest.(png|jpg) -> dota2_map.jpg -> synthetic
        candidate_paths = []
        if custom_path:
            candidate_paths.append(Path(custom_path))
        candidate_paths.extend([
            Path("assets/minimap_latest2.png"),
            Path("assets/minimap_latest2.jpg"),
            Path("assets/minimap_latest.png"),
            Path("assets/minimap_latest.jpg"),
            Path("assets/dota2_minimap.png"),
            Path("assets/dota2_minimap.jpg"),
            Path("dota2_map.jpg"),
        ])

        for map_file in candidate_paths:
            if map_file.exists():
                try:
                    map_img = Image.open(map_file)
                    if map_img.mode != 'RGB':
                        map_img = map_img.convert('RGB')
                    # Preserve original resolution for clarity; only upscale very small maps
                    w, h = map_img.size
                    if min(w, h) < 800:
                        scale_to = 1024
                        map_img = map_img.resize((max(scale_to, w), max(scale_to, h)), Image.Resampling.LANCZOS)
                    print(f"✅ Loaded Dota 2 map: {map_file} -> {map_img.size} (no downscale)")
                    return np.array(map_img)
                except Exception as e:
                    print(f"Error loading map {map_file}: {e}")
        
        print("Creating synthetic Dota 2-style map...")
        return self._create_synthetic_map()

    # Optional external data: tower positions
    def _load_tower_positions(self):
        if hasattr(self, '_tower_positions_loaded') and self._tower_positions_loaded:
            return getattr(self, '_tower_positions', None)
        self._tower_positions_loaded = True
        try:
            p = Path('assets/tower_positions.json')
            if not p.exists():
                self._tower_positions = None
                return None
            with open(p, 'r') as f:
                data = json.load(f)
            idx = {}
            for item in data:
                team = str(item.get('team', '')).lower()
                lane = str(item.get('lane', '')).lower()
                tier = int(item.get('tier', 0))
                key = (team, lane, tier)
                idx[key] = (float(item.get('x')), float(item.get('y')))
            self._tower_positions = idx
            return idx
        except Exception:
            self._tower_positions = None
            return None

    def _compute_kills_until(self, match_data: dict, t_seconds: int):
        rk = dk = 0
        for player in match_data.get('players', []) or []:
            is_radiant = (player.get('player_slot', 0) < 128)
            for k in player.get('kills_log', []) or []:
                if (k.get('time') or 0) <= t_seconds:
                    if is_radiant:
                        rk += 1
                    else:
                        dk += 1
        return rk, dk

    def _parse_objective_lane_tier(self, key: str):
        k = key.lower()
        lane = 'top' if 'top' in k else 'mid' if 'mid' in k else 'bot' if ('bot' in k or 'bottom' in k) else None
        tier = None
        for n in (1,2,3,4):
            if f'tier_{n}' in k or f't{n}' in k:
                tier = n
                break
        return lane, tier

    def _draw_towers_status(self, ax, match_data: dict, t_seconds: int):
        pos = self._load_tower_positions()
        if not pos:
            return
        # Determine destroyed towers up to t_seconds using objectives
        destroyed = set()
        for obj in match_data.get('objectives', []) or []:
            if obj.get('type') == 'CHAT_MESSAGE_TOWER_KILL' and (obj.get('time') or 0) <= t_seconds:
                key = obj.get('key', '') or ''
                lane, tier = self._parse_objective_lane_tier(key)
                # Team indicates which team got the kill; killed tower belongs to the opposite
                killer_team = obj.get('team')
                if lane and tier and killer_team in (2,3):
                    dead_team = 'dire' if killer_team == 2 else 'radiant'
                    destroyed.add((dead_team, lane, tier))
        # Draw
        for (team, lane, tier), (x, y) in pos.items():
            alive = (team, lane, tier) not in destroyed
            color = '#66ff66' if alive else '#777777'
            edge = '#2e8b57' if alive else '#555555'
            ax.scatter([x],[y], c=color, s=30, marker='s', alpha=0.9, edgecolors=edge, linewidths=0.8)

    # ------------------------------
    # Coordinate transforms
    # ------------------------------
    def _df_with_world_coords(self, df):
        """Return a copy of df with plotting coords (wx, wy) derived from x/y.
        Handles multiple possible coordinate systems from OpenDota data:
        - 128x128 grid: x,y in [0,127], origin top-left (common for obs_log/sen_log)
        - 256x256 grid: x,y in [0,255], origin top-left (seen in some exports)
        - World coords: x,y in approximately [-8000, 8000]
        All mapped to image pixel coordinates with origin='upper'.
        """
        if df is None or len(df) == 0:
            return df
        dfx = df.copy()
        dfx = dfx.dropna(subset=['x', 'y'])
        if len(dfx) == 0:
            return dfx
        # Convert to float
        xraw = dfx['x'].astype(float)
        yraw = dfx['y'].astype(float)
        xmin, xmax = float(xraw.min()), float(xraw.max())
        ymin, ymax = float(yraw.min()), float(yraw.max())

        # Choose mapping strategy
        use_world = False
        denom = None
        # Heuristic: if values clearly within 0..127 with small slack
        if xmin >= -1 and xmax <= 128.5 and ymin >= -1 and ymax <= 128.5:
            # Likely OpenDota 128-grid
            denom = 127.0
            xg = xraw.clip(lower=0.0, upper=denom)
            yg = yraw.clip(lower=0.0, upper=denom)
            dfx['wx'] = (xg / denom) * float(self.map_width)
            dfx['wy'] = (yg / denom) * float(self.map_height)
            return dfx
        # If within 0..255 (e.g., 256-grid)
        if xmin >= -1 and xmax <= 256.5 and ymin >= -1 and ymax <= 256.5:
            # 256-grid variant
            denom = 255.0
            xg = xraw.clip(lower=0.0, upper=denom)
            yg = yraw.clip(lower=0.0, upper=denom)
            dfx['wx'] = (xg / denom) * float(self.map_width)
            dfx['wy'] = (yg / denom) * float(self.map_height)
            return dfx
        # If appears to be world coordinates
        if xmin >= -20000 and xmax <= 20000 and ymin >= -20000 and ymax <= 20000:
            use_world = True
        if use_world:
            left = float(self.map_bounds.get('left', -8000.0))
            right = float(self.map_bounds.get('right', 8000.0))
            bottom = float(self.map_bounds.get('bottom', -8000.0))
            top = float(self.map_bounds.get('top', 8000.0))
            # Clamp to world bounds to avoid outliers
            xw = xraw.clip(lower=left, upper=right)
            yw = yraw.clip(lower=bottom, upper=top)
            # Map world x directly to pixel x
            dfx['wx'] = ((xw - left) / (right - left)) * float(self.map_width)
            # For origin='upper': top world (max y) should map to y=0 pixels
            dfx['wy'] = ((top - yw) / (top - bottom)) * float(self.map_height)
            return dfx
        # Fallback: scale by max range to avoid collapsing to corners
        rng = max(1.0, float(max(xmax - xmin, ymax - ymin)))
        dfx['wx'] = ((xraw - xmin) / rng) * float(self.map_width)
        # Invert y so larger y goes downward if raw looked increasing upward
        dfx['wy'] = ((ymax - yraw) / rng) * float(self.map_height)
        return dfx
    
    def _create_synthetic_map(self):
        """Create a synthetic Dota 2-style map"""
        size = 1024
        map_img = np.zeros((size, size, 3), dtype=np.uint8)
        
        # Create terrain-like colors
        center = size // 2
        y, x = np.ogrid[:size, :size]
        
        # Distance from center for river
        dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
        
        # River area (center diagonal)
        river_mask = (np.abs(x - y) < size * 0.05) | (dist_from_center < size * 0.08)
        map_img[river_mask] = [20, 40, 60]  # Dark blue river
        
        # Radiant side (bottom-left) - greener
        radiant_mask = (x + y > size) & (~river_mask)
        map_img[radiant_mask] = [30, 60, 20]  # Green forest
        
        # Dire side (top-right) - darker/redder  
        dire_mask = (x + y < size) & (~river_mask)
        map_img[dire_mask] = [50, 25, 15]  # Dark red/brown
        
        # Jungle areas - darker
        jungle_mask = ((x < center*0.7) | (x > center*1.3)) & ((y < center*0.7) | (y > center*1.3)) & (~river_mask)
        map_img[jungle_mask] = map_img[jungle_mask] * 0.7
        
        # Add noise for texture
        noise = np.random.randint(-8, 8, (size, size, 3))
        map_img = np.clip(map_img + noise, 0, 255)
        
        return map_img
    
    def _load_hero_data(self):
        """Load hero names from API"""
        try:
            heroes_data = self._make_request("heroes")
            if heroes_data:
                self.heroes = {hero['id']: hero['localized_name'] for hero in heroes_data}
                print(f"Loaded {len(self.heroes)} hero names")
        except Exception as e:
            print(f"Failed to load hero data: {e}")
    
    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """Make API request with rate limiting"""
        if params is None:
            params = {}
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        try:
            time.sleep(self.request_delay)
            response = self.session.get(f"{self.base_url}/{endpoint}", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API request failed: {e}")
            return {}
    
    def get_match_details(self, match_id: int) -> dict:
        """Get detailed match data"""
        return self._make_request(f"matches/{match_id}")
    
    def extract_game_progression(self, match_data: dict, team_id: int) -> dict:
        """Extract game progression markers"""
        progression = {
            'first_blood': match_data.get('first_blood_time', 0),
            'all_kills': []
        }
        
        is_radiant_team = match_data.get('radiant_team_id') == team_id
        
        all_kills = []
        for player in match_data.get('players', []):
            player_slot = player.get('player_slot', 0)
            is_radiant_player = player_slot < 128
            is_team_player = (is_radiant_team and is_radiant_player) or (not is_radiant_team and not is_radiant_player)
            
            if 'kills_log' in player and player['kills_log']:
                for kill in player['kills_log']:
                    all_kills.append({
                        'time': kill.get('time', 0),
                        'killer_team': 'our_team' if is_team_player else 'enemy_team',
                        'victim': kill.get('key', 'unknown')
                    })
        
        all_kills.sort(key=lambda x: x['time'])
        progression['all_kills'] = all_kills
        return progression
    
    def categorize_advanced_game_stage(self, current_time: int, progression: dict) -> str:
        """Advanced game stage categorization"""
        our_kills = sum(1 for kill in progression['all_kills'] 
                       if kill['time'] <= current_time and kill['killer_team'] == 'our_team')
        enemy_kills = sum(1 for kill in progression['all_kills'] 
                         if kill['time'] <= current_time and kill['killer_team'] == 'enemy_team')
        
        total_kills = our_kills + enemy_kills
        minutes = current_time // 60
        
        if minutes <= 10:
            return "Early Laning" if total_kills <= 3 else "Early Aggression"
        elif minutes <= 20:
            return "Mid Farm" if total_kills <= 10 else "Mid Fights"
        elif minutes <= 35:
            return "Late Farm" if total_kills <= 20 else "Late Fights"
        else:
            return "Very Late Game"
    
    def get_time_window(self, time_seconds: int) -> str:
        """Get 5-minute time window"""
        minutes = time_seconds // 60
        window_start = (minutes // 5) * 5
        window_end = window_start + 5
        return f"{window_start:02d}-{window_end:02d}min"
    
    def assess_team_momentum(self, current_time: int, progression: dict) -> str:
        """Assess team momentum based on recent events"""
        recent_window = 180  # 3 minutes
        our_recent_kills = sum(1 for kill in progression['all_kills']
                              if current_time - recent_window <= kill['time'] <= current_time
                              and kill['killer_team'] == 'our_team')
        enemy_recent_kills = sum(1 for kill in progression['all_kills']
                                if current_time - recent_window <= kill['time'] <= current_time
                                and kill['killer_team'] == 'enemy_team')
        
        if our_recent_kills > enemy_recent_kills + 1:
            return "Positive"
        elif enemy_recent_kills > our_recent_kills + 1:
            return "Negative"
        else:
            return "Neutral"
    
    def extract_enhanced_ward_data(self, match_data: dict, team_id: int):
        """Extract ward data with all categorizations"""
        ward_data = []
        progression = self.extract_game_progression(match_data, team_id)
        
        if 'players' not in match_data:
            return ward_data, progression
        
        is_radiant_team = match_data.get('radiant_team_id') == team_id
        
        for player in match_data['players']:
            player_slot = player.get('player_slot', 0)
            is_radiant_player = player_slot < 128
            
            if (is_radiant_team and is_radiant_player) or (not is_radiant_team and not is_radiant_player):
                # Observer wards
                for ward in player.get('obs_log', []):
                    if ward.get('x') is not None and ward.get('y') is not None:
                        ward_time = ward.get('time', 0)
                        ward_data.append({
                            'match_id': match_data['match_id'],
                            'team_id': team_id,
                            'ward_type': 'observer',
                            'time': ward_time,
                            'x': ward.get('x'),
                            'y': ward.get('y'),
                            'time_window': self.get_time_window(ward_time),
                            'game_stage': self.categorize_advanced_game_stage(ward_time, progression),
                            'momentum': self.assess_team_momentum(ward_time, progression),
                            'hero_name': self.heroes.get(player.get('hero_id'), 'Unknown')
                        })
                
                # Sentry wards
                for ward in player.get('sen_log', []):
                    if ward.get('x') is not None and ward.get('y') is not None:
                        ward_time = ward.get('time', 0)
                        ward_data.append({
                            'match_id': match_data['match_id'],
                            'team_id': team_id,
                            'ward_type': 'sentry',
                            'time': ward_time,
                            'x': ward.get('x'),
                            'y': ward.get('y'),
                            'time_window': self.get_time_window(ward_time),
                            'game_stage': self.categorize_advanced_game_stage(ward_time, progression),
                            'momentum': self.assess_team_momentum(ward_time, progression),
                            'hero_name': self.heroes.get(player.get('hero_id'), 'Unknown')
                        })
        
        return ward_data, progression

    def _draw_wards_with_optional_vision(self, ax, df_window):
        """Scatter ward points and optionally draw vision/truesight circles."""
        if df_window is None or len(df_window) == 0:
            return
        # Ensure world coords present
        if 'wx' not in df_window.columns or 'wy' not in df_window.columns:
            df_window = self._df_with_world_coords(df_window)
            if df_window is None or len(df_window) == 0:
                return
        obs_wards = df_window[df_window['ward_type'] == 'observer']
        sen_wards = df_window[df_window['ward_type'] == 'sentry']

        # Points
        if len(obs_wards) > 0:
            ax.scatter(obs_wards['wx'], obs_wards['wy'],
                       c='yellow', s=80, alpha=0.9, marker='o',
                       edgecolors='goldenrod', linewidths=1.5, label='Observer')
        if len(sen_wards) > 0:
            ax.scatter(sen_wards['wx'], sen_wards['wy'],
                       c='#87CEFA', s=70, alpha=0.9, marker='o',
                       edgecolors='royalblue', linewidths=1.5, label='Sentry')

        # Vision circles (approximate)
        if self.show_vision:
            # Convert world radius to pixel radius using horizontal scale
            try:
                pix_per_world = float(self.map_width) / float(self.map_bounds['right'] - self.map_bounds['left'])
            except Exception:
                pix_per_world = float(self.map_width) / 16000.0
            obs_r = self.observer_vision_radius * pix_per_world
            sen_r = self.sentry_truesight_radius * pix_per_world
            for _, row in obs_wards.iterrows():
                circ = patches.Circle((row['wx'], row['wy']), obs_r,
                                      facecolor='cyan', edgecolor='blue', alpha=0.12, lw=1)
                ax.add_patch(circ)
            for _, row in sen_wards.iterrows():
                circ = patches.Circle((row['wx'], row['wy']), sen_r,
                                      facecolor='yellow', edgecolor='red', alpha=0.10, lw=1)
                ax.add_patch(circ)

    def _annotate_coords(self, ax, df_points, text_color='white', fontsize=7):
        """Annotate each ward point with its (x, y) coordinates."""
        if df_points is None or len(df_points) == 0:
            return
        # Ensure world coords present
        if 'wx' not in df_points.columns or 'wy' not in df_points.columns:
            df_points = self._df_with_world_coords(df_points)
            if df_points is None or len(df_points) == 0:
                return
        for _, r in df_points.iterrows():
            x = r.get('wx'); y = r.get('wy')
            if x is None or y is None:
                continue
            try:
                # Show grid coords for readability and keep annotation at world coords
                gx = r.get('x'); gy = r.get('y')
                ax.text(x + 120, y + 120, f"g:({int(gx)}, {int(gy)})",
                        color=text_color, fontsize=fontsize,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5))
            except Exception:
                continue

    def create_time_windows_on_map(self, ward_data: list, match_id: int, team_name: str,
                                   folder_name: str = None, is_radiant_team: bool = None,
                                   team_tag: str = None, match_data: dict = None,
                                   team_id: int = None, custom_title: str = None,
                                   output_filename: str = None):
        """Create time window analysis on real Dota 2 map"""
        if not ward_data:
            return
        
        df = pd.DataFrame(ward_data)
        df_pos = df.dropna(subset=['x', 'y'])
        
        if df_pos.empty:
            return
        
        time_windows = sorted(df_pos['time_window'].unique())
        if len(time_windows) < 1:
            return
        
        # Create subplots
        n_windows = len(time_windows)
        # Arrange subplots in exactly two columns for readability
        n_cols = 2
        n_rows = (n_windows + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 6*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        title = custom_title if custom_title else f'{team_name} - Match {match_id}\nTime Window Analysis on Dota 2 Map'
        fig.suptitle(title, 
                     fontsize=18, color='white', y=0.98)
        fig.patch.set_facecolor('black')

        for i, time_window in enumerate(time_windows):
            if i >= len(axes):
                break
            
            # Show map background (pixel coordinates, origin at top-left)
            axes[i].imshow(self.map_image, origin='upper', extent=[
                0, self.map_width,
                0, self.map_height
            ])
            # Optional team side marker (Radiant bottom-left -> 15%,15%; Dire top-right -> 85%,85%)
            if is_radiant_team is not None:
                tag = team_tag if team_tag else team_name
                if is_radiant_team:
                    ax_x, ax_y = 0.15, 0.15
                    ha, va = 'left', 'bottom'
                else:
                    ax_x, ax_y = 0.85, 0.85
                    ha, va = 'right', 'top'
                axes[i].text(ax_x, ax_y, str(tag), color='white', fontsize=12,
                             ha=ha, va=va, transform=axes[i].transAxes,
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='white'))
            
            df_window = df_pos[df_pos['time_window'] == time_window]
            df_window = self._df_with_world_coords(df_window)
            if len(df_window) > 0:
                self._draw_wards_with_optional_vision(axes[i], df_window)
                if self.show_labels:
                    # Annotate coordinates for precise locations
                    self._annotate_coords(axes[i], df_window, text_color='white', fontsize=6)
                axes[i].legend(loc='upper right', fontsize=10, framealpha=0.8)

            # Metrics annotation per window start (gold/xp adv + towers)
            if match_data is not None and is_radiant_team is not None:
                try:
                    import math
                    m = re.match(r"\s*(-?\d+)\s*-", str(time_window))
                    win_start_min = int(m.group(1)) if m else 0
                    minute_index = max(0, win_start_min)
                    gold_adv = match_data.get('radiant_gold_adv') or []
                    xp_adv = match_data.get('radiant_xp_adv') or []
                    # Compute adv for our team (radiant positive if radiant team; invert if dire)
                    def fmt_adv(arr):
                        if not arr or minute_index >= len(arr):
                            return '—'
                        val = arr[minute_index]
                        if not is_radiant_team:
                            val = -val
                        # thousands
                        return (f"{val/1000.0:+.1f}k")
                    gold_str = fmt_adv(gold_adv)
                    xp_str = fmt_adv(xp_adv)
                    # Kills up to t_start
                    t_start = max(0, win_start_min) * 60
                    rk, dk = self._compute_kills_until(match_data, t_start)
                    # Display gold/xp adv and kills (Radiant:Dire)
                    axes[i].text(0.02, 0.98,
                                 f"Gold {gold_str} | XP {xp_str}\nKills {rk}:{dk}",
                                 color='white', fontsize=9, ha='left', va='top',
                                 transform=axes[i].transAxes,
                                 bbox=dict(boxstyle='round,pad=0.25', facecolor='black', alpha=0.6))
                    # Draw towers alive/destroyed if positions available
                    self._draw_towers_status(axes[i], match_data, t_start)
                except Exception:
                    pass
            
            # Day/Night icon in subtitle based on 5-min cycle (0-5 day, 5-10 night, ...)
            try:
                m = re.match(r"\s*(-?\d+)\s*-", str(time_window))
                win_start = int(m.group(1)) if m else 0
            except Exception:
                win_start = 0
            # Night/day rule: interval [5n, 5(n+1)) is NIGHT when n is odd; otherwise DAY.
            # Example: -5–0 (n=-1) is night; 0–5 (n=0) is day.
            n = (win_start // 5)
            is_night = (n % 2 != 0)
            icon = '☾' if is_night else '☀'

            axes[i].set_title(f'{time_window}\n{icon} ({len(df_window)} wards)', 
                              fontsize=12, color='white', fontweight='bold',
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.8))
            axes[i].set_xlim(0, self.map_width)
            axes[i].set_ylim(0, self.map_height)
            axes[i].grid(True, alpha=0.3, color='white')
            axes[i].set_facecolor('black')
        
        # Hide unused subplots
        for i in range(len(time_windows), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        folder = folder_name if folder_name else f"match_{match_id}"
        match_dir = self.output_dir / team_name.replace(' ', '_') / folder
        match_dir.mkdir(parents=True, exist_ok=True)
        # Custom filename support (e.g., "TeamName-天辉-眼位分析.png")
        out_name = output_filename if output_filename else "time_windows_map_analysis.png"
        file_path = match_dir / out_name
        plt.savefig(file_path, dpi=300, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        plt.close()

        print(f"✅ Created time windows map: {file_path}")

    def create_combined_side_daynight_aggregate(self, radiant_entries: list, dire_entries: list,
                                                team_name: str):
        """Create a combined aggregate plot with 4 columns per time window row:
        Col 0: Night Radiant, Col 1: Night Dire, Col 2: Day Radiant, Col 3: Day Dire.
        """
        dfr = pd.DataFrame(radiant_entries).dropna(subset=['x','y']) if radiant_entries else pd.DataFrame()
        dfd = pd.DataFrame(dire_entries).dropna(subset=['x','y']) if dire_entries else pd.DataFrame()
        if dfr.empty and dfd.empty:
            print(f"⚠️ No data for combined aggregate of {team_name}")
            return
        # Map to pixel coords
        if not dfr.empty:
            dfr = self._df_with_world_coords(dfr)
        if not dfd.empty:
            dfd = self._df_with_world_coords(dfd)

        # Collect all time windows
        windows = []
        if not dfr.empty:
            windows.extend(list(dfr['time_window'].unique()))
        if not dfd.empty:
            windows.extend(list(dfd['time_window'].unique()))
        windows = sorted(pd.unique(windows))
        if not windows:
            print(f"⚠️ No time windows for combined aggregate of {team_name}")
            return

        n_rows = len(windows)
        n_cols = 4
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = np.array([axes])

        fig.suptitle(f"{team_name} - TI Aggregate (Radiant/Dire x Night/Day)\nWard Placement by Time Windows",
                     fontsize=18, color='white', y=0.98)
        fig.patch.set_facecolor('black')

        def is_night_window(win_label: str) -> bool:
            try:
                m = re.match(r"\s*(-?\d+)\s*-", str(win_label))
                ws = int(m.group(1)) if m else 0
            except Exception:
                ws = 0
            n = (ws // 5)
            return (n % 2 != 0)

        for r, win in enumerate(windows):
            night = is_night_window(win)
            # Background for all 4 columns in this row
            for c in range(n_cols):
                ax = axes[r, c]
                ax.imshow(self.map_image, origin='upper', extent=[0, self.map_width, 0, self.map_height])
                # Team tag per column: Radiant on cols 0,2; Dire on cols 1,3
                is_rad_col = (c in (0,2))
                tag = team_name.split()[0].upper()
                ax_x, ax_y = (0.15, 0.15) if is_rad_col else (0.85, 0.85)
                ha, va = ('left','bottom') if is_rad_col else ('right','top')
                ax.text(ax_x, ax_y, tag, color='white', fontsize=10, ha=ha, va=va,
                        transform=ax.transAxes, bbox=dict(boxstyle='round,pad=0.3',
                        facecolor='black', alpha=0.7, edgecolor='white'))
                ax.set_xlim(0, self.map_width)
                ax.set_ylim(0, self.map_height)
                ax.grid(True, alpha=0.3, color='white')
                ax.set_facecolor('black')

            # Select per panel data
            # Night Radiant (col 0)
            if not dfr.empty and night:
                dsub = dfr[dfr['time_window'] == win]
                if len(dsub) > 0:
                    obs = dsub[dsub['ward_type']=='observer']
                    sen = dsub[dsub['ward_type']=='sentry']
                    if len(obs)>0:
                        axes[r,0].scatter(obs['wx'], obs['wy'], c='yellow', s=80, alpha=0.9, marker='o', edgecolors='goldenrod', linewidths=1.5)
                    if len(sen)>0:
                        axes[r,0].scatter(sen['wx'], sen['wy'], c='#87CEFA', s=70, alpha=0.9, marker='o', edgecolors='royalblue', linewidths=1.5)
            # Night Dire (col 1)
            if not dfd.empty and night:
                dsub = dfd[dfd['time_window'] == win]
                if len(dsub) > 0:
                    obs = dsub[dsub['ward_type']=='observer']
                    sen = dsub[dsub['ward_type']=='sentry']
                    if len(obs)>0:
                        axes[r,1].scatter(obs['wx'], obs['wy'], c='yellow', s=80, alpha=0.9, marker='o', edgecolors='goldenrod', linewidths=1.5)
                    if len(sen)>0:
                        axes[r,1].scatter(sen['wx'], sen['wy'], c='#87CEFA', s=70, alpha=0.9, marker='o', edgecolors='royalblue', linewidths=1.5)
            # Day Radiant (col 2)
            if not dfr.empty and not night:
                dsub = dfr[dfr['time_window'] == win]
                if len(dsub) > 0:
                    obs = dsub[dsub['ward_type']=='observer']
                    sen = dsub[dsub['ward_type']=='sentry']
                    if len(obs)>0:
                        axes[r,2].scatter(obs['wx'], obs['wy'], c='yellow', s=80, alpha=0.9, marker='o', edgecolors='goldenrod', linewidths=1.5)
                    if len(sen)>0:
                        axes[r,2].scatter(sen['wx'], sen['wy'], c='#87CEFA', s=70, alpha=0.9, marker='o', edgecolors='royalblue', linewidths=1.5)
            # Day Dire (col 3)
            if not dfd.empty and not night:
                dsub = dfd[dfd['time_window'] == win]
                if len(dsub) > 0:
                    obs = dsub[dsub['ward_type']=='observer']
                    sen = dsub[dsub['ward_type']=='sentry']
                    if len(obs)>0:
                        axes[r,3].scatter(obs['wx'], obs['wy'], c='yellow', s=80, alpha=0.9, marker='o', edgecolors='goldenrod', linewidths=1.5)
                    if len(sen)>0:
                        axes[r,3].scatter(sen['wx'], sen['wy'], c='#87CEFA', s=70, alpha=0.9, marker='o', edgecolors='royalblue', linewidths=1.5)

            # Row titles (on col 0) include window label and icon
            icon = '☾' if night else '☀'
            axes[r,0].set_title(f"{win} {icon}", fontsize=10, color='white')

        plt.tight_layout()
        out_dir = self.output_dir / team_name.replace(' ', '_') / 'aggregate_TI_combined'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'time_windows_combined.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
        plt.close()
        print(f"✅ Created combined side/day-night aggregate: {out_path}")

    def stitch_side_by_side(self, left_path: Path, right_path: Path, out_path: Path):
        """Horizontally concatenate two images (simple stitch) and save to out_path.
        If heights differ, scale the right image to match the left image's height.
        """
        try:
            from PIL import Image
            left = Image.open(left_path).convert('RGB')
            right = Image.open(right_path).convert('RGB')
            lh, lw = left.size[1], left.size[0]
            rh, rw = right.size[1], right.size[0]
            if rh != lh:
                # Resize right to match left height, preserve aspect
                new_rw = int(rw * (lh / rh))
                right = right.resize((new_rw, lh), Image.Resampling.LANCZOS)
                rw = new_rw; rh = lh
            stitched = Image.new('RGB', (lw + rw, lh), (0, 0, 0))
            stitched.paste(left, (0, 0))
            stitched.paste(right, (lw, 0))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            stitched.save(out_path, format='PNG')
            print(f"✅ Stitched image saved: {out_path}")
        except Exception as e:
            print(f"❌ Stitch failed for {out_path}: {e}")

    def create_time_window_reports(self, ward_data: list, match_id: int, team_name: str):
        """Create individual images per 5-minute window on real map."""
        if not ward_data:
            return
        df = pd.DataFrame(ward_data)
        df_pos = df.dropna(subset=['x', 'y'])
        if df_pos.empty:
            return
        time_windows = sorted(df_pos['time_window'].unique())
        match_dir = self.output_dir / team_name.replace(' ', '_') / f"match_{match_id}" / "reports" / "time_windows"
        match_dir.mkdir(parents=True, exist_ok=True)

        # Simple summary CSV
        summary_rows = []

        for time_window in tqdm(time_windows, desc=f"{team_name} {match_id} per 5-min reports"):
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.imshow(self.map_image, origin='upper', extent=[0, self.map_width, 0, self.map_height])
            df_window = df_pos[df_pos['time_window'] == time_window]
            df_window = self._df_with_world_coords(df_window)
            self._draw_wards_with_optional_vision(ax, df_window)
            if self.show_labels:
                # Annotate coordinates for each ward point in per-window report
                self._annotate_coords(ax, df_window, text_color='white', fontsize=7)
            ax.set_title(f'{team_name} - Match {match_id}  {time_window}\n({len(df_window)} wards)',
                         fontsize=12, color='white', fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.8))
            ax.set_xlim(0, self.map_width)
            ax.set_ylim(0, self.map_height)
            ax.grid(True, alpha=0.3, color='white')
            ax.set_facecolor('black')
            ax.legend(loc='upper right', fontsize=10, framealpha=0.8)
            out_path = match_dir / f"report_{time_window}.png"
            plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
            plt.close()

            # Summaries per window
            if len(df_window) > 0:
                obs_cnt = int((df_window['ward_type'] == 'observer').sum())
                sen_cnt = int((df_window['ward_type'] == 'sentry').sum())
                summary_rows.append({
                    'time_window': time_window,
                    'total_wards': int(len(df_window)),
                    'observer_count': obs_cnt,
                    'sentry_count': sen_cnt,
                })
        # Write summary CSV if there are rows
        if summary_rows:
            import csv
            csv_path = match_dir / "summary.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)
        print(f"✅ Created per-window reports under: {match_dir}")
    
    def create_game_stages_on_map(self, ward_data: list, match_id: int, team_name: str):
        """Create game stage analysis on real Dota 2 map"""
        if not ward_data:
            return
        
        df = pd.DataFrame(ward_data)
        df_pos = df.dropna(subset=['x', 'y'])
        
        if df_pos.empty:
            return
        
        game_stages = sorted(df_pos['game_stage'].unique())
        if len(game_stages) < 1:
            return
        
        # Create subplots  
        n_stages = len(game_stages)
        n_cols = min(3, n_stages)
        n_rows = (n_stages + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'{team_name} - Match {match_id}\nGame Stage Analysis on Dota 2 Map', 
                     fontsize=18, color='white', y=0.98)
        fig.patch.set_facecolor('black')
        
        stage_colors = {
            'Early Laning': 'lightgreen',
            'Early Aggression': 'orange', 
            'Mid Farm': 'yellow',
            'Mid Fights': 'red',
            'Late Farm': 'purple',
            'Late Fights': 'magenta',
            'Very Late Game': 'white'
        }
        
        for i, stage in enumerate(game_stages):
            if i >= len(axes):
                break
            
            # Show map background
            axes[i].imshow(self.map_image, origin='upper', extent=[0, self.map_width, 0, self.map_height])
            
            df_stage = df_pos[df_pos['game_stage'] == stage]
            df_stage = self._df_with_world_coords(df_stage)
            
            if len(df_stage) > 0:
                obs_wards = df_stage[df_stage['ward_type'] == 'observer']
                sen_wards = df_stage[df_stage['ward_type'] == 'sentry']
                
                stage_color = stage_colors.get(stage, 'white')
                
                if len(obs_wards) > 0:
                    axes[i].scatter(obs_wards['wx'], obs_wards['wy'], 
                                  c=stage_color, s=120, alpha=0.9, marker='o', 
                                  edgecolors='darkblue', linewidths=2, label='Observer')
                
                if len(sen_wards) > 0:
                    axes[i].scatter(sen_wards['wx'], sen_wards['wy'], 
                                  c=stage_color, s=120, alpha=0.9, marker='^', 
                                  edgecolors='darkred', linewidths=2, label='Sentry')
                
                if len(obs_wards) > 0 or len(sen_wards) > 0:
                    axes[i].legend(loc='upper right', fontsize=10, framealpha=0.8)
                if self.show_labels:
                    # Annotate coordinates
                    self._annotate_coords(axes[i], df_stage, text_color='white', fontsize=6)
            
            axes[i].set_title(f'{stage}\n({len(df_stage)} wards)', 
                            fontsize=12, color='white', fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.8))
            axes[i].set_xlim(0, self.map_width)
            axes[i].set_ylim(0, self.map_height)
            axes[i].grid(True, alpha=0.3, color='white')
            axes[i].set_facecolor('black')
        
        # Hide unused subplots
        for i in range(len(game_stages), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        match_dir = self.output_dir / team_name.replace(' ', '_') / f"match_{match_id}"
        match_dir.mkdir(parents=True, exist_ok=True)
        file_path = match_dir / "game_stages_map_analysis.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        plt.close()
        
        print(f"✅ Created game stages map: {file_path}")

    def create_offense_defense_on_map(self, ward_data: list, progression: dict, match_id: int, team_name: str, match_data: dict, team_id: int):
        """Create offensive vs defensive warding map.
        Prefer recent objectives (±120s). If objectives unavailable, fall back to recent kills.
        """
        if not ward_data:
            return
        df = pd.DataFrame(ward_data)
        df_pos = df.dropna(subset=['x', 'y', 'time'])
        if df_pos.empty:
            return
        is_radiant_team = match_data.get('radiant_team_id') == team_id

        # Prepare recent event extractor
        objectives_list = match_data.get('objectives') or []
        use_objectives = isinstance(objectives_list, list) and len(objectives_list) > 0

        # Classify context
        contexts = []
        for _, row in df_pos.iterrows():
            t = int(row['time'] or 0)
            our, enemy = 0, 0
            if use_objectives:
                recent = [o for o in objectives_list if (o.get('time', 0) or 0) <= t and (o.get('time', 0) or 0) > t - 120]
                for obj in recent:
                    if obj.get('type') in ['CHAT_MESSAGE_TOWER_KILL', 'CHAT_MESSAGE_BARRACKS_KILLED']:
                        team_val = obj.get('team')
                        if team_val == (2 if is_radiant_team else 3):
                            our += 1
                        else:
                            enemy += 1
            else:
                # Fallback to kills in last 120s using progression['all_kills'] if available
                kills = (progression or {}).get('all_kills') or []
                for k in kills:
                    kt = int(k.get('time', 0) or 0)
                    if t - 120 < kt <= t:
                        if k.get('killer_team') == 'our_team':
                            our += 1
                        else:
                            enemy += 1
            if our > enemy:
                contexts.append('Offensive')
            elif enemy > our:
                contexts.append('Defensive')
            else:
                contexts.append('Neutral')
        df_pos['context'] = contexts

        categories = [c for c in ['Offensive', 'Defensive', 'Neutral'] if c in df_pos['context'].values]
        if not categories:
            return

        fig, axes = plt.subplots(1, len(categories), figsize=(8*len(categories), 6))
        if len(categories) == 1:
            axes = [axes]
        fig.suptitle(f'{team_name} - Match {match_id}\nOffensive vs Defensive Warding', fontsize=18, color='white', y=0.95)
        fig.patch.set_facecolor('black')
        color_map = {'Offensive': 'red', 'Defensive': 'blue', 'Neutral': 'green'}
        for i, cat in enumerate(categories):
            ax = axes[i]
            ax.imshow(self.map_image, origin='upper', extent=[0, self.map_width, 0, self.map_height])
            df_cat = df_pos[df_pos['context'] == cat]
            df_cat = self._df_with_world_coords(df_cat)
            if not df_cat.empty:
                # Temporarily override colors for clarity
                obs = df_cat[df_cat['ward_type'] == 'observer']
                sen = df_cat[df_cat['ward_type'] == 'sentry']
                if len(obs) > 0:
                    ax.scatter(obs['wx'], obs['wy'], c=color_map[cat], s=120, alpha=0.9, marker='o', edgecolors='black', linewidths=2, label='Observer')
                if len(sen) > 0:
                    ax.scatter(sen['wx'], sen['wy'], c=color_map[cat], s=120, alpha=0.9, marker='^', edgecolors='black', linewidths=2, label='Sentry')
                if self.show_labels:
                    # Annotate coordinates for precise locations
                    self._annotate_coords(ax, df_cat, text_color='white', fontsize=7)
                ax.legend(loc='upper right', fontsize=10, framealpha=0.8)
            ax.set_title(f'{cat}\n({len(df_cat)} wards)', fontsize=12, color='white', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.8))
            ax.set_xlim(0, self.map_width)
            ax.set_ylim(0, self.map_height)
            ax.grid(True, alpha=0.3, color='white')
            ax.set_facecolor('black')
        plt.tight_layout()
        match_dir = self.output_dir / team_name.replace(' ', '_') / f"match_{match_id}"
        match_dir.mkdir(parents=True, exist_ok=True)
        file_path = match_dir / "offense_defense_map_analysis.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
        plt.close()
        print(f"✅ Created offense/defense map: {file_path}")

    def create_advantage_state_on_map(self, ward_data: list, match_id: int, team_name: str, match_data: dict, team_id: int):
        """Create maps for advantage/neutral/disadvantage states using radiant_gold_adv timeline."""
        if not ward_data:
            return
        gold_adv = match_data.get('radiant_gold_adv') or []
        if not gold_adv:
            return
        is_radiant_team = match_data.get('radiant_team_id') == team_id
        df = pd.DataFrame(ward_data)
        df_pos = df.dropna(subset=['x', 'y', 'time'])
        if df_pos.empty:
            return

        # Classify advantage state at ward time (minute granularity)
        states = []
        for _, row in df_pos.iterrows():
            minute = max(0, int((row['time'] or 0) // 60))
            if minute >= len(gold_adv):
                minute = len(gold_adv) - 1
            adv = gold_adv[minute]
            team_adv = adv if is_radiant_team else -adv
            if team_adv >= self.advantage_threshold:
                states.append('Advantage')
            elif team_adv <= -self.advantage_threshold:
                states.append('Disadvantage')
            else:
                states.append('Even')
        df_pos['adv_state'] = states

        categories = [c for c in ['Advantage', 'Even', 'Disadvantage'] if c in df_pos['adv_state'].values]
        if not categories:
            return

        fig, axes = plt.subplots(1, len(categories), figsize=(8*len(categories), 6))
        if len(categories) == 1:
            axes = [axes]
        fig.suptitle(f'{team_name} - Match {match_id}\nAdvantage State Warding (±{self.advantage_threshold} net worth)', fontsize=18, color='white', y=0.95)
        fig.patch.set_facecolor('black')
        color_map = {'Advantage': 'lime', 'Even': 'orange', 'Disadvantage': 'red'}
        for i, cat in enumerate(categories):
            ax = axes[i]
            ax.imshow(self.map_image, origin='upper', extent=[0, self.map_width, 0, self.map_height])
            df_cat = df_pos[df_pos['adv_state'] == cat]
            df_cat = self._df_with_world_coords(df_cat)
            if not df_cat.empty:
                obs = df_cat[df_cat['ward_type'] == 'observer']
                sen = df_cat[df_cat['ward_type'] == 'sentry']
                if len(obs) > 0:
                    ax.scatter(obs['wx'], obs['wy'], c=color_map[cat], s=120, alpha=0.9, marker='o', edgecolors='black', linewidths=2, label='Observer')
                if len(sen) > 0:
                    ax.scatter(sen['wx'], sen['wy'], c=color_map[cat], s=120, alpha=0.9, marker='^', edgecolors='black', linewidths=2, label='Sentry')
                if self.show_labels:
                    # Annotate coordinates for precise locations
                    self._annotate_coords(ax, df_cat, text_color='white', fontsize=7)
                ax.legend(loc='upper right', fontsize=10, framealpha=0.8)
            ax.set_title(f'{cat}\n({len(df_cat)} wards)', fontsize=12, color='white', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.8))
            ax.set_xlim(0, self.map_width)
            ax.set_ylim(0, self.map_height)
            ax.grid(True, alpha=0.3, color='white')
            ax.set_facecolor('black')
        plt.tight_layout()
        match_dir = self.output_dir / team_name.replace(' ', '_') / f"match_{match_id}"
        match_dir.mkdir(parents=True, exist_ok=True)
        file_path = match_dir / "advantage_state_map_analysis.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
        plt.close()
        print(f"✅ Created advantage-state map: {file_path}")
    
    def create_momentum_on_map(self, ward_data: list, match_id: int, team_name: str):
        """Create momentum analysis on real Dota 2 map"""
        if not ward_data:
            return
        
        df = pd.DataFrame(ward_data)
        df_pos = df.dropna(subset=['x', 'y'])
        
        if df_pos.empty:
            return
        
        momentums = ['Positive', 'Neutral', 'Negative']
        available_momentums = [m for m in momentums if m in df_pos['momentum'].values]
        
        if len(available_momentums) < 1:
            return
        
        # Create subplots
        fig, axes = plt.subplots(1, len(available_momentums), figsize=(8*len(available_momentums), 6))
        if len(available_momentums) == 1:
            axes = [axes]
        
        fig.suptitle(f'{team_name} - Match {match_id}\nTeam Momentum Analysis on Dota 2 Map', 
                     fontsize=18, color='white', y=0.95)
        fig.patch.set_facecolor('black')
        
        momentum_colors = {
            'Positive': 'lime',
            'Neutral': 'gray', 
            'Negative': 'red'
        }
        
        for i, momentum in enumerate(available_momentums):
            # Show map background
            axes[i].imshow(self.map_image, extent=[
                self.map_bounds['left'], self.map_bounds['right'],
                self.map_bounds['bottom'], self.map_bounds['top']
            ])
            
            df_momentum = df_pos[df_pos['momentum'] == momentum]
            df_momentum = self._df_with_world_coords(df_momentum)
            
            if len(df_momentum) > 0:
                obs_wards = df_momentum[df_momentum['ward_type'] == 'observer']
                sen_wards = df_momentum[df_momentum['ward_type'] == 'sentry']
                
                momentum_color = momentum_colors[momentum]
                
                if len(obs_wards) > 0:
                    axes[i].scatter(obs_wards['wx'], obs_wards['wy'], 
                                  c=momentum_color, s=140, alpha=0.9, marker='o', 
                                  edgecolors='black', linewidths=3, label='Observer')
                
                if len(sen_wards) > 0:
                    axes[i].scatter(sen_wards['wx'], sen_wards['wy'], 
                                  c=momentum_color, s=140, alpha=0.9, marker='^', 
                                  edgecolors='black', linewidths=3, label='Sentry')
                
                if len(obs_wards) > 0 or len(sen_wards) > 0:
                    axes[i].legend(loc='upper right', fontsize=12, framealpha=0.8)
                if self.show_labels:
                    # Annotate coordinates
                    self._annotate_coords(axes[i], df_momentum, text_color='white', fontsize=7)
            
            axes[i].set_title(f'{momentum} Momentum\n({len(df_momentum)} wards)', 
                            fontsize=14, color='white', fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.8))
            axes[i].set_xlim(0, self.map_width)
            axes[i].set_ylim(0, self.map_height)
            axes[i].grid(True, alpha=0.3, color='white')
            axes[i].set_facecolor('black')
        
        plt.tight_layout()
        
        match_dir = self.output_dir / team_name.replace(' ', '_') / f"match_{match_id}"
        match_dir.mkdir(parents=True, exist_ok=True)
        file_path = match_dir / "momentum_map_analysis.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        plt.close()
        
        print(f"✅ Created momentum map: {file_path}")
    
    def create_ultimate_comprehensive_map_analysis(self, ward_data: list, match_id: int, team_name: str):
        """Create the ultimate comprehensive analysis with all dimensions on real map"""
        if not ward_data:
            return
        
        df = pd.DataFrame(ward_data)
        df_pos = df.dropna(subset=['x', 'y'])
        
        if df_pos.empty:
            return
        
        # Create massive comprehensive figure
        fig = plt.figure(figsize=(24, 18), facecolor='black')
        fig.suptitle(f'{team_name} - Match {match_id}\nUltimate Comprehensive Ward Analysis on Dota 2 Map', 
                     fontsize=24, color='white', y=0.98)
        
        # 1. Time windows (top row)
        time_windows = sorted(df_pos['time_window'].unique())[:8]  # Limit to 8
        gs_time = fig.add_gridspec(4, 8, top=0.90, bottom=0.68, hspace=0.3, wspace=0.2)
        
        for i, time_window in enumerate(time_windows):
            ax = fig.add_subplot(gs_time[0, i])
            ax.imshow(self.map_image, origin='upper', extent=[0, self.map_width, 0, self.map_height])
            
            df_time = df_pos[df_pos['time_window'] == time_window]
            df_time = self._df_with_world_coords(df_time)
            
            if len(df_time) > 0:
                obs_wards = df_time[df_time['ward_type'] == 'observer']
                sen_wards = df_time[df_time['ward_type'] == 'sentry']
                
                if len(obs_wards) > 0:
                    ax.scatter(obs_wards['wx'], obs_wards['wy'], c='cyan', s=40, alpha=0.9, marker='o')
                if len(sen_wards) > 0:
                    ax.scatter(sen_wards['wx'], sen_wards['wy'], c='yellow', s=40, alpha=0.9, marker='^')
            
            ax.set_title(f'{time_window}\n({len(df_time)})', fontsize=10, color='white')
            ax.set_xlim(0, self.map_width)
            ax.set_ylim(0, self.map_height)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # 2. Game stages (second row)
        game_stages = sorted(df_pos['game_stage'].unique())[:8]
        stage_colors = ['lightgreen', 'orange', 'yellow', 'red', 'purple', 'magenta', 'white', 'pink']
        
        for i, stage in enumerate(game_stages):
            ax = fig.add_subplot(gs_time[1, i])
            ax.imshow(self.map_image, origin='upper', extent=[0, self.map_width, 0, self.map_height])
            
            df_stage = df_pos[df_pos['game_stage'] == stage]
            df_stage = self._df_with_world_coords(df_stage)
            
            if len(df_stage) > 0:
                obs_wards = df_stage[df_stage['ward_type'] == 'observer']
                sen_wards = df_stage[df_stage['ward_type'] == 'sentry']
                
                stage_color = stage_colors[i % len(stage_colors)]
                
                if len(obs_wards) > 0:
                    ax.scatter(obs_wards['wx'], obs_wards['wy'], c=stage_color, s=40, alpha=0.9, marker='o')
                if len(sen_wards) > 0:
                    ax.scatter(sen_wards['wx'], sen_wards['wy'], c=stage_color, s=40, alpha=0.9, marker='^')
            
            ax.set_title(f'{stage}\n({len(df_stage)})', fontsize=10, color='white')
            ax.set_xlim(0, self.map_width)
            ax.set_ylim(0, self.map_height)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # 3. Momentum analysis (third row)
        momentums = ['Positive', 'Neutral', 'Negative']
        momentum_colors = ['lime', 'gray', 'red']
        
        for i, momentum in enumerate(momentums):
            ax = fig.add_subplot(gs_time[2, i*2:(i+1)*2])
            ax.imshow(self.map_image, origin='upper', extent=[0, self.map_width, 0, self.map_height])
            
            df_momentum = df_pos[df_pos['momentum'] == momentum]
            df_momentum = self._df_with_world_coords(df_momentum)
            
            if len(df_momentum) > 0:
                obs_wards = df_momentum[df_momentum['ward_type'] == 'observer']
                sen_wards = df_momentum[df_momentum['ward_type'] == 'sentry']
                
                momentum_color = momentum_colors[i]
                
                if len(obs_wards) > 0:
                    ax.scatter(obs_wards['wx'], obs_wards['wy'], 
                             c=momentum_color, s=60, alpha=0.9, marker='o', 
                             edgecolors='black', linewidths=2, label='Observer')
                if len(sen_wards) > 0:
                    ax.scatter(sen_wards['wx'], sen_wards['wy'], 
                             c=momentum_color, s=60, alpha=0.9, marker='^', 
                             edgecolors='black', linewidths=2, label='Sentry')
            
            ax.set_title(f'{momentum} Momentum\n({len(df_momentum)} wards)', fontsize=12, color='white')
            ax.set_xlim(0, self.map_width)
            ax.set_ylim(0, self.map_height)
            ax.set_xticks([])
            ax.set_yticks([])
            if len(df_momentum) > 0:
                ax.legend(fontsize=10, framealpha=0.8)
        
        # 4. Combined overview (bottom row)
        ax_overview = fig.add_subplot(gs_time[3, :4])
        ax_overview.imshow(self.map_image, origin='upper', extent=[0, self.map_width, 0, self.map_height])
        
        dfo = self._df_with_world_coords(df_pos)
        obs_wards = dfo[dfo['ward_type'] == 'observer']
        sen_wards = dfo[dfo['ward_type'] == 'sentry']
        
        if len(obs_wards) > 0:
            ax_overview.scatter(obs_wards['wx'], obs_wards['wy'], 
                              c='cyan', s=80, alpha=0.8, marker='o', 
                              edgecolors='blue', linewidths=2, label='Observer')
        if len(sen_wards) > 0:
            ax_overview.scatter(sen_wards['wx'], sen_wards['wy'], 
                              c='yellow', s=80, alpha=0.8, marker='^', 
                              edgecolors='red', linewidths=2, label='Sentry')
        
        ax_overview.set_title(f'All Wards Overview\n({len(df_pos)} total wards)', fontsize=14, color='white')
        ax_overview.set_xlim(0, self.map_width)
        ax_overview.set_ylim(0, self.map_height)
        ax_overview.legend(fontsize=12, framealpha=0.8)
        
        # 5. Ward density heatmap (bottom right)
        ax_density = fig.add_subplot(gs_time[3, 4:])
        ax_density.imshow(self.map_image, origin='upper', extent=[0, self.map_width, 0, self.map_height])
        
        if len(df_pos) > 10:
            try:
                from scipy import ndimage
                dfo = self._df_with_world_coords(df_pos)
                heatmap, xedges, yedges = np.histogram2d(
                    dfo['wx'], dfo['wy'], 
                    bins=40, 
                    range=[[0, float(self.map_width)],[0, float(self.map_height)]]
                )
                heatmap_smooth = ndimage.gaussian_filter(heatmap, sigma=1.5)
                im = ax_density.imshow(heatmap_smooth.T, 
                                     extent=[0, self.map_width, 0, self.map_height], 
                                     alpha=0.7, cmap='hot', origin='upper')
                plt.colorbar(im, ax=ax_density, shrink=0.8)
            except ImportError:
                # Fallback if scipy not available
                dfo = self._df_with_world_coords(df_pos)
                ax_density.scatter(dfo['wx'], dfo['wy'], c=dfo['time'], 
                                 s=60, alpha=0.8, cmap='viridis')
        
        ax_density.set_title('Ward Density Heatmap', fontsize=14, color='white')
        ax_density.set_xlim(0, self.map_width)
        ax_density.set_ylim(0, self.map_height)
        
        # Add section labels
        fig.text(0.02, 0.79, 'TIME\nPROGRESSION', rotation=90, fontsize=16, fontweight='bold', 
                va='center', ha='center', color='white')
        fig.text(0.02, 0.59, 'GAME\nSTAGES', rotation=90, fontsize=16, fontweight='bold', 
                va='center', ha='center', color='white')
        fig.text(0.02, 0.39, 'TEAM\nMOMENTUM', rotation=90, fontsize=16, fontweight='bold', 
                va='center', ha='center', color='white')
        fig.text(0.02, 0.19, 'OVERVIEW &\nDENSITY', rotation=90, fontsize=16, fontweight='bold', 
                va='center', ha='center', color='white')
        
        plt.tight_layout()
        
        match_dir = self.output_dir / team_name.replace(' ', '_') / f"match_{match_id}"
        match_dir.mkdir(parents=True, exist_ok=True)
        file_path = match_dir / "ultimate_comprehensive_map_analysis.png"
        plt.savefig(file_path, dpi=300, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        plt.close()
        
        print(f"✅ Created ultimate comprehensive map: {file_path}")
    
    def process_all_matches_complete_map_analysis(self):
        """Generate all types of map-based analysis for all matches"""
        teams = {
            "Team_Falcons": 9247354,
            "PVISION": 9572001
        }
        
        print("🗺️ CREATING COMPLETE MAP-BASED ANALYSIS FOR ALL MATCHES")
        print("=" * 80)
        
        total_processed = 0
        
        for team_folder, team_id in teams.items():
            team_dir = self.output_dir / team_folder
            if not team_dir.exists():
                continue
            
            print(f"\n🎯 Processing {team_folder.replace('_', ' ')}...")
            
            match_folders = [d for d in team_dir.iterdir() if d.is_dir() and d.name.startswith('match_')]
            
            for match_folder in tqdm(sorted(match_folders), desc=f"{team_folder} matches"):
                match_id = int(match_folder.name.replace('match_', ''))
                
                print(f"  🔄 Creating complete map analysis for match {match_id}...")
                
                # Get match data
                match_data = self.get_match_details(match_id)
                if not match_data:
                    print(f"     ❌ Could not get match data for {match_id}")
                    continue
                
                # Extract enhanced ward data
                ward_data, progression = self.extract_enhanced_ward_data(match_data, team_id)
                
                if not ward_data:
                    print(f"     ⚠️  No ward data for match {match_id}")
                    continue
                
                # Create all map-based analyses
                self.create_time_windows_on_map(ward_data, match_id, team_folder.replace('_', ' '))
                self.create_time_window_reports(ward_data, match_id, team_folder.replace('_', ' '))
                self.create_game_stages_on_map(ward_data, match_id, team_folder.replace('_', ' '))
                self.create_momentum_on_map(ward_data, match_id, team_folder.replace('_', ' '))
                # Extra situational maps using objectives and net worth
                self.create_offense_defense_on_map(ward_data, progression, match_id, team_folder.replace('_', ' '), match_data, team_id)
                self.create_advantage_state_on_map(ward_data, match_id, team_folder.replace('_', ' '), match_data, team_id)
                self.create_ultimate_comprehensive_map_analysis(ward_data, match_id, team_folder.replace('_', ' '))
                
                total_processed += 1
        
        print(f"\n🎉 Complete! Created comprehensive map analysis for {total_processed} matches")


def main():
    """Generate complete map-based analysis"""
    analyzer = CompleteMapOverlayGenerator()
    analyzer.process_all_matches_complete_map_analysis()
    
    print(f"\n🗺️ COMPLETE MAP-BASED ANALYSIS FINISHED!")
    print(f"📁 Each match folder now contains:")
    print(f"   - time_windows_map_analysis.png")
    print(f"   - game_stages_map_analysis.png") 
    print(f"   - momentum_map_analysis.png")
    print(f"   - ultimate_comprehensive_map_analysis.png")
    print(f"   + all previous visualizations")


if __name__ == "__main__":
    main()
