#!/usr/bin/env python3
"""
Generate comprehensive CSV with ward positions grouped by time windows
For better debugging and analysis of the coordinate transformation fix.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def load_debug_data(debug_dir):
    """Load all debug CSV files and combine them."""
    debug_path = Path(debug_dir)
    if not debug_path.exists():
        print(f"Debug directory {debug_dir} not found")
        return None
    
    csv_files = list(debug_path.glob("coords_debug_*.csv"))
    if not csv_files:
        print(f"No debug CSV files found in {debug_dir}")
        return None
    
    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

def generate_time_window_summary(df, output_dir):
    """Generate comprehensive CSV grouped by time windows."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Group by time window and calculate statistics
    grouped = df.groupby(['time_window', 'ward_type']).agg({
        'x_original': ['count', 'mean', 'std', 'min', 'max'],
        'y_original': ['mean', 'std', 'min', 'max'],
        'wx': ['mean', 'std', 'min', 'max'],
        'wy': ['mean', 'std', 'min', 'max'],
        'coord_type': 'first',
        'transform_enabled': 'first',
        'transform_scale': 'first',
    }).round(2)
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns]
    grouped = grouped.reset_index()
    
    # Save summary
    summary_path = output_path / "ward_positions_by_time_window_summary.csv"
    grouped.to_csv(summary_path, index=False)
    print(f"📊 Summary CSV saved: {summary_path}")
    
    # Save detailed data
    detailed = df[['match_id', 'time_window', 'ward_type', 'time', 
                   'x_original', 'y_original', 'x_processed', 'y_processed',
                   'wx', 'wy', 'coord_type', 'transform_enabled', 'transform_scale']].copy()
    detailed = detailed.sort_values(['time_window', 'ward_type', 'time'])
    
    detail_path = output_path / "ward_positions_by_time_window_detailed.csv"
    detailed.to_csv(detail_path, index=False)
    print(f"📊 Detailed CSV saved: {detail_path}")
    
    # Create time window analysis
    time_windows = sorted(df['time_window'].unique())
    analysis_data = []
    
    for tw in time_windows:
        tw_data = df[df['time_window'] == tw]
        
        for ward_type in ['observer', 'sentry']:
            ward_data = tw_data[tw_data['ward_type'] == ward_type]
            
            if len(ward_data) > 0:
                analysis_data.append({
                    'time_window': tw,
                    'ward_type': ward_type,
                    'count': len(ward_data),
                    'x_orig_mean': ward_data['x_original'].mean(),
                    'y_orig_mean': ward_data['y_original'].mean(),
                    'x_orig_spread': ward_data['x_original'].std(),
                    'y_orig_spread': ward_data['y_original'].std(),
                    'wx_mean': ward_data['wx'].mean(),
                    'wy_mean': ward_data['wy'].mean(),
                    'wx_spread': ward_data['wx'].std(),
                    'wy_spread': ward_data['wy'].std(),
                    'coord_coverage_x': ward_data['x_original'].max() - ward_data['x_original'].min(),
                    'coord_coverage_y': ward_data['y_original'].max() - ward_data['y_original'].min(),
                    'pixel_coverage_x': ward_data['wx'].max() - ward_data['wx'].min(),
                    'pixel_coverage_y': ward_data['wy'].max() - ward_data['wy'].min(),
                })
    
    analysis_df = pd.DataFrame(analysis_data).round(2)
    analysis_path = output_path / "ward_positions_analysis.csv"
    analysis_df.to_csv(analysis_path, index=False)
    print(f"📊 Analysis CSV saved: {analysis_path}")
    
    return grouped, detailed, analysis_df

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_comprehensive_csv.py <debug_dir> [output_dir]")
        return
    
    debug_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output2/comprehensive_analysis"
    
    # Load data
    df = load_debug_data(debug_dir)
    if df is None:
        return
    
    print(f"Loaded {len(df)} ward position records")
    print(f"Time windows: {sorted(df['time_window'].unique())}")
    print(f"Ward types: {sorted(df['ward_type'].unique())}")
    print(f"Coordinate types: {sorted(df['coord_type'].unique())}")
    
    # Generate comprehensive analysis
    summary, detailed, analysis = generate_time_window_summary(df, output_dir)
    
    print("\n=== COORDINATE TRANSFORMATION ANALYSIS ===")
    print("Original coordinates range:")
    print(f"  X: {df['x_original'].min():.1f} to {df['x_original'].max():.1f}")
    print(f"  Y: {df['y_original'].min():.1f} to {df['y_original'].max():.1f}")
    
    print("Processed coordinates range:")
    print(f"  X: {df['x_processed'].min():.1f} to {df['x_processed'].max():.1f}")
    print(f"  Y: {df['y_processed'].min():.1f} to {df['y_processed'].max():.1f}")
    
    print("Final pixel coordinates range:")
    print(f"  WX: {df['wx'].min():.1f} to {df['wx'].max():.1f}")
    print(f"  WY: {df['wy'].min():.1f} to {df['wy'].max():.1f}")
    
    print(f"\nTransform config: scale={df['transform_scale'].iloc[0]}, enabled={df['transform_enabled'].iloc[0]}")

if __name__ == "__main__":
    main()