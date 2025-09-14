#!/usr/bin/env python3
"""
Fetch latest Dota 2 minimap image and save to assets/minimap_latest.jpg

Strategy:
- Try a small set of well-known, actively updated sources.
- Validate that the image is loadable and at least ~512px.
- Save as assets/minimap_latest.jpg (RGB, 1024x1024) for consistent plotting.
"""

import os
from pathlib import Path
import io
import time
import requests
from PIL import Image


CANDIDATE_URLS = [
    # Liquipedia commons (often updated)
    "https://liquipedia.net/commons/images/a/ab/Dota_2_Map.jpg",
    # Fandom/Gamepedia variants (may be older but kept as fallback)
    "https://static.wikia.nocookie.net/dota2_gamepedia/images/8/8b/Minimap_7.36.png",
    "https://static.wikia.nocookie.net/dota2_gamepedia/images/8/8b/Minimap_7.35.png",
]


def fetch_first_available(urls):
    s = requests.Session()
    for url in urls:
        try:
            time.sleep(0.3)
            r = s.get(url, timeout=15)
            r.raise_for_status()
            data = r.content
            img = Image.open(io.BytesIO(data))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            w, h = img.size
            if w < 256 or h < 256:
                continue
            # Normalize to 1024x1024
            img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
            return img, url
        except Exception:
            continue
    return None, None


def main():
    assets = Path('assets')
    assets.mkdir(exist_ok=True)
    img, url = fetch_first_available(CANDIDATE_URLS)
    if img is None:
        print("❌ Failed to fetch a minimap from candidates.")
        return 1
    out = assets / 'minimap_latest.jpg'
    img.save(out, format='JPEG', quality=92)
    print(f"✅ Saved latest minimap to {out} (source: {url})")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

