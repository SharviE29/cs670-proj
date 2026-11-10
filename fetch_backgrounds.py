"""Download landmark background photos from Wikimedia Commons (freely licensed).

Creates backgrounds/<country>/*.jpg, plus:
  backgrounds/credits.csv   - title, author, license, source page for every image
                              (cite these in your report/poster)
  backgrounds/preview.jpg   - contact sheet, so you can delete bad picks quickly

Usage (run on your own machine or Colab, needs internet):
  python fetch_backgrounds.py                      # all 10 countries, ~2 images per landmark
  python fetch_backgrounds.py --countries japan italy france mexico india --per_query 2

After downloading, LOOK at preview.jpg and delete anything that isn't a clear,
outdoor, recognisable scene (maps, interiors, close-ups, people's faces, food).
"""

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia asks every client to send a descriptive User-Agent.
UA = "CS670-course-project-background-fetcher/1.0 (educational use)"

LANDMARKS = {
    "japan":    ["Kinkaku-ji", "Fushimi Inari Taisha torii", "Chureito Pagoda Mount Fuji"],
    "italy":    ["Colosseum Rome exterior", "Grand Canal Venice", "Florence Cathedral Duomo"],
    "france":   ["Eiffel Tower Champ de Mars", "Louvre Pyramid", "Montmartre street Paris"],
    "mexico":   ["Chichen Itza El Castillo", "Zocalo Mexico City Metropolitan Cathedral",
                 "Guanajuato city colorful houses"],
    "india":    ["Taj Mahal", "Hawa Mahal Jaipur", "India Gate New Delhi"],
    "china":    ["Great Wall Mutianyu", "Forbidden City Beijing", "Temple of Heaven Beijing"],
    "korea":    ["Gyeongbokgung", "Bukchon Hanok Village", "Bulguksa"],
    "thailand": ["Wat Arun", "Grand Palace Bangkok", "Wat Phra That Doi Suthep"],
    "vietnam":  ["Ha Long Bay", "Hoi An Ancient Town", "Hanoi Old Quarter street"],
    "spain":    ["Sagrada Familia", "Plaza de Espana Seville", "Alhambra Granada"],
}


def api_get(params):
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def download(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        path.write_bytes(r.read())


def strip_html(s):
    return re.sub(r"<[^>]+>", "", s or "").strip()


def good_candidate(ii, min_width=900):
    """Landscape JPEG/PNG photos, big enough to use as a 1024x768 background."""
    if ii.get("mime") not in ("image/jpeg", "image/png"):
        return False
    w, h = ii.get("width", 0), ii.get("height", 0)
    return w >= min_width and w >= 1.2 * h  # landscape


def search(query, n_candidates=15, thumb_width=1280):
    data = api_get({
        "action": "query", "format": "json",
        "generator": "search", "gsrsearch": f"{query} filetype:bitmap",
        "gsrnamespace": 6, "gsrlimit": n_candidates,
        "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": thumb_width,
    })
    pages = sorted(data.get("query", {}).get("pages", {}).values(),
                   key=lambda p: p.get("index", 0))  # keep search-relevance order
    out = []
    for p in pages:
        ii = (p.get("imageinfo") or [{}])[0]
        if not good_candidate(ii):
            continue
        meta = ii.get("extmetadata", {})
        out.append({
            "title": p["title"],
            "url": ii.get("thumburl") or ii["url"],
            "page": ii.get("descriptionurl", ""),
            "author": strip_html(meta.get("Artist", {}).get("value")),
            "license": strip_html(meta.get("LicenseShortName", {}).get("value")),
        })
    return out


def contact_sheet(root, countries, out_path, thumb=(240, 160)):
    from PIL import Image, ImageDraw
    rows = []
    for c in countries:
        files = sorted((root / c).glob("*.jpg"))
        if files:
            rows.append((c, files))
    if not rows:
        return
    cols = max(len(f) for _, f in rows)
    label_w = 110
    sheet = Image.new("RGB", (label_w + cols * thumb[0], len(rows) * thumb[1]), "white")
    d = ImageDraw.Draw(sheet)
    for r, (c, files) in enumerate(rows):
        d.text((8, r * thumb[1] + thumb[1] // 2), c, fill="black")
        for k, f in enumerate(files):
            im = Image.open(f).convert("RGB")
            im.thumbnail(thumb)
            sheet.paste(im, (label_w + k * thumb[0], r * thumb[1]))
    sheet.save(out_path, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="backgrounds")
    ap.add_argument("--countries", nargs="*", default=list(LANDMARKS))
    ap.add_argument("--per_query", type=int, default=2, help="images per landmark")
    ap.add_argument("--sleep", type=float, default=1.0, help="pause between requests")
    args = ap.parse_args()

    root = Path(args.out)
    credits_path = root / "credits.csv"
    root.mkdir(parents=True, exist_ok=True)
    new_file = not credits_path.exists()
    with open(credits_path, "a", newline="") as fcsv:
        w = csv.writer(fcsv)
        if new_file:
            w.writerow(["file", "country", "query", "title", "author", "license", "source_page"])
        for country in args.countries:
            (root / country).mkdir(exist_ok=True)
            for q in LANDMARKS[country]:
                try:
                    cands = search(q)
                except Exception as e:  # noqa: BLE001
                    print(f"[{country}] search failed for '{q}': {e}")
                    continue
                got = 0
                for c in cands:
                    if got >= args.per_query:
                        break
                    slug = re.sub(r"[^a-z0-9]+", "_", q.lower()).strip("_")
                    path = root / country / f"{slug}_{got}.jpg"
                    try:
                        download(c["url"], path)
                    except Exception as e:  # noqa: BLE001
                        print(f"  download failed: {c['title']}: {e}")
                        continue
                    w.writerow([str(path), country, q, c["title"], c["author"],
                                c["license"], c["page"]])
                    got += 1
                    time.sleep(args.sleep)
                print(f"[{country}] {q}: {got} image(s)")
                time.sleep(args.sleep)

    try:
        contact_sheet(root, args.countries, root / "preview.jpg")
        print(f"\nPreview: {root / 'preview.jpg'}  <- check it and delete bad images")
    except Exception as e:  # noqa: BLE001
        print(f"(preview skipped: {e})")
    print(f"Credits: {credits_path}")


if __name__ == "__main__":
    main()
