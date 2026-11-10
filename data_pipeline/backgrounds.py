"""Step 3: background scenes from Wikimedia Commons, with per-image license metadata.

    python -m data_pipeline.backgrounds fetch [--regions japan italy ...]   # needs internet
    python -m data_pipeline.backgrounds audit                               # CLIP region/attribute check
    python -m data_pipeline.backgrounds report                              # counts + contact sheet

fetch: crawls the configured categories (OPEN ITEM: `backgrounds.categories`) to
`category_depth`, keeps bitmap photos above the size limits whose LicenseShortName is on the
whitelist, downloads a `thumb_width` rendition and writes backgrounds/licenses.csv
(license, author, URL per image). Images without a usable license are dropped and logged.
Region label = the region whose category the file was found in; files found under two
regions are dropped (ambiguous_region).

audit: CLIP zero-shot (not an evaluation-panel model). Keeps an image if its labelled region
is in CLIP's top-k over all regions and no "reject" prompt (map, food close-up, portrait,
drawing) beats the reference prompt. Adds indoor/outdoor and water attributes. Writes
backgrounds/index.jsonl (accepted only). A human-check sample comes from audit.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .classes import REGIONS, region_name
from .utils import (Config, RejectionLog, derive_seed, get_logger, load_config, pick_device,
                    read_jsonl, seed_everything, write_json, write_jsonl)

API = "https://commons.wikimedia.org/w/api.php"
LICENSE_FIELDS = ["bg_id", "region", "file", "title", "author", "license", "license_url",
                  "source_page", "file_url", "category"]


def bg_id_for(title: str) -> str:
    return "wm_" + hashlib.sha1(title.encode()).hexdigest()[:10]


def strip_html(s) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", s or "")).strip()


class Commons:
    def __init__(self, cfg: Config):
        bc = cfg["backgrounds"]
        self.ua, self.sleep = bc["user_agent"], bc["request_sleep_s"]

    def get(self, params: dict) -> dict:
        url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
        req = urllib.request.Request(url, headers={"User-Agent": self.ua})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    out = json.loads(r.read().decode())
                time.sleep(self.sleep)
                return out
            except Exception:  # noqa: BLE001
                time.sleep(2 ** attempt * self.sleep)
        raise RuntimeError(f"API failed: {params}")

    def download(self, url: str, path: Path) -> None:
        req = urllib.request.Request(url, headers={"User-Agent": self.ua})
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
        time.sleep(self.sleep)

    def subcats(self, cat: str) -> list[str]:
        out, cont = [], {}
        while True:
            d = self.get({"action": "query", "list": "categorymembers", "cmtitle": cat,
                          "cmtype": "subcat", "cmlimit": 500, **cont})
            out += [m["title"] for m in d.get("query", {}).get("categorymembers", [])]
            if "continue" not in d:
                return out
            cont = d["continue"]

    def files(self, cat: str, thumb_width: int):
        cont = {}
        while True:
            d = self.get({"action": "query", "generator": "categorymembers", "gcmtitle": cat,
                          "gcmtype": "file", "gcmlimit": 50, "prop": "imageinfo",
                          "iiprop": "url|size|mime|extmetadata", "iiurlwidth": thumb_width, **cont})
            for p in sorted(d.get("query", {}).get("pages", {}).values(), key=lambda p: p["title"]):
                yield p
            if "continue" not in d:
                return
            cont = d["continue"]


def license_ok(lic: str, bc: dict) -> bool:
    l = lic.lower().strip()
    if not l:
        return False
    if any(s in re.split(r"[\s\-]", l) for s in bc["rejected_license_substrings"]) or "fair use" in l:
        return False
    return any(l.startswith(p) for p in bc["allowed_license_prefixes"])


def fetch(cfg: Config, regions: list[str]) -> dict:
    log = get_logger("backgrounds", cfg)
    bc = cfg["backgrounds"]
    api = Commons(cfg)
    root = cfg.path("backgrounds_dir")
    (root / "img").mkdir(parents=True, exist_ok=True)
    rej = RejectionLog(root / "fetch_rejections.jsonl")
    lic_path = cfg.path("backgrounds_licenses")
    existing = {r["title"]: r for r in csv.DictReader(open(lic_path))} if lic_path.exists() else {}
    # pass 1: candidates per region (title -> info); detect cross-region files
    cand: dict[str, dict] = {}
    regions_of = defaultdict(set)
    for reg in regions:
        n_reg = 0
        cats, frontier = [], list(bc["categories"][reg])
        for depth in range(bc["category_depth"] + 1):
            cats += frontier
            if depth < bc["category_depth"]:
                frontier = sorted({s for c in frontier for s in api.subcats(c)} - set(cats))
        for cat in cats:
            for p in api.files(cat, bc["thumb_width"]):
                t = p["title"]
                regions_of[t].add(reg)
                if t in cand:
                    continue
                ii = (p.get("imageinfo") or [{}])[0]
                meta = ii.get("extmetadata", {})
                w, h = ii.get("width", 0), ii.get("height", 0)
                lic = strip_html(meta.get("LicenseShortName", {}).get("value"))
                reason = None
                if ii.get("mime") not in ("image/jpeg", "image/png"):
                    reason = "bad_mime"
                elif w < bc["min_width"] or h < bc["min_height"]:
                    reason = "too_small"
                elif max(w / h, h / w) > bc["max_aspect"]:
                    reason = "bad_aspect"
                elif not lic:
                    reason = "license_missing"
                elif not license_ok(lic, bc):
                    reason = "license_not_allowed"
                if reason:
                    rej(t, reason, region=reg, license=lic)
                    cand[t] = None
                    continue
                cand[t] = {
                    "bg_id": bg_id_for(t), "region": reg, "title": t,
                    "author": strip_html(meta.get("Artist", {}).get("value")) or "unknown",
                    "license": lic, "license_url": strip_html(meta.get("LicenseUrl", {}).get("value")),
                    "source_page": ii.get("descriptionurl", ""),
                    "file_url": ii.get("thumburl") or ii.get("url"), "category": cat,
                }
                n_reg += 1
        log.info("%s: %d categories, %d license-OK candidates", reg, len(cats), n_reg)
    # pass 2: drop ambiguous, download up to an oversampled per-region quota (seeded order)
    seed = derive_seed(cfg["seed"], "backgrounds")
    quota = int(bc["per_region_target"] * bc["oversample"])
    by_reg = defaultdict(list)
    for t, c in cand.items():
        if c is None:
            continue
        if len(regions_of[t]) > 1:
            rej(t, "ambiguous_region", regions=sorted(regions_of[t]))
            continue
        by_reg[c["region"]].append(c)
    rows = list(existing.values())
    have = Counter(r["region"] for r in rows)
    import random

    for reg in regions:
        lst = sorted(by_reg[reg], key=lambda c: c["title"])
        random.Random(derive_seed(seed, reg)).shuffle(lst)
        for c in lst:
            if have[reg] >= quota:
                break
            if c["title"] in existing:
                continue
            path = root / "img" / f"{c['bg_id']}.jpg"
            try:
                api.download(c["file_url"], path)
                Image.open(path).verify()
            except Exception as e:  # noqa: BLE001
                rej(c["title"], "download_failed", error=str(e)[:200])
                path.unlink(missing_ok=True)
                continue
            rows.append({**c, "file": f"img/{c['bg_id']}.jpg"})
            have[reg] += 1
        log.info("%s: %d downloaded (quota %d)", reg, have[reg], quota)
    rej.close()
    with open(lic_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LICENSE_FIELDS)
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["bg_id"]):
            w.writerow({k: r.get(k, "") for k in LICENSE_FIELDS})
    log.info("licenses: %d rows -> %s; rejections %s", len(rows), lic_path, rej.counts)
    return {"downloaded": dict(have), "rejections": rej.counts}


def import_pilot(cfg: Config, pilot_dir: Path) -> int:
    """Register the pilot's 30 landmark photos (pilot credits.csv) in licenses.csv format.
    Used for local demos; the release should be built from `fetch`."""
    root = cfg.path("backgrounds_dir")
    (root / "img").mkdir(parents=True, exist_ok=True)
    rows = []
    for r in csv.DictReader(open(pilot_dir / "credits.csv")):
        src = Path(r["file"])
        src = src if src.is_absolute() else pilot_dir.parent / src
        if not src.exists():
            continue
        bid = bg_id_for(r["title"])
        dst = root / "img" / f"{bid}.jpg"
        dst.write_bytes(src.read_bytes())
        rows.append({"bg_id": bid, "region": r["country"], "file": f"img/{bid}.jpg", "title": r["title"],
                     "author": r["author"], "license": r["license"], "license_url": "",
                     "source_page": r["source_page"], "file_url": "", "category": f"pilot:{r['query']}"})
    with open(cfg.path("backgrounds_licenses"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LICENSE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


# ---------------------------------------------------------------- audit
def audit(cfg: Config) -> dict:
    from .crops import ClipClassifier

    log = get_logger("backgrounds", cfg)
    bc, ac = cfg["backgrounds"], cfg["backgrounds"]["audit"]
    seed_everything(derive_seed(cfg["seed"], "bg_audit"), log)
    device = pick_device(cfg["crops"]["device"])
    root = cfg.path("backgrounds_dir")
    lic = list(csv.DictReader(open(cfg.path("backgrounds_licenses"))))
    rej = RejectionLog(cfg.path("backgrounds_rejections"))
    names = [region_name(r) for r in REGIONS]
    region_clf = ClipClassifier(ac["model"], names, ac["region_prompt"], device)
    # generic prompt classifiers reuse the same model weights via a raw prompt list
    other = ClipClassifier(ac["model"], [ac["reject_reference_prompt"], *ac["reject_prompts"]], "{}", device)
    attrs = {k: ClipClassifier(ac["model"], v, "{}", device) for k, v in ac["attribute_prompts"].items()}
    rows = []
    for r in lic:
        if not license_ok(r["license"], bc) and not r["category"].startswith("pilot:"):
            rej(r["bg_id"], "license_not_allowed", license=r["license"])
            continue
        img = Image.open(root / r["file"]).convert("RGB")
        p = region_clf.probs(img)
        rank = int((p > p[REGIONS.index(r["region"])]).sum()) + 1
        po = other.probs(img)
        worst = int(po[1:].argmax())
        if po[1 + worst] > po[0]:
            rej(r["bg_id"], "not_a_scene", clip_label=ac["reject_prompts"][worst], p=round(float(po[1 + worst]), 3))
            continue
        if rank > ac["region_topk_keep"]:
            rej(r["bg_id"], "clip_region_rank", rank=rank, clip_top1=REGIONS[int(p.argmax())])
            continue
        row = {"bg_id": r["bg_id"], "region": r["region"], "path": r["file"], "license": r["license"],
               "clip_region_rank": rank, "clip_region_p": round(float(p[REGIONS.index(r["region"])]), 4),
               "clip_top1": REGIONS[int(p.argmax())]}
        for k, clf in attrs.items():
            row[k] = bool(clf.probs(img)[0] > 0.5)
        rows.append(row)
    rej.close()
    # cap at per_region_target (seeded) after the audit
    import random

    by = defaultdict(list)
    for r in rows:
        by[r["region"]].append(r)
    final = []
    for reg in sorted(by):
        lst = sorted(by[reg], key=lambda r: r["bg_id"])
        random.Random(derive_seed(cfg["seed"], "bg_cap", reg)).shuffle(lst)
        final += lst[: bc["per_region_target"]]
    write_jsonl(cfg.path("backgrounds_index"), sorted(final, key=lambda r: r["bg_id"]))
    stats = {"accepted": dict(Counter(r["region"] for r in final)), "rejections": rej.counts}
    write_json(root / "audit_stats.json", stats)
    log.info("background audit: %s", stats)
    return stats


def report(cfg: Config) -> None:
    root = cfg.path("backgrounds_dir")
    rows = read_jsonl(cfg.path("backgrounds_index"))
    by = defaultdict(list)
    for r in rows:
        by[r["region"]].append(r)
    for reg in REGIONS:
        print(f"  {reg:9s} {len(by[reg]):4d}")
    print(f"  {'TOTAL':9s} {len(rows):4d}")
    tw, th, per = 160, 120, 12
    lw = 110
    sheet = Image.new("RGB", (lw + per * tw, max(1, len(by)) * th), "white")
    d = ImageDraw.Draw(sheet)
    for i, reg in enumerate(sorted(by)):
        d.text((6, i * th + th // 2), f"{reg} ({len(by[reg])})", fill="black")
        for j, r in enumerate(by[reg][:per]):
            im = Image.open(root / r["path"]).convert("RGB")
            im.thumbnail((tw - 4, th - 4))
            sheet.paste(im, (lw + j * tw, i * th))
    sheet.save(root / "contact_sheet.jpg", quality=88)
    print("contact sheet:", root / "contact_sheet.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "audit", "report", "import-pilot"])
    ap.add_argument("--config")
    ap.add_argument("--regions", nargs="*")
    ap.add_argument("--pilot_dir", default="backgrounds", help="for import-pilot")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.cmd == "fetch":
        regs = args.regions or REGIONS
        assert set(regs) <= set(REGIONS), set(regs) - set(REGIONS)
        fetch(cfg, regs)
    elif args.cmd == "audit":
        audit(cfg)
    elif args.cmd == "import-pilot":
        n = import_pilot(cfg, Path(args.pilot_dir).resolve())
        print(f"imported {n} pilot backgrounds")
    else:
        report(cfg)


if __name__ == "__main__":
    main()
