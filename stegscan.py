#!/usr/bin/env python3
"""
stegscan.py - heuristic steganography detector for images.

Analyze a single image, several images, or whole folders:

    python stegscan.py photo.png
    python stegscan.py ./images -r
    python stegscan.py ./images -r --json report.json --csv report.csv -v

Checks performed
  Structural (file level)
    * Data appended after the official end of image (PNG IEND, JPEG EOI,
      GIF trailer, BMP/WebP declared size) - the most common "hide a zip in a
      picture" trick.
    * Embedded file signatures (zip, rar, 7z, pdf, elf, ...) in trailing data
      and in metadata segments.
    * Non-standard PNG chunks, oversized text chunks, base64-looking blobs,
      oversized JPEG comments.
    * Strings left behind by known stego tools.
    * Extension / real-format mismatch.
  Pixel level (lossless formats; skipped for JPEG unless --force-pixel-tests)
    * Chi-square "pairs of values" test (Westfeld & Pfitzenmaier) run on
      growing prefixes of each channel, so partially filled images are caught.
    * LSB extraction test: pulls the least-significant bits in several
      channel orders / bit orders and checks whether they decode to a file
      signature or printable text (what zsteg-style tools look for).

Limitations
  * Heuristics, not proof. "NO INDICATORS" never proves an image is clean.
  * Cannot detect DCT-domain JPEG stego (steghide, F5, OutGuess, JSteg) or
    well-implemented adaptive/encrypted embedding (e.g. HUGO, WOW, S-UNIWARD).
    Those need statistical/ML steganalysis (e.g. SRM features + classifier).

Requires: Python 3.8+, Pillow, numpy   (scipy optional, used if present)
"""

import argparse
import csv
import json
import math
import os
import re
import struct
import sys
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
WEIGHTS = {"info": 0, "low": 1, "medium": 3, "high": 5}

SIGNATURES = {
    b"PK\x03\x04": "ZIP/DOCX/JAR",
    b"PK\x05\x06": "ZIP (empty)",
    b"Rar!\x1a\x07": "RAR",
    b"7z\xbc\xaf\x27\x1c": "7-Zip",
    b"\x1f\x8b\x08": "GZIP",
    b"BZh": "BZIP2",
    b"\xfd7zXZ\x00": "XZ",
    b"%PDF-": "PDF",
    b"\x7fELF": "ELF",
    b"MZ": "PE/DOS executable",
    b"\x89PNG\r\n\x1a\n": "PNG",
    b"\xff\xd8\xff": "JPEG",
    b"GIF8": "GIF",
    b"SQLite format 3": "SQLite",
    b"OggS": "OGG",
    b"ID3": "MP3",
    b"RIFF": "RIFF",
    b"-----BEGIN": "PEM/PGP block",
    b"{\\rtf": "RTF",
    b"\xd0\xcf\x11\xe0": "OLE2 (old Office)",
}
# Signatures that legitimately appear inside metadata (thumbnails etc.)
NOISY = {"JPEG", "PNG", "GIF", "RIFF", "MP3", "OLE2 (old Office)"}

TOOL_STRINGS = [
    b"steghide", b"openstego", b"outguess", b"stegosuite", b"xsteg",
    b"digital invisible ink", b"silenteye", b"stegano", b"f5 steganography",
    b"camouflage", b"invisible secrets", b"deepsound", b"snow.exe",
]

STD_PNG = {
    "IHDR", "PLTE", "IDAT", "IEND", "cHRM", "gAMA", "iCCP", "sBIT", "sRGB",
    "bKGD", "hIST", "tRNS", "pHYs", "sPLT", "tIME", "iTXt", "tEXt", "zTXt",
    "eXIf", "acTL", "fcTL", "fdAT", "cICP", "mDCV", "cLLI", "oFFs", "pCAL",
    "sCAL", "gIFg", "gIFx", "gIFt", "sTER", "dSIG",
}


@dataclass
class Finding:
    severity: str  # info | low | medium | high
    check: str
    detail: str


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def entropy(b: bytes) -> float:
    if not b:
        return 0.0
    c = np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256)
    p = c[c > 0] / len(b)
    return float(-(p * np.log2(p)).sum())


def printable_ratio(b: bytes) -> float:
    if not b:
        return 0.0
    ok = sum(1 for x in b if 32 <= x <= 126 or x in (9, 10, 13))
    return ok / len(b)


def magic_at_start(b: bytes):
    for sig, name in SIGNATURES.items():
        if b.startswith(sig):
            return name
    return None


def find_sigs(b: bytes, skip_noisy=False):
    """Find signatures (>=4 bytes, to limit false hits) anywhere in b."""
    hits = []
    for sig, name in SIGNATURES.items():
        if len(sig) < 4 or (skip_noisy and name in NOISY):
            continue
        i = b.find(sig)
        if i >= 0:
            hits.append((name, i))
    return hits


def looks_base64(b: bytes) -> bool:
    return len(b) >= 200 and re.fullmatch(rb"[A-Za-z0-9+/=\s]+", b) is not None


def chi_sf(x: float, k: int) -> float:
    """Survival function of chi-square distribution."""
    try:
        from scipy.stats import chi2
        return float(chi2.sf(x, k))
    except Exception:
        z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
        return 0.5 * math.erfc(z / math.sqrt(2))


def detect_format(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if data.startswith(b"\xff\xd8"):
        return "JPEG"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "GIF"
    if data.startswith(b"BM"):
        return "BMP"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "TIFF"
    return "UNKNOWN"


EXT_FOR_FMT = {
    "PNG": {".png"}, "JPEG": {".jpg", ".jpeg"}, "GIF": {".gif"},
    "BMP": {".bmp"}, "WEBP": {".webp"}, "TIFF": {".tif", ".tiff"},
}


# --------------------------------------------------------------------------
# container parsers: each returns (segments, end_offset_of_valid_data)
# --------------------------------------------------------------------------
def png_parse(data):
    pos, chunks, end = 8, [], None
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        name = data[pos + 4:pos + 8].decode("latin1")
        chunks.append((name, length, pos + 8))
        pos += 12 + length
        if name == "IEND":
            end = min(pos, len(data))
            break
    return chunks, end


def jpeg_parse(data):
    n, pos, segs, end = len(data), 2, [], None
    while pos + 4 <= n:
        if data[pos] != 0xFF:
            break
        marker = data[pos + 1]
        if marker == 0xFF:
            pos += 1
            continue
        if marker == 0xD9:
            end = pos + 2
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        length = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        segs.append((marker, pos + 4, max(length - 2, 0)))
        pos += 2 + length
        if marker == 0xDA:  # skip entropy-coded data up to next real marker
            i = pos
            while True:
                i = data.find(b"\xff", i)
                if i < 0 or i + 1 >= n:
                    pos = n
                    break
                b = data[i + 1]
                if b == 0xFF:
                    i += 1
                elif b == 0x00 or 0xD0 <= b <= 0xD7:
                    i += 2
                else:
                    pos = i
                    break
    return segs, end


def gif_end(data):
    if len(data) < 13:
        return None
    flags, pos = data[10], 13
    if flags & 0x80:
        pos += 3 * (2 ** ((flags & 7) + 1))

    def skip_sub(p):
        while p < len(data):
            s = data[p]
            p += 1
            if s == 0:
                return p
            p += s
        return None

    while pos is not None and pos < len(data):
        b = data[pos]
        if b == 0x3B:
            return pos + 1
        if b == 0x21:
            pos = skip_sub(pos + 2)
        elif b == 0x2C:
            if pos + 10 > len(data):
                return None
            f = data[pos + 9]
            pos += 10
            if f & 0x80:
                pos += 3 * (2 ** ((f & 7) + 1))
            pos = skip_sub(pos + 1)
        else:
            return None
    return None


# --------------------------------------------------------------------------
# structural checks
# --------------------------------------------------------------------------
def check_trailing(data, end, findings):
    if end is None or end >= len(data):
        return
    t = data[end:]
    if not t.strip(b"\x00\r\n \t"):
        findings.append(Finding("info", "trailing-padding",
                                f"{len(t)} bytes of null/whitespace padding after end of image"))
        return
    ent, pr, sigs = entropy(t), printable_ratio(t[:4096]), find_sigs(t)
    sev, notes = ("low" if len(t) < 8 else "medium"), []
    if sigs:
        sev = "high"
        notes.append("signatures: " + ", ".join(f"{n}@+{o}" for n, o in sigs))
    if len(t) >= 64 and ent > 7.5:
        sev = "high"
        notes.append("high entropy (encrypted/compressed?)")
    if len(t) >= 16 and pr > 0.9:
        sev = "high"
        notes.append("mostly printable text")
    findings.append(Finding(
        sev, "trailing-data",
        f"{len(t)} bytes after end of image data (offset {end}); entropy={ent:.2f} bits/byte"
        + ("; " + "; ".join(notes) if notes else "")))


def check_png(data, findings):
    chunks, end = png_parse(data)
    if end is None:
        findings.append(Finding("info", "png-truncated", "no IEND chunk found"))
    for name, length, off in chunks:
        payload = data[off:off + length]
        if name not in STD_PNG:
            findings.append(Finding("medium", "png-private-chunk",
                                    f"non-standard chunk '{name}' ({length} bytes) at offset {off - 8}"))
        if name in ("tEXt", "zTXt", "iTXt"):
            if length > 1024:
                findings.append(Finding("low", "png-large-text",
                                        f"{name} chunk of {length} bytes"))
            if looks_base64(payload):
                findings.append(Finding("medium", "png-base64-text",
                                        f"{name} chunk looks like a base64 blob ({length} bytes)"))
        if name not in ("IDAT", "fdAT", "IEND"):
            for sname, o in find_sigs(payload, skip_noisy=True):
                findings.append(Finding("high", "png-chunk-signature",
                                        f"{sname} signature inside '{name}' chunk (+{o})"))
    check_trailing(data, end, findings)


def check_jpeg(data, findings):
    segs, end = jpeg_parse(data)
    if end is None:
        findings.append(Finding("info", "jpeg-no-eoi", "no EOI marker found (truncated or damaged)"))
    for marker, start, length in segs:
        payload = data[start:start + length]
        if marker == 0xFE:
            if length > 256:
                findings.append(Finding("low", "jpeg-large-comment", f"COM segment of {length} bytes"))
            if looks_base64(payload):
                findings.append(Finding("medium", "jpeg-base64-comment", "COM segment looks like base64"))
        if 0xE0 <= marker <= 0xEF or marker == 0xFE:
            for sname, o in find_sigs(payload, skip_noisy=True):
                findings.append(Finding("high", "jpeg-segment-signature",
                                        f"{sname} signature inside segment FF{marker:02X} (+{o})"))
    check_trailing(data, end, findings)


def check_gif(data, findings):
    check_trailing(data, gif_end(data), findings)


def check_bmp(data, findings):
    if len(data) >= 6:
        declared = struct.unpack("<I", data[2:6])[0]
        if 0 < declared < len(data):
            check_trailing(data, declared, findings)


def check_webp(data, findings):
    if len(data) >= 12:
        declared = struct.unpack("<I", data[4:8])[0] + 8
        if declared < len(data):
            check_trailing(data, declared, findings)


def check_strings(data, findings):
    low = data.lower()
    for s in TOOL_STRINGS:
        if s in low:
            findings.append(Finding("medium", "tool-string",
                                    f"found string '{s.decode()}' in file"))


# --------------------------------------------------------------------------
# pixel-level checks
# --------------------------------------------------------------------------
def chi_square_pov(samples):
    """Return (p, chi2, bins) or None. High p => LSBs look 'equalised' (embedded)."""
    h = np.bincount(samples, minlength=256).astype(np.float64)
    even, odd = h[0::2], h[1::2]
    exp = (even + odd) / 2
    mask = exp >= 5
    k = int(mask.sum())
    if k < 20:
        return None
    chi = float((((even - exp)[mask]) ** 2 / exp[mask]).sum())
    return chi_sf(chi, k - 1), chi, k


def load_array(path):
    with Image.open(path) as im:
        im.load()
        if im.mode == "L":
            arr = np.asarray(im)[:, :, None]
        elif "A" in im.mode or "transparency" in im.info:
            arr = np.asarray(im.convert("RGBA"))
        else:
            arr = np.asarray(im.convert("RGB"))
    return np.ascontiguousarray(arr)


def check_chi_square(arr, findings):
    h, w, c = arr.shape
    names = "L" if c == 1 else "RGBA"[:c]
    fractions = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    per_channel, worst = [], None
    for ci in range(c):
        flat = arr[:, :, ci].ravel()
        best = None
        for frac in fractions:
            n = int(len(flat) * frac)
            if n < 5000:
                continue
            r = chi_square_pov(flat[:n])
            if r and (best is None or r[0] > best[0]):
                best = (r[0], r[1], r[2], frac)
        if best:
            per_channel.append(f"{names[ci]}: max p={best[0]:.4f} @ first {best[3]:.0%}")
            if worst is None or best[0] > worst[0]:
                worst = (*best, names[ci])
    if worst is None:
        findings.append(Finding("info", "chi-square", "not enough data / bins for a reliable test"))
        return
    p, chi, k, frac, ch = worst
    if p >= 0.99:
        sev = "high"
    elif p >= 0.90:
        sev = "medium"
    else:
        sev = "info"
    findings.append(Finding(
        sev, "chi-square-lsb",
        f"channel {ch}: p={p:.4f} over first {frac:.0%} of pixels (chi2={chi:.1f}, {k} bins). "
        f"p near 1 => LSB pairs equalised (typical of LSB replacement). [{'; '.join(per_channel)}]"))


def check_lsb_extraction(arr, findings, nbytes=512):
    h, w, c = arr.shape
    names = "L" if c == 1 else "RGBA"[:c]
    streams = {names[i]: arr[:, :, i].ravel() for i in range(c)}
    if c >= 3:
        streams["RGB"] = arr[:, :, :3].ravel()
        streams["BGR"] = arr[:, :, 2::-1].ravel()
    if c == 4:
        streams["RGBA"] = arr.ravel()
    hits = []
    for sname, stream in streams.items():
        s = stream[:nbytes * 8]
        if len(s) < 96 * 8:
            continue
        bits = (s & 1).astype(np.uint8)
        for bitorder in ("big", "little"):
            buf = np.packbits(bits, bitorder=bitorder).tobytes()
            for off in (0, 4, 8):  # allow a small length header
                win = buf[off:off + 48]
                m = magic_at_start(win)
                if m and m not in ("MZ", "ID3", "BZh") or (m and len(set(win[:16])) > 6):
                    hits.append(f"{sname}/{bitorder}-first/offset{off}: {m} signature")
                    break
                if len(win) == 48 and printable_ratio(win) >= 0.95 and len(set(win)) >= 6:
                    preview = win[:32].decode("latin1").replace("\n", " ")
                    hits.append(f"{sname}/{bitorder}-first/offset{off}: printable text '{preview}'")
                    break
    if hits:
        findings.append(Finding("high", "lsb-extraction",
                                "LSB stream decodes to structured data -> " + " | ".join(hits[:4])))
    else:
        findings.append(Finding("info", "lsb-extraction",
                                "no file signature / readable text in LSB streams (simple sequential layout)"))


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def analyze(path, force_pixel=False):
    res = {"file": path, "format": None, "size": None, "dimensions": None,
           "score": 0, "verdict": None, "findings": [], "error": None}
    findings = []
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        res["size"] = len(data)
        fmt = detect_format(data)
        res["format"] = fmt

        ext = os.path.splitext(path)[1].lower()
        if fmt in EXT_FOR_FMT and ext and ext not in EXT_FOR_FMT[fmt]:
            findings.append(Finding("low", "extension-mismatch",
                                    f"extension '{ext}' but content is {fmt}"))

        {"PNG": check_png, "JPEG": check_jpeg, "GIF": check_gif,
         "BMP": check_bmp, "WEBP": check_webp}.get(fmt, lambda d, f: None)(data, findings)
        check_strings(data, findings)

        try:
            if fmt == "JPEG" and not force_pixel:
                findings.append(Finding("info", "pixel-tests-skipped",
                                        "JPEG: spatial LSB tests are not meaningful on decoded pixels "
                                        "(use --force-pixel-tests to run anyway)"))
                with Image.open(path) as im:
                    res["dimensions"] = f"{im.width}x{im.height}"
            else:
                arr = load_array(path)
                res["dimensions"] = f"{arr.shape[1]}x{arr.shape[0]}"
                if arr.shape[0] * arr.shape[1] < 1000:
                    findings.append(Finding("info", "pixel-tests-skipped", "image too small"))
                else:
                    check_chi_square(arr, findings)
                    check_lsb_extraction(arr, findings)
        except Exception as e:  # unreadable pixel data is itself worth noting
            findings.append(Finding("low", "decode-error", f"Pillow could not decode pixels: {e}"))
    except Exception as e:
        res["error"] = str(e)

    score = sum(WEIGHTS[f.severity] for f in findings)
    res["score"] = score
    res["verdict"] = ("LIKELY STEGO" if score >= 5 else
                      "SUSPICIOUS" if score >= 2 else "NO INDICATORS")
    if res["error"]:
        res["verdict"] = "ERROR"
    res["findings"] = [asdict(f) for f in findings]
    return res


def collect(paths, recursive):
    out = []
    for p in paths:
        if os.path.isdir(p):
            if recursive:
                for root, _, files in os.walk(p):
                    out += [os.path.join(root, f) for f in files
                            if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
            else:
                out += [os.path.join(p, f) for f in os.listdir(p)
                        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
                        and os.path.isfile(os.path.join(p, f))]
        elif os.path.isfile(p):
            out.append(p)
        else:
            print(f"[!] not found: {p}", file=sys.stderr)
    return sorted(set(out))


def print_result(r, verbose):
    print(f"\n=== {r['file']}")
    print(f"    {r['format']}  {r['dimensions'] or '?'}  {r['size']} bytes   "
          f"-> {r['verdict']} (score {r['score']})")
    if r["error"]:
        print(f"    error: {r['error']}")
    for f in r["findings"]:
        if f["severity"] == "info" and not verbose:
            continue
        print(f"    [{f['severity'].upper():6}] {f['check']}: {f['detail']}")


def main():
    ap = argparse.ArgumentParser(description="Heuristic steganography detector for images.")
    ap.add_argument("paths", nargs="+", help="image file(s) and/or folder(s)")
    ap.add_argument("-r", "--recursive", action="store_true", help="recurse into subfolders")
    ap.add_argument("-v", "--verbose", action="store_true", help="show informational findings too")
    ap.add_argument("--json", metavar="FILE", help="write full results as JSON")
    ap.add_argument("--csv", metavar="FILE", help="write summary as CSV")
    ap.add_argument("--force-pixel-tests", action="store_true",
                    help="run spatial LSB tests on JPEGs too (usually meaningless)")
    args = ap.parse_args()

    files = collect(args.paths, args.recursive)
    if not files:
        print("No images found.", file=sys.stderr)
        sys.exit(1)

    results = []
    for i, f in enumerate(files, 1):
        r = analyze(f, args.force_pixel_tests)
        results.append(r)
        print_result(r, args.verbose)

    if len(results) > 1:
        print("\n" + "=" * 70 + "\nSUMMARY (highest score first)")
        for r in sorted(results, key=lambda x: -x["score"]):
            print(f"  {r['verdict']:14} {r['score']:3}  {r['file']}")
        counts = {}
        for r in results:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        print("  " + ", ".join(f"{k}: {v}" for k, v in counts.items()))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nJSON written to {args.json}")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "format", "dimensions", "size", "verdict", "score", "findings"])
            for r in results:
                w.writerow([r["file"], r["format"], r["dimensions"], r["size"], r["verdict"],
                            r["score"], " | ".join(f"{x['check']}({x['severity']})"
                                                   for x in r["findings"] if x["severity"] != "info")])
        print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()