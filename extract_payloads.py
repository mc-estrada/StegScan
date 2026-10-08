#!/usr/bin/env python3
"""
extract_payloads.py - pull out data hidden by make_steg_tests.py (naive methods).

    python extract_payloads.py steg_tests [-o extracted]
    python extract_payloads.py steg_tests/lsb_text.png

Needs stegscan.py in the same folder (reuses its container parsers).
Methods tried per file:
  * data appended after end-of-image   (PNG/JPEG/GIF/BMP/WebP)
  * non-standard PNG chunks
  * JPEG comment segments (base64-decoded when possible)
  * sequential RGB LSB stream, MSB-first (lossless formats only)
"""
import argparse
import base64
import binascii
import io
import os
import struct
import zipfile

import numpy as np
from PIL import Image

from stegscan import (STD_PNG, detect_format, gif_end, jpeg_parse, png_parse,
                      printable_ratio)


def trailing(data, fmt):
    end = None
    if fmt == "PNG":
        end = png_parse(data)[1]
    elif fmt == "JPEG":
        end = jpeg_parse(data)[1]
    elif fmt == "GIF":
        end = gif_end(data)
    elif fmt == "BMP":
        end = struct.unpack("<I", data[2:6])[0]
    elif fmt == "WEBP":
        end = struct.unpack("<I", data[4:8])[0] + 8
    if end and end < len(data) and data[end:].strip(b"\x00\r\n \t"):
        return data[end:]
    return None


def lsb_stream(path, max_bytes=65536):
    """Sequential RGB LSBs, MSB-first; stop at the first non-text byte."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB")).reshape(-1)
    bits = (arr[:max_bytes * 8] & 1).astype(np.uint8)
    buf = np.packbits(bits).tobytes()
    n = 0
    while n < len(buf) and (32 <= buf[n] <= 126 or buf[n] in (9, 10, 13)):
        n += 1
    return buf[:n] if n >= 16 else None


def extract(path):
    with open(path, "rb") as fh:
        data = fh.read()
    fmt = detect_format(data)
    found = []

    t = trailing(data, fmt)
    if t:
        found.append(("trailing", t))
        i = t.find(b"PK\x03\x04")
        if i >= 0:
            try:
                with zipfile.ZipFile(io.BytesIO(t[i:])) as z:
                    for name in z.namelist():
                        found.append((f"zip_{name}", z.read(name)))
            except zipfile.BadZipFile:
                pass

    if fmt == "PNG":
        for name, length, off in png_parse(data)[0]:
            if name not in STD_PNG:
                found.append((f"chunk_{name}", data[off:off + length]))

    if fmt == "JPEG":
        for marker, start, length in jpeg_parse(data)[0]:
            if marker == 0xFE:
                payload = data[start:start + length]
                try:
                    found.append(("comment_b64decoded", base64.b64decode(payload, validate=True)))
                except (binascii.Error, ValueError):
                    found.append(("comment", payload))

    if fmt in ("PNG", "BMP", "TIFF", "WEBP", "GIF"):
        try:
            s = lsb_stream(path)
            if s:
                found.append(("lsb_rgb", s))
        except Exception:
            pass
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-o", "--out", default="extracted")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = []
    for p in args.paths:
        if os.path.isdir(p):
            for root, _, fs in os.walk(p):
                files += [os.path.join(root, f) for f in fs]
        else:
            files.append(p)

    for f in sorted(files):
        if "controls" in f.split(os.sep):
            continue
        try:
            results = extract(f)
        except Exception as e:
            print(f"{f}: error {e}")
            continue
        if not results:
            print(f"{f}: nothing extracted")
            continue
        stem = os.path.basename(f).replace(".", "_")
        for label, blob in results:
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)
            out = os.path.join(args.out, f"{stem}__{safe}.bin")
            with open(out, "wb") as fh:
                fh.write(blob)
            prev = blob[:60].decode("latin1") if printable_ratio(blob[:60]) > 0.9 else "(binary)"
            print(f"{f}: [{label}] {len(blob)} bytes -> {out}\n    {prev!r}")


if __name__ == "__main__":
    main()
