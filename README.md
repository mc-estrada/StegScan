# StegScan
Basic steganalysis tool. Heuristic steganography detector for images.

Example Execution Syntax: 
python stegscan.py photo.png
python stegscan.py ./images -r --json report.json --csv report.csv
python stegscan.py ./images -v

Example Syntax for extract_payloads.py:
python extract_payloads.py steg_tests -o extracted

usage format: stegscan.py [-h] [-r] [-v] [--json FILE] [--csv FILE] [--force-pixel-tests] paths [paths ...]

positional arguments:
  paths                image file(s) and/or folder(s)

options:
  -h, --help           show this help message and exit
  -r, --recursive      recurse into subfolders
  -v, --verbose        show informational findings too
  --json FILE          write full results as JSON
  --csv FILE           write summary as CSV
  --force-pixel-tests  run spatial LSB tests on JPEGs too (usually meaningless
