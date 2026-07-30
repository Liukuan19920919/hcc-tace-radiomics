#!/bin/bash
# =============================================================================
# HCC Delta Vasculomics - Auto Download Script
# =============================================================================
# Downloads all 3 required public datasets with proxy support.
#
# Usage:
#   PROXY=http://127.0.0.1:7897 bash download_all.sh
#   bash download_all.sh  (if proxy already exported)
#
# Prerequisites:
#   - Working international network (VPN/proxy)
#   - ~130 GB free disk space
#   - python3 with requests installed
# =============================================================================

set -e

PROXY="${PROXY:-http://127.0.0.1:7897}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$PROJECT_DIR/data"
TMP_DIR="/tmp/hcc_vasc_dl"

echo "============================================"
echo " HCC Delta Vasculomics - Data Download"
echo " Proxy: $PROXY"
echo " Target: $DATA_DIR"
echo "============================================"
echo ""

# Network test
echo "[0/3] Testing network..."
if curl -s --connect-timeout 10 -x "$PROXY" -o /dev/null -w "%{http_code}" https://zenodo.org | grep -q "200\|301\|302"; then
    echo "  ✓ Zenodo accessible"
else
    echo "  ✗ Zenodo NOT accessible - please fix proxy and retry"
    echo "  Test: curl -x $PROXY https://zenodo.org"
    exit 1
fi

# ---- 1. HVM Dataset (Zenodo, ~15 GB) ----
echo ""
echo "[1/3] HVM Dataset (liver vessel annotations)"
echo "      DOI: 10.5281/zenodo.15034031"
echo "      ~15 GB"

HVM_DIR="$DATA_DIR/HVM"
mkdir -p "$HVM_DIR" "$TMP_DIR"

python3 - "$PROXY" "$HVM_DIR" "$TMP_DIR" << 'PYEOF'
import sys, os, json, subprocess, urllib.request

proxy = sys.argv[1]
hvm_dir = sys.argv[2]
tmp_dir = sys.argv[3]

# Query Zenodo API for file list
api_url = "https://zenodo.org/api/records/15034031"
req = urllib.request.Request(api_url)
req.set_proxy(proxy, 'http')

try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
except Exception as e:
    print(f"  ✗ Cannot query Zenodo API: {e}")
    sys.exit(1)

files = data.get('files', [])
total_size = sum(f.get('size', 0) for f in files)
print(f"  Found {len(files)} files, total {total_size//1024//1024}MB")

for i, f in enumerate(files, 1):
    fname = f['key']
    url = f['links']['self']
    size = f.get('size', 0)
    outpath = os.path.join(hvm_dir, fname)

    if os.path.exists(outpath) and os.path.getsize(outpath) == size:
        print(f"  [{i}/{len(files)}] {fname} - already downloaded ✓")
        continue

    print(f"  [{i}/{len(files)}] {fname} ({size//1024//1024}MB)")
    cmd = f"curl -L --connect-timeout 30 --retry 3 -x {proxy} -o '{outpath}' '{url}'"
    ret = os.system(cmd)
    if ret != 0:
        print(f"    ✗ Failed to download {fname}")
        continue

print("  ✓ HVM download complete")
PYEOF

# ---- 2. WAW-TACE (Zenodo, ~48 GB) ----
echo ""
echo "[2/3] WAW-TACE Dataset (radiomics features + clinical)"
echo "      DOI: 10.5281/zenodo.12741586"
echo "      ~48 GB"

WAW_DIR="$DATA_DIR/WAW-TACE"
mkdir -p "$WAW_DIR"

python3 - "$PROXY" "$WAW_DIR" "$TMP_DIR" << 'PYEOF'
import sys, os, json, urllib.request

proxy = sys.argv[1]
waw_dir = sys.argv[2]
tmp_dir = sys.argv[3]

api_url = "https://zenodo.org/api/records/12741586"
req = urllib.request.Request(api_url)
req.set_proxy(proxy, 'http')

try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
except Exception as e:
    print(f"  ✗ Cannot query Zenodo API: {e}")
    sys.exit(1)

files = data.get('files', [])
total_size = sum(f.get('size', 0) for f in files)
print(f"  Found {len(files)} files, total {total_size//1024//1024}MB")

for i, f in enumerate(files, 1):
    fname = f['key']
    url = f['links']['self']
    size = f.get('size', 0)
    outpath = os.path.join(waw_dir, fname)

    if os.path.exists(outpath) and os.path.getsize(outpath) == size:
        print(f"  [{i}/{len(files)}] {fname} - already downloaded ✓")
        continue

    print(f"  [{i}/{len(files)}] {fname} ({size//1024//1024}MB)")
    cmd = f"curl -L --connect-timeout 30 --retry 3 -x {proxy} -o '{outpath}' '{url}'"
    ret = os.system(cmd)
    if ret != 0:
        print(f"    ✗ Failed to download {fname}")
        continue

print("  ✓ WAW-TACE download complete")
PYEOF

# ---- 3. HCC-TACE-Seg (TCIA, ~27 GB) ----
echo ""
echo "[3/3] HCC-TACE-Seg (TCIA, ~27 GB)"
echo "      Collection: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230233"
echo ""
echo "  TCIA requires NBIA Data Retriever (Java GUI app)."
echo "  Download steps:"
echo "  1. Install NBIA Data Retriever:"
echo "     https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever+Download+Page"
echo "  2. Open manifest from TCIA collection page"
echo "  3. Set output to: $DATA_DIR/HCC-TACE-Seg"
echo "  4. Set proxy to: $PROXY in NBIA settings"
echo "  5. Click Download"
echo ""
echo "  Alternative: Use NBIA REST API"
echo "  See: https://wiki.cancerimagingarchive.net/display/Public/NBIA+Search+REST+API+Guide"
echo ""

# ---- Summary ----
echo "============================================"
echo " Download Summary"
echo "============================================"
echo "  HVM:      $HVM_DIR"
echo "  WAW-TACE: $WAW_DIR"
echo "  HCC-TACE: Manual via NBIA → $DATA_DIR/HCC-TACE-Seg"
echo ""
echo "  Total disk needed: ~90 GB"
echo "  Available: $(df -h "$DATA_DIR" | tail -1 | awk '{print $4}')"
echo "============================================"
