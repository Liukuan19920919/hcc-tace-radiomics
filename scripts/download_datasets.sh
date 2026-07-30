#!/bin/bash
# =============================================================================
# HCC Delta Vasculomics - Dataset Download Script
# =============================================================================
# Downloads all required public datasets for the replication study
#
# Datasets:
#   1. HCC-TACE-Seg (TCIA) - Primary cohort with paired CT + mRECIST
#   2. HVM (Zenodo) - Liver vessel annotations for segmentation training
#   3. WAW-TACE (Zenodo) - External validation with pre-extracted radiomics
#   4. TCGA-LIHC (TCIA) - Imaging-genomics correlation
#
# Prerequisites:
#   - NBIA Data Retriever (for TCIA downloads)
#   - python-wget or curl
#   - ~150 GB free disk space
# =============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$PROJECT_DIR/data"

echo "============================================"
echo "HCC Delta Vasculomics - Data Download"
echo "Project: $PROJECT_DIR"
echo "Data: $DATA_DIR"
echo "============================================"

# ---- 1. HCC-TACE-Seg (TCIA) ----
echo ""
echo "[1/4] HCC-TACE-Seg (~26.6 GB)"
echo "URL: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230233"
echo ""
echo "MANUAL STEP REQUIRED:"
echo "  1. Install NBIA Data Retriever from: https://wiki.cancerimagingarchive.net/display/NBIA/National+Biomedical+Imaging+Archive+NBIA+Downloader"
echo "  2. Download the manifest file (.tcia) from the collection page"
echo "  3. Run: NBIA Data Retriever --manifest <manifest.tcia> --output $DATA_DIR/HCC-TACE-Seg"
echo ""

# ---- 2. HVM Dataset (Zenodo) ----
echo "[2/4] HVM Hepatic Vessel Morphology Dataset (~15 GB estimated)"
echo "DOI: 10.1038/s41597-026-07550-3"
echo ""
HVM_DIR="$DATA_DIR/HVM"
mkdir -p "$HVM_DIR"

# HVM - try Zenodo record
HVM_ZENODO="https://zenodo.org/records/15034031"
echo "Checking Zenodo for HVM dataset..."
echo "Visit: $HVM_ZENODO"
echo ""
echo "MANUAL STEP: Download all files from Zenodo to $HVM_DIR"
echo "  wget -r -np -nH --cut-dirs=3 -P $HVM_DIR $HVM_ZENODO/files/"
echo ""

# ---- 3. WAW-TACE (Zenodo) ----
echo "[3/4] WAW-TACE (~48 GB)"
echo "DOI: 10.5281/zenodo.11063784"
echo ""
WAW_DIR="$DATA_DIR/WAW-TACE"
mkdir -p "$WAW_DIR"

WAW_ZENODO="https://zenodo.org/records/12741586"
echo "Visit: $WAW_ZENODO"
echo ""
echo "MANUAL STEP: Download all files from Zenodo to $WAW_DIR"
echo "  wget -r -np -nH --cut-dirs=3 -P $WAW_DIR $WAW_ZENODO/files/"
echo ""

# ---- 4. TCGA-LIHC (TCIA) ----
echo "[4/4] TCGA-LIHC (~40 GB estimated)"
echo "URL: https://www.cancerimagingarchive.net/collection/tcga-lihc/"
echo ""
echo "MANUAL STEP:"
echo "  1. Use NBIA Data Retriever with the TCGA-LIHC manifest"
echo "  2. Download clinical data from GDC: https://portal.gdc.cancer.gov/projects/TCGA-LIHC"
echo "  3. Store images in: $DATA_DIR/TCGA-LIHC/images"
echo "  4. Store clinical in: $DATA_DIR/TCGA-LIHC/clinical"
echo ""

echo "============================================"
echo "Download instructions complete."
echo "Total estimated size: ~130 GB"
echo "Please complete manual download steps above."
echo "============================================"
