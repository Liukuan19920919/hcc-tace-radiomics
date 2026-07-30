#!/usr/bin/env python3
"""
HCC Delta Vasculomics - Main Pipeline Runner
=============================================
Full pipeline: Data → QVMF Extraction → Delta Features → ML Modeling → SHAP

Usage:
    python scripts/run_pipeline.py --data_dir data/HCC-TACE-Seg --output outputs
    python scripts/run_pipeline.py --synthetic  # Test with synthetic data
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.features.extract_qvmf import (
    extract_qvmf_from_mask,
    compute_delta_features,
    build_feature_matrix,
)
from src.modeling.train_models import run_full_pipeline
from src.visualization.plots import (
    plot_morphological_fingerprint,
    plot_clinical_table,
    plot_univariate_forest,
)


def run_with_synthetic_data(output_dir: str = "outputs"):
    """Run pipeline with synthetic data for testing."""
    print("=" * 60)
    print("RUNNING WITH SYNTHETIC DATA")
    print("=" * 60)

    np.random.seed(42)
    n_patients = 100

    # Generate synthetic QVMFs
    features = {}
    radius_bins = ['R01', 'R12', 'R23', 'R34', 'R45', 'R56', 'R67', 'R78', 'R89', 'R910', 'R1011']
    vessel_types = ['', 'A_', 'V_']

    for vtype in vessel_types:
        prefix = vtype
        # Global features
        for feat in ['NSeg', 'mLen', 'totalLen', 'mTort', 'mRadius', 'End', 'NBif', 'Vol']:
            features[f'{prefix}{feat}'] = np.random.lognormal(
                mean=1.0, sigma=0.5, size=n_patients
            )
        # Radius-bin features
        for rb in radius_bins:
            for feat in ['NSeg', 'mLen', 'mTort']:
                features[f'{prefix}{feat}_{rb}'] = np.random.lognormal(
                    mean=0.5, sigma=0.3, size=n_patients
                )

    # Create baseline DataFrame
    df_base = pd.DataFrame(features, index=[f'P{i:03d}' for i in range(n_patients)])

    # Create follow-up DataFrame (slightly different, using concat to avoid fragmentation)
    fu_cols = {}
    for col in df_base.columns:
        effect = np.random.normal(0.2, 0.5, n_patients)
        fu_cols[col] = df_base[col].values + effect
    df_fu = pd.DataFrame(fu_cols, index=df_base.index)

    # Compute delta features
    delta_features = []  # list of dicts
    for idx in df_base.index:
        base_dict = df_base.loc[idx].to_dict()
        fu_dict = df_fu.loc[idx].to_dict()
        delta = compute_delta_features(base_dict, fu_dict)
        delta['patient_id'] = idx
        delta_features.append(delta)

    df_delta = pd.DataFrame(delta_features).set_index('patient_id')

    # Merge baseline + delta
    df_merged = pd.concat([df_base.add_prefix('base_'), df_delta], axis=1)

    # Generate response labels (biased by some delta features)
    delta_cols = [c for c in df_delta.columns if 'mTort' in c or 'mLen' in c]
    score = df_delta[delta_cols[:5]].mean(axis=1)
    df_merged['response'] = ((score > score.median()).astype(int))

    # Add clinical variable
    df_merged['LOT'] = np.random.choice([1, 2, 3], n_patients)

    print(f"Synthetic data: {df_merged.shape[0]} patients, {df_merged.shape[1]} features")
    print(f"Response distribution: {df_merged['response'].value_counts().to_dict()}")

    # Run pipeline
    results = run_full_pipeline(
        df_merged,
        response_col='response',
        clinical_cols=['LOT'],
        output_dir=output_dir,
        random_state=42,
    )

    # Generate visualizations
    print("\nGenerating publication figures...")
    plot_morphological_fingerprint(
        df_merged,
        response_col='response',
        save_path=f"{output_dir}/figures/morph_fingerprint.png"
    )
    plot_univariate_forest(
        df_merged,
        response_col='response',
        save_path=f"{output_dir}/figures/univariate_forest.png"
    )

    print(f"\nPipeline complete. Results saved to: {output_dir}/")
    return results


def run_with_real_data(data_dir: str, output_dir: str = "outputs"):
    """Run pipeline with real data from HCC-TACE-Seg."""
    print("=" * 60)
    print("RUNNING WITH REAL DATA")
    print("=" * 60)

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Step 1: Process each patient
    print("\n[1/5] Processing patient data...")
    patient_dirs = sorted(
        [d for d in data_path.iterdir() if d.is_dir()]
    )

    if not patient_dirs:
        raise ValueError(f"No patient directories found in {data_dir}")

    print(f"Found {len(patient_dirs)} patient directories")

    all_baseline_features = []
    all_followup_features = []
    all_clinical = []

    # Import data loading module
    from src.data.load_data import (
        find_paired_scans, find_segmentations, load_nifti,
    )

    for pdir in patient_dirs:
        pid = pdir.name
        try:
            # Find paired scans and segmentations
            paired = find_paired_scans(str(pdir))
            segs = find_segmentations(str(pdir))

            if 'baseline_pvp' not in paired:
                print(f"  {pid}: skipping (no baseline PVP)")
                continue

            # Extract features from baseline
            for seg_type, seg_path in segs.items():
                if 'vessel' not in seg_type.lower():
                    continue
                try:
                    mask_data, affine = load_nifti(seg_path)
                    spacing = tuple(abs(x) for x in affine.diagonal()[:-1])
                    base_feats = extract_qvmf_from_mask(
                        mask_data > 0, voxel_spacing=spacing,
                        prefix=f"{seg_type}_"
                    )
                    base_feats['patient_id'] = pid
                    all_baseline_features.append(base_feats)
                except Exception as e:
                    print(f"  {pid}: error extracting {seg_type} - {e}")

        except Exception as e:
            print(f"  {pid}: skipped ({e})")
            continue

    if not all_baseline_features:
        print("\nNo features extracted. Check data structure.")
        print("Expected structure: data/HCC-TACE-Seg/<patient>/<study>/<phase>/DICOM files")
        print("With segmentation NIfTI files in patient directories.")
        return

    print(f"\nExtracted features from {len(all_baseline_features)} scans")

    # Step 2: Build feature matrix
    print("\n[2/5] Building feature matrix...")
    df_base = build_feature_matrix(all_baseline_features)

    # Step 3: Load clinical data
    print("\n[3/5] Loading clinical data...")
    clin_files = list(data_path.rglob("*.csv")) + list(data_path.rglob("*.xlsx"))
    if clin_files:
        clin_file = str(clin_files[0])
        if clin_file.endswith('.csv'):
            clin_df = pd.read_csv(clin_file)
        else:
            clin_df = pd.read_excel(clin_file)
        print(f"Clinical data: {clin_df.shape[0]} patients, {clin_df.shape[1]} variables")

        # Parse response labels
        for col in clin_df.columns:
            if 'mrecist' in col.lower() or 'recist' in col.lower():
                print(f"  Response column: {col}")
                # Map to binary
                clin_df['response'] = clin_df[col].astype(str).str.upper().map(
                    lambda x: 1 if x in ['CR', 'PR', 'COMPLETE', 'PARTIAL']
                    else (0 if x in ['SD', 'PD', 'STABLE', 'PROGRESSIVE'] else None)
                )
                break
    else:
        print("No clinical file found. Using synthetic labels for testing.")
        np.random.seed(42)
        clin_df = pd.DataFrame({
            'patient_id': df_base.index,
            'response': np.random.binomial(1, 0.5, len(df_base))
        })

    # Step 4: Merge and prepare
    print("\n[4/5] Preparing analysis dataset...")
    df_full = df_base.copy()
    df_full['response'] = clin_df.set_index('patient_id')['response']

    # Step 5: Run ML pipeline
    print("\n[5/5] Running ML pipeline...")
    results = run_full_pipeline(
        df_full.dropna(subset=['response']),
        response_col='response',
        output_dir=output_dir,
    )

    print(f"\nPipeline complete. Results saved to: {output_dir}/")
    return results


def main():
    parser = argparse.ArgumentParser(
        description='HCC Delta Vasculomics Pipeline'
    )
    parser.add_argument('--data_dir', type=str, default='data/HCC-TACE-Seg',
                       help='Path to HCC-TACE-Seg dataset')
    parser.add_argument('--output', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--synthetic', action='store_true',
                       help='Run with synthetic test data')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.synthetic:
        run_with_synthetic_data(args.output)
    else:
        run_with_real_data(args.data_dir, args.output)


if __name__ == "__main__":
    main()
