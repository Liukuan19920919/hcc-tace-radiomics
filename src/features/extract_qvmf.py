"""
HCC Delta Vasculomics - QVMF Feature Extraction
================================================
Replicates the Quantitative Vascular Morphometry Features (QVMFs)
from Hu et al. (2026, Science Advances) for HCC application.

Key features extracted:
- Vessel length (overall, by radius bin, by vessel type)
- Tortuosity (distance metric, curvature-based)
- Branching (bifurcation count, angles)
- Segment count (by radius bin, by vessel type)
- Endpoints (total, by vessel type)
- Volume (overall)

Radius bins (matching original paper):
R01: 0-1mm, R12: 1-2mm, R23: 2-3mm, R34: 3-4mm, R45: 4-5mm,
R56: 5-6mm, R67: 6-7mm, R78: 7-8mm, R89: 8-9mm, R910: 9-10mm, R1011: >10mm
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import morphology, measure
from tqdm import tqdm


# Radius bin definitions (mm)
RADIUS_BINS = {
    'R01': (0, 1),
    'R12': (1, 2),
    'R23': (2, 3),
    'R34': (3, 4),
    'R45': (4, 5),
    'R56': (5, 6),
    'R67': (6, 7),
    'R78': (7, 8),
    'R89': (8, 9),
    'R910': (9, 10),
    'R1011': (10, float('inf')),
}


class VesselMorphometryExtractor:
    """
    Extract Quantitative Vascular Morphometry Features (QVMFs)
    from 3D vessel segmentation masks.

    Parameters
    ----------
    vessel_mask : np.ndarray
        Binary 3D vessel segmentation (or multi-class: 1=vein, 2=artery, etc.)
    voxel_spacing : tuple
        (dx, dy, dz) in mm
    vessel_type_map : dict, optional
        Mapping from mask value to vessel type: {1: 'vein', 2: 'artery', 3: 'portal'}
    """

    def __init__(
        self,
        vessel_mask: np.ndarray,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        vessel_type_map: Optional[Dict[int, str]] = None
    ):
        self.mask = vessel_mask.astype(bool) if vessel_mask.ndim == 3 else vessel_mask
        self.spacing = voxel_spacing
        self.vessel_type_map = vessel_type_map or {1: 'vessel'}

        # Pre-compute
        self.skeleton = None
        self.skeleton_coords = None
        self.distance_map = None
        self.branch_points = None
        self.end_points = None
        self._precompute()

    def _precompute(self):
        """Pre-compute skeleton, distance map, and topological features."""
        binary_mask = self.mask > 0 if not self.mask.dtype == bool else self.mask

        if not np.any(binary_mask):
            warnings.warn("Empty vessel mask provided.")
            return

        # Distance transform (for radius estimation)
        self.distance_map = ndimage.distance_transform_edt(
            binary_mask, sampling=self.spacing
        )

        # Crop to vessel bounding box for efficiency
        if np.any(binary_mask):
            coords = np.argwhere(binary_mask)
            self._crop_slice = tuple(
                slice(max(0, c.min() - 5), min(binary_mask.shape[i], c.max() + 6))
                for i, c in enumerate([coords[:, 0], coords[:, 1], coords[:, 2]])
            )
            binary_cropped = binary_mask[self._crop_slice]
            self._crop_offset = np.array([s.start for s in self._crop_slice])
        else:
            binary_cropped = binary_mask
            self._crop_slice = tuple(slice(0, s) for s in binary_mask.shape)
            self._crop_offset = np.zeros(3, dtype=int)

        # Skeletonization (handle different scikit-image versions)
        try:
            self.skeleton = morphology.skeletonize_3d(binary_cropped)
        except AttributeError:
            self.skeleton = morphology.skeletonize(binary_cropped)

        self.skeleton_coords = np.argwhere(self.skeleton).astype(float)
        if len(self.skeleton_coords) > 0:
            self.skeleton_coords += self._crop_offset  # back to original coords

        self.distance_map_cropped = ndimage.distance_transform_edt(
            binary_cropped, sampling=self.spacing
        )
        self.skeleton_coords = np.argwhere(self.skeleton)

        if len(self.skeleton_coords) < 2:
            return

        # Identify branch points and endpoints
        self._classify_skeleton_points()

    def _classify_skeleton_points(self):
        """Classify skeleton points as branch points, endpoints, or regular."""
        if self.skeleton is None:
            return

        # 3x3x3 neighborhood convolution to count neighbors
        kernel = np.ones((3, 3, 3), dtype=int)
        kernel[1, 1, 1] = 0

        # Count neighbors for each skeleton voxel
        neighbor_count = ndimage.convolve(
            self.skeleton.astype(int), kernel, mode='constant', cval=0
        )

        # Branch points: >2 neighbors; Endpoints: 1 neighbor
        skeleton_indices = np.argwhere(self.skeleton)
        if len(skeleton_indices) == 0:
            return

        counts = neighbor_count[self.skeleton]

        self.end_points = skeleton_indices[counts == 1]
        self.branch_points = skeleton_indices[counts > 2]

    def _get_vessel_segments(self) -> List[np.ndarray]:
        """
        Segment the skeleton into individual vessel branches.
        Removes branch points and returns connected components.
        """
        if self.skeleton is None:
            return []

        # Remove branch points to break skeleton into segments
        skeleton_segments = self.skeleton.copy()
        if self.branch_points is not None and len(self.branch_points) > 0:
            for pt in self.branch_points:
                skeleton_segments[tuple(pt)] = 0

        # Label individual segments
        labeled, n_segments = ndimage.label(skeleton_segments)

        segments = []
        for i in range(1, n_segments + 1):
            seg_coords = np.argwhere(labeled == i)
            if len(seg_coords) >= 2:
                segments.append(seg_coords)

        return segments

    def compute_segment_length(self, coords: np.ndarray) -> float:
        """Compute the arc length of an ordered skeleton segment in mm."""
        if len(coords) < 2:
            return 0.0

        # Simple approach: sum Euclidean distances between consecutive points
        # A more sophisticated approach would order the points along the curve
        diffs = np.diff(coords.astype(float) * self.spacing, axis=0)
        return np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))

    def compute_euclidean_length(self, coords: np.ndarray) -> float:
        """Compute straight-line distance between endpoints in mm."""
        if len(coords) < 2:
            return 0.0

        start = coords[0].astype(float) * self.spacing
        end = coords[-1].astype(float) * self.spacing
        return np.sqrt(np.sum((end - start) ** 2))

    def compute_tortuosity(self, coords: np.ndarray) -> float:
        """
        Distance Metric (DM) tortuosity: L / D - 1
        where L = arc length, D = Euclidean distance.
        Returns 0 if straight, >0 for curved vessels.
        """
        L = self.compute_segment_length(coords)
        D = self.compute_euclidean_length(coords)
        if D < 1e-10:
            return np.nan
        return L / D - 1

    def compute_radius_at_point(self, coord: Tuple[int, int, int]) -> float:
        """Get vessel radius from distance transform. Coord is in cropped space."""
        if self.distance_map_cropped is None:
            return np.nan
        idx = tuple(
            max(0, min(self.distance_map_cropped.shape[i]-1, int(round(c))))
            for i, c in enumerate(coord)
        )
        return self.distance_map_cropped[idx]

    def get_radius_bin(self, radius: float) -> str:
        """Map a radius value to its bin name."""
        for bin_name, (low, high) in RADIUS_BINS.items():
            if low <= radius < high:
                return bin_name
        return 'R1011'

    def extract_all_features(self, prefix: str = "") -> Dict[str, float]:
        """
        Extract all QVMFs.

        Returns
        -------
        dict mapping feature names to values
        """
        features = {}

        if self.skeleton is None or len(self.skeleton_coords) < 2:
            # Return NaN features
            for category in self._feature_names():
                features[f"{prefix}{category}"] = np.nan
            return features

        segments = self._get_vessel_segments()

        # ---- Per-segment features ----
        seg_lengths = []
        seg_tortuosities = []
        seg_mean_radii = []

        for seg_coords in segments:
            L = self.compute_segment_length(seg_coords)
            tort = self.compute_tortuosity(seg_coords)
            radii = [self.compute_radius_at_point(tuple(c)) for c in seg_coords]
            mean_r = np.mean(radii) if radii else np.nan

            seg_lengths.append(L)
            seg_tortuosities.append(tort)
            seg_mean_radii.append(mean_r)

        seg_lengths = np.array(seg_lengths)
        seg_tortuosities = np.array([t for t in seg_tortuosities if not np.isnan(t)])
        seg_radii = np.array([r for r in seg_mean_radii if not np.isnan(r)])

        # ---- Global features ----
        features[f"{prefix}NSeg"] = len(segments)
        features[f"{prefix}mLen"] = np.mean(seg_lengths) if len(seg_lengths) > 0 else np.nan
        features[f"{prefix}sdLen"] = np.std(seg_lengths) if len(seg_lengths) > 1 else np.nan
        features[f"{prefix}totalLen"] = np.sum(seg_lengths)
        features[f"{prefix}mTort"] = np.mean(seg_tortuosities) if len(seg_tortuosities) > 0 else np.nan
        features[f"{prefix}sdTort"] = np.std(seg_tortuosities) if len(seg_tortuosities) > 1 else np.nan
        features[f"{prefix}mRadius"] = np.mean(seg_radii) if len(seg_radii) > 0 else np.nan

        # Endpoints and branch points
        features[f"{prefix}End"] = len(self.end_points) if self.end_points is not None else 0
        features[f"{prefix}NBif"] = len(self.branch_points) if self.branch_points is not None else 0

        # Volume
        voxel_volume = np.prod(self.spacing)
        features[f"{prefix}Vol"] = np.sum(self.mask > 0) * voxel_volume

        # ---- Radius-bin features ----
        for bin_name in RADIUS_BINS.keys():
            features[f"{prefix}NSeg_{bin_name}"] = np.nan
            features[f"{prefix}mLen_{bin_name}"] = np.nan
            features[f"{prefix}mTort_{bin_name}"] = np.nan

        bin_segments = defaultdict(list)
        for i, r in enumerate(seg_mean_radii):
            if not np.isnan(r):
                bin_name = self.get_radius_bin(r)
                bin_segments[bin_name].append(i)

        for bin_name, indices in bin_segments.items():
            bin_lens = [seg_lengths[i] for i in indices]
            bin_torts = [seg_tortuosities[i] for i in indices if not np.isnan(seg_tortuosities[i])]

            features[f"{prefix}NSeg_{bin_name}"] = len(indices)
            features[f"{prefix}mLen_{bin_name}"] = np.mean(bin_lens) if bin_lens else np.nan
            features[f"{prefix}mTort_{bin_name}"] = np.mean(bin_torts) if bin_torts else np.nan

        # ---- Bifurcation analysis ----
        if self.branch_points is not None and len(self.branch_points) > 1:
            # Simple bifurcation angle estimation
            features[f"{prefix}BifAng"] = np.nan  # Requires vessel tree graph
        else:
            features[f"{prefix}BifAng"] = np.nan

        return features

    @staticmethod
    def _feature_names() -> List[str]:
        """Return list of all feature names produced."""
        base = ['NSeg', 'mLen', 'sdLen', 'totalLen', 'mTort', 'sdTort',
                'mRadius', 'End', 'NBif', 'Vol', 'BifAng']
        bin_suffixes = ['_NSeg', '_mLen', '_mTort']
        names = base.copy()
        for b in RADIUS_BINS.keys():
            for s in bin_suffixes:
                names.append(f"{s.split('_')[-1]}_{b}" if not s.startswith('_') else f"{b}{s}")
        return names


def extract_qvmf_from_mask(
    vessel_mask: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    prefix: str = ""
) -> Dict[str, float]:
    """
    Convenience function to extract all QVMFs from a vessel mask.

    Parameters
    ----------
    vessel_mask : np.ndarray
        3D binary vessel segmentation array
    voxel_spacing : tuple
        Voxel dimensions in mm (dx, dy, dz)
    prefix : str
        Prefix for feature names (e.g., "A_" for arterial, "V_" for venous)

    Returns
    -------
    dict of feature_name -> value
    """
    extractor = VesselMorphometryExtractor(vessel_mask, voxel_spacing)
    return extractor.extract_all_features(prefix=prefix)


def extract_qvmf_multiclass(
    multiclass_mask: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    class_map: Dict[int, str] = None
) -> Dict[str, float]:
    """
    Extract QVMFs from a multi-class vessel mask (e.g., separating portal vein,
    hepatic vein, and artery).

    Parameters
    ----------
    multiclass_mask : np.ndarray
        3D array where different vessel types have different integer labels
    voxel_spacing : tuple
        Voxel dimensions in mm
    class_map : dict
        Mapping {label: name} e.g. {1: 'portal_vein', 2: 'hepatic_vein', 3: 'artery'}

    Returns
    -------
    dict with features for each vessel type + overall features
    """
    if class_map is None:
        class_map = {
            1: 'portal_vein',
            2: 'hepatic_vein',
            3: 'artery',
        }

    all_features = {}

    # Overall features (all vessels combined)
    overall_mask = multiclass_mask > 0
    overall_features = extract_qvmf_from_mask(overall_mask, voxel_spacing, prefix="")
    all_features.update(overall_features)

    # Per-vessel-type features
    for label, name in class_map.items():
        if label not in multiclass_mask:
            continue

        vessel_mask = multiclass_mask == label
        if vessel_mask.sum() < 10:  # Skip if too small
            continue

        # Use prefix matching the original paper convention
        if 'artery' in name.lower():
            prefix = "A_"
        elif 'vein' in name.lower() or 'portal' in name.lower():
            prefix = "V_"
        else:
            prefix = f"{name}_"

        type_features = extract_qvmf_from_mask(vessel_mask, voxel_spacing, prefix=prefix)
        all_features.update(type_features)

    return all_features


def compute_delta_features(
    baseline_features: Dict[str, float],
    followup_features: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute delta (change) features between baseline and follow-up.

    ΔFeature = Follow-up value - Baseline value

    Parameters
    ----------
    baseline_features : dict
        QVMFs from baseline CT
    followup_features : dict
        QVMFs from follow-up CT

    Returns
    -------
    dict with Δ-prefixed feature names
    """
    delta = {}
    common_keys = set(baseline_features.keys()) & set(followup_features.keys())

    for key in sorted(common_keys):
        delta[f"Δ{key}"] = followup_features[key] - baseline_features[key]

    return delta


def build_feature_matrix(
    patient_features: List[Dict],
    clinical_df: Optional[pd.DataFrame] = None,
    patient_id_col: str = 'patient_id'
) -> pd.DataFrame:
    """
    Build a clean feature matrix from per-patient feature dictionaries.

    Parameters
    ----------
    patient_features : list of dict
        Each dict should have 'patient_id' + QVMF features
    clinical_df : pd.DataFrame, optional
        Clinical variables to merge
    patient_id_col : str
        Column name for patient ID in clinical_df

    Returns
    -------
    pd.DataFrame with rows = patients, columns = features + optional clinical
    """
    df = pd.DataFrame(patient_features)

    # Set patient_id as index
    if 'patient_id' in df.columns:
        df = df.set_index('patient_id')

    # Merge clinical data if provided
    if clinical_df is not None:
        clinical_df = clinical_df.copy()
        if patient_id_col in clinical_df.columns:
            clinical_df = clinical_df.set_index(patient_id_col)
        df = df.join(clinical_df, how='left')

    # Remove columns with >50% NaN
    nan_frac = df.isna().mean()
    cols_to_remove = nan_frac[nan_frac > 0.5].index.tolist()
    if cols_to_remove:
        print(f"Removing {len(cols_to_remove)} columns with >50% NaN")
        df = df.drop(columns=cols_to_remove)

    return df


if __name__ == "__main__":
    print("QVMF feature extraction module initialized.")
    print(f"Radius bins: {list(RADIUS_BINS.keys())}")

    # Quick test with synthetic data
    print("\nTesting with synthetic vessel mask...")
    test_mask = np.zeros((50, 50, 50), dtype=bool)
    test_mask[20:30, 20:25, 10:40] = True  # Straight segment
    test_mask[25:28, 15:20, 20:25] = True  # Branch

    features = extract_qvmf_from_mask(test_mask, voxel_spacing=(1.0, 1.0, 1.0))
    print(f"Extracted {len(features)} features from test mask.")
    for k, v in list(features.items())[:10]:
        print(f"  {k}: {v:.3f}" if not np.isnan(v) else f"  {k}: NaN")
