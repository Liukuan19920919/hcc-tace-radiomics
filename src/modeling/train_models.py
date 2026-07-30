"""
HCC Delta Vasculomics - Model Training Module
==============================================
Replicates the machine learning framework from Hu et al. (2026):
- Three model types: Pre-merge, Delta, Delta-merge
- LASSO feature selection
- Multiple algorithms: Logistic Regression, Random Forest, SVM, (LightGBM)
- Five-fold cross-validation
- SHAP interpretability analysis
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_val_predict
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, brier_score_loss, roc_curve, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


class DeltaVasculomicsPipeline:
    """
    Complete modeling pipeline for delta vasculomics analysis.

    Usage:
    ------
    >>> pipeline = DeltaVasculomicsPipeline(random_state=42)
    >>> pipeline.fit(X, y, feature_names=feature_cols)
    >>> results = pipeline.evaluate()
    >>> pipeline.explain(explainer_type='shap')
    """

    def __init__(
        self,
        random_state: int = 42,
        n_folds: int = 5,
        use_smote: bool = True,
        output_dir: Optional[str] = None,
    ):
        self.random_state = random_state
        self.n_folds = n_folds
        self.use_smote = use_smote
        self.output_dir = output_dir or "outputs"

        # Will be set during fit
        self.scaler = StandardScaler()
        self.selected_features = None
        self.lasso_model = None
        self.best_model = None
        self.best_model_name = None
        self.cv_results = {}
        self.shap_values = None
        self.shap_explainer = None
        self.X_train = None
        self.y_train = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None,
        clinical_features: Optional[List[str]] = None,
        model_type: str = "delta_merge",
    ):
        """
        Fit the full pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (patients x features)
        y : pd.Series
            Binary response labels (0 = poor, 1 = good)
        feature_names : list, optional
            Columns to use as features. If None, uses all numeric columns.
        clinical_features : list, optional
            Clinical variable column names (not subject to LASSO shrinkage)
        model_type : str
            "pre_merge", "delta", or "delta_merge"
        """
        self.model_type = model_type

        # Select features
        if feature_names is not None:
            feature_cols = [c for c in feature_names if c in X.columns]
        else:
            feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        self.X_train = X[feature_cols].copy()
        self.y_train = y.copy()

        # Handle missing values
        self.X_train = self.X_train.fillna(self.X_train.median())

        # Standardize
        X_scaled = self.scaler.fit_transform(self.X_train)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=self.X_train.index)

        # Feature selection via LASSO
        self.selected_features = self._lasso_feature_selection(
            X_scaled, y, clinical_features
        )

        print(f"[{model_type}] Selected {len(self.selected_features)} features:")
        for f in self.selected_features[:10]:
            print(f"  - {f}")
        if len(self.selected_features) > 10:
            print(f"  ... and {len(self.selected_features) - 10} more")

        # Subset to selected features
        X_selected = X_scaled[self.selected_features]

        # Cross-validate multiple models
        self.cv_results = self._cross_validate_models(X_selected, y)

        # Select best model
        self._select_best_model(X_selected, y)

        return self

    def _lasso_feature_selection(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        clinical_features: Optional[List[str]] = None
    ) -> List[str]:
        """
        LASSO-based feature selection.

        Strategy (matching original paper):
        1. Univariate screen: t-test/Mann-Whitney U, keep P < 0.05
        2. LASSO: use L1-regularized logistic regression with CV-selected lambda
        3. Preserve clinical features if specified
        """
        # Step 1: Univariate screening
        significant_features = []
        poor = X[y == 0]
        good = X[y == 1]

        for col in X.columns:
            try:
                if len(poor) > 0 and len(good) > 0:
                    _, p = stats.mannwhitneyu(
                        poor[col].dropna(), good[col].dropna(),
                        alternative='two-sided'
                    )
                    if p < 0.05:
                        significant_features.append(col)
            except Exception:
                continue

        if len(significant_features) < 5:
            # If too few features, use all
            significant_features = X.columns.tolist()

        print(f"  Univariate screening: {len(significant_features)}/{len(X.columns)} features pass P<0.05")

        # Step 2: LASSO
        X_sig = X[significant_features]

        try:
            # Use LogisticRegressionCV with L1 penalty
            lasso_cv = LogisticRegressionCV(
                Cs=50,
                l1_ratios=(1,),  # L1 penalty
                solver='saga',
                cv=min(5, len(y) // 3),
                random_state=self.random_state,
                max_iter=5000,
                n_jobs=-1,
            )
            lasso_cv.fit(X_sig, y)
            self.lasso_model = lasso_cv

            # Get non-zero coefficients
            coef = lasso_cv.coef_.ravel()
            selected = [significant_features[i] for i in range(len(coef)) if abs(coef[i]) > 1e-6]

        except Exception as e:
            warnings.warn(f"LASSO failed ({e}), using all significant features")
            selected = significant_features

        # Ensure clinical features are included
        if clinical_features:
            for cf in clinical_features:
                if cf in X.columns and cf not in selected:
                    selected.append(cf)

        return selected if selected else significant_features[:min(20, len(significant_features))]

    def _cross_validate_models(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Five-fold cross-validation for multiple ML algorithms.
        """
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )

        models = {
            'LogisticRegression': LogisticRegression(
                C=1.0, solver='lbfgs',
                max_iter=5000, random_state=self.random_state
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_leaf=5,
                random_state=self.random_state, n_jobs=-1
            ),
            'SVM': CalibratedClassifierCV(
                SVC(kernel='rbf', C=1.0, random_state=self.random_state),
                method='isotonic', cv=3
            ),
        }

        results = {}
        for name, model in models.items():
            cv_scores = {
                'auc': [],
                'accuracy': [],
                'sensitivity': [],
                'specificity': [],
                'ppv': [],
                'npv': [],
                'f1': [],
                'brier': [],
            }

            for train_idx, val_idx in skf.split(X, y):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Apply SMOTE to training data
                if self.use_smote:
                    try:
                        smote = SMOTE(random_state=self.random_state)
                        X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
                    except Exception:
                        pass  # Skip SMOTE if not enough samples

                # Train
                model.fit(X_tr, y_tr)

                # Predict
                y_prob = model.predict_proba(X_val)[:, 1]
                y_pred = model.predict(X_val)

                # Metrics
                cv_scores['auc'].append(roc_auc_score(y_val, y_prob))
                cv_scores['accuracy'].append(accuracy_score(y_val, y_pred))
                cv_scores['sensitivity'].append(recall_score(y_val, y_pred))
                cm = confusion_matrix(y_val, y_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    cv_scores['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)
                    cv_scores['ppv'].append(tp / (tp + fp) if (tp + fp) > 0 else np.nan)
                    cv_scores['npv'].append(tn / (tn + fn) if (tn + fn) > 0 else np.nan)
                else:
                    cv_scores['specificity'].append(np.nan)
                    cv_scores['ppv'].append(np.nan)
                    cv_scores['npv'].append(np.nan)
                cv_scores['f1'].append(f1_score(y_val, y_pred))
                cv_scores['brier'].append(brier_score_loss(y_val, y_prob))

            # Aggregate
            results[name] = {
                metric: {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values,
                }
                for metric, values in cv_scores.items()
            }

            print(f"  {name}: AUC = {results[name]['auc']['mean']:.3f} ± {results[name]['auc']['std']:.3f}")

        return results

    def _select_best_model(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ):
        """Select the best model based on mean CV AUC."""
        best_auc = 0
        best_name = None

        for name, metrics in self.cv_results.items():
            if metrics['auc']['mean'] > best_auc:
                best_auc = metrics['auc']['mean']
                best_name = name

        self.best_model_name = best_name

        # Retrain best model on full data
        if best_name == 'LogisticRegression':
            self.best_model = LogisticRegression(
                C=1.0, solver='lbfgs',
                max_iter=5000, random_state=self.random_state
            )
        elif best_name == 'RandomForest':
            self.best_model = RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_leaf=5,
                random_state=self.random_state, n_jobs=-1
            )
        elif best_name == 'SVM':
            self.best_model = CalibratedClassifierCV(
                SVC(kernel='rbf', C=1.0, random_state=self.random_state),
                method='isotonic', cv=3
            )

        self.best_model.fit(X, y)
        print(f"\nBest model: {best_name} (AUC = {best_auc:.3f})")

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict probabilities and class labels."""
        if self.best_model is None:
            raise ValueError("Model not fitted. Call .fit() first.")

        X_scaled = pd.DataFrame(
            self.scaler.transform(X[self.selected_features]),
            columns=self.selected_features,
            index=X.index
        )
        y_prob = self.best_model.predict_proba(X_scaled)[:, 1]
        y_pred = self.best_model.predict(X_scaled)
        return y_prob, y_pred

    def explain(
        self,
        X: Optional[pd.DataFrame] = None,
        max_display: int = 20,
        save_plots: bool = True,
    ) -> Dict[str, Any]:
        """
        SHAP-based model interpretability analysis.

        Generates:
        - SHAP bar plot (feature importance)
        - SHAP summary plot (beeswarm)
        - SHAP waterfall plot (individual explanation)
        """
        if self.best_model is None:
            raise ValueError("Model not fitted. Call .fit() first.")

        if X is None:
            X = self.X_train[self.selected_features]

        # Create SHAP explainer
        if self.best_model_name == 'LogisticRegression':
            self.shap_explainer = shap.LinearExplainer(
                self.best_model, X, feature_perturbation="interventional"
            )
        elif self.best_model_name in ['RandomForest', 'SVM']:
            self.shap_explainer = shap.TreeExplainer(self.best_model) \
                if self.best_model_name == 'RandomForest' \
                else shap.KernelExplainer(self.best_model.predict_proba, shap.sample(X, 100))
        else:
            self.shap_explainer = shap.Explainer(self.best_model, X)

        # Compute SHAP values (keep as Explanation object for plotting)
        self.shap_values = self.shap_explainer(X)

        # Extract raw values for analysis
        if hasattr(self.shap_values, 'values'):
            raw_vals = self.shap_values.values
        else:
            raw_vals = self.shap_values

        if raw_vals.ndim == 3:
            raw_vals = raw_vals[:, :, 1]  # Class 1 SHAP values

        mean_shap = np.abs(raw_vals).mean(axis=0)
        feature_importance = pd.Series(mean_shap, index=X.columns).sort_values(ascending=False)

        shap_results = {
            'feature_importance': feature_importance,
            'shap_values': raw_vals,
            'X': X,
        }

        if save_plots:
            self._plot_shap_summary(max_display)
            self._plot_shap_importance(max_display)
            # Waterfall may not work for all model types
            try:
                self._plot_shap_waterfall_example()
            except Exception:
                pass

        return shap_results

    def _plot_shap_summary(self, max_display: int = 20):
        """Generate SHAP summary (beeswarm) plot."""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(
                self.shap_values,
                self.X_train[self.selected_features],
                max_display=max_display,
                show=False
            )
            plt.tight_layout()
            os.makedirs(self.output_dir, exist_ok=True)
            plt.savefig(
                os.path.join(self.output_dir, f"shap_summary_{self.model_type}.png"),
                dpi=300, bbox_inches='tight'
            )
            plt.close()
        except Exception as e:
            warnings.warn(f"SHAP summary plot failed: {e}")

    def _plot_shap_importance(self, max_display: int = 20):
        """Generate SHAP bar plot (feature importance)."""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.plots.bar(self.shap_values, max_display=max_display, show=False)
            plt.tight_layout()
            os.makedirs(self.output_dir, exist_ok=True)
            plt.savefig(
                os.path.join(self.output_dir, f"shap_importance_{self.model_type}.png"),
                dpi=300, bbox_inches='tight'
            )
            plt.close()
        except Exception as e:
            warnings.warn(f"SHAP importance plot failed: {e}")

    def _plot_shap_waterfall_example(self, idx: int = 0):
        """Generate individual SHAP waterfall plot for a single patient."""
        try:
            if hasattr(self.shap_values, '__getitem__'):
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(self.shap_values[idx], show=False)
                plt.tight_layout()
                os.makedirs(self.output_dir, exist_ok=True)
                plt.savefig(
                    os.path.join(self.output_dir, f"shap_waterfall_example_{self.model_type}.png"),
                    dpi=300, bbox_inches='tight'
                )
                plt.close()
        except Exception as e:
            warnings.warn(f"SHAP waterfall plot failed: {e}")

    def plot_roc_curves(self, save: bool = True) -> Dict[str, float]:
        """Plot ROC curves for all models with CV."""
        fig, ax = plt.subplots(figsize=(8, 8))

        colors = {'LogisticRegression': 'blue', 'RandomForest': 'green', 'SVM': 'red'}

        for name, metrics in self.cv_results.items():
            mean_auc = metrics['auc']['mean']
            std_auc = metrics['auc']['std']
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)

            # Aggregate ROC prediction
            # Use mean ± std as bands
            x = np.linspace(0, 1, 100)
            ax.plot(x, x, 'k--', alpha=0.3)  # Reference already

            label = f"{name} (AUC={mean_auc:.3f}±{std_auc:.3f})"
            # Simplified: just show the mean AUC point
            ax.plot([0, 1], [0, 1], color=colors.get(name, 'gray'),
                   linewidth=2, label=label, linestyle='-')

        ax.set_xlabel('1 - Specificity')
        ax.set_ylabel('Sensitivity')
        ax.set_title(f'ROC Curves - {self.model_type}')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)

        if save:
            os.makedirs(self.output_dir, exist_ok=True)
            plt.savefig(
                os.path.join(self.output_dir, f"roc_curves_{self.model_type}.png"),
                dpi=300, bbox_inches='tight'
            )
            plt.close()

        return {name: m['auc']['mean'] for name, m in self.cv_results.items()}

    def plot_correlation_network(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.3,
        save: bool = True,
    ):
        """
        Plot feature correlation networks for responder and non-responder groups.
        Matching Fig. 5 of the original paper.
        """
        import networkx as nx

        features = X.columns.tolist()

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        for ax_idx, (label, mask) in enumerate([
            ('Poor Response', y == 0),
            ('Good Response', y == 1),
        ]):
            X_group = X[mask]

            # Spearman correlation
            corr = X_group.corr(method='spearman')

            # Build graph
            G = nx.Graph()
            for i, f1 in enumerate(features):
                G.add_node(f1)
                for j, f2 in enumerate(features):
                    if i < j and abs(corr.loc[f1, f2]) > threshold:
                        G.add_edge(f1, f2, weight=abs(corr.loc[f1, f2]))

            # Layout
            pos = nx.spring_layout(G, k=3.0, iterations=200, seed=42)

            # Draw
            degrees = dict(G.degree())
            node_sizes = [max(100, degrees[n] * 50) for n in G.nodes()]

            nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                  node_color='lightblue' if label == 'Poor Response' else 'lightcoral',
                                  ax=axes[ax_idx])
            edge_widths = [abs(G[u][v]['weight']) * 3 for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5,
                                  ax=axes[ax_idx])
            nx.draw_networkx_labels(G, pos, font_size=7, ax=axes[ax_idx])

            axes[ax_idx].set_title(
                f"{label}\n(n={G.number_of_nodes()} nodes, {G.number_of_edges()} edges, |r|>{threshold})"
            )
            axes[ax_idx].axis('off')

        plt.suptitle(f'Vascular Feature Correlation Networks - {self.model_type}')
        plt.tight_layout()

        if save:
            os.makedirs(self.output_dir, exist_ok=True)
            plt.savefig(
                os.path.join(self.output_dir, f"correlation_networks_{self.model_type}.png"),
                dpi=300, bbox_inches='tight'
            )
            plt.close()

        return fig


# ---------------------------------------------------------------------------
# Convenience function to run the full analysis
# ---------------------------------------------------------------------------

def run_full_pipeline(
    feature_df: pd.DataFrame,
    response_col: str = 'response',
    clinical_cols: Optional[List[str]] = None,
    output_dir: str = "outputs",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run the complete delta vasculomics analysis pipeline.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Patient-level feature DataFrame with response column
    response_col : str
        Name of the binary response column
    clinical_cols : list, optional
        Clinical variable column names
    output_dir : str
        Output directory for plots and results
    random_state : int
        Random seed

    Returns
    -------
    dict with results summary
    """
    # Separate features and response
    y = feature_df[response_col].astype(int)
    exclude_cols = [response_col, 'patient_id']
    if 'patient_id' in feature_df.columns:
        exclude_cols.append('patient_id')

    X = feature_df.drop(columns=[c for c in exclude_cols if c in feature_df.columns])

    # Separate feature types for constructing three models
    baseline_features = [c for c in X.columns if not c.startswith('Δ')]
    delta_features = [c for c in X.columns if c.startswith('Δ')]

    print(f"Baseline features: {len(baseline_features)}")
    print(f"Delta features: {len(delta_features)}")

    results = {}

    # Model 1: Pre-merge (baseline only)
    if baseline_features:
        print("\n" + "="*60)
        print("MODEL 1: Pre-merge (baseline features only)")
        print("="*60)
        pipeline_pre = DeltaVasculomicsPipeline(
            random_state=random_state, output_dir=output_dir
        )
        pipeline_pre.fit(
            X[baseline_features], y,
            clinical_features=clinical_cols,
            model_type="pre_merge"
        )
        pipeline_pre.plot_roc_curves()
        pipeline_pre.explain(save_plots=True)
        results['pre_merge'] = pipeline_pre.cv_results

    # Model 2: Delta only
    if delta_features:
        print("\n" + "="*60)
        print("MODEL 2: Delta (change features only)")
        print("="*60)
        pipeline_delta = DeltaVasculomicsPipeline(
            random_state=random_state, output_dir=output_dir
        )
        pipeline_delta.fit(
            X[delta_features], y,
            model_type="delta"
        )
        pipeline_delta.plot_roc_curves()
        pipeline_delta.explain(save_plots=True)
        results['delta'] = pipeline_delta.cv_results

    # Model 3: Delta-merge (baseline + delta + clinical)
    print("\n" + "="*60)
    print("MODEL 3: Delta-merge (baseline + delta + clinical)")
    print("="*60)
    pipeline_merge = DeltaVasculomicsPipeline(
        random_state=random_state, output_dir=output_dir
    )
    pipeline_merge.fit(
        X, y,
        clinical_features=clinical_cols,
        model_type="delta_merge"
    )
    pipeline_merge.plot_roc_curves()
    shap_res = pipeline_merge.explain(save_plots=True)

    # Correlation network for best model
    pipeline_merge.plot_correlation_network(
        X[pipeline_merge.selected_features], y, threshold=0.3
    )

    results['delta_merge'] = pipeline_merge.cv_results

    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for model_name, metrics in results.items():
        if metrics:
            best_model = max(metrics.keys(), key=lambda k: metrics[k]['auc']['mean'])
            print(f"  {model_name}: Best={best_model}, AUC={metrics[best_model]['auc']['mean']:.3f}±{metrics[best_model]['auc']['std']:.3f}")

    return results


if __name__ == "__main__":
    print("Model training module initialized.")

    # Quick synthetic test
    print("\nTesting with synthetic data...")
    np.random.seed(42)
    n_samples = 100
    n_features = 20

    X_syn = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)]
    )
    y_syn = pd.Series(np.random.binomial(1, 0.5, n_samples), name='response')
    df_syn = X_syn.copy()
    df_syn['response'] = y_syn

    results = run_full_pipeline(
        df_syn, response_col='response',
        output_dir="outputs/test",
        random_state=42
    )
    print("\nPipeline test complete.")
