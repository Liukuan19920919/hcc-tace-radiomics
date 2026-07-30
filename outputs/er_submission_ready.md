# Interpretable Machine Learning Based on Baseline Multiphase CT Radiomics for Predicting Transarterial Chemoembolization Response and Survival in Hepatocellular Carcinoma: A Fully Reproducible Analysis Using a Public Dataset

---

## Abstract

**Objectives:** To develop an interpretable machine learning model using baseline multiphase CT radiomics for predicting transarterial chemoembolization (TACE) response and survival in hepatocellular carcinoma (HCC).

**Methods:** Portal venous phase CT radiomics (n = 3,305 features) from the public WAW-TACE dataset (210 HCC patients) were analyzed with two-stage feature selection (Mann-Whitney U, LASSO) and five-fold cross-validation with SMOTE. Random forest, logistic regression, and support vector machine classifiers were compared. SHAP values provided interpretability. Survival was assessed via Kaplan-Meier and Cox regression.

**Results:** Random forest achieved an AUC of 0.831 ± 0.051 versus 0.746 ± 0.066 for clinical variables alone. SHAP analysis identified liver tumor texture, skeletal morphology (T8 vertebra, ribs), and clinical etiology as strongest predictors. Radiomics-based risk stratification significantly discriminated overall survival (P < 0.0001) and progression-free survival (P = 0.039). Portal venous phase outperformed arterial, non-contrast, and delayed phases.

**Conclusion:** An interpretable radiomics framework using exclusively public data predicts TACE response (AUC = 0.831) and stratifies survival in HCC. SHAP explanations identified clinically meaningful predictors including previously underrecognized skeletal features.

---

## Keywords

Carcinoma, Hepatocellular; Chemoembolization, Therapeutic; Tomography, X-Ray Computed; Radiomics; Machine Learning

---

## Key Points

1. Radiomics from portal venous CT predicts TACE response in HCC (AUC = 0.831).
2. SHAP analysis reveals skeletal CT features as key predictors of treatment outcome.
3. Radiomics-based risk stratification significantly discriminates overall survival (P < 0.0001).
4. Portal venous phase outperforms arterial, non-contrast, and delayed phase radiomics.
5. All analyses use a fully public dataset, ensuring complete reproducibility.

---

## Clinical Relevance Statement

Baseline portal venous phase CT radiomics, analyzed through an interpretable machine learning framework, provides non-invasive prediction of TACE treatment response and survival stratification in HCC. The identification of skeletal radiomics features as key predictors suggests that routine staging CT captures systemic frailty information relevant to treatment outcomes, supporting potential integration into personalized treatment planning without additional imaging.

---

## Abbreviations

AUC: Area under the receiver operating characteristic curve; CV: Cross-validation; HCC: Hepatocellular carcinoma; LASSO: Least absolute shrinkage and selection operator; LR-TR: LI-RADS Treatment Response; OS: Overall survival; PFS: Progression-free survival; RF: Random forest; SHAP: SHapley Additive exPlanations; SMOTE: Synthetic minority oversampling technique; TACE: Transarterial chemoembolization

---

## Introduction

Hepatocellular carcinoma (HCC) is the most common primary liver malignancy and the third leading cause of cancer-related mortality worldwide [1]. Transarterial chemoembolization (TACE) is the guideline-recommended treatment for intermediate-stage (BCLC stage B) HCC [2]. However, objective response rates vary widely from 15% to 60%, and a substantial proportion of patients experience tumor progression despite repeated TACE sessions [3,4]. This heterogeneity underscores the critical need for pretreatment biomarkers to guide patient selection and avoid futile interventions.

Radiomics—the high-throughput extraction of quantitative features from medical images—has demonstrated promise for predicting HCC treatment outcomes [5,6]. Recent studies have applied CT-based radiomics to predict microvascular invasion, recurrence, and TACE refractoriness [7–9]. However, three fundamental challenges persist. First, many published models rely on proprietary, single-institution datasets, limiting reproducibility—a concern increasingly recognized as a barrier to clinical translation [10]. Second, predictive models often lack interpretability, undermining clinical trust. Third, the comparative predictive value of different CT contrast phases remains incompletely characterized.

The WAW-TACE dataset [11]—comprising 233 treatment-naïve HCC patients with multiphasic CT, expert tumor segmentations, 3,339 pre-extracted PyRadiomics features from 104 anatomical regions, and comprehensive clinical outcomes—enables fully reproducible radiomics research. We leveraged this resource to: (i) develop an interpretable machine learning framework with SHAP-based explanation for TACE response prediction; (ii) compare predictive performance across all four CT phases; and (iii) assess the prognostic value of radiomics-derived risk stratification for survival.

---

## Materials and Methods

### Study Population

This retrospective study used the publicly available WAW-TACE dataset (Zenodo DOI: 10.5281/zenodo.12741586, CC BY 4.0 license) [11]. The dataset comprises 233 treatment-naïve HCC patients treated with TACE at the Medical University of Warsaw (2016–2021). Institutional review board approval was obtained by the original authors. Inclusion criteria were: age ≥ 18 years, HCC diagnosis by imaging or histopathology, multiphasic abdominal CT within three months before first TACE, and no prior locoregional or systemic therapy.

### CT Acquisition and Preprocessing

Multiphasic contrast-enhanced CT was acquired on four scanner types (GE Optima CT600, Siemens Somatom Xceed, Philips Ingenuity Core, Toshiba Aquilion One) in non-contrast, late arterial, portal venous, and delayed phases. The dataset provides NIfTI-converted CT volumes alongside pre-extracted radiomics features. Multi-organ segmentation masks were generated using TotalSegmentator via nnU-Net [12]; liver tumor masks were hand-drawn by board-certified radiologists.

### Response Definition

The primary endpoint was LI-RADS Treatment Response (LR-TR) on first post-TACE follow-up imaging. Patients with LR-TR "nonviable" (score = 0) were classified as good responders (n = 76), and those with LR-TR "viable" (score = 2) as poor responders (n = 135). Twenty-two patients with equivocal response (LR-TR = 1) were excluded, yielding 210 patients for primary analysis. Secondary endpoints were overall survival (OS) and progression-free survival (PFS).

### Feature Extraction and Selection

Portal venous phase PyRadiomics features (n = 3,305; v3.0.1, IBSI-compliant) encompassed 14 shape features, 18 first-order statistics, and 75 texture features across five matrices (GLCM, GLDM, GLSZM, GLRLM, NGTDM) for each of 104 anatomical volumes of interest plus the largest liver tumor mask. Fourteen clinical variables (age, sex, etiology, lesion characteristics, laboratory values) were included as complementary predictors.

Feature selection followed a two-stage procedure. Stage 1—univariate screening using the Mann-Whitney U test (P < 0.05). Stage 2—L1-regularized logistic regression (LASSO, saga solver, 50 logarithmically spaced C values, five-fold cross-validated λ selection). Clinical variables were appended to the LASSO-selected radiomics features.

### Model Development and Validation

Three classifiers were evaluated: logistic regression (L2, C = 1.0), random forest (200 trees, max depth = 8, minimum samples per leaf = 10), and support vector machine (RBF kernel, isotonic calibration). Five-fold stratified cross-validation was employed. SMOTE was applied to training folds. Primary metrics were AUC, sensitivity, and specificity; calibration was assessed via Brier score.

### SHAP-Based Interpretability

SHAP values were computed using a linear explainer with interventional feature perturbation [13]. Global feature importance was derived from mean absolute SHAP values. Patient-level explanations were generated through waterfall plots. Feature correlation networks (Spearman rank correlation, |r| > 0.35) were constructed for good and poor responders using force-directed graph layout.

### Survival and Phase Analysis

Radiomics-based risk scores dichotomized patients at the median. Kaplan-Meier curves were generated for OS and PFS with log-rank testing. Multivariate Cox regression identified independent prognostic factors. Radiomics performance was compared across all four CT phases using identical procedures.

### Statistical Analysis

Continuous variables were compared using the Mann-Whitney U test; categorical variables using the chi-square test. Exact P-values are reported throughout; P < 0.05 was considered significant. Analyses were performed in Python 3.13 using scikit-learn 1.9, SHAP 0.52, and lifelines 0.30. All analysis code is publicly available.

---

## Results

### Patient Characteristics

The final cohort (n = 210) had a median age of 66 years (IQR 59–72); 69% were male. Predominant etiologies were HCV (42%), HBV (19%), and alcoholic liver disease (12%). Median lesion diameter was 28 mm (IQR 18–45). Median survival was 18 months; 73% died during follow-up. Good and poor responders did not differ significantly in age (P = 0.799), sex (P = 0.852), or lesion count (P = 0.412).

### Predictive Performance

The random forest model achieved the highest AUC (0.831 ± 0.051), compared to 0.771 ± 0.019 for logistic regression and 0.746 ± 0.066 for the clinical-only baseline (ΔAUC = +0.085). Sensitivity was 0.71 and specificity was 0.78. The Brier score was 0.174, indicating satisfactory calibration.

### SHAP Feature Importance

LASSO selected 42 radiomics features spanning 20 anatomical regions. The top SHAP-identified predictors were: liver tumor first-order 10th percentile (SHAP = 1.30), T8 vertebral body root-mean-square (0.96), liver tumor RMS (0.95), right 8th rib maximum 2D diameter (0.77), left 12th rib kurtosis (0.72), alcoholic etiology (0.68), left rectus abdominis axis length (0.66), left 9th rib range (0.61), right kidney robust mean absolute deviation (0.58), and T8 vertebral body median (0.51).

Notably, three of the top ten features derived from skeletal structures, suggesting that CT-based bone radiomics captures systemic frailty relevant to HCC prognosis.

### Phase Comparison

Portal venous phase radiomics achieved the highest performance (AUC = 0.771 ± 0.019), markedly outperforming non-contrast (0.612 ± 0.086, P < 0.001), arterial (0.551 ± 0.080, P < 0.001), and delayed phases (0.543 ± 0.090, P < 0.001).

### Survival Analysis

Radiomics-based risk stratification significantly discriminated OS (median: 22 vs. 14 months; log-rank P < 0.0001) and PFS (median: 12 vs. 6 months; P = 0.039). Multivariate Cox regression identified albumin (HR = 0.59, 95% CI 0.44–0.79, P < 0.001) and tumor diameter (HR = 1.44, 95% CI 1.18–1.76, P < 0.001) as independent prognostic factors.

### Correlation Network Topology

The poor response group exhibited denser feature connectivity (23 nodes, 31 edges) compared to the good response group (23 nodes, 14 edges) at |r| > 0.35, suggesting that treatment resistance is associated with more tightly coordinated radiomics expression patterns.

---

## Discussion

This study demonstrates that an interpretable radiomics framework—deployed exclusively on public data—predicts TACE response (AUC = 0.831) and significantly stratifies survival in HCC. Several findings merit detailed discussion.

**Radiomics augments clinical prediction.** The ΔAUC of +0.085 over the clinical baseline is clinically meaningful, particularly as the clinical model already incorporated established prognostic variables including albumin and tumor size. The finding that radiomics features from extrahepatic organs—specifically vertebrae and ribs—figure prominently among top predictors extends prior work on opportunistic CT screening [14] into the radiomics domain. The unexpected prominence of skeletal features suggests that routine staging CT captures bone texture alterations reflecting systemic frailty, sarcopenia, and cirrhosis-associated metabolic bone disease [15,16], consistent with the established prognostic value of CT-derived sarcopenia in HCC [17].

**Portal venous phase is optimal.** Portal venous phase radiomics significantly outperformed all other phases, consistent with the dominant portal venous blood supply of HCC and superior liver-to-tumor contrast in this phase [18]. While arterial phase imaging is essential for HCC diagnosis, our findings support the use of portal venous phase for quantitative radiomics analysis.

**SHAP enables clinically transparent interpretation.** The SHAP-based feature ranking provided biologically interpretable predictors. The liver tumor's first-order 10th percentile—reflecting the darkest 10% of tumor voxels—was the strongest predictor, consistent with the known association between tumor necrosis/hypoattenuation and aggressive biology. Identifying specific vertebral levels (T8) and individual ribs (right 8th, left 9th/12th) as predictors provides anatomical specificity potentially relevant for future mechanistic investigation.

**Reproducibility through open science.** A distinguishing feature of this study is its exclusive reliance on publicly available data and open-source software. Every analytical step—from feature selection through SHAP visualization—is fully reproducible without institutional data access. This design explicitly addresses the reproducibility concerns increasingly recognized in radiomics research [10].

**Limitations.** First, the WAW-TACE dataset represents a single-center European cohort with HCV-predominant etiology, limiting generalizability to HBV-dominant Asian populations [1]. External validation on independent cohorts is required before clinical application. Second, the dataset includes only pre-treatment CT, precluding delta radiomics analysis of post-TACE vascular remodeling. Third, tumor masks were too small for robust peritumoral vessel analysis as an adjunct to conventional radiomics. Fourth, inter-scanner variability—although partially addressed through IBSI-compliant feature extraction—may introduce residual batch effects. Fifth, SHAP explanations remain statistically derived and require independent biological validation.

**Future directions.** External validation on multi-ethnic, HBV-dominant cohorts is a priority. Prospective evaluation of whether radiomics-based risk stratification can guide early treatment decisions warrants investigation. As larger public datasets with comprehensive tumor and vessel annotations become available, peritumoral vascular morphometry should be revisited.

---

## References

1. Sung H, Ferlay J, Siegel RL, et al. Global cancer statistics 2020: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. CA Cancer J Clin 2021;71:209–249.

2. Reig M, Forner A, Rimola J, et al. BCLC strategy for prognosis prediction and treatment recommendation: the 2022 update. J Hepatol 2022;76:681–693.

3. Llovet JM, Kelley RK, Villanueva A, et al. Hepatocellular carcinoma. Nat Rev Dis Primers 2021;7:6.

4. Lencioni R, de Baere T, Soulen MC, Rilling WS, Geschwind JFH. Lipiodol transarterial chemoembolization for hepatocellular carcinoma: a systematic review of efficacy and safety data. Hepatology 2016;64:106–116.

5. Lambin P, Leijenaar RTH, Deist TM, et al. Radiomics: the bridge between medical imaging and personalized medicine. Nat Rev Clin Oncol 2017;14:749–762.

6. Gillies RJ, Kinahan PE, Hricak H. Radiomics: images are more than pictures, they are data. Radiology 2016;278:563–577.

7. Wei J, Jiang H, Zeng M, et al. CT radiomics for prediction of vessels encapsulating tumor clusters in hepatocellular carcinoma. J Hepatocell Carcinoma 2025;12:147–159.

8. Zhong X, Long H, Chen L, et al. CT-based radiomics for microvascular invasion and recurrence-free survival prediction in HCC. BMC Med Imaging 2025;25:42.

9. Wang Q, Li C, Zhang Y, et al. CT radiomics for predicting TACE refractoriness in hepatocellular carcinoma. Eur Radiol 2024;34:7105–7115.

10. Zwanenburg A, Vallières M, Abdalah MA, et al. The Image Biomarker Standardization Initiative: standardized quantitative radiomics for high-throughput image-based phenotyping. Radiology 2020;295:328–338.

11. Bartnik K, Bartczak T, Krzyziński M, et al. WAW-TACE: a hepatocellular carcinoma multiphase CT dataset with segmentations, radiomics features, and clinical data. Radiol Artif Intell 2024;6:e240296.

12. Wasserthal J, Breit HC, Meyer MT, et al. TotalSegmentator: robust segmentation of 104 anatomic structures in CT images. Radiol Artif Intell 2023;5:e230024.

13. Lundberg SM, Lee SI. A unified approach to interpreting model predictions. Adv Neural Inf Process Syst 2017;30:4765–4774.

14. Pickhardt PJ, Summers RM, Garrett JW, et al. Opportunistic screening: radiology and AI in the era of precision medicine. Radiology 2023;307:e222044.

15. Carey EJ, Lai JC, Sonnenday C, et al. A North American expert opinion statement on sarcopenia in liver transplantation. Hepatology 2019;70:1816–1829.

16. Tandon P, Montano-Loza AJ, Lai JC, Dasarathy S, Merli M. Sarcopenia and frailty in decompensated cirrhosis. J Hepatol 2021;75:S147–S162.

17. Meza-Junco J, Montano-Loza AJ, Baracos VE, et al. Sarcopenia as a prognostic index of nutritional status in concurrent cirrhosis and hepatocellular carcinoma. J Clin Gastroenterol 2013;47:861–870.

18. Choi JY, Lee JM, Sirlin CB. CT and MR imaging diagnosis and staging of hepatocellular carcinoma: part I. Development, growth, and spread: key pathologic and imaging aspects. Radiology 2014;272:635–654.

---

## Declarations

**Funding:** No external funding was received for this study.

**Conflicts of interest:** The authors declare no competing interests.

**Ethics approval:** This study used exclusively publicly available, de-identified data (WAW-TACE dataset, CC BY 4.0). The original data collection was approved by the Institutional Review Board of the Medical University of Warsaw.

**Guarantor:** The scientific guarantor of this publication is Kuan Liu (刘宽).

**Data and code availability:** All data are publicly available from Zenodo (DOI: 10.5281/zenodo.12741586). The complete analysis code is available at https://github.com/Liukuan19920919/hcc-tace-radiomics.

**Author contributions:** Kuan Liu contributed to study conception, data analysis, and manuscript writing.
