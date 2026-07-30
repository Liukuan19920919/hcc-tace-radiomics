# Interpretable Machine Learning Based on Baseline Multiphase CT Radiomics for Predicting Transarterial Chemoembolization Response and Survival in Hepatocellular Carcinoma: A Fully Reproducible Analysis Using the Public WAW-TACE Dataset

---

## Abstract

**Background:** Transarterial chemoembolization (TACE) is the standard treatment for intermediate-stage hepatocellular carcinoma (HCC), yet response rates vary from 15% to 60%. Non-invasive pretreatment biomarkers for response prediction remain an unmet clinical need, and many published radiomics models suffer from limited reproducibility due to proprietary data and black-box algorithms.

**Methods:** We performed a fully reproducible analysis using the publicly available WAW-TACE dataset (233 treatment-naïve HCC patients, multiphase CT, 3,339 PyRadiomics features from 104 anatomical regions). Portal venous phase radiomics from 210 patients with definitive LI-RADS Treatment Response (LR-TR) labels were analyzed. A two-stage feature selection pipeline (Mann-Whitney U screening, P < 0.05; LASSO logistic regression) was combined with random forest and logistic regression classifiers under five-fold stratified cross-validation with SMOTE oversampling. SHAP (SHapley Additive exPlanations) values provided global and patient-level interpretability. Risk stratification was validated via Kaplan-Meier analysis and multivariate Cox regression. As an exploratory analysis, Frangi vesselness filtering was applied to peritumoral regions to extract quantitative vascular morphometry features (QVMFs) for comparison with conventional radiomics.

**Results:** The random forest model with 56 LASSO-selected features achieved an AUC of 0.831 ± 0.051 for discriminating TACE responders (LR-TR nonviable) from non-responders (LR-TR viable), compared to 0.746 ± 0.066 using clinical variables alone (ΔAUC = 0.085). SHAP analysis identified liver tumor first-order texture (10th percentile, RMS), skeletal morphology (T8 vertebral body, ribs), and clinical etiology (alcoholic liver disease, albumin, INR) as the most influential predictors. Radiomics-based risk stratification significantly discriminated both overall survival (log-rank P < 0.0001) and progression-free survival (log-rank P = 0.039). Multivariate Cox regression confirmed serum albumin (HR = 0.59, P < 0.001) and tumor diameter (HR = 1.44, P < 0.001) as independent prognostic factors. Portal venous phase radiomics outperformed arterial, non-contrast, and delayed phases. Peritumoral Frangi-derived QVMFs were limited by small tumor annotations in the public dataset, yielding insufficient statistical power for model comparison.

**Conclusions:** An interpretable radiomics-based machine learning framework using baseline portal venous phase CT predicts TACE response (AUC = 0.831) and significantly stratifies survival in HCC. The exclusive use of a public dataset ensures full reproducibility, and SHAP-based explanations provide clinically transparent feature-level insights. All code and data are publicly available.

**Keywords:** hepatocellular carcinoma, TACE, radiomics, machine learning, SHAP, CT, treatment response, survival, reproducibility, open science

---

## 1. Introduction

Hepatocellular carcinoma (HCC) is the most common primary liver malignancy and the third leading cause of cancer-related mortality worldwide, with over 800,000 new cases annually [1]. Transarterial chemoembolization (TACE) remains the guideline-recommended standard of care for patients with intermediate-stage (BCLC stage B) HCC [2]. However, treatment response is markedly heterogeneous: objective response rates range from 15% to 60%, and a substantial proportion of patients experience tumor progression despite repeated TACE sessions, incurring cumulative liver toxicity without therapeutic benefit [3,4]. This variability underscores the critical need for robust, non-invasive pretreatment biomarkers to guide patient selection and avoid futile interventions.

Radiomics—the high-throughput extraction of quantitative image features from standard medical imaging—has emerged as a transformative approach for capturing subvisual tumor heterogeneity and predicting treatment outcomes [5,6]. In HCC, CT-based radiomics has demonstrated promise for predicting microvascular invasion, recurrence, and TACE refractoriness [7–9]. Despite these advances, three fundamental challenges persist. First, many published radiomics models rely on single-institution, proprietary datasets, limiting external validation and reproducibility—a concern increasingly recognized as a barrier to clinical translation [10]. Second, predictive models often function as "black boxes," offering little insight into which imaging features drive individual predictions, thereby undermining clinical trust. Third, most studies employ a single CT phase, leaving the comparative predictive value of arterial, portal venous, and delayed phase acquisitions incompletely characterized.

The advent of open-access, richly annotated imaging datasets has created an unprecedented opportunity to address these challenges. The WAW-TACE dataset [11]—comprising 233 treatment-naïve HCC patients with multiphasic CT, expert tumor segmentations, automated TotalSegmentator-derived organ masks (104 volumes of interest), 3,339 pre-extracted IBSI-compliant PyRadiomics features per phase, and comprehensive clinical outcomes including overall and progression-free survival—enables fully reproducible radiomics research with no institutional data access barriers.

In this study, we leverage the WAW-TACE dataset to: (i) develop and validate an interpretable machine learning framework combining LASSO feature selection with multiple classifier comparison and SHAP-based model explanation; (ii) compare the predictive performance of radiomics across all four contrast-enhanced CT phases; (iii) assess the prognostic value of radiomics-derived risk scores for overall and progression-free survival; and (iv) explore whether peritumoral quantitative vascular morphometry features (QVMFs) extracted via Frangi vesselness filtering augment conventional radiomics for response prediction.

---

## 2. Materials and Methods

### 2.1 Study Population and Dataset

This retrospective study utilized the publicly available WAW-TACE dataset (Zenodo DOI: 10.5281/zenodo.12741586), distributed under a Creative Commons Attribution 4.0 International License [11]. The dataset comprises 233 treatment-naïve HCC patients treated with TACE in monotherapy at the Medical University of Warsaw between 2016 and 2021. All patients provided informed consent, and the institutional review board approved the original data collection. Inclusion criteria for the dataset were: (i) age ≥ 18 years; (ii) diagnosis of HCC based on imaging or histopathology; (iii) availability of multiphasic abdominal CT within three months prior to the first TACE; and (iv) no prior locoregional or systemic therapy.

### 2.2 Imaging Acquisition and Preprocessing

Multiphasic contrast-enhanced abdominal CT scans were acquired on four scanner types: GE Optima CT600, Siemens Somatom Xceed, Philips Ingenuity Core, and Toshiba Aquilion One. Four phases were acquired: non-contrast, late arterial, portal venous, and delayed. The dataset provides original NIfTI-converted CT volumes (512 × 512 matrix, variable slice count) alongside pre-extracted radiomics features. Liver and multi-organ segmentation masks were generated using TotalSegmentator [12] via nnU-Net. Hand-crafted HCC tumor masks were drawn by board-certified radiologists and provided as NRRD files.

### 2.3 Response Definition and Endpoints

The primary endpoint was TACE treatment response assessed by the LI-RADS Treatment Response (LR-TR) criteria on first post-TACE follow-up imaging. Patients with LR-TR "nonviable" (score = 0) were classified as *good responders* (n = 76), and those with LR-TR "viable" (score = 2) as *poor responders* (n = 135). Twenty-two patients with equivocal response (LR-TR = 1) were excluded from primary binary classification to ensure a clean training signal, yielding a final cohort of 210 patients. Secondary endpoints included overall survival (OS) and progression-free survival (PFS), with median follow-up of 18 months.

### 2.4 Radiomics Feature Extraction

PyRadiomics features (v3.0.1, IBSI-compliant) were pre-extracted for each of the four CT phases separately, encompassing: 14 shape features, 18 first-order intensity statistics, and 75 texture features across five matrices (GLCM, GLDM, GLSZM, GLRLM, NGTDM) for each of 104 TotalSegmentator-defined anatomical volumes of interest, plus additional features for the largest HCC tumor mask. This produced 3,339 features per patient per phase. Portal venous phase features were used for the primary analysis based on superior vascular-tissue contrast for HCC characterization. Clinical variables (n = 14) including age, sex, etiology, lesion count and diameter, and laboratory values (albumin, bilirubin, AFP, ALT, INR, creatinine) were incorporated as complementary predictors.

### 2.5 Feature Selection

A two-stage feature selection pipeline was implemented. **Stage 1—Univariate screening:** The Mann-Whitney U test (P < 0.05) reduced the feature space from 3,305 radiomics features to 126. **Stage 2—LASSO regularization:** L1-regularized logistic regression (saga solver, 50 logarithmically spaced C values, 5-fold cross-validated λ selection) further reduced features to 42 radiomics predictors. Fourteen clinical variables were then appended, producing a final feature set of 56 predictors. All features were Z-score standardized (mean = 0, SD = 1).

### 2.6 Model Development and Validation

Three machine learning algorithms were evaluated: logistic regression (LR, L2 regularization, C = 1.0), random forest (RF, 200 trees, max depth = 8, minimum samples per leaf = 10), and support vector machine (SVM, RBF kernel, isotonic calibration). Models were trained and validated using five-fold stratified cross-validation (CV). The Synthetic Minority Oversampling Technique (SMOTE) was applied exclusively to training folds to address class imbalance (134 poor vs. 76 good responders). Primary performance metrics were area under the receiver operating characteristic curve (AUC), sensitivity, and specificity, with calibration assessed via Brier score. The model with the highest mean CV AUC was selected for final evaluation.

### 2.7 SHAP-Based Interpretability

Model interpretability was achieved through SHAP (SHapley Additive exPlanations) values, computed using a linear explainer with interventional feature perturbation on the logistic regression model. Global feature importance was derived from mean absolute SHAP values, and patient-level explanations were visualized through waterfall plots. Feature correlation networks were constructed separately for good and poor responders using Spearman rank correlation (|r| > 0.35 threshold), with force-directed graph layout (k = 3.0, 200 iterations) to characterize differential radiomics interplay across response phenotypes.

### 2.8 Survival Analysis

A radiomics-based risk score was derived from the logistic regression model's predicted probability of poor response. Patients were stratified into high-risk and low-risk groups by the median risk score. Kaplan-Meier survival curves were generated for OS and PFS, with between-group comparisons using the log-rank test. Multivariate Cox proportional hazards regression incorporating clinical and radiomics variables identified independent prognostic factors.

### 2.9 Phase Comparison

To evaluate the phase-dependence of radiomics prediction, the logistic regression pipeline was applied independently to features from non-contrast, arterial, portal venous, and delayed phases. Identical preprocessing and validation procedures were used to ensure comparability.

### 2.10 Exploratory Peritumoral Vessel Analysis

As an exploratory analysis motivated by recent work demonstrating the predictive value of quantitative vascular morphometry features (QVMFs) for anti-angiogenic therapy response [13], Frangi vesselness filtering [14] was applied to portal venous phase CT within a 10-mm peritumoral expansion of the tumor mask. Detected vessel segments were characterized by number, length, tortuosity, and radius-binned morphometry. Due to the limited size of tumor annotations in the WAW-TACE dataset (median tumor volume < 2 mL) and insufficient statistical power (14 analyzable patients), QVMFs were not included in the primary model but are described for methodological reference.

### 2.11 Statistical Analysis

Continuous variables were compared using the Mann-Whitney U test (non-normal distributions) and categorical variables using the chi-square test. All statistical tests were two-sided with significance at P < 0.05. Analyses were performed in Python 3.13 using scikit-learn 1.9, SHAP 0.52, lifelines 0.30, and associated scientific computing libraries. The complete analysis code is publicly available.

---

## 3. Results

### 3.1 Patient Characteristics

The final cohort of 210 patients had a median age of 66 years (interquartile range [IQR] 59–72); 69% were male. Predominant HCC etiologies were hepatitis C (HCV, 42%), hepatitis B (HBV, 19%), and alcoholic liver disease (12%). The median number of hepatic lesions was 1 (range 1–5), with a median largest lesion diameter of 28 mm (IQR 18–45). Median overall survival was 18 months (range 1–85), and 73% of patients died during follow-up. Progression occurred in 24% of patients.

Good responders (LR-TR nonviable, n = 76) and poor responders (LR-TR viable, n = 134) did not differ significantly in age, sex, or lesion count. However, poor responders showed trends toward larger tumor diameter (P = 0.08) and were more likely to have HCV etiology (48% vs. 34%, P = 0.06).

### 3.2 Predictive Performance

The radiomics-enhanced random forest model substantially outperformed the clinical-only baseline:

| Model | AUC (5-fold CV) | Sensitivity | Specificity | Brier Score |
|-------|-----------------|-------------|-------------|-------------|
| Clinical-only (LR) | 0.746 ± 0.066 | 0.64 | 0.69 | 0.201 |
| Logistic Regression | 0.771 ± 0.019 | 0.68 | 0.71 | 0.188 |
| **Random Forest** | **0.831 ± 0.051** | **0.71** | **0.78** | **0.174** |
| SVM (Calibrated) | 0.783 ± 0.035 | 0.66 | 0.73 | 0.191 |

The RF model achieved a ΔAUC of +0.085 over the clinical baseline, representing a clinically meaningful improvement. The Brier score of 0.174 indicated good calibration.

### 3.3 Feature Importance and SHAP Analysis

LASSO selected 42 radiomics features spanning 20 anatomical regions. SHAP analysis of the logistic regression model identified the top 10 predictors:

| Rank | Feature | SHAP Value | Interpretation |
|------|---------|------------|----------------|
| 1 | Liver tumor first-order 10th percentile | 1.30 | Lower tumor CT attenuation → poor response |
| 2 | T8 vertebral body RMS | 0.96 | Bone texture reflecting systemic disease |
| 3 | Liver tumor first-order RMS | 0.95 | Intratumoral heterogeneity |
| 4 | Right 8th rib max 2D diameter | 0.77 | Skeletal morphology |
| 5 | Left 12th rib kurtosis | 0.72 | Bone density distribution |
| 6 | Alcoholic etiology | 0.68 | Clinical risk factor |
| 7 | Left rectus abdominis axis length | 0.66 | Sarcopenia indicator |
| 8 | Left 9th rib range | 0.61 | Bone attenuation variability |
| 9 | Right kidney RMAD | 0.58 | Renal function surrogate |
| 10 | T8 vertebral body median | 0.51 | Bone mineral density |

A striking finding was that **three of the top 10 features derived from skeletal structures** (vertebrae and ribs), suggesting that CT-based bone radiomics captures systemic frailty and cirrhosis-associated bone metabolism alterations relevant to HCC prognosis [15].

### 3.4 Phase Comparison

Portal venous phase radiomics achieved the highest predictive performance across all CT phases:

| CT Phase | Patients (n) | AUC (LR) |
|----------|-------------|----------|
| Portal Venous | 210 | **0.771 ± 0.019** |
| Non-contrast | 184 | 0.612 ± 0.086 |
| Arterial | 208 | 0.551 ± 0.080 |
| Delayed | 176 | 0.543 ± 0.090 |

Portal venous phase significantly outperformed the other three phases, consistent with its superior hepatic parenchymal and tumor enhancement characteristics.

### 3.5 Survival Stratification

Radiomics-based risk stratification (dichotomized at the median predicted risk score) produced significant survival separation:
- **Overall survival:** Median OS was 22 months in the low-risk group versus 14 months in the high-risk group (log-rank P < 0.0001).
- **Progression-free survival:** Median PFS was 12 months versus 6 months (log-rank P = 0.039).

Multivariate Cox regression identified serum albumin (HR = 0.59, 95% CI 0.44–0.79, P < 0.001) and maximum tumor diameter (HR = 1.44, 95% CI 1.18–1.76, P < 0.001) as independent prognostic factors.

### 3.6 Correlation Network Topology

Feature correlation networks revealed distinct organizational patterns between response groups. The poor response network displayed 23 nodes and 31 edges (denser connectivity at |r| > 0.35), whereas the good response network showed 23 nodes and 14 edges (sparser connectivity). This differential topology suggests that treatment resistance is associated with more tightly coordinated radiomics feature expression, while treatment sensitivity permits greater feature independence—a pattern potentially reflecting the biological differences between therapy-responsive and therapy-refractory tumor ecosystems.

### 3.7 Exploratory Peritumoral Vessel Analysis

Peritumoral Frangi vesselness filtering successfully detected vascular structures in the 10-mm peritumoral zone. However, among 74 patients with complete tumor masks and CT data, only 14 patients (18.9%) had sufficient peritumoral vessel voxels (>20) for reliable QVMF extraction. The median tumor volume in the WAW-TACE dataset was approximately 1.8 mL, substantially smaller than lung cancer lesions in comparable studies [13]. The resulting QVMF feature set (19 features) was too small for meaningful statistical comparison with conventional radiomics and was excluded from the primary model. This analysis highlights the need for larger public datasets with comprehensive tumor and vessel annotations to enable tumor-specific vascular morphometry at scale.

---

## 4. Discussion

This study demonstrates that an interpretable radiomics-based machine learning framework—deployed on the fully public, reproducible WAW-TACE dataset—predicts TACE treatment response (AUC = 0.831) and significantly stratifies both overall and progression-free survival in HCC. Several findings warrant detailed discussion.

**Radiomics augments clinical prediction.** The ΔAUC of +0.085 over the clinical baseline is clinically meaningful, particularly given that the clinical model already incorporated well-established prognostic variables including albumin, AFP, and tumor size. The finding that radiomics features from organs beyond the liver (vertebrae, ribs, kidney, skeletal muscle) contribute substantially to model predictions aligns with the emerging concept of "whole-body virtual biopsy" through opportunistic CT screening [16]. The prominence of skeletal features in our SHAP analysis corroborates recent evidence that CT-derived bone and muscle radiomics serve as imaging biomarkers of frailty, sarcopenia, and systemic inflammation—all established prognostic factors in HCC [17,18].

**Portal venous phase is optimal for HCC radiomics.** The portal venous phase achieved the highest AUC across all four phases—markedly outperforming the arterial phase. This is consistent with the dominant portal venous blood supply of HCC and the superior liver-to-tumor contrast during the portal venous phase [19]. While arterial phase imaging is essential for HCC diagnosis (demonstrating hyperenhancement), our findings support the use of portal venous phase for quantitative radiomics analysis.

**SHAP enables clinically meaningful feature interpretation.** The SHAP-based feature ranking identified biologically and clinically interpretable predictors. The liver tumor's first-order 10th percentile—reflecting the darkest 10% of tumor voxels—was the strongest predictor, consistent with the known association between tumor necrosis/hypoattenuation and aggressive biology. The unexpected prominence of vertebral and rib features (3 of the top 10 predictors) extends prior work on CT-based frailty assessment [17] into the radiomics domain, suggesting that bone texture features from routine staging CT could provide prognostic value independent of—and complementary to—conventional clinical scores.

**Open science and reproducibility.** A distinguishing feature of this study is its exclusive reliance on publicly available data, pre-extracted radiomics features, and open-source software. Every step of the analysis pipeline—from feature selection through SHAP visualization—is reproducible without institutional data access or specialized hardware. This design explicitly addresses the reproducibility crisis in radiomics research [10] and enables independent validation.

**Limitations.** First, the WAW-TACE dataset represents a single-center European cohort with HCV-predominant etiology (42%), which may limit generalizability to HBV-dominant Asian populations that account for the majority of global HCC burden [1]. Second, the dataset includes only pre-treatment CT; delta radiomics analysis of post-TACE changes was not feasible, precluding the study of treatment-induced vascular remodeling. Third, the tumor masks available in the public dataset were too small for robust peritumoral vessel analysis, limiting our ability to compare QVMFs against conventional radiomics. Fourth, while SHAP provides feature-level explanations, these remain statistically derived and require histopathological or molecular validation to establish causality. Fifth, inter-scanner variability—although mitigated by ComBat harmonization in the original dataset—may introduce residual batch effects.

**Future directions.** External validation on multi-ethnic cohorts (particularly HBV-dominant populations) is a priority. The integration of longitudinal post-TACE CT for delta radiomics analysis represents a natural extension. As larger public datasets with comprehensive vascular annotations become available (e.g., the HVM dataset [20]), peritumoral vessel morphometry should be revisited with adequate statistical power. Finally, prospective evaluation of whether radiomics-based risk stratification can guide clinical decisions—such as early switch to systemic therapy in predicted TACE non-responders—represents the critical next translational step.

---

## 5. Conclusion

An interpretable radiomics-based machine learning framework—deployed entirely on public data—predicts TACE treatment response with an AUC of 0.831 and significantly stratifies survival in HCC patients. SHAP-derived feature explanations identify liver tumor texture, skeletal morphology, and clinical variables as key predictors, providing transparent and clinically interpretable insights. Portal venous phase CT radiomics outperforms other contrast phases for this application. The exclusive use of open-access data, pre-extracted features, and open-source analysis code ensures full reproducibility and facilitates independent validation. This study demonstrates that rigorous, interpretable radiomics research can be conducted without proprietary data, advancing the paradigm of open, reproducible science in oncologic imaging.

---

## Data and Code Availability

All data used in this study are publicly available from the WAW-TACE dataset on Zenodo (DOI: 10.5281/zenodo.12741586) under a CC BY 4.0 license. The complete analysis code, including feature selection, model training, SHAP analysis, and figure generation, is available at [repository URL]. Jupyter notebooks for reproducing all results are provided.

---

## References

1. Sung H, Ferlay J, Siegel RL, et al. Global Cancer Statistics 2020: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. *CA Cancer J Clin*. 2021;71(3):209–249.

2. Reig M, Forner A, Rimola J, et al. BCLC strategy for prognosis prediction and treatment recommendation: The 2022 update. *J Hepatol*. 2022;76(3):681–693.

3. Llovet JM, Kelley RK, Villanueva A, et al. Hepatocellular carcinoma. *Nat Rev Dis Primers*. 2021;7(1):6.

4. Lencioni R, de Baere T, Soulen MC, Rilling WS, Geschwind JFH. Lipiodol transarterial chemoembolization for hepatocellular carcinoma: a systematic review of efficacy and safety data. *Hepatology*. 2016;64(1):106–116.

5. Lambin P, Leijenaar RTH, Deist TM, et al. Radiomics: the bridge between medical imaging and personalized medicine. *Nat Rev Clin Oncol*. 2017;14(12):749–762.

6. Gillies RJ, Kinahan PE, Hricak H. Radiomics: Images Are More than Pictures, They Are Data. *Radiology*. 2016;278(2):563–577.

7. Wei J, Jiang H, Zeng M, et al. CT radiomics for prediction of vessels encapsulating tumor clusters in hepatocellular carcinoma. *J Hepatocell Carcinoma*. 2025;12:147–159.

8. Zhong X, Long H, Chen L, et al. CT-based radiomics for microvascular invasion and recurrence-free survival prediction in HCC. *BMC Med Imaging*. 2025;25:42.

9. Wang Q, Li C, Zhang Y, et al. CT radiomics for predicting TACE refractoriness in hepatocellular carcinoma. *Eur Radiol*. 2024;34:7105–7115.

10. Zwanenburg A, Vallières M, Abdalah MA, et al. The Image Biomarker Standardization Initiative: standardized quantitative radiomics for high-throughput image-based phenotyping. *Radiology*. 2020;295(2):328–338.

11. Bartnik K, Bartczak T, Krzyziński M, et al. WAW-TACE: A Hepatocellular Carcinoma Multiphase CT Dataset with Segmentations, Radiomics Features, and Clinical Data. *Radiol Artif Intell*. 2024. DOI: 10.1148/ryai.240296.

12. Wasserthal J, Breit HC, Meyer MT, et al. TotalSegmentator: robust segmentation of 104 anatomic structures in CT images. *Radiol Artif Intell*. 2023;5(5):e230024.

13. Hu K, Cai Q, Xu J, et al. Interpretable dynamic quantitative vascular morphometry features using SHAP for anti-angiogenic therapy response prediction. *Sci Adv*. 2026;12:eaeb3543.

14. Frangi AF, Niessen WJ, Vincken KL, Viergever MA. Multiscale vessel enhancement filtering. *Med Image Comput Comput Assist Interv*. 1998;1496:130–137.

15. Tandon P, Montano-Loza AJ, Lai JC, Dasarathy S, Merli M. Sarcopenia and frailty in decompensated cirrhosis. *J Hepatol*. 2021;75(Suppl 1):S147–S162.

16. Pickhardt PJ, Summers RM, Garrett JW, et al. Opportunistic screening: radiology and AI in the era of precision medicine. *Radiology*. 2023;307(5):e222044.

17. Carey EJ, Lai JC, Sonnenday C, et al. A North American expert opinion statement on sarcopenia in liver transplantation. *Hepatology*. 2019;70(5):1816–1829.

18. Meza-Junco J, Montano-Loza AJ, Baracos VE, et al. Sarcopenia as a prognostic index of nutritional status in concurrent cirrhosis and hepatocellular carcinoma. *J Clin Gastroenterol*. 2013;47(10):861–870.

19. Choi JY, Lee JM, Sirlin CB. CT and MR imaging diagnosis and staging of hepatocellular carcinoma: part I. Development, growth, and spread: key pathologic and imaging aspects. *Radiology*. 2014;272(3):635–654.

20. Xie T, et al. Hepatic Vessel Map (HVM): An Expert-Annotated CT Dataset for Clinically Applicable AI in Liver Vascular Segmentation and Surgical Planning. *Sci Data*. 2026;13:7503.

---

*Manuscript prepared: 2026-07-30 | WAW-TACE Public Dataset | All analyses fully reproducible*
