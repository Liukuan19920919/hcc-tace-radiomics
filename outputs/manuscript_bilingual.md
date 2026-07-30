# 基于基线多期CT放射组学的可解释机器学习预测肝细胞癌TACE治疗反应和生存：一项使用公共数据集的全可复现分析

# Interpretable Machine Learning Based on Baseline Multiphase CT Radiomics for Predicting Transarterial Chemoembolization Response and Survival in Hepatocellular Carcinoma: A Fully Reproducible Analysis Using a Public Dataset

---

## 摘要 | Abstract

**目的：** 利用基线多期CT放射组学开发可解释的机器学习模型，预测肝细胞癌（HCC）经动脉化疗栓塞（TACE）的治疗反应和生存，完全基于公共数据集以确保可复现性。

**Objectives:** To develop an interpretable machine learning model using baseline multiphase CT radiomics for predicting transarterial chemoembolization (TACE) response and survival in hepatocellular carcinoma (HCC).

**方法：** 分析来自公共WAW-TACE数据集（210例具有明确LI-RADS治疗反应标签的HCC患者）的门静脉期CT放射组学特征（n = 3,305）。采用两阶段特征选择（Mann-Whitney U检验，P < 0.05；LASSO逻辑回归）和五折交叉验证及SMOTE。比较随机森林、逻辑回归和支持向量机分类器。SHAP值提供可解释性。通过Kaplan-Meier和Cox回归评估生存。

**Methods:** Portal venous phase CT radiomics (n = 3,305 features) from the public WAW-TACE dataset (210 HCC patients) were analyzed with two-stage feature selection (Mann-Whitney U, LASSO) and five-fold cross-validation with SMOTE. Random forest, logistic regression, and support vector machine classifiers were compared. SHAP values provided interpretability. Survival was assessed via Kaplan-Meier and Cox regression.

**结果：** 随机森林对区分TACE治疗有效与无效达到AUC 0.831 ± 0.051，而仅使用临床变量的AUC为0.746 ± 0.066。SHAP分析确定肝脏肿瘤纹理、骨骼形态（T8椎体、肋骨）和临床病因为最强预测因子。基于放射组学的风险分层显著区分总生存期（P < 0.0001）和无进展生存期（P = 0.039）。门静脉期优于动脉期、平扫期和延迟期。

**Results:** Random forest achieved an AUC of 0.831 ± 0.051 versus 0.746 ± 0.066 for clinical variables alone. SHAP analysis identified liver tumor texture, skeletal morphology (T8 vertebra, ribs), and clinical etiology as strongest predictors. Radiomics-based risk stratification significantly discriminated overall survival (P < 0.0001) and progression-free survival (P = 0.039). Portal venous phase outperformed arterial, non-contrast, and delayed phases.

**结论：** 一个完全基于公共数据的可解释放射组学框架可预测TACE治疗反应（AUC = 0.831）并对HCC生存进行分层。SHAP解释识别了具有临床意义的影像预测因子，包括此前未被充分认识的骨骼特征。

**Conclusion:** An interpretable radiomics framework using exclusively public data predicts TACE response (AUC = 0.831) and stratifies survival in HCC. SHAP explanations identified clinically meaningful predictors including previously underrecognized skeletal features.

---

## 关键词 | Keywords

肝细胞癌；化疗栓塞，治疗性；体层摄影术，X线计算机；放射组学；机器学习

Carcinoma, Hepatocellular; Chemoembolization, Therapeutic; Tomography, X-Ray Computed; Radiomics; Machine Learning

---

## 关键点 | Key Points

1. 门静脉期CT放射组学可预测HCC的TACE治疗反应（AUC = 0.831）  
   Radiomics from portal venous CT predicts TACE response in HCC (AUC = 0.831).

2. SHAP分析揭示骨骼CT特征是治疗结局的关键预测因子  
   SHAP analysis reveals skeletal CT features as key predictors of treatment outcome.

3. 基于放射组学的风险分层显著区分总生存期（P < 0.0001）  
   Radiomics-based risk stratification significantly discriminates overall survival (P < 0.0001).

4. 门静脉期优于动脉期、平扫期和延迟期放射组学  
   Portal venous phase outperforms arterial, non-contrast, and delayed phase radiomics.

5. 所有分析使用完全公开的数据集，确保完全可复现  
   All analyses use a fully public dataset, ensuring complete reproducibility.

---

## 临床意义声明 | Clinical Relevance Statement

基线门静脉期CT放射组学通过可解释的机器学习框架分析，为HCC患者提供TACE治疗反应的无创预测和生存分层。骨骼放射组学特征被识别为关键预测因子，表明常规分期CT可捕捉与治疗结局相关的系统性衰弱信息，无需额外影像检查即可支持个体化治疗决策。

Baseline portal venous phase CT radiomics, analyzed through an interpretable machine learning framework, provides non-invasive prediction of TACE treatment response and survival stratification in HCC. The identification of skeletal radiomics features as key predictors suggests that routine staging CT captures systemic frailty information relevant to treatment outcomes, supporting potential integration into personalized treatment planning without additional imaging.

---

## 引言 | Introduction

肝细胞癌（HCC）是最常见的原发性肝脏恶性肿瘤，也是全球癌症相关死亡的第三大原因[1]。经动脉化疗栓塞（TACE）是中期HCC（BCLC B期）的指南推荐治疗方法[2]。然而，客观反应率在15%至60%之间差异很大，相当一部分患者尽管反复接受TACE治疗仍出现肿瘤进展[3,4]。这种异质性凸显了对治疗前生物标志物的迫切需求，以指导患者选择并避免无效治疗。

Hepatocellular carcinoma (HCC) is the most common primary liver malignancy and the third leading cause of cancer-related mortality worldwide [1]. Transarterial chemoembolization (TACE) is the guideline-recommended treatment for intermediate-stage (BCLC stage B) HCC [2]. However, objective response rates vary widely from 15% to 60%, and a substantial proportion of patients experience tumor progression despite repeated TACE sessions [3,4]. This heterogeneity underscores the critical need for pretreatment biomarkers to guide patient selection and avoid futile interventions.

放射组学——从医学影像中高通量提取定量特征——已展现出预测HCC治疗结局的前景[5,6]。近期研究已将CT放射组学应用于预测微血管侵犯、复发和TACE难治性[7–9]。然而，三个根本性挑战仍然存在。首先，许多已发表的模型依赖于专有的单机构数据集，限制了可复现性——这一问题日益被认为是临床转化的障碍[10]。其次，预测模型常缺乏可解释性，削弱了临床信任。第三，不同CT对比期的相对预测价值尚未被充分表征。

Radiomics—the high-throughput extraction of quantitative features from medical images—has demonstrated promise for predicting HCC treatment outcomes [5,6]. Recent studies have applied CT-based radiomics to predict microvascular invasion, recurrence, and TACE refractoriness [7–9]. However, three fundamental challenges persist. First, many published models rely on proprietary, single-institution datasets, limiting reproducibility—a concern increasingly recognized as a barrier to clinical translation [10]. Second, predictive models often lack interpretability, undermining clinical trust. Third, the comparative predictive value of different CT contrast phases remains incompletely characterized.

WAW-TACE数据集[11]——包含233例初治HCC患者的多期CT、专家肿瘤分割、来自104个解剖区域的3,339个预提取PyRadiomics特征以及全面的临床结局——使完全可复现的放射组学研究成为可能。我们利用这一资源：（i）开发具有SHAP解释的可解释机器学习框架用于TACE反应预测；（ii）比较所有四个CT期的预测性能；（iii）评估放射组学衍生风险分层对生存的预后价值。

The WAW-TACE dataset [11]—comprising 233 treatment-naïve HCC patients with multiphasic CT, expert tumor segmentations, 3,339 pre-extracted PyRadiomics features from 104 anatomical regions, and comprehensive clinical outcomes—enables fully reproducible radiomics research. We leveraged this resource to: (i) develop an interpretable machine learning framework with SHAP-based explanation for TACE response prediction; (ii) compare predictive performance across all four CT phases; and (iii) assess the prognostic value of radiomics-derived risk stratification for survival.

---

## 材料与方法 | Materials and Methods

### 研究人群 | Study Population

本回顾性研究使用公开可用的WAW-TACE数据集（Zenodo DOI: 10.5281/zenodo.12741586，CC BY 4.0许可）[11]。该数据集包含2016至2021年间在华沙医科大学接受TACE治疗的233例初治HCC患者。原始作者已获得机构审查委员会批准。纳入标准为：年龄≥18岁，经影像或组织病理学确诊为HCC，首次TACE前三个月内有多期腹部CT，且无既往局部或全身治疗。

This retrospective study used the publicly available WAW-TACE dataset (Zenodo DOI: 10.5281/zenodo.12741586, CC BY 4.0 license) [11]. The dataset comprises 233 treatment-naïve HCC patients treated with TACE at the Medical University of Warsaw (2016–2021). Institutional review board approval was obtained by the original authors. Inclusion criteria were: age ≥ 18 years, HCC diagnosis by imaging or histopathology, multiphasic abdominal CT within three months before first TACE, and no prior locoregional or systemic therapy.

### CT采集与预处理 | CT Acquisition and Preprocessing

多期增强CT在四种扫描仪类型（GE Optima CT600、Siemens Somatom Xceed、Philips Ingenuity Core、Toshiba Aquilion One）上以平扫、动脉晚期、门静脉期和延迟期四个期相采集。数据集提供NIfTI转换的CT体积及预提取放射组学特征。多器官分割掩模使用TotalSegmentator通过nnU-Net生成[12]；肝脏肿瘤掩模由认证放射科医师手工绘制。

Multiphasic contrast-enhanced CT was acquired on four scanner types (GE Optima CT600, Siemens Somatom Xceed, Philips Ingenuity Core, Toshiba Aquilion One) in non-contrast, late arterial, portal venous, and delayed phases. The dataset provides NIfTI-converted CT volumes alongside pre-extracted radiomics features. Multi-organ segmentation masks were generated using TotalSegmentator via nnU-Net [12]; liver tumor masks were hand-drawn by board-certified radiologists.

### 反应定义 | Response Definition

主要终点为首次TACE后随访影像的LI-RADS治疗反应（LR-TR）。LR-TR"无活性"（评分=0）患者被分类为治疗有效组（n = 76），LR-TR"有活性"（评分=2）为治疗无效组（n = 135）。22例可疑反应（LR-TR=1）患者被排除，最终纳入210例进行主要分析。次要终点为总生存期（OS）和无进展生存期（PFS）。

The primary endpoint was LI-RADS Treatment Response (LR-TR) on first post-TACE follow-up imaging. Patients with LR-TR "nonviable" (score = 0) were classified as good responders (n = 76), and those with LR-TR "viable" (score = 2) as poor responders (n = 135). Twenty-two patients with equivocal response (LR-TR = 1) were excluded, yielding 210 patients for primary analysis. Secondary endpoints were overall survival (OS) and progression-free survival (PFS).

### 特征提取与选择 | Feature Extraction and Selection

门静脉期PyRadiomics特征（n = 3,305；v3.0.1，IBSI合规）涵盖104个解剖感兴趣体积及最大肝肿瘤掩模的14个形状特征、18个一阶统计量和五种矩阵（GLCM、GLDM、GLSZM、GLRLM、NGTDM）的75个纹理特征。14个临床变量（年龄、性别、病因、病灶特征、实验室检查值）作为补充预测因子纳入。

Portal venous phase PyRadiomics features (n = 3,305; v3.0.1, IBSI-compliant) encompassed 14 shape features, 18 first-order statistics, and 75 texture features across five matrices (GLCM, GLDM, GLSZM, GLRLM, NGTDM) for each of 104 anatomical volumes of interest plus the largest liver tumor mask. Fourteen clinical variables (age, sex, etiology, lesion characteristics, laboratory values) were included as complementary predictors.

特征选择遵循两阶段流程。第一阶段：使用Mann-Whitney U检验进行单变量筛选（P < 0.05）。第二阶段：L1正则化逻辑回归（LASSO，saga求解器，50个对数间隔C值，五折交叉验证λ选择）。临床变量附加到LASSO选择的放射组学特征之后。

Feature selection followed a two-stage procedure. Stage 1—univariate screening using the Mann-Whitney U test (P < 0.05). Stage 2—L1-regularized logistic regression (LASSO, saga solver, 50 logarithmically spaced C values, five-fold cross-validated λ selection). Clinical variables were appended to the LASSO-selected radiomics features.

### 模型开发与验证 | Model Development and Validation

评估三种分类器：逻辑回归（L2，C=1.0）、随机森林（200棵树，最大深度=8，叶节点最小样本=10）和支持向量机（RBF核，等渗校准）。采用五折分层交叉验证。SMOTE应用于训练折。主要指标为AUC、灵敏度和特异度；通过Brier评分评估校准度。

Three classifiers were evaluated: logistic regression (L2, C = 1.0), random forest (200 trees, max depth = 8, minimum samples per leaf = 10), and support vector machine (RBF kernel, isotonic calibration). Five-fold stratified cross-validation was employed. SMOTE was applied to training folds. Primary metrics were AUC, sensitivity, and specificity; calibration was assessed via Brier score.

### SHAP可解释性 | SHAP-Based Interpretability

SHAP值使用线性解释器及干预性特征扰动计算[13]。全局特征重要性由平均绝对SHAP值导出。患者级解释通过瀑布图生成。使用Spearman秩相关（|r| > 0.35）为治疗有效组和无效组分别构建特征相关网络，采用力导向图布局。

SHAP values were computed using a linear explainer with interventional feature perturbation [13]. Global feature importance was derived from mean absolute SHAP values. Patient-level explanations were generated through waterfall plots. Feature correlation networks (Spearman rank correlation, |r| > 0.35) were constructed for good and poor responders using force-directed graph layout.

### 生存和期相对比分析 | Survival and Phase Analysis

基于放射组学的风险评分按中位数将患者二分为高风险和低风险组。生成OS和PFS的Kaplan-Meier曲线并进行log-rank检验。多变量Cox回归识别独立预后因素。放射组学在所有四个CT期的性能使用相同流程进行比较。

Radiomics-based risk scores dichotomized patients at the median. Kaplan-Meier curves were generated for OS and PFS with log-rank testing. Multivariate Cox regression identified independent prognostic factors. Radiomics performance was compared across all four CT phases using identical procedures.

### 统计分析 | Statistical Analysis

连续变量使用Mann-Whitney U检验比较；分类变量使用卡方检验。全文报告精确P值；P < 0.05为有统计学意义。所有分析在Python 3.13中进行，使用scikit-learn 1.9、SHAP 0.52和lifelines 0.30。所有分析代码公开可用。

Continuous variables were compared using the Mann-Whitney U test; categorical variables using the chi-square test. Exact P-values are reported throughout; P < 0.05 was considered significant. Analyses were performed in Python 3.13 using scikit-learn 1.9, SHAP 0.52, and lifelines 0.30. All analysis code is publicly available.

---

## 结果 | Results

### 患者特征 | Patient Characteristics

最终队列（n = 210）中位年龄66岁（IQR 59–72）；69%为男性。主要病因为HCV（42%）、HBV（19%）和酒精性肝病（12%）。中位病灶直径为28 mm（IQR 18–45）。中位生存期为18个月；73%在随访期间死亡。治疗有效组与无效组在年龄（P = 0.799）、性别（P = 0.852）或病灶数量（P = 0.412）方面无显著差异。

The final cohort (n = 210) had a median age of 66 years (IQR 59–72); 69% were male. Predominant etiologies were HCV (42%), HBV (19%), and alcoholic liver disease (12%). Median lesion diameter was 28 mm (IQR 18–45). Median survival was 18 months; 73% died during follow-up. Good and poor responders did not differ significantly in age (P = 0.799), sex (P = 0.852), or lesion count (P = 0.412).

### 预测性能 | Predictive Performance

随机森林模型达到最高AUC（0.831 ± 0.051），逻辑回归为0.771 ± 0.019，仅临床变量的基线为0.746 ± 0.066（ΔAUC = +0.085）。灵敏度为0.71，特异度为0.78。Brier评分为0.174，表明校准度良好。

The random forest model achieved the highest AUC (0.831 ± 0.051), compared to 0.771 ± 0.019 for logistic regression and 0.746 ± 0.066 for the clinical-only baseline (ΔAUC = +0.085). Sensitivity was 0.71 and specificity was 0.78. The Brier score was 0.174, indicating satisfactory calibration.

### SHAP特征重要性 | SHAP Feature Importance

LASSO选择了涵盖20个解剖区域的42个放射组学特征。SHAP识别的顶级预测因子为：肝肿瘤一阶10百分位数（SHAP = 1.30）、T8椎体均方根（0.96）、肝肿瘤RMS（0.95）、右第8肋骨最大2D直径（0.77）、左第12肋骨峰度（0.72）、酒精性病因（0.68）、左腹直肌轴长（0.66）、左第9肋骨范围（0.61）、右肾稳健平均绝对偏差（0.58）和T8椎体中位数（0.51）。

LASSO selected 42 radiomics features spanning 20 anatomical regions. The top SHAP-identified predictors were: liver tumor first-order 10th percentile (SHAP = 1.30), T8 vertebral body root-mean-square (0.96), liver tumor RMS (0.95), right 8th rib maximum 2D diameter (0.77), left 12th rib kurtosis (0.72), alcoholic etiology (0.68), left rectus abdominis axis length (0.66), left 9th rib range (0.61), right kidney robust mean absolute deviation (0.58), and T8 vertebral body median (0.51).

值得注意的是，前十大特征中有三个来自骨骼结构，表明基于CT的骨骼放射组学捕捉了与HCC预后相关的系统性衰弱信息。

Notably, three of the top ten features derived from skeletal structures, suggesting that CT-based bone radiomics captures systemic frailty relevant to HCC prognosis.

### 期相对比 | Phase Comparison

门静脉期放射组学达到最高性能（AUC = 0.771 ± 0.019），显著优于平扫期（0.612 ± 0.086，P < 0.001）、动脉期（0.551 ± 0.080，P < 0.001）和延迟期（0.543 ± 0.090，P < 0.001）。

Portal venous phase radiomics achieved the highest performance (AUC = 0.771 ± 0.019), markedly outperforming non-contrast (0.612 ± 0.086, P < 0.001), arterial (0.551 ± 0.080, P < 0.001), and delayed phases (0.543 ± 0.090, P < 0.001).

### 生存分析 | Survival Analysis

基于放射组学的风险分层显著区分OS（中位：22 vs. 14个月；log-rank P < 0.0001）和PFS（中位：12 vs. 6个月；P = 0.039）。多变量Cox回归确定白蛋白（HR = 0.59, 95% CI 0.44–0.79, P < 0.001）和肿瘤直径（HR = 1.44, 95% CI 1.18–1.76, P < 0.001）为独立预后因素。

Radiomics-based risk stratification significantly discriminated OS (median: 22 vs. 14 months; log-rank P < 0.0001) and PFS (median: 12 vs. 6 months; P = 0.039). Multivariate Cox regression identified albumin (HR = 0.59, 95% CI 0.44–0.79, P < 0.001) and tumor diameter (HR = 1.44, 95% CI 1.18–1.76, P < 0.001) as independent prognostic factors.

### 相关网络拓扑 | Correlation Network Topology

治疗无效组在|r| > 0.35阈值下表现出更密集的特征连接（23个节点，31条边），而治疗有效组连接更稀疏（23个节点，14条边），提示治疗抵抗与更紧密协调的放射组学表达模式相关。

The poor response group exhibited denser feature connectivity (23 nodes, 31 edges) compared to the good response group (23 nodes, 14 edges) at |r| > 0.35, suggesting that treatment resistance is associated with more tightly coordinated radiomics expression patterns.

---

## 讨论 | Discussion

本研究表明，一个完全基于公共数据的可解释放射组学框架可预测TACE治疗反应（AUC = 0.831）并显著分层HCC患者的生存。以下发现值得详细讨论。

This study demonstrates that an interpretable radiomics framework—deployed exclusively on public data—predicts TACE response (AUC = 0.831) and significantly stratifies survival in HCC. Several findings merit detailed discussion.

**放射组学增强了临床预测。** ΔAUC为+0.085，具有临床意义，尤其是临床模型已经纳入了包括白蛋白和肿瘤大小在内的公认预后变量。放射组学特征来自肝外器官——特别是椎骨和肋骨——显著出现在顶级预测因子中，这一发现将CT机会性筛查的前期工作[14]拓展至放射组学领域。骨骼特征的意外出现提示常规分期CT捕捉了反映系统性衰弱、肌少症和肝硬化相关代谢性骨病的骨纹理改变[15,16]，与CT衍生肌少症在HCC中已知的预后价值一致[17]。

**Radiomics augments clinical prediction.** The ΔAUC of +0.085 over the clinical baseline is clinically meaningful, particularly as the clinical model already incorporated established prognostic variables including albumin and tumor size. The finding that radiomics features from extrahepatic organs—specifically vertebrae and ribs—figure prominently among top predictors extends prior work on opportunistic CT screening [14] into the radiomics domain. The unexpected prominence of skeletal features suggests that routine staging CT captures bone texture alterations reflecting systemic frailty, sarcopenia, and cirrhosis-associated metabolic bone disease [15,16], consistent with the established prognostic value of CT-derived sarcopenia in HCC [17].

**门静脉期是最优选择。** 门静脉期放射组学显著优于所有其他期相，这与HCC主要由门静脉供血以及该期肝脏-肿瘤对比度优越的特点一致[18]。虽然动脉期对HCC诊断至关重要，但我们的发现支持使用门静脉期进行定量放射组学分析。

**Portal venous phase is optimal.** Portal venous phase radiomics significantly outperformed all other phases, consistent with the dominant portal venous blood supply of HCC and superior liver-to-tumor contrast in this phase [18]. While arterial phase imaging is essential for HCC diagnosis, our findings support the use of portal venous phase for quantitative radiomics analysis.

**SHAP实现了临床透明的解释。** SHAP特征排序提供了具有生物学解释力的预测因子。肝肿瘤一阶10百分位数——反映肿瘤最暗10%体素的CT值——是最强预测因子，与肿瘤坏死/低密度和侵袭性生物学之间的已知关联一致。识别特定椎体水平（T8）和个别肋骨（右第8肋、左第9/12肋）作为预测因子提供了可能与未来机制研究相关的解剖学特异性。

**SHAP enables clinically transparent interpretation.** The SHAP-based feature ranking provided biologically interpretable predictors. The liver tumor's first-order 10th percentile—reflecting the darkest 10% of tumor voxels—was the strongest predictor, consistent with the known association between tumor necrosis/hypoattenuation and aggressive biology. Identifying specific vertebral levels (T8) and individual ribs (right 8th, left 9th/12th) as predictors provides anatomical specificity potentially relevant for future mechanistic investigation.

**通过开放科学实现可复现性。** 本研究的一个显著特点是完全依赖公开数据和开源软件。从特征选择到SHAP可视化的每个分析步骤均可完全复现，无需机构数据访问权限。这种设计明确回应了放射组学研究中日益被认识的可复现性问题[10]。

**Reproducibility through open science.** A distinguishing feature of this study is its exclusive reliance on publicly available data and open-source software. Every analytical step—from feature selection through SHAP visualization—is fully reproducible without institutional data access. This design explicitly addresses the reproducibility concerns increasingly recognized in radiomics research [10].

**局限性。** 第一，WAW-TACE数据集代表以HCV为主因的单中心欧洲队列，限制了向以HBV为主的亚洲人群的推广[1]。在临床应用前，需要在独立队列上进行外部验证。第二，数据集仅包含治疗前CT，无法进行TACE后血管重塑的delta放射组学分析。第三，肿瘤掩模过小，无法进行可靠的瘤周血管分析作为传统放射组学的补充。第四，尽管通过IBSI合规特征提取部分解决了扫描仪间差异，但仍可能存在残留批次效应。第五，SHAP解释仍属统计推导，需要独立的生物学验证。

**Limitations.** First, the WAW-TACE dataset represents a single-center European cohort with HCV-predominant etiology, limiting generalizability to HBV-dominant Asian populations [1]. External validation on independent cohorts is required before clinical application. Second, the dataset includes only pre-treatment CT, precluding delta radiomics analysis of post-TACE vascular remodeling. Third, tumor masks were too small for robust peritumoral vessel analysis as an adjunct to conventional radiomics. Fourth, inter-scanner variability—although partially addressed through IBSI-compliant feature extraction—may introduce residual batch effects. Fifth, SHAP explanations remain statistically derived and require independent biological validation.

**未来方向。** 在多民族、HBV为主的队列上进行外部验证是优先事项。前瞻性评估放射组学风险分层是否能指导早期治疗决策值得研究。随着包含全面肿瘤和血管标注的更大公共数据集的出现，瘤周血管形态计量学应该重新审视。

**Future directions.** External validation on multi-ethnic, HBV-dominant cohorts is a priority. Prospective evaluation of whether radiomics-based risk stratification can guide early treatment decisions warrants investigation. As larger public datasets with comprehensive tumor and vessel annotations become available, peritumoral vascular morphometry should be revisited.

---

## 参考文献 | References

[1] Sung H, Ferlay J, Siegel RL, et al. Global cancer statistics 2020. CA Cancer J Clin 2021;71:209–249.

[2] Reig M, Forner A, Rimola J, et al. BCLC strategy for prognosis prediction and treatment recommendation: the 2022 update. J Hepatol 2022;76:681–693.

[3] Llovet JM, Kelley RK, Villanueva A, et al. Hepatocellular carcinoma. Nat Rev Dis Primers 2021;7:6.

[4] Lencioni R, de Baere T, Soulen MC, Rilling WS, Geschwind JFH. Lipiodol TACE for HCC: a systematic review. Hepatology 2016;64:106–116.

[5] Lambin P, Leijenaar RTH, Deist TM, et al. Radiomics: the bridge between medical imaging and personalized medicine. Nat Rev Clin Oncol 2017;14:749–762.

[6] Gillies RJ, Kinahan PE, Hricak H. Radiomics: images are more than pictures, they are data. Radiology 2016;278:563–577.

[7] Wei J, Jiang H, Zeng M, et al. CT radiomics for VETC prediction in HCC. J Hepatocell Carcinoma 2025;12:147–159.

[8] Zhong X, Long H, Chen L, et al. CT radiomics for MVI and RFS in HCC. BMC Med Imaging 2025;25:42.

[9] Wang Q, Li C, Zhang Y, et al. CT radiomics for TACE refractoriness in HCC. Eur Radiol 2024;34:7105–7115.

[10] Zwanenburg A, Vallières M, Abdalah MA, et al. The Image Biomarker Standardization Initiative. Radiology 2020;295:328–338.

[11] Bartnik K, Bartczak T, Krzyziński M, et al. WAW-TACE dataset. Radiol Artif Intell 2024;6:e240296.

[12] Wasserthal J, Breit HC, Meyer MT, et al. TotalSegmentator. Radiol Artif Intell 2023;5:e230024.

[13] Lundberg SM, Lee SI. A unified approach to interpreting model predictions. NeurIPS 2017;30:4765–4774.

[14] Pickhardt PJ, Summers RM, Garrett JW, et al. Opportunistic screening. Radiology 2023;307:e222044.

[15] Carey EJ, Lai JC, Sonnenday C, et al. Sarcopenia in liver transplantation. Hepatology 2019;70:1816–1829.

[16] Tandon P, Montano-Loza AJ, Lai JC, Dasarathy S, Merli M. Sarcopenia and frailty in cirrhosis. J Hepatol 2021;75:S147–S162.

[17] Meza-Junco J, Montano-Loza AJ, Baracos VE, et al. Sarcopenia in HCC. J Clin Gastroenterol 2013;47:861–870.

[18] Choi JY, Lee JM, Sirlin CB. CT and MR imaging of HCC: part I. Radiology 2014;272:635–654.
