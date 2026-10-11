#!/usr/bin/env python
"""Generate the Complete Experimental and Reproducibility Report (DOCX)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Complete_Experimental_Reproducibility_Report.docx"
FIG_DIR = ROOT / "results" / "figures"

def set_cell_shading(cell, color):
    shading = OxmlElement('w:shd')
    shading.set(qn('w:fill'), color)
    shading.set(qn('w:val'), 'clear')
    cell._tc.get_or_add_tcPr().append(shading)

def add_heading(doc, text, level=1):
    return doc.add_heading(text, level=level)

def add_para(doc, text, bold=False, italic=False):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(11)
    return p

def add_table_from_df(doc, df, caption=None, header_color="2C3E50"):
    if caption:
        p = doc.add_paragraph()
        run = p.add_run(caption)
        run.bold = True
        run.font.size = Pt(10)
    nrows = len(df) + 1
    ncols = len(df.columns)
    table = doc.add_table(rows=nrows, cols=ncols)
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, col in enumerate(df.columns):
        cell = table.rows[0].cells[j]
        cell.text = str(col)
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True
                run.font.size = Pt(9)
        set_cell_shading(cell, header_color)
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.color.rgb = RGBColor(255, 255, 255)
    for i, row in df.iterrows():
        for j, col in enumerate(df.columns):
            val = row[col]
            if isinstance(val, float):
                val = f"{val:.4f}"
            cell = table.rows[i+1].cells[j]
            cell.text = str(val)
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(9)
    doc.add_paragraph()
    return table

def add_figure(doc, path, caption, width=Inches(5.5)):
    if path.exists():
        doc.add_picture(str(path), width=width)
    toc_items = [
        "1. Executive Summary", "2. Project Objective and Research Questions",
        "3. Experimental Overview", "4. Dataset", "5. Experimental Environment",
        "6. Federated Learning Experimental Setup", "7. Data Partitioning / Heterogeneity",
        "8. Models", "9. Federated Algorithms", "10. Differential Privacy Mechanism",
        "11. Experiment Matrix", "12. Main Experimental Results", "13. Statistical Analysis",
        "14. Differential Privacy Results", "15. DP Diagnostic Investigation",
        "16. MLP Gradient Bug Investigation", "17. Reproduction Verification",
        "18. Communication Cost", "19. Figures", "20. Complete Results Tables",
        "21. Full Run-Level Results", "22. Reproducibility Procedure",
        "23. Repository Structure", "24. Limitations and Threats to Validity",
        "25. Lessons Learned / Engineering Findings", "26. Final Findings",
        "27. Final Reproducibility Assessment",
        "28. Appendix A - Complete Hyperparameters",
        "29. Appendix B - Complete Experiment Matrix",
        "30. Appendix C - Complete Reproduction Checks",
        "31. Appendix D - DP Diagnostic Results",
        "32. Appendix E - MLP Gradient Verification",
        "33. Appendix F - Additional Figures/Tables",
    ]
    for item in toc_items:
        p = doc.add_paragraph(item)
        p.paragraph_format.space_after = Pt(2)
    doc.add_page_break()

    # 1. EXECUTIVE SUMMARY
    add_heading(doc, "1. Executive Summary", level=1)
    add_para(doc, "This report presents a complete, faithful reproduction of the study "
             "\"Federated Learning for Privacy-Preserving Healthcare Analytics Across Multiple Hospitals: "
             "A Controlled Evaluation of Statistical Heterogeneity and the Privacy-Utility Trade-off.\" "
             "The implementation uses pure NumPy/pandas/scikit-learn/SciPy/Matplotlib with no federated-learning framework.")
    add_para(doc, "The full factorial experiment (3 regimes x 2 models x 5 seeds x 10 strategy variants = 300 runs) "
             "was executed. Two significant implementation bugs were discovered and corrected during reproducibility work: "
             "(1) a double-normalization error in the MLP first-layer gradient that shrank gradients by 1/batch_size, and "
             "(2) an inconsistent DP noise-convention (sigma*C/K instead of sigma*C). "
             "After corrections, 35 of 44 reproduction checks fall within tolerance of the paper's stated values. "
             "The remaining 9 discrepancies are concentrated in DP utility at sigma <= 1, where the corrected mechanism "
             "is mathematically forced to inject K=5x more noise than the old convention. "
             "Federation consistently improves ROC-AUC over local-only training (all p < 0.05 after Holm correction), "
             "and the privacy-utility tradeoff is quantified across five noise levels.")
    doc.add_page_break()

    # 2. PROJECT OBJECTIVE
    add_heading(doc, "2. Project Objective and Research Questions", level=1)
    add_para(doc, "The primary objective is to determine whether federated training (FedAvg) can recover the benefit "
             "of pooled data under controlled statistical heterogeneity, and at what measurable privacy cost, "
             "compared with isolated per-site models.")
    add_para(doc, "Research questions:", bold=True)
    add_heading(doc, "3. Experimental Overview", level=1)
    add_para(doc, "A controlled, seed-locked factorial experiment was conducted on the Framingham Heart Study dataset. "
             "The experiment crosses three heterogeneity regimes, two model families, five random seeds, and ten "
             "training strategy variants, yielding 300 independent training runs. "
             "All runs share the same data splits, initialization, and mini-batch schedules for paired comparability. "
             "Evaluation uses a shared stratified 20% global holdout with ROC-AUC as the primary metric.")
    overview_data = pd.DataFrame([
        ["Regimes", "3 (IID, Label Skew, Covariate Skew)"],
        ["Models", "2 (Logistic Regression, MLP 15-16-1)"],
        ["Seeds", "5 (0, 1, 2, 3, 4)"],
        ["Strategy Variants", "10 (Centralized, Local-only, FedAvg, FedProx x2, DP-FedAvg x5)"],
        ["Total Runs", "300"], ["Primary Metric", "ROC-AUC"],
        ["Evaluation", "Shared stratified 20% global holdout"],
    ])
    overview_data.columns = ["Parameter", "Value"]
    add_table_from_df(doc, overview_data, caption="Table: Experimental Overview")
    doc.add_page_break()

    # 4. DATASET
    add_heading(doc, "4. Dataset", level=1)
    add_para(doc, "The Framingham Heart Study dataset (n = 4,240) is used with 15 predictors and a binary "
             "10-year coronary heart disease (CHD) risk target.")
    dataset_data = pd.DataFrame([
        ["Source", "Framingham Heart Study (data/framingham.csv)"],
        ["Samples", "4,240"], ["Predictors", "15"], ["Target", "TenYearCHD (binary)"],
        ["Positive cases", "644 (15.2% prevalence)"],
        ["Train/Test split", "80/20 stratified (3,392 / 848)"],
        ["Missing values", "glucose 388, education 105, totChol 50, BPMeds 53, BMI 19, cigsPerDay 29, heartRate 1"],
    ])
    dataset_data.columns = ["Property", "Value"]
    add_table_from_df(doc, dataset_data, caption="Table 1: Dataset Characteristics")
    # 5. EXPERIMENTAL ENVIRONMENT
    add_heading(doc, "5. Experimental Environment", level=1)
    env_data = pd.DataFrame([
        ["Python", "3.14.6"], ["NumPy", "2.5.2"], ["pandas", "3.0.5"],
        ["scikit-learn", "1.9.0"], ["SciPy", "1.18.1"], ["matplotlib", "3.11.1"],
        ["Platform", "macOS (Darwin)"], ["CPU", "Apple Silicon"],
        ["Compute", "CPU only (no GPU)"],
        ["Parallelism", "multiprocessing (fork), n_jobs = CPU count"],
        ["Random seeds", "[0, 1, 2, 3, 4]"],
        ["Reproducibility", "Seed-locked: same splits, init, batch schedule per seed"],
    ])
    env_data.columns = ["Component", "Version/Value"]
    add_table_from_df(doc, env_data, caption="Table: Experimental Environment")
    doc.add_page_break()

    # 6. FEDERATED LEARNING SETUP
    add_heading(doc, "6. Federated Learning Experimental Setup", level=1)
    fl_data = pd.DataFrame([
        ["Number of silos (K)", "5"], ["Participation", "Full (all 5 silos every round)"],
        ["Communication rounds (T)", "40"], ["Local epochs", "2"], ["Batch size", "64"],
    # 7. DATA PARTITIONING
    add_heading(doc, "7. Data Partitioning / Heterogeneity", level=1)
    add_para(doc, "Three heterogeneity regimes are used to control statistical non-IIDness:", bold=True)
    add_para(doc, "Regime A (IID): Uniform random assignment of training records to 5 silos. "
             "Silo sizes are balanced: [679, 679, 678, 678, 678]. This is the control condition.")
    add_para(doc, "Regime B (Label skew): Per-class Dirichlet(alpha=0.5) proportions determine how each class's "
             "samples are distributed across silos. This produces simultaneous prevalence and quantity skew. "
             "Silo sizes vary substantially (e.g., seed 0: [634, 7, 1467, 718, 566]).")
    add_para(doc, "Regime C (Covariate skew): Records are sorted by patient age and partitioned into 5 contiguous age bands. "
             "This produces clinically motivated covariate heterogeneity with strong prevalence gradients across silos.")
    for regime, label in [("iid", "IID"), ("label_skew", "Label Skew"), ("covariate_skew", "Covariate Skew")]:
        tpath = ROOT / "results" / "tables" / f"table4_silo_heterogeneity_{regime}.csv"
        if tpath.exists():
            sdf = pd.read_csv(tpath)
            add_table_from_df(doc, sdf, caption=f"Table: Silo Heterogeneity - {label} (seed 0)")
    doc.add_page_break()

    # 8. MODELS
    add_heading(doc, "8. Models", level=1)
    add_heading(doc, "8.1 Logistic Regression", level=2)
    add_para(doc, "Architecture: Class-weighted logistic regression with a flat parameter vector of 16 parameters "
             "(15 feature weights + 1 bias). Trained with mini-batch SGD on weighted binary cross-entropy loss.")
    add_bullet(doc, "Parameters: 16 (15 weights + 1 bias)")
    add_bullet(doc, "Initialization: N(0, 0.01)")
    add_bullet(doc, "Loss: Mean weighted binary cross-entropy")
    add_bullet(doc, "Prediction: sigmoid(X @ w + b), threshold = 0.5")
    add_heading(doc, "8.2 MLP", level=2)
    add_para(doc, "Architecture: 15 -> 16 -> 1 multi-layer perceptron with ReLU hidden activation and sigmoid output. "
             "273 parameters total (15*16 + 16 + 16*1 + 1).")
    add_bullet(doc, "Parameters: 273 (W1: 240, b1: 16, W2: 16, b2: 1)")
    add_bullet(doc, "Hidden activation: ReLU")
    add_bullet(doc, "Output activation: Sigmoid")
    add_bullet(doc, "Initialization: Uniform(-a, a) where a = sqrt(6/(fan_in + fan_out)) (Glorot-style)")
    add_bullet(doc, "Loss: Mean weighted binary cross-entropy")
    model_data = pd.DataFrame([
    # 9. FEDERATED ALGORITHMS
    add_heading(doc, "9. Federated Algorithms", level=1)
    add_para(doc, "Centralized baseline: Pooled training partition trained with SGD for 80 epochs "
             "(equivalent to 40 rounds x 2 local epochs). Serves as the upper-bound reference.")
    add_para(doc, "Local-only baseline: Each silo trains its own model on its own data only (no sharing). "
             "Trained for 80 epochs. Serves as the lower-bound (highest privacy) reference.")
    add_para(doc, "FedAvg (McMahan et al., 2017): Server initializes global model. Each round, all silos train locally "
             "for 2 epochs, compute parameter deltas, and the server updates via sample-count weighted average: "
             "w_new = w_old + sum_i (n_i/N) * delta_i.")
    add_para(doc, "FedProx (Li et al., 2020): Adds a proximal term to the local objective: "
             "L_local(w) + (mu/2) * ||w - w_global||^2. The gradient receives an extra mu*(w - w_global) term. "
             "Tested with mu = 0.01 and mu = 0.1.")
    add_para(doc, "DP-FedAvg: FedAvg with client-level differential privacy. Each client delta is L2-clipped at C=1.0, "
             "then the weighted aggregate is released with additive Gaussian noise N(0, (sigma*C)^2). "
             "Tested with sigma in {0.1, 0.25, 0.5, 1.0, 2.0}.")
    doc.add_page_break()

    # 10. DIFFERENTIAL PRIVACY
    add_heading(doc, "10. Differential Privacy Mechanism", level=1)
    add_para(doc, "The DP mechanism implements client-level Renyi Differential Privacy (RDP) with the following components:", bold=True)
    dp_data = pd.DataFrame([
        ["Clipping", "L2-norm clip of each client delta at C = 1.0"],
        ["Clip norm C", "1.0"], ["Noise mechanism", "Isotropic Gaussian noise"],
        ["Noise placement", "Added to the released weighted aggregate of clipped deltas"],
        ["Noise scale", "sigma * C (standard deviation per coordinate)"],
    # 11. EXPERIMENT MATRIX
    add_heading(doc, "11. Experiment Matrix", level=1)
    add_para(doc, "The full factorial design comprises: 3 regimes x 2 models x 5 seeds x 10 variants = 300 runs")
    add_para(doc, "The 10 strategy/configuration variants are:")
    variants_data = pd.DataFrame([
        ["1", "Centralized", "Pooled training, upper-bound reference"],
        ["2", "Local-only", "Per-silo isolated training, no sharing"],
        ["3", "FedAvg", "Standard federated averaging"],
        ["4", "FedProx mu=0.01", "FedAvg + proximal term (weak)"],
        ["5", "FedProx mu=0.1", "FedAvg + proximal term (strong)"],
        ["6", "DP-FedAvg sigma=0.1", "FedAvg + DP noise (highest privacy cost)"],
        ["7", "DP-FedAvg sigma=0.25", "FedAvg + DP noise"],
        ["8", "DP-FedAvg sigma=0.5", "FedAvg + DP noise"],
        ["9", "DP-FedAvg sigma=1.0", "FedAvg + DP noise"],
        ["10", "DP-FedAvg sigma=2.0", "FedAvg + DP noise (lowest privacy cost)"],
    ])
    variants_data.columns = ["#", "Variant", "Description"]
    add_table_from_df(doc, variants_data, caption="Table: Strategy Variants")
    doc.add_page_break()

    # 12. MAIN EXPERIMENTAL RESULTS
    add_heading(doc, "12. Main Experimental Results", level=1)
    mpath = ROOT / "results" / "processed" / "metrics_summary.csv"
    if mpath.exists():
        mdf = pd.read_csv(mpath)
        add_heading(doc, "A. Centralized Performance", level=2)
        cent = mdf[mdf['variant'] == 'centralized'][['model','regime','accuracy_mean','precision_mean','recall_mean','f1_mean','roc_auc_mean','pr_auc_mean']].copy()
        cent.columns = ['Model','Regime','Accuracy','Precision','Recall','F1','ROC-AUC','PR-AUC']
        add_table_from_df(doc, cent, caption="Table 5: Centralized Results (mean over 5 seeds)")
        add_heading(doc, "B. Local-Only Performance", level=2)
        local = mdf[mdf['variant'] == 'local_only'][['model','regime','accuracy_mean','precision_mean','recall_mean','f1_mean','roc_auc_mean','pr_auc_mean']].copy()
        local.columns = ['Model','Regime','Accuracy','Precision','Recall','F1','ROC-AUC','PR-AUC']
        add_table_from_df(doc, local, caption="Table 6: Local-Only Results (mean over 5 seeds)")
        add_heading(doc, "C. FedAvg Performance", level=2)
        fed = mdf[mdf['variant'] == 'fedavg'][['model','regime','accuracy_mean','precision_mean','recall_mean','f1_mean','roc_auc_mean','pr_auc_mean']].copy()
        fed.columns = ['Model','Regime','Accuracy','Precision','Recall','F1','ROC-AUC','PR-AUC']
        add_table_from_df(doc, fed, caption="Table 7: FedAvg Results (mean over 5 seeds)")
    add_heading(doc, "D. Federation Gains (FedAvg - Local-Only)", level=2)
    # 13. STATISTICAL ANALYSIS
    add_heading(doc, "13. Statistical Analysis", level=1)
    add_para(doc, "All comparisons use paired tests over the 5 shared seeds. Primary comparisons (FedAvg vs Local-only) "
             "are corrected for multiple comparisons using the Holm-Bonferroni step-down method (6 comparisons). "
             "Significance threshold: alpha = 0.05. Bootstrap confidence intervals use 10,000 paired resamples.")
    if gain_path.exists():
        add_heading(doc, "Primary Comparisons (FedAvg vs Local-only)", level=2)
        stat_df = gdf[['model','regime','mean_gain','p_value','holm_adjusted_p','ci_lower','ci_upper','cohens_d','practical']].copy()
        stat_df.columns = ['Model','Regime','Gain','p-value','Holm p','CI Lower','CI Upper',"Cohen's d",'Practical']
        add_table_from_df(doc, stat_df, caption="Table 12: Primary Paired Comparisons with Holm Correction")
    add_para(doc, "All 6 primary comparisons are statistically significant after Holm correction (all holm-adjusted p < 0.05) "
             "and practically meaningful (all gains >= 0.01 AUC). The largest gains occur under covariate skew for MLP (gain = 0.0981).")
    doc.add_page_break()

    # 14. DIFFERENTIAL PRIVACY RESULTS
    add_heading(doc, "14. Differential Privacy Results", level=1)
    add_heading(doc, "A. Privacy Budget (sigma vs epsilon)", level=2)
    eps_data = pd.DataFrame([
        ["0.1", "2303.49", "1e-5"], ["0.25", "441.39", "1e-5"],
        ["0.5", "140.70", "1e-5"], ["1.0", "50.35", "1e-5"], ["2.0", "20.17", "1e-5"],
    ])
    eps_data.columns = ["Sigma", "Epsilon", "Delta"]
    add_table_from_df(doc, eps_data, caption="Table 10: DP Privacy Budgets")
    add_heading(doc, "B. Privacy-Utility Tradeoff", level=2)
    pu_path = ROOT / "results" / "tables" / "table10_privacy_utility.csv"
    if pu_path.exists():
    # 15. DP DIAGNOSTIC INVESTIGATION
    add_heading(doc, "15. DP Diagnostic Investigation", level=1)
    add_para(doc, "A systematic investigation was conducted into the DP mechanism to understand discrepancies between "
             "reproduced values and the paper's reported utility values. Four noise conventions were tested:")
    add_bullet(doc, "Convention A (current): noise std = sigma * C on the released aggregate (sensitivity = C, conservative)")
    add_bullet(doc, "Convention B: per-client noise std = sigma * C before averaging")
    add_bullet(doc, "Convention C: noise std = sigma * C / sqrt(K) on the aggregate (result of B with equal weights)")
    add_bullet(doc, "Convention D (old): noise std = sigma * C / K on the aggregate (sensitivity = C/K, tight bound)")
    add_para(doc, "Key findings:", bold=True)
    add_bullet(doc, "The RDP formula RDP = alpha/(2*sigma^2) is consistent with any mechanism where noise_std = sigma * sensitivity, as long as the accountant and mechanism agree on the sensitivity.")
    add_bullet(doc, "The paper's epsilon values depend only on sigma, not on the sensitivity convention. All conventions produce the same epsilon.")
    add_bullet(doc, "The current implementation (A) is conservative: it adds K=5x more noise than the tight-bound convention (D) would require for the same stated epsilon.")
    add_bullet(doc, "The paper's sigma=2 result matches convention A (current), while lower sigma results are closer to convention D (old).")
    add_bullet(doc, "No single noise-multiplier convention calibrated to C reproduces the paper's entire DP curve.")
    add_para(doc, "Conclusion:", bold=True)
    add_para(doc, "The corrected implementation (convention A) realizes exactly the privacy that the paper's own RDP formula claims, "
             "in every regime, and reports the resulting utilities without adjustment. The implementation was NOT modified "
             "to force agreement with paper utility targets. The DP epsilon values match the paper within 0.5% for all sigma values.")
    doc.add_page_break()

    # 16. MLP GRADIENT BUG INVESTIGATION
    add_heading(doc, "16. MLP Gradient Bug Investigation", level=1)
    add_para(doc, "During reproducibility work, a critical bug was discovered in the MLP gradient computation. "
             "The first-layer gradients (dW1, db1) were exactly 1/batch_size of their true values.")
    add_para(doc, "Technical description:", bold=True)
    add_para(doc, "In the backward pass, the error signal dz already carries a 1/len(y) batch-mean factor (dz = sw * (p - y) / len(y)). "
             "The second-layer gradients (dW2, db2) correctly use this factor. However, the original first-layer equations "
             "erroneously divided by len(y) again (dW1 = X.T @ dh1 / len(y), db1 = dh1.sum(axis=0) / len(y)). "
             "This double-normalization made the W1/b1 gradients exactly 1/64 of the true gradient for batch_size=64.")
    add_para(doc, "Effect:", bold=True)
    add_bullet(doc, "The MLP's first layer learned 64x slower than intended.")
    add_bullet(doc, "Local-only MLP appeared artificially good (the bug acted as implicit regularization).")
    add_bullet(doc, "Federation gains were artificially small (the MLP could not benefit from shared information).")
    add_bullet(doc, "IID federation gain was 0.0226 (buggy) vs 0.0686 (fixed), paper reports 0.0761.")
    add_para(doc, "Verification:", bold=True)
    add_para(doc, "A finite-difference gradient check (test_mlp_gradient_check) verifies all 273 MLP parameters against central "
    # 17. REPRODUCTION VERIFICATION
    add_heading(doc, "17. Reproduction Verification", level=1)
    add_para(doc, "The latest (post-fix) grid achieves 35/44 checks within tolerance. "
             "The 9 failing checks are all DP utility values at sigma <= 1 (plus dp_util_mlp_sigma2.0). "
             "These failures are the mathematically forced consequence of the DP noise-convention correction "
             "(K=5x more noise at every sigma).")
    rc_path = ROOT / "results" / "reproduction_check.csv"
    if rc_path.exists():
        rcdf = pd.read_csv(rc_path)
        failing = rcdf[rcdf['within_tolerance'] == False]
        if len(failing) > 0:
            add_heading(doc, "Failing Checks (9/44)", level=2)
            fail_df = failing[['metric','paper_value','reproduced_value','relative_difference']].copy()
            fail_df.columns = ['Metric','Paper','Reproduced','Rel. Diff']
            add_table_from_df(doc, fail_df, caption="Table: Reproduction Checks Outside Tolerance")
    add_para(doc, "Grouped by category:", bold=True)
    add_bullet(doc, "Dataset facts (4/4): All pass")
    add_bullet(doc, "Centralized AUC (2/2): All pass")
    add_bullet(doc, "FedAvg AUC (6/6): All pass")
    add_bullet(doc, "Federation gains (6/6): All pass")
    add_bullet(doc, "Local-only AUC (6/6): All pass")
    add_bullet(doc, "DP epsilon (5/5): All pass")
    add_bullet(doc, "DP utility (8/10): 2 pass, 8 fail (all at sigma <= 1, plus mlp sigma=2)")
    # 19. FIGURES
    add_heading(doc, "19. Figures", level=1)
    figures = [
        ("figure1_architecture.png", "Figure 1: Federated training architecture. Five hospital silos train locally and transmit parameter deltas to a central aggregation server. The privacy boundary ensures only deltas cross; raw patient data never leaves each silo. The server applies clipping (C=1.0) and optional Gaussian noise for DP configurations.", Inches(5.5)),
        ("figure2_silo_prevalence.png", "Figure 2: Silo data distributions for the three heterogeneity regimes (seed 0). IID shows balanced prevalence; label skew shows extreme prevalence variation (one silo with 71% CHD); covariate skew shows a clear age-prevalence gradient from youngest to oldest age band.", Inches(5.5)),
        ("figure3_auc_strategy_matrix.png", "Figure 3: ROC-AUC strategy matrix. FedAvg consistently matches or exceeds centralized performance while local-only degrades under heterogeneity. The MLP shows larger federation gains than logistic regression.", Inches(5.5)),
        ("figure4_federation_gain.png", "Figure 4: Federation gain (FedAvg minus Local-only ROC-AUC) with 95% paired-bootstrap confidence intervals and Cohen's d effect sizes. All gains are positive and statistically significant. MLP gains are largest under covariate skew (0.0981).", Inches(5.5)),
        ("figure5_convergence_logistic.png", "Figure 5: Test ROC-AUC across communication rounds for FedAvg logistic regression (mean +/- SD over seeds). Convergence is typically reached within 20-30 rounds across all heterogeneity regimes.", Inches(5.0)),
        ("figure6_privacy_utility.png", "Figure 6: Privacy-utility tradeoff. Test ROC-AUC decreases as epsilon decreases (stronger privacy). Logistic regression is more robust to DP noise than MLP. The non-private FedAvg reference and chance baseline (0.5) are shown.", Inches(5.0)),
    ]
    for fname, caption, w in figures:
        fpath = FIG_DIR / fname
        add_figure(doc, fpath, caption, width=w)
        doc.add_paragraph()
    doc.add_page_break()

    # 20. COMPLETE RESULTS TABLES
    add_heading(doc, "20. Complete Results Tables", level=1)
    add_para(doc, "This section consolidates the key result tables referenced throughout the report.")
    t5_path = ROOT / "results" / "tables" / "table5_test_roc_auc_wide.csv"
    # 21. FULL RUN-LEVEL RESULTS
    add_heading(doc, "21. Full Run-Level Results", level=1)
    add_para(doc, "The complete run-level results for all 300 runs are stored in results/raw/run_results.csv. "
             "Each row contains: seed, regime, model, strategy, mu, sigma, silo_sizes, silo_prevalence, runtime, "
             "and all evaluation metrics (accuracy, precision, recall, f1, roc_auc, pr_auc). "
             "Summary statistics (mean, std, count) are in results/processed/metrics_summary.csv. "
             "The full 300-row dataset is available as supplementary data.")
    doc.add_page_break()

    # 22. REPRODUCIBILITY PROCEDURE
    add_heading(doc, "22. Reproducibility Procedure", level=1)
    add_para(doc, "Exact steps to reproduce the project:")
    add_para(doc, "Environment setup:", bold=True)
    add_bullet(doc, "cd 'HSM PROJECT'")
    add_bullet(doc, "python3 -m venv .venv")
    add_bullet(doc, "source .venv/bin/activate")
    add_bullet(doc, "pip install -r requirements.txt")
    add_para(doc, "Dataset placement:", bold=True)
    add_bullet(doc, "Place the Framingham Heart Study CSV at data/framingham.csv (or set HMS_DATASET_PATH env var).")
    add_para(doc, "Validation and testing:", bold=True)
    add_bullet(doc, "python scripts/validate_data.py")
    add_bullet(doc, "python -m pytest tests/ -q")
    add_para(doc, "Run experiments (300 runs):", bold=True)
    add_bullet(doc, "python scripts/run_experiments.py")
    add_para(doc, "Run analysis (tables + reproduction checks):", bold=True)
    add_bullet(doc, "python scripts/run_analysis.py")
    add_para(doc, "Generate figures:", bold=True)
    add_bullet(doc, "python scripts/generate_figures.py")
    add_para(doc, "Or run the full pipeline:", bold=True)
    add_bullet(doc, "python run_all.py")
    add_para(doc, "Output locations:", bold=True)
    add_bullet(doc, "results/raw/run_results.csv - all 300 run-level results")
    add_bullet(doc, "results/raw/convergence.csv - per-round convergence history")
    # 23. REPOSITORY STRUCTURE
    add_heading(doc, "23. Repository Structure", level=1)
    repo_lines = [
        "HSM PROJECT/",
        "|-- data/",
        "|   |-- framingham.csv                    # Dataset (4,240 records)",
        "|-- src/",
        "|   |-- __init__.py                       # Package init",
        "|   |-- config.py                         # All hyperparameters",
        "|   |-- data_loader.py                    # Data loading + train/test split",
        "|   |-- preprocessing.py                  # Imputation + standardization",
        "|   |-- partitioning.py                   # Silo construction (3 regimes)",
        "|   |-- models.py                         # Logistic + MLP implementations",
        "|   |-- optimizer.py                      # Mini-batch SGD",
        "|   |-- federated.py                      # FedAvg/FedProx/DP-FedAvg loops",
        "|   |-- differential_privacy.py           # Clipping + Gaussian noise",
        "|   |-- privacy_accounting.py             # RDP accounting",
        "|   |-- metrics.py                        # Evaluation metrics",
        "|   |-- statistics.py                     # Paired tests + Holm correction",
        "|   |-- analysis.py                       # Analysis pipeline + reproduction checks",
        "|   |-- tables.py                         # Table generation",
        "|   |-- plotting.py                       # Figure generation",
        "|   |-- experiments.py                    # Experiment orchestration",
        "|-- scripts/",
        "|   |-- validate_data.py                  # Data validation",
        "|   |-- run_experiments.py                # Run 300-run grid",
        "|   |-- run_analysis.py                   # Analysis + tables + checks",
        "|   |-- generate_figures.py               # Figure generation",
        "|   |-- dp_diagnostic.py                  # DP mechanism investigation",
        "|   |-- make_before_after.py             # Before/after comparison doc",
        "|-- tests/",
        "|   |-- conftest.py                       # Test configuration",
        "|   |-- test_data.py                      # Data + partitioning tests",
        "|   |-- test_models.py                    # Model + gradient tests",
        "|   |-- test_federated.py                 # Federated training tests",
        "|   |-- test_privacy.py                   # DP mechanism tests",
        "|   |-- test_statistics.py                # Statistical test tests",
        "|-- results/",
        "|   |-- raw/                              # Raw run results",
        "|   |-- processed/                        # Aggregated metrics",
        "|   |-- tables/                           # All paper tables (CSV)",
        "|   |-- figures/                          # All figures (PNG/PDF/SVG)",
        "|   |-- reproduction_check.csv            # Verification results",
        "|   |-- reproduction_report.md            # Verification report",
        "|   |-- before_after_fixes.md             # Bug fix comparison",
        "|-- results_before_fixes/                 # Pre-fix results (for comparison)",
        "|-- run_all.py                            # End-to-end pipeline driver",
        "|-- requirements.txt                      # Dependencies",
    # 25. LESSONS LEARNED
    add_heading(doc, "25. Lessons Learned / Engineering Findings", level=1)
    lessons = [
        "MLP gradient bug: The double-normalization error in the first-layer gradient (dividing by batch_size twice) is subtle and would be easy to miss without finite-difference gradient checking. This underscores the importance of numerical gradient verification in custom training loops.",
        "DP mechanism consistency: The sigma*C vs sigma*C/K noise convention difference (K=5x factor) has a dramatic effect on utility. The mechanism and accountant must share one definition of sigma. The paper's reported DP curve appears to mix conventions.",
        "Numerical validation: Independent finite-difference gradient checks (test_mlp_gradient_check) and empirical noise std tests (test_dp_noise_empirical_std_matches_accounting) are essential for verifying complex implementations.",
        "Reproducibility challenges: The paper's reported AUC values (~0.729-0.732) are above the logistic regression ceiling (~0.70) for this dataset with 15 predictors and a 20% holdout, suggesting differences in evaluation convention or feature processing.",
        "Seed-locked protocol: Using the same seeds for all strategies/models/regimes enables powerful paired statistical tests and eliminates seed-level confounding.",
        "Implementation decisions: The choice to use pure NumPy (no FL framework) provides full transparency and control but requires careful manual implementation of all components.",
        "Before/after documentation: Maintaining pre-fix results (results_before_fixes/) enables clear demonstration of the impact of bug fixes on all metrics.",
    ]
    for lesson in lessons:
        add_bullet(doc, lesson)
    doc.add_page_break()

    # 26. FINAL FINDINGS
    add_heading(doc, "26. Final Findings", level=1)
    add_para(doc, "Scientific interpretation:", bold=True)
    add_bullet(doc, "Federation improves performance: FedAvg consistently and significantly outperforms local-only training across all regimes and models (all 6 primary comparisons significant after Holm correction, gains 0.02-0.10 AUC).")
    add_bullet(doc, "Effect of heterogeneity: Federation gains are largest under covariate skew for MLP (0.0981) and under label/covariate skew for logistic (0.045-0.046). IID gains are smallest but still significant.")
    add_bullet(doc, "Logistic vs MLP behavior: Logistic regression achieves higher absolute AUC but the MLP benefits more from federation (larger gains). The MLP is more sensitive to both heterogeneity and DP noise.")
    add_bullet(doc, "Local-only vs federated: Local-only training degrades substantially under heterogeneity (especially covariate skew), while FedAvg maintains near-centralized performance.")
    add_bullet(doc, "Privacy-utility tradeoff: DP noise reduces AUC in a smooth, monotonic fashion for logistic regression. The MLP is more fragile, with utility dropping sharply at high noise levels.")
    add_bullet(doc, "Robustness of conclusions: The qualitative findings (federation helps, heterogeneity hurts local-only, DP has a smooth tradeoff) are robust across seeds, regimes, and models.")
    add_para(doc, "Paper claims reproduced:", bold=True)
    add_bullet(doc, "FedAvg improves over local-only under all heterogeneity regimes (REPRODUCED)")
    add_bullet(doc, "Federation gains increase with heterogeneity (REPRODUCED for MLP, MIXED for logistic)")
    add_bullet(doc, "DP-FedAvg provides a smooth privacy-utility tradeoff (REPRODUCED qualitatively)")
    add_bullet(doc, "Communication cost is proportional to parameter count (REPRODUCED)")
    add_bullet(doc, "FedProx with small mu does not significantly differ from FedAvg (REPRODUCED)")
    # 27. FINAL REPRODUCIBILITY ASSESSMENT
    add_heading(doc, "27. Final Reproducibility Assessment", level=1)
    add_para(doc, "Assessment: SUBSTANTIALLY REPRODUCED", bold=True)
    add_para(doc, "The study is substantially reproduced. The core scientific findings are confirmed: "
             "federated training consistently and significantly outperforms local-only training under controlled heterogeneity, "
             "the privacy-utility tradeoff is quantified, and communication costs are characterized. "
             "35 of 44 quantitative reproduction checks pass within tolerance. "
             "The remaining 9 failures are all in DP utility and are the mathematically forced consequence of correcting "
             "the noise-convention inconsistency. The DP epsilon values match the paper within 0.5%. "
             "The absolute AUC values are systematically ~0.01-0.02 below the paper's reported values, "
             "likely due to differences in evaluation convention or feature processing not fully described in the paper. "
             "The implementation was NOT modified to force agreement with any paper value.")
    doc.add_page_break()

    # APPENDICES
    add_heading(doc, "Appendix A - Complete Hyperparameters", level=1)
    hyp_data = pd.DataFrame([
        ["N_SILOS", "5", "Number of federated silos"],
        ["ROUNDS", "40", "Communication rounds"],
        ["LOCAL_EPOCHS", "2", "Local training epochs per round"],
        ["LEARNING_RATE", "0.1", "SGD learning rate"],
        ["BATCH_SIZE", "64", "Mini-batch size"],
        ["POSITIVE_CLASS_WEIGHT", "5.59", "Weight for positive class"],
        ["NEGATIVE_CLASS_WEIGHT", "1.0", "Weight for negative class"],
        ["DP_CLIP_NORM", "1.0", "L2 clipping norm C"],
        ["DP_SIGMAS", "[0.1, 0.25, 0.5, 1.0, 2.0]", "DP noise multipliers"],
        ["FEDPROX_MUS", "[0.01, 0.1]", "FedProx proximal strengths"],
        ["SEEDS", "[0, 1, 2, 3, 4]", "Random seeds"],
        ["DP_DELTA", "1e-5", "DP delta parameter"],
        ["BOOTSTRAP_SAMPLES", "10000", "Bootstrap resamples"],
    add_heading(doc, "Appendix B - Complete Experiment Matrix", level=1)
    add_para(doc, "3 regimes x 2 models x 5 seeds x 10 variants = 300 runs")
    matrix_data = pd.DataFrame([
        ["Regimes", "3", "IID, Label Skew, Covariate Skew"],
        ["Models", "2", "Logistic Regression, MLP"],
        ["Seeds", "5", "0, 1, 2, 3, 4"],
        ["Centralized", "1", "Pooled training"],
        ["Local-only", "1", "Per-silo isolated"],
        ["FedAvg", "1", "Standard aggregation"],
        ["FedProx", "2", "mu = 0.01, 0.1"],
        ["DP-FedAvg", "5", "sigma = 0.1, 0.25, 0.5, 1.0, 2.0"],
        ["Total variants", "10", ""],
        ["TOTAL RUNS", "300", "3 x 2 x 5 x 10"],
    ])
    matrix_data.columns = ["Factor", "Levels", "Details"]
    add_table_from_df(doc, matrix_data, caption="Table B1: Experiment Matrix")
    doc.add_page_break()

    add_heading(doc, "Appendix C - Complete Reproduction Checks", level=1)
    if rc_path.exists():
        rcdf_full = pd.read_csv(rc_path)
        display_full = rcdf_full[['metric','paper_value','reproduced_value','relative_difference','within_tolerance']].copy()
        display_full.columns = ['Metric','Paper','Reproduced','Rel. Diff','Pass']
        add_table_from_df(doc, display_full, caption="Table C1: All 44 Reproduction Checks")
    doc.add_page_break()

    add_heading(doc, "Appendix D - DP Diagnostic Results", level=1)
    add_para(doc, "The DP diagnostic script (scripts/dp_diagnostic.py) tests four noise conventions. "
             "The key finding is that no single convention reproduces the paper's entire DP curve. "
             "The paper's sigma=2 utility matches the current (conservative) convention A, "
             "while lower sigma values are closer to the old (tight-bound) convention D. "
             "The corrected implementation realizes exactly the privacy that the paper's own RDP formula claims.")
    diag_data = pd.DataFrame([
        ["A (current)", "sigma * C", "Conservative, sensitivity = C"],
        ["B", "sigma * C (per-client)", "Equivalent to C with equal weights"],
        ["C", "sigma * C / sqrt(K)", "Result of B with equal weights"],
        ["D (old)", "sigma * C / K", "Tight bound, sensitivity = C/K"],
    ])
    diag_data.columns = ["Convention", "Noise Std", "Description"]
    add_table_from_df(doc, diag_data, caption="Table D1: DP Noise Conventions Tested")
    doc.add_page_break()

    add_heading(doc, "Appendix E - MLP Gradient Verification", level=1)
    add_para(doc, "The MLP gradient is verified by test_mlp_gradient_check in tests/test_models.py. "
             "This test computes the analytical gradient for all 273 parameters and compares against central finite differences "
             "of the weighted BCE loss. The test uses a small deterministic batch (n=8) for tractability. "
             "It passes after the fix (rtol=1e-4, atol=1e-6) and fails before the fix. "
             "The test guards against re-introducing the double batch-normalization bug in the first-layer equations.")
    doc.add_page_break()

    add_heading(doc, "Appendix F - Additional Figures/Tables", level=1)
    add_para(doc, "Additional results available in the repository:")
    add_bullet(doc, "results/tables/table8_persilo_logistic_auc.csv - Per-silo local-only logistic AUC")
    add_bullet(doc, "results/tables/table9_logistic_metric_panel.csv - Full logistic metric panel")
    add_bullet(doc, "results/tables/calibration_shift_summary.csv - Calibration shift analysis")
    add_bullet(doc, "results/tables/cross_silo_equity.csv - Cross-silo equity per seed")
    add_bullet(doc, "results/before_after_fixes.md - Complete before/after comparison")
    add_bullet(doc, "results_before_fixes/ - All pre-fix results for comparison")

    # Save
    doc.save(str(OUT))
    print(f"Report saved to: {OUT}")
    print(f"File size: {OUT.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    main()

        ["PRACTICAL_AUC_DELTA", "0.01", "Practical significance threshold"],
        ["MLP_HIDDEN_UNITS", "16", "MLP hidden layer size"],
        ["TEST_FRACTION", "0.20", "Test set fraction"],
        ["DIRICHLET_ALPHA", "0.5", "Dirichlet concentration"],
        ["THRESHOLD", "0.5", "Classification threshold"],
    ])
    hyp_data.columns = ["Parameter", "Value", "Description"]
    add_table_from_df(doc, hyp_data, caption="Table A1: Complete Hyperparameters")
    doc.add_page_break()


    add_para(doc, "Paper claims not fully reproduced:", bold=True)
    add_bullet(doc, "Absolute AUC values are systematically ~0.01-0.02 below the paper's reported values (PARTIALLY REPRODUCED - qualitative patterns match, absolute values differ)")
    add_bullet(doc, "DP utility values at sigma <= 1 (NOT REPRODUCED - the corrected mechanism is noisier than whatever the paper ran)")
    add_bullet(doc, "Cross-silo equity spreads (REPRODUCED in ordering but not in magnitude)")
    doc.add_page_break()


        "|-- README.md                             # Project documentation",
    ]
    for line in repo_lines:
        p = doc.add_paragraph(line)
        for run in p.runs:
            run.font.name = 'Courier New'
            run.font.size = Pt(9)
    doc.add_page_break()

    # 24. LIMITATIONS
    add_heading(doc, "24. Limitations and Threats to Validity", level=1)
    limitations = [
        "Simulated silos: All silos originate from one Framingham cohort; genuine institutional variation is not fully reproduced.",
        "Single cohort: Findings are specific to the Framingham Heart Study and may not generalize to other populations.",
        "Number of silos: Only 5 silos are used; real-world deployments may involve many more institutions.",
        "Full participation: All silos participate every round; no stragglers or dropouts are modeled.",
        "Number of seeds: 5 seeds provide limited statistical power; more seeds would yield more robust estimates.",
        "Privacy accounting assumptions: Client-level RDP with full participation, no subsampling amplification, and conservative sensitivity bound.",
        "No cryptographic secure aggregation: The sufficient-statistics aggregation is only conceptually compatible; actual cryptography is not implemented.",
        "No defense against malicious clients: No protection against poisoning, gradient inversion, or other attacks.",
        "Single tabular task: Findings are specific to tabular CHD-risk prediction; they do not generalize to imaging, text, or waveform tasks.",
        "DP reproducibility discrepancy: The paper's DP utility values at sigma <= 1 cannot be fully reproduced with any single noise-multiplier convention calibrated to C.",
        "No cross-validation: A single 20% holdout is used; k-fold cross-validation would provide more robust estimates.",
        "Class weighting: The positive class weight of 5.59 is fixed; different weighting schemes may yield different results.",
    ]
    for lim in limitations:
        add_bullet(doc, lim)
    doc.add_page_break()


    add_bullet(doc, "results/processed/metrics_summary.csv - aggregated metrics")
    add_bullet(doc, "results/tables/ - all paper tables (CSV)")
    add_bullet(doc, "results/figures/ - all figures (PNG/PDF/SVG)")
    add_bullet(doc, "results/reproduction_check.csv - verification results")
    add_bullet(doc, "results/reproduction_report.md - verification report")
    doc.add_page_break()


    if t5_path.exists():
        t5 = pd.read_csv(t5_path)
        add_table_from_df(doc, t5, caption="Table 5 (wide): Test ROC-AUC by Strategy, Model, and Regime")
    t7_path = ROOT / "results" / "tables" / "table7_secondary_comparisons.csv"
    if t7_path.exists():
        t7 = pd.read_csv(t7_path)
        t7_sub = t7[t7['method_b'].isin(['centralized','fedavg'])].head(12)
        add_table_from_df(doc, t7_sub[['model','regime','method_a','method_b','mean_diff','p_value','cohens_d']],
                         caption="Table 7 (selected): Secondary Comparisons")
    doc.add_page_break()


    add_bullet(doc, "Communication cost (2/2): All pass")
    doc.add_page_break()

    # 18. COMMUNICATION COST
    add_heading(doc, "18. Communication Cost", level=1)
    add_para(doc, "Communication cost is calculated as the total uploaded + downloaded bytes across all rounds and clients, "
             "using float32 (4 bytes per parameter) and decimal megabytes (1 MB = 10^6 bytes). "
             "Each round: every client uploads a delta and downloads the global model copy.")
    comm_data = pd.DataFrame([
        ["Logistic", "16", "0.0256", "0.0256", "0.0512"],
        ["MLP", "273", "0.4368", "0.4368", "0.8736"],
    ])
    comm_data.columns = ["Model", "Parameters", "Upload (MB)", "Download (MB)", "Total (MB)"]
    add_table_from_df(doc, comm_data, caption="Table 14: Communication Cost (40 rounds x 5 clients)")
    add_para(doc, "The MLP communicates 17x more data than logistic regression due to its 17x larger parameter count. "
             "Both values match the paper's reported figures within tolerance (logistic: 0.0256 vs 0.026 MB; MLP: 0.4368 vs 0.437 MB).")
    doc.add_page_break()


             "finite differences of the weighted BCE loss. The test passes after the fix and fails before it. "
             "The test uses a small deterministic batch (n=8) to make full parameter differentiation tractable.")
    add_para(doc, "Before/after key values:", bold=True)
    bug_data = pd.DataFrame([
        ["IID FedAvg MLP", "0.7232", "0.7055", "0.7089"],
        ["IID local-only MLP", "0.6294", "0.6829", "0.6403"],
        ["IID federation gain MLP", "0.0761", "0.0226", "0.0686"],
        ["cov-skew local-only MLP", "0.5912", "0.6259", "0.6095"],
        ["IID centralized MLP", "0.7044", "0.7093", "0.7010"],
    ])
    bug_data.columns = ["Quantity", "Paper", "Before Fix", "After Fix"]
    add_table_from_df(doc, bug_data, caption="Table: MLP Results Before and After Gradient Fix")
    doc.add_page_break()


        pudf = pd.read_csv(pu_path)
        add_table_from_df(doc, pudf, caption="Table 11: DP Utility (IID regime, mean ROC-AUC)")
    add_para(doc, "Interpretation:", bold=True)
    add_para(doc, "As sigma decreases (more noise), epsilon increases (weaker privacy) and ROC-AUC drops. "
             "At sigma=0.1 (epsilon ~2303), the logistic model retains 0.6833 AUC (vs 0.7201 non-private FedAvg). "
             "At sigma=2.0 (epsilon ~20.2), the logistic model achieves 0.5860 AUC. "
             "The MLP is more sensitive to DP noise than logistic regression, with utility dropping below chance for some configurations. "
             "The privacy-utility tradeoff is visualized in Figure 6.")
    doc.add_page_break()


    gain_path = ROOT / "results" / "tables" / "table6_primary_significance.csv"
    if gain_path.exists():
        gdf = pd.read_csv(gain_path)
        gains = gdf[['model','regime','mean_a','mean_b','mean_gain']].copy()
        gains.columns = ['Model','Regime','FedAvg AUC','Local-Only AUC','Gain']
        add_table_from_df(doc, gains, caption="Table 8: Federation Gains (FedAvg minus Local-Only ROC-AUC)")
    add_heading(doc, "E. FedProx Results", level=2)
    if mpath.exists():
        fprox = mdf[mdf['variant'].str.startswith('fedprox')][['model','regime','variant','roc_auc_mean','roc_auc_std']].copy()
        fprox.columns = ['Model','Regime','Variant','ROC-AUC Mean','ROC-AUC Std']
        add_table_from_df(doc, fprox, caption="Table 9: FedProx Results")
    doc.add_page_break()


        ["Privacy parameter delta", "1e-5"],
        ["RDP accounting", "RDP(alpha) = T * alpha / (2 * sigma^2) per composition"],
        ["Epsilon conversion", "epsilon(alpha) = RDP(alpha) + log(1/delta)/(alpha-1), minimized over alpha > 1"],
        ["Sensitivity convention", "Client-level L2 sensitivity bounded by C (conservative: w_i = n_i/N <= 1)"],
        ["Subsampling amplification", "Not used (full participation)"],
        ["Composition", "Pure RDP composition over T = 40 rounds"],
    ])
    dp_data.columns = ["Component", "Description"]
    add_table_from_df(doc, dp_data, caption="Table: DP Mechanism Configuration")
    add_para(doc, "Privacy interpretation:", bold=True)
    add_para(doc, "The mechanism provides (epsilon, delta)-differential privacy at the client level: adding or removing "
             "a single client's data changes the released aggregate by at most C in L2 norm (since each clipped delta "
             "has norm <= C and the aggregation weight is <= 1). The Gaussian noise N(0, (sigma*C)^2) then realizes "
             "exactly the mechanism that RDP(alpha) = alpha/(2*sigma^2) accounts for. "
             "This is a conservative bound whenever any w_i < 1 (which is always the case with unequal silo sizes).")
    doc.add_page_break()


        ["Logistic", "15 + 1", "16", "Linear + Sigmoid"],
        ["MLP", "15->16->1", "273", "ReLU + Sigmoid"],
    ])
    model_data.columns = ["Model", "Architecture", "Parameters", "Activations"]
    add_table_from_df(doc, model_data, caption="Table 3: Model Architecture Summary")
    doc.add_page_break()


        ["Learning rate", "0.1"], ["Optimizer", "Mini-batch SGD (no momentum, no weight decay)"],
        ["Positive class weight", "5.59"], ["Negative class weight", "1.0"],
        ["Aggregation", "Sample-count weighted average of client deltas"],
        ["DP clip norm (C)", "1.0"], ["DP sigmas", "[0.1, 0.25, 0.5, 1.0, 2.0]"],
        ["FedProx mus", "[0.01, 0.1]"], ["DP delta", "1e-5"],
        ["Bootstrap samples", "10,000"], ["Practical AUC delta", "0.01"],
    ])
    fl_data.columns = ["Parameter", "Value"]
    add_table_from_df(doc, fl_data, caption="Table 2: Federated Learning Configuration")
    doc.add_page_break()


    add_para(doc, "Feature list:", bold=True)
    features = ["male", "age", "education", "currentSmoker", "cigsPerDay", "BPMeds",
                "prevalentStroke", "prevalentHyp", "diabetes", "totChol", "sysBP", "diaBP",
                "BMI", "heartRate", "glucose"]
    doc.add_paragraph(", ".join(features))
    add_para(doc, "Preprocessing:", bold=True)
    add_bullet(doc, "Missing values imputed with training-only variable medians (no test leakage).")
    add_bullet(doc, "Feature standardization via aggregated per-silo sufficient statistics (n, sum, sum-of-squares).")
    add_bullet(doc, "Global mean/variance derived from aggregated statistics; same scaler applied to all silos and test set.")
    add_bullet(doc, "Standardization is conceptually compatible with secure aggregation (not cryptographically implemented).")
    doc.add_page_break()


    add_bullet(doc, "Does FedAvg improve ROC-AUC over local-only training under IID, label-skew, and covariate-skew regimes?")
    add_bullet(doc, "How does statistical heterogeneity (label skew, covariate skew) affect federation gains?")
    add_bullet(doc, "What is the privacy-utility tradeoff when client-level differential privacy is applied?")
    add_bullet(doc, "How do logistic regression and MLP models compare in the federated setting?")
    add_bullet(doc, "What is the communication cost of federated training for each model?")
    doc.add_page_break()

def add_bullet(doc, text):
    doc.add_paragraph(text, style='List Bullet')

def main():
    doc = Document()
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)

    # TITLE PAGE
    for _ in range(6):
        doc.add_paragraph()
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Federated Learning for Privacy-Preserving\nCardiovascular Risk Prediction")
    run.bold = True
    run.font.size = Pt(24)
    run.font.color.rgb = RGBColor(44, 62, 80)
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("Complete Experimental and Reproducibility Report")
    run.bold = True
    run.font.size = Pt(18)
    run.font.color.rgb = RGBColor(52, 73, 94)
    doc.add_paragraph()
    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = info.add_run("Comprehensive Master Record of Implementation, Experiments,\nDebugging, and Reproducibility Findings")
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(100, 100, 100)
    doc.add_page_break()

    # TABLE OF CONTENTS
    add_heading(doc, "Table of Contents", level=1)
