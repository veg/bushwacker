#!/usr/bin/env python3
"""
Generate publication-grade vector PDF figures for the MDL Phylogenetic Graph
Induction technical report strictly from executed results in benchmark_results.json.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    bench_dir = Path(__file__).resolve().parent
    paper_dir = bench_dir.parent / "paper"
    with open(bench_dir / "benchmark_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # ------------------------------------------------------------------
    # Figure 1: Synthetic Four-Regime Benchmark Summary (4 Panels)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    fig.subplots_adjust(hspace=0.38, wspace=0.28)

    # Panel A: Test 1 Polytomy Calibration
    ax = axes[0, 0]
    t1 = data["test1_polytomy_calibration"]
    methods_t1 = ["MDL", "FastTree2", "IQTREE2"]
    labels_t1 = ["Bushwhacker\n(MDL)", "FastTree 2\n(GTR+CAT)", "IQ-TREE 2\n(GTR+G)"]
    tp_vals = [t1[m]["inferred"] - t1[m]["fp"] for m in methods_t1]
    fp_vals = [t1[m]["fp"] for m in methods_t1]
    rf_vals = [t1[m]["rf_error"] for m in methods_t1]

    x = np.arange(len(methods_t1))
    w = 0.34
    b1 = ax.bar(x - w / 2, tp_vals, width=w, color="#1b7837", label="True Positive Splits (max 13)")
    b2 = ax.bar(x + w / 2, fp_vals, width=w, color="#d73027", label="False Positive Splits (16 zero-length)")
    ax.axhline(13.0, color="#1b7837", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_t1)
    ax.set_ylabel("Number of Internal Splits")
    ax.set_ylim(0, 24.0)
    ax.set_title("A. Test 1: Polytomy Calibration (13 True, 16 Zero-Length)")
    ax.legend(loc="upper left", frameon=False)
    for bar, rf in zip(b2, rf_vals):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2 + (0.06 if h < 0.1 else 0.0),
            h + 0.4,
            f"FP={h:.2f}\n(RF={rf:.1f})",
            ha="center",
            va="bottom",
            fontsize=7.6,
            fontweight="bold",
            color="#a50026" if h > 1.0 else "#1b7837",
        )

    # Panel B: Test 2 Felsenstein Zone LBA
    ax = axes[0, 1]
    t2 = data["test2_felsenstein_zone"]
    methods_t2 = [
        ("Unweighted\nParsimony", t2["MP_true_pct"], t2["MP_lba_pct"], 0.0),
        ("Hamming\nNeighbor-Joining", t2["NJ_true_pct"], t2["NJ_lba_pct"], 0.0),
        ("Bushwhacker\n(MDL)", t2["MDL_true_pct"], t2["MDL_lba_pct"], t2["MDL_polytomy_pct"]),
    ]
    x2 = np.arange(len(methods_t2))
    true_p = [m[1] for m in methods_t2]
    lba_p = [m[2] for m in methods_t2]
    poly_p = [m[3] for m in methods_t2]

    ax.bar(x2 - 0.25, true_p, width=0.25, color="#2166ac", label="True Split (A,B)|(C,D)")
    ax.bar(x2, lba_p, width=0.25, color="#b2182b", label="LBA Trap (A,C)|(B,D)")
    ax.bar(x2 + 0.25, poly_p, width=0.25, color="#878787", label="Cautious Star Polytomy")
    ax.set_xticks(x2)
    ax.set_xticklabels([m[0] for m in methods_t2])
    ax.set_ylabel("Replicates (%)")
    ax.set_ylim(0, 145)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("B. Test 2: Felsenstein Zone LBA (l_long=0.45, l_int=0.025)")
    ax.legend(loc="upper right", frameon=False)
    for idx_m, (tp_v, lba_v) in enumerate(zip(true_p, lba_p)):
        ax.text(idx_m - 0.25, tp_v + 2.0, f"{tp_v:.0f}%", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#2166ac")
        ax.text(idx_m, lba_v + 2.0, f"{lba_v:.0f}%", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#b2182b")

    # Panel C: Test 3 Two-Site Convergent Hijack
    ax = axes[1, 0]
    t3 = data["test3_two_site_hijack"]
    m_keys = ["m=0", "m=2", "m=3", "m=5"]
    m_x = [0, 2, 3, 5]
    mdl_r = [t3[k]["MDL_pair_recall_pct"] for k in m_keys]
    ft_r = [t3[k]["FastTree2_pair_recall_pct"] for k in m_keys]
    iq_r = [t3[k]["IQTREE2_pair_recall_pct"] for k in m_keys]
    sh_r = [t3[k]["MDL_shunt_accuracy_pct"] for k in m_keys]

    ax.plot(m_x, mdl_r, marker="o", linewidth=2.4, color="#1b7837", label="Bushwhacker Host Recall")
    ax.plot(m_x, sh_r, marker="s", linewidth=1.8, linestyle=":", color="#008837", label="Bushwhacker Shunt Site Recall")
    ax.plot(m_x, iq_r, marker="^", linewidth=2.0, linestyle="--", color="#d95f02", label="IQ-TREE 2 (GTR+G) Host Recall")
    ax.plot(m_x, ft_r, marker="v", linewidth=2.0, linestyle="-.", color="#7570b3", label="FastTree 2 Host Recall")
    ax.set_xticks(m_x)
    ax.set_xticklabels([f"m = {v}\ncodons" for v in m_x])
    ax.set_ylabel("Recall (%)")
    ax.set_ylim(-5, 110)
    ax.set_title("C. Test 3: Convergent Replacement Hijack (16 Host Pairs)")
    ax.legend(loc="lower left", frameon=False)

    # Panel D: Test 4 Shunt Mode Discrimination
    ax = axes[1, 1]
    t4 = data["test4_shunt_mode"]
    metrics_t4 = ["INTERVAL Mode\nClassification (%)", "Breakpoint Localization\nMean IoU (%)"]
    vals_t4 = [t4["recomb_interval_mode_pct"], t4["recomb_breakpoint_iou_pct"]]
    bars4 = ax.bar([0, 1], vals_t4, width=0.45, color=["#2b8cbe", "#045a8d"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(metrics_t4)
    ax.set_ylabel("Accuracy / IoU (%)")
    ax.set_ylim(0, 115)
    ax.set_title("D. Test 4: Recombination Tract [400..750] MDL Detection")
    for b in bars4:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 2.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color="#045a8d")

    fig.savefig(paper_dir / "fig1_synthetic_benchmarks.pdf", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: Empirical Datasets Compression & Graph Structure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    fig.subplots_adjust(wspace=0.28)

    t5 = data["test5_empirical"]
    ds_keys = ["bglobin.fasta", "camelid.fasta", "prestin.fasta", "HIV_RT.fasta", "SARS-CoV-2-spike.fasta"]
    ds_labels = ["bglobin\n(17x432)", "Camelid VHH\n(212x288)", "Prestin\n(116x2235)", "HIV-1 RT\n(476x1005)", "SARS-CoV-2 Spike\n(180x3852)"]

    # Panel A: Code Length Cascade (kilonats)
    ax = axes[0]
    x5 = np.arange(len(ds_keys))
    w5 = 0.20
    raw_k = [t5[k]["raw_nats"] / 1000.0 for k in ds_keys]
    star_k = [t5[k]["star_nats"] / 1000.0 for k in ds_keys]
    bb_k = [t5[k]["backbone_nats"] / 1000.0 for k in ds_keys]
    fg_k = [t5[k]["full_graph_nats"] / 1000.0 for k in ds_keys]
    comp_r = [t5[k]["compression_ratio"] for k in ds_keys]

    ax.bar(x5 - 1.5 * w5, raw_k, width=w5, color="#bababa", label="Raw Uncompressed (N*L*ln4)")
    ax.bar(x5 - 0.5 * w5, star_k, width=w5, color="#fdae61", label="Pure Star Polytomy")
    ax.bar(x5 + 0.5 * w5, bb_k, width=w5, color="#4393c3", label="MDL Backbone Tree")
    b_fg = ax.bar(x5 + 1.5 * w5, fg_k, width=w5, color="#1b7837", label="Full MDL Graph (Tree + Shunts)")
    ax.set_yscale("log")
    ax.set_xticks(x5)
    ax.set_xticklabels(ds_labels)
    ax.set_ylabel("Total Description Length (kilonats, log scale)")
    ax.set_ylim(2.0, 2000.0)
    ax.set_title("A. Lossless MDL Information Compression Cascade")
    ax.legend(loc="upper left", frameon=False, fontsize=7.8)
    for bar, cr in zip(b_fg, comp_r):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h * 1.15,
            f"{cr:.1f}x",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color="#1b7837",
        )

    # Panel B: Supported Vertical Resolution (%) vs Contracted Polytomy (%)
    ax = axes[1]
    res_pct = [t5[k]["resolution_pct"] for k in ds_keys]
    poly_pct = [100.0 - r for r in res_pct]
    int_sh = [t5[k]["interval_shunts"] for k in ds_keys]
    sp_sh = [t5[k]["sparse_shunts"] for k in ds_keys]

    ax.bar(x5, res_pct, width=0.48, color="#2166ac", label="MDL-Certified Internal Branches (%)")
    ax.bar(x5, poly_pct, bottom=res_pct, width=0.48, color="#e0e0e0", label="Contracted Polytomy Slots (%)")
    ax.set_xticks(x5)
    ax.set_xticklabels(ds_labels)
    ax.set_ylabel("Percentage of Binary Tree Slots (%)")
    ax.set_ylim(0, 118)
    ax.set_title("B. Backbone Resolution vs. Native Polytomies & Shunts")
    ax.legend(loc="upper right", frameon=False, fontsize=7.8)
    for i, (r_p, n_i, n_s) in enumerate(zip(res_pct, int_sh, sp_sh)):
        ax.text(i, r_p / 2, f"{r_p:.1f}%", ha="center", va="center", fontsize=8.0, fontweight="bold", color="white" if r_p > 18 else "#2166ac")
        ax.text(i, 102.0, f"{n_i} INT / {n_s} SP\nshunts", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#333333")

    fig.savefig(paper_dir / "fig2_empirical_compression.pdf", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 3: Five-Axis Ruthless Stress-Test Suite vs. Classical Paradigms
    # ------------------------------------------------------------------
    stress_path = bench_dir / "stress_test_results.json"
    if stress_path.exists():
        with open(stress_path, "r", encoding="utf-8") as f:
            sdata = json.load(f)

        fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2))
        fig.subplots_adjust(hspace=0.40, wspace=0.30)

        # Panel A: Stress Test 1 - The Likelihood Overfitting Paradox
        ax = axes[0, 0]
        s1 = sdata["stress1_likelihood_paradox"]

        cats = [
            "Polytomy\nIn-Sample d(ln L)\n[T_ML - T_true]",
            "Polytomy\nHeld-Out d(MDL)\n[T_ML - T_MDL]",
            "Convergence (m=3)\nIn-Sample d(ln L)\n[T_ML - T_true]",
            "Convergence (m=3)\nMDL Savings\n[T_ML - G_MDL]",
        ]
        vals_a = [
            s1["polytomy_mean_excess_train_logl"],
            s1["polytomy_heldout_test_mdl_savings_nats"],
            s1["convergent_mean_excess_logl"],
            s1["convergent_mean_mdl_savings_over_ml_nats"],
        ]
        colors_a = ["#d73027", "#1b7837", "#d73027", "#1b7837"]
        bars_a = ax.bar(np.arange(4), vals_a, width=0.50, color=colors_a)
        ax.set_yscale("symlog", linthresh=5.0)
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(cats, fontsize=7.6)
        ax.set_ylabel("Delta Log-Likelihood / Delta MDL (nats, symlog)")
        ax.set_ylim(0, 1200)
        ax.set_title("A. Stress Test 1: The ML Overfitting Paradox vs. MDL")
        for b, v in zip(bars_a, vals_a):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v * 1.18 + 0.5,
                f"+{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.0,
                fontweight="bold",
                color=b.get_facecolor(),
            )

        # Panel B: Stress Test 2 - ML Home Turf (Strictly Binary GTR+F+G4 Precision)
        ax = axes[0, 1]
        s2 = sdata["stress2_ml_home_turf"]
        L_keys = ["150", "300", "600", "1500"]
        L_x = [150, 300, 600, 1500]
        m_specs = [
            ("MDL", "Bushwhacker (MDL)", "#1b7837", "o", "-"),
            ("IQTREE_aBayes95", "IQ-TREE 2 (aBayes>=0.95)", "#045a8d", "P", "-."),
            ("IQTREE_Raw", "IQ-TREE 2 (GTR+G Raw)", "#d95f02", "^", "--"),
            ("FastTree_Raw", "FastTree 2 (GTR+CAT)", "#7570b3", "v", "-."),
            ("NJ_JC", "Neighbor-Joining", "#e7298a", "s", ":"),
            ("MP", "Max Parsimony", "#66a61e", "d", ":"),
        ]
        for key_m, lbl_m, col_m, mk_m, ls_m in m_specs:
            y_prec = [s2[lk]["methods"][key_m]["precision_pct"] for lk in L_keys]
            ax.plot(L_x, y_prec, marker=mk_m, linestyle=ls_m, linewidth=2.1, color=col_m, label=lbl_m)
        ax.set_xscale("log")
        ax.set_xticks(L_x)
        ax.set_xticklabels([f"L={v} bp" for v in L_x])
        ax.set_ylabel("Realized Clade Precision (%)")
        ax.set_ylim(72, 103)
        ax.set_title("B. Stress Test 2: ML Home Turf (Binary GTR+F+G4)")
        ax.legend(loc="lower right", frameon=False, fontsize=7.4)

        # Panel C: Stress Test 3 - Support-Collapsed ML & Bayesian Trees vs. MDL
        ax = axes[1, 0]
        s3 = sdata["stress3_support_collapsed"]["polytomy_methods"]
        m3_order = [
            ("MDL (No Bootstrap)", "Bushwhacker\n(No Boot)"),
            ("IQTREE aBayes>=0.95", "IQ-TREE 2\naBayes≥.95"),
            ("IQTREE SH-aLRT>=80", "IQ-TREE 2\nSH-aLRT≥80"),
            ("IQTREE UFBoot>=95", "IQ-TREE 2\nUFBoot≥95"),
            ("FastTree SH>=0.90", "FastTree 2\nSH≥0.90"),
            ("FastTree SH>=0.95", "FastTree 2\nSH≥0.95"),
        ]
        x3 = np.arange(len(m3_order))
        rf3 = [s3[k]["rf_error"] for k, _ in m3_order]
        fp3 = [s3[k]["fp"] for k, _ in m3_order]
        w3 = 0.36
        b_rf = ax.bar(x3 - w3 / 2, rf3, width=w3, color="#2166ac", label="Robinson-Foulds Error (RF)")
        b_fp = ax.bar(x3 + w3 / 2, fp3, width=w3, color="#d73027", label="False Positive Splits (FP/16)")
        ax.set_xticks(x3)
        ax.set_xticklabels([lbl for _, lbl in m3_order], fontsize=5.9)
        ax.set_ylabel("Mean Splits / RF Error")
        ax.set_ylim(0, 5.2)
        ax.set_title("C. Stress Test 3: Support-Collapsed ML/Bayesian vs. Bushwhacker")
        ax.legend(loc="upper left", frameon=False, fontsize=7.8)
        for bar in b_rf:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.10, f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#2166ac")

        # Panel D: Stress Test 4 - Empirical Mammalian Prestin (116 Mammals)
        ax = axes[1, 1]
        s5 = dict(sdata["stress5_prestin_head_to_head"])
        prestin_cmp_path = bench_dir.parent / "results" / "prestin_collapsed_ml_comparison.json"
        if prestin_cmp_path.exists():
            pcmp = json.loads(prestin_cmp_path.read_text(encoding="utf-8"))["models"]["Bushwhacker_MDL"]
            s5["MDLPhyloEngine (Ours)"] = {
                "true_positives": pcmp["tp"],
                "false_positives": pcmp["fp"],
                "rf_error": pcmp["fp"] + pcmp["fn"],
            }
        m5_order = [
            ("MDLPhyloEngine (Ours)", "Bushwhacker\n(+2 shunts)"),
            ("IQTREE2 (aBayes>=0.95)", "IQ-TREE 2\naBayes≥.95"),
            ("IQTREE2 (SH-aLRT>=80)", "IQ-TREE 2\nSH-aLRT≥80"),
            ("FastTree2 (SH>=0.90)", "FastTree 2\nSH≥0.90"),
            ("IQTREE2 (GTR+G Raw)", "IQ-TREE 2\nRaw"),
            ("Max Parsimony (NNI)", "Max\nParsimony"),
            ("Neighbor-Joining (JC)", "Neighbor\nJoining"),
        ]
        x5_s = np.arange(len(m5_order))
        tp5 = [s5[k]["true_positives"] for k, _ in m5_order]
        fp5 = [s5[k]["false_positives"] for k, _ in m5_order]
        rf5 = [s5[k]["rf_error"] for k, _ in m5_order]
        w5_s = 0.36
        ax.bar(x5_s - w5_s / 2, tp5, width=w5_s, color="#1b7837", label="True Positive Clades (TP)")
        b_fp5 = ax.bar(x5_s + w5_s / 2, fp5, width=w5_s, color="#d73027", label="False Positive Clades (FP)")
        ax.set_xticks(x5_s)
        ax.set_xticklabels([lbl for _, lbl in m5_order], fontsize=5.9)
        ax.set_ylabel("Clade Count (103 Shared Mammals)")
        ax.set_ylim(0, 102)
        ax.set_title("D. Stress Test 4: Empirical Mammalian Prestin vs. Species Tree")
        ax.legend(loc="upper right", frameon=False, fontsize=7.6)
        for idx_5, (t_v, f_v, r_v) in enumerate(zip(tp5, fp5, rf5)):
            ax.text(idx_5, max(t_v, f_v) + 2.0, f"RF={r_v}", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="#222222")

        fig.savefig(paper_dir / "fig3_stress_tests.pdf", bbox_inches="tight")
        plt.close(fig)

    print("Generated fig1_synthetic_benchmarks.pdf, fig2_empirical_compression.pdf, and fig3_stress_tests.pdf")


if __name__ == "__main__":
    main()
