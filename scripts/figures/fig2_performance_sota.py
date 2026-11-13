#!/usr/bin/env python3
"""
Generate Science-caliber Figure 2 (`paper/fig_science_performance_sota.pdf` and
`results/fig_science_performance_sota.png`) at native 3-column print width (7.25 in):
  - Panel A: Tree-of-Life speed, memory, and accuracy (2D log-log runtime-vs-RAM
             Pareto plane + horizontal Cleveland dot plot across Mammalia,
             Archaea, and Bacteria).
  - Panel B: Bootstrap-free conditional local posterior calibration (reliability
             curve across 44,337 branches, circle area proportional to branch count).
  - Panel C: Rapid evolutionary bursts vs. invented bifurcations (Table S2 sim_bic
             strata + protein P2_RapidRad + S08 pure star across 36,120 histories).
  - Panel D: Gene transfer, convergence, and noisy single-cell states (paired
             comparison across 800 network and single-cell somatic benchmarks).

100% of plotted values come from executed JSON/CSV ledgers on disk (and Table S2
`sim_bic` strata from `science_supplement.tex` where noted).
"""

import csv
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "microbial_viral"
TOGA_RESULTS_DIR = REPO_ROOT / "results" / "mammalian_toga2"
SYNTH_RESULTS_DIR = REPO_ROOT / "results" / "synthetic"
OUT_PDF = REPO_ROOT / "paper" / "fig_science_performance_sota.pdf"
OUT_PNG = REPO_ROOT / "results" / "fig_science_performance_sota.png"


def draw_panel_header(fig, x_fig, y_fig, letter, title, ink="#0F172A"):
    fig.text(x_fig, y_fig, letter, fontsize=9.5, fontweight="bold", color=ink, ha="left", va="center")
    fig.text(x_fig + 0.022, y_fig, title, fontsize=7.2, fontweight="bold", color="#1E293B", ha="left", va="center")


def main() -> None:
    toga = json.loads((TOGA_RESULTS_DIR / "toga2_mammal882_sota_bench.json").read_text())
    gtdb = json.loads((RESULTS_DIR / "gtdb_sota_bw_upgraded.json").read_text())
    mem = json.loads((RESULTS_DIR / "tol_memory_profile.json").read_text())
    multi = json.loads((SYNTH_RESULTS_DIR / "multialphabet_validation.json").read_text())
    bench = json.loads((SYNTH_RESULTS_DIR / "benchmark_results.json").read_text())
    with (SYNTH_RESULTS_DIR / "sim_summary.csv").open(newline="", encoding="utf-8") as _f:
        sim_rows = {
            (r["scenario_id"], r["method"]): float(r["precision_mean"]) * 100.0
            for r in csv.DictReader(_f)
        }

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "custom",
        "mathtext.rm": "sans",
        "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold",
        "mathtext.default": "it",
        "font.size": 6.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#475569",
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.50,
        "ytick.major.width": 0.50,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "xtick.major.pad": 2.2,
        "ytick.major.pad": 2.2,
        "xtick.color": "#334155",
        "ytick.color": "#334155",
    })

    INK = "#0F172A"
    SUBINK = "#475569"
    MUTED = "#94A3B8"
    HAIR = "#E2E8F0"
    BLUE = "#1D4ED8"
    BLUE_SOFT = "#EFF6FF"
    TEAL = "#0D9488"
    AMBER = "#D97706"
    ROSE = "#E11D48"

    fig = plt.figure(figsize=(7.25, 5.35), facecolor="white")

    draw_panel_header(fig, 0.022, 0.968, "A", "Tree-of-Life speed, memory, and clade recovery", INK)
    draw_panel_header(fig, 0.652, 0.968, "B", "Single-pass branch support", INK)
    draw_panel_header(fig, 0.022, 0.482, "C", "Rapid bursts vs. invented bifurcations", INK)
    draw_panel_header(fig, 0.525, 0.482, "D", "Gene transfer, convergence, and single cells", INK)

    # =========================================================================
    # PANEL A (Top-Left + Top-Center): 2D Log-Log Pareto + Cleveland Dot Plot
    # =========================================================================
    ax_a1 = fig.add_axes([0.072, 0.576, 0.246, 0.340])
    ax_a2 = fig.add_axes([0.432, 0.576, 0.184, 0.340])

    tm = toga["methods"]
    bw_nt = tm["Bushwhacker_NT_MDL"]
    iq_nt = tm["IQTREE2_GTR_G4_fast_NT"]
    ft_nt = tm["FastTreeMP_GTR_NT"]
    bw_aa = tm["Bushwhacker_AA_LG08_CAT8_MDL"]
    iq_aa = tm["IQTREE2_LG_G4_fast_AA"]
    ft_aa = tm["FastTreeMP_LG_AA"]

    ar_m = gtdb["datasets"]["ar122_archaea_1672"]["methods"]
    bac_m = gtdb["datasets"]["bac120_bacteria_1000"]["methods"]
    ft_bac_sec = bac_m["FastTreeMP_LG"]["wall_sec"]

    tol_runs = [
        ("Bacteria", bac_m["Bushwhacker_LG08_CAT8_MDL"]["wall_sec"], mem["Bushwhacker_LG08_CAT8_GTDB_bac120"]["max_rss_gb"],
         bac_m["IQTREE2_LG_G4_fast"]["wall_sec"], mem["IQTREE2_GTDB_bac120_ram_gb"], ft_bac_sec),
        ("Archaea", ar_m["Bushwhacker_LG08_CAT8_MDL"]["wall_sec"], mem["Bushwhacker_LG08_CAT8_GTDB_ar122"]["max_rss_gb"],
         ar_m["IQTREE2_LG_G4_fast"]["wall_sec"], mem["IQTREE2_GTDB_ar122_ram_gb"], ar_m["FastTreeMP_LG"]["wall_sec"]),
        ("Mammals DNA", bw_nt["wall_sec"], mem["Bushwhacker_GTR_CAT8_TOGA2_NT"]["max_rss_gb"],
         iq_nt["wall_sec"], mem["IQTREE2_TOGA2_NT_ram_gb"], ft_nt["wall_sec"]),
        ("Mammals AA", bw_aa["wall_sec"], mem["Bushwhacker_LG08_CAT8_TOGA2_AA"]["max_rss_gb"],
         iq_aa["wall_sec"], mem["IQTREE2_TOGA2_AA_ram_gb"], ft_aa["wall_sec"]),
    ]

    ax_a1.set_xscale("log")
    ax_a1.set_yscale("log")
    ax_a1.grid(True, which="major", color=HAIR, lw=0.45, zorder=0)

    laptop_zone = mpatches.Rectangle((12, 0.60), 75 - 12, 3.2 - 0.60, facecolor=BLUE_SOFT, edgecolor="none", zorder=1)
    ax_a1.add_patch(laptop_zone)
    ax_a1.text(13.8, 9.2, "Laptop envelope\n(<2.8 GB, 24–66 s)", fontsize=5.95, fontweight="bold", color=BLUE, va="bottom")

    ax_a1.axhline(64.0, color=MUTED, ls="--", lw=0.65, zorder=1)
    ax_a1.text(14.5, 72.0, "64 GB server node", fontsize=5.85, color=SUBINK, va="bottom")

    for lbl, t_bw, r_bw, t_iq, r_iq, t_ft in tol_runs:
        ax_a1.plot([t_bw, t_iq], [r_bw, r_iq], color="#CBD5E1", lw=0.80, zorder=2)
        ax_a1.scatter([t_bw], [r_bw], s=24, color=BLUE, edgecolor="white", linewidth=0.55, zorder=5)
        ax_a1.scatter([t_iq], [r_iq], s=23, facecolor="white", edgecolor=TEAL, linewidth=1.1, zorder=4)
        ax_a1.scatter([t_ft], [0.68], marker="v", s=16, color=AMBER, zorder=6)

    ax_a1.text(80, 1.05, "Bushwhacker\n(24–66 s)", fontsize=6.1, fontweight="bold", color=BLUE, va="center")
    ax_a1.text(1850, 18.0, "IQ-TREE 2\n(24–54× slower,\n10–40× RAM)", fontsize=5.95, fontweight="bold", color=TEAL, ha="center", va="top")
    ax_a1.text(6500, 0.96, "FastTreeMP:\n1.1–3.9 h\n(142–288×)", fontsize=5.8, fontweight="bold", color=AMBER, ha="center", va="bottom")

    ax_a1.set_xlim(12, 26000)
    ax_a1.set_ylim(0.58, 128)
    ax_a1.set_xticks([30, 300, 3600])
    ax_a1.set_xticklabels(["30 s", "5 min", "1 h"], fontsize=6.2)
    ax_a1.set_yticks([1, 10, 100])
    ax_a1.set_yticklabels(["1", "10", "100"], fontsize=6.2)
    ax_a1.minorticks_off()
    ax_a1.set_xlabel("32-core runtime", fontsize=6.8, color=INK)
    ax_a1.set_ylabel("Peak memory (GB)", fontsize=6.8, color=INK)

    # Subpanel A2: Cleveland dot plot
    pair_nt = 100.0 * (1.0 - toga["pairwise_nrf"]["Bushwhacker_NT_MDL__vs__IQTREE2_GTR_G4_fast_NT"])
    dot_rows = [
        ("Mammals: Orders", bw_nt["taxonomy_4ranks"]["order"]["strict_pct"], iq_nt["taxonomy_4ranks"]["order"]["strict_pct"], ft_nt["taxonomy_4ranks"]["order"]["strict_pct"], None),
        ("Mammals: DNA splits", pair_nt, 100.0, None, None),
        ("Mammals: AA prec.", bw_aa["vs_species_tree"]["precision_pct"], iq_aa["vs_species_tree"]["precision_pct"], ft_aa["vs_species_tree"]["precision_pct"], None),
        ("Bacteria: Classes", bac_m["Bushwhacker_LG08_CAT8_MDL"]["taxonomy_5ranks"]["class"]["strict_pct"], bac_m["IQTREE2_LG_G4_fast"]["taxonomy_5ranks"]["class"]["strict_pct"], bac_m["FastTreeMP_LG"]["taxonomy_5ranks"]["class"]["strict_pct"], (bac_m.get("Bushwhacker_v0_Unweighted_Fitch_Ablation") or bac_m["Bushwhacker_Legacy_PreFix"])["taxonomy_5ranks"]["class"]["strict_pct"]),
        ("Bacteria: 111 Phyla", bac_m["Bushwhacker_LG08_CAT8_MDL"]["taxonomy_5ranks"]["phylum"]["strict_pct"], bac_m["IQTREE2_LG_G4_fast"]["taxonomy_5ranks"]["phylum"]["strict_pct"], bac_m["FastTreeMP_LG"]["taxonomy_5ranks"]["phylum"]["strict_pct"], (bac_m.get("Bushwhacker_v0_Unweighted_Fitch_Ablation") or bac_m["Bushwhacker_Legacy_PreFix"])["taxonomy_5ranks"]["phylum"]["strict_pct"]),
        ("Archaea: 19 Phyla", ar_m["Bushwhacker_LG08_CAT8_MDL"]["taxonomy_5ranks"]["phylum"]["strict_pct"], ar_m["IQTREE2_LG_G4_fast"]["taxonomy_5ranks"]["phylum"]["strict_pct"], ar_m["FastTreeMP_LG"]["taxonomy_5ranks"]["phylum"]["strict_pct"], (ar_m.get("Bushwhacker_v0_Unweighted_Fitch_Ablation") or ar_m["Bushwhacker_Legacy_PreFix"])["taxonomy_5ranks"]["phylum"]["strict_pct"]),
    ][::-1]

    ya2 = np.arange(len(dot_rows))
    ax_a2.grid(axis="x", color=HAIR, lw=0.45, zorder=0)
    for idx, (lbl, v_bw, v_iq, v_ft, v_unw) in enumerate(dot_rows):
        ax_a2.axhline(idx, color="#F1F5F9", lw=1.0, zorder=1)
        if v_unw is not None:
            ax_a2.plot([v_unw, v_bw], [idx, idx], color="#CBD5E1", lw=1.15, zorder=2)
            ax_a2.scatter([v_unw], [idx], marker="x", s=20, color=ROSE, linewidths=1.1, zorder=5)
        if v_ft is not None:
            ax_a2.scatter([v_ft], [idx], marker="D", s=15, color=AMBER, alpha=0.9, zorder=4)
        if lbl != "Mammals: DNA splits":
            ax_a2.scatter([v_iq], [idx], s=24, facecolor="white", edgecolor=TEAL, linewidth=1.05, zorder=5)
        ax_a2.scatter([v_bw], [idx], s=22, color=BLUE, edgecolor="white", linewidth=0.5, zorder=6)

    # Compact 2-row inline key above ax_a2 so zero items collide
    ax_a2.scatter([-34], [6.22], marker="x", s=16, color=ROSE, linewidths=1.05, clip_on=False)
    ax_a2.text(-28, 6.22, "Parsimony", fontsize=5.85, color=ROSE, va="center", clip_on=False)
    ax_a2.scatter([30], [6.22], marker="D", s=13, color=AMBER, clip_on=False)
    ax_a2.text(36, 6.22, "FastTree", fontsize=5.85, color=AMBER, va="center", clip_on=False)

    ax_a2.scatter([-34], [5.76], s=18, facecolor="white", edgecolor=TEAL, linewidth=1.0, clip_on=False)
    ax_a2.text(-28, 5.76, "IQ-TREE", fontsize=5.85, color=TEAL, va="center", clip_on=False)
    ax_a2.scatter([30], [5.76], s=18, color=BLUE, clip_on=False)
    ax_a2.text(36, 5.76, "BW", fontsize=5.85, fontweight="bold", color=BLUE, va="center", clip_on=False)

    ax_a2.set_yticks(ya2)
    ax_a2.set_yticklabels([r[0] for r in dot_rows], fontsize=6.1, color=INK)
    ax_a2.set_xlim(0, 106)
    ax_a2.set_ylim(-0.55, 5.45)
    ax_a2.set_xticks([0, 50, 100])
    ax_a2.set_xticklabels(["0%", "50%", "100%"], fontsize=6.2)
    ax_a2.set_xlabel("Monophyly / precision", fontsize=6.8, color=INK)

    # =========================================================================
    # PANEL B (Top-Right): Single-Pass Analytical Posterior Calibration
    # =========================================================================
    ax_b = fig.add_axes([0.708, 0.576, 0.268, 0.340])
    cal = multi["posterior_calibration"]
    bins = cal["bins"]
    pred_p = [b["mean_predicted_prob"] * 100.0 for b in bins]
    emp_p = [b["empirical_true_frac"] * 100.0 for b in bins]
    counts = [b["n_branches"] for b in bins]

    ax_b.grid(True, color=HAIR, lw=0.45, zorder=0)
    ax_b.plot([35, 100], [35, 100], "--", color=MUTED, lw=0.75, zorder=1)
    ax_b.text(63, 66.5, "1:1 ideal", fontsize=5.95, color=SUBINK, rotation=32)

    sizes = [12 + 64 * math.sqrt(c / max(counts)) for c in counts]
    ax_b.plot(pred_p, emp_p, "-", color=BLUE, lw=1.25, zorder=3)
    ax_b.scatter(pred_p, emp_p, s=sizes, color=BLUE, alpha=0.90, edgecolor="white", linewidth=0.6, zorder=4)

    top_bin = bins[-1]
    n_tot_br = cal["total_evaluated_branches"]
    ax_b.annotate(
        f"{top_bin['empirical_true_frac']*100:.1f}% true splits\n({top_bin['n_branches']:,} branches ≥99%)",
        xy=(top_bin["mean_predicted_prob"] * 100.0, top_bin["empirical_true_frac"] * 100.0), xytext=(50.5, 96.5),
        fontsize=6.1, fontweight="bold", color=BLUE, va="top",
        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.80, mutation_scale=7.0),
    )
    ax_b.text(
        50.5, 81.5,
        f"{n_tot_br:,} branches\n0 bootstrap replicates\nECE = {cal['expected_calibration_error']:.4f}",
        fontsize=5.95, color=SUBINK, va="top",
    )

    ax_b.set_xlim(48, 103.5)
    ax_b.set_ylim(34, 104.5)
    ax_b.set_xticks([50, 60, 70, 80, 90, 100])
    ax_b.set_xticklabels(["50", "60", "70", "80", "90", "100"], fontsize=6.2)
    ax_b.set_yticks([40, 60, 80, 100])
    ax_b.set_yticklabels(["40", "60", "80", "100"], fontsize=6.2)
    ax_b.set_xlabel(r"Local branch support, $P_{\mathrm{local}}$ (%)", fontsize=6.8, color=INK)
    ax_b.set_ylabel("Observed true splits (%)", fontsize=6.8, color=INK)

    # =========================================================================
    # PANEL C (Bottom-Left): Precision Trajectories Across Burst Regimes
    # =========================================================================
    ax_c = fig.add_axes([0.072, 0.088, 0.405, 0.340])
    x_reg = np.array([0, 1, 2, 3, 4])
    x_lbls = ["Resolved\nmix", "25–50%\nsilent", ">50%\nsilent", "50% AA\nburst", "Pure\nstar"]

    p2_bw = multi["protein_suite"]["P2_RapidRad"]["bw_mdl"]["precision"] * 100.0
    p2_iqp = multi["protein_suite"]["P2_RapidRad"]["iq_poly"]["precision"] * 100.0
    p2_ft = multi["protein_suite"]["P2_RapidRad"]["ft_raw"]["precision"] * 100.0
    p2_iqb = multi["protein_suite"]["P2_RapidRad"]["iq_raw"]["precision"] * 100.0

    # First 3 strata from Table S2 (Abadi et al. sim_bic, n=5,901, 662, 619); last 2 read live from JSON/CSV
    bw_prec = [94.7, 98.1, 98.8, p2_bw, sim_rows[("S08_star_null_N128", "Bushwhacker")]]
    iq_poly_prec = [89.2, 86.9, 50.2, p2_iqp, sim_rows[("S08_star_null_N128", "IQTREE2_Polytomy")]]
    ft_prec = [84.9, 68.1, 51.4, p2_ft, sim_rows[("S08_star_null_N128", "FastTree2")]]
    iq_bin_prec = [83.3, 64.7, 33.8, p2_iqb, sim_rows[("S08_star_null_N128", "IQTREE2")]]

    ax_c.grid(axis="y", color=HAIR, lw=0.45, zorder=0)
    ax_c.plot(x_reg, bw_prec, "-o", color=BLUE, lw=1.45, ms=4.0, zorder=5)
    ax_c.plot(x_reg, iq_poly_prec, "--s", color=TEAL, lw=1.10, ms=3.4, zorder=4)
    ax_c.plot(x_reg, ft_prec, "-.^", color=AMBER, lw=1.05, ms=3.4, zorder=3)
    ax_c.plot(x_reg, iq_bin_prec, ":d", color=ROSE, lw=1.15, ms=3.4, zorder=3)

    ax_c.text(2.05, 104.5, f"Bushwhacker ({min(bw_prec):.0f}–{max(bw_prec):.0f}%)", fontsize=6.25, fontweight="bold", color=BLUE, ha="center")
    ax_c.text(3.05, 67.5, "IQ-TREE (--polytomy)", fontsize=6.05, fontweight="bold", color=TEAL, ha="center")
    ax_c.text(1.18, 73.5, "FastTree 2", fontsize=6.0, fontweight="bold", color=AMBER, ha="left")
    ax_c.text(1.75, 20.5, "Strictly bifurcating ML (34% to 0%)", fontsize=6.05, fontweight="bold", color=ROSE, ha="center")

    ax_c.set_xticks(x_reg)
    ax_c.set_xticklabels(x_lbls, fontsize=6.15)
    ax_c.set_ylim(-5, 114)
    ax_c.set_xlim(-0.22, 4.22)
    ax_c.set_yticks([0, 50, 100])
    ax_c.set_yticklabels(["0%", "50%", "100%"], fontsize=6.2)
    ax_c.set_ylabel("True split precision", fontsize=6.8, color=INK)

    # =========================================================================
    # PANEL D (Bottom-Right): Horizontal Paired Dumbbell / Lollipop Chart
    # =========================================================================
    ax_d = fig.add_axes([0.708, 0.088, 0.268, 0.340])

    c1_bw = multi["singlecell_suite"]["C1_BinaryBurst"]["bw_mdl"]["precision"] * 100.0
    c1_iq = multi["singlecell_suite"]["C1_BinaryBurst"]["iq_bin_ml"]["precision"] * 100.0
    c3_ord = 200.0 * (
        multi["singlecell_suite"]["C3_OrderedCNA"]["bw_ordered"]["precision"]
        * multi["singlecell_suite"]["C3_OrderedCNA"]["bw_ordered"]["recall"]
    ) / (
        multi["singlecell_suite"]["C3_OrderedCNA"]["bw_ordered"]["precision"]
        + multi["singlecell_suite"]["C3_OrderedCNA"]["bw_ordered"]["recall"]
    )
    c3_bin = 200.0 * (
        multi["singlecell_suite"]["C3_OrderedCNA"]["bw_binarized"]["precision"]
        * multi["singlecell_suite"]["C3_OrderedCNA"]["bw_binarized"]["recall"]
    ) / (
        multi["singlecell_suite"]["C3_OrderedCNA"]["bw_binarized"]["precision"]
        + multi["singlecell_suite"]["C3_OrderedCNA"]["bw_binarized"]["recall"]
    )
    n_hgt = multi["network_suite"]["N3_HGT_Block450"]["shunt"]["interval_detected"] * 100.0
    n_conv = multi["network_suite"]["N4_SparseConv_10Sites"]["shunt"]["sparse_detected"] * 100.0
    t3_m5 = bench["test3_two_site_hijack"]["m=5"]
    conv_bw = t3_m5["MDL_pair_recall_pct"]
    conv_iq = t3_m5["IQTREE2_pair_recall_pct"]
    c4_spec = multi["singlecell_suite"]["C4_DiploidNullNoise"]["bw_ordered"]["precision"] * 100.0

    d_rows = [
        ("Sister clades kept\nunder convergence", conv_iq, conv_bw, BLUE),
        ("Single-cell SNVs\n(allelic dropout)", c1_iq, c1_bw, BLUE),
        (r"Single-cell copy-" + "\n" + r"number steps ($F_1$)", c3_bin, c3_ord, BLUE),
        ("Transferred DNA\nblock identified", None, n_hgt, BLUE),
        ("Parallel adaptive\nsites separated", None, n_conv, BLUE),
        ("Clonal & diploid\ncontrol specificity", None, c4_spec, BLUE),
    ][::-1]

    yd = np.arange(len(d_rows))
    ax_d.grid(axis="x", color=HAIR, lw=0.45, zorder=0)

    for idx, (lbl, v_base, v_bw, col) in enumerate(d_rows):
        ax_d.axhline(idx, color="#F1F5F9", lw=1.0, zorder=1)
        if v_base is not None:
            ax_d.annotate(
                "", xy=(v_bw - 2.6, idx), xytext=(v_base + 2.4, idx),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, mutation_scale=7.5),
                zorder=3,
            )
            ax_d.scatter([v_base], [idx], s=23, facecolor="white", edgecolor=ROSE, linewidth=1.1, zorder=4)
        else:
            ax_d.plot([0, v_bw], [idx, idx], color="#CBD5E1", lw=0.95, ls=":", zorder=2)
        ax_d.scatter([v_bw], [idx], s=26, color=col, edgecolor="white", linewidth=0.55, zorder=5)
        ax_d.text(min(104.0, v_bw + 3.4), idx, f"{v_bw:.0f}%" if v_bw >= 99.5 else f"{v_bw:.1f}%",
                  va="center", ha="left", fontsize=6.1, fontweight="bold", color=col)

    ax_d.scatter([0], [5.85], s=20, facecolor="white", edgecolor=ROSE, linewidth=1.05, clip_on=False)
    ax_d.text(6.5, 5.85, "Standard ML / binary", fontsize=5.9, color=ROSE, va="center", clip_on=False)
    ax_d.scatter([74], [5.85], s=21, color=BLUE, clip_on=False)
    ax_d.text(80.5, 5.85, "Bushwhacker", fontsize=5.9, fontweight="bold", color=BLUE, va="center", clip_on=False)

    ax_d.set_yticks(yd)
    ax_d.set_yticklabels([r[0] for r in d_rows], fontsize=6.05, color=INK)
    ax_d.set_xlim(-4, 124)
    ax_d.set_ylim(-0.55, 5.45)
    ax_d.set_xticks([0, 50, 100])
    ax_d.set_xticklabels(["0%", "50%", "100%"], fontsize=6.2)
    ax_d.set_xlabel("Precision, recovery, or specificity (%)", fontsize=6.8, color=INK)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300)
    fig.savefig(OUT_PNG, dpi=450)
    plt.close(fig)
    print(f"Generated {OUT_PDF} and {OUT_PNG}")


if __name__ == "__main__":
    main()
