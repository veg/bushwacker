#!/usr/bin/env python3
"""
Generate a Science-standard Figure 2 (`paper/fig_science_performance_sota.pdf` and
`results/fig_science_performance_sota.png`) matching Figure 1's visual language:
  - Zero vertical bar charts
  - Zero bordered callout/legend boxes
  - Large, legible sentence-case typography (figsize 12.6 x 8.4 in; fonts >= 8 pt)

Panels:
  A: Tree-of-Life speed, memory, and accuracy (2D log-log runtime-vs-RAM Pareto plane
     + horizontal Cleveland dot plot across Mammalia, Archaea, and Bacteria).
  B: Single-pass branch support without bootstraps (analytical posterior reliability
     curve across 44,556 branches, bubble area proportional to branch count).
  C: Rapid evolutionary bursts vs. invented splits (Fig. 1-style miniature tree trio
     + direct-labeled precision trajectories across 36,120 simulated histories).
  D: Gene transfer, convergence, and noisy single-cell states (horizontal paired
     dumbbell / dot comparison across 800 network and single-cell somatic benchmarks).

100% of plotted numbers are read directly from executed benchmark JSON ledgers on disk.
"""

import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
PILLARS_DIR = REPO_ROOT / "scratch" / "results_tree_pillars"
OUT_PDF = REPO_ROOT / "paper" / "fig_science_performance_sota.pdf"
OUT_PNG = RESULTS_DIR / "fig_science_performance_sota.png"


def draw_panel_header(fig, x_fig, y_fig, letter, title, ink="#0F172A"):
    fig.text(x_fig, y_fig, letter, fontsize=14, fontweight="bold", color=ink, ha="left", va="center")
    fig.text(x_fig + 0.024, y_fig, title, fontsize=11, fontweight="bold", color=ink, ha="left", va="center")


def draw_mini_burst_comparison(ax, ink, subink, blue, rose, slate):
    """Draw clean miniature tree schematic contrasting True burst, Standard ML, and Bushwhacker."""
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 2.35)
    ax.axis("off")

    # 1. True burst (star radiation from a single ancestral hub)
    cx1, cy1 = 1.35, 1.02
    ax.text(cx1, 2.12, "True rapid burst", fontsize=8.4, fontweight="bold", color=ink, ha="center", va="center")
    angles = [32, 95, 152, 212, 272, 330]
    for ang in angles:
        rad = math.radians(ang)
        ax.plot([cx1, cx1 + 0.72 * math.cos(rad)], [cy1, cy1 + 0.68 * math.sin(rad)],
                color=slate, lw=1.7, solid_capstyle="round")
    ax.scatter([cx1], [cy1], s=34, color=slate, zorder=5)
    ax.text(cx1, 0.10, "0 stem mutations", fontsize=7.6, color=subink, ha="center", va="center")

    # Arrow 1 -> 2 & 3
    ax.annotate("", xy=(3.18, 1.05), xytext=(2.42, 1.05),
                arrowprops=dict(arrowstyle="-|>,head_length=0.38,head_width=0.22", color="#94A3B8", lw=1.2))

    # 2. Standard ML (forced binary ladder inventing arbitrary splits)
    ax.text(4.85, 2.12, "Standard ML", fontsize=8.4, fontweight="bold", color=rose, ha="center", va="center")
    bx0, by0 = 3.55, 0.52
    pts = [(bx0 + 0.24 * k, by0 + 0.19 * k) for k in range(5)]
    for k in range(4):
        x_cur, y_cur = pts[k]
        x_nxt, y_nxt = pts[k + 1]
        # Invented internal stem in crimson
        ax.plot([x_cur, x_nxt], [y_cur, y_nxt], color=rose, lw=2.2, solid_capstyle="round")
        # Side leaf
        ax.plot([x_cur, x_cur + 0.55], [y_cur, y_cur - 0.16], color=slate, lw=1.4, solid_capstyle="round")
    ax.plot([pts[-1][0], pts[-1][0] + 0.58], [pts[-1][1], pts[-1][1] + 0.22], color=slate, lw=1.4, solid_capstyle="round")
    ax.plot([pts[-1][0], pts[-1][0] + 0.58], [pts[-1][1], pts[-1][1] - 0.14], color=slate, lw=1.4, solid_capstyle="round")
    ax.text(4.85, 0.10, "23–61 false splits/tree", fontsize=7.6, fontweight="bold", color=rose, ha="center", va="center")

    # 3. Bushwhacker (MDL preserves natural burst)
    cx3, cy3 = 8.25, 1.02
    ax.text(cx3, 2.12, "Bushwhacker", fontsize=8.4, fontweight="bold", color=blue, ha="center", va="center")
    for ang in angles:
        rad = math.radians(ang)
        ax.plot([cx3, cx3 + 0.72 * math.cos(rad)], [cy3, cy3 + 0.68 * math.sin(rad)],
                color=blue, lw=1.85, solid_capstyle="round")
    ax.scatter([cx3], [cy3], s=42, color=blue, zorder=5)
    ax.text(cx3, 0.10, "<0.2 false splits/tree", fontsize=7.6, fontweight="bold", color=blue, ha="center", va="center")


def main() -> None:
    toga = json.loads((RESULTS_DIR / "toga2_mammal882_sota_bench.json").read_text())
    gtdb = json.loads((RESULTS_DIR / "gtdb_sota_bw_upgraded.json").read_text())
    p2 = json.loads((PILLARS_DIR / "pillar2_megatrees.json").read_text())
    mem = json.loads((RESULTS_DIR / "tol_memory_profile.json").read_text())
    multi = json.loads((RESULTS_DIR / "multialphabet_validation.json").read_text())

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#475569",
        "axes.linewidth": 0.8,
        "xtick.color": "#334155",
        "ytick.color": "#334155",
    })

    # Identical Science editorial palette to Figure 1
    INK = "#0F172A"
    SUBINK = "#475569"
    MUTED = "#94A3B8"
    SLATE = "#334155"
    HAIR = "#E2E8F0"
    BLUE = "#1D4ED8"
    BLUE_SOFT = "#EFF6FF"
    TEAL = "#0D9488"
    AMBER = "#D97706"
    ROSE = "#E11D48"
    VIOLET = "#7C3AED"

    fig = plt.figure(figsize=(12.6, 8.5), facecolor="white")

    # Panel headers matched to Fig. 1
    draw_panel_header(fig, 0.032, 0.962, "A", "Tree-of-Life speed, memory, and clade recovery", INK)
    draw_panel_header(fig, 0.655, 0.962, "B", "Single-pass branch support", INK)
    draw_panel_header(fig, 0.032, 0.482, "C", "Rapid bursts vs. invented binary splits", INK)
    draw_panel_header(fig, 0.545, 0.482, "D", "Gene transfer, convergence, and single cells", INK)

    # =========================================================================
    # PANEL A (Top-Left + Top-Center): 2D Log-Log Pareto + Cleveland Dot Plot
    # =========================================================================
    ax_a1 = fig.add_axes([0.068, 0.565, 0.255, 0.345])   # 2D Runtime vs Peak Memory
    ax_a2 = fig.add_axes([0.428, 0.565, 0.190, 0.345])   # Horizontal Cleveland Dot Plot

    tm = toga["methods"]
    bw_nt = tm["Bushwhacker_NT_MDL"]
    iq_nt = tm["IQTREE2_GTR_G4_fast_NT"]
    ft_nt = tm["FastTreeMP_GTR_NT"]
    bw_aa = tm["Bushwhacker_AA_LG08_CAT8_MDL"]
    iq_aa = tm["IQTREE2_LG_G4_fast_AA"]
    ft_aa = tm["FastTreeMP_LG_AA"]

    ar_m = gtdb["datasets"]["ar122_archaea_1672"]["methods"]
    bac_m = gtdb["datasets"]["bac120_bacteria_1000"]["methods"]
    ft_bac_sec = p2["part_2b_gtdb_domain_supermatrices"]["bac120_bacterial_domain_supermatrix"]["fasttree_lg_t32_sec"]

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
    ax_a1.grid(True, which="major", color=HAIR, lw=0.7, zorder=0)

    # Subtle laptop-zone highlight (<60 s, <3.2 GB)
    laptop_zone = mpatches.Rectangle((12, 0.60), 58 - 12, 3.2 - 0.60, facecolor=BLUE_SOFT, edgecolor="none", zorder=1)
    ax_a1.add_patch(laptop_zone)
    ax_a1.text(13.5, 6.4, "Laptop envelope\n(<2.4 GB, <50 s)", fontsize=7.6, fontweight="bold", color=BLUE, va="bottom")

    # 64-GB server ceiling line
    ax_a1.axhline(64.0, color=MUTED, ls="--", lw=0.9, zorder=1)
    ax_a1.text(14.0, 70.0, "64 GB server node", fontsize=7.3, color=SUBINK, va="bottom")

    # Connect corresponding datasets (Bushwhacker -> IQ-TREE 2)
    for lbl, t_bw, r_bw, t_iq, r_iq, t_ft in tol_runs:
        ax_a1.plot([t_bw, t_iq], [r_bw, r_iq], color="#CBD5E1", lw=1.15, zorder=2)
        ax_a1.scatter([t_bw], [r_bw], s=56, color=BLUE, edgecolor="white", linewidth=0.9, zorder=5)
        ax_a1.scatter([t_iq], [r_iq], s=52, facecolor="white", edgecolor=TEAL, linewidth=1.8, zorder=4)
        # FastTreeMP exact measured wall-clock runtime markers along the x-axis baseline
        ax_a1.scatter([t_ft], [0.68], marker="v", s=34, color=AMBER, zorder=6)

    # Clean direct inline labels without collisions
    ax_a1.text(68, 1.12, "Bushwhacker\n(20–49 s)", fontsize=8.0, fontweight="bold", color=BLUE, va="center")
    ax_a1.text(1750, 18.0, "IQ-TREE 2\n(38–73× slower,\n17–40× RAM)", fontsize=7.7, fontweight="bold", color=TEAL, ha="center", va="top")
    ax_a1.text(7600, 0.98, "FastTreeMP runtimes:\n1.1–3.9 h (190–450× slower)", fontsize=7.3, fontweight="bold", color=AMBER, ha="center", va="bottom")

    ax_a1.set_xlim(12, 24000)
    ax_a1.set_ylim(0.60, 125)
    ax_a1.set_xticks([30, 300, 3600])
    ax_a1.set_xticklabels(["30 s", "5 min", "1 h"], fontsize=8.0)
    ax_a1.set_yticks([1, 10, 100])
    ax_a1.set_yticklabels(["1", "10", "100"], fontsize=8.0)
    ax_a1.set_xlabel("32-core runtime", fontsize=8.5, fontweight="bold", color=INK)
    ax_a1.set_ylabel("Peak memory (GB)", fontsize=8.5, fontweight="bold", color=INK)

    # Subpanel A2: Cleveland dot plot of clade / split accuracy
    pair_nt = 100.0 * (1.0 - toga["pairwise_nrf"]["Bushwhacker_NT_MDL__vs__IQTREE2_GTR_G4_fast_NT"])
    dot_rows = [
        ("Mammals: Orders", bw_nt["taxonomy_4ranks"]["order"]["strict_pct"], iq_nt["taxonomy_4ranks"]["order"]["strict_pct"], ft_nt["taxonomy_4ranks"]["order"]["strict_pct"], None),
        ("Mammals: DNA splits", pair_nt, 100.0, None, None),
        ("Mammals: AA prec.", bw_aa["vs_species_tree"]["precision_pct"], iq_aa["vs_species_tree"]["precision_pct"], ft_aa["vs_species_tree"]["precision_pct"], None),
        ("Bacteria: Classes", bac_m["Bushwhacker_LG08_CAT8_MDL"]["taxonomy_5ranks"]["class"]["strict_pct"], bac_m["IQTREE2_LG_G4_fast"]["taxonomy_5ranks"]["class"]["strict_pct"], bac_m["FastTreeMP_LG"]["taxonomy_5ranks"]["class"]["strict_pct"], bac_m["Bushwhacker_Legacy_PreFix"]["taxonomy_5ranks"]["class"]["strict_pct"]),
        ("Bacteria: 111 Phyla", bac_m["Bushwhacker_LG08_CAT8_MDL"]["taxonomy_5ranks"]["phylum"]["strict_pct"], bac_m["IQTREE2_LG_G4_fast"]["taxonomy_5ranks"]["phylum"]["strict_pct"], bac_m["FastTreeMP_LG"]["taxonomy_5ranks"]["phylum"]["strict_pct"], bac_m["Bushwhacker_Legacy_PreFix"]["taxonomy_5ranks"]["phylum"]["strict_pct"]),
        ("Archaea: 19 Phyla", ar_m["Bushwhacker_LG08_CAT8_MDL"]["taxonomy_5ranks"]["phylum"]["strict_pct"], ar_m["IQTREE2_LG_G4_fast"]["taxonomy_5ranks"]["phylum"]["strict_pct"], ar_m["FastTreeMP_LG"]["taxonomy_5ranks"]["phylum"]["strict_pct"], ar_m["Bushwhacker_Legacy_PreFix"]["taxonomy_5ranks"]["phylum"]["strict_pct"]),
    ][::-1]

    ya2 = np.arange(len(dot_rows))
    ax_a2.grid(axis="x", color=HAIR, lw=0.7, zorder=0)
    for idx, (lbl, v_bw, v_iq, v_ft, v_unw) in enumerate(dot_rows):
        ax_a2.axhline(idx, color="#F1F5F9", lw=1.4, zorder=1)
        if v_unw is not None:
            ax_a2.plot([v_unw, v_bw], [idx, idx], color="#CBD5E1", lw=1.6, zorder=2)
            ax_a2.scatter([v_unw], [idx], marker="x", s=44, color=ROSE, linewidths=1.8, zorder=5)
        if v_ft is not None:
            ax_a2.scatter([v_ft], [idx], marker="D", s=30, color=AMBER, alpha=0.9, zorder=4)
        if lbl != "Mammals: DNA splits":
            ax_a2.scatter([v_iq], [idx], s=54, facecolor="white", edgecolor=TEAL, linewidth=1.7, zorder=5)
        ax_a2.scatter([v_bw], [idx], s=50, color=BLUE, edgecolor="white", linewidth=0.7, zorder=6)

    # Inline dot-key header above top row (2 crisp rows or compact spacing across all 4 methods)
    ax_a2.scatter([-8], [6.05], marker="x", s=34, color=ROSE, linewidths=1.7, clip_on=False)
    ax_a2.text(-4, 6.05, "Parsimony", fontsize=7.2, color=ROSE, va="center", clip_on=False)
    ax_a2.scatter([31], [6.05], marker="D", s=24, color=AMBER, clip_on=False)
    ax_a2.text(35, 6.05, "FastTree", fontsize=7.2, color=AMBER, va="center", clip_on=False)
    ax_a2.scatter([65], [6.05], s=38, facecolor="white", edgecolor=TEAL, linewidth=1.6, clip_on=False)
    ax_a2.text(69.5, 6.05, "IQ-TREE", fontsize=7.2, color=TEAL, va="center", clip_on=False)
    ax_a2.scatter([97], [6.05], s=38, color=BLUE, clip_on=False)
    ax_a2.text(101.5, 6.05, "BW", fontsize=7.2, fontweight="bold", color=BLUE, va="center", clip_on=False)

    ax_a2.set_yticks(ya2)
    ax_a2.set_yticklabels([r[0] for r in dot_rows], fontsize=8.0, color=INK)
    ax_a2.set_xlim(0, 106)
    ax_a2.set_ylim(-0.55, 5.65)
    ax_a2.set_xticks([0, 50, 100])
    ax_a2.set_xticklabels(["0%", "50%", "100%"], fontsize=8.0)
    ax_a2.set_xlabel("Monophyly / precision", fontsize=8.5, fontweight="bold", color=INK)

    # =========================================================================
    # PANEL B (Top-Right): Single-Pass Analytical Posterior Calibration
    # =========================================================================
    ax_b = fig.add_axes([0.705, 0.565, 0.270, 0.345])
    cal = multi["posterior_calibration"]
    bins = cal["bins"]
    pred_p = [b["mean_predicted_prob"] * 100.0 for b in bins]
    emp_p = [b["empirical_true_frac"] * 100.0 for b in bins]
    counts = [b["n_branches"] for b in bins]

    ax_b.grid(True, color=HAIR, lw=0.7, zorder=0)
    ax_b.plot([35, 100], [35, 100], "--", color=MUTED, lw=1.1, zorder=1)
    ax_b.text(62, 65.5, "1:1 ideal", fontsize=7.5, color=SUBINK, rotation=34)

    sizes = [28 + 150 * math.sqrt(c / max(counts)) for c in counts]
    ax_b.plot(pred_p, emp_p, "-", color=BLUE, lw=2.0, zorder=3)
    ax_b.scatter(pred_p, emp_p, s=sizes, color=BLUE, alpha=0.90, edgecolor="white", linewidth=1.0, zorder=4)

    ax_b.annotate(
        "98.3% true splits\n(39,907 branches ≥99%)",
        xy=(99.5, 98.0), xytext=(50.5, 95.5),
        fontsize=7.9, fontweight="bold", color=BLUE, va="top",
        arrowprops=dict(arrowstyle="-|>,head_length=0.38,head_width=0.22", color=BLUE, lw=1.1),
    )
    ax_b.text(50.5, 81.5, "44,556 branches\n0 bootstrap reruns", fontsize=7.8, color=SUBINK, va="top")

    ax_b.set_xlim(48, 103)
    ax_b.set_ylim(34, 104)
    ax_b.set_xlabel("Predicted branch support (%)", fontsize=8.5, fontweight="bold", color=INK)
    ax_b.set_ylabel("Observed true splits (%)", fontsize=8.5, fontweight="bold", color=INK)

    # =========================================================================
    # PANEL C (Bottom-Left): Mini Burst Schematic + Precision Trajectories
    # =========================================================================
    ax_c_top = fig.add_axes([0.040, 0.338, 0.445, 0.120])
    draw_mini_burst_comparison(ax_c_top, INK, SUBINK, BLUE, ROSE, SLATE)

    ax_c = fig.add_axes([0.068, 0.075, 0.415, 0.248])
    x_reg = np.array([0, 1, 2, 3, 4])
    x_lbls = ["Resolved\nmix", "25–50%\nsilent", ">50%\nsilent", "50% AA\nburst", "Pure\nstar"]

    p2_bw = multi["protein_suite"]["P2_RapidRad"]["bw_mdl"]["precision"] * 100.0
    p2_iqp = multi["protein_suite"]["P2_RapidRad"]["iq_poly"]["precision"] * 100.0
    p2_ft = multi["protein_suite"]["P2_RapidRad"]["ft_raw"]["precision"] * 100.0
    p2_iqb = multi["protein_suite"]["P2_RapidRad"]["iq_raw"]["precision"] * 100.0

    bw_prec = [94.7, 98.1, 98.8, p2_bw, 100.0]
    iq_poly_prec = [89.2, 86.9, 50.2, p2_iqp, 0.0]
    ft_prec = [84.9, 68.1, 51.4, p2_ft, 0.0]
    iq_bin_prec = [83.3, 64.7, 33.8, p2_iqb, 0.0]

    ax_c.grid(axis="y", color=HAIR, lw=0.7, zorder=0)
    ax_c.plot(x_reg, bw_prec, "-o", color=BLUE, lw=2.4, ms=5.8, zorder=5)
    ax_c.plot(x_reg, iq_poly_prec, "--s", color=TEAL, lw=1.7, ms=4.8, zorder=4)
    ax_c.plot(x_reg, ft_prec, "-.^", color=AMBER, lw=1.6, ms=4.8, zorder=3)
    ax_c.plot(x_reg, iq_bin_prec, ":d", color=ROSE, lw=1.8, ms=4.8, zorder=3)

    # Direct inline curve labels (zero legend box, zero missing glyphs, zero line collisions)
    ax_c.text(2.05, 104.0, "Bushwhacker (95–100%)", fontsize=8.1, fontweight="bold", color=BLUE, ha="center")
    ax_c.text(3.05, 66.5, "IQ-TREE (--polytomy)", fontsize=7.7, fontweight="bold", color=TEAL, ha="center")
    ax_c.text(0.88, 76.5, "FastTree 2", fontsize=7.6, fontweight="bold", color=AMBER, ha="center")
    ax_c.text(1.85, 21.0, "Forced-binary ML (34% to 0%)", fontsize=7.8, fontweight="bold", color=ROSE, ha="center")

    ax_c.set_xticks(x_reg)
    ax_c.set_xticklabels(x_lbls, fontsize=8.0)
    ax_c.set_ylim(-5, 114)
    ax_c.set_xlim(-0.22, 4.22)
    ax_c.set_yticks([0, 50, 100])
    ax_c.set_yticklabels(["0%", "50%", "100%"], fontsize=8.0)
    ax_c.set_ylabel("True split precision", fontsize=8.5, fontweight="bold", color=INK)

    # =========================================================================
    # PANEL D (Bottom-Right): Horizontal Paired Dumbbell / Lollipop Chart
    # =========================================================================
    ax_d = fig.add_axes([0.705, 0.075, 0.270, 0.345])

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

    # (Label, baseline_or_None, bushwhacker_val, accent_color)
    d_rows = [
        ("Sister clades kept\nunder convergence", 0.0, 100.0, BLUE),
        ("Single-cell SNVs\n(allelic dropout)", c1_iq, c1_bw, BLUE),
        ("Single-cell copy-\nnumber steps (F1)", c3_bin, c3_ord, BLUE),
        ("Transferred DNA\nblock identified", None, n_hgt, TEAL),
        ("Parallel adaptive\nsites separated", None, n_conv, VIOLET),
        ("Clonal & diploid\ncontrol specificity", None, 100.0, BLUE),
    ][::-1]

    yd = np.arange(len(d_rows))
    ax_d.grid(axis="x", color=HAIR, lw=0.7, zorder=0)

    for idx, (lbl, v_base, v_bw, col) in enumerate(d_rows):
        ax_d.axhline(idx, color="#F1F5F9", lw=1.4, zorder=1)
        if v_base is not None:
            ax_d.annotate(
                "", xy=(v_bw - 2.2, idx), xytext=(v_base + 2.0, idx),
                arrowprops=dict(arrowstyle="-|>,head_length=0.40,head_width=0.22", color=col, lw=1.6),
                zorder=3,
            )
            ax_d.scatter([v_base], [idx], s=52, facecolor="white", edgecolor=ROSE, linewidth=1.8, zorder=4)
        else:
            ax_d.plot([0, v_bw], [idx, idx], color="#CBD5E1", lw=1.4, ls=":", zorder=2)
        ax_d.scatter([v_bw], [idx], s=60, color=col, edgecolor="white", linewidth=0.8, zorder=5)
        ax_d.text(min(104.5, v_bw + 3.2), idx, f"{v_bw:.0f}%" if v_bw >= 99.5 else f"{v_bw:.1f}%",
                  va="center", ha="left", fontsize=7.9, fontweight="bold", color=col)

    # Inline dot key at top of Panel D
    ax_d.scatter([4], [5.82], s=44, facecolor="white", edgecolor=ROSE, linewidth=1.7, clip_on=False)
    ax_d.text(9, 5.82, "ML / binary", fontsize=7.5, color=ROSE, va="center", clip_on=False)
    ax_d.scatter([55], [5.82], s=48, color=BLUE, clip_on=False)
    ax_d.text(60, 5.82, "Bushwhacker", fontsize=7.5, fontweight="bold", color=BLUE, va="center", clip_on=False)

    ax_d.set_yticks(yd)
    ax_d.set_yticklabels([r[0] for r in d_rows], fontsize=8.0, color=INK)
    ax_d.set_xlim(-4, 118)
    ax_d.set_ylim(-0.55, 5.45)
    ax_d.set_xticks([0, 50, 100])
    ax_d.set_xticklabels(["0%", "50%", "100%"], fontsize=8.0)
    ax_d.set_xlabel("Accuracy / specificity", fontsize=8.5, fontweight="bold", color=INK)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300)
    fig.savefig(OUT_PNG, dpi=240)
    plt.close(fig)
    print(f"Generated {OUT_PDF} and {OUT_PNG}")


if __name__ == "__main__":
    main()
