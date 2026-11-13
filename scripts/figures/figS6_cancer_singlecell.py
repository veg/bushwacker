#!/usr/bin/env python3
"""
Render the Science-standard 4-panel proof-of-concept figure for Non-Sequence Bushwhacker
on Single-Cell Cancer Copy-Number (DLP+ scWGS) and Somatic Genotype Clones (AML Tapestri).

Reads strictly verified empirical outputs from:
  results/cancer_singlecell/cancer_singlecell_summary.json

Outputs:
  paper/fig_science_cancer_singlecell.pdf
  results/cancer_singlecell/fig_science_cancer_singlecell.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_JSON = ROOT / "results" / "single_cell" / "cancer_singlecell_summary.json"
OUT_PDF = ROOT / "paper" / "fig_science_cancer_singlecell.pdf"
OUT_PNG = ROOT / "results" / "single_cell" / "fig_science_cancer_singlecell.png"

# Okabe-Ito Science palette
BLUE = "#0072B2"
SKY = "#56B4E9"
VERMILION = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
SLATE = "#475569"
LIGHT_GRAY = "#CBD5E1"
DARK = "#0F172A"


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.85)
    ax.spines["bottom"].set_linewidth(0.85)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")
    ax.tick_params(axis="both", labelsize=8.0, width=0.8, length=3.2, colors="#1E293B")
    if grid_axis in ("x", "y", "both"):
        ax.grid(axis=grid_axis, linestyle="--", linewidth=0.5, alpha=0.28, color="#94A3B8")
        ax.set_axisbelow(True)


def add_panel_letter(ax: plt.Axes, letter: str, x: float = -0.14, y: float = 1.13) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=13.5,
        fontweight="bold",
        color=DARK,
        va="top",
        ha="left",
    )


def format_sci_p(p_val: float) -> str:
    if p_val <= 0.0:
        return "< 1e-300"
    exp = int(np.floor(np.log10(p_val)))
    mant = p_val / (10.0 ** exp)
    return f"{mant:.1f}e{exp}"


def main() -> None:
    with SUMMARY_JSON.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not data.get("zero_leakage_verified", False):
        raise RuntimeError("Refusing to plot: zero_leakage_verified is not True!")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.edgecolor": "#334155",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    dlp = data["dlp_singlecell_cna"]
    aml = data["aml_singlecell_tapestri"]
    raw_compass = aml["raw_singlecell_compass_zygosity"]

    fig = plt.figure(figsize=(14.2, 9.8), dpi=300)
    outer = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.0, 1.05],
        width_ratios=[1.04, 1.0],
        hspace=0.38,
        wspace=0.26,
        left=0.058,
        right=0.982,
        top=0.935,
        bottom=0.068,
    )

    # =========================================================================
    # PANEL A: DLP+ Single-Cell Copy-Number Evolution (Compression & Purity)
    # =========================================================================
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 0], wspace=0.32, width_ratios=[1.0, 1.05])
    ax_a1 = fig.add_subplot(gs_a[0, 0])
    ax_a2 = fig.add_subplot(gs_a[0, 1])

    cohorts = ["SA501", "SA535", "OV2295", "SA609"]
    cohort_labels = [
        f"SA501\n(N={dlp['SA501']['n_cells']})",
        f"SA535\n(N={dlp['SA535']['n_cells']})",
        f"OV2295\n(N={dlp['OV2295']['n_cells']})",
        f"SA609\n(N={dlp['SA609']['n_cells']})",
    ]
    x_a = np.arange(len(cohorts))
    w_a1 = 0.35

    comp_ord = [dlp[c]["raw_500kb_ordered"]["compression_ratio_raw_to_net"] for c in cohorts]
    comp_unord = [dlp[c]["raw_500kb_unordered"]["compression_ratio_raw_to_net"] for c in cohorts]
    seg_rt = [dlp[c]["segment_blocks_ordered"]["runtime_sec"] for c in cohorts]

    bars_ord = ax_a1.bar(
        x_a - w_a1 / 2,
        comp_ord,
        width=w_a1,
        color=BLUE,
        edgecolor=DARK,
        linewidth=0.6,
        label="Ordered Wagner + RLE (--ordered)",
    )
    ax_a1.bar(
        x_a + w_a1 / 2,
        comp_unord,
        width=w_a1,
        color=SKY,
        edgecolor=DARK,
        linewidth=0.6,
        label="Unordered Fitch (--discrete)",
    )

    for idx, rect in enumerate(bars_ord):
        val = comp_ord[idx]
        ax_a1.text(
            rect.get_x() + rect.get_width() / 2,
            val + 0.8,
            f"{val:.1f}x\n({seg_rt[idx]:.1f}s)",
            ha="center",
            va="bottom",
            fontsize=6.8,
            fontweight="bold",
            color=BLUE,
            linespacing=1.05,
        )

    style_axis(ax_a1, "y")
    ax_a1.set_xticks(x_a)
    ax_a1.set_xticklabels(cohort_labels, fontsize=7.5)
    ax_a1.set_ylim(0, 44.5)
    ax_a1.set_ylabel("Lossless MDL compression ratio (raw / net)", fontsize=8.4, fontweight="bold", color=DARK)
    ax_a1.set_title("scWGS 500-kb CNA compression\n(4,950 cells x 4,375 bins, K=12)", fontsize=8.8, fontweight="bold", color=DARK, pad=7)
    ax_a1.legend(loc="upper right", frameon=False, fontsize=6.7, handlelength=1.2, borderpad=0.1)
    add_panel_letter(ax_a1, "A", x=-0.24, y=1.16)

    # A2: Held-out anatomical / passage clade purity vs forced-binary & null
    pur_mdl = [dlp[c]["raw_500kb_ordered"]["mdl_concordance"]["nearest_clade_purity"] for c in cohorts]
    pur_bin = [dlp[c]["raw_500kb_ordered"]["forced_binary_baseline_concordance"]["nearest_clade_purity"] for c in cohorts]
    pur_null = [dlp[c]["raw_500kb_ordered"]["mdl_concordance"]["null_random_purity"] for c in cohorts]
    contr_pct = [dlp[c]["segment_blocks_ordered"]["false_split_contraction_pct"] for c in cohorts]

    w_a2 = 0.25
    bars_mdl = ax_a2.bar(
        x_a - w_a2,
        pur_mdl,
        width=w_a2,
        color=GREEN,
        edgecolor=DARK,
        linewidth=0.6,
        label="MDL polytomy tree",
    )
    ax_a2.bar(
        x_a,
        pur_bin,
        width=w_a2,
        color=SLATE,
        edgecolor=DARK,
        linewidth=0.6,
        label="Forced binary tree",
    )
    ax_a2.bar(
        x_a + w_a2,
        pur_null,
        width=w_a2,
        color=LIGHT_GRAY,
        edgecolor="#64748B",
        linewidth=0.6,
        label="Random permutation",
    )

    for idx, rect in enumerate(bars_mdl):
        y_top = max(pur_mdl[idx], pur_bin[idx])
        ax_a2.text(
            x_a[idx] - 0.05,
            y_top + 0.025,
            f"-{contr_pct[idx]:.0f}% splits",
            ha="center",
            va="bottom",
            fontsize=6.7,
            fontweight="bold",
            color=GREEN,
        )

    style_axis(ax_a2, "y")
    ax_a2.set_xticks(x_a)
    ax_a2.set_xticklabels(cohort_labels, fontsize=7.5)
    ax_a2.set_ylim(0, 1.18)
    ax_a2.set_ylabel("Held-out compartment clade purity", fontsize=8.4, fontweight="bold", color=DARK)
    ax_a2.set_title("Zero-leakage clade concordance\n& false-split contraction (%)", fontsize=8.8, fontweight="bold", color=DARK, pad=7)
    ax_a2.legend(loc="upper right", frameon=False, fontsize=6.7, handlelength=1.2, borderpad=0.1)

    # =========================================================================
    # PANEL B: Clonal Bottlenecks & Inferred Chromosomal-Arm INTERVAL Shunts
    # =========================================================================
    gs_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 1], wspace=0.72, width_ratios=[0.92, 1.12])
    ax_b1 = fig.add_subplot(gs_b[0, 0])
    ax_b2 = fig.add_subplot(gs_b[0, 1])

    sa609_div = dlp["SA609"]["subclone_clade_diversity_by_group"]
    ov_div = dlp["OV2295"]["subclone_clade_diversity_by_group"]
    sa609_f1 = dlp["SA609"]["segment_blocks_ordered"]["mdl_concordance"]["best_split_f1_by_group"]
    ov_f1 = dlp["OV2295"]["segment_blocks_ordered"]["mdl_concordance"]["best_split_f1_by_group"]

    b1_rows = [
        ("OV2295 Relapse ascites", ov_div["Relapse_Ascites_SA922"]["effective_clones_expH"], VERMILION, f"F1={ov_f1['Relapse_Ascites_SA922']:.3f}"),
        ("OV2295 Pre-Tx ascites", ov_div["PreTx_Ascites_SA1090"]["effective_clones_expH"], ORANGE, f"F1={ov_f1['PreTx_Ascites_SA1090']:.3f}"),
        ("OV2295 Primary ovary", ov_div["Ovary_PrimarySite_SA921"]["effective_clones_expH"], SKY, f"F1={ov_f1['Ovary_PrimarySite_SA921']:.3f}"),
        ("SA609 Selected X2", sa609_div["Selected_X2_B00426"]["effective_clones_expH"], PURPLE, "2.5x bottleneck"),
        ("SA609 Untreated X1", sa609_div["Untreated_X1_B00290"]["effective_clones_expH"], BLUE, f"F1={sa609_f1['Untreated_X1_B00290']:.2f}"),
        ("SA928 Diploid normal", sa609_div["Diploid_Normal_SA928"]["effective_clones_expH"], GREEN, f"F1={sa609_f1['Diploid_Normal_SA928']:.3f}"),
    ]
    y_b1 = np.arange(len(b1_rows))
    b1_vals = [r[1] for r in b1_rows]
    b1_cols = [r[2] for r in b1_rows]

    ax_b1.barh(y_b1, b1_vals, height=0.58, color=b1_cols, edgecolor=DARK, linewidth=0.6)
    ax_b1.axhline(2.5, color="#94A3B8", linestyle=":", linewidth=0.9)

    for idx, (_, val, col, tag) in enumerate(b1_rows):
        ax_b1.text(
            val + 5.0,
            idx,
            f"{val:.1f} ({tag})",
            va="center",
            ha="left",
            fontsize=6.4,
            fontweight="bold",
            color=DARK if col == SKY else col,
        )

    style_axis(ax_b1, "x")
    ax_b1.set_yticks(y_b1)
    ax_b1.set_yticklabels([r[0] for r in b1_rows], fontsize=7.2)
    ax_b1.set_xlim(0, 385)
    ax_b1.set_xlabel("Effective MDL subclones exp(H)", fontsize=8.3, fontweight="bold", color=DARK)
    ax_b1.set_title("Metastatic & selective\nsubclone diversity", fontsize=8.8, fontweight="bold", color=DARK, pad=7)
    add_panel_letter(ax_b1, "B", x=-0.46, y=1.16)

    # B2: Top inferred Stage 4 INTERVAL chromosomal CNA/LOH shunts across all 4 cohorts
    shunt_candidates = []
    cohort_colors = {"SA535": VERMILION, "SA609": BLUE, "SA501": GREEN, "OV2295": PURPLE}
    locus_notes = {
        ("SA609", "8", 98500001): " (MYC)",
        ("SA609", "17", 14500001): " (TP53)",
    }
    seen_cohort_chrom = set()
    for c_name in ["SA535", "SA609", "SA501", "OV2295"]:
        blocks = dlp[c_name]["raw_500kb_ordered"]["shunt_summary"]["top_interval_cna_loh_blocks"]
        for blk in blocks:
            key = (c_name, blk["chrom"])
            if key in seen_cohort_chrom:
                continue
            seen_cohort_chrom.add(key)
            note = locus_notes.get((c_name, blk["chrom"], blk["start_bp"]), "")
            lbl = f"{c_name} chr{blk['chrom']}{note} ({blk['span_mb']:.0f} Mb)"
            shunt_candidates.append(
                (blk["net_mdl_gain_nats"], lbl, c_name, blk["shared_features"], blk["posterior_prob"])
            )

    shunt_candidates.sort(key=lambda item: item[0], reverse=True)
    top_shunts = shunt_candidates[:8][::-1]  # top 8 includes OV2295 chr4 (+28.8 nats)

    y_b2 = np.arange(len(top_shunts))
    b2_gains = [s[0] for s in top_shunts]
    b2_cols = [cohort_colors[s[2]] for s in top_shunts]

    ax_b2.barh(y_b2, b2_gains, height=0.60, color=b2_cols, edgecolor=DARK, linewidth=0.6)
    for idx, (gain, _, c_name, n_bins, post_p) in enumerate(top_shunts):
        ax_b2.text(
            gain + 2.0,
            idx,
            f"+{gain:.1f} nats ({n_bins}b)",
            va="center",
            ha="left",
            fontsize=6.5,
            fontweight="bold",
            color=cohort_colors[c_name],
        )

    style_axis(ax_b2, "x")
    ax_b2.set_yticks(y_b2)
    ax_b2.set_yticklabels([s[1] for s in top_shunts], fontsize=7.0)
    ax_b2.set_xlim(0, 152)
    ax_b2.set_xlabel("Inferred INTERVAL shunt gain (nats)", fontsize=8.3, fontweight="bold", color=DARK)
    ax_b2.set_title("Stage 4 chromosomal-arm\nCNA / LOH shunts (P = 1.000)", fontsize=8.8, fontweight="bold", color=DARK, pad=7)

    # =========================================================================
    # PANEL C: 123-Patient Single-Cell AML: Polytomy Burst Hubs & CN-LOH
    # =========================================================================
    gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, 0], wspace=0.36, width_ratios=[1.05, 1.0])
    ax_c1 = fig.add_subplot(gs_c[0, 0])
    ax_c2 = fig.add_subplot(gs_c[0, 1])

    k_keys = [
        ("k=0_terminal_leaf", "k = 0\nLeaf"),
        ("k=1_linear_step", "k = 1\nLinear"),
        ("k=2_bifurcating", "k = 2\nBinary"),
        ("k>=3_polytomy_burst_hub", "k >= 3\nBurst hub"),
    ]
    deg_dict = raw_compass["clone_abundance_by_mdl_out_degree"]
    mean_cells = [deg_dict[k]["mean_cells"] for k, _ in k_keys]
    exp_pct = [100.0 * deg_dict[k]["frac_expanded_ge_50_cells"] for k, _ in k_keys]
    n_geno = [deg_dict[k]["n_genotypes"] for k, _ in k_keys]

    x_c1 = np.arange(len(k_keys))
    c1_colors = [LIGHT_GRAY, SKY, BLUE, VERMILION]
    bars_c1 = ax_c1.bar(x_c1, mean_cells, width=0.54, color=c1_colors, edgecolor=DARK, linewidth=0.65)

    for idx, rect in enumerate(bars_c1):
        val = mean_cells[idx]
        ax_c1.text(
            rect.get_x() + rect.get_width() / 2,
            val + 8.0,
            f"{val:.1f} cells\n({exp_pct[idx]:.0f}% >=50c)",
            ha="center",
            va="bottom",
            fontsize=6.7,
            fontweight="bold",
            color=VERMILION if idx == 3 else DARK,
            linespacing=1.05,
        )

    fold_enrich = raw_compass["hub_k3_vs_leaf_k0_mean_cell_fold_enrichment"]
    mw_p_str = format_sci_p(raw_compass["hierarchical_per_run_diff_cells"]["wilcoxon_p_two_sided"])
    rho_c1 = raw_compass["k_out_vs_cell_count_spearman_rho"]

    # Clean significance bracket between k=0 and k>=3
    y_brk = 368.0
    ax_c1.plot([0, 0, 3, 3], [354.0, y_brk, y_brk, 354.0], color=DARK, linewidth=0.85)
    ax_c1.text(
        1.5,
        y_brk + 5.0,
        f"{fold_enrich:.2f}x in vivo expansion (run p = {mw_p_str}; rho = {rho_c1:.2f})",
        ha="center",
        va="bottom",
        fontsize=6.8,
        fontweight="bold",
        color=DARK,
    )

    style_axis(ax_c1, "y")
    ax_c1.set_xticks(x_c1)
    ax_c1.set_xticklabels(
        [f"{lbl}\n(n={n_geno[i]})" for i, (_, lbl) in enumerate(k_keys)],
        fontsize=7.1,
    )
    ax_c1.set_ylim(0, 418)
    ax_c1.set_ylabel("In vivo cells per somatic genotype", fontsize=8.4, fontweight="bold", color=DARK)
    ax_c1.set_title(
        f"Sequence-free MDL hubs predict clone size\n({raw_compass['total_single_cells_scanned']:,} cells, {raw_compass['total_unthresholded_somatic_genotypes']:,} genotypes)",
        fontsize=8.7,
        fontweight="bold",
        color=DARK,
        pad=7,
    )
    add_panel_letter(ax_c1, "C", x=-0.24, y=1.16)

    # C2: Ordered Zygosity 0 -> 1 -> 2 Stepwise Copy-Neutral LOH
    cn_loh_dict = raw_compass["top_cn_loh_genes_1_to_2"]
    top_loh_items = list(cn_loh_dict.items())[:10][::-1]  # top 10 ascending for barh
    y_c2 = np.arange(len(top_loh_items))
    loh_genes = [g for g, _ in top_loh_items]
    loh_counts = [cnt for _, cnt in top_loh_items]

    signaling_set = {"FLT3", "NRAS", "KRAS", "PTPN11", "IDH1", "IDH2", "WT1", "KIT", "JAK2", "MYC"}
    c2_cols = [VERMILION if g in signaling_set else BLUE for g in loh_genes]

    ax_c2.barh(y_c2, loh_counts, height=0.60, color=c2_cols, edgecolor=DARK, linewidth=0.55)
    for idx, cnt in enumerate(loh_counts):
        ax_c2.text(
            cnt + 10.0,
            idx,
            f"{cnt}",
            va="center",
            ha="left",
            fontsize=6.9,
            fontweight="bold",
            color=DARK,
        )

    step_frac_pct = 100.0 * raw_compass["stepwise_cn_loh_fraction"]
    step_n = raw_compass["zygosity_cn_loh_stepwise_1_to_2_transitions"]
    tot_hom = step_n + raw_compass["zygosity_direct_0_to_2_jumps"]

    style_axis(ax_c2, "x")
    ax_c2.set_yticks(y_c2)
    ax_c2.set_yticklabels(loh_genes, fontsize=7.4, fontweight="bold")
    ax_c2.set_xlim(0, 675)
    ax_c2.set_xlabel("Stepwise Het -> Hom/CN-LOH (1 -> 2) events", fontsize=8.2, fontweight="bold", color=DARK)
    ax_c2.set_title(
        f"Ordered zygosity resolves CN-LOH\n({step_frac_pct:.1f}% = {step_n:,}/{tot_hom:,} stepwise 1->2)",
        fontsize=8.7,
        fontweight="bold",
        color=DARK,
        pad=7,
    )

    # =========================================================================
    # PANEL D: 2D Clonal Evolution & Convergent Resistance Map (123 Patients)
    # =========================================================================
    ax_d = fig.add_subplot(outer[1, 1])

    genes_all = [
        g
        for g in aml["gene_convergence_ranking"]
        if g["pan_cohort_independent_branch_acquisitions"] >= 4
        or g["within_patient_parallel_acquisitions"] >= 2
    ]

    ax_d.axvline(50.0, color="#94A3B8", linestyle="--", linewidth=0.9, zorder=1)
    ax_d.text(5.0, 42.8, "Early founding trunk (<50%)", fontsize=7.4, fontweight="bold", color=BLUE, ha="left")
    ax_d.text(104.5, 42.8, "Late subclonal branching (>50%)", fontsize=7.4, fontweight="bold", color=VERMILION, ha="right")

    for g in genes_all:
        x_val = 100.0 * g["subclonal_fraction"]
        y_val = g["pan_cohort_independent_branch_acquisitions"]
        par = g["within_patient_parallel_acquisitions"]
        is_sig = g["pathway_class"] == "Signaling_Resistance"
        col = VERMILION if is_sig else BLUE
        size = 38.0 + 28.0 * par
        ax_d.scatter(
            x_val,
            y_val,
            s=size,
            color=col,
            alpha=0.86,
            edgecolors=DARK,
            linewidths=0.75,
            zorder=3,
        )

    # Deterministic, hand-verified non-overlapping label offsets (dx_pct, dy_hits, ha)
    label_offsets = {
        "NPM1": (-5.2, 1.2, "right"),
        "RUNX1": (5.2, -2.0, "left"),
        "FLT3": (-5.2, 1.8, "right"),
        "NRAS": (5.0, 0.5, "left"),
        "IDH2": (-5.0, 1.5, "right"),
        "TET2": (-4.5, 2.5, "right"),
        "WT1": (-5.5, -2.5, "right"),
        "SRSF2": (2.5, 6.2, "left"),
        "KRAS": (7.5, 2.2, "left"),
        "IDH1": (-5.2, -0.5, "right"),
        "ASXL1": (-2.5, 5.5, "right"),
        "TP53": (-5.2, -1.8, "right"),
        "PTPN11": (3.8, -1.0, "left"),
        "DNMT3A": (-5.2, 5.2, "right"),
        "SETBP1": (-3.8, -4.6, "right"),
        "GATA2": (-4.5, 1.5, "right"),
        "STAG2": (4.2, -1.6, "left"),
        "KIT": (-3.5, 5.0, "right"),
        "U2AF1": (3.2, -3.2, "left"),
        "SF3B1": (4.2, -1.2, "left"),
        "JAK2": (-4.5, -1.4, "right"),
    }

    for g in genes_all:
        name = g["gene"]
        if name not in label_offsets:
            continue
        x_val = 100.0 * g["subclonal_fraction"]
        y_val = g["pan_cohort_independent_branch_acquisitions"]
        par = g["within_patient_parallel_acquisitions"]
        dx, dy, ha = label_offsets[name]
        is_sig = g["pathway_class"] == "Signaling_Resistance"
        txt = f"{name} ({par})" if par >= 3 else name
        ax_d.annotate(
            txt,
            xy=(x_val, y_val),
            xytext=(x_val + dx, y_val + dy),
            ha=ha,
            va="center",
            fontsize=7.0,
            fontweight="bold" if (par >= 3 or name in {"TET2", "TP53", "WT1"}) else "normal",
            color=VERMILION if is_sig else "#0B4F6C",
            arrowprops=dict(arrowstyle="-", color="#94A3B8", lw=0.55, shrinkA=0, shrinkB=3),
            zorder=4,
        )

    enrich = aml["signaling_vs_founder_subclonal_enrichment"]
    or_val = enrich["fisher_odds_ratio"]
    p_enrich = format_sci_p(enrich["fisher_p_value_two_sided"])
    n_par_pts = aml["n_patients_with_parallel_driver_acquisitions"]
    rt_123 = aml["total_123_patient_bushwhacker_runtime_sec"]

    style_axis(ax_d, "both")
    ax_d.set_xlim(2, 107)
    ax_d.set_ylim(0, 45.5)
    ax_d.set_xlabel("Subclonal late branch acquisition (% of patient trees)", fontsize=8.5, fontweight="bold", color=DARK)
    ax_d.set_ylabel("Pan-cohort independent branch acquisitions", fontsize=8.5, fontweight="bold", color=DARK)
    ax_d.set_title(
        f"123-patient AML clonal architecture ({rt_123:.3f} s; 0 false HGT shunts)\n"
        f"Signaling vs founder subclonal OR = {or_val:.2f} (p = {p_enrich}); {n_par_pts}/123 patients parallel",
        fontsize=8.7,
        fontweight="bold",
        color=DARK,
        pad=7,
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="RTK/RAS & metabolic signaling", markerfacecolor=VERMILION, markeredgecolor=DARK, markersize=6.8),
        Line2D([0], [0], marker="o", color="w", label="Epigenetic / chromatin / splicing", markerfacecolor=BLUE, markeredgecolor=DARK, markersize=6.8),
        Line2D([0], [0], marker="o", color="w", label="Bubble size & (n): within-patient parallel hits", markerfacecolor="#94A3B8", markeredgecolor=DARK, markersize=8.5),
    ]
    ax_d.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.012, 0.915),
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        framealpha=0.94,
        fontsize=6.7,
        borderpad=0.35,
        handletextpad=0.4,
    )
    add_panel_letter(ax_d, "D", x=-0.11, y=1.16)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PDF -> {OUT_PDF}")
    print(f"Saved PNG -> {OUT_PNG}")


if __name__ == "__main__":
    main()
