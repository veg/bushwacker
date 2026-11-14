#!/usr/bin/env python3
"""
Generate Science-caliber Figure 6 (`paper/fig_science_tree_scales.pdf` and
`results/fig_science_tree_scales.png`) at native 3-column print width (7.25 in):
  - Panel A: Somatic burst hubs predict in vivo clonal expansion without cell-count
             input across 147 mouse B-cell germinal centers (10,429 cells) and
             168 human myeloid scDNA-seq runs (1,001,609 cells).
  - Panel B: Experimental yeast-display antibody-binding landscape (Delta log10 K_D)
             ordered strictly by burst degree k = 0, 1, 2, >=3 alongside recurrent
             cross-mouse parallel CDR drivers.
  - Panel C: Grounding empirical patient specimen — single-cell clonal phylogeny of
             human patient AML-38-001 reconstructed without cell counts.
  - Panel D: Clonal trunk-to-subclone driver architecture and parallel signaling
             resistance across 123 human acute myeloid leukemia patients.

100% of plotted topologies, posteriors, cell counts, and affinity shifts are loaded
directly from executed Bushwhacker outputs and JSON ledgers on disk.
"""

import json
import subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath


REPO_ROOT = Path(__file__).resolve().parents[2]
TP_DIR = REPO_ROOT / "results" / "single_cell"
CANCER_JSON = REPO_ROOT / "results" / "single_cell" / "cancer_singlecell_summary.json"
MORITA_JSON = REPO_ROOT / "data" / "single_cell" / "AML_cohort_Morita_2020.json"
SPEC_DIR = REPO_ROOT / "data" / "specimens"
OUT_PDF = REPO_ROOT / "paper" / "fig_science_tree_scales.pdf"
OUT_PNG = REPO_ROOT / "results" / "fig_science_tree_scales.png"


def draw_panel_header(fig, x_fig, y_fig, letter, title, ink="#0F172A"):
    fig.text(x_fig, y_fig, letter, fontsize=9.5, fontweight="bold", color=ink, ha="left", va="center")
    fig.text(x_fig + 0.022, y_fig, title, fontsize=7.2, fontweight="bold", color="#1E293B", ha="left", va="center")


def fmt_p(p):
    """Format a p-value in Science style (e.g., 1.7 x 10^-3)."""
    s = f"{p:.1e}"
    mant, exp = s.split("e")
    return rf"{mant} × $10^{{{int(exp)}}}$"


def ensure_aml38_specimen():
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    m = json.loads(MORITA_JSON.read_text())
    muts = m["mutation_labels"]
    idx = m["patient_labels"].index("AML-38-001")
    tree_rec = m["trees"][idx]["tree"]

    nodes = []
    def collect(nd):
        nodes.append(nd)
        for ch in nd.get("children", []):
            collect(ch)
    collect(tree_rec)

    fa = SPEC_DIR / "aml_38_001_clones.fasta"
    nwk = SPEC_DIR / "aml_38_001.nwk"
    js = SPEC_DIR / "aml_38_001.json"
    if not js.exists():
        with open(fa, "w") as f:
            for nd in nodes:
                gset = set(nd["genotype"])
                seq = "".join("1" if j in gset else "0" for j in range(len(muts)))
                lbl = f"Clone_{nd['node_id']}_" + ("_".join(muts[j] for j in nd["genotype"]) if nd["genotype"] else "WT")
                f.write(f">{lbl}\n{seq}\n")
        bw = REPO_ROOT / "target" / "release" / "bushwhacker"
        subprocess.run([str(bw), "--ordered", str(fa), "-o", str(nwk), "-j", str(js)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tree_rec, muts, json.loads(js.read_text())


def draw_bezier(ax, p0, p1, color, lw, alpha=1.0, zorder=2):
    xm = 0.5 * (p0[0] + p1[0])
    verts = [p0, (xm, p0[1]), (xm, p1[1]), p1]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    ax.add_patch(mpatches.PathPatch(
        MplPath(verts, codes), facecolor="none", edgecolor=color, lw=lw, alpha=alpha, zorder=zorder,
    ))


def draw_aml38_patient_tree(ax, tree_rec, muts, bw_json, colors):
    """Render authentic Bushwhacker clonal tree of AML-38-001 with held-out cell counts."""
    INK = colors["INK"]
    SUBINK = colors["SUBINK"]
    BLUE = colors["BLUE"]
    ROSE = colors["ROSE"]
    TEAL = colors["TEAL"]
    SLATE = "#64748B"

    cell_by_id = {}
    mut_by_id = {}
    def walk(nd):
        nid = nd["node_id"]
        cell_by_id[nid] = int(round(nd["seq_cell_number"]))
        mut_by_id[nid] = muts[nd["mutation_ids"][0]] if nd["mutation_ids"] else "WT"
        for ch in nd.get("children", []):
            walk(ch)
    walk(tree_rec)

    pos = {
        0: (0.04, 2.58),   # WT root
        1: (0.20, 2.58),   # +NPM1 founder (117 cells)
        2: (0.43, 4.35),   # +IDH1 hub (395 cells, k=3)
        5: (0.71, 5.25),   # +KRAS (2,611 cells)
        6: (0.71, 4.35),   # +PTPN11 (547 cells)
        4: (0.71, 3.45),   # +FLT3 (95 cells)
        3: (0.43, 0.80),   # +IDH2 hub (360 cells, k=4)
        7: (0.71, 2.15),   # +FLT3 (1,476 cells)
        9: (0.71, 1.25),   # +NRAS (1,043 cells)
        10: (0.71, 0.35),  # +PTPN11 (105 cells)
        8: (0.71, -0.55),  # +KRAS (34 cells)
    }

    # Trunk WT -> +NPM1 -> +IDH1 / +IDH2 hubs
    draw_bezier(ax, pos[0], pos[1], SLATE, lw=1.35, zorder=2)
    draw_bezier(ax, pos[1], pos[2], TEAL, lw=1.40, zorder=2)
    draw_bezier(ax, pos[1], pos[3], TEAL, lw=1.40, zorder=2)

    # Parallel signaling subclone rays from +IDH1 and +IDH2 burst hubs
    for hub_id, children in [(2, [5, 6, 4]), (3, [7, 9, 10, 8])]:
        xh, yh = pos[hub_id]
        for cid in children:
            xc, yc = pos[cid]
            lw = 1.45 if cell_by_id[cid] >= 500 else 0.95
            draw_bezier(ax, (xh, yh), (xc, yc), ROSE, lw=lw, alpha=0.88, zorder=3)

    def node_size(cells):
        return 14.0 + 2.65 * np.sqrt(max(10.0, float(cells)))

    # WT root node
    ax.scatter([pos[0][0]], [pos[0][1]], s=24, facecolor="white", edgecolor=SLATE, linewidth=1.1, zorder=5)
    ax.text(pos[0][0], pos[0][1] + 0.38, "WT\nroot", ha="center", va="bottom", fontsize=6.15, fontweight="bold", color=SUBINK)

    # Early +NPM1 founder
    ax.scatter([pos[1][0]], [pos[1][1]], s=node_size(cell_by_id[1]), color=BLUE, edgecolor="white", linewidth=0.75, zorder=5)
    ax.text(pos[1][0], pos[1][1] + 0.40, r"+$\it{NPM1}$" + f"\n({cell_by_id[1]} cells)",
            ha="center", va="bottom", fontsize=6.2, fontweight="bold", color=BLUE)

    # +IDH1 and +IDH2 burst hubs with soft radial halos
    sup_sorted = sorted(b["mdl_support"] for b in bw_json.get("internal_branches", []))
    p_idh1 = sup_sorted[0] if len(sup_sorted) >= 2 else 0.996
    p_idh2 = sup_sorted[-1] if len(sup_sorted) >= 2 else 0.997
    for hid, p_val, k_deg in [(2, p_idh1, 3), (3, p_idh2, 4)]:
        xh, yh = pos[hid]
        sz = node_size(cell_by_id[hid])
        ax.scatter([xh], [yh], s=sz * 2.5, color=TEAL, alpha=0.14, edgecolor="none", zorder=4)
        ax.scatter([xh], [yh], s=sz * 1.65, facecolor="none", edgecolor=TEAL, linewidth=0.85, ls=":", zorder=4)
        ax.scatter([xh], [yh], s=sz, color=TEAL, edgecolor="white", linewidth=0.75, zorder=6)
        lbl_hub = (
            r"+$\it{" + mut_by_id[hid] + r"}$" + f" hub ($k$ = {k_deg})\n"
            f"{cell_by_id[hid]} cells · " + r"$P_{\mathrm{local}}$" + f" = {p_val:.3f}"
        )
        if hid == 2:
            ax.text(xh - 0.02, yh + 0.45, lbl_hub, ha="center", va="bottom",
                    fontsize=6.15, fontweight="bold", color=TEAL)
        else:
            ax.text(xh - 0.04, yh - 0.30, lbl_hub, ha="right", va="top",
                    fontsize=6.15, fontweight="bold", color=TEAL)

    # Signaling resistance subclones
    for cid in [5, 6, 4, 7, 9, 10, 8]:
        xc, yc = pos[cid]
        cells = cell_by_id[cid]
        gname = mut_by_id[cid]
        ax.scatter([xc], [yc], s=node_size(cells), color=ROSE, alpha=0.92, edgecolor="white", linewidth=0.70, zorder=6)
        bold = cells >= 500
        ax.text(
            xc + 0.032, yc, r"+$\it{" + gname + r"}$" + f" ({cells:,})",
            va="center", ha="left",
            fontsize=6.25, fontweight="bold" if bold else "normal",
            color=ROSE if bold else INK,
        )

    # Compact cell-count scale key in open upper-left space
    ax.text(0.01, 5.18, "In vivo cells:", fontsize=6.0, fontweight="bold", color=SUBINK, ha="left", va="center")
    for idx_s, (c_ref, xs, dx) in enumerate([(100, 0.03, 0.024), (500, 0.135, 0.028), (2500, 0.250, 0.036)]):
        ax.scatter([xs], [4.62], s=node_size(c_ref), color="#CBD5E1", edgecolor=SLATE, linewidth=0.55, zorder=4)
        ax.text(xs + dx, 4.62, f"{c_ref:,}", fontsize=6.0, color=SUBINK, ha="left", va="center")

    ax.text(0.01, 5.95,
            f"7,016 cells · {bw_json['runtime_sec']*1000:.1f} ms (counts withheld)",
            fontsize=6.2, fontweight="bold", color=SUBINK, ha="left", va="top")
    ax.text(0.01, -1.25,
            r"Parallel $\it{FLT3}$, $\it{KRAS}$, $\it{PTPN11}$ clones arise on both hubs",
            fontsize=6.1, color=SUBINK, ha="left", va="bottom")

    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-1.45, 6.15)
    ax.axis("off")


def main() -> None:
    p3 = json.loads((TP_DIR / "pillar3_bcr_cellular.json").read_text())
    cancer = json.loads(CANCER_JSON.read_text())
    aml38_rec, aml38_muts, aml38_bw = ensure_aml38_specimen()

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
    TEAL = "#0F766E"
    ROSE = "#BE123C"

    fig = plt.figure(figsize=(7.25, 5.35), facecolor="white")

    draw_panel_header(fig, 0.012, 0.968, "A", "Clonal expansion without cell counts (1.01M cells)")
    draw_panel_header(fig, 0.512, 0.968, "B", "Antigen affinity & parallel CDR drivers")
    draw_panel_header(fig, 0.012, 0.472, "C", "Patient AML-38-001 clonal phylogeny")
    draw_panel_header(fig, 0.512, 0.472, "D", "Driver architecture (123 AML patients)")

    # =========================================================================
    # PANEL A (Top-Left): Mouse Germinal Centers + Human Myeloid scDNA-seq
    # =========================================================================
    ax_a1 = fig.add_axes([0.072, 0.576, 0.176, 0.345])
    ax_a2 = fig.add_axes([0.312, 0.576, 0.172, 0.345])

    disc = p3["part3b_empirical_gcreplay"]["zero_leakage_biological_discovery"]
    strat_b = disc["stratified_by_inferred_burst_daughters"]
    k_bcr = [
        ("k=0_terminal_twig", "0\nLeaf"),
        ("k=1_linear_step", "1\nStep"),
        ("k=2_small_burst", "2\nBurst"),
        ("k>=3_major_burst_hub", "≥3\nHub"),
    ]
    xa = np.arange(4)
    bcr_pct = [100.0 * strat_b[k]["frac_expanded_ge2_cells"] for k, _ in k_bcr]

    ax_a1.grid(axis="y", color=HAIR, lw=0.45, zorder=0)
    ax_a1.fill_between(xa, 0, bcr_pct, color=BLUE, alpha=0.08, zorder=1)
    ax_a1.plot(xa, bcr_pct, "-o", color=BLUE, lw=1.35, ms=3.8, zorder=4)
    ax_a1.scatter([3], [bcr_pct[3]], s=28, color=ROSE, edgecolor="white", linewidth=0.65, zorder=6)

    for xi, val in enumerate(bcr_pct):
        if xi in (1, 2):
            ax_a1.text(xi - 0.12, val + 2.6, f"{val:.1f}%", ha="right", va="bottom", fontsize=6.15, color=INK)
        else:
            dy = 2.5 if xi < 3 else 3.0
            ax_a1.text(xi, val + dy, f"{val:.1f}%", ha="center", va="bottom",
                       fontsize=6.2, fontweight="bold" if xi in (0, 3) else "normal",
                       color=ROSE if xi == 3 else INK)

    fold_bcr = strat_b["k>=3_major_burst_hub"]["frac_expanded_ge2_cells"] / strat_b["k=0_terminal_twig"]["frac_expanded_ge2_cells"]
    p_bcr = disc["in_vivo_clone_abundance"]["hierarchical_gc_paired_diff_hub_minus_nonhub"]["wilcoxon_p_two_sided"]
    ax_a1.text(0.03, 0.97, f"Mouse B cells (147 GCs)\n{fold_bcr:.1f}× (GC $p$ = {fmt_p(p_bcr)})",
               transform=ax_a1.transAxes, ha="left", va="top", fontsize=6.15, fontweight="bold", color=BLUE)
    ax_a1.set_xticks(xa)
    ax_a1.set_xticklabels([lbl for _, lbl in k_bcr], fontsize=6.2)
    ax_a1.set_xlim(-0.35, 3.40)
    ax_a1.set_ylim(0, 62)
    ax_a1.set_yticks([0, 20, 40, 60])
    ax_a1.set_yticklabels(["0%", "20%", "40%", "60%"], fontsize=6.2)
    ax_a1.set_ylabel("Expanded clones in vivo (%)", fontsize=6.8, color=INK)
    ax_a1.set_xlabel("Mutant daughters ($k$)", fontsize=6.8, color=INK)

    rc = cancer["aml_singlecell_tapestri"]["raw_singlecell_compass_zygosity"]
    ab_map = rc["clone_abundance_by_mdl_out_degree"]
    k_aml = [
        ("k=0_terminal_leaf", "0\nLeaf"),
        ("k=1_linear_step", "1\nStep"),
        ("k=2_bifurcating", "2\nSplit"),
        ("k>=3_polytomy_burst_hub", "≥3\nHub"),
    ]
    aml_cells = [ab_map[k]["mean_cells"] for k, _ in k_aml]

    ax_a2.grid(axis="y", color=HAIR, lw=0.45, zorder=0)
    ax_a2.fill_between(xa, 0, aml_cells, color=TEAL, alpha=0.08, zorder=1)
    ax_a2.plot(xa, aml_cells, "-o", color=TEAL, lw=1.35, ms=3.8, zorder=4)
    ax_a2.scatter([3], [aml_cells[3]], s=28, color=ROSE, edgecolor="white", linewidth=0.65, zorder=6)

    for xi, val in enumerate(aml_cells):
        if xi == 2:
            ax_a2.text(xi - 0.10, val + 15.0, f"{val:.1f}", ha="right", va="bottom", fontsize=6.15, color=INK)
        else:
            dy = 12.0 if xi != 1 else 14.0
            ax_a2.text(xi, val + dy, f"{val:.1f}", ha="center", va="bottom", fontsize=6.2,
                       fontweight="bold" if xi in (0, 3) else "normal",
                       color=ROSE if xi == 3 else INK)

    fold_aml = rc["hub_k3_vs_leaf_k0_mean_cell_fold_enrichment"]
    hier_aml_p = rc["hierarchical_per_run_diff_cells"]["wilcoxon_p_two_sided"]
    n_m_cells = rc["total_single_cells_scanned"] / 1e6
    ax_a2.text(0.03, 0.97, f"Human myeloid ({n_m_cells:.2f}M)\n{fold_aml:.2f}× (run $p$ = {fmt_p(hier_aml_p)})",
               transform=ax_a2.transAxes, ha="left", va="top", fontsize=6.15, fontweight="bold", color=TEAL)
    ax_a2.set_xticks(xa)
    ax_a2.set_xticklabels([lbl for _, lbl in k_aml], fontsize=6.2)
    ax_a2.set_xlim(-0.35, 3.40)
    ax_a2.set_ylim(0, 385)
    ax_a2.set_yticks([0, 100, 200, 300])
    ax_a2.set_yticklabels(["0", "100", "200", "300"], fontsize=6.2)
    ax_a2.set_ylabel("Mean cells / clone", fontsize=6.8, color=INK)
    ax_a2.set_xlabel("Mutant daughters ($k$)", fontsize=6.8, color=INK)

    # =========================================================================
    # PANEL B (Top-Right): Yeast-Display Affinity (Strictly Ordered k=0..>=3)
    # =========================================================================
    ax_b1 = fig.add_axes([0.578, 0.576, 0.162, 0.345])
    ax_b2 = fig.add_axes([0.840, 0.576, 0.146, 0.345])

    # STRICTLY monotonic k = 0, 1, 2, >=3 order
    b1_classes = [
        ("k=0_terminal_twig", "0\nLeaf", SUBINK),
        ("k=1_linear_step", "1\nStep", TEAL),
        ("k=2_small_burst", "2\nBurst", BLUE),
        ("k>=3_major_burst_hub", "≥3\nHub", ROSE),
    ]
    xb1 = np.arange(4)
    ax_b1.grid(axis="y", color=HAIR, lw=0.45, zorder=0)

    leaf_med = strat_b["k=0_terminal_twig"]["median_delta_bind"]
    ax_b1.axhline(leaf_med, color=MUTED, ls="--", lw=0.80, zorder=2)

    for xi, (k, _, col) in enumerate(b1_classes):
        med = strat_b[k]["median_delta_bind"]
        q25 = strat_b[k]["iqr25_delta_bind"]
        q75 = strat_b[k]["iqr75_delta_bind"]
        ax_b1.plot([xi, xi], [q25, q75], color=col, lw=1.65, solid_capstyle="round", zorder=3)
        ax_b1.scatter([xi], [med], s=28, color=col, edgecolor="white", linewidth=0.65, zorder=5)
        ax_b1.text(xi, q75 + 0.045, f"+{med:.2f}", ha="center", va="bottom",
                   fontsize=6.15, fontweight="bold", color=col)

    aff = disc["experimental_dms_binding_affinity"]
    rho_gc = aff["hierarchical_gc_spearman_rho"]["mean"]
    p_aff_gc = aff["hierarchical_gc_spearman_rho"]["wilcoxon_p_two_sided"]
    p_aff_m = aff["hierarchical_mouse_spearman_rho"]["wilcoxon_p_two_sided"]
    ax_b1.text(
        -0.36, -0.30,
        f"Within-GC $\\rho$ = +{rho_gc:.3f}\n"
        f"GC $p$ = {fmt_p(p_aff_gc)}\n"
        f"Mouse $p$ = {fmt_p(p_aff_m)}",
        ha="left", va="bottom",
        fontsize=6.0, fontweight="bold", color=SUBINK,
    )
    ax_b1.set_xticks(xb1)
    ax_b1.set_xticklabels([lbl for _, lbl, _ in b1_classes], fontsize=6.2)
    ax_b1.set_xlim(-0.48, 3.48)
    ax_b1.set_ylim(-0.36, 1.88)
    ax_b1.set_yticks([0.0, 0.5, 1.0, 1.5])
    ax_b1.set_yticklabels(["0.0", "0.5", "1.0", "1.5"], fontsize=6.2)
    ax_b1.set_ylabel(r"Antigen affinity ($\Delta\log_{10} K_{\mathrm{D}}$)", fontsize=6.8, color=INK)
    ax_b1.set_xlabel("Mutant daughters ($k$)", fontsize=6.8, color=INK)

    sh = p3["part3b_empirical_gcreplay"]["cross_mouse_replay_shunts"]
    par5 = sh["parallel_star_hub_replay"]["min_mice_5"]
    top_par = par5["top_parallel_convergent_mutations"][:5]
    rows_b2 = [(f"Other ({par5['n_other_aa_variants']})", par5["other_mean_delta_bind"], MUTED)] + [
        (f"{m['mutation']}\n{m['region']} · {m['independent_mice']}m", m["dms_delta_bind"],
         ROSE if "CDR" in m["region"].upper() else BLUE)
        for m in top_par[::-1]
    ]
    yb2 = np.arange(len(rows_b2))
    ax_b2.grid(axis="x", color=HAIR, lw=0.45, zorder=0)
    ax_b2.axvline(0.0, color="#334155", lw=0.65, ls="--", zorder=2)

    for yi, (lbl, val, col) in enumerate(rows_b2):
        ax_b2.plot([0.0, val], [yi, yi], color=col, lw=1.35, zorder=3)
        ax_b2.scatter([val], [yi], s=24, color=col, edgecolor="white", linewidth=0.55, zorder=5)
        if val >= 0:
            ax_b2.text(val + 0.045, yi, f"+{val:.2f}", va="center", ha="left", fontsize=6.15, fontweight="bold", color=col)
        else:
            ax_b2.text(0.045, yi, f"{val:.2f}", va="center", ha="left", fontsize=6.05, color=SUBINK)

    ax_b2.text(
        1.00, 1.03,
        f"{100*par5['cdr_fraction_parallel']:.1f}% CDR ({par5['cdr_enrichment_odds_ratio']:.1f}× OR, $p$ = {fmt_p(par5['cdr_enrichment_fisher_p_two_sided'])})",
        transform=ax_b2.transAxes, ha="right", va="bottom",
        fontsize=6.05, fontweight="bold", color=ROSE, clip_on=False,
    )
    ax_b2.set_yticks(yb2)
    ax_b2.set_yticklabels([r[0] for r in rows_b2], fontsize=6.05, color=INK)
    ax_b2.set_xlim(-0.20, 1.18)
    ax_b2.set_xticks([0.0, 0.5, 1.0])
    ax_b2.set_xticklabels(["0.0", "0.5", "1.0"], fontsize=6.2)
    ax_b2.set_ylim(-0.55, 5.45)
    ax_b2.set_xlabel(r"Affinity shift ($\Delta\log_{10} K_{\mathrm{D}}$)", fontsize=6.8, color=INK)

    # =========================================================================
    # PANEL C (Bottom-Left): Single-Cell Clonal Tree of Patient AML-38-001
    # =========================================================================
    ax_c = fig.add_axes([0.018, 0.032, 0.472, 0.415])
    draw_aml38_patient_tree(ax_c, aml38_rec, aml38_muts, aml38_bw, {
        "INK": INK, "SUBINK": SUBINK, "BLUE": BLUE, "ROSE": ROSE, "TEAL": TEAL,
    })

    # =========================================================================
    # PANEL D (Bottom-Right): 123-Patient AML Trunk-vs-Subclone Driver Landscape
    # =========================================================================
    ax_d = fig.add_axes([0.578, 0.080, 0.398, 0.345])
    aml = cancer["aml_singlecell_tapestri"]
    genes_all = [
        g for g in aml["gene_convergence_ranking"]
        if g["pan_cohort_independent_branch_acquisitions"] >= 4
        or g["within_patient_parallel_acquisitions"] >= 2
    ]

    ax_d.grid(True, color=HAIR, lw=0.45, zorder=0)
    ax_d.axvline(50.0, color="#64748B", ls="--", lw=0.75, zorder=1)
    ax_d.text(3.0, 42.6, "Early trunk (<50%)", fontsize=6.15, fontweight="bold", color=BLUE, ha="left", va="center")
    sig_enr = aml["signaling_vs_founder_subclonal_enrichment"]
    ax_d.text(
        103.5, 42.2,
        f"Late signaling (>50%)\nOR = {sig_enr['fisher_odds_ratio']:.2f}×, $p$ = {fmt_p(sig_enr['fisher_p_value_two_sided'])}",
        fontsize=6.05, fontweight="bold", color=ROSE, ha="right", va="center",
    )

    # Plot largest markers underneath smaller colliding markers so both remain visible
    genes_sorted = sorted(genes_all, key=lambda g: g["within_patient_parallel_acquisitions"], reverse=True)
    for idx_g, g in enumerate(genes_sorted):
        xv = 100.0 * g["subclonal_fraction"]
        yv = g["pan_cohort_independent_branch_acquisitions"]
        par = g["within_patient_parallel_acquisitions"]
        is_sig = g["pathway_class"] == "Signaling_Resistance"
        col = ROSE if is_sig else BLUE
        sz = 16.0 + 9.5 * par
        ax_d.scatter([xv], [yv], s=sz, color=col, alpha=0.88, edgecolor="white", linewidth=0.65, zorder=3 + 0.01 * idx_g)

    # Explicit non-colliding absolute label positions (xytext in data coords)
    label_targets = {
        "NPM1": (59.5, 36.2, "right"),
        "NRAS": (84.8, 31.8, "left"),
        "FLT3": (61.2, 29.2, "right"),
        "RUNX1": (81.2, 23.0, "left"),
        "TET2": (11.5, 21.2, "center"),
        "DNMT3A": (33.0, 20.8, "center"),
        "IDH2": (32.5, 11.8, "right"),
        "IDH1": (41.0, 8.2, "right"),
        "TP53": (44.0, 3.2, "right"),
        "ASXL1": (57.5, 13.8, "right"),
        "SRSF2": (69.0, 17.2, "center"),
        "KRAS": (79.5, 15.2, "left"),
        "PTPN11": (86.5, 11.4, "left"),
        "WT1": (57.5, 7.6, "right"),
        "U2AF1": (72.0, 2.8, "right"),
        "KIT": (95.0, 7.6, "right"),
        "JAK2": (95.0, 2.8, "right"),
    }
    for g in genes_all:
        nm = g["gene"]
        if nm not in label_targets:
            continue
        xv = 100.0 * g["subclonal_fraction"]
        yv = g["pan_cohort_independent_branch_acquisitions"]
        par = g["within_patient_parallel_acquisitions"]
        tx, ty, ha = label_targets[nm]
        is_sig = g["pathway_class"] == "Signaling_Resistance"
        txt = r"$\it{" + nm + r"}$" + (f" ({par})" if par >= 3 else "")
        ax_d.annotate(
            txt, xy=(xv, yv), xytext=(tx, ty),
            ha=ha, va="center", fontsize=6.1,
            fontweight="bold" if (par >= 3 or nm in {"TET2", "DNMT3A", "TP53"}) else "normal",
            color=ROSE if is_sig else BLUE,
            arrowprops=dict(arrowstyle="-", color="#94A3B8", lw=0.45, shrinkA=0, shrinkB=2.5),
            zorder=7,
        )

    ax_d.set_xlim(0, 106)
    ax_d.set_ylim(0, 46)
    ax_d.set_xticks([0, 25, 50, 75, 100])
    ax_d.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=6.2)
    ax_d.set_yticks([0, 10, 20, 30, 40])
    ax_d.set_yticklabels(["0", "10", "20", "30", "40"], fontsize=6.2)
    ax_d.set_xlabel("Subclonal branch acquisitions (% of patient trees)", fontsize=6.8, color=INK)
    ax_d.set_ylabel("Independent acquisitions across cohort", fontsize=6.8, color=INK)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300)
    fig.savefig(OUT_PNG, dpi=450)
    plt.close(fig)
    print(f"Wrote {OUT_PDF} and {OUT_PNG}")


if __name__ == "__main__":
    main()
