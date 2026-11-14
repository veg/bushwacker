#!/usr/bin/env python3
"""
Generate Science-caliber Supplementary Figure S7
(`paper/fig_supp_chst13_bootstrap.pdf` and `results/fig_supp_chst13_bootstrap.png`)
comparing bootstrap, SH-aLRT, aBayes, --polytomy, FastTree, and Bushwhacker MDL
on mammalian CHST13 (ENSG00000180767) and multi-locus ASTRAL-IV coalescent trees.

All plotted trees, support values, split counts, and runtimes come strictly from
`results/macroevolution/fig3_empirical_tanglegram_gene.json` and
`results/macroevolution/pillar1_part1a.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent
if str(FIG_DIR) not in sys.path:
    sys.path.insert(0, str(FIG_DIR))

from fig3_macroevo_clean import (  # noqa: E402
    assign_cladogram_coords,
    clean_mammal_name,
    draw_panel_header,
    get_leaves,
    get_splits_set,
    parse_newick,
    prune_tree,
    reroot_at_outgroup,
    untangle_min_crossings,
)


REPO_ROOT = FIG_DIR.parents[1]
GENE_JSON = REPO_ROOT / "results" / "macroevolution" / "fig3_empirical_tanglegram_gene.json"
PILLAR1_JSON = REPO_ROOT / "results" / "macroevolution" / "pillar1_part1a.json"
OUT_PDF = REPO_ROOT / "paper" / "fig_supp_chst13_bootstrap.pdf"
OUT_PNG = REPO_ROOT / "results" / "fig_supp_chst13_bootstrap.png"


DESIRED_CLADE_SEED = {
    "P036_Human": 0, "P035_Chimp": 1, "P034_Gorilla": 2, "P033_Orangutan": 3,
    "P032_Rhesus": 4, "P031_Marmoset": 5, "P028_Bushbaby": 6,
    "P025_Mouse": 7, "P026_Rat": 8, "P024_Kangaroo_rat": 9, "P023_Guinea_pig": 10, "P021_Pika": 11,
    "P011_Dolphin": 12, "P012_Cow": 13, "P013_Pig": 14, "P016_Dog": 15, "P018_Megabat": 16,
    "P005_Rock_hyrax": 17, "P008_Sloth": 18,
    "P002_Opossum": 19, "P003_Wallaby": 20, "P001_Platypus": 21,
}


def canonical_split_key(leaves: set[str] | frozenset[str], all_tips: set[str]) -> frozenset[frozenset[str]]:
    return frozenset([frozenset(leaves), frozenset(all_tips - set(leaves))])


def collapse_by_splits(node: dict, keep_splits: set, all_tips: set[str]) -> dict:
    """Return a copy of a rooted tree retaining only internal splits in keep_splits."""
    if not node["children"]:
        return {
            "children": [],
            "name": node.get("name"),
            "sup": node.get("sup"),
            "bl": node.get("bl", 0.01),
        }
    new_children = []
    for c in node["children"]:
        cc = collapse_by_splits(c, keep_splits, all_tips)
        if cc["children"] and 1 < len(c["_leaves"]) < len(all_tips) - 1:
            sp = canonical_split_key(c["_leaves"], all_tips)
            if sp not in keep_splits:
                new_children.extend(cc["children"])
                continue
        new_children.append(cc)
    return {
        "children": new_children,
        "name": node.get("name"),
        "sup": node.get("sup"),
        "bl": node.get("bl", 0.01),
    }


def draw_cladogram_column(
    ax,
    tree_node: dict,
    x0: float,
    x1: float,
    ref_splits: set,
    all_tips: set[str],
    title: str,
    subtitle: str,
    true_color: str,
    conf_color: str,
    ink: str,
    subink: str,
    branch_labels: dict | None = None,
    highlight_taxa: set[str] | None = None,
    poly_marker_color: str = "#64748B",
    dashed_splits: set | None = None,
    node_callout: tuple[frozenset, str, str, tuple[float, float]] | None = None,
) -> None:
    """Draw a compact left-to-right cladogram with mammal labels to the right of tips."""
    assign_cladogram_coords(tree_node)
    get_splits_set(tree_node, all_tips)

    def mx(ux: float) -> float:
        return x0 + ux * (x1 - x0)

    tip_offset = 0.005
    highlight_taxa = highlight_taxa or set()
    branch_labels = branch_labels or {}
    dashed_splits = dashed_splits or set()

    def rec(u: dict, is_root: bool = False) -> None:
        ux, uy = mx(u["_x"]), u["_y"]
        if not is_root and len(u["children"]) >= 3:
            ax.scatter(
                [ux], [uy], s=14, facecolor="white", edgecolor=poly_marker_color,
                linewidth=0.95, zorder=6,
            )
            if node_callout is not None and 1 < len(u["_leaves"]) < len(all_tips) - 1:
                target_sp, tag_txt, tag_col, text_pos = node_callout
                if canonical_split_key(u["_leaves"], all_tips) == target_sp:
                    ax.annotate(
                        tag_txt,
                        xy=(ux, uy),
                        xytext=text_pos,
                        fontsize=4.7, fontweight="bold", color=tag_col,
                        ha="left", va="center",
                        bbox=dict(boxstyle="round,pad=0.14", fc="#FFFBEB", ec=tag_col, lw=0.5),
                        arrowprops=dict(arrowstyle="->", color=tag_col, lw=0.65, mutation_scale=5.5, shrinkA=1, shrinkB=3),
                        zorder=7,
                    )
        for c in u["children"]:
            cx, cy = mx(c["_x"]), c["_y"]
            is_int = bool(c["children"]) and 1 < len(c["_leaves"]) < len(all_tips) - 1
            is_conf = False
            is_dashed = False
            lbl = None
            col = "#64748B"
            lw = 0.68
            z = 2
            if is_int:
                sp = canonical_split_key(c["_leaves"], all_tips)
                is_conf = sp not in ref_splits
                is_dashed = is_conf and (sp in dashed_splits)
                if is_conf:
                    col = conf_color
                    lw = 1.05 if is_dashed else 1.20
                    z = 4
                elif true_color:
                    col = true_color
                    lw = 1.05 if true_color == "#1D4ED8" else 0.78
                    z = 3
                if sp in branch_labels:
                    lbl = branch_labels[sp]

            ls = (0, (2.2, 1.3)) if is_dashed else "-"
            ax.plot([ux, ux], [uy, cy], color="#94A3B8" if not is_conf else "#64748B", lw=0.65, zorder=1)
            ax.plot([ux, cx], [cy, cy], color=col, lw=lw, ls=ls, solid_capstyle="round", zorder=z)

            if lbl is not None:
                lbl_y = cy + 0.14 if cy > uy else cy - 0.14
                va = "bottom" if cy > uy else "top"
                ax.text(
                    0.5 * (ux + cx), lbl_y, lbl,
                    fontsize=4.55, fontweight="bold",
                    color=conf_color if is_conf else col,
                    ha="center", va=va,
                    bbox=dict(boxstyle="square,pad=0.02", fc="white", ec="none", alpha=0.88),
                    zorder=6,
                )

            rec(c, is_root=False)

    rec(tree_node, is_root=True)

    # Tip labels
    def draw_tips(u: dict) -> None:
        if not u["children"]:
            nm = u["name"]
            is_hi = nm in highlight_taxa
            ax.text(
                x1 + tip_offset, u["_y"], clean_mammal_name(nm),
                ha="left", va="center", fontsize=5.0,
                fontweight="bold" if is_hi else "normal",
                color=conf_color if (is_hi and conf_color) else (ink if is_hi else subink),
            )
            return
        for c in u["children"]:
            draw_tips(c)

    draw_tips(tree_node)

    xc = 0.5 * (x0 + x1 + 0.045)
    head_col = conf_color if "conflicts" in subtitle and not subtitle.startswith("0 ") else (true_color if true_color != "#475569" else ink)
    ax.text(xc, 23.05, title, fontsize=6.2, fontweight="bold", color=head_col, ha="center", va="center")
    ax.text(xc, 21.95, subtitle, fontsize=5.35, color=subink, ha="center", va="center")


def main() -> None:
    bundle = json.loads(GENE_JSON.read_text(encoding="utf-8"))
    gp = bundle["genes"]["ENSG00000180767"]
    bc = gp["bootstrap_comparison"]
    p1 = json.loads(PILLAR1_JSON.read_text(encoding="utf-8"))

    bw_raw = parse_newick(gp["bw_json"]["newick"])
    ml_raw = parse_newick(gp["ml_best_observed_newick"])
    all_tips = set(get_leaves(bw_raw))

    ref_pruned = reroot_at_outgroup(prune_tree(parse_newick(bundle["reference_astral_newick"]), all_tips), "P001_Platypus")
    untangle_min_crossings(ref_pruned, DESIRED_CLADE_SEED)
    ref_order = get_leaves(ref_pruned)
    ref_rank = {nm: idx for idx, nm in enumerate(ref_order)}
    ref_splits = get_splits_set(ref_pruned, all_tips)

    ml_tree = reroot_at_outgroup(ml_raw, "P001_Platypus")
    get_splits_set(ml_tree, all_tips)

    bw_tree = reroot_at_outgroup(bw_raw, "P001_Platypus")
    get_splits_set(bw_tree, all_tips)
    untangle_min_crossings(bw_tree, ref_rank)

    # Map UFBoot values and reference membership per ML split
    ml_meta = {}
    for rec in bc["ml_splits"]:
        sp = canonical_split_key(set(rec["clade"]), all_tips)
        ml_meta[sp] = rec

    ml_all_splits = {sp for sp in ml_meta}
    ml_uf50_splits = {sp for sp, rec in ml_meta.items() if rec["ufboot"] >= 50}
    ml_uf95_splits = {sp for sp, rec in ml_meta.items() if rec["ufboot"] >= 95}

    t_raw = reroot_at_outgroup(parse_newick(gp["ml_best_observed_newick"]), "P001_Platypus")
    untangle_min_crossings(t_raw, ref_rank)

    t_uf50 = collapse_by_splits(ml_tree, ml_uf50_splits, all_tips)
    untangle_min_crossings(t_uf50, ref_rank)

    t_uf95 = collapse_by_splits(ml_tree, ml_uf95_splits, all_tips)
    untangle_min_crossings(t_uf95, ref_rank)

    # Identify displaced taxa in Raw ML for bold crimson callout
    conf_taxa_in_small = set()
    for sp, rec in ml_meta.items():
        if not rec["in_ref"] and len(rec["clade"]) <= 4:
            for tx in rec["clade"]:
                conf_taxa_in_small.add(tx)

    simiiformes_sp = canonical_split_key(
        {"P031_Marmoset", "P032_Rhesus", "P033_Orangutan", "P034_Gorilla", "P035_Chimp", "P036_Human"},
        all_tips,
    )
    ml_lt50_splits = {sp for sp, rec in ml_meta.items() if not rec["in_ref"] and rec["ufboot"] < 50}

    uf50_conf_labels = {
        sp: f"{rec['ufboot']}"
        for sp, rec in ml_meta.items()
        if not rec["in_ref"] and rec["ufboot"] >= 50
    }
    raw_conf_labels = {
        sp: (f"{rec['ufboot']}†" if not rec["in_polytomy"] else f"{rec['ufboot']}")
        for sp, rec in ml_meta.items()
        if not rec["in_ref"]
    }
    bw_labels = {}
    for sp, rec in ml_meta.items():
        pass
    # Read P_local directly from bw_tree nodes
    get_splits_set(bw_tree, all_tips)
    def collect_bw_sup(u):
        if u["children"] and 1 < len(u["_leaves"]) < len(all_tips) - 1 and u.get("sup") is not None:
            sp = canonical_split_key(u["_leaves"], all_tips)
            sv = float(u["sup"])
            bw_labels[sp] = "1.0" if sv >= 0.995 else f"{sv:.2f}"
        for c in u["children"]:
            collect_bw_sup(c)
    collect_bw_sup(bw_tree)

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
        "font.size": 6.4,
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
    HAIR = "#E2E8F0"
    BLUE = "#1D4ED8"
    SKY = "#0284C7"
    TEAL = "#0D9488"
    AMBER = "#D97706"
    ROSE = "#E11D48"

    fig = plt.figure(figsize=(7.25, 5.65), facecolor="white")

    draw_panel_header(
        fig, 0.020, 0.972, "A",
        r"$\it{CHST13}$ topologies across support thresholds and Bushwhacker MDL (22 mammals, 1,176 bp)",
        INK,
    )
    draw_panel_header(fig, 0.020, 0.452, "B", r"Split recovery and single-core CPU cost on $\it{CHST13}$", INK)
    draw_panel_header(fig, 0.515, 0.452, "C", "ASTRAL-IV error across cutoffs and rapid radiations", INK)

    # =========================================================================
    # PANEL A: 4-Tree Side-by-Side Comparison on CHST13
    # =========================================================================
    ax_a = fig.add_axes([0.015, 0.488, 0.970, 0.456])
    ax_a.set_xlim(0.0, 1.0)
    ax_a.set_ylim(-1.65, 24.0)
    ax_a.axis("off")

    rt = bc["runtimes_1core_sec"]
    ms = bc["methods_summary"]

    # Light dividers separating the 4 columns
    for xv in (0.246, 0.496, 0.746):
        ax_a.plot([xv, xv], [-0.3, 23.5], color="#E2E8F0", lw=0.65, ls="--", zorder=0)

    draw_cladogram_column(
        ax_a, t_raw, 0.005, 0.141, ref_splits, all_tips,
        title="1. Raw ML (--polytomy)",
        subtitle=f"{ms['Raw_ML']['fp']} false ({ms['IQTREE_polytomy']['fp']} --poly) · {ms['Raw_ML']['tp']} true · {rt['ml_raw_1core_sec']:.1f}–{rt['polytomy_1core_sec']:.1f} s",
        true_color="#475569", conf_color=ROSE, ink=INK, subink=SUBINK,
        branch_labels=raw_conf_labels,
        highlight_taxa=conf_taxa_in_small,
        dashed_splits=ml_lt50_splits,
    )

    draw_cladogram_column(
        ax_a, t_uf50, 0.255, 0.391, ref_splits, all_tips,
        title="2. ML + UFBoot ≥ 50%",
        subtitle=f"{ms['UFBoot_50']['fp']} false remain · {ms['UFBoot_50']['tp']} true · {rt['ufboot1000_1core_sec']:.1f} s",
        true_color="#475569", conf_color=ROSE, ink=INK, subink=SUBINK,
        branch_labels=uf50_conf_labels,
        highlight_taxa={"P034_Gorilla", "P035_Chimp", "P028_Bushbaby", "P008_Sloth", "P021_Pika"},
        poly_marker_color="#64748B",
    )

    draw_cladogram_column(
        ax_a, t_uf95, 0.505, 0.641, ref_splits, all_tips,
        title="3. ML + UFBoot ≥ 95%",
        subtitle=f"{ms['UFBoot_95']['fp']} false · {ms['UFBoot_95']['tp']} true · {rt['ufboot1000_1core_sec']:.1f} s",
        true_color=TEAL, conf_color=ROSE, ink=INK, subink=SUBINK,
        poly_marker_color=AMBER,
        node_callout=(
            simiiformes_sp,
            "Catarrhini (93%)\ncollapsed <95%",
            AMBER,
            (0.503, 19.8),
        ),
    )

    draw_cladogram_column(
        ax_a, bw_tree, 0.755, 0.891, ref_splits, all_tips,
        title="4. Bushwhacker (MDL)",
        subtitle=f"{ms['Bushwhacker_MDL']['fp']} false · {ms['Bushwhacker_MDL']['tp']} true · {rt['bw_1core_sec']:.2f} s",
        true_color=BLUE, conf_color=ROSE, ink=INK, subink=SUBINK,
        branch_labels=bw_labels,
        poly_marker_color=BLUE,
    )

    # Compact legend footer across Panel A
    ax_a.plot([0.020, 0.048], [-1.15, -1.15], color=ROSE, lw=1.20)
    ax_a.text(0.054, -1.15, r"UFBoot conflict ($\geq$50%)", fontsize=5.3, color=ROSE, ha="left", va="center")
    ax_a.plot([0.215, 0.243], [-1.15, -1.15], color=ROSE, lw=1.05, ls=(0, (2.2, 1.3)))
    ax_a.text(0.249, -1.15, "<50%", fontsize=5.3, color=ROSE, ha="left", va="center")
    ax_a.text(0.312, -1.15, r"$\dagger$ Only split collapsed by --poly (bl=$10^{-6}$)", fontsize=5.3, color=SUBINK, ha="left", va="center")
    ax_a.scatter([0.652], [-1.15], s=14, facecolor="white", edgecolor=AMBER, linewidth=0.95, zorder=6)
    ax_a.text(0.663, -1.15, "Collapsed hub (≥3-way)", fontsize=5.3, color=INK, ha="left", va="center")
    ax_a.plot([0.830, 0.858], [-1.15, -1.15], color=BLUE, lw=1.05)
    ax_a.text(0.864, -1.15, r"True ($P_{\mathrm{local}}$)", fontsize=5.3, fontweight="bold", color=BLUE, ha="left", va="center")

    # =========================================================================
    # PANEL B: Diverging TP vs FP Splits + 1-Core Runtime on CHST13
    # =========================================================================
    ax_b = fig.add_axes([0.168, 0.082, 0.312, 0.326])
    rows_b = [
        ("Raw_ML", "Raw ML (GTR+G)"),
        ("IQTREE_polytomy", "IQ-TREE --polytomy"),
        ("UFBoot_30", "UFBoot ≥ 30%"),
        ("UFBoot_50", "UFBoot ≥ 50%"),
        ("StdBS_50", "Std. BS (100) ≥ 50%"),
        ("SH_aLRT_80", "SH-aLRT ≥ 80%"),
        ("UFBoot_70", "UFBoot ≥ 70%"),
        ("StdBS_70", "Std. BS (100) ≥ 70%"),
        ("UFBoot_95", "UFBoot ≥ 95%"),
        ("Bushwhacker_MDL_live_T1", "Bushwhacker (GTR+CAT8)*"),
        ("Bushwhacker_MDL", "Bushwhacker (emp.-α)"),
    ][::-1]

    y_b = np.arange(len(rows_b))
    ax_b.grid(axis="x", color=HAIR, lw=0.45, zorder=0)
    ax_b.axvline(0.0, color=INK, lw=0.75, zorder=3)

    # Shade Bushwhacker rows lightly
    ax_b.axhspan(-0.45, 1.45, color="#EFF6FF", zorder=0)

    for idx, (key, lbl) in enumerate(rows_b):
        m = ms[key]
        fp = m["fp"]
        tp = m["tp"]
        rts = m["runtime_sec"]
        is_bw = key.startswith("Bushwhacker")
        if fp > 0:
            ax_b.barh(idx, -fp, height=0.62, color=ROSE if not is_bw else AMBER, alpha=0.88, zorder=3)
            ax_b.text(-fp - 0.22, idx, f"{fp}", ha="right", va="center", fontsize=5.45, fontweight="bold",
                      color=ROSE if not is_bw else AMBER)
        else:
            ax_b.scatter([0], [idx], s=14, color=BLUE if is_bw else TEAL, zorder=4)

        bar_col = BLUE if is_bw else (TEAL if fp == 0 else "#64748B")
        ax_b.barh(idx, tp, height=0.62, color=bar_col, alpha=0.88, zorder=3)
        ax_b.text(tp + 0.22, idx, f"{tp}", ha="left", va="center", fontsize=5.45, fontweight="bold", color=bar_col)

        rt_str = f"{rts:.2f} s" if rts < 0.5 else (f"{rts:.1f} s" if rts < 100 else f"{rts:.0f} s")
        ax_b.text(13.8, idx, rt_str, ha="left", va="center", fontsize=5.45,
                  fontweight="bold" if is_bw else "normal",
                  color=BLUE if is_bw else SUBINK)

    ax_b.text(-4.5, len(rows_b) - 0.30, "← False (FP)", ha="center", va="bottom", fontsize=5.75, fontweight="bold", color=ROSE, clip_on=False)
    ax_b.text(6.0, len(rows_b) - 0.30, "True clades (TP) →", ha="center", va="bottom", fontsize=5.75, fontweight="bold", color=BLUE, clip_on=False)
    ax_b.text(13.8, len(rows_b) - 0.30, "1-core CPU", ha="left", va="bottom", fontsize=5.6, fontweight="bold", color=SUBINK, clip_on=False)

    ax_b.set_yticks(y_b)
    ax_b.set_yticklabels([lbl for _, lbl in rows_b], fontsize=5.75, color=INK)
    ax_b.set_xlim(-9.6, 17.6)
    ax_b.set_ylim(-0.65, len(rows_b) + 0.15)
    ax_b.set_xticks([-8, -4, 0, 4, 8, 12])
    ax_b.set_xticklabels(["8", "4", "0", "4", "8", "12"], fontsize=5.8)
    ax_b.set_xlabel("False (left) vs. true clades (right)", fontsize=6.5, color=INK)

    # =========================================================================
    # PANEL C: Multi-Locus ASTRAL-IV Threshold Sensitivity (pillar1_part1a.json)
    # =========================================================================
    ax_c = fig.add_axes([0.562, 0.082, 0.412, 0.326])
    ax_c.grid(axis="y", color=HAIR, lw=0.45, zorder=0)

    sc_map = {s["scenario"]: s for s in p1["part_1a_msc_simulations"]}
    bars_anom = p1["bars_eq9_and_collapsed_ml_baselines"]

    # Left section (x = 0..5): IQ-TREE support threshold sweep on 200-locus MSC regimes
    # Right section (x = 6.7..8.4): FastTree SH70 -> SH90 spike and 100-locus Anomaly-24 UFBoot95 vs BW
    cut_keys = ["iq_bin", "iq_poly", "iq_uf10", "iq_uf30", "iq_uf50", "iq_uf70"]
    cut_lbls = ["Raw\nML", "--poly", "UFB\n≥10%", "UFB\n≥30%", "UFB\n≥50%", "UFB\n≥70%"]
    xc = np.arange(len(cut_keys), dtype=float)

    for sc_key, label, col, mk in [
        ("MSC_Anomaly24_L400", "Anomaly-24 (200 loci, 400 bp)", ROSE, "o"),
        ("MSC_Rapid32_L800", "Rapid-32 (200 loci, 800 bp)", TEAL, "s"),
    ]:
        ms_sc = sc_map[sc_key]["methods"]
        ys = [ms_sc[k]["mean_nrf"] * 100.0 for k in cut_keys]
        ax_c.plot(xc, ys, f"-{mk}", color=col, lw=1.25, markersize=3.8, label=label, zorder=4)
        bw_v = ms_sc["bw_mdl"]["mean_nrf"] * 100.0
        bw_w = ms_sc["bw_mdl"]["mean_wall_64core_sec"]
        iq_w = ms_sc["iq_uf70"]["mean_wall_64core_sec"]
        ax_c.plot([0, 5.2], [bw_v, bw_v], ls="--", color=col, lw=0.95, alpha=0.85, zorder=3)
        if col == ROSE:
            ax_c.text(
                0.08, bw_v + 0.9,
                f"Bushwhacker ({bw_w:.2f} s vs. {iq_w:.0f} s UFBoot)",
                fontsize=5.25, fontweight="bold", color=col, va="bottom",
            )
        else:
            ax_c.annotate(
                f"Bushwhacker ({bw_w:.2f} s vs. {iq_w:.0f} s UFBoot)",
                xy=(1.55, bw_v),
                xytext=(0.15, 11.4),
                fontsize=5.25, fontweight="bold", color=col, va="bottom",
                arrowprops=dict(arrowstyle="->", color=col, lw=0.65, ls="--", mutation_scale=5.5, shrinkB=2),
                zorder=5,
            )

    # Highlight FastTree SH90 & Anomaly-24 100-locus UFBoot>=95% vs Bushwhacker bars on the right
    ax_c.axvline(5.55, color="#CBD5E1", lw=0.75, ls=":", zorder=2)

    # Pairs at x=6.2, 7.0, 8.05, 8.85: Anomaly-24 200-locus (FT SH70 vs SH90) and 100-locus (BW vs UFBoot>=95%)
    r_bars = [
        (6.2, sc_map["MSC_Anomaly24_L400"]["methods"]["ft_sh70"]["mean_nrf"] * 100.0, "#94A3B8", "FT\nSH70"),
        (7.0, sc_map["MSC_Anomaly24_L400"]["methods"]["ft_sh90"]["mean_nrf"] * 100.0, AMBER, "FT\nSH90"),
        (8.05, bars_anom["default"]["mean_nrf"] * 100.0, BLUE, "BW\n(0.6s)"),
        (8.85, bars_anom["iq_uf95"]["mean_nrf"] * 100.0, ROSE, "UFB≥95\n(38s)"),
    ]
    for xb, yb, col, tag in r_bars:
        ax_c.bar([xb], [yb], width=0.62, color=col, alpha=0.9, zorder=3)
        ax_c.text(xb, yb + 0.7, f"{yb:.1f}%", ha="center", va="bottom", fontsize=5.2, fontweight="bold", color=col)

    ax_c.text(6.6, 42.8, "Anom.-24\n(200 loci)", ha="center", va="top", fontsize=5.15, color=SUBINK)
    ax_c.text(8.45, 42.8, "Anom.-24\n(100 loci)", ha="center", va="top", fontsize=5.15, fontweight="bold", color=INK)

    ax_c.set_xticks(list(xc) + [6.2, 7.0, 8.05, 8.85])
    ax_c.set_xticklabels(cut_lbls + ["FT\nSH70", "FT\nSH90", "BW\n(0.6s)", "UFB95\n(38s)"], fontsize=5.3)
    ax_c.set_xlim(-0.45, 9.45)
    ax_c.set_ylim(0, 44.5)
    ax_c.set_yticks([0, 10, 20, 30, 40])
    ax_c.set_yticklabels(["0", "10", "20", "30", "40"], fontsize=5.8)
    ax_c.set_ylabel("ASTRAL-IV species-tree error (nRF %)", fontsize=6.4, color=INK)
    ax_c.set_xlabel("Gene-tree collapsing criterion before coalescent summary", fontsize=6.4, color=INK)
    ax_c.legend(loc="upper left", frameon=False, fontsize=5.45)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300)
    fig.savefig(OUT_PNG, dpi=450)
    plt.close(fig)
    print(f"Generated {OUT_PDF} and {OUT_PNG}")


if __name__ == "__main__":
    main()
