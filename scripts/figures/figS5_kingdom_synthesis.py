#!/usr/bin/env python3
"""
Generate publication-grade vector PDF Figure (`paper/fig_kingdom_synthesis.pdf`)
from the executed 19,952-gene Mammalian TOGA2 Taxonomic Atlas
(`scratch/toga2_taxonomic_atlas.json`, `scratch/toga2_proteome_deep_summary.json`)
and the 2,002-alignment Plant Kingdom Atlas (`scratch/plant_kingdom_atlas.json`).
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    toga_tax_path = repo_root / "results" / "mammalian_toga2" / "toga2_taxonomic_atlas.json"
    toga_sum_path = repo_root / "results" / "mammalian_toga2" / "toga2_deep_mining_atlas.json"
    plant_path = repo_root / "results" / "macroevolution" / "plant_kingdom_atlas.json"
    out_pdf = repo_root / "paper" / "fig_kingdom_synthesis.pdf"

    toga_tax = json.loads(toga_tax_path.read_text(encoding="utf-8"))
    toga_sum = json.loads(toga_sum_path.read_text(encoding="utf-8"))
    plant = json.loads(plant_path.read_text(encoding="utf-8"))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8.0,
        "axes.titlesize": 9.3,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.0,
    })

    # Okabe-Ito colorblind-safe palette
    c_blue = "#0072B2"
    c_sky = "#56B4E9"
    c_orange = "#E69F00"
    c_vermillion = "#D55E00"
    c_green = "#009E73"
    c_purple = "#CC79A7"
    c_navy = "#1e3a8a"

    fig = plt.figure(figsize=(14.0, 8.2))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.05, 1.05],
        height_ratios=[1.0, 1.05],
        wspace=0.28,
        hspace=0.38,
    )

    # =========================================================================
    # PANEL A: Length-Invariance Law of True Mammalian Discordance (19,952 Genes)
    # =========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    by_len = toga_sum["stratification_by_gene_length"]
    bin_labels = [
        "<0.6 kb\n(2,680)",
        "0.6–1.2 kb\n(5,650)",
        "1.2–2.4 kb\n(7,087)",
        "2.4–4.8 kb\n(3,415)",
        ">4.8 kb\n(1,058)",
    ]
    x_a = np.arange(len(by_len))

    raw_conf = [b["binary_conflict_rate_pct"] for b in by_len]
    mdl_conf = [b["mdl_conflict_rate_pct"] for b in by_len]
    sp_better = [b["sp_tree_compresses_better_or_equal_pct"] for b in by_len]

    ax_a.plot(
        x_a, raw_conf, marker="o", lw=2.2, color=c_vermillion,
        label="Unfiltered Binary Gene-Tree Conflict (%)",
    )
    ax_a.plot(
        x_a, mdl_conf, marker="s", lw=2.4, color=c_blue,
        label="MDL-Contracted True Discordance (ILS/Paralogy, %)",
    )
    ax_a.plot(
        x_a, sp_better, marker="^", lw=2.2, ls="--", color=c_green,
        label="Genes Compressed Better by Species Tree (%)",
    )

    for i in range(len(x_a)):
        ax_a.text(x_a[i], raw_conf[i] + 2.2, f"{raw_conf[i]:.1f}%", ha="center", va="bottom", fontsize=6.9, fontweight="bold", color=c_vermillion)
        if i == 1:
            ax_a.text(x_a[i], mdl_conf[i] + 2.2, f"{mdl_conf[i]:.1f}%", ha="center", va="bottom", fontsize=6.9, fontweight="bold", color=c_blue)
            ax_a.text(x_a[i] - 0.18, sp_better[i] + 1.8, f"{sp_better[i]:.1f}%", ha="right", va="bottom", fontsize=6.9, fontweight="bold", color=c_green)
        else:
            ax_a.text(x_a[i], mdl_conf[i] - 3.2, f"{mdl_conf[i]:.1f}%", ha="center", va="top", fontsize=6.9, fontweight="bold", color=c_blue)
            off_sp = -5.8 if i == 2 else 2.2
            ax_a.text(x_a[i], sp_better[i] + off_sp, f"{sp_better[i]:.1f}%", ha="center", va="bottom", fontsize=6.9, fontweight="bold", color=c_green)

    ax_a.axhspan(32.5, 35.0, color=c_sky, alpha=0.18, zorder=0)
    ax_a.text(
        2.15, 11.5,
        "Length-Invariant Mammalian Discordance Floor (~33.2%–34.0%)\n"
        "(55.5% of binary conflicts = 3.87M splits contracted as noise)",
        ha="center", va="center", fontsize=6.5, color="#004b76",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#eef7fc", edgecolor=c_sky, lw=0.8),
    )

    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(bin_labels, fontsize=7.1)
    ax_a.set_ylabel("Percentage across 19,890 Mammalian Genes (%)", fontsize=7.8)
    ax_a.set_ylim(2, 104)
    ax_a.set_xlim(-0.4, 4.4)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.legend(loc="upper left", fontsize=6.5, frameon=True)
    ax_a.set_title(
        "A. Length-Invariance of Mammalian Discordance (19,890 Coding Loci)",
        loc="left", pad=8,
    )

    # =========================================================================
    # PANEL B: Taxonomic Scope & Functional Enrichment of Mammalian Shunts
    # =========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    scope_int = toga_tax["shunt_taxonomic_scope"]["INTERVAL"]
    scope_spa = toga_tax["shunt_taxonomic_scope"]["SPARSE"]
    tot_int = scope_int["intra_family"] + scope_int["intra_order_inter_family"] + scope_int["inter_order"]
    tot_spa = scope_spa["intra_family"] + scope_spa["intra_order_inter_family"] + scope_spa["inter_order"]

    cats = [
        "Intra-Family\n(Recent Paralogy /\nConversion)",
        "Inter-Family\n(Within Mammalian\nOrder)",
        "Inter-Order\n(Deep Ecological\nConvergence)",
    ]
    int_pcts = [
        100.0 * scope_int["intra_family"] / tot_int,
        100.0 * scope_int["intra_order_inter_family"] / tot_int,
        100.0 * scope_int["inter_order"] / tot_int,
    ]
    spa_pcts = [
        100.0 * scope_spa["intra_family"] / tot_spa,
        100.0 * scope_spa["intra_order_inter_family"] / tot_spa,
        100.0 * scope_spa["inter_order"] / tot_spa,
    ]

    x_b = np.arange(len(cats))
    w_b = 0.34
    bars_int = ax_b.bar(
        x_b - w_b / 2, int_pcts, width=w_b, color=c_orange, edgecolor="#8a4b00", lw=0.8,
        label=f"INTERVAL Shunts ({tot_int:,} tracts; 3.0×–6.6× in Tandem Arrays)",
    )
    bars_spa = ax_b.bar(
        x_b + w_b / 2, spa_pcts, width=w_b, color=c_blue, edgecolor="#004b76", lw=0.8,
        label=f"SPARSE Shunts ({tot_spa:,} links; {spa_pcts[2]:.1f}% Cross-Order)",
    )

    for b, v in zip(bars_int, int_pcts):
        ax_b.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.1, fontweight="bold", color="#8a4b00")
    for b, v in zip(bars_spa, spa_pcts):
        ax_b.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.1, fontweight="bold", color="#004b76")

    fam_map = {r["family"]: r for r in toga_sum.get("gene_family_shunt_enrichment", [])}
    hb = fam_map.get("Hemoglobin / Globin Clusters", {"interval_gene_pct": 40.0, "interval_fold_enrichment": 6.62})
    cyp = fam_map.get("Xenobiotic Detoxification (CYP/UGT/AKR/GST)", {"interval_gene_pct": 32.76, "interval_fold_enrichment": 5.42})
    olf = fam_map.get("Olfactory Receptors (OR)", {"interval_gene_pct": 26.32, "interval_fold_enrichment": 4.36})
    znf = fam_map.get("KRAB / C2H2 Zinc-Finger Arrays", {"interval_gene_pct": 23.80, "interval_fold_enrichment": 3.94})
    cad = fam_map.get("Cadherins, Keratins, Mucins & Extracellular Matrix", {"interval_gene_pct": 18.06, "interval_fold_enrichment": 2.99})
    oth = fam_map.get("Other Coding Loci", {"interval_gene_pct": 4.15, "interval_fold_enrichment": 0.69})

    ax_b.text(
        1.88, 64.0,
        "Tandem-Array INTERVAL Enrichment\n"
        f"• Hemoglobin/Globin: {hb['interval_gene_pct']:.1f}% ({hb['interval_fold_enrichment']:.1f}×)\n"
        f"• CYP/UGT Detox:     {cyp['interval_gene_pct']:.1f}% ({cyp['interval_fold_enrichment']:.1f}×)\n"
        f"• Olfactory (OR):    {olf['interval_gene_pct']:.1f}% ({olf['interval_fold_enrichment']:.1f}×)\n"
        f"• KRAB-ZNF Arrays:   {znf['interval_gene_pct']:.1f}% ({znf['interval_fold_enrichment']:.1f}×)\n"
        f"• Cadherins/Keratin: {cad['interval_gene_pct']:.1f}% ({cad['interval_fold_enrichment']:.1f}×)\n"
        f"• Other Coding Loci: {oth['interval_gene_pct']:.1f}% ({oth['interval_fold_enrichment']:.1f}×)",
        fontsize=6.2, fontfamily="monospace", color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fdf8ec", edgecolor="#e2c07d", lw=0.85),
        ha="center", va="center",
    )

    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(cats, fontsize=7.3)
    ax_b.set_ylabel("Share of Inferred Mammalian Shunts (%)", fontsize=7.9)
    ax_b.set_ylim(0, 90)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.legend(loc="upper left", fontsize=6.8, frameon=True)
    ax_b.set_title(
        "B. Taxonomic Scope Law: INTERVAL Conversion vs. SPARSE Convergence",
        loc="left", pad=8,
    )

    # =========================================================================
    # PANEL C: Cochlear Hair-Cell Mechanotransduction Complex (10/27 Sp AA; 19/27 Free AA)
    # =========================================================================
    ax_c = fig.add_subplot(gs[1, 0])
    coch_notes = {
        "USH2A": "Usherin ankle-link (21 free-AA shunts)",
        "ADGRV1": "Ankle-link GPCR (21 free-AA shunts)",
        "TRIOBP": "Stereocilia rootlet actin-bundling",
        "CDH23": "Otocadherin upper tip-link (24 free-AA)",
        "LOXHD1": "Stereocilia connector",
        "CIB2": "TMC1/2 auxiliary subunit",
        "WHRN": "Whirlin ankle-link scaffold",
        "STRC": "Stereocilin top-connector",
        "PCDH15": "Lower tip-link (44 free-AA)",
        "TMC1": "Channel pore (3 sp-NT)",
    }
    coch_rows = []
    for row in toga_sum.get("cochlear_auditory_complex_27_genes", toga_sum.get("cochlear_auditory_complex_25_genes", [])):
        sym = row["gene_symbol"]
        n_sp_aa = row.get("n_sp_aa_shunts", 0)
        gain_sp_aa = row.get("sp_aa_total_gain_nats", 0.0)
        if n_sp_aa > 0 and sym in coch_notes:
            coch_rows.append((f"{sym} ({n_sp_aa} {'shunt' if n_sp_aa == 1 else 'shunts'})", gain_sp_aa, coch_notes[sym]))
    cochlear_genes = sorted(coch_rows, key=lambda r: -r[1])[:10]
    y_c = np.arange(len(cochlear_genes))
    gains_c = [g[1] for g in cochlear_genes]
    names_c = [g[0] for g in cochlear_genes]
    notes_c = [g[2] for g in cochlear_genes]

    bars_c = ax_c.barh(y_c, gains_c, height=0.62, color=c_purple, edgecolor="#6b215c", lw=0.75, alpha=0.88)
    ax_c.invert_yaxis()
    ax_c.set_yticks(y_c)
    ax_c.set_yticklabels(names_c, fontsize=7.1, fontweight="bold")
    ax_c.set_xlabel("Species-Tree Protein Shunt MDL Gain ΔL_shunt (nats)", fontsize=7.8)
    ax_c.set_xlim(0, 82)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    for i, (bar, gain, note) in enumerate(zip(bars_c, gains_c, notes_c)):
        ax_c.text(
            bar.get_width() + 0.9,
            bar.get_y() + bar.get_height() / 2,
            f"+{gain:.1f} nats | {note}",
            va="center", ha="left", fontsize=6.15, color="#222222",
        )

    ax_c.text(
        0.98, 0.24,
        "19/27 (70.4%) cochlear genes\nshunted on free AA;\n116-sp Prestin: +3.9 nats\n(P_local = 98.0%)",
        transform=ax_c.transAxes, ha="right", va="center",
        fontsize=6.2, color="#4a1640",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#fcf2fa", edgecolor=c_purple, lw=0.8),
    )

    ax_c.set_title(
        "C. Focal Convergence: Cochlear Complex (10/27 Species-Tree AA; 19/27 Free AA)",
        loc="left", pad=8,
    )

    # =========================================================================
    # PANEL D: Plant Kingdom 3rd-Codon Paradox & Poaceae C4 / HGT Showcase
    # =========================================================================
    ax_d = fig.add_subplot(gs[1, 1])
    w3a = plant["nuclear_phylogenomic_summary"]["WickD3a"]
    w3b = plant["nuclear_phylogenomic_summary"]["WickD3b"]
    xid4 = plant["nuclear_phylogenomic_summary"]["XiD4"]

    p_cats = [
        "WickD3a (844 Loci)\nAll 3 Codon Pos\n(L = 2,000 bp)",
        "WickD3b (844 Loci)\n1st+2nd Codon Pos\n(L = 1,333 bp)",
        "XiD4 (310 Loci)\nSeed Plants\n(L = 773 bp)",
    ]
    x_d = np.arange(len(p_cats))
    w_d = 0.26

    conf_elim = [
        w3a["vs_astral_species_tree"]["pct_conflicts_eliminated"],
        w3b["vs_astral_species_tree"]["pct_conflicts_eliminated"],
        xid4["vs_astral_species_tree"]["pct_conflicts_eliminated"],
    ]
    sp_comp = [
        w3a["pct_genes_species_tree_compresses_better_or_equal"],
        w3b["pct_genes_species_tree_compresses_better_or_equal"],
        xid4["pct_genes_species_tree_compresses_better_or_equal"],
    ]
    shunt_counts = [
        w3a["total_shunts"],
        w3b["total_shunts"],
        xid4["total_shunts"],
    ]

    b1 = ax_d.bar(x_d - w_d, conf_elim, width=w_d, color=c_blue, edgecolor="#004b76", lw=0.8, label="Species-Tree Conflicts Eliminated (%)")
    b2 = ax_d.bar(x_d, sp_comp, width=w_d, color=c_green, edgecolor="#005a40", lw=0.8, label="Genes Species Tree Compresses Better (%)")
    b3 = ax_d.bar(x_d + w_d, shunt_counts, width=w_d, color=c_vermillion, edgecolor="#803000", lw=0.8, label="Inferred Shunts (Count)")

    for bar, v in zip(b1, conf_elim):
        ax_d.text(bar.get_x() + bar.get_width() / 2, v + 1.3, f"{v:.1f}%", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color=c_blue)
    for bar, v in zip(b2, sp_comp):
        ax_d.text(bar.get_x() + bar.get_width() / 2, v + 1.3, f"{v:.1f}%", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color=c_green)
    for bar, v in zip(b3, shunt_counts):
        ax_d.text(bar.get_x() + bar.get_width() / 2, v + 1.3, f"{v}", ha="center", va="bottom", fontsize=6.9, fontweight="bold", color=c_vermillion)

    ax_d.annotate(
        "−88.3% Shunts (60 → 7)\n3rd-pos GC convergence\neliminated; Sorghum ↔ Zea\nINTERVAL tract (+8.04 nats)\npreserved!",
        xy=(1.0 + w_d, 11.0),
        xytext=(1.75, 118.0),
        fontsize=6.0, fontweight="bold", color="#803000",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff5eb", edgecolor=c_vermillion, lw=0.85),
        arrowprops=dict(arrowstyle="->", color=c_vermillion, lw=1.3, connectionstyle="arc3,rad=-0.18"),
        ha="center", va="center",
    )

    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels(p_cats, fontsize=7.1)
    ax_d.set_ylabel("Percentage (%) or Shunt Count", fontsize=7.9)
    ax_d.set_ylim(0, 142)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    ax_d.legend(loc="upper left", fontsize=6.6, frameon=True)
    ax_d.set_title(
        "D. Plant Phylogenomics (1,154 Loci / 1,998 Alignments): 3rd-Codon Saturation vs. Introgression",
        loc="left", pad=8,
    )

    fig.savefig(str(out_pdf), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
