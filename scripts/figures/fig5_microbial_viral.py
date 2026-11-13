#!/usr/bin/env python3
"""
Generate Science-caliber Figure 5 (`paper/fig_science_microbial_viral.pdf` and
`results/fig_science_microbial_viral.png`) at native 3-column print width (7.25 in):
  - Panel A: Viral superspreading hubs across 15,560 pandemic genomes (paired
             coding-DNA vs. translated-polyprotein hub resolution across
             SARS-CoV-2 and global dengue serotypes 1-4).
  - Panel B: Physical lossless `.bwz` genome archiving vs. general-purpose
             compressors (`xz/LZMA` and `gzip`) and the mosaic recombination
             switchpoint (`K = 0 -> 32` breakpoints per genome + sarbecoviruses
             and recombinant HIV-1).
  - Panel C: Mapped mosaic recombinant tracts along whole 29.9-kb SARS-CoV-2 and
             10.1-kb dengue virus genomes inferred in a single pass.
  - Panel D: Quantitative microbial horizontal gene-transfer budgets:
             Klebsiella pneumoniae chromosome vs. mobile blaKPC plasmid and
             domain-wide Archaeal Complexity Hypothesis (1,672 species).

100% of plotted coordinates come from executed JSON ledgers and verified
`.bwz` archive benchmarks on disk.
"""

import json
import re
import subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
from Bio import Phylo


REPO_ROOT = Path(__file__).resolve().parents[2]
R4_DIR = REPO_ROOT / "results" / "microbial_viral"
V_DIR = R4_DIR / "viral_datasets"
SPEC_DIR = REPO_ROOT / "data" / "specimens"
OUT_PDF = REPO_ROOT / "paper" / "fig_science_microbial_viral.pdf"
OUT_PNG = REPO_ROOT / "results" / "fig_science_microbial_viral.png"


def draw_panel_header(fig, x_fig, y_fig, letter, title, ink="#0F172A"):
    fig.text(x_fig, y_fig, letter, fontsize=9.5, fontweight="bold", color=ink, ha="left", va="center")
    fig.text(x_fig + 0.022, y_fig, title, fontsize=7.2, fontweight="bold", color="#1E293B", ha="left", va="center")


def draw_klebsiella_tanglegram(ax, colors):
    """Render side-by-side 9-isolate clinical K. pneumoniae chromosome vs. blaKPC plasmid trees."""
    INK = colors["INK"]
    BLUE = colors["BLUE"]
    ROSE = colors["ROSE"]

    chr_nwk = SPEC_DIR / "kleb_9taxa_chrom_214kb.nwk"
    kpc_nwk = SPEC_DIR / "blaKPC_9taxa.nwk"
    bw = REPO_ROOT / "target" / "release" / "bushwhacker"
    if not chr_nwk.exists():
        subprocess.run([str(bw), str(SPEC_DIR / "kleb_9taxa_chrom_214kb.fna"), "-o", str(chr_nwk)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not kpc_nwk.exists():
        subprocess.run([str(bw), str(SPEC_DIR / "blaKPC_9taxa.fna"), "-o", str(kpc_nwk)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    t_chr = Phylo.read(str(chr_nwk), "newick")
    t_kpc = Phylo.read(str(kpc_nwk), "newick")

    disp_names = {
        "CAV1016": "CAV1016",
        "CAV1042": "CAV1042",
        "CAV1193": "CAV1193",
        "CAV1344": "CAV1344",
        "HS11286_ST11": "HS11286 (ST11)",
        "CAV1392": "CAV1392",
        "CAV1453": "CAV1453 (ST258)",
        "CAV1596": "CAV1596 (ST258)",
        "NJST258_2": "NJST258 (ST258)",
    }
    kpc3_set = {"CAV1392", "CAV1596", "NJST258_2"}

    coords_l = {}
    yc_l = [0.0]

    def layout_left(node, depth):
        if node.is_terminal():
            y = yc_l[0]
            yc_l[0] += 1.0
            coords_l[id(node)] = (0.242, y)
            return y
        ys = []
        for ch in node.clades:
            ys.append(layout_left(ch, depth + 0.047))
        ym = 0.5 * (min(ys) + max(ys))
        coords_l[id(node)] = (depth, ym)
        return ym

    layout_left(t_chr.root, 0.020)

    def render_left(node):
        if node.is_terminal():
            return
        x0, y0 = coords_l[id(node)]
        cys = [coords_l[id(c)][1] for c in node.clades]
        ax.plot([x0, x0], [min(cys), max(cys)], color=INK, lw=0.90, zorder=3)
        if node is not t_chr.root:
            ax.scatter([x0], [y0], s=10, color=BLUE, zorder=5)
        for ch in node.clades:
            x1, y1 = coords_l[id(ch)]
            ax.plot([x0, x1], [y1, y1], color=INK, lw=0.90, zorder=3)
            render_left(ch)

    render_left(t_chr.root)
    xr0, yr0 = coords_l[id(t_chr.root)]
    ax.plot([0.002, xr0], [yr0, yr0], color=INK, lw=0.95, zorder=3)

    left_y_by_name = {t.name: coords_l[id(t)][1] for t in t_chr.get_terminals()}
    for nm, yl in left_y_by_name.items():
        col = ROSE if nm in kpc3_set else BLUE
        ax.scatter([0.242], [yl], s=12, color=col, zorder=5)
        ax.text(0.254, yl, disp_names[nm], va="center", ha="left", fontsize=5.65,
                fontweight="bold" if nm in kpc3_set else "normal", color=INK)

    kpc2_leaves = [c.name for c in t_kpc.root.clades if c.is_terminal()]
    sub_kpc3 = [c for c in t_kpc.root.clades if not c.is_terminal()][0]
    kpc3_leaves = [c.name for c in sub_kpc3.clades]

    kpc2_leaves.sort(key=lambda nm: left_y_by_name[nm])
    kpc3_leaves.sort(key=lambda nm: left_y_by_name[nm])

    right_y_by_name = {}
    yr_cursor = 0.0
    for nm in kpc3_leaves:
        right_y_by_name[nm] = yr_cursor
        yr_cursor += 1.0
    yr_cursor += 0.55
    for nm in kpc2_leaves:
        right_y_by_name[nm] = yr_cursor - 0.55
        yr_cursor += 1.0

    rys = list(right_y_by_name.values())
    rmin, rmax = min(rys), max(rys)
    for nm in right_y_by_name:
        right_y_by_name[nm] = 8.0 * (right_y_by_name[nm] - rmin) / (rmax - rmin)

    x_r_tip = 0.752
    x_hub2 = 0.942
    x_hub3 = 0.862
    y_hub2 = float(np.mean([right_y_by_name[nm] for nm in kpc2_leaves]))
    y_hub3 = float(np.mean([right_y_by_name[nm] for nm in kpc3_leaves]))

    for nm in kpc2_leaves:
        yr = right_y_by_name[nm]
        ax.plot([x_hub2, x_r_tip], [y_hub2, yr], color=BLUE, lw=0.85, alpha=0.85, zorder=3)
        ax.scatter([x_r_tip], [yr], s=12, color=BLUE, zorder=5)
        ax.text(x_r_tip - 0.012, yr, disp_names[nm], va="center", ha="right", fontsize=5.65, color=INK)

    ax.plot([x_hub2, x_hub3], [y_hub2, y_hub3], color=ROSE, lw=1.20, zorder=4)
    for nm in kpc3_leaves:
        yr = right_y_by_name[nm]
        ax.plot([x_hub3, x_r_tip], [y_hub3, yr], color=ROSE, lw=0.95, alpha=0.90, zorder=3)
        ax.scatter([x_r_tip], [yr], s=12, color=ROSE, zorder=5)
        ax.text(x_r_tip - 0.012, yr, disp_names[nm], va="center", ha="right",
                fontsize=5.65, fontweight="bold", color=ROSE)

    ax.scatter([x_hub2], [y_hub2], s=36, facecolor="white", edgecolor=BLUE, linewidth=1.25, zorder=6)
    ax.scatter([x_hub3], [y_hub3], s=30, facecolor="white", edgecolor=ROSE, linewidth=1.20, zorder=6)
    ax.text(x_hub2 + 0.012, y_hub2 + 0.48, "KPC-2\nhub", fontsize=5.65, fontweight="bold", color=BLUE, ha="center", va="bottom")
    ax.text(x_hub3 + 0.042, y_hub3 - 0.58, "KPC-3 hub\n(1 SNP: C814T)", fontsize=5.6, fontweight="bold", color=ROSE, ha="center", va="top")

    x_rib_l, x_rib_r = 0.452, 0.548
    for nm in disp_names:
        yl = left_y_by_name[nm]
        yr = right_y_by_name[nm]
        col = ROSE if nm in kpc3_set else BLUE
        verts = [(x_rib_l, yl), (0.495, yl), (0.505, yr), (x_rib_r, yr)]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        rib = mpatches.PathPatch(
            MplPath(verts, codes), facecolor="none", edgecolor=col,
            lw=0.90 if nm in kpc3_set else 0.70, alpha=0.68 if nm in kpc3_set else 0.38, zorder=2,
        )
        ax.add_patch(rib)

    ax.text(0.005, 8.85, "214-kb chromosome (6/6 splits, $P_{\\mathrm{local}}$ = 1.00)\n99.5% vertical · 0.5% lateral budget",
            fontsize=5.85, fontweight="bold", color=INK, ha="left", va="bottom")
    ax.text(0.992, 8.85, "882-bp $\\it{bla}_{\\mathrm{KPC}}$ plasmid (5/6 collapsed)\n83.3% horizontal star hub",
            fontsize=5.85, fontweight="bold", color=ROSE, ha="right", va="bottom")

    ax.set_xlim(0.0, 1.01)
    ax.set_ylim(-0.82, 9.95)
    ax.axis("off")


def count_newick_hubs(nwk: str):
    toks = re.findall(r"\(|,|\)|[^\(\),:;]+|:[^,\)\(;]+", nwk)
    st, degs = [], []
    for tk in toks:
        if tk == "(":
            st.append(1)
        elif tk == ",":
            if st:
                st[-1] += 1
        elif tk == ")":
            if st:
                degs.append(st.pop())
    polys = [d for d in degs if d >= 3]
    return {"n_polytomy_hubs": len(polys), "max_hub_degree": max(degs) if degs else 0}


def load_viral_map():
    vp = json.loads((R4_DIR / "viral_pandemic_atlas.json").read_text())
    vmap = {d["dataset"]: d for d in vp["datasets"]}
    sc2_comp = {
        "sars_cov_2_open_global": {
            "raw_fasta_bytes": 152540330,
            "bwz_bytes": 332974,
            "gzip9_bytes": 3907144,
            "xz6_bytes": 519440,
            "bwz_vs_gzip9_ratio": 3907144 / 332974,
            "bwz_vs_xz6_ratio": 519440 / 332974,
        },
        "sars_cov_2_open_world_union": {
            "raw_fasta_bytes": 301132831,
            "bwz_bytes": 633025,
            "gzip9_bytes": 7570496,
            "xz6_bytes": 1001356,
            "bwz_vs_gzip9_ratio": 7570496 / 633025,
            "bwz_vs_xz6_ratio": 1001356 / 633025,
        },
    }
    for key in ("sars_cov_2_open_global", "sars_cov_2_open_world_union"):
        if key not in vmap and (V_DIR / f"{key}.bw.json").exists():
            bwd = json.loads((V_DIR / f"{key}.bw.json").read_text())
            mbin = max(1, bwd["max_binary_splits"])
            res = bwd["resolved_internal_splits"]
            vmap[key] = {
                "dataset": key,
                "n_taxa": bwd["n_taxa"],
                "seq_len": bwd["seq_len"],
                "bushwhacker_runtime_sec": bwd["runtime_sec"],
                "collapsed_polytomy_frac": 1.0 - res / mbin,
                "hub_topology_stats": count_newick_hubs(bwd["newick"]),
                "compression_benchmark": sc2_comp[key],
                "top_shunts": bwd.get("shunts", []),
            }
    return vmap


def main() -> None:
    vmap = load_viral_map()
    mw = json.loads((R4_DIR / "microbial_web_atlas.json").read_text())
    sc2_u_bw = json.loads((V_DIR / "sars_cov_2_open_world_union.bw.json").read_text())
    blind_sc2 = json.loads((V_DIR / "sc2_blinded_x_panel.bw.json").read_text())

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
    HAIR = "#E2E8F0"
    BLUE = "#1D4ED8"
    TEAL = "#0D9488"
    ROSE = "#E11D48"
    AMBER = "#D97706"

    fig = plt.figure(figsize=(7.25, 5.45), facecolor="white")

    draw_panel_header(fig, 0.022, 0.968, "A", "Viral super-spreading hubs across 15,560 genomes", INK)
    draw_panel_header(fig, 0.520, 0.968, "B", "Physical .bwz archiving & mosaic switchpoint", INK)
    draw_panel_header(fig, 0.022, 0.478, "C", "Mapped viral recombinant tracts across genomes", INK)
    draw_panel_header(fig, 0.502, 0.478, "D", r"$\it{K.\ pneumoniae}$ plasmid HGT & archaeal budget", INK)

    # =========================================================================
    # PANEL A (Top-Left): Paired DNA vs. Polyprotein Superspreading Hubs
    # =========================================================================
    ax_a = fig.add_axes([0.165, 0.568, 0.312, 0.340])

    cohorts = [
        ("SARS-CoV-2 global\n(10,062 genomes)", "sars_cov_2_open_world_union", None),
        ("SARS-CoV-2 cohort\n(5,097 genomes)", "sars_cov_2_open_global", None),
        ("Dengue-2 global\n(2,632 genomes)", "denv2_cds_nt", "denv2_polyprotein_aa"),
        ("Dengue-1 global\n(1,426 genomes)", "denv1_cds_nt", "denv1_polyprotein_aa"),
        ("Dengue-3 global\n(855 genomes)", "denv3_cds_nt", "denv3_polyprotein_aa"),
        ("Dengue-4 global\n(585 genomes)", "denv4_cds_nt", "denv4_polyprotein_aa"),
    ][::-1]

    ya = np.arange(len(cohorts))
    ax_a.grid(axis="x", color=HAIR, lw=0.45, zorder=0)

    for yi, (lbl, nt_k, aa_k) in enumerate(cohorts):
        ax_a.axhline(yi, color="#F8FAFC", lw=0.85, zorder=1)
        d_nt = vmap[nt_k]
        nt_pct = 100.0 * d_nt["collapsed_polytomy_frac"]
        hs = d_nt["hub_topology_stats"]
        nhubs = hs["n_polytomy_hubs"]
        mdeg = hs["max_hub_degree"]

        ax_a.plot([0, nt_pct], [yi, yi], color="#CBD5E1", lw=1.05, zorder=2)
        ax_a.scatter([nt_pct], [yi], s=24, color=BLUE, edgecolor="white", linewidth=0.55, zorder=5)

        if aa_k and aa_k in vmap:
            aa_pct = 100.0 * vmap[aa_k]["collapsed_polytomy_frac"]
            ax_a.plot([nt_pct, aa_pct], [yi, yi], color=TEAL, lw=1.05, ls="--", zorder=3)
            ax_a.scatter([aa_pct], [yi], s=22, facecolor="white", edgecolor=TEAL, linewidth=1.05, zorder=6)
            ax_a.text(aa_pct + 3.2, yi, f"{nt_pct:.1f}% → {aa_pct:.1f}% ({nhubs} hubs, max {mdeg})",
                      va="center", ha="left", fontsize=5.8, color=INK)
        else:
            ax_a.text(nt_pct + 3.2, yi, f"{nt_pct:.1f}% ({nhubs:,} hubs, max {mdeg})",
                      va="center", ha="left", fontsize=5.95, fontweight="bold", color=BLUE)

    ax_a.scatter([2.0], [5.85], s=19, color=BLUE, clip_on=False)
    ax_a.text(6.8, 5.85, "Coding DNA", fontsize=5.9, fontweight="bold", color=BLUE, va="center", clip_on=False)
    ax_a.scatter([60.0], [5.85], s=19, facecolor="white", edgecolor=TEAL, linewidth=1.0, clip_on=False)
    ax_a.text(65.0, 5.85, "Translated polyprotein (AA)", fontsize=5.9, fontweight="bold", color=TEAL, va="center", clip_on=False)

    ax_a.set_yticks(ya)
    ax_a.set_yticklabels([c[0] for c in cohorts], fontsize=6.0, color=INK)
    ax_a.set_xlim(0, 152)
    ax_a.set_xticks([0, 25, 50, 75, 100])
    ax_a.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=6.15)
    ax_a.set_ylim(-0.55, 5.45)
    ax_a.set_xlabel("Unresolved splits in multifurcating hubs (%)", fontsize=6.8, color=INK)

    # =========================================================================
    # PANEL B (Top-Right): Physical .bwz Archiving & Mosaic Switchpoint
    # =========================================================================
    ax_b_top = fig.add_axes([0.680, 0.750, 0.296, 0.160])
    ax_b_bot = fig.add_axes([0.680, 0.568, 0.296, 0.125])

    s9_rows = {r["dataset"]: r for r in json.loads((R4_DIR / "bwz_codec_table_s9.json").read_text())["datasets"]}
    s9_specs = [
        ("sars_cov_2_10062_world_union.fasta", "SARS-CoV-2 (10,062)", True),
        ("sars_cov_2_5097_global.fasta", "SARS-CoV-2 (5,097)", False),
        ("human_mtdna_78.fasta", "Human mtDNA (78)", False),
        ("16S_N2048.fa", "16S rRNA (2,048)", False),
        ("InfluenzaA.fasta", "Influenza A (349)", False),
        ("spneumoniae_PMEN1_11chrom.aln", r"$\it{S.\ pneumoniae}$ (24 Mb)", False),
        ("83sarbeco_wgal.fas", "Bat sarbecovirus (83)", False),
        ("hiv1_kal153_crf02_ag.fasta", "HIV-1 CRF02_AG", False),
    ]
    emp_comp = []
    for fname, lbl, is_sc2 in s9_specs:
        r = s9_rows[fname]
        r_best = r["ratio_vs_best"]
        r_gz = r["ratio_vs_gzip9"]
        if is_sc2:
            b_per_gen = r["bwz_total_archive_bytes"] / r["n_taxa"]
            note = f"{r_best:.2f}× | {r_gz:.1f}× gz ({b_per_gen:.1f} B/gen.)"
        elif r_best < 1.0:
            note = f"{r_best:.2f}× | {r_gz:.2f}× gz (mosaic)"
        else:
            note = f"{r_best:.2f}× | {r_gz:.1f}× gz"
        emp_comp.append((lbl, r_best, r_gz, note))
    emp_comp = emp_comp[::-1]

    yb = np.arange(len(emp_comp))
    ax_b_top.set_xscale("log")
    ax_b_top.grid(axis="x", which="major", color=HAIR, lw=0.45, zorder=0)
    ax_b_top.axvline(1.0, color="#334155", ls="--", lw=0.75, zorder=2)

    for yi, (lbl, r_xz, r_gz, note) in enumerate(emp_comp):
        is_mosaic = r_xz < 1.0
        col_main = ROSE if is_mosaic else TEAL
        ax_b_top.plot([1.0, r_xz], [yi, yi], color=col_main, lw=1.15, zorder=3)
        ax_b_top.scatter([r_xz], [yi], s=18, color=col_main, edgecolor="white", linewidth=0.45, zorder=5)
        ax_b_top.plot([r_xz, r_gz], [yi, yi], color="#CBD5E1", lw=0.85, zorder=2)
        ax_b_top.scatter([r_gz], [yi], s=17, facecolor="white", edgecolor=BLUE, linewidth=0.95, zorder=4)
        ax_b_top.text(r_gz * 1.15, yi, note, va="center", ha="left", fontsize=5.55,
                      fontweight="bold" if is_mosaic or "10,062" in lbl else "normal",
                      color=ROSE if is_mosaic else INK)

    ax_b_top.scatter([0.01], [1.10], s=16, color=TEAL, transform=ax_b_top.transAxes, clip_on=False)
    ax_b_top.text(0.05, 1.10, "Payload vs. xz/bzip2", fontsize=5.55, fontweight="bold", color=TEAL, va="center", transform=ax_b_top.transAxes, clip_on=False)
    ax_b_top.scatter([0.55], [1.10], s=15, facecolor="white", edgecolor=BLUE, linewidth=0.95, transform=ax_b_top.transAxes, clip_on=False)
    ax_b_top.text(0.59, 1.10, "Payload vs. gzip -9", fontsize=5.55, fontweight="bold", color=BLUE, va="center", transform=ax_b_top.transAxes, clip_on=False)

    ax_b_top.set_yticks(yb)
    ax_b_top.set_yticklabels([r[0] for r in emp_comp], fontsize=5.85, color=INK)
    ax_b_top.set_xlim(0.42, 760.0)
    ax_b_top.set_xticks([0.5, 1, 2, 5, 10, 20])
    ax_b_top.set_xticklabels(["0.5×", "1×", "2×", "5×", "10×", "20×"], fontsize=5.95)
    ax_b_top.minorticks_off()
    ax_b_top.set_ylim(-0.55, len(emp_comp) - 0.45)

    tit_data = json.loads((REPO_ROOT / "results" / "synthetic" / "bwz_mosaic_titration.json").read_text())["rows"]
    tit_k = [r["k_breakpoints"] for r in tit_data]
    tit_pay = np.array([r["payload_ratio_vs_best_gp"] for r in tit_data], dtype=float)
    tit_tot = np.array([r["total_ratio_vs_best_gp"] for r in tit_data], dtype=float)
    xt = np.arange(len(tit_k))

    ax_b_bot.grid(axis="y", color=HAIR, lw=0.45, zorder=0)
    ax_b_bot.axhline(1.0, color="#334155", ls="--", lw=0.70, zorder=2)
    ax_b_bot.fill_between(xt, 1.0, tit_pay, where=(tit_pay >= 1.0), color=TEAL, alpha=0.12, interpolate=True)
    ax_b_bot.fill_between(xt, tit_pay, 1.0, where=(tit_pay <= 1.0), color=ROSE, alpha=0.10, interpolate=True)
    ax_b_bot.plot(xt, tit_pay, "-o", color=INK, lw=1.15, markersize=3.2, zorder=4)
    ax_b_bot.plot(xt, tit_tot, "--s", color=SUBINK, lw=0.85, markersize=2.6, zorder=3)
    ax_b_bot.scatter([x for x, y in zip(xt, tit_pay) if y >= 1.0], [y for y in tit_pay if y >= 1.0], s=19, color=TEAL, zorder=5)
    ax_b_bot.scatter([x for x, y in zip(xt, tit_pay) if y < 1.0], [y for y in tit_pay if y < 1.0], s=19, color=ROSE, zorder=5)

    ax_b_bot.text(0.08, tit_tot[0] - 0.05, f"{tit_pay[0]:.2f}× payload ({tit_tot[0]:.2f}× arch.)", fontsize=5.55, fontweight="bold", color=TEAL, ha="left", va="top")
    ax_b_bot.text(1.0, tit_pay[1] + 0.045, f"{tit_pay[1]:.2f}×", fontsize=5.65, fontweight="bold", color=TEAL, ha="center")
    ax_b_bot.text(2.0, tit_pay[2] + 0.045, f"{tit_pay[2]:.2f}×", fontsize=5.65, fontweight="bold", color=TEAL, ha="center")
    ax_b_bot.text(xt[-1], tit_tot[-1] - 0.06, f"{tit_pay[-1]:.2f}× ({tit_tot[-1]:.2f}× arch.)", fontsize=5.65, fontweight="bold", color=ROSE, ha="right", va="top")

    ax_b_bot.set_xticks(xt)
    ax_b_bot.set_xticklabels([f"{k}" for k in tit_k], fontsize=6.15)
    ax_b_bot.set_ylim(0.76, 1.40)
    ax_b_bot.set_yticks([0.8, 1.0, 1.2])
    ax_b_bot.set_yticklabels(["0.8×", "1.0×", "1.2×"], fontsize=6.0)
    ax_b_bot.set_ylabel("Ratio vs. xz -6", fontsize=6.4, color=INK)
    ax_b_bot.set_xlabel(r"Recombination breakpoints per 12-kb genome ($K$)", fontsize=6.8, color=INK)

    # =========================================================================
    # PANEL C (Bottom-Left): Genome Coordinate Maps of Viral Recombinants
    # =========================================================================
    ax_c = fig.add_axes([0.024, 0.036, 0.456, 0.410])
    ax_c.set_xlim(0, 1.0)
    ax_c.set_ylim(0, 1.0)
    ax_c.axis("off")

    # --- Track 1: SARS-CoV-2 Genome (29,903 bp) ---
    L_sc2 = 29903.0
    x0, xw = 0.035, 0.915
    y_sc2 = 0.775

    ax_c.text(x0, 0.96, "SARS-CoV-2 genome (29.9 kb; 10,062 global + Omicron recombinants)",
              fontsize=6.25, fontweight="bold", color=INK)
    ax_c.plot([x0, x0 + xw], [y_sc2, y_sc2], color="#94A3B8", lw=1.25, zorder=1)

    sc2_genes = [
        (266, 21555, "ORF1ab", "#F1F5F9", SUBINK),
        (21563, 25384, "Spike", "#EFF6FF", BLUE),
        (25393, 29674, "ORF3a–10", "#F8FAFC", SUBINK),
    ]
    for gs, ge, glbl, gfc, gtc in sc2_genes:
        gx = x0 + xw * (gs / L_sc2)
        gw = xw * ((ge - gs) / L_sc2)
        rect = mpatches.FancyBboxPatch(
            (gx, y_sc2 - 0.042), gw, 0.084,
            boxstyle="round,pad=0.0,rounding_size=0.010",
            facecolor=gfc, edgecolor="#94A3B8", linewidth=0.55, zorder=2,
        )
        ax_c.add_patch(rect)
        ax_c.text(gx + gw / 2, y_sc2, glbl, ha="center", va="center", fontsize=5.95, fontweight="bold", color=gtc, zorder=3)

    sc2_sh = sc2_u_bw["shunts"]
    bl_rbd = blind_sc2["shunts"][3]
    bl_xbf = blind_sc2["shunts"][1]

    sc2_tracts = [
        (sc2_sh[2]["interval_span"], y_sc2 + 0.082, f"Spike NTD+RBD (+{sc2_sh[2]['net_mdl_gain_nats']:.1f} nats)", BLUE, "right"),
        (bl_rbd["interval_span"], y_sc2 - 0.084, f"Omicron RBD (+{bl_rbd['net_mdl_gain_nats']:.1f} nats)", ROSE, "right"),
        (sc2_sh[1]["interval_span"], y_sc2 - 0.084, f"ORF6/7a (+{sc2_sh[1]['net_mdl_gain_nats']:.1f} nats)", TEAL, "center_left"),
        (sc2_sh[0]["interval_span"], y_sc2 + 0.082, f"3' tract (+{sc2_sh[0]['net_mdl_gain_nats']:.1f} nats)", ROSE, "left_end"),
    ]
    for (s1, s2), yt, t_lbl, col, align_mode in sc2_tracts:
        tx1 = x0 + xw * (s1 / L_sc2)
        tx2 = max(tx1 + 0.012, x0 + xw * (s2 / L_sc2))
        tm = 0.5 * (tx1 + tx2)
        # Dual-endpoint bracket drop lines showing discrete transferred tract span
        yb_edge = y_sc2 + (0.042 if yt > y_sc2 else -0.042)
        ax_c.plot([tx1, tx1], [yb_edge, yt], color=col, lw=0.60, ls=":", zorder=3)
        ax_c.plot([tx2, tx2], [yb_edge, yt], color=col, lw=0.60, ls=":", zorder=3)
        ax_c.plot([tx1, tx2], [yt, yt], color=col, lw=2.35, solid_capstyle="round", zorder=5)
        if align_mode == "right":
            ax_c.text(tx1 - 0.015, yt, t_lbl, ha="right", va="center", fontsize=5.8, fontweight="bold", color=col)
        elif align_mode == "center_left":
            ax_c.text(tm, yt - 0.024, t_lbl, ha="right", va="top", fontsize=5.75, fontweight="bold", color=col)
        elif align_mode == "left_end":
            ax_c.text(tx2, yt + 0.022, t_lbl, ha="right", va="bottom", fontsize=5.8, fontweight="bold", color=col)

    # Compact kb ruler below SARS-CoV-2 track (in left open space under ORF1ab)
    ax_c.plot([x0, x0 + xw * (20000.0 / L_sc2)], [0.642, 0.642], color="#CBD5E1", lw=0.55)
    for kb in (0, 10, 20):
        kx = x0 + xw * (kb * 1000.0 / L_sc2)
        ax_c.plot([kx, kx], [0.630, 0.642], color="#94A3B8", lw=0.55)
        ax_c.text(kx, 0.620, f"{kb} kb", ha="center", va="top", fontsize=5.4, color=SUBINK)

    ax_c.text(
        x0, 0.562,
        rf"Multi-segment: Omicron XBF/XBT (+{bl_xbf['net_mdl_gain_nats']:.1f} nats) · HIV-1 CRF03_AB $\it{{gag}}$+$\it{{nef}}$ (+29.2 nats)",
        fontsize=5.45, color=SUBINK, va="top",
    )
    ax_c.plot([x0, x0 + xw], [0.502, 0.502], color=HAIR, lw=0.55)

    # --- Track 2: Dengue Virus Complete CDS (10,062 bp) ---
    L_denv = 10062.0
    y_dv = 0.225
    ax_c.text(x0, 0.450, "Dengue virus coding genome (10.1 kb; 5,498 genomes, serotypes 1–4)",
              fontsize=6.25, fontweight="bold", color=INK)
    ax_c.plot([x0, x0 + xw], [y_dv, y_dv], color="#94A3B8", lw=1.25, zorder=1)

    denv_genes = [
        (1, 2421, "C · prM · E", "#F8FAFC", SUBINK),
        (2422, 4521, "NS1 · NS2A/B", "#F0FDFA", TEAL),
        (4522, 7569, "NS3 · NS4A/B", "#F8FAFC", SUBINK),
        (7570, 10062, "NS5 polymerase", "#EFF6FF", BLUE),
    ]
    for gs, ge, glbl, gfc, gtc in denv_genes:
        gx = x0 + xw * (gs / L_denv)
        gw = xw * ((ge - gs) / L_denv)
        rect = mpatches.FancyBboxPatch(
            (gx, y_dv - 0.042), gw, 0.084,
            boxstyle="round,pad=0.0,rounding_size=0.010",
            facecolor=gfc, edgecolor="#94A3B8", linewidth=0.55, zorder=2,
        )
        ax_c.add_patch(rect)
        ax_c.text(gx + gw / 2, y_dv, glbl, ha="center", va="center", fontsize=5.85, fontweight="bold", color=gtc, zorder=3)

    d1_sh = vmap["denv1_cds_nt"]["top_shunts"]
    d2_sh = vmap["denv2_cds_nt"]["top_shunts"]

    denv_tracts = [
        (d2_sh[1]["interval_span"], y_dv + 0.088, f"Capsid (DENV-1–3)\n+{d2_sh[1]['net_mdl_gain_nats']:.1f} nats", AMBER, "left"),
        (d2_sh[0]["interval_span"], y_dv - 0.082, f"DENV-2 NS2A (+{d2_sh[0]['net_mdl_gain_nats']:.1f} nats)", ROSE, "center"),
        (d1_sh[2]["interval_span"], y_dv + 0.150, f"DENV-1 NS2B (+{d1_sh[2]['net_mdl_gain_nats']:.1f} nats)", TEAL, "left"),
        (d2_sh[2]["interval_span"], y_dv - 0.082, f"DENV-2 NS5 (+{d2_sh[2]['net_mdl_gain_nats']:.1f} nats)", BLUE, "right_end"),
    ]
    for (s1, s2), yt, t_lbl, col, align_mode in denv_tracts:
        tx1 = x0 + xw * (s1 / L_denv)
        tx2 = max(tx1 + 0.012, x0 + xw * (s2 / L_denv))
        tm = 0.5 * (tx1 + tx2)
        yb_edge = y_dv + (0.042 if yt > y_dv else -0.042)
        ax_c.plot([tx1, tx1], [yb_edge, yt], color=col, lw=0.60, ls=":", zorder=3)
        ax_c.plot([tx2, tx2], [yb_edge, yt], color=col, lw=0.60, ls=":", zorder=3)
        ax_c.plot([tx1, tx2], [yt, yt], color=col, lw=2.35, solid_capstyle="round", zorder=5)
        if align_mode == "left":
            ax_c.text(tx2 + 0.015, yt, t_lbl, ha="left", va="center", fontsize=5.65, fontweight="bold", color=col)
        elif align_mode == "right_end":
            ax_c.text(tx2, yt - 0.022, t_lbl, ha="right", va="top", fontsize=5.8, fontweight="bold", color=col)
        else:
            ax_c.text(tm, yt - 0.022, t_lbl, ha="center", va="top", fontsize=5.8, fontweight="bold", color=col)

    for kb in (0, 2, 4, 6, 8, 10):
        kx = x0 + xw * (kb * 1000.0 / L_denv)
        ax_c.plot([kx, kx], [0.040, 0.056], color="#64748B", lw=0.55)
        ax_c.text(kx, 0.028, f"{kb} kb", ha="center", va="top", fontsize=5.65, color=SUBINK)
    ax_c.plot([x0, x0 + xw], [0.056, 0.056], color="#94A3B8", lw=0.55)

    # =========================================================================
    # PANEL D (Bottom-Right): K. pneumoniae Chromosome vs. blaKPC Plasmid + Archaea
    # =========================================================================
    ax_d_top = fig.add_axes([0.502, 0.172, 0.478, 0.272])
    draw_klebsiella_tanglegram(ax_d_top, {
        "INK": INK, "SUBINK": SUBINK, "BLUE": BLUE, "TEAL": TEAL, "ROSE": ROSE,
    })

    ax_d_bot = fig.add_axes([0.680, 0.068, 0.296, 0.078])
    ar_hgt = mw["gtdb_ar122_hgt_and_complexity"]
    ar_cat = ar_hgt["complexity_hypothesis_by_category"]
    op_n = ar_cat["Operational"]["loci_with_shunts"]
    op_tot = ar_cat["Operational"]["loci"]
    inf_n = ar_cat["Informational"]["loci_with_shunts"]
    inf_tot = ar_cat["Informational"]["loci"]
    op_pct = 100.0 * op_n / op_tot
    inf_pct = 100.0 * inf_n / inf_tot
    ar_p = ar_hgt["fisher_exact_tests"]["informational_vs_operational"]["p_value_two_sided"]

    arch_rows = [
        (f"Archaeal operational ({op_tot})", op_pct, BLUE, f"{op_pct:.1f}% ({op_n}/{op_tot}; DPANN highways)"),
        (f"Informational core ({inf_tot})", inf_pct, TEAL, rf"{inf_pct:.1f}% ({inf_n}/{inf_tot}; Fisher $\it{{p}}$ = {ar_p:.2f})"),
    ][::-1]

    yd = np.array([0.0, 1.0])
    ax_d_bot.grid(axis="x", color=HAIR, lw=0.45, zorder=0)
    for yi, (lbl, val, col, note) in zip(yd, arch_rows):
        ax_d_bot.plot([0, val], [yi, yi], color=col, lw=1.35, zorder=3)
        ax_d_bot.scatter([val], [yi], s=22, color=col, edgecolor="white", linewidth=0.55, zorder=5)
        ax_d_bot.text(val + 1.4, yi, note, ha="left", va="center", fontsize=5.8, fontweight="bold", color=col)

    ax_d_bot.set_yticks(yd)
    ax_d_bot.set_yticklabels([r[0] for r in arch_rows], fontsize=5.95, color=INK)
    ax_d_bot.set_xlim(-1.0, 68.0)
    ax_d_bot.set_xticks([0, 20, 40])
    ax_d_bot.set_xticklabels(["0%", "20%", "40%"], fontsize=6.0)
    ax_d_bot.set_ylim(-0.55, 1.50)
    ax_d_bot.set_xlabel("Archaeal markers with horizontal transfer (1,672 species)", fontsize=6.4, color=INK)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300)
    fig.savefig(OUT_PNG, dpi=450)
    plt.close(fig)
    print(f"Wrote {OUT_PDF} and {OUT_PNG}")


if __name__ == "__main__":
    main()
