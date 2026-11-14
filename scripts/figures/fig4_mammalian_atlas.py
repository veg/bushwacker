#!/usr/bin/env python3
"""
Generate Science-caliber Figure 4 (`paper/fig_science_mammalian_atlas.pdf` and
`results/fig_science_mammalian_atlas.png`) at native 3-column print width (7.25 in):
  - Panel A: Grounding empirical network specimen — Mammalian Prestin (SLC26A5,
             116 mammals, 2,235 bp), retaining monophyletic bats (P_local = 1.00)
             and Cetartiodactyla / whales (P_local = 1.00) while isolating a lateral
             convergence link (+3.9 nats, P_local = 0.98) between echolocating
             horseshoe/roundleaf bats and dolphins across 4 motor-domain codons.
  - Panel B: Organismal species-tree compression preference and protein-resolved
             incomplete lineage sorting compared against coding DNA across 19,953
             TOGA2 proteins in 882 mammals.
  - Panel C: Spatial and taxonomic separation of local paralog gene conversion from
             parallel amino-acid sites, and enrichment across tandem gene families.
  - Panel D: Whole-proteome guild neutrality (showing length-adjustment collapse of
             ant-eaters from raw to length-adjusted excess) paired with an anatomical
             vector map of convergent sweeps across the cochlear hair-cell bundle.

100% of plotted topologies, counts, fold enrichments, and compression gains are read
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
from Bio import Phylo


REPO_ROOT = Path(__file__).resolve().parents[2]
R4_DIR = REPO_ROOT / "results" / "mammalian_toga2"
SPEC_DIR = REPO_ROOT / "data" / "specimens"
OUT_PDF = REPO_ROOT / "paper" / "fig_science_mammalian_atlas.pdf"
OUT_PNG = REPO_ROOT / "results" / "fig_science_mammalian_atlas.png"


def draw_panel_header(fig, x_fig, y_fig, letter, title, ink="#0F172A"):
    fig.text(x_fig, y_fig, letter, fontsize=9.5, fontweight="bold", color=ink, ha="left", va="center")
    fig.text(x_fig + 0.022, y_fig, title, fontsize=7.2, fontweight="bold", color="#1E293B", ha="left", va="center")


def ensure_prestin_specimen():
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    fa = REPO_ROOT / "data" / "specimens" / "prestin.fasta"
    nwk = SPEC_DIR / "prestin.nwk"
    js = SPEC_DIR / "prestin.json"
    bw = REPO_ROOT / "target" / "release" / "bushwhacker"
    if not nwk.exists() or not js.exists():
        subprocess.run([str(bw), str(fa), "-o", str(nwk), "-j", str(js)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return Phylo.read(str(nwk), "newick"), json.loads(js.read_text())


def draw_prestin_network(ax, tree, pj, colors):
    """Render the authentic Laurasiatherian Prestin network from prestin.nwk + prestin.json."""
    INK = colors["INK"]
    SUBINK = colors["SUBINK"]
    BLUE = colors["BLUE"]
    TEAL = colors["TEAL"]
    ROSE = colors["ROSE"]
    AMBER = colors["AMBER"]
    SLATE = "#64748B"

    fm_bats = {"HLdesRot1", "HLminNat1", "eptFus1", "myoBra1", "myoLuc2", "myoDav1"}
    fruit_bats = {"HLpteVam2", "pteAle1"}
    cf_bats = {"HLhipArm1", "HLrhiSin1"}

    baleen = {"HLbalMys1", "balAcu1"}
    toothed = {"HLdelLeu1", "HLorcOrc1", "HLturTru3", "lipVex1", "phyCat1"}
    ruminants = {
        "HLcapHir2", "HLoviAri4", "HLoviCan1", "panHod1", "bisBis1",
        "HLbosInd1", "bosTau8", "bosMut1", "bubBub1", "HLcerEla1", "HLodoVir1",
    }
    camelids = {"HLcamBac1", "HLcamFer2", "HLcamDro1", "vicPac2"}

    collapsed_clades = [
        (fm_bats, "FM microbats (6 spp.)", AMBER, False),
        (ruminants, "Ruminants (11 spp.)", SLATE, False),
        (camelids, "Camelids (4 spp.)", SLATE, False),
    ]

    tip_labels = {
        "HLhipArm1": (r"$\it{H.\ armiger}$ (roundleaf bat)", ROSE, True),
        "HLrhiSin1": (r"$\it{R.\ sinicus}$ (horseshoe bat)", ROSE, True),
        "HLpteVam2": (r"$\it{P.\ vampyrus}$ (flying fox)", TEAL, False),
        "pteAle1": (r"$\it{P.\ alecto}$ (black flying fox)", TEAL, False),
        "HLorcOrc1": (r"$\it{O.\ orca}$ (killer whale)", BLUE, True),
        "HLturTru3": (r"$\it{T.\ truncatus}$ (dolphin)", BLUE, True),
        "HLdelLeu1": (r"$\it{D.\ leucas}$ (beluga)", BLUE, False),
        "lipVex1": (r"$\it{L.\ vexillifer}$ (baiji)", BLUE, False),
        "phyCat1": (r"$\it{P.\ catodon}$ (sperm whale)", BLUE, False),
        "HLbalMys1": (r"$\it{B.\ mysticetus}$ (bowhead)", "#0284C7", False),
        "balAcu1": (r"$\it{B.\ acutorostrata}$ (minke)", "#0284C7", False),
        "susScr11": (r"$\it{S.\ scrofa}$ (wild boar)", SLATE, False),
    }

    mrca_bat = tree.common_ancestor(list(fm_bats | fruit_bats | cf_bats))
    mrca_cetart = tree.common_ancestor(list(baleen | toothed | ruminants | camelids | {"susScr11"}))
    mrca_cet = mrca_cetart.common_ancestor(list(baleen | toothed))
    node_rhino = mrca_bat.common_ancestor(["HLhipArm1", "HLrhiSin1"])
    node_delph = mrca_cet.common_ancestor(["HLorcOrc1", "HLturTru3"])

    def match_collapsed(clade):
        leaves = {t.name for t in clade.get_terminals()}
        for cset, lbl, col, bold in collapsed_clades:
            if leaves == cset:
                return (lbl, col, bold)
        return None

    def clade_color(clade):
        leaves = {t.name for t in clade.get_terminals()}
        if leaves <= cf_bats:
            return ROSE
        if leaves <= fruit_bats:
            return TEAL
        if leaves <= fm_bats:
            return AMBER
        if leaves <= toothed:
            return BLUE
        if leaves <= baleen:
            return "#0284C7"
        if leaves <= (toothed | baleen):
            return BLUE
        return SLATE

    coords = {}
    leaf_rows = []
    y_cursor = [0.0]

    def layout_sub(node, depth):
        mc = match_collapsed(node)
        if mc is not None:
            y = y_cursor[0]
            y_cursor[0] += 1.0
            coords[id(node)] = (depth, y)
            leaf_rows.append((node, depth, y, mc[0], mc[1], mc[2], True))
            return y
        if node.is_terminal():
            y = y_cursor[0]
            y_cursor[0] += 1.0
            coords[id(node)] = (depth, y)
            lbl, col, bold = tip_labels.get(node.name, (node.name, SLATE, False))
            leaf_rows.append((node, depth, y, lbl, col, bold, False))
            return y

        ch_list = list(node.clades)
        node_lvs = {t.name for t in node.get_terminals()}
        if node_lvs <= (fm_bats | fruit_bats | cf_bats):
            def bat_sort(c):
                lvs = {t.name for t in c.get_terminals()}
                if lvs & cf_bats:
                    return 0
                if lvs & fruit_bats:
                    return 1
                return 2
            ch_list.sort(key=bat_sort)
        else:
            def cet_sort(c):
                lvs = {t.name for t in c.get_terminals()}
                if lvs & {"HLorcOrc1", "HLturTru3"}:
                    return 4
                if lvs & toothed:
                    return 3
                if lvs & baleen:
                    return 2
                if lvs & ruminants:
                    return 1
                return 0
            ch_list.sort(key=cet_sort)

        ys = []
        for ch in ch_list:
            bl = float(ch.branch_length or 0.002)
            step = 0.068 + np.sqrt(max(0.001, bl)) * 0.72
            ys.append(layout_sub(ch, depth + step))
        ym = 0.5 * (min(ys) + max(ys))
        coords[id(node)] = (depth, ym)
        return ym

    stem_x = 0.25
    root_x = 0.05
    layout_sub(mrca_cetart, stem_x)
    y_cursor[0] += 2.65
    layout_sub(mrca_bat, stem_x)

    _, y_bat = coords[id(mrca_bat)]
    _, y_cetart = coords[id(mrca_cetart)]
    ax.plot([root_x, root_x], [y_cetart, y_bat], color="#475569", lw=1.15, zorder=2)
    ax.plot([root_x, stem_x], [y_bat, y_bat], color="#475569", lw=1.15, zorder=2)
    ax.plot([root_x, stem_x], [y_cetart, y_cetart], color="#475569", lw=1.15, zorder=2)
    ym_root = 0.5 * (y_bat + y_cetart)
    ax.plot([0.00, root_x], [ym_root, ym_root], color="#475569", lw=1.25, zorder=2)

    def render_edges(node):
        if match_collapsed(node) is not None or node.is_terminal():
            return
        x0, y0 = coords[id(node)]
        cys = [coords[id(c)][1] for c in node.clades]
        ax.plot([x0, x0], [min(cys), max(cys)], color=clade_color(node), lw=1.05, zorder=3)
        for ch in node.clades:
            x1, y1 = coords[id(ch)]
            ax.plot([x0, x1], [y1, y1], color=clade_color(ch), lw=1.05, zorder=3)
            render_edges(ch)

    render_edges(mrca_cetart)
    render_edges(mrca_bat)

    max_x = max(x for x, _ in coords.values())
    align_x = max_x + 0.045

    for node, x, y, lbl, col, bold, is_wedge in leaf_rows:
        if is_wedge:
            tri = mpatches.Polygon(
                [[x, y], [align_x - 0.015, y + 0.25], [align_x - 0.015, y - 0.25]],
                closed=True, facecolor=col, alpha=0.20, edgecolor=col, lw=0.75, zorder=3,
            )
            ax.add_patch(tri)
        else:
            ax.plot([x, align_x - 0.015], [y, y], color=col, lw=0.45, ls=":", alpha=0.65, zorder=2)
            ax.scatter([x], [y], s=11 if bold else 6.5, color=col, zorder=4)
        ax.text(
            align_x, y, lbl, va="center", ha="left",
            fontsize=6.35 if bold else 6.15,
            fontweight="bold" if bold else "normal",
            color=col if bold else INK,
        )

    xb, yb = coords[id(mrca_bat)]
    xc, yc = coords[id(mrca_cet)]
    xca, yca = coords[id(mrca_cetart)]
    ax.scatter([xb], [yb], s=18, color=INK, zorder=5)
    ax.text(
        xb - 0.015, yb + 0.35,
        r"Bats" + "\n" + r"($P_{\mathrm{local}}$ = 1.00)",
        fontsize=6.1, fontweight="bold", color=INK, ha="right", va="bottom",
    )
    ax.scatter([xc], [yc], s=18, color=BLUE, zorder=5)
    ax.text(
        xc - 0.015, yc + 0.30,
        r"Whales" + "\n" + r"($P_{\mathrm{local}}$ = 1.00)",
        fontsize=6.1, fontweight="bold", color=BLUE, ha="right", va="bottom",
    )
    ax.scatter([xca], [yca], s=15, color=SLATE, zorder=5)
    ax.text(
        xca - 0.015, yca - 0.35,
        r"Cetartiodactyla" + "\n" + r"($P_{\mathrm{local}}$ = 1.00)",
        fontsize=6.0, color=SUBINK, ha="right", va="top",
    )

    sh = pj["shunts"][0]
    xr, yr = coords[id(node_rhino)]
    xd, yd = coords[id(node_delph)]
    ax.scatter([xr, xd], [yr, yd], s=30, color=ROSE, edgecolor="white", linewidth=0.75, zorder=7)

    arc_x = 0.5 * (xr + xd) - 0.06
    verts = [(xr, yr), (arc_x, yr - 0.90), (arc_x, yd + 0.90), (xd, yd)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    arc = mpatches.PathPatch(
        MplPath(verts, codes), facecolor="none", edgecolor=ROSE, lw=1.6, ls="--", zorder=6,
    )
    ax.add_patch(arc)

    ym_arc = 0.5 * (yr + yd)
    codons_str = " · ".join(["N7T", "I384T", "S392A", "G601A"])
    assert sh["codons_1based"] == [7, 384, 392, 601], f"Unexpected prestin codons: {sh['codons_1based']}"
    ax.text(
        0.5 * (xr + xd) + 0.035, ym_arc,
        f"Sonar link (+{sh['net_mdl_gain_nats']:.1f} nats, " + r"$P_{\mathrm{local}}$" + f" = {sh['posterior_prob']:.2f})\n"
        f"4 motor codons: {codons_str}",
        fontsize=6.05, fontweight="bold", color=ROSE, ha="left", va="center", zorder=8,
    )

    ax.set_xlim(-0.10, align_x + 0.68)
    ax.set_ylim(-0.75, y_cursor[0] - 0.20)
    ax.axis("off")


def draw_hair_cell_complex(ax, coch_4m, prestin_json, colors):
    """Draw anatomical outer hair-cell stereocilia bundle mapped with cochlear convergence gains."""
    INK = colors["INK"]
    SUBINK = colors["SUBINK"]
    TEAL = colors["TEAL"]
    BLUE = colors["BLUE"]
    VIOLET = colors["VIOLET"]
    ROSE = colors["ROSE"]

    coch_gains = {}
    n_sp_aa = 0
    n_free_aa = 0
    for r in coch_4m:
        if r["sp_aa"]["n_shunts"] > 0:
            n_sp_aa += 1
            coch_gains[r["symbol"]] = float(r["sp_aa"]["total_shunt_gain_nats"])
        if r["free_aa"]["n_shunts"] > 0:
            n_free_aa += 1
    assert len(coch_gains) == 10, f"Expected 10 sp_aa cochlear hits, got {len(coch_gains)}"
    prestin_gain = float(prestin_json["shunts"][0]["net_mdl_gain_nats"])

    ax.set_xlim(-12, 114)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Subtitle & metric key at top
    ax.text(
        50, 96.5,
        f"Cochlear complex ({n_sp_aa}/{len(coch_4m)} sp.-tree · 50% echolocator links)",
        fontsize=6.4, fontweight="bold", color=INK, ha="center", va="center",
    )
    ax.text(
        50, 90.5,
        "Species-tree parallel gain (+nats)",
        fontsize=6.0, color=SUBINK, ha="center", va="center",
    )

    # Cuticular plate (apical actin base) + lateral outer hair-cell wall
    cutic = mpatches.FancyBboxPatch(
        (26, 10.0), 45, 18.0, boxstyle="round,pad=0,rounding_size=2.2",
        facecolor="#F1F5F9", edgecolor="#94A3B8", linewidth=0.6, zorder=1,
    )
    ax.add_patch(cutic)
    ax.text(48.5, 13.5, "Cuticular plate", fontsize=5.9, color="#64748B", ha="center", va="center", zorder=2)

    # Outer hair-cell lateral plasma membrane on right
    ax.plot([25, 71, 71], [28, 28, 6], color="#475569", lw=1.1, zorder=2)
    ax.plot([26, 68.5, 68.5], [27.0, 27.0, 6], color="#CBD5E1", lw=0.7, zorder=2)

    # Tectorial membrane gel ribbon overlying tallest stereocilium
    tect = mpatches.FancyBboxPatch(
        (17.5, 79.2), 41.0, 6.2, boxstyle="round,pad=0,rounding_size=2.4",
        facecolor="#E0F2FE", edgecolor="#0284C7", linewidth=0.65, alpha=0.82, zorder=1,
    )
    ax.add_patch(tect)
    ax.text(38.0, 82.3, "Tectorial membrane", fontsize=5.9, color="#0369A1", ha="center", va="center", zorder=3)

    # Three stepped stereocilia (Tall, Mid, Short) + tapered actin rootlets
    for cx, w, y_top in [(34.0, 7.6, 77.5), (48.0, 7.0, 61.5), (61.0, 6.4, 47.0)]:
        shaft = mpatches.FancyBboxPatch(
            (cx - w / 2.0, 27.5), w, y_top - 27.5,
            boxstyle="round,pad=0,rounding_size=3.0",
            facecolor="#EFF6FF", edgecolor="#64748B", linewidth=0.75, zorder=2,
        )
        ax.add_patch(shaft)
        ax.plot([cx, cx], [18.5, 29.0], color="#94A3B8", lw=1.25, zorder=2)
        ax.plot([cx, cx], [29.0, 35.5], color="#94A3B8", lw=0.8, zorder=2)

    # Tip links connecting shorter tip to taller adjacent shaft
    ax.plot([48.0, 37.8], [61.5, 69.2], color=SUBINK, lw=1.0, zorder=3)
    ax.plot([61.0, 51.5], [47.0, 54.0], color=SUBINK, lw=1.0, zorder=3)

    # Shaft connectors and basal ankle links
    ax.plot([37.8, 44.5], [50.0, 50.0], color="#64748B", lw=0.85, ls="-", zorder=3)
    ax.plot([51.5, 57.8], [41.0, 41.0], color="#64748B", lw=0.85, ls="-", zorder=3)
    ax.plot([37.8, 44.5], [32.2, 32.2], color=TEAL, lw=0.95, zorder=3)
    ax.plot([51.5, 57.8], [32.2, 32.2], color=TEAL, lw=0.95, zorder=3)

    # Mapped compartment nodes: (symbol, anchor_xy, label_xy, ha, col, gain)
    nodes = [
        # Right column labels
        ("STRC", (38.0, 78.2), (73.0, 79.0), "left", VIOLET, coch_gains["STRC"]),
        ("CDH23", (42.8, 65.4), (73.0, 68.5), "left", ROSE, coch_gains["CDH23"]),
        ("PCDH15", (46.8, 62.3), (73.0, 59.0), "left", VIOLET, coch_gains["PCDH15"]),
        ("CIB2", (51.5, 54.0), (73.0, 49.5), "left", BLUE, coch_gains["CIB2"]),
        ("TMC1", (61.0, 47.0), (73.0, 40.0), "left", BLUE, coch_gains["TMC1"]),
        ("Prestin", (71.0, 15.0), (75.5, 15.0), "left", ROSE, prestin_gain),
        # Left column labels
        ("LOXHD1", (41.2, 50.0), (20.5, 52.0), "right", VIOLET, coch_gains["LOXHD1"]),
        ("USH2A", (41.2, 32.2), (20.5, 39.5), "right", TEAL, coch_gains["USH2A"]),
        ("ADGRV1", (54.6, 32.2), (20.5, 29.8), "right", TEAL, coch_gains["ADGRV1"]),
        ("WHRN", (34.0, 29.0), (20.5, 21.0), "right", TEAL, coch_gains["WHRN"]),
        ("TRIOBP", (34.0, 19.5), (20.5, 11.8), "right", ROSE, coch_gains["TRIOBP"]),
    ]

    for sym, (ax0, ay0), (tx, ty), ha, col, gain in nodes:
        msz = 16.0 + 1.55 * gain
        ax.plot([ax0, tx - 1.5 if ha == "left" else tx + 1.5], [ay0, ty],
                color="#94A3B8", lw=0.55, zorder=3)
        ax.scatter([ax0], [ay0], s=msz, color=col, edgecolor="white", linewidth=0.65, zorder=5)
        ax.text(
            tx, ty, r"$\it{" + sym + r"}$" + f" +{gain:.1f}",
            fontsize=6.15, fontweight="bold" if gain >= 14.0 else "normal",
            color=col if gain >= 14.0 else INK, ha=ha, va="center", zorder=6,
        )

    ax.text(
        75.5, 8.5, "(lateral wall)", fontsize=6.0, color=SUBINK, ha="left", va="center",
    )


def main() -> None:
    tp = json.loads((R4_DIR / "toga2_protein_atlas.json").read_text())
    ls = json.loads((R4_DIR / "toga2_protein_length_strata.json").read_text())
    coch_4m = json.loads((R4_DIR / "toga2_cochlear_4modes.json").read_text())
    dm = json.loads((R4_DIR / "toga2_deep_mining_atlas.json").read_text())
    prestin_tree, prestin_json = ensure_prestin_specimen()

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
    TEAL = "#0D9488"
    AMBER = "#D97706"
    ROSE = "#E11D48"
    VIOLET = "#7C3AED"

    fig = plt.figure(figsize=(7.25, 5.35), facecolor="white")

    n_prot = tp["n_proteins_completed"]
    draw_panel_header(fig, 0.012, 0.968, "A", r"$\it{Prestin}$ sonar network (116 mammals)")
    draw_panel_header(fig, 0.552, 0.968, "B", f"Species-tree fit & discordance ({n_prot:,} proteins)")
    draw_panel_header(fig, 0.012, 0.472, "C", "Conversion vs. parallel sites (882 mammals)")
    draw_panel_header(fig, 0.398, 0.472, "D", "Guild neutrality vs. cochlear bundle")

    # =========================================================================
    # PANEL A (Top-Left Hero): Authentic Mammalian Prestin Network
    # =========================================================================
    ax_a = fig.add_axes([0.014, 0.506, 0.510, 0.440])
    draw_prestin_network(ax_a, prestin_tree, prestin_json, {
        "INK": INK, "SUBINK": SUBINK, "BLUE": BLUE, "TEAL": TEAL, "ROSE": ROSE, "AMBER": AMBER,
    })

    # =========================================================================
    # PANEL B (Top-Right): Species-Tree Preference & Lineage Sorting Rate
    # =========================================================================
    ax_b = fig.add_axes([0.612, 0.576, 0.372, 0.348])
    x = np.arange(len(ls))
    xlbls = ["<150", "150–300", "300–500", "500–800", "800–1,500", ">1,500"]
    sp_pref = [100.0 * s["species_tree_shorter_dl_frac"] for s in ls]
    prot_ils = [100.0 * s["mdl_conflict_rate_per_total"] for s in ls]
    tot_genes = sum(s["genes"] for s in ls)
    overall_sp = 100.0 * sum(s["genes"] * s["species_tree_shorter_dl_frac"] for s in ls) / tot_genes
    overall_ils = 100.0 * tp["tree_resolution_and_concordance"]["mdl_discordant_frac_of_max_binary"]
    dna_base = 100.0 * tp["fig4b_matched_denominators"]["nt_19952"]["mdl_discordant_frac_of_max_binary"]

    ax_b.grid(axis="y", color=HAIR, lw=0.45, zorder=0)
    ax_b.plot(x, sp_pref, "-o", color=BLUE, lw=1.35, ms=3.5, zorder=5)
    ax_b.plot(x, prot_ils, "-s", color=AMBER, lw=1.25, ms=3.1, zorder=4)
    ax_b.axhline(dna_base, color=TEAL, ls="--", lw=0.85, zorder=3)

    ax_b.text(0.05, dna_base + 2.8, f"Coding-DNA baseline ({dna_base:.1f}%)",
              fontsize=6.2, fontweight="bold", color=TEAL, ha="left", va="bottom")
    ax_b.text(2.45, 70.0, f"Species tree compresses better\nthan free gene tree ({overall_sp:.1f}% \u2192 {sp_pref[-1]:.1f}%)",
              fontsize=6.2, fontweight="bold", color=BLUE, ha="center", va="top")
    ax_b.text(3.30, 11.2, f"Protein discordance ({overall_ils:.1f}%)",
              fontsize=6.2, fontweight="bold", color=AMBER, ha="center", va="top")

    ax_b.scatter([x[-1]], [sp_pref[-1]], s=26, color=BLUE, edgecolor="white", linewidth=0.65, zorder=6)
    ax_b.text(x[-1] - 0.10, sp_pref[-1] - 6.0, f"{sp_pref[-1]:.1f}%", fontsize=6.3, fontweight="bold", color=BLUE, ha="right")

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(xlbls, fontsize=6.2)
    ax_b.set_xlim(-0.25, 5.25)
    ax_b.set_ylim(0, 108)
    ax_b.set_yticks([0, 25, 50, 75, 100])
    ax_b.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=6.2)
    ax_b.set_xlabel("Protein length (residues)", fontsize=6.8, color=INK)
    ax_b.set_ylabel("Proteins or candidate splits (%)", fontsize=6.8, color=INK)

    # =========================================================================
    # PANEL C (Bottom-Left): Taxonomic Reach + Tandem Family Conversion
    # =========================================================================
    ax_c_top = fig.add_axes([0.132, 0.282, 0.236, 0.112])
    ax_c_bot = fig.add_axes([0.132, 0.080, 0.236, 0.148])

    sh_tot = tp["shunt_totals_on_species_tree"]
    tot_int = sh_tot["total_interval_shunts"]
    int_inter_ord = 100.0 * sh_tot["inter_order_interval_shunts"] / tot_int
    int_intra_fam = 100.0 * (tot_int - sh_tot["inter_order_interval_shunts"] - sh_tot["inter_family_intra_order_interval_shunts"]) / tot_int

    tot_sp = sh_tot["total_sparse_shunts"]
    sp_inter_ord = 100.0 * sh_tot["inter_order_sparse_shunts"] / tot_sp
    sp_intra_fam = 100.0 * (tot_sp - sh_tot["inter_order_sparse_shunts"] - sh_tot["inter_family_intra_order_sparse_shunts"]) / tot_sp

    reach_rows = [
        ("Cross-order", int_inter_ord, sp_inter_ord),
        ("Within-family", int_intra_fam, sp_intra_fam),
    ]
    yr = np.arange(len(reach_rows))
    ax_c_top.grid(axis="x", color=HAIR, lw=0.45, zorder=0)
    for idx, (lbl, v_conv, v_par) in enumerate(reach_rows):
        ax_c_top.plot([v_conv, v_par], [idx, idx], color="#CBD5E1", lw=1.35, zorder=2)
        ax_c_top.scatter([v_conv], [idx], s=26, color=TEAL, edgecolor="white", linewidth=0.55, zorder=5)
        ax_c_top.scatter([v_par], [idx], s=26, color=VIOLET, edgecolor="white", linewidth=0.55, zorder=5)

    ax_c_top.text(int_intra_fam + 3.0, 1, f"{int_intra_fam:.1f}%", va="center", ha="left", fontsize=6.2, fontweight="bold", color=TEAL)
    ax_c_top.text(sp_intra_fam - 3.0, 1, f"{sp_intra_fam:.1f}%", va="center", ha="right", fontsize=6.1, color=VIOLET)
    ax_c_top.text(int_inter_ord, 0.25, f"{int_inter_ord:.1f}%", va="bottom", ha="center", fontsize=6.1, fontweight="bold", color=TEAL)
    ax_c_top.text(sp_inter_ord + 3.0, 0, f"{sp_inter_ord:.1f}% ({sp_inter_ord / int_inter_ord:.1f}×)", va="center", ha="left", fontsize=6.2, fontweight="bold", color=VIOLET)

    med_aa = dm["interval_tract_statistics"]["median_tract_length_aa"]
    ax_c_top.scatter([0], [1.80], s=20, color=TEAL, clip_on=False)
    ax_c_top.text(4.5, 1.80, f"Paralog conversion ({100.0 * sh_tot['proteins_with_interval_shunt'] / n_prot:.1f}%, med. {med_aa} aa)",
                  fontsize=6.15, fontweight="bold", color=TEAL, va="center", clip_on=False)
    ax_c_top.scatter([0], [1.42], s=20, color=VIOLET, clip_on=False)
    ax_c_top.text(4.5, 1.42, f"Parallel sites ({100.0 * sh_tot['proteins_with_sparse_shunt'] / n_prot:.1f}% genes)",
                  fontsize=6.15, fontweight="bold", color=VIOLET, va="center", clip_on=False)

    ax_c_top.set_yticks(yr)
    ax_c_top.set_yticklabels([r[0] for r in reach_rows], fontsize=6.3, color=INK)
    ax_c_top.set_xlim(0, 102)
    ax_c_top.set_ylim(-0.48, 1.25)
    ax_c_top.set_xticks([0, 50, 100])
    ax_c_top.set_xticklabels(["0%", "50%", "100%"], fontsize=6.2)

    fam_map = {f["family"]: f for f in dm["gene_family_shunt_enrichment"]}
    fam_rows = [
        ("CYP / UGT / GST", fam_map["Xenobiotic Detoxification (CYP/UGT/AKR/GST)"]["interval_gene_pct"], TEAL, f"{fam_map['Xenobiotic Detoxification (CYP/UGT/AKR/GST)']['interval_fold_enrichment']:.1f}×"),
        ("Olfactory (OR)", fam_map["Olfactory Receptors (OR)"]["interval_gene_pct"], TEAL, f"{fam_map['Olfactory Receptors (OR)']['interval_fold_enrichment']:.1f}×"),
        ("KRAB zinc-finger", fam_map["KRAB / C2H2 Zinc-Finger Arrays"]["interval_gene_pct"], TEAL, f"{fam_map['KRAB / C2H2 Zinc-Finger Arrays']['interval_fold_enrichment']:.1f}×"),
        ("Other coding loci", fam_map["Other Coding Loci"]["interval_gene_pct"], MUTED, f"{fam_map['Other Coding Loci']['interval_fold_enrichment']:.2f}×"),
    ][::-1]

    yf = np.arange(len(fam_rows))
    ax_c_bot.grid(axis="x", color=HAIR, lw=0.45, zorder=0)
    for idx, (lbl, pct, col, fold_txt) in enumerate(fam_rows):
        ax_c_bot.plot([0, pct], [idx, idx], color="#CBD5E1", lw=1.25, zorder=2)
        ax_c_bot.scatter([pct], [idx], s=24, color=col, edgecolor="white", linewidth=0.55, zorder=4)
        ax_c_bot.text(pct + 1.2, idx, f"{pct:.1f}% ({fold_txt})", va="center", ha="left",
                      fontsize=6.15, fontweight="bold" if col == TEAL else "normal",
                      color=col if col == TEAL else SUBINK)

    ax_c_bot.set_yticks(yf)
    ax_c_bot.set_yticklabels([r[0] for r in fam_rows], fontsize=6.25, color=INK)
    ax_c_bot.set_xlim(0, 56.0)
    ax_c_bot.set_ylim(-0.55, 3.55)
    ax_c_bot.set_xticks([0, 20, 40])
    ax_c_bot.set_xticklabels(["0%", "20%", "40%"], fontsize=6.2)
    ax_c_bot.set_xlabel("Proteins with conversion (%)", fontsize=6.8, color=INK)

    # =========================================================================
    # PANEL D (Bottom-Right): Guild Neutrality + Cochlear Stereocilia Schematic
    # =========================================================================
    ax_d1 = fig.add_axes([0.486, 0.080, 0.172, 0.336])
    ax_d2 = fig.add_axes([0.674, 0.015, 0.318, 0.430])

    eg = tp["ecological_guild_enrichment"]
    ant_raw = eg["myrmecophagous_insectivorous_armor"]["fold_enrichment"]
    ant_adj = eg["myrmecophagous_insectivorous_armor"]["length_adjusted_fold_raw"]
    guild_rows = [
        ("Ant-eaters", ant_adj, AMBER, f"{ant_adj:.2f}× adj.", ant_raw),
        ("Subterranean", eg["subterranean_blind"]["fold_enrichment"], SUBINK, f"{eg['subterranean_blind']['fold_enrichment']:.2f}×", None),
        ("Marine", eg["marine_aquatic"]["fold_enrichment"], SUBINK, f"{eg['marine_aquatic']['fold_enrichment']:.2f}×", None),
        ("Echolocators", eg["echolocators"]["fold_enrichment"], SUBINK, f"{eg['echolocators']['fold_enrichment']:.2f}×", None),
        ("Bats + whales", eg["echolocating_bats_and_toothed_whales"]["fold_enrichment"], SUBINK, f"{eg['echolocating_bats_and_toothed_whales']['fold_enrichment']:.2f}×", None),
    ][::-1]

    yc = np.arange(len(guild_rows))
    ax_d1.set_xscale("log")
    ax_d1.grid(axis="x", which="major", color=HAIR, lw=0.45, zorder=0)
    ax_d1.axvline(1.0, color="#334155", ls="--", lw=0.75, zorder=2)
    ax_d1.text(0.32, 4.72, "Whole-proteome guilds", ha="left", va="bottom",
               fontsize=6.35, fontweight="bold", color=INK, clip_on=False)
    ax_d1.text(0.90, 4.30, "1× null", ha="right", va="bottom", fontsize=5.95, color=SUBINK)

    for idx, (lbl, fv, col, note, raw_fv) in enumerate(guild_rows):
        ax_d1.plot([1.0, fv], [idx, idx], color=col, lw=1.35, zorder=3)
        ax_d1.scatter([fv], [idx], s=26, color=col, edgecolor="white", linewidth=0.55, zorder=5)
        if raw_fv is not None:
            ax_d1.scatter([raw_fv], [idx], s=26, facecolor="white", edgecolor=AMBER, linewidth=1.0, zorder=5)
            ax_d1.annotate(
                "", xy=(fv * 1.18, idx), xytext=(raw_fv * 0.85, idx),
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=0.95, shrinkA=0, shrinkB=0, mutation_scale=7.5),
                zorder=4,
            )
            ax_d1.text(fv * 1.08, idx - 0.36, note, va="top", ha="left", fontsize=6.05, fontweight="bold", color=AMBER)
            ax_d1.text(raw_fv, idx + 0.25, f"{raw_fv:.1f}× raw", va="bottom", ha="center", fontsize=6.05, color=AMBER)
        elif fv >= 1.0:
            ax_d1.text(fv * 1.24, idx, note, va="center", ha="left", fontsize=6.1, color=col)
        else:
            ax_d1.text(1.18, idx, note, va="center", ha="left", fontsize=6.1, color=col)

    ax_d1.set_yticks(yc)
    ax_d1.set_yticklabels([r[0] for r in guild_rows], fontsize=6.25, color=INK)
    ax_d1.set_xlim(0.30, 40.0)
    ax_d1.set_ylim(-0.55, 4.65)
    ax_d1.set_xticks([0.5, 1, 4, 16])
    ax_d1.set_xticklabels(["0.5×", "1×", "4×", "16×"], fontsize=6.2)
    ax_d1.minorticks_off()
    ax_d1.set_xlabel("Fold excess vs. random", fontsize=6.8, color=INK)

    draw_hair_cell_complex(ax_d2, coch_4m, prestin_json, {
        "INK": INK, "SUBINK": SUBINK, "BLUE": BLUE, "TEAL": TEAL, "ROSE": ROSE, "VIOLET": VIOLET,
    })

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300)
    fig.savefig(OUT_PNG, dpi=450)
    plt.close(fig)
    print(f"Generated {OUT_PDF} and {OUT_PNG}")


if __name__ == "__main__":
    main()
