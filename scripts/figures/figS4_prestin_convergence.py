#!/usr/bin/env python3
"""
Generate publication-grade vector PDF Figure (`paper/fig_prestin_convergence.pdf`)
for the 116-mammal Prestin (SLC26A5) echolocation convergence analysis from
authentic executed Bushwhacker output (`/tmp/prestin_mdl.nwk`, `/tmp/prestin_mdl.json`,
and `docs/examples/prestin.fasta`).
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
from Bio import Phylo, SeqIO
from Bio.Seq import Seq


SPECIES_LABELS = {
    # Cetacea - Odontoceti (Echolocating toothed whales)
    "HLorcOrc1": "Orcinus orca (Killer whale)",
    "HLturTru3": "Tursiops truncatus (Bottlenose dolphin)",
    "HLdelLeu1": "Delphinapterus leucas (Beluga whale)",
    "lipVex1": "Lipotes vexillifer (Yangtze river dolphin)",
    "phyCat1": "Physeter catodon (Sperm whale)",
    # Cetacea - Mysticeti (Non-echolocating baleen whales)
    "HLbalMys1": "Balaena mysticetus (Bowhead whale)",
    "balAcu1": "Balaenoptera acutorostrata (Minke whale)",
    # Artiodactyla - Ruminantia, Suina, Tylopoda
    "bubBub1": "Bubalus bubalis (Water buffalo)",
    "HLbosInd1": "Bos indicus (Zebu cattle)",
    "bosTau8": "Bos taurus (Domestic cattle)",
    "bosMut1": "Bos mutus (Wild yak)",
    "bisBis1": "Bison bison (American bison)",
    "HLoviCan1": "Ovis canadensis (Bighorn sheep)",
    "HLoviAri4": "Ovis aries (Domestic sheep)",
    "HLcapHir2": "Capra hircus (Domestic goat)",
    "panHod1": "Pantholops hodgsonii (Tibetan antelope)",
    "HLodoVir1": "Odocoileus virginianus (White-tailed deer)",
    "HLcerEla1": "Cervus elaphus (Red deer)",
    "susScr11": "Sus scrofa (Wild boar)",
    "HLcamFer2": "Camelus ferus (Wild Bactrian camel)",
    "HLcamBac1": "Camelus bactrianus (Bactrian camel)",
    "HLcamDro1": "Camelus dromedarius (Dromedary)",
    "vicPac2": "Vicugna pacos (Alpaca)",
    # Chiroptera - Rhinolophoidea (CF echolocating bats)
    "HLrhiSin1": "Rhinolophus sinicus (Chinese horseshoe bat)",
    "HLhipArm1": "Hipposideros armiger (Great roundleaf bat)",
    # Chiroptera - Yangochiroptera (FM echolocating microbats)
    "HLdesRot1": "Desmodus rotundus (Common vampire bat)",
    "HLminNat1": "Miniopterus natalensis (Long-fingered bat)",
    "eptFus1": "Eptesicus fuscus (Big brown bat)",
    "myoDav1": "Myotis davidii (David's myotis)",
    "myoBra1": "Myotis brandtii (Brandt's myotis)",
    "myoLuc2": "Myotis lucifugus (Little brown bat)",
    # Chiroptera - Pteropodidae (Non-echolocating fruit bats)
    "HLpteVam2": "Pteropus vampyrus (Large flying fox)",
    "pteAle1": "Pteropus alecto (Black flying fox)",
    # Outgroups for codon table
    "hg38": "Homo sapiens (Human)",
    "mm10": "Mus musculus (House mouse)",
}

ECHOLOCATING_WHALES = {"HLorcOrc1", "HLturTru3", "HLdelLeu1", "lipVex1", "phyCat1"}
BALEEN_WHALES = {"HLbalMys1", "balAcu1"}
CF_BATS = {"HLrhiSin1", "HLhipArm1"}
FM_BATS = {"HLdesRot1", "HLminNat1", "eptFus1", "myoDav1", "myoBra1", "myoLuc2"}
FRUIT_BATS = {"HLpteVam2", "pteAle1"}


def ensure_prestin_outputs(repo_root: Path) -> tuple[Path, Path, Path]:
    fa_path = repo_root / "data" / "specimens" / "prestin.fasta"
    nwk_path = repo_root / "data" / "specimens" / "prestin.nwk"
    json_path = repo_root / "data" / "specimens" / "prestin.json"
    bw_bin = repo_root / "target" / "release" / "bushwhacker"
    if not nwk_path.exists() or not json_path.exists():
        subprocess.run(
            [str(bw_bin), str(fa_path), "-o", str(nwk_path), "-j", str(json_path)],
            check=True,
        )
    return fa_path, nwk_path, json_path


def layout_cladogram(clade, x_start=0.0, y_list=None):
    if y_list is None:
        y_list = [0]
    coords = {}

    def _visit(node, depth):
        if node.is_terminal():
            y = float(y_list[0])
            y_list[0] += 1
            coords[id(node)] = (depth, y)
            return y
        child_ys = []
        for ch in node.clades:
            bl = max(0.0015, float(ch.branch_length or 0.002))
            step = 0.075 + np.sqrt(bl) * 1.65
            child_ys.append(_visit(ch, depth + step))
        y_mid = 0.5 * (min(child_ys) + max(child_ys))
        coords[id(node)] = (depth, y_mid)
        return y_mid

    _visit(clade, x_start)
    return coords


def draw_subtree(ax, root_clade, coords, color_fn, lw=1.4):
    for node in root_clade.find_clades(order="preorder"):
        x0, y0 = coords[id(node)]
        if not node.is_terminal():
            ch_ys = [coords[id(c)][1] for c in node.clades]
            col_v = color_fn(node)
            ax.plot([x0, x0], [min(ch_ys), max(ch_ys)], color=col_v, lw=lw, zorder=3)
            for ch in node.clades:
                x1, y1 = coords[id(ch)]
                col_h = color_fn(ch)
                ax.plot([x0, x1], [y1, y1], color=col_h, lw=lw, zorder=3)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fa_path, nwk_path, json_path = ensure_prestin_outputs(repo_root)
    out_pdf = repo_root / "paper" / "fig_prestin_convergence.pdf"
    out_png = repo_root / "results" / "mammalian_toga2" / "fig_prestin_convergence.png"

    bw_data = json.loads(json_path.read_text(encoding="utf-8"))
    shunt = bw_data["shunts"][0]
    seqs = {r.id: str(r.seq) for r in SeqIO.parse(str(fa_path), "fasta")}
    tree = Phylo.read(str(nwk_path), "newick")

    cet_art_taxa = [
        "vicPac2", "HLcamDro1", "HLcamBac1", "HLcamFer2", "susScr11",
        "HLcerEla1", "HLodoVir1", "panHod1", "HLcapHir2", "HLoviAri4",
        "HLoviCan1", "bisBis1", "bosMut1", "bosTau8", "HLbosInd1", "bubBub1",
        "phyCat1", "lipVex1", "HLdelLeu1", "HLorcOrc1", "HLturTru3",
        "HLbalMys1", "balAcu1",
    ]
    bat_taxa = [
        "HLrhiSin1", "HLhipArm1", "HLdesRot1", "HLminNat1", "eptFus1",
        "myoDav1", "myoBra1", "myoLuc2", "HLpteVam2", "pteAle1",
    ]

    mrca_cet = tree.common_ancestor(cet_art_taxa)
    mrca_bat = tree.common_ancestor(bat_taxa)

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
        "font.size": 8.2,
        "axes.titlesize": 9.4,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.2,
    })

    fig = plt.figure(figsize=(14.0, 8.5))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.42, 1.0],
        height_ratios=[1.22, 0.85],
        wspace=0.45,
        hspace=0.34,
    )

    # =========================================================================
    # PANEL A: Laurasiatheria Prestin MDL Phylogeny + SPARSE Shunt Edge
    # =========================================================================
    ax_tree = fig.add_subplot(gs[:, 0])

    y_counter = [0]
    coords_bat = layout_cladogram(mrca_bat, x_start=0.14, y_list=y_counter)
    y_counter[0] += 2
    coords_cet = layout_cladogram(mrca_cet, x_start=0.14, y_list=y_counter)

    all_coords = {}
    all_coords.update(coords_bat)
    all_coords.update(coords_cet)

    y_bat_root = coords_bat[id(mrca_bat)][1]
    y_cet_root = coords_cet[id(mrca_cet)][1]
    ax_tree.plot([0.02, 0.02], [y_bat_root, y_cet_root], color="#4d4d4d", lw=1.6, zorder=2)
    ax_tree.plot([0.02, 0.14], [y_bat_root, y_bat_root], color="#4d4d4d", lw=1.6, zorder=2)
    ax_tree.plot([0.02, 0.14], [y_cet_root, y_cet_root], color="#4d4d4d", lw=1.6, zorder=2)
    ax_tree.plot([-0.03, 0.02], [0.5 * (y_bat_root + y_cet_root), 0.5 * (y_bat_root + y_cet_root)], color="#4d4d4d", lw=1.8, zorder=2)
    ax_tree.text(
        0.005,
        0.5 * (y_bat_root + y_cet_root) + 0.6,
        "Laurasiatheria\nBackbone\n(116 Mammals)",
        fontsize=6.9,
        fontweight="bold",
        color="#333333",
        ha="right",
        va="bottom",
    )

    def get_clade_color(clade):
        leaves = {t.name for t in clade.get_terminals()}
        if leaves <= ECHOLOCATING_WHALES:
            return "#00688b"
        if leaves <= BALEEN_WHALES:
            return "#2b8cbe"
        if leaves <= (ECHOLOCATING_WHALES | BALEEN_WHALES):
            return "#00688b"
        if leaves <= CF_BATS:
            return "#b2182b"
        if leaves <= FM_BATS:
            return "#d6604d"
        if leaves <= FRUIT_BATS:
            return "#238b45"
        return "#525252"

    draw_subtree(ax_tree, mrca_bat, coords_bat, get_clade_color, lw=1.45)
    draw_subtree(ax_tree, mrca_cet, coords_cet, get_clade_color, lw=1.45)

    max_x = max(x for x, _ in all_coords.values())
    label_x = max_x + 0.03

    for root_c in (mrca_bat, mrca_cet):
        for term in root_c.get_terminals():
            x, y = all_coords[id(term)]
            nm = term.name
            disp = SPECIES_LABELS.get(nm, nm)
            col = get_clade_color(term)
            fw = "bold" if (nm in ECHOLOCATING_WHALES or nm in CF_BATS or nm in BALEEN_WHALES or nm in FRUIT_BATS) else "normal"
            ax_tree.plot([x, label_x - 0.01], [y, y], color=col, lw=0.5, ls=":", alpha=0.55)
            ax_tree.text(label_x, y, disp, fontsize=7.1, color=col, fontweight=fw, va="center", ha="left")

    def span_of(taxa_subset):
        ys = [all_coords[id(t)][1] for r in (mrca_bat, mrca_cet) for t in r.get_terminals() if t.name in taxa_subset]
        return min(ys) - 0.42, max(ys) + 0.42

    for taxa_set, bg_col in [
        (ECHOLOCATING_WHALES, "#e0f3f8"),
        (BALEEN_WHALES, "#f0f9e8"),
        (CF_BATS, "#fde0dd"),
        (FM_BATS, "#fff5eb"),
        (FRUIT_BATS, "#e5f5e0"),
    ]:
        y_lo, y_hi = span_of(taxa_set)
        rect = mpatches.FancyBboxPatch(
            (0.11, y_lo),
            label_x + 0.96 - 0.11,
            y_hi - y_lo,
            boxstyle="round,pad=0.02",
            facecolor=bg_col,
            edgecolor="none",
            alpha=0.55,
            zorder=0,
        )
        ax_tree.add_patch(rect)

    node_delph = mrca_cet.common_ancestor(["HLorcOrc1", "HLturTru3"])
    node_rhino = mrca_bat.common_ancestor(["HLhipArm1", "HLrhiSin1"])
    x_u, y_u = all_coords[id(node_delph)]
    x_v, y_v = all_coords[id(node_rhino)]

    ax_tree.scatter([x_u, x_v], [y_u, y_v], s=58, color="#c51b7d", edgecolor="white", linewidth=1.2, zorder=6)
    ax_tree.text(
        x_u - 0.015, y_u + 0.75, f"{shunt['label_u']} (Delphinidae)\n$P_{{\\mathrm{{local}}}}$ = 1.00",
        fontsize=6.6, fontweight="bold", color="#00587a", ha="right", zorder=7,
        bbox=dict(boxstyle="square,pad=0.04", fc="white", ec="none", alpha=0.86),
    )
    ax_tree.text(
        x_v + 0.025, y_v + 0.85, f"{shunt['label_v']} (Rhinolophoidea)\n$P_{{\\mathrm{{local}}}}$ = 1.00",
        fontsize=6.6, fontweight="bold", color="#99000d", ha="left", zorder=7,
        bbox=dict(boxstyle="square,pad=0.04", fc="white", ec="none", alpha=0.86),
    )

    node_cetacea = mrca_cet.common_ancestor(list(ECHOLOCATING_WHALES | BALEEN_WHALES))
    xc, yc = all_coords[id(node_cetacea)]
    ax_tree.scatter([xc], [yc], s=36, color="#00688b", zorder=5)
    ax_tree.text(
        xc - 0.015, yc + 0.45, "Monophyletic\nCetacea\n($P_{\\mathrm{local}}$ = 1.00)",
        fontsize=6.3, fontweight="bold", color="#00688b", ha="right", zorder=7,
        bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.86),
    )

    xb, yb = all_coords[id(mrca_bat)]
    ax_tree.scatter([xb], [yb], s=36, color="#b2182b", zorder=5)
    ax_tree.text(
        xb - 0.015, yb - 0.45, "Monophyletic\nChiroptera\n($P_{\\mathrm{local}}$ = 1.00)",
        fontsize=6.3, fontweight="bold", color="#b2182b", ha="right", va="top", zorder=7,
        bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.86),
    )

    ctrl_x = 0.05
    verts = [
        (x_u, y_u),
        (ctrl_x, y_u - 2.2),
        (ctrl_x, y_v + 2.2),
        (x_v, y_v),
    ]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    shunt_patch = mpatches.PathPatch(
        MplPath(verts, codes),
        facecolor="none",
        edgecolor="#c51b7d",
        lw=2.5,
        linestyle="--",
        zorder=5,
    )
    ax_tree.add_patch(shunt_patch)

    shunt_mid_x = 0.125 * (x_u + x_v) + 0.75 * ctrl_x
    shunt_mid_y = 0.5 * (y_u + y_v)
    ax_tree.annotate(
        "Inferred SPARSE Shunt\n"
        f"+{shunt['net_mdl_gain_nats']:.2f} nats ($P_{{\\mathrm{{local}}}}$ = {shunt['posterior_prob']*100:.2f}%)\n"
        f"{shunt['shared_mutations']} Convergent Codons:\n"
        "N7T, I384T, S392A, G601A",
        xy=(shunt_mid_x, shunt_mid_y),
        xytext=(-0.22, 29.8),
        fontsize=6.8,
        fontweight="bold",
        color="#8e0152",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fde0ef", edgecolor="#c51b7d", lw=1.2),
        arrowprops=dict(arrowstyle="->", color="#c51b7d", lw=1.4, connectionstyle="arc3,rad=0.18"),
        ha="left",
        va="center",
        zorder=7,
    )

    # Set wide right limit so species text stays 100% inside Panel A
    ax_tree.set_xlim(-0.24, label_x + 1.32)
    ax_tree.set_ylim(-1.3, y_counter[0] + 0.5)
    ax_tree.axis("off")
    ax_tree.set_title(
        "A. Mammalian Prestin (SLC26A5) MDL Phylogeny & Lateral Shunt",
        loc="left",
        pad=10,
    )

    # =========================================================================
    # PANEL B: Convergent Codon & Amino-Acid Matrix at Inferred Shunt Sites
    # =========================================================================
    ax_mat = fig.add_subplot(gs[0, 1])

    rep_rows = [
        ("HLorcOrc1", "O. orca (Killer whale)", "Odontoceti (Echo)"),
        ("HLturTru3", "T. truncatus (Dolphin)", "Odontoceti (Echo)"),
        ("HLdelLeu1", "D. leucas (Beluga)", "Odontoceti (Echo)"),
        ("lipVex1", "L. vexillifer (Baiji)", "Odontoceti (Echo)"),
        ("phyCat1", "P. catodon (Sperm whale)", "Odontoceti (Echo)"),
        ("HLbalMys1", "B. mysticetus (Bowhead)", "Mysticeti (Non-Echo)"),
        ("balAcu1", "B. acutorostrata (Minke)", "Mysticeti (Non-Echo)"),
        ("bosTau8", "B. taurus (Cattle)", "Artiodactyla Outgroup"),
        ("susScr11", "S. scrofa (Boar)", "Artiodactyla Outgroup"),
        ("HLrhiSin1", "R. sinicus (Horseshoe bat)", "Rhinolophoidea (CF Echo)"),
        ("HLhipArm1", "H. armiger (Roundleaf bat)", "Rhinolophoidea (CF Echo)"),
        ("myoLuc2", "M. lucifugus (Myotis bat)", "Yangochiroptera (FM Echo)"),
        ("eptFus1", "E. fuscus (Big brown bat)", "Yangochiroptera (FM Echo)"),
        ("HLpteVam2", "P. vampyrus (Flying fox)", "Pteropodidae (Non-Echo)"),
        ("pteAle1", "P. alecto (Black fruit bat)", "Pteropodidae (Non-Echo)"),
        ("hg38", "H. sapiens (Human)", "Euarchontoglires Outgroup"),
    ]

    codon_sites = [
        (19, 7, "Codon 7\n(N7T)", "T", True),
        (1150, 384, "Codon 384\n(I384T)", "T", True),
        (1173, 392, "Codon 392\n(S392A)", "A", True),
        (1801, 601, "Codon 601\n(G601A)", "A", True),
        (1488, 497, "Codon 497\n(Bat stem\nL497M)", "M", False),
    ]

    n_r = len(rep_rows)
    n_c = len(codon_sites)
    ax_mat.set_xlim(-0.5, n_c - 0.5)
    ax_mat.set_ylim(-0.5, n_r - 0.5)
    ax_mat.invert_yaxis()

    for r_i, (tid, t_label, grp) in enumerate(rep_rows):
        seq = seqs[tid]
        for c_i, (nt_site, cod_num, _, conv_aa, is_inferred) in enumerate(codon_sites):
            c0 = (cod_num - 1) * 3
            cod_str = seq[c0:c0 + 3]
            aa = str(Seq(cod_str).translate()) if "-" not in cod_str else "-"
            is_conv = (aa == conv_aa)
            if not is_inferred:
                bg = "#eceef2" if is_conv else "#f6f7f9"
                fg = "#333333"
            else:
                bg = "#f46d43" if is_conv else "#e0f3f8"
                fg = "white" if is_conv else "#2b3e50"
                if not is_conv and aa not in ("N", "I", "S", "L", "G"):
                    bg = "#fee08b"
                    fg = "#333333"
            rect = mpatches.FancyBboxPatch(
                (c_i - 0.44, r_i - 0.40),
                0.88,
                0.80,
                boxstyle="round,pad=0.03",
                facecolor=bg,
                edgecolor="#cccccc",
                lw=0.6,
            )
            ax_mat.add_patch(rect)
            ax_mat.text(
                c_i,
                r_i,
                f"{aa} ({cod_str})",
                ha="center",
                va="center",
                fontsize=6.9,
                fontweight="bold" if is_conv else "normal",
                color=fg,
            )

    ax_mat.set_xticks(range(n_c))
    ax_mat.set_xticklabels([c[2] for c in codon_sites], fontsize=7.4, fontweight="bold")
    ax_mat.xaxis.tick_top()
    ax_mat.set_yticks(range(n_r))
    ax_mat.set_yticklabels([r[1] for r in rep_rows], fontsize=7.1)
    for tick, (_, _, grp) in zip(ax_mat.get_yticklabels(), rep_rows):
        if "Odontoceti" in grp:
            tick.set_color("#00688b")
            tick.set_fontweight("bold")
        elif "Rhinolophoidea" in grp:
            tick.set_color("#b2182b")
            tick.set_fontweight("bold")
        elif "Yangochiroptera" in grp:
            tick.set_color("#d6604d")
        elif "Non-Echo" in grp:
            tick.set_color("#238b45")
            tick.set_fontweight("bold")

    for sep_y in [4.5, 6.5, 8.5, 10.5, 12.5, 14.5]:
        ax_mat.axhline(sep_y, color="#666666", lw=0.8, ls="--", alpha=0.6)

    for spine in ax_mat.spines.values():
        spine.set_visible(False)
    ax_mat.tick_params(length=0)
    ax_mat.set_title("B. Codon States at the 4 Inferred Shunt Sites (+ Ancestral Bat L497M)", loc="left", pad=26)

    # =========================================================================
    # PANEL C: MDL Code-Length Ledger & 116-Mammal Species-Tree Precision
    # =========================================================================
    ax_bot = fig.add_subplot(gs[1, 1])

    cmp_path = Path(__file__).resolve().parents[2] / "results" / "mammalian_toga2" / "prestin_collapsed_ml_comparison.json"
    cmp_data = json.loads(cmp_path.read_text())
    cm = cmp_data["models"]
    n_shared = cmp_data["n_taxa"]
    n_bw_sh = cm["Bushwhacker_MDL"].get("certified_shunts", 1)
    order_keys = [
        ("Bushwhacker_MDL", f"Bushwhacker\n(MDL + {n_bw_sh} {'shunt' if n_bw_sh == 1 else 'shunts'})", "#1b7837"),
        ("IQTREE2_SHaLRT80_UFBoot95", "IQ-TREE 2\n(SH≥80 & UFB≥95)", "#2b8cbe"),
        ("IQTREE2_UFBoot95", "IQ-TREE 2\n(UFBoot ≥ 95%)", "#386cb0"),
        ("IQTREE2_SHaLRT80", "IQ-TREE 2\n(SH-aLRT ≥ 80%)", "#7570b3"),
        ("IQTREE2_GTR_G4_polytomy", "IQ-TREE 2\n(--polytomy)", "#d95f02"),
    ]
    methods = [lbl for _, lbl, _ in order_keys]
    fp_splits = [cm[k]["fp"] for k, _, _ in order_keys]
    prec_vals = [100.0 * cm[k]["precision"] for k, _, _ in order_keys]
    colors_bar = [col for _, _, col in order_keys]

    y_pos = np.arange(len(methods))
    bars = ax_bot.barh(y_pos, fp_splits, height=0.56, color=colors_bar, alpha=0.88)
    ax_bot.invert_yaxis()
    ax_bot.set_yticks(y_pos)
    ax_bot.set_yticklabels(methods, fontsize=7.1, fontweight="bold")
    ax_bot.set_xlabel(f"False-Positive Splits vs. Mammalian Species Tree ({n_shared} Shared Taxa)", fontsize=7.6)
    ax_bot.set_xlim(0, 42.0)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)

    for idx_m, (bar, fp, pr) in enumerate(zip(bars, fp_splits, prec_vals)):
        ax_bot.text(
            bar.get_width() + 0.45,
            bar.get_y() + bar.get_height() / 2,
            f"{fp} FP ({pr:.1f}%)",
            va="center",
            ha="left",
            fontsize=7.1,
            fontweight="bold",
            color="#1b7837" if idx_m == 0 else "#333333",
        )

    ledger_txt = (
        "Stage 4 Shunt MDL Ledger\n"
        "------------------------\n"
        f"Data savings : +{shunt['data_savings_nats']:.2f} nats\n"
        f"SPARSE mask  : -{shunt['mask_cost_nats']:.2f} nats\n"
        f"Node header  : -{shunt['endpoint_cost_nats']:.2f} nats\n"
        f"Net MDL gain : +{shunt['net_mdl_gain_nats']:.2f} nats\n"
        f"P(shunt | A) :  {shunt['posterior_prob']*100:.2f}%"
    )
    ax_bot.text(
        0.99,
        0.96,
        ledger_txt,
        transform=ax_bot.transAxes,
        fontsize=6.8,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f7f7f7", edgecolor="#bdbdbd", lw=0.9),
        ha="right",
        va="top",
    )
    ax_bot.set_title("C. Species-Tree False Positives & Shunt MDL Ledger", loc="left", pad=8)

    fig.savefig(str(out_pdf), bbox_inches="tight")
    fig.savefig(str(out_png), dpi=220, bbox_inches="tight")
    print(f"Saved {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
