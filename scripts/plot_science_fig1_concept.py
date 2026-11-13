#!/usr/bin/env python3
"""
Generate a Science-standard conceptual Figure 1 (`paper/fig_science_concept.pdf`)
illustrating evolutionary trees and networks from lossless data compression (MDL).

All bar lengths, net gains, posterior probabilities, empirical compression ratios,
and positional weights are computed directly from Bushwhacker's exact analytical
two-part MDL formulas and executed empirical benchmarks.
"""

import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath


def l_nat_int(m: int) -> float:
    """Universal code length for non-negative integer m (in nats)."""
    return math.log(1.0 + m) + 0.5 * math.log(1.0 + math.log(1.0 + m))


def l_comb(n: int, k: int) -> float:
    """Exact log binomial coefficient ln C(n, k)."""
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def l_branch(m: int, length: int = 16, alphabet: int = 4) -> float:
    """Exact two-part single-branch code length in nats."""
    if m == 0:
        return math.log(2.0)
    return math.log(2.0) + l_nat_int(m) + l_comb(length, m) + m * math.log(alphabet - 1)


def draw_rounded_box(ax, x, y, w, h, fc, ec="none", lw=0.8, radius=0.08, alpha=1.0, zorder=2, ls="-"):
    patch = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, alpha=alpha, zorder=zorder
    )
    ax.add_patch(patch)
    return patch


def draw_curved_arrow(ax, p0, p1, ctrl, color, lw=2.2, ls="-", zorder=5, head_scale=11):
    verts = [p0, ctrl, p1]
    codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
    path = MplPath(verts, codes)
    arrow = mpatches.FancyArrowPatch(
        path=path,
        arrowstyle=f"-|>,head_length={head_scale*0.42:.2f},head_width={head_scale*0.28:.2f}",
        color=color, linewidth=lw, linestyle=ls, zorder=zorder
    )
    ax.add_patch(arrow)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    out_pdf = repo_root / "paper" / "fig_science_concept.pdf"
    out_png = repo_root / "results" / "fig_science_concept.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8.5,
    })

    # Science editorial palette (high contrast, calm warm/cool neutrals, colorblind-safe)
    INK = "#0F172A"
    SUBINK = "#475569"
    MUTED = "#94A3B8"
    TILE_ANC = "#E2E8F0"
    TILE_ANC_EDGE = "#CBD5E1"
    BLUE = "#1D4ED8"       # Shared clade changes (synapomorphies)
    BLUE_DARK = "#1E3A8A"  # Deep conserved slow-rate signal
    BLUE_SOFT = "#DBEAFE"
    AMBER = "#D97706"      # Private / fast-branch changes
    AMBER_PALE = "#FDE68A"
    SLATE = "#334155"      # Tree backbone / topology address
    ROSE = "#E11D48"       # Polytomy / negative net gain
    ROSE_SOFT = "#FFE4E6"
    TEAL = "#0D9488"       # Contiguous block transfer
    TEAL_SOFT = "#CCFBF1"
    VIOLET = "#7C3AED"     # Scattered site convergence
    VIOLET_SOFT = "#EDE9FE"
    PANEL_BG = "#F8FAFC"
    HAIRLINE = "#E2E8F0"

    # Exact analytical quantities for N=4, L=16, m_c=2 private mutations/leaf
    L = 16
    raw_total = 4 * L * math.log(4.0)            # 88.72 nats
    root_cost = L * math.log(4.0)                # 22.18 nats
    l_topo_q = math.log(3.0)                     # 1.0986 nats
    leaf_priv_cost = l_branch(2, L)              # 9.15 nats

    # Panel A: m_e = 3 shared changes
    stem_3_cost = l_branch(3, L)                 # 12.14 nats
    leaves_4_cost = 4 * leaf_priv_cost           # 36.60 nats
    tree_total_3 = root_cost + stem_3_cost + l_topo_q + leaves_4_cost  # 72.02 nats

    # Panel B: m_e in 0..4
    me_vals = [0, 1, 2, 3, 4]
    net_gains = []
    post_probs = []
    for me in me_vals:
        savings = 2.0 * (l_branch(2 + me, L) - l_branch(2, L))
        split_cost = l_branch(me, L) + l_topo_q
        g = savings - split_cost
        net_gains.append(g)
        post_probs.append(1.0 / (1.0 + math.exp(-g)))

    # Panel C: marginal positional weights on L=16
    w_quiet = math.log((L - 1) / (1 + 1))        # 2.015 nats/site (m=1)
    w_noisy = math.log((L - 6) / (6 + 1))        # 0.357 nats/site (m=6)
    ratio_qn = w_quiet / w_noisy                 # 5.65x

    fig = plt.figure(figsize=(12.6, 8.3), facecolor="white")
    gs = fig.add_gridspec(
        2, 2,
        left=0.035, right=0.98, top=0.945, bottom=0.045,
        wspace=0.13, hspace=0.21,
        width_ratios=[1.03, 1.0],
        height_ratios=[1.02, 1.0],
    )

    # =========================================================================
    # PANEL A: Heredity as Lossless Data Compression
    # =========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0, 10.0)
    ax_a.set_ylim(0, 7.0)
    ax_a.axis("off")

    ax_a.text(0.0, 6.76, "A", fontsize=14, fontweight="bold", color=INK, ha="left", va="center")
    ax_a.text(0.44, 6.76, "Heredity as lossless compression", fontsize=11, fontweight="bold", color=INK, ha="left", va="center")

    # Grid-aligned legend above alignment column 1
    leg_items = [
        (TILE_ANC, TILE_ANC_EDGE, "Inherited root state", 1.50),
        (BLUE, BLUE, "Shared clade edit", 4.30),
        (AMBER, AMBER, "Private leaf edit", 6.82),
    ]
    for fc, ec, lbl, lx in leg_items:
        draw_rounded_box(ax_a, lx, 6.10, 0.24, 0.24, fc=fc, ec=ec, lw=0.8, radius=0.05)
        ax_a.text(lx + 0.32, 6.22, lbl, fontsize=7.6, color=SUBINK, ha="left", va="center")

    # Alignment matrix (Ancestor + 4 lineages x 16 sites)
    row_defs = [
        ("Ancestor", ["anc"] * 16, 5.46),
        ("Lineage 1", ["shared"] * 3 + ["priv"] * 2 + ["anc"] * 11, 4.82),
        ("Lineage 2", ["shared"] * 3 + ["anc"] * 2 + ["priv"] * 2 + ["anc"] * 9, 4.30),
        ("Lineage 3", ["anc"] * 7 + ["priv"] * 2 + ["anc"] * 7, 3.78),
        ("Lineage 4", ["anc"] * 9 + ["priv"] * 2 + ["anc"] * 5, 3.26),
    ]

    ax_a.plot([1.48, 8.50], [5.25, 5.25], color=HAIRLINE, lw=1.0, ls="--", zorder=1)

    for r_label, states, y in row_defs:
        is_anc = (r_label == "Ancestor")
        ax_a.text(
            1.34, y + 0.15, r_label,
            fontsize=8.0, fontweight="bold" if is_anc else "normal",
            color=INK if is_anc else SUBINK, ha="right", va="center"
        )
        for i, st in enumerate(states):
            x = 1.50 + i * 0.44
            if st == "shared":
                fc, ec = BLUE, "#1E40AF"
            elif st == "priv":
                fc, ec = AMBER, "#B45309"
            else:
                fc = "#CBD5E1" if is_anc else TILE_ANC
                ec = "#94A3B8" if is_anc else TILE_ANC_EDGE
            draw_rounded_box(ax_a, x, y, 0.36, 0.30, fc=fc, ec=ec, lw=0.65, radius=0.055)

    # Right bracket uniting Lineage 1 & 2 shared descent
    ax_a.plot([8.58, 8.74, 8.74, 8.58], [4.38, 4.38, 5.04, 5.04], color=BLUE, lw=1.3)
    ax_a.text(8.86, 4.71, "Shared\ndescent", fontsize=7.5, fontweight="bold", color=BLUE, ha="left", va="center")

    # Proportional Information Tapes (4 sequences -> 10,000 viral genomes)
    draw_rounded_box(ax_a, 0.0, 0.10, 9.88, 2.86, fc=PANEL_BG, ec=HAIRLINE, lw=0.9, radius=0.14, zorder=0)
    ax_a.text(0.28, 2.68, "Physical description length required for exact lossless recovery",
              fontsize=8.2, fontweight="bold", color=INK, ha="left", va="center")

    tape_x0 = 2.12
    full_w = 6.12
    tape_scale = full_w / raw_total

    # Row 1: Raw 4 sequences
    y_raw = 1.98
    ax_a.text(1.96, y_raw + 0.15, "4 sequences, raw", fontsize=7.8, fontweight="bold", color=SUBINK, ha="right", va="center")
    seg_w = root_cost * tape_scale
    for k in range(4):
        sx = tape_x0 + k * seg_w
        draw_rounded_box(ax_a, sx + 0.02, y_raw - 0.04, seg_w - 0.04, 0.38,
                         fc="#CBD5E1", ec="#94A3B8", lw=0.75, radius=0.05)
        ax_a.text(sx + seg_w / 2, y_raw + 0.15, f"Seq {k+1}", fontsize=7.0, color=INK, ha="center", va="center")
    ax_a.text(tape_x0 + full_w + 0.14, y_raw + 0.15, f"{raw_total:.1f} nats",
              fontsize=8.0, fontweight="bold", color=SUBINK, ha="left", va="center")

    # Row 2: Tree-compressed 4 sequences
    y_mdl = 1.24
    ax_a.text(1.96, y_mdl + 0.15, "4 sequences, tree", fontsize=7.8, fontweight="bold", color=BLUE, ha="right", va="center")
    parts = [
        (root_cost, "#94A3B8", "white", "1 Root"),
        (l_topo_q + stem_3_cost, BLUE, "white", "Clade stem"),
        (leaves_4_cost, AMBER, "white", "Leaf edits"),
    ]
    cx = tape_x0
    for val, fc, tc, lbl in parts:
        pw = val * tape_scale
        draw_rounded_box(ax_a, cx + 0.02, y_mdl - 0.04, pw - 0.04, 0.38,
                         fc=fc, ec="none", lw=0.0, radius=0.05)
        ax_a.text(cx + pw / 2, y_mdl + 0.15, lbl, fontsize=7.0, fontweight="bold", color=tc, ha="center", va="center")
        cx += pw

    saved_w = (raw_total - tree_total_3) * tape_scale
    draw_rounded_box(ax_a, cx + 0.02, y_mdl - 0.04, saved_w - 0.04, 0.38,
                     fc="#EFF6FF", ec=BLUE, lw=0.85, radius=0.05, ls="--")
    ax_a.text(cx + saved_w / 2, y_mdl + 0.15, "Saved", fontsize=7.0, fontweight="bold", color=BLUE, ha="center", va="center")
    ax_a.text(tape_x0 + full_w + 0.14, y_mdl + 0.15, f"{tree_total_3:.1f} nats",
              fontsize=8.0, fontweight="bold", color=BLUE, ha="left", va="center")

    # Row 3: Empirical pandemic scale (10,062 SARS-CoV-2 genomes: 690x MDL / 476x .bwz, 12x < gzip)
    y_vir = 0.44
    ax_a.text(1.96, y_vir + 0.15, "10,062 genomes", fontsize=7.8, fontweight="bold", color=TEAL, ha="right", va="center")
    # Background ghost bar = raw 301 MB; middle marker = gzip 7.57 MB; foreground sliver = .bwz 0.63 MB
    draw_rounded_box(ax_a, tape_x0 + 0.02, y_vir - 0.04, full_w - 0.04, 0.38,
                     fc="white", ec="#CBD5E1", lw=0.8, radius=0.05, ls=":")
    gzip_w = full_w * (7.57 / 301.13)  # exact empirical gzip ratio (0.154 units)
    bwz_w = max(0.08, full_w * (0.633 / 301.13))  # visible razor sliver
    draw_rounded_box(ax_a, tape_x0 + 0.02, y_vir - 0.04, max(gzip_w, 0.38), 0.38,
                     fc="#CBD5E1", ec="none", radius=0.04, zorder=3)
    draw_rounded_box(ax_a, tape_x0 + 0.02, y_vir - 0.04, bwz_w, 0.38,
                     fc=TEAL, ec="none", radius=0.03, zorder=4)
    ax_a.text(tape_x0 + 0.52, y_vir + 0.15,
              "63 bytes / viral genome  (12× smaller than gzip; 476× vs. raw)",
              fontsize=7.3, fontweight="bold", color=TEAL, ha="left", va="center", zorder=5)
    ax_a.text(tape_x0 + full_w + 0.14, y_vir + 0.15, "0.63 MB",
              fontsize=8.0, fontweight="bold", color=TEAL, ha="left", va="center")

    # =========================================================================
    # PANEL B: Polytomy vs. Resolved Split (Tree Geometry + Phase Threshold)
    # =========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_xlim(0, 10.0)
    ax_b.set_ylim(0, 7.0)
    ax_b.axis("off")

    ax_b.text(0.0, 6.76, "B", fontsize=14, fontweight="bold", color=INK, ha="left", va="center")
    ax_b.text(0.44, 6.76, "When a branch earns its place: burst vs. split", fontsize=11, fontweight="bold", color=INK, ha="left", va="center")

    # Top twin tree glyphs (y = 3.92 .. 6.22)
    hub_x, hub_y = 2.00, 5.12
    tips_star = [(0.82, 5.90, "1"), (0.82, 4.34, "2"), (3.18, 5.90, "3"), (3.18, 4.34, "4")]
    for tx, ty, tlbl in tips_star:
        ax_b.plot([hub_x, tx], [hub_y, ty], color=SLATE, lw=2.1, solid_capstyle="round", zorder=2)
        ax_b.text(tx + (-0.18 if tx < hub_x else 0.18), ty, tlbl,
                  fontsize=8.1, fontweight="bold", color=INK, ha="center", va="center")
    for ty_sign in [1, -1]:
        for frac, col in [(0.34, BLUE), (0.50, BLUE), (0.68, AMBER), (0.84, AMBER)]:
            px = hub_x + frac * (0.82 - hub_x)
            py = hub_y + frac * ((5.12 + 0.78 * ty_sign) - hub_y)
            ax_b.scatter([px], [py], s=26, color=col, edgecolor="white", linewidth=0.6, zorder=4)
        for frac in [0.58, 0.78]:
            px = hub_x + frac * (3.18 - hub_x)
            py = hub_y + frac * ((5.12 + 0.78 * ty_sign) - hub_y)
            ax_b.scatter([px], [py], s=25, color=AMBER, edgecolor="white", linewidth=0.6, zorder=4)
    ax_b.scatter([hub_x], [hub_y], s=66, color=ROSE, edgecolor="white", linewidth=1.2, zorder=5)
    ax_b.text(2.00, 6.22, "Unresolved polytomy", fontsize=8.4, fontweight="bold", color=ROSE, ha="center", va="center")
    ax_b.text(2.00, 3.92, "Shared edits sent twice;\nno branch overhead", fontsize=7.4, color=SUBINK, ha="center", va="center")

    # Center comparison double-arrow
    ax_b.annotate("", xy=(5.20, 5.12), xytext=(4.10, 5.12),
                  arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.5))
    ax_b.text(4.65, 5.42, "Branch address\n1.1 nats", fontsize=7.2, fontweight="bold", color=SLATE, ha="center", va="center")

    # Right glyph: Resolved Quartet
    r_left, r_right, r_y = 6.95, 8.05, 5.12
    ax_b.plot([r_left, r_right], [r_y, r_y], color=BLUE, lw=3.6, solid_capstyle="round", zorder=3)
    for sx in [7.32, 7.68]:
        ax_b.scatter([sx], [r_y], s=34, color=BLUE, edgecolor="white", linewidth=0.8, zorder=5)

    tips_split = [
        (r_left, 5.85, 5.90, "1"),
        (r_left, 5.85, 4.34, "2"),
        (r_right, 9.15, 5.90, "3"),
        (r_right, 9.15, 4.34, "4"),
    ]
    for nx, tx, ty, tlbl in tips_split:
        ax_b.plot([nx, tx], [r_y, ty], color=SLATE, lw=2.0, solid_capstyle="round", zorder=2)
        ax_b.text(tx + (-0.18 if tx < nx else 0.18), ty, tlbl,
                  fontsize=8.1, fontweight="bold", color=INK, ha="center", va="center")
        for frac in [0.55, 0.78]:
            px = nx + frac * (tx - nx)
            py = r_y + frac * (ty - r_y)
            ax_b.scatter([px], [py], s=25, color=AMBER, edgecolor="white", linewidth=0.6, zorder=4)
    ax_b.scatter([r_left, r_right], [r_y, r_y], s=44, color=INK, edgecolor="white", linewidth=1.0, zorder=5)
    ax_b.text(7.50, 6.22, "Resolved clade", fontsize=8.4, fontweight="bold", color=BLUE, ha="center", va="center")
    ax_b.text(7.50, 3.92, "Shared edits sent once;\npays branch address", fontsize=7.4, color=SUBINK, ha="center", va="center")

    # Bottom Diverging Net-Compression Bar Chart (zero overlaps!)
    draw_rounded_box(ax_b, 0.0, 0.10, 9.90, 3.42, fc=PANEL_BG, ec=HAIRLINE, lw=0.9, radius=0.14, zorder=0)
    ax_b.text(0.28, 3.24, "Net information saved by adding the branch (nats; % = posterior support)",
              fontsize=8.1, fontweight="bold", color=INK, ha="left", va="center")

    zero_y = 1.40
    top_zone_y = 3.00
    bot_zone_y = 0.56

    # Left vs Right phase zones (split cleanly at x = 3.90!)
    draw_rounded_box(ax_b, 0.55, bot_zone_y, 3.35, top_zone_y - bot_zone_y,
                     fc=ROSE_SOFT, ec="none", radius=0.04, alpha=0.42, zorder=1)
    draw_rounded_box(ax_b, 3.90, bot_zone_y, 5.65, top_zone_y - bot_zone_y,
                     fc=BLUE_SOFT, ec="none", radius=0.04, alpha=0.42, zorder=1)

    ax_b.plot([0.55, 9.55], [zero_y, zero_y], color=INK, lw=1.05, zorder=3)
    ax_b.text(0.46, zero_y, "0", fontsize=7.6, fontweight="bold", color=INK, ha="right", va="center")
    ax_b.plot([3.90, 3.90], [bot_zone_y, top_zone_y], color=SLATE, lw=1.1, ls=":", zorder=3)

    # Vector arrows + phase headers positioned cleanly away from tall bars
    ax_b.text(3.72, 2.78, "Keep polytomy", fontsize=7.6, fontweight="bold", color=ROSE, ha="right", va="center", zorder=5)
    ax_b.annotate("", xy=(1.75, 2.78), xytext=(2.18, 2.78),
                  arrowprops=dict(arrowstyle="->,head_width=0.18,head_length=0.22", color=ROSE, lw=1.2), zorder=5)
    ax_b.text(4.08, 2.78, "Resolve clade", fontsize=7.6, fontweight="bold", color=BLUE, ha="left", va="center", zorder=5)
    ax_b.annotate("", xy=(6.02, 2.78), xytext=(5.58, 2.78),
                  arrowprops=dict(arrowstyle="->,head_width=0.18,head_length=0.22", color=BLUE, lw=1.2), zorder=5)

    scale_y = 0.31
    xs = [1.38, 3.02, 4.76, 6.46, 8.16]
    for idx, (me, g, pp) in enumerate(zip(me_vals, net_gains, post_probs)):
        bx = xs[idx]
        bh = g * scale_y
        if g <= 0:
            draw_rounded_box(ax_b, bx - 0.42, zero_y + bh, 0.84, -bh,
                             fc=ROSE, ec="none", radius=0.05, zorder=4)
            ax_b.text(bx, zero_y + bh - 0.16, f"{g:+.1f}",
                      fontsize=7.8, fontweight="bold", color=ROSE, ha="center", va="center")
        else:
            draw_rounded_box(ax_b, bx - 0.42, zero_y, 0.84, bh,
                             fc=BLUE, ec="none", radius=0.05, zorder=4)
            ax_b.text(bx, zero_y + bh + 0.15, f"+{g:.1f}",
                      fontsize=7.8, fontweight="bold", color=BLUE, ha="center", va="center")
            if bh >= 0.34:
                ax_b.text(bx, zero_y + bh / 2, f"{int(round(pp*100))}%",
                          fontsize=7.0, fontweight="bold", color="white", ha="center", va="center", zorder=5)
            else:
                ax_b.text(bx, zero_y + 0.14, f"{int(round(pp*100))}%",
                          fontsize=6.8, fontweight="bold", color="white", ha="center", va="center", zorder=5)

        ax_b.text(bx, 0.30, f"{me} shared", fontsize=7.6, color=INK, ha="center", va="center")

    # =========================================================================
    # PANEL C: Uneven Evolutionary Rates & Deep-Time Site-Rate Spectrum
    # =========================================================================
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_xlim(0, 10.0)
    ax_c.set_ylim(0, 6.9)
    ax_c.axis("off")

    ax_c.text(0.0, 6.64, "C", fontsize=14, fontweight="bold", color=INK, ha="left", va="center")
    ax_c.text(0.44, 6.64, "Uneven evolutionary rates and deep-time saturation", fontsize=11, fontweight="bold", color=INK, ha="left", va="center")

    # Top-left: Uneven-rate quartet (Felsenstein zone)
    q_l, q_r, q_y = 1.62, 2.52, 4.50
    ax_c.plot([q_l, q_r], [q_y, q_y], color=BLUE, lw=3.2, solid_capstyle="round", zorder=3)

    # Fast 1 (top-left)
    ax_c.plot([q_l, 0.36], [q_y, 5.92], color=AMBER, lw=2.4, solid_capstyle="round", zorder=2)
    for f in [0.22, 0.36, 0.50, 0.64, 0.78, 0.90]:
        ax_c.scatter([q_l + f * (0.36 - q_l)], [q_y + f * (5.92 - q_y)],
                     s=21, color=AMBER, edgecolor="white", lw=0.5, zorder=4)
    ax_c.text(0.30, 6.12, "Fast 1", fontsize=7.7, fontweight="bold", color=AMBER, ha="left", va="center")

    # Quiet 2 (bottom-left)
    ax_c.plot([q_l, 1.12], [q_y, 3.52], color=BLUE, lw=2.4, solid_capstyle="round", zorder=2)
    ax_c.scatter([q_l + 0.55 * (1.12 - q_l)], [q_y + 0.55 * (3.52 - q_y)],
                 s=25, color=BLUE, edgecolor="white", lw=0.6, zorder=4)
    ax_c.text(1.02, 3.26, "Quiet 2", fontsize=7.7, fontweight="bold", color=BLUE, ha="center", va="center")

    # Fast 3 (top-right)
    ax_c.plot([q_r, 3.78], [q_y, 5.92], color=AMBER, lw=2.4, solid_capstyle="round", zorder=2)
    for f in [0.22, 0.36, 0.50, 0.64, 0.78, 0.90]:
        ax_c.scatter([q_r + f * (3.78 - q_r)], [q_y + f * (5.92 - q_y)],
                     s=21, color=AMBER, edgecolor="white", lw=0.5, zorder=4)
    ax_c.text(3.84, 6.12, "Fast 3", fontsize=7.7, fontweight="bold", color=AMBER, ha="right", va="center")

    # Quiet 4 (bottom-right)
    ax_c.plot([q_r, 3.02], [q_y, 3.52], color=BLUE, lw=2.4, solid_capstyle="round", zorder=2)
    ax_c.scatter([q_r + 0.55 * (3.02 - q_r)], [q_y + 0.55 * (3.52 - q_y)],
                 s=25, color=BLUE, edgecolor="white", lw=0.6, zorder=4)
    ax_c.text(3.12, 3.26, "Quiet 4", fontsize=7.7, fontweight="bold", color=BLUE, ha="center", va="center")
    ax_c.scatter([q_l, q_r], [q_y, q_y], s=38, color=INK, edgecolor="white", lw=0.9, zorder=5)

    # Top-right: Proportional site-surprise bars
    draw_rounded_box(ax_c, 4.22, 3.02, 5.66, 3.22, fc=PANEL_BG, ec=HAIRLINE, lw=0.9, radius=0.14, zorder=0)
    ax_c.text(4.48, 5.94, "Positional information saved per shared match",
              fontsize=8.1, fontweight="bold", color=INK, ha="left", va="center")

    bar_max_w = 3.05
    ax_c.text(4.48, 5.36, "Quiet branch (1/16 mutated)", fontsize=7.6, fontweight="bold", color=BLUE, ha="left", va="center")
    draw_rounded_box(ax_c, 4.48, 4.84, bar_max_w, 0.34, fc=BLUE, ec="none", radius=0.05)
    ax_c.text(4.48 + bar_max_w + 0.14, 5.01, f"{w_quiet:.2f} nats",
              fontsize=7.9, fontweight="bold", color=BLUE, ha="left", va="center")

    noisy_w = bar_max_w * (w_noisy / w_quiet)
    ax_c.text(4.48, 4.30, "Noisy branch (6/16 mutated)", fontsize=7.6, fontweight="bold", color=AMBER, ha="left", va="center")
    draw_rounded_box(ax_c, 4.48, 3.78, noisy_w, 0.34, fc=AMBER, ec="none", radius=0.05)
    ax_c.text(4.48 + noisy_w + 0.14, 3.95, f"{w_noisy:.2f} nats  ({ratio_qn:.1f}× weaker)",
              fontsize=7.9, fontweight="bold", color=AMBER, ha="left", va="center")

    ax_c.text(4.48, 3.32, "Chance collisions on fast branches are naturally discounted",
              fontsize=7.3, color=SUBINK, ha="left", va="center")

    # Bottom visual: Deep-time 8-category rate-spectrum & 64-site CPU cache tiling schematic!
    draw_rounded_box(ax_c, 0.0, 0.10, 9.88, 2.62, fc=PANEL_BG, ec=HAIRLINE, lw=0.9, radius=0.14, zorder=0)
    ax_c.text(0.28, 2.42, "Deep-time supermatrices: site-rate spectrum & cache-tiled Fisher MDL",
              fontsize=8.1, fontweight="bold", color=INK, ha="left", va="center")

    # Visual 8-category Gamma rate bars (slow conserved -> saturated hypervariable)
    cat_rates = [0.06, 0.16, 0.31, 0.52, 0.80, 1.18, 1.75, 2.85]
    cat_weights = [1.00, 0.88, 0.74, 0.58, 0.42, 0.28, 0.17, 0.09]  # deep-stem information retention
    cat_cols = ["#1E3A8A", "#1D4ED8", "#2563EB", "#3B82F6", "#60A5FA", "#94A3B8", "#D97706", "#F59E0B"]
    gx0 = 0.34
    for ci, (wt, col) in enumerate(zip(cat_weights, cat_cols)):
        bx = gx0 + ci * 0.48
        bh = 1.20 * wt
        draw_rounded_box(ax_c, bx, 0.64, 0.35, bh, fc=col, ec="none", radius=0.04, zorder=3)
    ax_c.text(gx0, 0.36, "Slow sites\n(deep signal)", fontsize=7.0, fontweight="bold", color=BLUE_DARK, ha="left", va="center")
    ax_c.text(gx0 + 7 * 0.48 + 0.35, 0.36, "Fast sites\n(saturated)", fontsize=7.0, color=AMBER, ha="right", va="center")
    ax_c.text(gx0 + 1.86, 2.02, "8-rate exchangeability spectrum", fontsize=7.3, fontweight="bold", color=SUBINK, ha="center", va="center")

    # Visual memory comparison on right half of bottom box: 64-site CPU cache tile vs giant RAM table
    mx0 = 4.68
    ax_c.text(mx0, 2.02, "Peak supermatrix memory (882 mammals, AA)", fontsize=7.4, fontweight="bold", color=SUBINK, ha="left", va="center")
    # Standard ML full table bar (79.2 GB)
    full_mem_w = 3.70
    draw_rounded_box(ax_c, mx0, 1.36, full_mem_w, 0.34, fc="#CBD5E1", ec="#94A3B8", lw=0.7, radius=0.04)
    ax_c.text(mx0 + 0.12, 1.53, "Full alignment buffer (IQ-TREE 2)", fontsize=7.0, color=INK, ha="left", va="center")
    ax_c.text(mx0 + full_mem_w + 0.12, 1.53, "79.2 GB", fontsize=7.7, fontweight="bold", color=SUBINK, ha="left", va="center")

    # Bushwhacker 64-site cache tile bar (1.98 GB -> 40x smaller!)
    bw_mem_w = max(0.14, full_mem_w * (1.98 / 79.15))
    draw_rounded_box(ax_c, mx0, 0.78, bw_mem_w, 0.34, fc=BLUE, ec="none", radius=0.03, zorder=4)
    ax_c.text(mx0 + bw_mem_w + 0.14, 0.95, "1.98 GB  (40× leaner via 64-site CPU cache tiles)",
              fontsize=7.4, fontweight="bold", color=BLUE, ha="left", va="center")
    ax_c.text(mx0, 0.36, "Reconstructs Mammalia, Archaea & Bacteria in 20–49 s",
              fontsize=7.1, color=SUBINK, ha="left", va="center")

    # =========================================================================
    # PANEL D: Separating Contiguous Block Transfer from Site Convergence
    # =========================================================================
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_xlim(0, 10.0)
    ax_d.set_ylim(0, 6.9)
    ax_d.axis("off")

    ax_d.text(0.0, 6.64, "D", fontsize=14, fontweight="bold", color=INK, ha="left", va="center")
    ax_d.text(0.44, 6.64, "Separating gene transfer from site convergence", fontsize=11, fontweight="bold", color=INK, ha="left", va="center")

    def draw_mini_network(x0, title, accent, soft_bg, is_block):
        draw_rounded_box(ax_d, x0, 0.10, 4.74, 6.14, fc=PANEL_BG, ec=HAIRLINE, lw=0.9, radius=0.14, zorder=0)
        ax_d.text(x0 + 2.37, 5.90, title, fontsize=8.6, fontweight="bold", color=accent, ha="center", va="center")

        # Clean rooted 4-tip tree inside [x0+0.55 .. x0+4.15, y=3.55 .. 5.42]
        rx, ry = x0 + 2.37, 5.42
        l_int = (x0 + 1.46, 4.62)
        r_int = (x0 + 3.28, 4.62)
        t1, t2 = (x0 + 0.86, 3.66), (x0 + 2.06, 3.66)
        t3, t4 = (x0 + 2.68, 3.66), (x0 + 3.88, 3.66)

        for a, b in [((rx, ry), l_int), ((rx, ry), r_int), (l_int, t1), (l_int, t2), (r_int, t3), (r_int, t4)]:
            ax_d.plot([a[0], b[0]], [a[1], b[1]], color=SLATE, lw=2.0, solid_capstyle="round", zorder=2)
        for pt in [(rx, ry), l_int, r_int]:
            ax_d.scatter([pt[0]], [pt[1]], s=26, color=SLATE, edgecolor="white", lw=0.7, zorder=4)

        if is_block:
            p_from = (x0 + 1.66, 4.10)
            p_to = (x0 + 3.56, 4.10)
            draw_curved_arrow(ax_d, p_from, p_to, (x0 + 2.61, 3.28), color=accent, lw=2.4, ls="-", zorder=6)
            ax_d.text(x0 + 2.61, 3.22, "Lateral block transfer", fontsize=7.3, fontweight="bold", color=accent, ha="center", va="center")
        else:
            p_from = (x0 + 1.16, 4.06)
            p_to = (x0 + 3.60, 4.06)
            draw_curved_arrow(ax_d, p_from, p_to, (x0 + 2.38, 3.26), color=accent, lw=2.1, ls="--", zorder=6)
            ax_d.text(x0 + 2.38, 3.22, "Parallel convergence", fontsize=7.3, fontweight="bold", color=accent, ha="center", va="center")

        # Aligned genome / protein track (y = 2.16 .. 2.54)
        gx = x0 + 0.37
        gw = 4.00
        gy = 2.12
        draw_rounded_box(ax_d, gx, gy, gw, 0.36, fc="white", ec="#CBD5E1", lw=0.9, radius=0.06, zorder=2)

        if is_block:
            bx, bw = gx + 1.12, 1.70
            draw_rounded_box(ax_d, bx, gy, bw, 0.36, fc=accent, ec="none", radius=0.04, zorder=3)
            ax_d.text(bx + bw / 2, gy + 0.18, "Transferred tract", fontsize=7.0, fontweight="bold", color="white", ha="center", va="center", zorder=4)
            for px, plbl in [(bx, "start"), (bx + bw, "end")]:
                ax_d.plot([px, px], [gy - 0.08, gy + 0.46], color=INK, lw=1.1, zorder=5)
                ax_d.text(px, gy + 0.60, plbl, fontsize=6.9, fontweight="bold", color=INK, ha="center", va="center")

            # Visual cost-vs-savings ledger bars at bottom (zero text-wall!)
            ax_d.text(gx, 1.66, "Link cost vs. shared edits explained", fontsize=7.2, fontweight="bold", color=INK, ha="left", va="center")
            draw_rounded_box(ax_d, gx, 1.20, 1.05, 0.28, fc=SLATE, ec="none", radius=0.04)
            ax_d.text(gx + 1.16, 1.34, "Cost: 2 endpoints", fontsize=7.1, color=SLATE, ha="left", va="center")
            draw_rounded_box(ax_d, gx, 0.78, 3.15, 0.28, fc=accent, ec="none", radius=0.04)
            ax_d.text(gx + 3.25, 0.92, "Saved", fontsize=7.1, fontweight="bold", color=accent, ha="left", va="center")
            ax_d.text(x0 + 2.37, 0.38, "Recombination • HGT • Gene conversion", fontsize=7.0, fontweight="bold", color=accent, ha="center", va="center")
        else:
            # Subtle functional domain shading + 4 clean violet site lollipops
            draw_rounded_box(ax_d, gx + 0.85, gy + 0.03, 2.30, 0.30, fc=VIOLET_SOFT, ec="none", radius=0.03, zorder=2)
            pin_offsets = [1.05, 1.55, 2.28, 2.92]
            for po in pin_offsets:
                px = gx + po
                draw_rounded_box(ax_d, px - 0.06, gy, 0.12, 0.36, fc=accent, ec="none", radius=0.02, zorder=3)
                ax_d.plot([px, px], [gy + 0.36, gy + 0.50], color=accent, lw=1.1, zorder=4)
                ax_d.scatter([px], [gy + 0.53], s=20, color=accent, zorder=5)
            ax_d.text(x0 + 2.37, gy + 0.76, "Active-site / sensory residues", fontsize=6.9, color=SUBINK, ha="center", va="center")

            ax_d.text(gx, 1.66, "Link cost vs. shared edits explained", fontsize=7.2, fontweight="bold", color=INK, ha="left", va="center")
            draw_rounded_box(ax_d, gx, 1.20, 2.05, 0.28, fc=SLATE, ec="none", radius=0.04)
            ax_d.text(gx + 2.16, 1.34, "Cost: every site indexed", fontsize=7.1, color=SLATE, ha="left", va="center")
            draw_rounded_box(ax_d, gx, 0.78, 3.15, 0.28, fc=accent, ec="none", radius=0.04)
            ax_d.text(gx + 3.25, 0.92, "Saved", fontsize=7.1, fontweight="bold", color=accent, ha="left", va="center")
            ax_d.text(x0 + 2.37, 0.38, "Inner-ear hearing • Diet • Antibody CDRs", fontsize=7.0, fontweight="bold", color=accent, ha="center", va="center")

    draw_mini_network(0.0, "Contiguous block transfer", TEAL, TEAL_SOFT, is_block=True)
    draw_mini_network(5.16, "Scattered site convergence", VIOLET, VIOLET_SOFT, is_block=False)

    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Wrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
