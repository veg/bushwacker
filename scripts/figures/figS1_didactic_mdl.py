#!/usr/bin/env python3
"""
Generate publication-grade vector PDF Figure (`paper/fig_didactic_mdl.pdf`)
for the 4-taxon, 16-site didactic MDL ledger (Table 1 and Section 2.3),
using exact two-part branch differences, exact quartet topology cost
L_topo(2,2) = ln 3 = 1.10 nats (matching `src/lib.rs:split_topology_cost_nats`),
Okabe-Ito colorblind-safe palette, and unified sans-serif typography.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out_pdf = repo_root / "paper" / "fig_didactic_mdl.pdf"

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8.2,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.4,
    })

    # Okabe-Ito colorblind-safe palette
    c_blue = "#0072B2"
    c_sky = "#56B4E9"
    c_orange = "#E69F00"
    c_vermillion = "#D55E00"
    c_purple = "#CC79A7"
    c_gray = "#666666"
    c_light_bg = "#F7F9FB"

    fig = plt.figure(figsize=(13.6, 7.6))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.08, 1.0],
        height_ratios=[1.05, 1.0],
        wspace=0.22,
        hspace=0.34,
    )

    # =========================================================================
    # PANEL A: Alignment Matrix (N = 4, L = 16) & Single-Branch Code Anatomy
    # =========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0, 22.8)
    ax_a.set_ylim(-3.1, 6.4)
    ax_a.axis("off")

    ax_a.text(
        0.0, 6.05,
        "A   Didactic Alignment (N = 4 Taxa, L = 16 Sites) and Branch Code Anatomy",
        fontsize=9.6, color="#1a1a1a", ha="left", va="center",
    )
    ax_a.text(
        0.0, 5.42,
        "Raw Uncompressed Cost: 4 × 16 × ln 4 = 88.72 nats  |  Ancestral Root X_r Cost: 16 × ln 4 = 22.18 nats",
        fontsize=7.7, color="#333333", ha="left", va="center",
    )

    # Bracket headers for site partitions
    ax_a.plot([2.25, 4.85], [4.62, 4.62], color=c_blue, lw=1.6)
    ax_a.text(3.45, 4.82, "Shared (1–3)", fontsize=6.4, color=c_blue, ha="center", va="bottom")

    ax_a.plot([5.25, 12.85], [4.62, 4.62], color=c_orange, lw=1.6)
    ax_a.text(9.05, 4.82, "Private Substitutions (4–11; m_c=2)", fontsize=6.4, color="#995c00", ha="center", va="bottom")

    ax_a.plot([13.15, 17.75], [4.62, 4.62], color=c_gray, lw=1.6)
    ax_a.text(15.55, 4.82, "Invariant (12–16)", fontsize=6.4, color="#444444", ha="center", va="bottom")

    for s in range(1, 17):
        x = 1.65 + s * 0.98
        ax_a.text(x, 4.25, str(s), fontsize=7.0, color="#333333", ha="center", va="center")

    rows = [
        ("Root X_r", ["A"] * 16, "22.18 nats", 3.55),
        ("Taxon T1", ["G", "G", "G", "C", "C"] + ["."] * 11, "m_e + 2 muts", 2.75),
        ("Taxon T2", ["G", "G", "G", ".", ".", "T", "T"] + ["."] * 9, "m_e + 2 muts", 1.95),
        ("Taxon T3", ["."] * 7 + ["C", "C"] + ["."] * 7, "2 priv. muts", 1.15),
        ("Taxon T4", ["."] * 9 + ["T", "T"] + ["."] * 5, "2 priv. muts", 0.35),
    ]

    for label, states, right_note, y in rows:
        ax_a.text(1.95, y, label, fontsize=7.8, color="#222222", ha="right", va="center")
        for idx, st in enumerate(states):
            x = 1.65 + (idx + 1) * 0.98
            if label == "Root X_r":
                bg, fg, ec = "#eef2f6", "#222222", "#cbd5e1"
            elif st == "G":
                bg, fg, ec = "#d0e8f8", "#004b76", "#7bbce8"
            elif st in ("C", "T"):
                bg, fg, ec = "#fde6c4", "#8a4b00", "#f3b867"
            else:
                bg, fg, ec = "#fafafa", "#888888", "#e2e8f0"
            rect = mpatches.FancyBboxPatch(
                (x - 0.41, y - 0.31), 0.82, 0.62,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=bg, edgecolor=ec, lw=0.75,
            )
            ax_a.add_patch(rect)
            ax_a.text(x, y, st, fontsize=7.3, color=fg, ha="center", va="center")
        ax_a.text(17.95, y, right_note, fontsize=7.4, color="#333333", ha="left", va="center")

    box_a = mpatches.FancyBboxPatch(
        (0.05, -2.95), 22.55, 2.72,
        boxstyle="round,pad=0.08,rounding_size=0.18",
        facecolor=c_light_bg, edgecolor="#94a3b8", lw=0.9,
    )
    ax_a.add_patch(box_a)
    ax_a.text(
        0.4, -0.62,
        "Single-Branch Code Length (Eq. 1):  L_branch(m, 16) = L_N(m) + ln C(16, m) + m ln 3   (L_N(0) = ln 2)",
        fontsize=7.0, color="#004b76", ha="left", va="center",
    )
    ax_a.text(
        0.4, -1.25,
        "• m = 1 mut:  1.69 + ln(16) [2.77] + 1.10 =  5.56 nats   (Unamortized new-branch startup)",
        fontsize=6.65, color="#222222", ha="left", va="center",
    )
    ax_a.text(
        0.4, -1.85,
        "• m = 2 muts: 2.73 + ln(120) [4.79] + 2.20 =  9.71 nats  (3rd mut on leaf: +3.27 nats)",
        fontsize=6.65, color="#222222", ha="left", va="center",
    )
    ax_a.text(
        0.4, -2.45,
        "• m = 3 muts: 3.36 + ln(560) [6.33] + 3.30 = 12.99 nats  (m = 4: 15.72; m = 5: 18.05 nats)",
        fontsize=6.65, color="#222222", ha="left", va="center",
    )

    # =========================================================================
    # PANEL B: Star Polytomy (T_star) vs. Resolved Quartet (T_split)
    # =========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_xlim(-0.55, 10.75)
    ax_b.set_ylim(0, 6.4)
    ax_b.axis("off")

    ax_b.text(
        -0.45, 6.05,
        "B   Star Polytomy (T_star) vs. Resolved Quartet (T_split)",
        fontsize=9.6, color="#1a1a1a", ha="left", va="center",
    )
    ax_b.text(
        -0.45, 5.42,
        "Factoring m_e shared A→G sites onto stem e saves ΔS(m_e) = 2 [L_branch(2+m_e, 16) − L_branch(2, 16)]",
        fontsize=6.8, color="#333333", ha="left", va="center",
    )

    # Left: Star tree
    hub_x, hub_y = 1.65, 3.35
    ax_b.plot([hub_x, 0.75], [hub_y, 4.40], color=c_vermillion, lw=2.3)
    ax_b.plot([hub_x, 0.75], [hub_y, 2.30], color=c_vermillion, lw=2.3)
    ax_b.plot([hub_x, 2.55], [hub_y, 4.40], color="#555555", lw=1.8)
    ax_b.plot([hub_x, 2.55], [hub_y, 2.30], color="#555555", lw=1.8)
    ax_b.scatter([hub_x], [hub_y], s=46, color="#1e3a8a", zorder=5)
    ax_b.text(hub_x, hub_y + 0.35, "Hub r", fontsize=7.0, color="#1e3a8a", ha="center")

    ax_b.text(0.65, 4.45, "T1\n2+m_e muts", fontsize=6.6, color="#222222", ha="right", va="center")
    ax_b.text(0.65, 2.25, "T2\n2+m_e muts", fontsize=6.6, color="#222222", ha="right", va="center")
    ax_b.text(2.65, 4.45, "T3\n2 muts (9.71)", fontsize=6.5, color="#333333", ha="left", va="center")
    ax_b.text(2.65, 2.25, "T4\n2 muts (9.71)", fontsize=6.5, color="#333333", ha="left", va="center")
    ax_b.text(
        1.65, 1.52,
        "Unresolved Star T_star\n(0 internal splits; m_e duplicated\nin parallel on both T1 and T2)",
        fontsize=6.6, color="#444444", ha="center", va="center",
    )

    # Right: Split quartet
    u_x, v_x, q_y = 7.35, 8.65, 3.35
    ax_b.plot([u_x, v_x], [q_y, q_y], color=c_blue, lw=2.8)
    ax_b.plot([u_x, 6.55], [q_y, 4.40], color="#555555", lw=1.8)
    ax_b.plot([u_x, 6.55], [q_y, 2.30], color="#555555", lw=1.8)
    ax_b.plot([v_x, 9.45], [q_y, 4.40], color="#555555", lw=1.8)
    ax_b.plot([v_x, 9.45], [q_y, 2.30], color="#555555", lw=1.8)
    ax_b.scatter([u_x, v_x], [q_y, q_y], s=38, color=c_blue, zorder=5)

    ax_b.text(8.00, q_y + 0.42, "Split e (m_e muts)", fontsize=6.7, color=c_blue, ha="center")
    ax_b.text(8.00, q_y - 0.56, "L_branch(m_e)\n+ L_topo = 1.10", fontsize=6.4, color=c_blue, ha="center")
    ax_b.text(6.45, 4.45, "T1\n2 muts (9.71)", fontsize=6.5, color="#333333", ha="right", va="center")
    ax_b.text(6.45, 2.25, "T2\n2 muts (9.71)", fontsize=6.5, color="#333333", ha="right", va="center")
    ax_b.text(9.55, 4.45, "T3\n2 muts (9.71)", fontsize=6.5, color="#333333", ha="left", va="center")
    ax_b.text(9.55, 2.25, "T4\n2 muts (9.71)", fontsize=6.5, color="#333333", ha="left", va="center")
    ax_b.text(
        8.00, 1.52,
        "Resolved Quartet T_split\n(1 of 3 quartets: L_topo = ln 3 = 1.10;\nshared muts encoded once on stem e)",
        fontsize=6.6, color="#444444", ha="center", va="center",
    )

    box_b = mpatches.FancyBboxPatch(
        (-0.48, 0.08), 11.16, 0.98,
        boxstyle="round,pad=0.06,rounding_size=0.15",
        facecolor="#fdf8ec", edgecolor="#e2c07d", lw=0.9,
    )
    ax_b.add_patch(box_b)
    ax_b.text(
        5.10, 0.72,
        "Two-Part MDL Split Law (Eq. 5): G_MDL(e) = L_MDL(T_star) − L_MDL(T_split) = ΔS(m_e) − L_branch(m_e, 16) − ln 3",
        fontsize=5.35, color="#6b4400", ha="center", va="center",
    )
    ax_b.text(
        5.10, 0.32,
        "Split e is retained iff G_MDL(e) > 0 nats  (Posterior Support P(e | A) = 1 / [1 + exp(−G_MDL(e))])",
        fontsize=5.55, color="#444444", ha="center", va="center",
    )

    # =========================================================================
    # PANEL C: MDL Split Threshold Bar Chart (Exact Reconciled Ledger)
    # =========================================================================
    ax_c = fig.add_subplot(gs[1, 0])
    regimes = [
        "m_e = 1 Shared Site\n(Polytomy Regime)",
        "m_e = 2 Shared Sites\n(Resolved Clade)",
        "m_e = 3 Shared Sites\n(Strong Clade)",
    ]
    x_pos = [0.0, 1.40, 2.80]
    w = 0.27

    leaf_sav = [6.55, 12.02, 16.69]
    split_cost = [6.66, 10.81, 14.08]
    net_gain = [-0.11, 1.21, 2.60]
    post_probs = [0.47, 0.77, 0.93]

    ax_c.bar(
        [x - w for x in x_pos], leaf_sav, width=w,
        color=c_blue, edgecolor="#004b76", lw=0.8,
        label="Exact Leaf Savings ΔS(m_e) (nats)",
    )
    ax_c.bar(
        x_pos, split_cost, width=w,
        color=c_orange, edgecolor="#995c00", lw=0.8,
        label="Internal Split Cost (L_branch + ln 3)",
    )
    net_colors = [c_vermillion, c_sky, c_sky]
    ax_c.bar(
        [x + w for x in x_pos], net_gain, width=w,
        color=net_colors, edgecolor="#1a1a1a", lw=0.8,
        label="Net Two-Part MDL Gain G_MDL(e)",
    )

    ax_c.axhline(0.0, color="#222222", lw=0.9)
    for i, x in enumerate(x_pos):
        ax_c.text(x - w, leaf_sav[i] + 0.35, f"+{leaf_sav[i]:.2f}", ha="center", va="bottom", fontsize=7.2, color="#004b76")
        ax_c.text(x, split_cost[i] + 0.35, f"−{split_cost[i]:.2f}", ha="center", va="bottom", fontsize=7.2, color="#8a4b00")
        if net_gain[i] < 0:
            ax_c.text(
                x + w + 0.04, net_gain[i] - 0.32,
                f"−{abs(net_gain[i]):.2f} nats\nCONTRACT\n(P = {post_probs[i]:.2f})",
                ha="center", va="top", fontsize=6.9, color=c_vermillion,
            )
        else:
            ax_c.text(
                x + w + 0.16, net_gain[i] + 0.32,
                f"+{net_gain[i]:.2f} nats\nRESOLVE\n(P = {post_probs[i]:.2f})",
                ha="center", va="bottom", fontsize=6.9, color="#004b76",
            )

    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(regimes, fontsize=7.8)
    ax_c.set_ylabel("Information Code Length (nats)", fontsize=8.2)
    ax_c.set_xlim(-0.55, 3.55)
    ax_c.set_ylim(-3.8, 18.2)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.set_title(
        "C   MDL Split Threshold: Why m_e = 1 Contracts while m_e ≥ 2 Resolves",
        loc="left", fontsize=9.6, color="#1a1a1a", pad=8,
    )
    ax_c.legend(loc="upper left", frameon=True, fontsize=7.1)

    # =========================================================================
    # PANEL D: Asymmetric Branch Weights (LBA Immunity) & Lateral Shunts
    # =========================================================================
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_xlim(0, 10.95)
    ax_d.set_ylim(0, 6.4)
    ax_d.axis("off")

    ax_d.text(
        0.0, 6.10,
        "D   Asymmetric Branch Weights (LBA Immunity) & Lateral Shunts",
        fontsize=9.6, color="#1a1a1a", ha="left", va="center",
    )

    box_d1 = mpatches.FancyBboxPatch(
        (0.05, 3.05), 10.80, 2.68,
        boxstyle="round,pad=0.08,rounding_size=0.18",
        facecolor="#f5f0f9", edgecolor=c_purple, lw=0.95,
    )
    ax_d.add_patch(box_d1)
    ax_d.text(
        0.25, 5.36,
        "1. Why Combinadic Marginal Weights Resist Long-Branch Attraction (Eq. 4)",
        fontsize=6.8, color="#5b2166", ha="left", va="center",
    )
    ax_d.text(
        0.25, 4.82,
        "Suppose T1 & T3 are LONG branches (m_long = 6 muts on L = 16) and T2 & T4 are SHORT (m_short = 1 mut):",
        fontsize=5.45, color="#222222", ha="left", va="center",
    )
    ax_d.text(
        0.35, 4.32,
        "• Short-branch positional weight: w_short = ln[(16 − 1)/(1 + 1)] = ln(7.5)  = 2.01 nats/site",
        fontsize=5.55, color="#222222", ha="left", va="center",
    )
    ax_d.text(
        0.35, 3.84,
        "• Long-branch positional weight:  w_long  = ln[(16 − 6)/(6 + 1)] = ln(10/7) = 0.36 nats/site",
        fontsize=5.55, color="#222222", ha="left", va="center",
    )
    ax_d.text(
        0.35, 3.32,
        "→ Grouping the two LONG branches {T1, T3} on a chance collision saves only 2 w_long = 0.71 nats of site\n"
        "   address, whereas grouping the SHORT branches saves 2 w_short = 4.03 nats (5.6× stronger signal).",
        fontsize=5.45, color="#5b2166", ha="left", va="center",
    )

    box_d2 = mpatches.FancyBboxPatch(
        (0.05, 0.08), 10.80, 2.68,
        boxstyle="round,pad=0.08,rounding_size=0.18",
        facecolor="#eef7fc", edgecolor=c_blue, lw=0.95,
    )
    ax_d.add_patch(box_d2)
    ax_d.text(
        0.25, 2.40,
        "2. When Non-Sister Lineages Share Parallel States: Lateral Shunt Peeling (Eq. 7)",
        fontsize=6.8, color="#004b76", ha="left", va="center",
    )
    ax_d.text(
        0.25, 1.88,
        "Once backbone tree ((T1,T2),(T3,T4)) is fixed, suppose non-sister clades share m parallel replacements:",
        fontsize=5.45, color="#222222", ha="left", va="center",
    )
    ax_d.text(
        0.35, 1.40,
        "• Endpoint Address Header: 2 ln |V| nats to specify donor and recipient nodes",
        fontsize=5.55, color="#222222", ha="left", va="center",
    )
    ax_d.text(
        0.35, 0.94,
        "• Contiguous HGT Tract:    L_mask^INTERVAL = ln 2 + 2 ln L + L_branch(d_[a,b], b − a + 1)",
        fontsize=5.55, color="#222222", ha="left", va="center",
    )
    ax_d.text(
        0.35, 0.48,
        "• Scattered Convergence:   L_mask^SPARSE = ln 2 + ln C(M_pool, m) + L_N(m − m_min) (inferred iff savings > header + mask)",
        fontsize=5.20, color="#004b76", ha="left", va="center",
    )

    fig.savefig(str(out_pdf), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
