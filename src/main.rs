use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::env;
use std::fs;
use std::path::Path;
use std::process;

use bushwhacker::discrete::{
    encode_discrete_sequences, infer_discrete_mdl, parse_discrete_file, parse_discrete_str,
    DiscreteMode,
};
use bushwhacker::{
    compress_alignment_to_bwz, configure_thread_pool, decompress_alignment_from_bwz,
    detect_sequence_alphabet_kind, looks_like_csv_or_tsv_matrix, parse_alignment_str,
    BushwhackerEngine, BushwhackerResult,
};

fn print_usage(prog: &str) {
    eprintln!(
        "BUSHWHACKER: First-Principles Kolmogorov / MDL Phylogenetic Graph Induction Engine\n\
         (Bipartition Unification, Shunt Hunting, & Weighted Hub Aggregation via\n\
          Combinadic Kolmogorov Entropy Reduction)\n\n\
         Usage:\n  \
           {prog} <input.fasta|phylip|nexus|csv|tsv|bwz> [options]\n\n\
         Supported Input Formats (auto-detected):\n  \
           - FASTA   (.fa, .fasta, .fna)\n  \
           - PHYLIP  (.phy, .phylip; strict 10-char or relaxed, sequential or interleaved)\n  \
           - NEXUS   (.nex, .nexus, .nxs; DATA or CHARACTERS blocks, sequential or interleaved)\n  \
           - CSV/TSV (.csv, .tsv; K-state discrete or ordered copy-number / genotype matrices)\n  \
           - BWZ     (.bwz; Bushwhacker lossless phylogenetic archive for --decompress)\n\n\
         Options:\n  \
           -o, --out-newick <PATH>      Output Newick file path (with MDL branch supports & lengths)\n  \
           -b, --out-binary <PATH>      Output strictly binary Newick file path (for iqtree3 -t)\n  \
           -t, --contract-tree <PATH>   Contract unsupported splits & peel shunts on an input Newick tree\n  \
           -j, --out-json <PATH>        Output JSON report path (full MDL ledger, splits, & shunts)\n  \
           -c, --compress <PATH>        Write lossless .bwz phylogenetic archive to <PATH>\n  \
           -d, --decompress <PATH>      Decompress input .bwz archive into FASTA at <PATH>\n  \
           --protein                    Force 20-state amino-acid MDL mode (auto-detected by default)\n  \
           --translate                  Translate in-frame nucleotide CDS to 20-state protein before MDL\n  \
           --binary                     Force unordered 2-state (0/1) discrete character MDL\n  \
           --discrete                   Unordered K-state discrete character-matrix MDL (2 <= K <= 32)\n  \
           --ordered                    Ordered Wagner K-state copy-number / genotype MDL (0..K-1)\n  \
           --no-coding                  Disable automatic coding-frame 3rd-position entropy weighting\n  \
           --no-shunts                  Disable Stage 4 reticulate shunt peeling\n  \
           --min-shunt-sites <N>        Minimum parallel mutations to evaluate a shunt (default: 2)\n  \
           --no-mult-pen                Ablation: omit Rissanen branch multiplicity penalty\n  \
           --empirical-alpha            Ablation: estimate Gamma(alpha, alpha) rate shape from Fitch steps\n  \
           -T, --threads <N>            Worker threads for parallel MDL induction (default: all cores)\n  \
           -h, --help                   Print this help message\n  \
           -V, --version                Print version information"
    );
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn clade_leaves_for_shunt_endpoint(
    adj: &BTreeMap<usize, Vec<usize>>,
    names: &[String],
    n: usize,
    u: usize,
    other: usize,
) -> (usize, Vec<String>) {
    if u < n {
        return (1, vec![names[u].clone()]);
    }
    // Find neighbor of `u` on the path toward `other`
    let mut parent: HashMap<usize, usize> = HashMap::new();
    let mut q = VecDeque::from([u]);
    let mut visited: HashSet<usize> = HashSet::from([u]);
    while let Some(curr) = q.pop_front() {
        if curr == other {
            break;
        }
        if let Some(nbs) = adj.get(&curr) {
            for &nb in nbs {
                if visited.insert(nb) {
                    parent.insert(nb, curr);
                    q.push_back(nb);
                }
            }
        }
    }
    let mut step_to_other = None;
    let mut cur = other;
    while let Some(&p) = parent.get(&cur) {
        if p == u {
            step_to_other = Some(cur);
            break;
        }
        cur = p;
    }

    // Collect leaves in each branch attached to `u` except `step_to_other`
    let mut branch_leaf_sets: Vec<Vec<String>> = Vec::new();
    if let Some(nbs) = adj.get(&u) {
        for &nb in nbs {
            if Some(nb) == step_to_other {
                continue;
            }
            let mut leaves = Vec::new();
            let mut bq = VecDeque::from([nb]);
            let mut bvis = HashSet::from([u, nb]);
            while let Some(x) = bq.pop_front() {
                if x < n {
                    leaves.push(names[x].clone());
                } else if let Some(xnbs) = adj.get(&x) {
                    for &y in xnbs {
                        if bvis.insert(y) {
                            bq.push_back(y);
                        }
                    }
                }
            }
            if !leaves.is_empty() {
                branch_leaf_sets.push(leaves);
            }
        }
    }
    let total_away: usize = branch_leaf_sets.iter().map(|v| v.len()).sum();
    if total_away > 0 && total_away <= n / 2 {
        let mut all_away: Vec<String> = branch_leaf_sets.into_iter().flatten().collect();
        let full_sz = all_away.len();
        all_away.sort();
        all_away.truncate(64);
        return (full_sz, all_away);
    }
    if let Some(mut smallest) = branch_leaf_sets.into_iter().min_by_key(|v| v.len()) {
        let full_sz = smallest.len();
        smallest.sort();
        smallest.truncate(64);
        return (full_sz, smallest);
    }
    (0, Vec::new())
}

fn serialize_json(res: &BushwhackerResult, fasta_path: &str, alphabet_label: &str) -> String {
    let one_to_one_sites = alphabet_label != "nucleotide";
    let internal_branches: Vec<_> = res
        .branches
        .iter()
        .filter(|b| b.is_internal)
        .map(|b| {
            format!(
                "    {{\"u\": {}, \"v\": {}, \"mutations\": {:.2}, \"branch_length\": {:.6}, \"mdl_gain_nats\": {:.3}, \"mdl_support\": {:.4}, \"num_synapomorphies\": {}}}",
                b.u,
                b.v,
                b.mutation_count,
                b.branch_length,
                b.mdl_gain_nats,
                b.mdl_support,
                b.synapomorphy_sites.len()
            )
        })
        .collect();

    let shunt_entries: Vec<_> = res
        .shunts
        .iter()
        .map(|s| {
            let sites_str = s
                .sites
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let codons_str = s
                .sites
                .iter()
                .map(|x| {
                    if one_to_one_sites {
                        (x + 1).to_string()
                    } else {
                        (x / 3 + 1).to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            let span_str = match s.interval_span {
                Some((a, b)) => format!("[{a}, {b}]"),
                None => "null".to_string(),
            };
            let (sz_u, leaves_u_vec) = clade_leaves_for_shunt_endpoint(
                &res.adjacency,
                &res.names,
                res.n_taxa,
                s.node_u,
                s.node_v,
            );
            let leaves_u = leaves_u_vec
                .into_iter()
                .map(|x| json_escape(&x))
                .collect::<Vec<_>>()
                .join(", ");
            let (sz_v, leaves_v_vec) = clade_leaves_for_shunt_endpoint(
                &res.adjacency,
                &res.names,
                res.n_taxa,
                s.node_v,
                s.node_u,
            );
            let leaves_v = leaves_v_vec
                .into_iter()
                .map(|x| json_escape(&x))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "    {{\n      \"node_u\": {},\n      \"node_v\": {},\n      \"label_u\": {},\n      \"label_v\": {},\n      \"clade_size_u\": {},\n      \"clade_size_v\": {},\n      \"leaves_u\": [{}],\n      \"leaves_v\": [{}],\n      \"mode\": {},\n      \"shared_mutations\": {},\n      \"sites_0based\": [{}],\n      \"codons_1based\": [{}],\n      \"residues_1based\": [{}],\n      \"interval_span\": {},\n      \"data_savings_nats\": {:.3},\n      \"mask_cost_nats\": {:.3},\n      \"endpoint_cost_nats\": {:.3},\n      \"net_mdl_gain_nats\": {:.3},\n      \"posterior_prob\": {:.4}\n    }}",
                s.node_u,
                s.node_v,
                json_escape(&s.label_u),
                json_escape(&s.label_v),
                sz_u,
                sz_v,
                leaves_u,
                leaves_v,
                json_escape(&s.mode),
                s.shared_mutations,
                sites_str,
                codons_str,
                codons_str,
                span_str,
                s.data_savings_nats,
                s.mask_cost_nats,
                s.endpoint_cost_nats,
                s.net_mdl_gain_nats,
                s.posterior_prob
            )
        })
        .collect();

    format!(
        "{{\n  \"tool\": \"BUSHWHACKER\",\n  \"input\": {},\n  \"alphabet\": {},\n  \"n_taxa\": {},\n  \"seq_len\": {},\n  \"raw_Description_Length_nats\": {:.2},\n  \"star_tree_MDL_nats\": {:.2},\n  \"tree_MDL_nats\": {:.2},\n  \"network_MDL_nats\": {:.2},\n  \"compression_ratio_vs_star\": {:.4},\n  \"resolved_internal_splits\": {},\n  \"max_binary_splits\": {},\n  \"polytomy_nodes\": {},\n  \"certified_shunts\": {},\n  \"runtime_sec\": {:.4},\n  \"newick\": {},\n  \"binary_newick\": {},\n  \"internal_branches\": [\n{}\n  ],\n  \"shunts\": [\n{}\n  ]\n}}\n",
        json_escape(fasta_path),
        json_escape(alphabet_label),
        res.n_taxa,
        res.n_sites,
        res.raw_uncompressed_nats,
        res.star_tree_nats,
        res.backbone_tree_nats,
        res.full_graph_nats,
        res.full_graph_nats / res.star_tree_nats.max(1e-9),
        res.n_internal_branches,
        res.max_binary_internal_branches,
        res.polytomy_nodes,
        res.shunts.len(),
        res.runtime_sec,
        json_escape(&res.newick),
        json_escape(&res.binary_newick),
        internal_branches.join(",\n"),
        shunt_entries.join(",\n")
    )
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let prog = args.first().map(|s| s.as_str()).unwrap_or("bushwhacker");

    if args.len() < 2 {
        print_usage(prog);
        process::exit(1);
    }

    let mut fasta_path: Option<String> = None;
    let mut out_newick: Option<String> = None;
    let mut out_binary: Option<String> = None;
    let mut contract_tree: Option<String> = None;
    let mut out_json: Option<String> = None;
    let mut out_compress: Option<String> = None;
    let mut out_decompress: Option<String> = None;
    let mut no_coding = false;
    let mut no_shunts = false;
    let mut force_protein = false;
    let mut translate_coding = false;
    let mut discrete_mode: Option<DiscreteMode> = None;
    // Output `alphabet` label for the flagged discrete path; matches the
    // `alphabet_mode` vocabulary emitted by the WASM dispatcher.
    let mut discrete_label: &str = "discrete";
    let mut min_shunt_sites: usize = 2;
    let mut no_mult_pen = false;
    let mut empirical_alpha = false;
    let mut num_threads: Option<usize> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_usage(prog);
                process::exit(0);
            }
            "-V" | "--version" => {
                println!("bushwhacker {}", env!("CARGO_PKG_VERSION"));
                process::exit(0);
            }
            "-o" | "--out-newick" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: missing argument for {}", args[i - 1]);
                    process::exit(1);
                }
                out_newick = Some(args[i].clone());
            }
            "-b" | "--out-binary" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: missing argument for {}", args[i - 1]);
                    process::exit(1);
                }
                out_binary = Some(args[i].clone());
            }
            "-t" | "--contract-tree" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: missing argument for {}", args[i - 1]);
                    process::exit(1);
                }
                contract_tree = Some(args[i].clone());
            }
            "-j" | "--out-json" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: missing argument for {}", args[i - 1]);
                    process::exit(1);
                }
                out_json = Some(args[i].clone());
            }
            "-c" | "--compress" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: missing argument for {}", args[i - 1]);
                    process::exit(1);
                }
                out_compress = Some(args[i].clone());
            }
            "-d" | "--decompress" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: missing argument for {}", args[i - 1]);
                    process::exit(1);
                }
                out_decompress = Some(args[i].clone());
            }
            "--protein" => {
                force_protein = true;
            }
            "--translate" => {
                translate_coding = true;
            }
            "--binary" => {
                discrete_mode = Some(DiscreteMode::Unordered);
                discrete_label = "binary";
            }
            "--discrete" => {
                discrete_mode = Some(DiscreteMode::Unordered);
            }
            "--ordered" => {
                discrete_mode = Some(DiscreteMode::Ordered);
                discrete_label = "ordered";
            }
            "--no-coding" => {
                no_coding = true;
            }
            "--no-shunts" => {
                no_shunts = true;
            }
            "--no-mult-pen" => {
                no_mult_pen = true;
            }
            "--empirical-alpha" => {
                empirical_alpha = true;
            }
            "--min-shunt-sites" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: missing argument for --min-shunt-sites");
                    process::exit(1);
                }
                min_shunt_sites = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("Error: invalid integer for --min-shunt-sites: {}", args[i]);
                    process::exit(1);
                });
            }
            "-T" | "--threads" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: missing argument for {}", args[i - 1]);
                    process::exit(1);
                }
                let t: usize = args[i].parse().unwrap_or_else(|_| {
                    eprintln!("Error: invalid integer for --threads: {}", args[i]);
                    process::exit(1);
                });
                if t == 0 {
                    eprintln!("Error: --threads must be >= 1");
                    process::exit(1);
                }
                num_threads = Some(t);
            }
            arg if arg.starts_with('-') => {
                eprintln!("Error: unknown option '{arg}'");
                print_usage(prog);
                process::exit(1);
            }
            arg => {
                if fasta_path.is_none() {
                    fasta_path = Some(arg.to_string());
                } else {
                    eprintln!("Error: unexpected positional argument '{arg}'");
                    process::exit(1);
                }
            }
        }
        i += 1;
    }

    let Some(fasta) = fasta_path else {
        print_usage(prog);
        process::exit(1);
    };

    if let Some(t) = num_threads {
        configure_thread_pool(t);
    }

    if let Some(dec_fasta_out) = out_decompress {
        let bwz_bytes = fs::read(&fasta).unwrap_or_else(|e| {
            eprintln!("Error reading .bwz archive '{fasta}': {e}");
            process::exit(1);
        });
        let (names, seqs, newick) = decompress_alignment_from_bwz(&bwz_bytes).unwrap_or_else(|e| {
            eprintln!("Error decompressing '{fasta}': {e}");
            process::exit(1);
        });
        let mut out_str = String::with_capacity(names.len() * (seqs[0].len() + 32));
        for (nm, sq) in names.iter().zip(&seqs) {
            out_str.push('>');
            out_str.push_str(nm);
            out_str.push('\n');
            out_str.push_str(sq);
            out_str.push('\n');
        }
        fs::write(&dec_fasta_out, out_str).unwrap_or_else(|e| {
            eprintln!("Error writing decompressed FASTA to '{dec_fasta_out}': {e}");
            process::exit(1);
        });
        println!(
            "Decompressed {} taxa x {} sites from '{}' -> '{}'",
            names.len(),
            seqs.first().map(|s| s.len()).unwrap_or(0),
            fasta,
            dec_fasta_out
        );
        if let Some(nwk_path) = out_newick {
            fs::write(&nwk_path, format!("{newick}\n")).unwrap_or_else(|e| {
                eprintln!("Error writing Newick to '{nwk_path}': {e}");
                process::exit(1);
            });
            println!("Extracted embedded Newick tree to: {nwk_path}");
        }
        return;
    }

    let mut engine = BushwhackerEngine::new(!no_coding, !no_shunts, min_shunt_sites);
    engine.force_protein = force_protein;
    engine.translate_coding = translate_coding;
    engine.discrete_mode = discrete_mode;
    engine.no_mult_pen = no_mult_pen;
    engine.empirical_alpha = empirical_alpha;

    // The contract-tree input is shared by every inference path below; read it
    // once instead of duplicating the load in each branch.
    let starting_tree: Option<String> = contract_tree.as_ref().map(|p| {
        fs::read_to_string(p).unwrap_or_else(|e| {
            eprintln!("Error reading input tree '{p}': {e}");
            process::exit(1);
        })
    });
    let starting_newick: Option<(&str, bool)> = starting_tree.as_deref().map(|s| (s, false));

    let (names, seqs_opt, alphabet_label, res) = if let Some(d_mode) = discrete_mode {
        let cm = parse_discrete_file(Path::new(&fasta)).unwrap_or_else(|e| {
            eprintln!("Error reading discrete character matrix '{fasta}': {e}");
            process::exit(1);
        });
        let r = infer_discrete_mdl(&engine, &cm, d_mode, starting_newick).unwrap_or_else(|e| {
            eprintln!("Error running discrete BUSHWHACKER on '{fasta}': {e}");
            process::exit(1);
        });
        (cm.names, None, discrete_label, r)
    } else {
        // Mirror the WASM `auto` dispatch in `infer_any_alignment_json`: CSV/TSV
        // character matrices and 0/1 or 0..K-1 alignment-format inputs route to
        // the discrete engine; protein and nucleotide stay on the primary engine.
        let input_text = fs::read_to_string(&fasta).unwrap_or_else(|e| {
            eprintln!("Error reading alignment '{fasta}': {e}");
            process::exit(1);
        });
        if looks_like_csv_or_tsv_matrix(&input_text) {
            let cm = parse_discrete_str(&input_text).unwrap_or_else(|e| {
                eprintln!("Error reading discrete character matrix '{fasta}': {e}");
                process::exit(1);
            });
            let (d_mode, lbl) = if cm.k_states <= 2 {
                (DiscreteMode::Unordered, "binary")
            } else {
                (DiscreteMode::Ordered, "ordered")
            };
            engine.discrete_mode = Some(d_mode);
            let r = infer_discrete_mdl(&engine, &cm, d_mode, starting_newick).unwrap_or_else(|e| {
                eprintln!("Error running discrete BUSHWHACKER on '{fasta}': {e}");
                process::exit(1);
            });
            (cm.names, None, lbl, r)
        } else {
            let (nms, sqs) = parse_alignment_str(&input_text).unwrap_or_else(|e| {
                eprintln!("Error reading alignment '{fasta}': {e}");
                process::exit(1);
            });
            let alphabet_kind = detect_sequence_alphabet_kind(&sqs);
            let is_prot = force_protein || translate_coding || alphabet_kind == "protein";
            if !is_prot && matches!(alphabet_kind, "binary" | "ordered") {
                let cm = encode_discrete_sequences(&nms, &sqs).unwrap_or_else(|e| {
                    eprintln!("Error encoding discrete characters in '{fasta}': {e}");
                    process::exit(1);
                });
                let d_mode = if alphabet_kind == "ordered" {
                    DiscreteMode::Ordered
                } else {
                    DiscreteMode::Unordered
                };
                engine.discrete_mode = Some(d_mode);
                let r = infer_discrete_mdl(&engine, &cm, d_mode, starting_newick).unwrap_or_else(|e| {
                    eprintln!("Error running discrete BUSHWHACKER on '{fasta}': {e}");
                    process::exit(1);
                });
                (cm.names, None, alphabet_kind, r)
            } else {
                let r = if let Some((in_nwk, unrooted)) = starting_newick {
                    engine
                        .infer_with_starting_newick(&nms, &sqs, in_nwk, unrooted)
                        .unwrap_or_else(|e| {
                            eprintln!(
                                "Error contracting tree '{}' on '{fasta}': {e}",
                                contract_tree.as_deref().unwrap_or_default()
                            );
                            process::exit(1);
                        })
                } else {
                    engine.infer(&nms, &sqs).unwrap_or_else(|e| {
                        eprintln!("Error running BUSHWHACKER on '{fasta}': {e}");
                        process::exit(1);
                    })
                };
                (
                    nms,
                    Some(sqs),
                    if is_prot { "protein" } else { "nucleotide" },
                    r,
                )
            }
        }
    };
    let one_to_one_sites = alphabet_label != "nucleotide";

    println!("{}", "=".repeat(78));
    println!(
        "BUSHWHACKER Summary: {}  (N={} taxa, L={} sites)",
        fasta, res.n_taxa, res.n_sites
    );
    println!("{}", "-".repeat(78));
    println!("  Raw Uncompressed DL   : {:12.1} nats", res.raw_uncompressed_nats);
    println!("  Star Tree MDL (Null)  : {:12.1} nats", res.star_tree_nats);
    println!(
        "  Induced Tree MDL      : {:12.1} nats  ({:.2}% of Star)",
        res.backbone_tree_nats,
        100.0 * res.backbone_tree_nats / res.star_tree_nats.max(1e-9)
    );
    println!(
        "  Induced Network MDL   : {:12.1} nats  ({:.2}% of Star)",
        res.full_graph_nats,
        100.0 * res.full_graph_nats / res.star_tree_nats.max(1e-9)
    );
    println!(
        "  Resolved Splits       : {:4} / {:4} binary splits ({} polytomy hubs)",
        res.n_internal_branches, res.max_binary_internal_branches, res.polytomy_nodes
    );
    println!("  Certified Shunts      : {:4}", res.shunts.len());
    for (idx, s) in res.shunts.iter().enumerate() {
        let codons: Vec<usize> = s
            .sites
            .iter()
            .map(|&x| if one_to_one_sites { x + 1 } else { x / 3 + 1 })
            .collect();
        println!(
            "    [{}] {}  {} <-> {}  (+{:.2} nats, P={:.1}%, m={} sites, {}={:?})",
            idx + 1,
            s.mode,
            s.label_u,
            s.label_v,
            s.net_mdl_gain_nats,
            100.0 * s.posterior_prob,
            s.shared_mutations,
            match alphabet_label {
                "protein" => "residues",
                "nucleotide" => "codons",
                _ => "features_1based",
            },
            codons
        );
    }
    println!("  Runtime               : {:.4} s", res.runtime_sec);
    println!("{}", "=".repeat(78));

    if let Some(nwk_path) = out_newick {
        fs::write(&nwk_path, format!("{}\n", res.newick)).unwrap_or_else(|e| {
            eprintln!("Error writing Newick to '{nwk_path}': {e}");
            process::exit(1);
        });
        println!("Wrote Newick tree to: {nwk_path}");
    } else if out_compress.is_none() && out_binary.is_none() {
        println!("\nNewick Tree:\n{}", res.newick);
    }

    if let Some(bin_path) = out_binary {
        fs::write(&bin_path, format!("{}\n", res.binary_newick)).unwrap_or_else(|e| {
            eprintln!("Error writing binary Newick to '{bin_path}': {e}");
            process::exit(1);
        });
        println!("Wrote strictly binary Newick tree to: {bin_path}");
    }

    if let Some(json_path) = out_json {
        let json_str = serialize_json(&res, &fasta, alphabet_label);
        fs::write(&json_path, json_str).unwrap_or_else(|e| {
            eprintln!("Error writing JSON to '{json_path}': {e}");
            process::exit(1);
        });
        println!("Wrote JSON report to: {json_path}");
    }

    if let Some(bwz_path) = out_compress {
        let Some(seqs) = seqs_opt else {
            eprintln!("Error: .bwz compression currently requires sequence FASTA/PHYLIP/NEXUS input.");
            process::exit(1);
        };
        let (bwz_bytes, stats) =
            compress_alignment_to_bwz(&names, &seqs, &res).unwrap_or_else(|e| {
                eprintln!("Error compressing alignment to BWZ: {e}");
                process::exit(1);
            });
        fs::write(&bwz_path, &bwz_bytes).unwrap_or_else(|e| {
            eprintln!("Error writing BWZ archive to '{bwz_path}': {e}");
            process::exit(1);
        });
        println!(
            "Wrote lossless BWZ archive to: {bwz_path}\n  Sequence Payload : {:8} B ({:9.1} nats, {:5.1}x vs raw ASCII {:8} B)\n  Taxon Headers    : {:8} B\n  Total Archive    : {:8} B",
            stats.compressed_seq_bytes,
            stats.seq_mdl_nats,
            stats.raw_seq_bytes as f64 / stats.compressed_seq_bytes.max(1) as f64,
            stats.raw_seq_bytes,
            stats.compressed_header_bytes,
            stats.total_bwz_bytes
        );
    }
}
