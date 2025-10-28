# pedigree_tools.py (aDNA-ready, standalone)
# - Safe drop-in: does NOT assume your old file structure
# - Adds ROH-aware utilities via ibd_tools.load_haproh_roh / annotate_segments_with_roh
# - Provides pedigree loading/validation, degree prediction wrapper, and simple CLI
# - Optional imports guarded; designed to fail soft with clear errors

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import itertools as it
import networkx as nx
from collections import Counter
from typing import Dict, Tuple, List, Optional

# Optional plotting
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Optional sklearn (for degree prediction wrapper)
try:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    import pickle as pkl
except Exception:
    LinearDiscriminantAnalysis = None
    pkl = None

# Import ROH helpers from ibd_tools.py (the version I gave you)
try:
    from ibd_tools import load_haproh_roh, annotate_segments_with_roh
except Exception as e:
    load_haproh_roh = None
    annotate_segments_with_roh = None


############################################################
# Pedigree loading / utilities
############################################################

def load_pedigree_yaml(ped_yaml: str) -> nx.DiGraph:
    """
    Load a pedigree YAML into a directed graph (parent -> child).
    YAML schema is flexible; we support common shapes:
      - root mapping of individual IDs to dicts including:
          mother/father (IDs), sex (1=male,2=female or 'M'/'F'), age (optional)
    Nodes are strings; edges are parent->child.
    """
    if not os.path.exists(ped_yaml):
        raise FileNotFoundError(f"Pedigree file not found: {ped_yaml}")

    with open(ped_yaml, "r") as f:
        y = yaml.safe_load(f)

    g = nx.DiGraph()
    # First ensure nodes exist with attributes
    for person, data in y.items():
        if not g.has_node(person):
            g.add_node(person)
        # Normalize sex
        sex = str(data.get("sex", "")).strip()
        if sex.upper() in ["M", "MALE", "1"]:
            sex = "1"
        elif sex.upper() in ["F", "FEMALE", "2"]:
            sex = "2"
        else:
            sex = "0"  # unknown
        nx.set_node_attributes(g, {person: {
            "sex": sex,
            "age": data.get("age", np.nan),
        }})

    # Add parental edges if present
    for person, data in y.items():
        mother = data.get("mother", None)
        father = data.get("father", None)
        if mother not in [None, "", -1]:
            g.add_edge(str(mother), str(person))
        if father not in [None, "", -1]:
            g.add_edge(str(father), str(person))

    return g


def validate_pedigree(g: nx.DiGraph, allow_cycles: bool = False) -> None:
    """
    Basic consistency checks:
      - (optional) no cycles
      - in-degree per child ≤ 2 (<= 2 parents)
    Raises ValueError on problems.
    """
    if not allow_cycles:
        try:
            _ = nx.algorithms.dag.topological_sort(g)
        except nx.NetworkXUnfeasible:
            raise ValueError("Pedigree graph has cycles; set allow_cycles=True to skip this check.")

    bad = [n for n in g.nodes if g.in_degree(n) > 2]
    if bad:
        raise ValueError(f"Nodes with >2 parents found: {bad[:10]}{'...' if len(bad)>10 else ''}")


def get_full_sib_groups(g: nx.DiGraph) -> List[set]:
    """
    Return connected components of full-sibling relationships:
      siblings share BOTH parents (when known).
    """
    # Build undirected sib graph
    sib_g = nx.Graph()

    # Children grouped by their set of parents
    parent_sets = {}
    for child in g.nodes:
        parents = set(g.predecessors(child))
        if not parents:
            continue
        parent_sets.setdefault(frozenset(parents), []).append(child)

    for parents, kids in parent_sets.items():
        if len(parents) == 2 and len(kids) >= 2:
            for a, b in it.combinations(kids, 2):
                sib_g.add_edge(a, b)

    return list(nx.connected_components(sib_g))


############################################################
# IBD + ROH utilities for aDNA
############################################################

def apply_roh_annotation(
    ibd_path: str,
    roh_path: str,
    output_path: Optional[str] = None,
    roh_min_cM: float = 8.0,
    overlap_frac: float = 0.5,
) -> pd.DataFrame:
    """
    Load IBD (TSV) and hapROH (TSV), annotate IBD with ROH overlap columns.
    - ibd_path must have: chromosome|chrom, id1, id2, start_cm, end_cm
    - roh_path is the hapROH per-tract file
    """
    if annotate_segments_with_roh is None or load_haproh_roh is None:
        raise ImportError("ROH helpers unavailable. Ensure your updated ibd_tools.py is importable.")

    ibd_df = pd.read_csv(ibd_path, sep="\t")
    roh_dict = load_haproh_roh(roh_path)

    ann = annotate_segments_with_roh(
        ibd_df,
        roh_dict,
        roh_min_cM=roh_min_cM,
        overlap_frac=overlap_frac,
    )

    if output_path:
        ann.to_csv(output_path, sep="\t", index=False)
    return ann


def filter_pairs_by_roh_fraction(
    annotated_ibd: pd.DataFrame,
    require_non_roh: bool = False,
    max_roh_fraction_per_pair: float = 0.5,
) -> pd.DataFrame:
    """
    Summarize ROH overlap per pair and optionally filter:
      - If require_non_roh=True, drop pairs with ALL segments overlapping ROH in both samples.
      - Else, drop pairs whose total IBD2 fraction within ROH_both exceeds threshold.
    Expects columns added by annotate_segments_with_roh.
    """
    if annotated_ibd.empty:
        return annotated_ibd

    # derive cM length per segment
    tmp = annotated_ibd.copy()
    tmp["cm_len"] = (tmp["end_cm"] - tmp["start_cm"]).clip(lower=0)

    # Compute pair-level summaries
    grp = tmp.groupby(["id1", "id2"])
    sums = grp.agg(
        total_cm=("cm_len", "sum"),
        roh_both_cm=("cm_len", lambda s: s[tmp.loc[s.index, "in_roh_both"]].sum()),
        all_in_roh_both=("in_roh_both", "all"),
    ).reset_index()

    if require_non_roh:
        keep = sums[~sums["all_in_roh_both"]][["id1", "id2"]]
    else:
        frac = sums["roh_both_cm"] / sums["total_cm"].replace(0, np.nan)
        keep = sums[frac.fillna(0) <= float(max_roh_fraction_per_pair)][["id1", "id2"]]

    # Join back to segments
    key = ["id1", "id2"]
    keep_key = set(map(tuple, keep[key].values))
    mask = annotated_ibd[key].apply(tuple, axis=1).isin(keep_key)
    return annotated_ibd[mask].reset_index(drop=True)


############################################################
# Degree prediction wrapper (optional)
############################################################

def predict_degrees_with_lda(
    ibd_pairs_tsv: str,
    lda_pickle_path: str,
    out_path: Optional[str] = None,
    ibd1_col: str = "IBD1Seg",
    ibd2_col: str = "IBD2Seg",
) -> pd.DataFrame:
    """
    Convenience wrapper:
      - Load KING-like pair table (must contain ibd1_col, ibd2_col, ID1, ID2)
      - Load pre-trained LDA (e.g., degree classifier pkl)
      - Add 'predicted_degree' column and save/return
    """
    if LinearDiscriminantAnalysis is None or pkl is None:
        raise ImportError("scikit-learn not available; install to use degree prediction.")

    df = pd.read_csv(ibd_pairs_tsv, sep="\t", dtype={"ID1": str, "ID2": str})
    with open(lda_pickle_path, "rb") as f:
        lda = pkl.load(f)

    X = df[[ibd1_col, ibd2_col]].values
    df["predicted_degree"] = lda.predict(X)

    if out_path:
        df.to_csv(out_path, sep="\t", index=False)
    return df


############################################################
# Minimal plotting (optional)
############################################################

def plot_pair_roh_fraction(summary_df: pd.DataFrame, out_png: Optional[str] = None):
    """
    Quick diagnostic plot of roh_both_cm / total_cm by pair.
    Expects columns: id1, id2, total_cm, roh_both_cm
    """
    if plt is None:
        print("matplotlib not available; skipping plot.")
        return

    df = summary_df.copy()
    df["roh_frac"] = df["roh_both_cm"] / df["total_cm"].replace(0, np.nan)
    df["roh_frac"] = df["roh_frac"].fillna(0)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["roh_frac"], bins=30)
    ax.set_xlabel("Fraction of pair IBD within ROH (both)")
    ax.set_ylabel("Count")
    ax.set_title("Pair-level ROH burden within IBD")
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=200)
    plt.close(fig)
############################################################
# CLI
############################################################

def _cli():
    p = argparse.ArgumentParser(
        description="Pedigree utilities (aDNA-friendly): ROH-aware IBD annotation, pedigree loading, LDA degree prediction."
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # annotate IBD with ROH
    a = sub.add_parser("annotate_roh", help="Annotate an IBD TSV with ROH overlap columns.")
    a.add_argument("--ibd", required=True, help="IBD segments TSV (must include id1,id2,chromosome/start_cm/end_cm)")
    a.add_argument("--roh", required=True, help="hapROH tracts TSV")
    a.add_argument("--out", required=False, default=None, help="Output TSV (if omitted, prints path to stdout)")
    a.add_argument("--roh-min-cm", type=float, default=8.0, help="Minimum ROH tract length (cM) to consider")
    a.add_argument("--overlap-frac", type=float, default=0.5, help="Required fractional overlap for in_roh flags")

    # filter pairs by ROH burden (post-annotation)
    f = sub.add_parser("filter_roh", help="Filter annotated IBD by pair-level ROH burden.")
    f.add_argument("--annotated", required=True, help="Annotated IBD TSV produced by annotate_roh")
    f.add_argument("--out", required=True, help="Output filtered IBD TSV")
    f.add_argument("--require-non-roh", action="store_true", help="Drop pairs with ALL segments in ROH for both samples")
    f.add_argument("--max-roh-frac", type=float, default=0.5, help="Max allowed pair ROH-both fraction")

    # pedigree load + validate
    v = sub.add_parser("validate_ped", help="Validate a pedigree YAML (no cycles, <=2 parents).")
    v.add_argument("--yaml", required=True)

    # degree prediction wrapper (LDA)
    d = sub.add_parser("predict_degree", help="Predict degree of relatedness with a pre-trained LDA.")
    d.add_argument("--pairs", required=True, help="KING-like pairs TSV with IBD1/IBD2 columns")
    d.add_argument("--lda", required=True, help="Pickle file containing trained LDA")
    d.add_argument("--out", required=True, help="Output TSV with predicted_degree")
    d.add_argument("--ibd1-col", default="IBD1Seg")
    d.add_argument("--ibd2-col", default="IBD2Seg")

    args = p.parse_args()

    if args.cmd == "annotate_roh":
        ann = apply_roh_annotation(
            ibd_path=args.ibd,
            roh_path=args.roh,
            output_path=args.out,
            roh_min_cM=float(args.roh_min_cm),
            overlap_frac=float(args.overlap_frac),
        )
        if args.out:
            print(args.out)
        else:
            # if no path requested, just print a small preview
            print(ann.head().to_csv(sep="\t", index=False))

    elif args.cmd == "filter_roh":
        df = pd.read_csv(args.annotated, sep="\t")
        out = filter_pairs_by_roh_fraction(
            df,
            require_non_roh=bool(args.require_non_roh),
            max_roh_fraction_per_pair=float(args.max_roh_frac),
        )
        out.to_csv(args.out, sep="\t", index=False)
        print(args.out)

    elif args.cmd == "validate_ped":
        g = load_pedigree_yaml(args.yaml)
        validate_pedigree(g)
        print("OK")

    elif args.cmd == "predict_degree":
        res = predict_degrees_with_lda(
            ibd_pairs_tsv=args.pairs,
            lda_pickle_path=args.lda,
            out_path=args.out,
            ibd1_col=args.ibd1_col,
            ibd2_col=args.ibd2_col,
        )
        print(args.out)


if __name__ == "__main__":
    _cli()
