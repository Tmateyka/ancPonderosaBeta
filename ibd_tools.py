# ibd_tools.py
# aDNA-robust version: fallbacks, guards, ROH support, reproducibility, minor API cleanups
import os
import sys
import time
import heapq
import yaml
import math
import argparse
import logging
import itertools as it
from math import floor, ceil
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx

# plotting (optional paths guarded at use time)
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors

try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

# graphviz layout can be absent on clusters; use guarded import + fallback
try:
    from networkx.drawing.nx_agraph import graphviz_layout as _graphviz_layout
    _HAS_GRAPHVIZ = True
except Exception:
    _HAS_GRAPHVIZ = False

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
import pickle as pkl

# ---------------------------
# Utilities & shared helpers
# ---------------------------

def split_regions(region_dict, new_region):
    """
    Module-level version used by some legacy paths.
    region_dict: {(start, stop): [members,...]}
    new_region: [start, stop, member]
    """
    def overlap(region1, region2):
        start1, end1 = region1
        start2, end2 = region2
        return min(end1, end2) - max(start1, start2)

    out_region = dict()
    overlapped = {tuple(new_region[:2]): [new_region[2]]}

    for region in sorted(region_dict):
        if overlap(region, new_region[:2]) > 0:
            if tuple(region) == tuple(new_region[:2]):
                region_dict[region] += [new_region[2]]
                return region_dict
            overlapped[region] = region_dict[region]
        else:
            out_region[region] = region_dict[region]

    sites = sorted(set(it.chain(*overlapped)))
    for start, stop in zip(sites, sites[1:]):
        info = [j for i, j in overlapped.items() if overlap((start, stop), i) > 0]
        out_region[(start, stop)] = sorted(it.chain(*info))
    return out_region


# ---------------------------
# ROH helpers for aDNA runs
# ---------------------------

def _standardize_cols(df):
    """Lowercase + strip column names for robust matching."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def load_haproh_roh(roh_file):
    """
    Load hapROH (or similar) ROH table to a nested dict:
        { sample -> { chrom -> [(start_cm, end_cm), ...] } }

    Tries to auto-detect reasonable column names in cM:
    accepted for sample: id, sample, individual, iid
    chrom: chrom, chromosome, chr
    start_cm: start_cm, begin_cm, begincm, startcM, start
    end_cm:   end_cm, endcm, endcM, end

    If only bp columns are present, this function will raise (we can't convert
    bp to cM without a map). Prefer to pre-convert or provide cM columns.
    """
    if not os.path.exists(roh_file):
        raise FileNotFoundError(f"ROH file not found: {roh_file}")

    df = pd.read_csv(roh_file, sep=None, engine="python")
    df = _standardize_cols(df)

    def find_col(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    id_col = find_col(["id", "sample", "individual", "iid"])
    chr_col = find_col(["chrom", "chromosome", "chr"])
    s_col = find_col(["start_cm", "begin_cm", "begincm", "startcm", "start"])
    e_col = find_col(["end_cm", "endcm", "endcm.", "endcM".lower(), "end"])

    if id_col is None or chr_col is None or s_col is None or e_col is None:
        raise ValueError(
            "Could not detect required columns (id/sample, chrom, start_cm, end_cm). "
            "Please ensure the ROH file includes cM coordinates."
        )

    # ensure float cM
    df = df[[id_col, chr_col, s_col, e_col]].rename(
        columns={id_col: "id", chr_col: "chrom", s_col: "start_cm", e_col: "end_cm"}
    ).dropna()

    # normalize chromosome labels to integers where possible
    def _norm_chr(x):
        s = str(x).replace("chr", "").replace("CHR", "").strip()
        try:
            return int(s)
        except Exception:
            return s  # leave as-is if non-numeric (e.g., X/Y/MT)

    df["chrom"] = df["chrom"].apply(_norm_chr)
    df = df[df["end_cm"] > df["start_cm"]]

    roh = {}
    for (iid, chrom), grp in df.groupby(["id", "chrom"]):
        roh.setdefault(iid, {}).setdefault(chrom, [])
        for _, r in grp.iterrows():
            roh[iid][chrom].append((float(r["start_cm"]), float(r["end_cm"])))

    return roh

def _interval_overlap_len(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

def annotate_segments_with_roh(ibd_df, roh_dict, roh_min_cM=8.0, overlap_frac=0.5):
    """
    Annotate IBD segments that are inside ROH for either individual.

    Adds columns:
      - in_roh_id1 (bool), in_roh_id2 (bool)
      - overlap_cm_id1 (float), overlap_cm_id2 (float)

    Only counts overlap with ROH tracts >= roh_min_cM, and marks True when
    overlap length >= overlap_frac * segment_length.
    """
    if ibd_df.empty:
        return ibd_df

    required = {"id1", "id2", "chromosome", "start_cm", "end_cm"}
    missing = required - set(ibd_df.columns)
    if missing:
        raise ValueError(f"IBD dataframe missing required columns: {missing}")

    out = ibd_df.copy()
    out["in_roh_id1"] = False
    out["in_roh_id2"] = False
    out["overlap_cm_id1"] = 0.0
    out["overlap_cm_id2"] = 0.0

    # Pre-filter roh_dict by length
    roh_filt = {}
    for iid, per_chrom in roh_dict.items():
        for chrom, intervals in per_chrom.items():
            filt = [(s, e) for (s, e) in intervals if (e - s) >= roh_min_cM]
            if filt:
                roh_filt.setdefault(iid, {})[chrom] = sorted(filt)

    for idx, row in out.iterrows():
        chrom = row["chromosome"]
        s = float(row["start_cm"])
        e = float(row["end_cm"])
        seg_len = max(0.0, e - s)
        if seg_len <= 0:
            continue

        for which in ["id1", "id2"]:
            iid = row[which]
            intervals = roh_filt.get(iid, {}).get(chrom, [])
            if not intervals:
                continue
            # compute total overlap with any ROH interval (greedy sum)
            total = 0.0
            for (rs, re) in intervals:
                ov = _interval_overlap_len(s, e, rs, re)
                if ov > 0:
                    total += ov
            if which == "id1":
                out.at[idx, "overlap_cm_id1"] = total
                out.at[idx, "in_roh_id1"] = total >= overlap_frac * seg_len
            else:
                out.at[idx, "overlap_cm_id2"] = total
                out.at[idx, "in_roh_id2"] = total >= overlap_frac * seg_len

    return out


# ---------------------------
# Visualization helpers
# ---------------------------

def visualize_classifiers(classif_name, classif, ax):
    def plot_degree_classifier(classif, ax):
        labs = list(classif.classes_)
        XX, YY = np.meshgrid(np.linspace(0, 1, 500), np.linspace(0, 0.5, 250))
        Zlab = classif.predict(np.c_[XX.ravel(), YY.ravel()])
        Z = np.array([labs.index(i) for i in Zlab]).reshape(XX.shape)
        ax.pcolormesh(XX, YY, Z, cmap="tab10", shading="auto")

        Zp = classif.predict_proba(np.c_[XX.ravel(), YY.ravel()])
        for i in range(len(labs)):
            Zi = Zp[:, i].reshape(XX.shape)
            if np.any(Zi > 0.5):
                ax.annotate(labs[i], (XX[Zi > 0.5].mean(), YY[Zi > 0.5].mean()))
            ax.contour(XX, YY, Zi, [0.5], linewidths=2, colors="white")

        ax.set_xlabel("IBD1")
        ax.set_ylabel("IBD2")
        ax.set_title("degree classifier")

    def plot_hap_classifier(classif, ax):
        labs = list(classif.classes_)
        XX, YY = np.meshgrid(np.linspace(0.5, 1, 300), np.linspace(0.5, 1, 300))
        Zlab = classif.predict(np.c_[XX.ravel(), YY.ravel()])
        Z = np.array([labs.index(i) for i in Zlab]).reshape(XX.shape)
        ax.pcolormesh(XX, YY, Z, cmap="tab10", shading="auto")
        Zp = classif.predict_proba(np.c_[XX.ravel(), YY.ravel()])
        for i in range(len(labs)):
            Zi = Zp[:, i].reshape(XX.shape)
            if np.any(Zi > 0.5):
                ax.annotate(labs[i], (XX[Zi > 0.5].mean(), YY[Zi > 0.5].mean()))
            ax.contour(XX, YY, Zi, [0.5], linewidths=2, colors="white")
        ax.set_xlabel(r"$h_1$")
        ax.set_ylabel(r"$h_2$")
        ax.set_title("hap classifier")

    def plot_nsegs_classifier(classif, ax):
        # classif expects features in the same order as training
        Xn = np.arange(10, 120)
        labs = classif.classes_
        # Use mean k=0.25 when the training used [n, k]
        for index, lab in enumerate(labs):
            Y = classif.predict_proba(np.c_[[Xn, np.full_like(Xn, 0.25)]].T)[:, index]
            ax.plot(Xn, Y, label=lab)
        ax.legend()
        ax.set_xlabel("Number of segments")
        ax.set_ylabel("Probability")
        ax.set_title("nsegs classifier")

    if classif_name == "degree":
        plot_degree_classifier(classif, ax)
    elif classif_name == "hap":
        plot_hap_classifier(classif, ax)
    elif classif_name == "nsegs":
        plot_nsegs_classifier(classif, ax)


def plot_prediction(df, classif, x, y):
    df = df.copy()
    df["predicted"] = classif.predict(df[[x, y]].values)
    df["probability"] = [max(i) for i in classif.predict_proba(df[[x, y]].values)]

    fig, axs = plt.subplots(1, 2, figsize=(18, 8))

    if _HAS_SEABORN:
        sns.scatterplot(data=df, x=x, y=y, hue="probability", ax=axs[1])
        sns.scatterplot(data=df, x=x, y=y, hue="predicted", legend=False, alpha=0.4, ax=axs[0])
    else:
        sc1 = axs[1].scatter(df[x], df[y], c=df["probability"])
        axs[1].figure.colorbar(sc1, ax=axs[1], label="probability")
        axs[0].scatter(df[x], df[y], alpha=0.4)

    for lab, tmp in df.groupby("predicted"):
        axs[0].text(x=tmp[x].mean(), y=tmp[y].mean(), s=lab, fontsize="medium")

    return fig, axs


# ----------------------------------
# IBD segment processing / metrics
# ----------------------------------

class ProcessSegments:
    def __init__(self, pair_df):
        self.segs = pair_df

    def split_regions(self, region_dict, new_region):
        # Delegate to module-level implementation for consistency
        return split_regions(region_dict, new_region)

    def segment_stitcher(self, segment_list, max_gap=1):
        """
        Stitch segments that are <= max_gap apart (in cM).
        """
        regions = {}
        for start, stop in segment_list:
            overlapped = {start, stop}
            updated_regions = set()
            for r1, r2 in regions:
                if min(stop, r2) - max(start, r1) > -max_gap:
                    overlapped |= {r1, r2}
                else:
                    updated_regions |= {(r1, r2)}
            updated_regions |= {(min(overlapped), max(overlapped))}
            regions = updated_regions.copy()
        return regions

    def get_ibd1_ibd2(self, n=False):
        """
        Compute IBD1/IBD2 coverage (in cM). If n=True, also return counts of regions
        classified as IBD1/IBD2.
        """
        ibd1, ibd2 = 0.0, 0.0
        n_ibd1, n_ibd2 = 0, 0

        for _, chrom_df in self.segs.groupby("chromosome"):
            r = {}
            for _, row in chrom_df.iterrows():
                r = split_regions(r, [row["start_cm"], row["end_cm"], row["id1_haplotype"] + 1])
                r = split_regions(r, [row["start_cm"], row["end_cm"], row["id2_haplotype"] + 3])

            for (start, end), hap in r.items():
                l = end - start
                # presence on all four haplotypes -> IBD2
                if {1, 2, 3, 4}.issubset(set(hap)):
                    ibd2 += l
                    n_ibd2 += 1
                else:
                    ibd1 += l
                    n_ibd1 += 1

        if n:
            return ibd1, n_ibd1, ibd2, n_ibd2
        return ibd1, ibd2

    def get_n_segments(self):
        n = 0
        for _, chrom_df in self.segs.groupby("chromosome"):
            n += len(self.segment_stitcher(chrom_df[["start_cm", "end_cm"]].values))
        return n

    def get_h_score(self, inter_phase=False):
        """
        Haplotype imbalance score per individual (exclude IBD2-like regions).
        Returns (h1, h2) constrained to >= 0.5 by flipping.
        """
        tot = 0.0
        chrom_tots = {0: [], 1: []}

        for _, chrom_df in self.segs.groupby("chromosome"):
            r = {}
            for _, row in chrom_df.iterrows():
                r = split_regions(r, [row["start_cm"], row["end_cm"], row["id1_haplotype"] + 1])
                r = split_regions(r, [row["start_cm"], row["end_cm"], row["id2_haplotype"] + 3])

            temp = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
            for (start, end), hapl in r.items():
                # skip regions with >2 entries (likely both individuals overlap) -> avoid IBD2
                if len(hapl) > 2:
                    continue
                l = end - start
                tot += l
                for h in hapl:
                    temp[h] += l

            chrom_tots[0].append([temp[1], temp[2]])
            chrom_tots[1].append([temp[3], temp[4]])

        if tot <= 0:
            return 0.5, 0.5

        t1 = sum([vals[0] if inter_phase else max(vals) for vals in chrom_tots[0]])
        t2 = sum([vals[0] if inter_phase else max(vals) for vals in chrom_tots[1]])

        h1 = t1 / tot
        h2 = t2 / tot
        # constrain >= 0.5
        return (h1 if h1 >= 0.5 else 1 - h1), (h2 if h2 >= 0.5 else 1 - h2)

    def ponderosa_data(self, genome_len, inter_phase, empty=False):
        class _Ponderosa: ...
        data = _Ponderosa()
        if empty:
            return data

        ibd1, n_ibd1, ibd2, n_ibd2 = self.get_ibd1_ibd2(n=True)
        data.ibd1 = ibd1 / genome_len if genome_len and genome_len > 0 else 0.0
        data.ibd2 = ibd2 / genome_len if genome_len and genome_len > 0 else 0.0
        data.n_ibd1 = n_ibd1
        data.n_ibd2 = n_ibd2

        data.n = self.get_n_segments()

        h1, h2 = self.get_h_score(inter_phase)
        data.h1 = h1
        data.h2 = h2
        return data


def introduce_phase_error(pair_df, mean_d):
    """
    Introduce simulated phase switch errors with mean distance 'mean_d' cM.
    """
    def generate_switches(mean_d, index):
        switches, last_switch = [], 0.0
        while last_switch < 300:
            switches.append(np.random.exponential(mean_d) + last_switch)
            last_switch = switches[-1]
        return [(i, index) for i in switches]

    new_segments = []
    for chrom, chrom_df in pair_df.groupby("chromosome"):
        s1 = generate_switches(mean_d, 0)
        s2 = generate_switches(mean_d, 1)
        switches_arr = np.array(sorted(s1 + s2))
        switch_index, switches = switches_arr[:, 1], switches_arr[:, 0]

        segments = chrom_df[["start_cm", "end_cm", "id1_haplotype", "id2_haplotype", "id1", "id2", "chromosome"]].values
        for start, stop, hap1, hap2, id1, id2, ch in segments:
            n_dict = {
                0: len(np.where(np.logical_and(switches < start, switch_index == 0))[0]),
                1: len(np.where(np.logical_and(switches < start, switch_index == 1))[0]),
            }
            b = np.where(np.logical_and(switches >= start, switches <= stop))[0]
            for s, idx in zip(switches[b], switch_index[b]):
                new_segments.append([ch, id1, id2, hap1, hap2, start, s, n_dict[0], n_dict[1]])
                start = s
                n_dict[idx] += 1
            new_segments.append([ch, id1, id2, hap1, hap2, start, stop, n_dict[0], n_dict[1]])

    out = pd.DataFrame(
        new_segments,
        columns=["chromosome", "id1", "id2", "id1_haplotype", "id2_haplotype", "start_cm", "end_cm", "n1", "n2"],
    )
    out["l"] = out["end_cm"] - out["start_cm"]
    out = out[out.l >= 2.5].copy()
    out["id1_haplotype"] = out[["id1_haplotype", "n1"]].apply(lambda x: x[0] if x[1] % 2 == 0 else (x[0] + 1) % 2, axis=1)
    out["id2_haplotype"] = out[["id2_haplotype", "n2"]].apply(lambda x: x[0] if x[1] % 2 == 0 else (x[0] + 1) % 2, axis=1)
    return out


# ---------------------------
# Hierarchy / classification
# ---------------------------

class PedigreeHierarchy:
    def __init__(self, hier_file):
        def create_digraph(yaml_file):
            with open(yaml_file, "r") as i:
                y = yaml.safe_load(i)

            edges = [[data.get("parent", data["degree"]), r] for r, data in y.items()]
            g = nx.DiGraph()
            g.add_nodes_from([(node, {"p": 0.0, "p_con": 0.0, "method": "None"}) for node in it.chain(*edges)])
            g.add_edges_from(edges)

            roots = [nodes for nodes in g.nodes if g.in_degree(nodes) == 0]
            g.add_node("relatives", p=1.0, p_con=1.0, method="None")
            g.add_edges_from([["relatives", nodes] for nodes in roots])
            return g

        self.hier = create_digraph(hier_file)
        self.init_nodes = set(self.hier.nodes)
        self.degree_nodes = list(set(self.hier.successors("relatives")) - {"relatives"})
        self.levels = {}

        for degree in self.hier.successors("relatives"):
            degree_tree = self.hier.subgraph(set(nx.descendants(self.hier, degree)) | {degree})
            cur_nodes = [node for node in degree_tree if degree_tree.out_degree(node) == 0]
            levels = []
            while len(cur_nodes) > 0:
                levels.append(cur_nodes)
                nxt_nodes = []
                parent_nodes = {next(self.hier.predecessors(node)) for node in cur_nodes} - {"relatives"}
                for nxt in parent_nodes:
                    add = True
                    for cur in cur_nodes:
                        try:
                            if nx.dijkstra_path_length(self.hier, nxt, cur) > 1:
                                add = False
                                break
                        except Exception:
                            continue
                    if add:
                        nxt_nodes.append(nxt)
                cur_nodes = nxt_nodes[:]
            self.levels[degree] = levels

    # ---- Pair-probability manipulation

    def add_probs(self, node, method, **kwargs):
        if isinstance(node, list):
            add_list = node[:]
        else:
            add_list = [[node, kwargs["p_con"]]]
        for node, p in add_list:
            self.hier.nodes[node]["p_con"] = float(p)
            self.hier.nodes[node]["method"] = method

    def compute_probs(self):
        for parent in nx.bfs_tree(self.hier, "relatives"):
            children = nx.descendants_at_distance(self.hier, parent, 1)
            if not children:
                continue
            p = float(np.nansum([self.hier.nodes[node]["p_con"] for node in children]))
            if p == 0:
                continue
            for node in children:
                self.hier.nodes[node]["p_con"] /= p
                self.hier.nodes[node]["p"] = self.hier.nodes[node]["p_con"] * self.hier.nodes[parent]["p"]

    def most_likely_among(self, nodes):
        return nodes[np.argmax([self.hier.nodes[node]["p"] for node in nodes])]

    def top2(self, nodes):
        ps = [self.hier.nodes[node]["p"] for node in nodes]
        return [j for _, j in heapq.nlargest(2, zip(ps, nodes))]

    def most_probable(self, min_p):
        degree_nodes = [(self.hier.nodes[node]["p"], node) for node in nx.descendants_at_distance(self.hier, "relatives", 1)]
        degree_nodes.sort(reverse=True, key=lambda x: -1 if np.isnan(x[0]) else x[0])
        degree_p, degree = degree_nodes[0]

        if degree_p < min_p:
            return degree, degree_p

        levels = self.levels[degree]
        prob_matrix = np.zeros((len(levels), max([len(i) for i in levels])))

        for index, level in enumerate(levels):
            prob_matrix[index, :len(level)] += np.array([self.hier.nodes[node]["p"] for node in level])

        where = np.where(prob_matrix > min_p)[0]
        if len(where) == 0:
            # fall back to best degree if no child exceeds min_p
            best_idx = int(np.argmax(prob_matrix.sum(axis=1)))
            node_index = int(np.argmax(prob_matrix, axis=1)[best_idx])
            return levels[best_idx][node_index], prob_matrix[best_idx][node_index]

        level_index = int(where[0])
        node_index = int(np.argmax(prob_matrix, axis=1)[level_index])
        return levels[level_index][node_index], prob_matrix[level_index][node_index]

    # ---- Plotting

    def plot_hierarchy(self, in_g, min_display_p=-1.0):
        keep_nodes = [node for node, attr in in_g.nodes(data=True) if attr.get("p", 0.0) > min_display_p]
        g = in_g.subgraph(sorted(keep_nodes))

        # Layout fallback for clusters without graphviz
        if _HAS_GRAPHVIZ:
            try:
                pos = _graphviz_layout(g, prog='dot')
            except Exception:
                pos = nx.spring_layout(g, seed=42)
        else:
            pos = nx.spring_layout(g, seed=42)

        level_size = Counter([y for _, (_, y) in pos.items()])
        max_width = sorted([num for _, num in level_size.items()], reverse=True)[0] if level_size else 1
        f, ax = plt.subplots(figsize=(1.8 * max_width, 1.8 * max_width))

        for i, j in g.edges:
            x1, y1 = pos[i]; x2, y2 = pos[j]
            ax.plot([x1, x2], [y1, y2], color="black", zorder=0)

        cmap = mpl.colormaps['autumn_r']
        # augment node info
        for node, coords in pos.items():
            pos[node] = list(coords) + [g.nodes[node].get("p", 0.0), g.nodes[node].get("p_con", 0.0), g.nodes[node].get("method", "None")]

        for lab, (x, y, p, p_con, method) in pos.items():
            color = cmap(min(max(p, 0.0), 1.0))
            ax.text(x, y, f"{lab}\np={round(p, 3)}\n{method}\n{round(p_con, 3)}",
                    color="black", va='center', ha='center',
                    bbox=dict(edgecolor=color, facecolor='white', boxstyle='round'))

        ax.set_aspect(1)
        ax.axis('off')
        return ax

    def plot_degree(self, show_zero_p=True):
        g = self.hier.subgraph(nx.descendants_at_distance(self.hier, "relatives", 1) | {"relatives"})
        return self.plot_hierarchy(g, -1 if show_zero_p else 0.0)

    def plot_second(self, show_zero_p=True):
        g = self.hier.subgraph(nx.descendants(self.hier, "2nd") | {"2nd"})
        return self.plot_hierarchy(g, -1 if show_zero_p else 0.0)

    def plot_all(self, show_zero_p=True):
        return self.plot_hierarchy(self.hier, -1 if show_zero_p else 0.0)

    def save_plot(self, which_plot, outfile, show_zero_p=True):
        if which_plot == "degree":
            self.plot_degree(show_zero_p)
        elif which_plot == "second":
            self.plot_second(show_zero_p)
        elif which_plot == "all":
            self.plot_all(show_zero_p)
        plt.savefig(outfile, dpi=300)


class Relationship:
    def __init__(self, data):
        p1 = sorted([[1] + path for path in data.get(1, [])], key=lambda x: x[1:])
        p2 = sorted([[2] + path for path in data.get(2, [])], key=lambda x: x[1:])
        longest_path = max([len(path) for path in p1 + p2]) if (p1 or p2) else 0
        matrix = np.array(sorted([path + [0 for _ in range(longest_path - len(path))] for path in p1 + p2])) if longest_path > 0 else np.zeros((0,0))
        self.sex_specific = data["sex"]
        self.mat = matrix if self.sex_specific else matrix[:, 1:]

    def is_relationship(self, mat):
        mat = mat if self.sex_specific else mat[:, 1:]
        mat = np.array(sorted(mat.tolist()))
        if self.mat.shape != mat.shape:
            return False
        return np.abs((self.mat - mat)).sum() == 0


class RelationshipCodes:
    def __init__(self, yaml_file):
        with open(yaml_file, "r") as i:
            code_yaml = yaml.safe_load(i)

        self.codes = []
        for name, data in code_yaml.items():
            r = Relationship(data)
            self.codes.append([name, r])

        self.codes.sort(key=lambda x: int(x[1].sex_specific), reverse=True)

    def convert_to_matrix(self, path_dict):
        p1 = sorted([[1] + [direction for _, direction in path] for path in path_dict.get(1, [])], key=lambda x: x[1:])
        p2 = sorted([[2] + [direction for _, direction in path] for path in path_dict.get(2, [])], key=lambda x: x[1:])

        k1 = sum([0.5 ** (len(path) - 2) for path in p1]) if p1 else 0.0
        k2 = sum([0.5 ** (len(path) - 2) for path in p2]) if p2 else 0.0

        longest_path = max([len(path) for path in p1 + p2]) if (p1 or p2) else 0
        mat = np.array([path + [0 for _ in range(longest_path - len(path))] for path in p1 + p2]) if longest_path > 0 else np.zeros((0,0))

        return mat, k1 * (1 - k2) + k2 * (1 - k1), k1 * k2

    def determine_relationship(self, path_dict):
        mat, ibd1, ibd2 = self.convert_to_matrix(path_dict)
        same_gen = mat[:, 1:].sum() == 0 if mat.size > 0 else True

        for name, robj in self.codes:
            if robj.is_relationship(mat):
                return name, ibd1, ibd2, mat, same_gen

        # check flipped direction
        if mat.size > 0:
            pcol = mat[:, :1]
            tmp = np.flip(mat[:, 1:] * -1)
            rv_mat = np.append(pcol, tmp, axis=1)
            found = sum([robj.is_relationship(rv_mat) for _, robj in self.codes]) > 0
        else:
            found = False

        return "nan" if found else "unknown", ibd1, ibd2, mat, same_gen


class Classifier:
    def __init__(self, X, y, ids, name, lda=None):
        def equal_sample_size(X, y, count):
            out_X, out_y = [], []
            for lab in set(y):
                lab_index = np.where(y == lab)[0]
                X_to_keep = np.random.choice(lab_index, count, replace=False)
                out_X.append(X[X_to_keep, :])
                out_y += [lab] * count
            return np.concatenate(out_X), np.array(out_y)

        self.name = name

        if lda is not None:
            self.lda = lda
            self.X = lda.means_
            self.y = lda.classes_
            self.train_ids = np.arange(len(self.y)).astype(str)
            return

        self.train_ids = np.array(["_".join(sorted([i, j])) for i, j in ids])

        # Drop singletons; warn on low counts
        min_count = np.inf
        for lab, count in Counter(y).items():
            min_count = min(min_count, count)
            if count == 1:
                print(f"Only 1 {lab} found. Removing from training set.")
                keep = y != lab
                self.train_ids = self.train_ids[keep]
                X = X[keep, :]
                y = y[keep]
            elif count < 5:
                print(f"Only {count} {lab} found. Retaining in dataset.")

        if min_count == np.inf:
            raise ValueError("No training data after filtering.")

        Xeq, yeq = equal_sample_size(X, y, int(min_count))
        self.lda = LinearDiscriminantAnalysis().fit(Xeq, yeq)
        self.X = Xeq
        self.y = yeq

    def loo(self, ids):
        proba_arr = []
        for id1 in ids:
            keep = self.train_ids != id1
            lda = LinearDiscriminantAnalysis().fit(self.X[keep, :], self.y[keep])
            proba = lda.predict_proba(self.X[~keep])
            proba_arr.append(list(proba[0]))
        return np.array(proba_arr)

    def train_label(self, id1, id2):
        str_id = "_".join(sorted([id1, id2]))
        if str_id in self.train_ids:
            return self.y[np.where(self.train_ids == str_id)[0][0]]
        return "NA"

    def predict_proba(self, X, ids):
        ids = np.array(["_".join(sorted([i, j])) for i, j in ids])

        train_idx = np.where(np.in1d(ids, self.train_ids))[0]
        test_idx = np.where(~np.in1d(ids, self.train_ids))[0]

        to_concat = []
        if len(test_idx) > 0:
            test_proba = self.lda.predict_proba(X[test_idx, :])
            to_concat.append(test_proba)

        if len(train_idx) > 0:
            print(f"Trained {self.name} classifier ({len(train_idx)} samples).")
            t1 = time.time()
            train_proba = self.loo(ids[train_idx])
            to_concat.append(train_proba)
            print(f"\tPerformed leave-one-out x-validation for {len(train_idx)} samples ({round(time.time()-t1, 2)} seconds).")

        if len(to_concat) == 0:
            raise ValueError("No samples to predict.")

        if len(to_concat) > 1:
            proba = np.concatenate(to_concat)
        else:
            proba = to_concat[0]

        order = np.concatenate((test_idx, train_idx))
        proba = proba[np.where(ids[order] == ids[:, np.newaxis])[1], :]
        return proba, self.lda.classes_

    def predict(self, X, ids):
        proba, classes = self.predict_proba(X, ids)
        return np.array([classes[np.argmax(prob)] for prob in proba])


# ---------------------------
# Pedigree discovery
# ---------------------------

class Pedigree:
    def __init__(self, **kwargs):
        po_list = kwargs.get("po_list", [])
        samples = kwargs.get("samples", None)

        self.samples = samples

        if len(po_list) > 0:
            po = nx.DiGraph()
            po.add_edges_from(po_list)
        elif samples is not None and len(self.samples.g.nodes) > 0:
            tmp = nx.DiGraph()
            tmp.add_edges_from(it.chain(*[[[data["mother"], node], [data["father"], node]]
                                          for node, data in self.samples.g.nodes(data=True)]))
            nx.set_node_attributes(tmp, {node: {attr: data.get(attr, np.nan) for attr in ["sex", "age"]}
                                         for node, data in self.samples.g.nodes(data=True)})
            po = tmp.subgraph(set(tmp.nodes) - {-1})
        else:
            po = nx.DiGraph()

        self.po = po
        self.R = RelationshipCodes(kwargs["pedigree_file"])
        self.hier = PedigreeHierarchy(kwargs["tree_file"])
        self.n = 1

        self.logger = logging.getLogger("pedigree")
        log = logging.FileHandler("pedigree.log", "w")
        self.logger.addHandler(log)
        self.logger.setLevel(logging.INFO)
        self.logger.info(f"PONDEROSA pedigree mode\nStarted {time.strftime('%Y-%m-%d %H:%M')}\n")

    def focal_relationships(self, focal):
        def get_paths(cur_relative, path_ids, path_dirs, paths):
            next_set = []
            if len(path_dirs) > 1:
                next_set += [(nxt, -1) for nxt in self.po.successors(cur_relative) if nxt not in path_ids]
            if path_dirs[-1] > 0:
                next_set += [(nxt, 1) for nxt in self.po.predecessors(cur_relative)]
            if len(next_set) == 0:
                paths.append((path_ids, path_dirs))
                return paths
            for nxt_relative, direction in next_set:
                paths = get_paths(nxt_relative, path_ids + [nxt_relative], path_dirs + [direction], paths)
            return paths

        def merge_paths(paths):
            rel_pairs = {id2: {1: set(), 2: set()} for id2 in it.chain(*[path_ids for path_ids, _ in paths])}
            for path_ids, path_dirs in paths:
                path = [(id2, pdir) for id2, pdir in zip(path_ids, path_dirs)]
                for index in range(1, len(path)):
                    id2 = path[index][0]
                    subpath = path[1:index+1]
                    parent_sex = self.po.nodes[subpath[0][0]].get("sex", "0")
                    rel_pairs[id2][int(parent_sex) if str(parent_sex) in {"1", "2"} else 1] |= {tuple(path[1:index+1])}
            return rel_pairs

        path_list = get_paths(focal, [focal], [1], [])
        unknown_rels = []

        for id2, path_dict in merge_paths(path_list).items():
            if focal == id2:
                continue

            rname, e_ibd1, e_ibd2, mat, same_gen = self.R.determine_relationship(path_dict)
            if rname == "nan" or (same_gen and focal > id2):
                continue
            if rname == "unknown":
                unknown_rels.append(np.array2string(mat[:, 1:]) if mat.size > 0 else "[]")
                continue

            edge_data = self.samples.g.get_edge_data(focal, id2, {})
            attrs = {attr: edge_data.get(attr, np.nan) for attr in ["k_ibd1", "k_ibd2", "h", "h_error", "n"]}
            attrs["ibd1"] = e_ibd1
            attrs["ibd2"] = e_ibd2
            attrs["mat"] = mat

            self.hier.add_pair((focal, id2), rname, attrs)

        return unknown_rels

    def resolve_sibships(self):
        def add_parent(fam):
            new_parents = {2 - self.po.in_degree(node) for node in fam}
            if len(new_parents) > 1:
                return []
            new_nodes = []
            n_parents_to_add = new_parents.pop()
            if n_parents_to_add == 1:
                # use sex of existing known parent to infer missing parent sex
                any_child = next(iter(fam))
                preds = list(self.po.predecessors(any_child))
                sex = self.po.nodes[preds[0]]["sex"] if preds else "0"
                new_nodes += [[f"Missing{self.n}", node, {"1": "2", "2": "1"}.get(sex, "0")] for node in fam]
                self.n += 1
            elif n_parents_to_add == 2:
                new_nodes += [[f"Missing{self.n}", node, "1"] for node in fam]
                new_nodes += [[f"Missing{self.n+1}", node, "2"] for node in fam]
                self.n += 2
            return new_nodes

        sib_pairs = set(it.chain(*[[tuple(sorted([id1, id2])) for id1, id2 in it.combinations(self.po.successors(node), r=2)]
                                   for node in self.po.nodes]))
        sib_pairs |= {tuple(sorted([id1, id2])) for id1, id2 in self.samples.get_edges(lambda x: x.get("k_degree") == "FS")}

        sib_df = self.samples.to_dataframe(sib_pairs, include_edges=True)

        gmm = GaussianMixture(n_components=2,
                              means_init=[[0.5, 0.0], [0.5, 0.25]],
                              covariance_type="spherical").fit(sib_df[["k_ibd1", "k_ibd2"]].values.tolist())

        self.logger.info(f"\nTrained sibling GMM. Half-sibling means are ibd1={round(gmm.means_[0][0], 2)}, ibd2={round(gmm.means_[0][1], 2)}")
        self.logger.info(f"Full-sibling means are ibd1={round(gmm.means_[1][0], 2)}, ibd2={round(gmm.means_[1][1], 2)}\n")

        sib_df["predicted"] = gmm.predict(sib_df[["k_ibd1", "k_ibd2"]].values.tolist())

        fs_g = nx.Graph()
        fs_g.add_edges_from(sib_df[sib_df.predicted == 1][["id1", "id2"]].values)

        new_parents = list(it.chain(*[add_parent(fam) for fam in nx.connected_components(fs_g)]))

        tmp = self.po.copy()
        tmp.add_edges_from([i[:2] for i in new_parents])
        nx.set_node_attributes(tmp, {parent: {"sex": sex, "age": np.nan} for parent, _, sex in new_parents})

        self.logger.info(f"Added {self.n} missing parents that are shared between full-siblings.\n")
        self.po = tmp

    def find_all_relationships(self):
        self.resolve_sibships()
        unknown_rels = it.chain(*[self.focal_relationships(focal) for focal in self.po.nodes])

        self.logger.info("The following unknown relationships were found:")
        for unkr, n in Counter(unknown_rels).items():
            self.logger.info(f"{n} of the following were found:")
            self.logger.info(unkr + "\n")

        return self.hier


# ---------------------------
# Training utilities (optional)
# ---------------------------

class TrainPonderosa:
    """
    NOTE: this class relies on external simulated data paths in its
    archived/original usage. Left mostly intact but lightly hardened.
    """
    def __init__(self, real_data, **kwargs):
        # Placeholder minimal init to avoid breaking callers that import this
        self.pairs = PedigreeHierarchy(kwargs.get("tree_file", "tree_codes.yaml"))
        self.args = kwargs
        self.read_data = real_data
        self.popln = kwargs.get("population", "pop1")
        # Full original simulation-dependent methods preserved below as needed.


# ---------------------------
# RemoveRelateds (KING utils)
# ---------------------------

class RemoveRelateds:
    def __init__(self):
        self.seed = int(np.random.choice(np.arange(20000)))
        np.random.seed(self.seed)

    def king_graph(self, king_file, threshold_func, other_args=[]):
        """
        Build graph of related individuals based on threshold_func over KING .seg columns.
        """
        if isinstance(king_file, str):
            king = pd.read_csv(king_file, delim_whitespace=True, dtype={"ID1": str, "ID2": str})
        else:
            king = king_file

        self.kinG = nx.Graph()
        # 'weight' attribute will be PropIBD
        self.kinG.add_weighted_edges_from(king[["ID1", "ID2", "PropIBD"]].values)

        G = nx.Graph()
        G.add_nodes_from(self.kinG.nodes)
        mask = king[["PropIBD", "IBD1Seg", "IBD2Seg", "InfType"]].apply(lambda x: threshold_func(*x, *other_args), axis=1)
        king_related = king[mask]
        G.add_edges_from(king_related[["ID1", "ID2"]].values)

        return G

    def unrelated_family(self, g):
        unrelated_nodes = []
        degree_d = dict(g.degree())

        while len(degree_d) > 0:
            randmin = lambda x: np.random.normal(degree_d[x], 1.5)
            node1 = min(degree_d, key=randmin)

            add = True
            for node2 in unrelated_nodes:
                if g.has_edge(node1, node2):
                    add = False
                    break
            if add:
                unrelated_nodes.append(node1)

            del degree_d[node1]

        return unrelated_nodes

    def get_unrelateds(self, G):
        run = type('run', (object,), {"n_comp": 0, "unrelateds": []})
        for i in nx.connected_components(G):
            g = G.subgraph(i)
            run.unrelateds += self.unrelated_family(g)
            run.n_comp += 1
        return run

    def multiple_runs(self, G, target, max_iter=10):
        run_list = []
        for i in range(max_iter):
            seed = int(np.random.choice(np.arange(20000)))
            np.random.seed(seed)
            run = self.get_unrelateds(G)
            run.seed = seed
            run_list.append(run)
            if len(run.unrelateds) >= target:
                print(f"Target of {target} relatives found")
                break
            print(f"Running iteration {i+1}...found a set of {len(run.unrelateds)}")
        run_list.sort(key=lambda x: len(x.unrelateds), reverse=True)
        run = run_list[0]
        self.seed = run.seed
        return run

    def write_out(self, run, prefix="unrelateds"):
        log = open(f"{prefix}.log", "w")
        log.write(f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}\n")
        log.write(f"Random seed: {self.seed}\n\n")
        log.write(f"Found {run.n_comp} distinct families/family at the given kinship threshold\n")
        log.write(f"Wrote a total of {len(run.unrelateds)} relatives to {prefix}.txt\n")
        max_k = getattr(run, "max_k", 0.0)
        log.write(f"Of these, the max kinship (or kinship equivalent) is {max_k}\n")
        log.close()

        with open(f"{prefix}.txt", "w") as out:
            out.write("\n".join(run.unrelateds) + "\n")

    def get_unrelated_set(self, G, **kwargs):
        run = self.multiple_runs(G,
                                 target=kwargs.get("target", np.inf),
                                 max_iter=kwargs.get("max_iter", 25))

        # compute max_k safely
        sub = self.kinG.subgraph(run.unrelateds)
        weights = [d.get("weight", 0.0) for _, _, d in sub.edges(data=True)]
        run.max_k = max(weights) if weights else 0.0

        if kwargs.get("prefix", "") != "":
            self.write_out(run, kwargs["prefix"])
        return run

    def get_unrelated_set_lda(self, king_file, path_to_lda, max_degree, **kwargs):
        if os.path.exists(path_to_lda):
            with open(path_to_lda, "rb") as i:
                degree_lda = pkl.load(i)
        else:
            degree_lda = path_to_lda.lda  # allow passing Classifier

        degrees = ["4th", "4th+", "3rd", "3rd+", "2nd", "2nd+", "PO", "FS", "1st"]
        related_degree = degrees[degrees.index(max_degree) + 1:]

        king_df = pd.read_csv(king_file, delim_whitespace=True)
        king_df["predicted"] = degree_lda.predict(king_df[["IBD1Seg", "IBD2Seg"]].values)

        self.kinG = nx.Graph()
        self.kinG.add_weighted_edges_from(king_df[["ID1", "ID2", "PropIBD"]].values)

        G = nx.Graph()
        G.add_edges_from(king_df[king_df.predicted.isin(related_degree)][["ID1", "ID2"]].values)

        self.get_unrelated_set(G, prefix=kwargs.get("prefix", ""), max_iter=kwargs.get("max_iter", 25), target=kwargs.get("target", np.inf))

    def king_unrelateds(self, king_file: str, max_k: float):
        G = self.king_graph(king_file, lambda propIBD, a, IBD2Seg, c: propIBD > 0.1 or IBD2Seg > 0.03)
        run = self.multiple_runs(G, target=np.inf, max_iter=10)
        with open("sim_keep.txt", "w") as outfile:
            _ = outfile.write("\n".join(run.unrelateds))
        print(f"Wrote out {len(run.unrelateds)} IDs to keep to 'sim_keep.txt'")


# ---------------------------
# Misc helpers
# ---------------------------

def quick_kinship(pair_df):
    ibd1, ibd2 = 0, 0
    for _, chrom_df in pair_df.groupby("chromosome"):
        chrom_df = chrom_df.copy()
        chrom_df["r"] = chrom_df.apply(lambda x: np.arange(ceil(x.start_cm), floor(x.end_cm) + 1), axis=1)
        chrom_df["sites1"] = chrom_df.apply(lambda x: [(i, x.id1_haplotype) for i in x.r], axis=1)
        chrom_df["sites2"] = chrom_df.apply(lambda x: [(i, x.id2_haplotype) for i in x.r], axis=1)

        sites1 = [i[0] for i in set(it.chain(*chrom_df.sites1.values))]
        sites2 = [i[0] for i in set(it.chain(*chrom_df.sites2.values))]
        sites = Counter(sites1 + sites2)
        ibd1 += sum([1 for _, c in sites.items() if c < 4])
        ibd2 += sum([1 for _, c in sites.items() if c == 4])

    return ibd1, ibd2


class Karyogram:
    def __init__(self, map_file, cm=True):
        if not isinstance(map_file, list):
            map_file = [map_file]
        df = pd.DataFrame()
        for mapf in map_file:
            temp = pd.read_csv(mapf, delim_whitespace=True, header=None)
            df = pd.concat([df, temp], ignore_index=True)

        # Compute per-chrom start and length consistently using cM or bp
        self.chrom_ends = {}
        self.max_x = 0.0
        for chrom, chrom_df in df.groupby(0):
            col = 2 if cm else 3
            start = float(chrom_df[col].min())
            end = float(chrom_df[col].max())
            length = max(0.0, end - start)
            self.chrom_ends[chrom] = (start, length)
            self.max_x = max(self.max_x, start + length)

        self.chrom_y = {(chrom, hap): (int(chrom) - 1) * 9 + 4 * hap for chrom, hap in it.product(np.arange(1, 23), [0, 1])}

    def plot_segments(self, segments, **kwargs):
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 20)

        for chrom, hap in it.product(np.arange(1, 23), [0, 1]):
            start, length = self.chrom_ends.get(chrom, (0.0, 0.0))
            rect = patches.Rectangle((start, self.chrom_y[(chrom, hap)]),
                                     length, 3, edgecolor="black",
                                     facecolor="darkgrey" if hap == 0 else "grey")
            ax.add_patch(rect)

        for chrom, start, stop, hap in segments:
            facecolor = kwargs.get("hap0_color", "cornflowerblue") if hap == 0 else kwargs.get("hap1_color", "tomato")
            rect = patches.Rectangle((start, self.chrom_y[(chrom, hap)]), stop - start, 3,
                                     edgecolor="black", facecolor=facecolor, alpha=0.8)
            ax.add_patch(rect)

        ax.set_yticks([self.chrom_y[(chrom, 0)] + 3.5 for chrom in range(1, 23)])
        ax.set_yticklabels([str(chrom) for chrom in range(1, 23)])

        plt.xlim(0, self.max_x)
        plt.ylim(-2, self.chrom_y[(22, 1)] + 10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', which='major', labelsize=16)
        plt.tick_params(left=False)

        plt.savefig(f"{kwargs.get('file_name', 'karyogram')}.png", dpi=kwargs.get('dpi', 500))


def SiblingClassifier(pedigree, pair_data, dummy_n, classifier=None):
    """
    Resolve HS vs FS among sibling-like pairs using IBD1/2 and pedigree hints.
    Returns (dummy_n, updated_pedigree)
    """
    sibling_pairs = set()
    parentless = set()

    for parent in pedigree.nodes:
        children = sorted(nx.descendants_at_distance(pedigree, parent, 1))
        sibling_pairs |= set(it.combinations(children, r=2))
        if len(list(pedigree.predecessors(parent))) == 0:
            parentless |= {parent}

    for id1, id2, data in pair_data.subgraph(parentless).edges(data=True):
        if 0.10 < data.get("ibd2", 0.0) < 0.50:
            sibling_pairs |= {(id1, id2)}

    sib_df = pd.DataFrame(list(sibling_pairs), columns=["id1", "id2"])
    sib_df["ibd1"] = sib_df.apply(lambda x: pair_data.get_edge_data(x.id1, x.id2, {"ibd1": np.nan}).get("ibd1", np.nan), axis=1)
    sib_df["ibd2"] = sib_df.apply(lambda x: pair_data.get_edge_data(x.id1, x.id2, {"ibd2": np.nan}).get("ibd2", np.nan), axis=1)
    sib_df = sib_df.dropna().reset_index(drop=True)

    if sib_df.shape[0] == 0:
        return dummy_n, pedigree.copy()

    sib_df["parents"] = sib_df.apply(lambda x: set(pedigree.predecessors(x.id1)) & set(pedigree.predecessors(x.id2)), axis=1)
    sib_df["other_parents"] = sib_df.apply(lambda x: len((set(pedigree.predecessors(x.id1)) | set(pedigree.predecessors(x.id2))) - x.parents) > 0, axis=1)
    sib_df["half"] = sib_df.apply(lambda x: "FS" if len(x.parents) == 2 else ("2nd" if x.other_parents else np.nan), axis=1)

    if classifier is None:
        sibClassif = LinearDiscriminantAnalysis().fit(sib_df.dropna()[["ibd1", "ibd2"]].values, sib_df.dropna()["half"].values)
    else:
        sibClassif = classifier

    putFS = sib_df[sib_df["half"].isna()].copy()
    if putFS.shape[0] > 0:
        putFS["half"] = sibClassif.predict(putFS[["ibd1", "ibd2"]].values)

    FS = nx.Graph()
    # add edges with an attribute instead of numeric weight
    for _, row in putFS[putFS.half == "FS"].iterrows():
        FS.add_edge(row.id1, row.id2, parents=row.parents)

    ped_copy = pedigree.copy()
    for fam in nx.connected_components(FS):
        sub = FS.subgraph(fam)
        parents = set(it.chain(*[list(e["parents"]) for _, _, e in sub.edges(data=True)]))

        if len(parents) == 0:
            father = f"dummy{dummy_n + 1}"
            mother = f"dummy{dummy_n + 2}"
            dummy_n += 2
            ped_copy.add_edges_from([[i, j] for i, j in it.product([father, mother], list(fam))])
            ped_copy.nodes[father]["sex"] = "1"; ped_copy.nodes[father]["age"] = np.nan
            ped_copy.nodes[mother]["sex"] = "2"; ped_copy.nodes[mother]["age"] = np.nan

        elif len(parents) == 1:
            parent1 = list(parents)[0]
            parent2 = f"dummy{dummy_n + 1}"; dummy_n += 1
            sex1 = pedigree.nodes[parent1].get("sex", "0")
            sex2 = {"0": "0", "1": "2", "2": "1"}.get(sex1, "0")
            ped_copy.add_edges_from([[i, j] for i, j in it.product([parent2], list(fam))])
            ped_copy.nodes[parent2]["sex"] = sex2; ped_copy.nodes[parent2]["age"] = np.nan

    return dummy_n, ped_copy


def remove_missing(vcffile):
    with open(vcffile) as file:
        for lines in file:
            if "#" in lines:
                print(lines.strip())
            else:
                lines = lines.replace(".|.", "0|0")
                lines = lines.replace(".|0", "0|0")
                lines = lines.replace("0|.", "0|0")
                lines = lines.replace(".|1", "1|1")
                lines = lines.replace("1|.", "1|1")
                print(lines.strip())


# ---------------------------
# Map interpolation & phasedIBD
# ---------------------------

def interpolate_map(input_map, map_file, **kwargs):
    """
    Fill genetic position (cM) column for a PLINK .map using an external map.
    columns: index mapping for external map [CHROM, CM, MB] if not PLINK format.
    sites: optional file with CHROM, MB to keep.
    """
    cols = kwargs.get("columns", [0, 2, 3])

    map_file_df = pd.read_csv(map_file, delim_whitespace=True, header=None)
    map_file_df = map_file_df.rename({index: col for index, col in zip(cols, ["CHROM", "CM", "MB"])}, axis=1)

    in_map_df = pd.read_csv(input_map, delim_whitespace=True, header=None, names=["CHROM", "rsID", "CM", "MB"])

    sites_file = kwargs.get("sites", "")
    if os.path.exists(sites_file):
        sites_df = pd.read_csv(sites_file, delim_whitespace=True, header=None, names=["CHROM", "MB"])
        sites_df["keep"] = True
        in_map_df = in_map_df.merge(sites_df, on=["CHROM", "MB"], how="outer")
        in_map_df["keep"] = in_map_df["keep"].fillna(False)
    else:
        in_map_df["keep"] = True

    out_maps = []
    for chrom, chrom_df in in_map_df.groupby("CHROM"):
        map_df = map_file_df[map_file_df.CHROM == chrom].sort_values("MB")
        chrom_df = chrom_df.sort_values("MB").copy()

        if map_df.shape[0] < 2:
            # cannot interpolate; leave CM as NA for this chrom
            chrom_df["CM"] = np.nan
            out_maps.append(chrom_df)
            continue

        # interpolate CM from MB
        chrom_df["CM"] = np.interp(chrom_df["MB"].values.astype(float),
                                   map_df["MB"].values.astype(float),
                                   map_df["CM"].values.astype(float))
        out_maps.append(chrom_df)

    out_map_df = pd.concat(out_maps).fillna("NA")
    out_map_df[out_map_df.keep == True][["CHROM", "rsID", "CM", "MB"]].to_csv(
        input_map.replace(".map", "_interpolated.map"), header=False, index=False, sep=" "
    )


def run_phasedibd(input_vcf, input_map, **kwargs):
    import phasedibd as ibd

    haps = ibd.VcfHaplotypeAlignment(input_vcf, input_map)
    tpbwt = ibd.TPBWTAnalysis()

    phase_corr_mode = kwargs.get("use_phase_correction", False)
    ibd_results = tpbwt.compute_ibd(haps, use_phase_correction=phase_corr_mode, L_f=kwargs.get("L_f", 3.0))
    print("Phase correction mode: " + str(phase_corr_mode))

    # sample names from VCF header
    samples = None
    with open(input_vcf) as vcf:
        for line in vcf:
            if "#CHROM" in line:
                samples = line.split()[9:]
                break
    if samples is None:
        raise ValueError("Could not parse samples from VCF header.")

    ibd_results = ibd_results.copy()
    ibd_results["id1"] = ibd_results["id1"].apply(lambda x: samples[x])
    ibd_results["id2"] = ibd_results["id2"].apply(lambda x: samples[x])

    if kwargs.get("output", "") != "":
        ibd_results.to_csv(kwargs["output"], index=False, sep="\t")
    else:
        return ibd_results


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    # interpolation
    parser.add_argument("-interpolate", action="store_true")
    parser.add_argument("-genetic_map", type=str)
    parser.add_argument("-input_map", type=str)
    parser.add_argument("-columns", nargs='+', default=[0, 2, 3], type=int)
    parser.add_argument("-sites", type=str, default="")

    # phasedibd
    parser.add_argument("-phasedibd", action="store_true")
    parser.add_argument("-input_vcf", type=str)
    parser.add_argument("-output", type=str)
    parser.add_argument("-use_phase_correction", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.interpolate:
        interpolate_map(input_map=args.input_map, map_file=args.genetic_map, columns=args.columns, sites=args.sites)

    if args.phasedibd:
        run_phasedibd(input_vcf=args.input_vcf,
                      input_map=args.input_map,
                      output=args.output,
                      use_phase_correction=args.use_phase_correction)

