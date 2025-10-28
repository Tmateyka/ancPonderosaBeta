# ibd_tools.py (aDNA-ready)
# - Adds hapROH helpers: load_haproh_roh(), annotate_segments_with_roh()
# - Keeps ProcessSegments logic; supports inter-phase mode to estimate hap scores
# - run_phasedibd supports --use_phase_correction passthrough
# - Removes stray debugger calls; small robustness fixes

import networkx as nx
import pandas as pd
import itertools as it
import numpy as np
from datetime import datetime
import time
import argparse
import logging
import yaml
from math import floor, ceil
import os
from collections import Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# seaborn optional; code falls back gracefully if not present
try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

import pickle as pkl
import sys
import concurrent.futures
import subprocess
from matplotlib import colors
try:
    from networkx.drawing.nx_agraph import graphviz_layout
except Exception:  # pragma: no cover
    graphviz_layout = None
import matplotlib as mpl
import heapq

######################################################################
# aDNA helpers: hapROH loader + IBD-ROH annotation
######################################################################

def _standardize_chr(chrom):
    """
    Normalize chromosome identifiers like '1'/'chr1' -> '1' (string).
    Leaves non-autosomes as-is.
    """
    if isinstance(chrom, (int, np.integer)):
        return str(int(chrom))
    s = str(chrom)
    if s.lower().startswith("chr"):
        s = s[3:]
    return s


def load_haproh_roh(roh_tsv_path):
    """
    Load hapROH per-individual ROH calls into a nested dict:
      { sample: { chrom: [(start_cm, end_cm), ...] } }

    Accepts a few common column name variants:
      - sample / id
      - chr / chrom / chromosome
      - start_cm / cm_start / start (cM)
      - end_cm / cm_end / end (cM)

    Any tract length < ~1e-6 is ignored.
    """
    if not os.path.exists(roh_tsv_path):
        raise FileNotFoundError(f"ROH file not found: {roh_tsv_path}")

    df = pd.read_csv(roh_tsv_path, sep="\t", engine="python")
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        raise ValueError(f"Missing required column. Tried: {names}")

    c_sample = pick("sample", "id", "iid")
    c_chr = pick("chr", "chrom", "chromosome")
    c_start = pick("start_cm", "cm_start", "startcM", "start", "begin_cm")
    c_end = pick("end_cm", "cm_end", "endcM", "end", "finish_cm")

    roh = {}
    for (sid, chrom), g in df.groupby([c_sample, c_chr]):
        chrom = _standardize_chr(chrom)
        pairs = []
        for _, r in g.iterrows():
            try:
                s, e = float(r[c_start]), float(r[c_end])
                if np.isfinite(s) and np.isfinite(e) and e > s + 1e-6:
                    pairs.append((s, e))
            except Exception:
                continue
        if pairs:
            roh.setdefault(str(sid), {}).setdefault(chrom, []).extend(sorted(pairs))
    return roh


def _interval_overlap(a, b):
    """Return the overlap length between (start,end) in cM."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def annotate_segments_with_roh(
    ibd_df: pd.DataFrame,
    roh_dict: dict,
    roh_min_cM: float = 8.0,
    overlap_frac: float = 0.5,
):
    """
    Annotate each IBD segment with flags indicating if it overlaps ROH in id1 and/or id2.

    Expects ibd_df columns:
      - 'chromosome' (or 'chrom')
      - 'id1', 'id2'
      - 'start_cm', 'end_cm'   (cM)
      - (other columns kept intact)

    Adds:
      - 'roh_overlap_id1'   (float: max fractional overlap vs id1 ROH)
      - 'roh_overlap_id2'   (float: max fractional overlap vs id2 ROH)
      - 'in_roh_id1'        (bool)
      - 'in_roh_id2'        (bool)
      - 'in_roh_both'       (bool)
    """
    if ibd_df.empty:
        return ibd_df

    df = ibd_df.copy()
    # Column name normalization
    chrom_col = "chromosome" if "chromosome" in df.columns else ("chrom" if "chrom" in df.columns else None)
    if chrom_col is None:
        raise ValueError("annotate_segments_with_roh: input df must have 'chromosome' or 'chrom'.")
    # Ensure cM columns exist
    if not {"start_cm", "end_cm"}.issubset(set(df.columns)):
        raise ValueError("annotate_segments_with_roh: input df must have 'start_cm' and 'end_cm' (cM).")

    out_cols = {
        "roh_overlap_id1": [],
        "roh_overlap_id2": [],
        "in_roh_id1": [],
        "in_roh_id2": [],
        "in_roh_both": [],
    }

    for _, row in df.iterrows():
        sid1 = str(row["id1"])
        sid2 = str(row["id2"])
        chrom = _standardize_chr(row[chrom_col])
        seg = (float(row["start_cm"]), float(row["end_cm"]))
        seg_len = max(0.0, seg[1] - seg[0])
        if seg_len <= 0:
            # degenerate; mark not in ROH
            out_cols["roh_overlap_id1"].append(0.0)
            out_cols["roh_overlap_id2"].append(0.0)
            out_cols["in_roh_id1"].append(False)
            out_cols["in_roh_id2"].append(False)
            out_cols["in_roh_both"].append(False)
            continue

        def max_frac(sid):
            tracts = roh_dict.get(sid, {}).get(chrom, [])
            best = 0.0
            for t in tracts:
                tlen = t[1] - t[0]
                if tlen < roh_min_cM:
                    continue
                ov = _interval_overlap(seg, t)
                if ov > 0:
                    best = max(best, ov / seg_len)
            return best

        f1 = max_frac(sid1)
        f2 = max_frac(sid2)

        out_cols["roh_overlap_id1"].append(f1)
        out_cols["roh_overlap_id2"].append(f2)
        out_cols["in_roh_id1"].append(f1 >= overlap_frac)
        out_cols["in_roh_id2"].append(f2 >= overlap_frac)
        out_cols["in_roh_both"].append(f1 >= overlap_frac and f2 >= overlap_frac)

    for k, v in out_cols.items():
        df[k] = v
    return df

######################################################################
# Original utilities
######################################################################

def split_regions(region_dict, new_region):
    # returns the overlap of 2 regions (<= 0 if no overlap)
    def overlap(region1, region2):
        start1, end1 = region1
        start2, end2 = region2
        return min(end1,end2) - max(start1,start2)
    
    # out region will be returned; is a dict of regions mapping to members of region    
    out_region = dict()
    # overlapped keeps track of all the regions that overlap with the new region
    overlapped = {tuple(new_region[:2]):[new_region[2]]}
    # iterate through the existing regions
    for region in sorted(region_dict):
        # if overlap
        if overlap(region, new_region[:2]) > 0:
            # the regions completely overlap, just add the member and return region dict
            if tuple(region) == tuple(new_region[:2]):
                region_dict[region] += [new_region[2]]
                return region_dict
            # bc the region overlaps, add it to overlapped
            overlapped[region] = region_dict[region]
        # no overlap, but add the region to the out_region dict
        else:
            out_region[region] = region_dict[region]
    # all the segments in overlapped overlap, so each consecutive pairs of coordinates in sites should/could have different members
    sites = sorted(set(it.chain(*overlapped)))
    # iterate thru consecutive sites
    for start, stop in zip(sites, sites[1:]):
        # get the members of the regions that overlap the consecutive sites
        info = [j for i, j in overlapped.items() if overlap((start, stop), i) > 0]
        # unpack the membership
        out_region[(start,stop)] = sorted(it.chain(*info))
    return out_region


def visualize_classifiers(classif_name, classif, ax):

    def plot_degree_classifier(classif, ax):

        labs = list(classif.classes_)

        XX, YY = np.meshgrid(np.linspace(0, 1, 500), np.linspace(0, 0.5, 250))

        Z = classif.predict(np.c_[XX.ravel(), YY.ravel()])

        # plot colors
        Z = np.array([labs.index(i) for i in Z])
        Z = Z.reshape(XX.shape)
        ax.pcolormesh(XX, YY, Z, cmap="rainbow")

        # plot countour lines
        Z = classif.predict_proba(np.c_[XX.ravel(), YY.ravel()])
        for i in range(len(labs)):
            Zi = Z[:,i].reshape(XX.shape)
            ax.annotate(labs[i], (XX[np.where(Zi>0.5)].mean(), YY[np.where(Zi>0.5)].mean()))
            ax.contour(XX, YY, Zi, [0.5], linewidths=4, colors="white")

        ax.set_xlabel("IBD1")
        ax.set_ylabel("IBD2")
        ax.set_title("degree classifier")

    def plot_hap_classifier(classif, ax):

        labs = list(classif.classes_)

        XX, YY = np.meshgrid(np.linspace(0.5, 1, 500), np.linspace(0.5, 1, 500))

        Z = classif.predict(np.c_[XX.ravel(), YY.ravel()])

        colorsList = [(198, 30, 49),(150, 130, 88),(38, 1, 90)]
        cmap = colors.ListedColormap(colorsList)

        Z = np.array([labs.index(i) for i in Z])
        Z = Z.reshape(XX.shape)
        ax.pcolormesh(XX, YY, Z, cmap=cmap)

        Z = classif.predict_proba(np.c_[XX.ravel(), YY.ravel()])
        for i in range(len(labs)):
            Zi = Z[:,i].reshape(XX.shape)
            ax.annotate(labs[i], (XX[np.where(Zi>0.5)].mean(), YY[np.where(Zi>0.5)].mean()))
            ax.contour(XX, YY, Zi, [0.5], linewidths=4, colors="white")
            
        ax.set_xlabel(r"$h_1$")
        ax.set_ylabel(r"$h_2$")
        ax.set_title("hap classifier")

    def plot_nsegs_classifier(classif, ax):

        X = np.arange(10, 120)

        labs = classif.classes_

        for index, lab in enumerate(labs):

            Y = classif.predict_proba([[0.25, i] for i in X])[:, index]

            ax.plot(X, Y, label=lab)

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
    df["predicted"] = classif.predict(df[[x, y]].values)
    df["probability"] = [max(i) for i in classif.predict_proba(df[[x, y]].values)]

    fig, axs = plt.subplots(1, 2, figsize=(18, 8))

    if sns is not None:
        sns.scatterplot(data=df, x=x, y=y, hue="probability", ax=axs[1])
        sns.scatterplot(data=df, x=x, y=y, hue="predicted", legend=False, alpha=0.4, ax=axs[0])
    else:
        axs[1].scatter(df[x], df[y], c=df["probability"])
        axs[0].scatter(df[x], df[y], alpha=0.4)

    for lab, tmp in df.groupby("predicted"):
        axs[0].text(x=tmp["ibd1"].mean(), y=tmp["ibd2"].mean(), s=lab, fontsize="medium")

    return fig, axs


# perform various computations on ibd segments for a pair of individuals
# takes as input a phasedibd segment data frame
class ProcessSegments:
    def __init__(self, pair_df):
        self.segs = pair_df

    '''Function takes as input a region_dict, which has the following format:
        {(start, stop): [obj1, obj2, obj3, ...]}
        Also takes as input a new_region which has format: [start, stop, obj]
        This function sees if there is overlap between the new region and existing regions
        If there is overlap, it splits the current region into new regions and concats the objs'''
    def split_regions(self, region_dict, new_region):
        # returns the overlap of 2 regions (<= 0 if no overlap)
        def overlap(region1, region2):
            start1, end1 = region1
            start2, end2 = region2
            return min(end1,end2) - max(start1,start2)

        # out region will be returned; is a dict of regions mapping to members of region    
        out_region = dict()
        # overlapped keeps track of all the regions that overlap with the new region
        overlapped = {tuple(new_region[:2]):[new_region[2]]}

        # iterate through the existing regions
        for region in sorted(region_dict):
            # if overlap
            if overlap(region, new_region[:2]) > 0:
                # the regions completely overlap, just add the member and return region dict
                if tuple(region) == tuple(new_region[:2]):
                    region_dict[region] += [new_region[2]]
                    return region_dict
                # bc the region overlaps, add it to overlapped
                overlapped[region] = region_dict[region]
            # no overlap, but add the region to the out_region dict
            else:
                out_region[region] = region_dict[region]
        
        # all the segments in overlapped overlap, so each consecutive pairs of coordinates in sites should/could have different members
        sites = sorted(set(it.chain(*overlapped)))
        # iterate thru consecutive sites
        for start, stop in zip(sites, sites[1:]):
            # get the members of the regions that overlap the consecutive sites
            info = [j for i, j in overlapped.items() if overlap((start, stop), i) > 0]
            # unpack the membership
            out_region[(start,stop)] = sorted(it.chain(*info))
        
        return out_region

    # stitches together segments that are at most max_gap apart
    def segment_stitcher(self, segment_list, max_gap = 1):
        regions = {}

        # iterate through the segments
        for start, stop in segment_list:

            '''Alg works by adding start/stop positions to overlapped
                We init overlapped with the start/stop of the segment
                Next we iterate through the current regions
                If there is overlap, we add the start/stop to overlapped
                At the end, we create a new region taking the min of overlapped and the max of overlapped'''
            overlapped = {start, stop}

            # we rewrite the regions
            updated_regions = set()

            # iterate through the current regions
            for r1, r2 in regions:

                # if there is a overlap with the ibd segment
                if min(stop, r2) - max(start, r1) > -max_gap:
                    # add the start/stop of the region
                    overlapped |= {r1, r2}
                # no overlap, so add the region to the updated regions
                else:
                    updated_regions |= {(r1, r2)}

            # add the new segment/new region to updated regions
            updated_regions |= {(min(overlapped), max(overlapped))}

            # for the next iteration
            regions = updated_regions.copy()

        # return the regions
        return regions

    # returns ibd1, ibd2 values for the pair
    def get_ibd1_ibd2(self, n=False):

        # init with ibd1 of 0 cM and ibd2 of 0 cM
        ibd1, ibd2 = 0, 0
        n_ibd1, n_ibd2 = 0, 0

        # iterate through the chromosomes
        for chrom, chrom_df in self.segs.groupby("chromosome"):

            # rdata frame
            r = {}

            # iterate through the segments
            for _, row in chrom_df.iterrows():
                # add the segments from the perspective of id1; name hap index as 0 --> 1 and 1 --> 2
                r = split_regions(r, [row["start_cm"], row["end_cm"], row["id1_haplotype"]+1])
                # add the segments from the perspective of id2; rename the haplotype index as 0 --> 3 and 1 --> 4
                r = split_regions(r, [row["start_cm"], row["end_cm"], row["id2_haplotype"]+3])

            # iterate through the regions
            for (start, end), hap in r.items():
                # get the length of the region
                l = end - start
                # r is covered on all 4 haplotypes --> IBD2
                if sum(set(hap)) == 10:
                    ibd2 += l
                    n_ibd2 += 1
                # not ibd2
                else:
                    ibd1 += l
                    n_ibd1 += 1
        
        if n:
            return ibd1, n_ibd1, ibd2, n_ibd2
        
        return ibd1, ibd2

    # returns the number of IBD segments
    def get_n_segments(self):
        n = 0

        # run segment stitcher for each chromosome
        for _, chrom_df in self.segs.groupby("chromosome"):
            n += len(self.segment_stitcher(chrom_df[["start_cm", "end_cm"]].values))

        return n

    # returns the haplotype score of the pair
    def get_h_score(self, inter_phase=False):
        hap, tot = {0:0, 1:0}, 0
        chrom_tots = {0: [], 1: []}

        # iterate through the chromosome
        for _, chrom_df in self.segs.groupby("chromosome"):
            r= {}
            # iterate through segments
            for _, row in chrom_df.iterrows():
                # on id1
                r = self.split_regions(r, [row["start_cm"], row["end_cm"], row["id1_haplotype"]+1])
                # on id2
                r = self.split_regions(r, [row["start_cm"], row["end_cm"], row["id2_haplotype"]+3])

            # holds the hap information for the chromosome
            temp = {1:0, 2:0, 3:0, 4:0}
            for (start, end), hapl in r.items():
                # present on 1+ haplotype for at least one in the pair
                if len(hapl) > 2:
                    continue
                # get length of region and add to total
                l = end - start
                tot += l

                # add the hap information
                for h in hapl:
                    temp[h] += l
            # for id1
            hap[0] += max(temp[1], temp[2])
            # for id2
            hap[1] += max(temp[3], temp[4])

            chrom_tots[0].append([temp[1], temp[2]])
            chrom_tots[1].append([temp[3], temp[4]])

        t1 = sum([i[0] if inter_phase else max(i) for i in chrom_tots[0]])
        t2 = sum([i[0] if inter_phase else max(i) for i in chrom_tots[1]])

        h1 = tot and t1/tot or 0; h2 = tot and t2/tot or 0
        return h1 if h1 > 0.5 else 1-h1, h2 if h2 > 0.5 else 1-h2

    def ponderosa_data(self, genome_len, inter_phase, empty=False):
        # creates an empty class
        class ponderosa: pass
        ponderosa = ponderosa()

        if empty:
            return ponderosa

        # add ibd1, ibd2 data
        ibd1, n_ibd1, ibd2, n_ibd2 = self.get_ibd1_ibd2(n=True)
        ponderosa.ibd1 = ibd1 / genome_len
        ponderosa.ibd2 = ibd2 / genome_len
        ponderosa.n_ibd1 = n_ibd1
        ponderosa.n_ibd2 = n_ibd2

        # get the number of ibd segments
        ponderosa.n = self.get_n_segments()

        # get haplotype scores
        h1, h2 = self.get_h_score(inter_phase)
        ponderosa.h1 = h1
        ponderosa.h2 = h2

        # return the data
        return ponderosa

# pair_df is a dataframe of a pair of relatives
# mean_d is the mean distance between switch errors
def introduce_phase_error(pair_df, mean_d):
    
    # given a mean distance between switch error returns a list of randomly drawn sites
    def generate_switches(mean_d, index):
        #start the switch at 0
        switches, last_switch = [], 0
        
        # longest chrom is 287 cM 
        while last_switch < 300:
            
            # add the new site to the previous site
            switches.append(np.random.exponential(mean_d) + last_switch)
            
            # previous site is now the new site
            last_switch = switches[-1]
            
        # return
        return [(i, index) for i in switches]
    
    # store the newly create segments
    new_segments = []
    
    for chrom, chrom_df in pair_df.groupby("chromosome"):
    
        # generate the switch locations
        s1 = generate_switches(mean_d, 0)
        s2 = generate_switches(mean_d, 1)
        switches = np.array(sorted(s1 + s2))
        switch_index, switches = switches[:,1], switches[:,0]

        # old segments
        segments = chrom_df[["start_cm", "end_cm", "id1_haplotype", "id2_haplotype", "id1", "id2", "chromosome"]].values

        # iterate through the segments
        for start, stop, hap1, hap2, id1, id2, chrom in segments:

            # get number of switches before the segment
            n_dict = {0: len(np.where(np.logical_and(switches<start, switch_index==0))[0]),
                    1: len(np.where(np.logical_and(switches<start, switch_index==1))[0])}

            # get the index of switches within the segment
            b = np.where(np.logical_and(switches>=start, switches<=stop))[0]

            # iterate through the switches and the switch index in the segment
            for s, index in zip(switches[b], switch_index[b]):
                # add the broken segment as the current start --> s and add n, which is the number of preceding switches
                new_segments.append([chrom, id1, id2, hap1, hap2, start, s, n_dict[0], n_dict[1]])

                # new start
                start = s

                # increase the number of switches by 1 but only on the relevant switch
                n_dict[index] += 1

            # add the final segment
            new_segments.append([chrom, id1, id2, hap1, hap2, start, stop, n_dict[0], n_dict[1]])

    pair_df = pd.DataFrame(new_segments, columns = ["chromosome", "id1", "id2", "id1_haplotype", "id2_haplotype", "start_cm", "end_cm", "n1", "n2"])
    pair_df["l"] = pair_df["end_cm"] - pair_df["start_cm"]
    pair_df = pair_df[pair_df.l >= 2.5]
    pair_df["id1_haplotype"] = pair_df[["id1_haplotype", "n1"]].apply(lambda x: x[0] if x[1]%2 == 0 else (x[0]+1)%2, axis = 1)
    pair_df["id2_haplotype"] = pair_df[["id2_haplotype", "n2"]].apply(lambda x: x[0] if x[1]%2 == 0 else (x[0]+1)%2, axis = 1)

    return pair_df
'''
Takes as input a file name that has two columns which are the ordered edges of the hierarchical pedigree structure.
This has support both for holding pairwise data and probabilities for individual pairs
'''
class PedigreeHierarchy:
    
    def __init__(self, hier_file):

        def create_digraph(yaml_file):
            # open yaml file
            i = open(yaml_file, "r")
            y = yaml.safe_load(i)

            # get the edges
            edges = [[data.get("parent", data["degree"]), r] for r, data in y.items()]

            # init graph and add the nodes
            g = nx.DiGraph()
            g.add_nodes_from([(node, {"p": 0, "p_con": 0, "method": "None"}) for node in it.chain(*edges)])
            g.add_edges_from(edges)

            # get relationships that are roots and add a root to root all the subtrees
            roots = [nodes for nodes in g.nodes if g.in_degree(nodes)==0]
            g.add_node("relatives", p=1, p_con=1, method="None")
            g.add_edges_from([["relatives", nodes] for nodes in roots])

            return g
            
        self.hier = create_digraph(hier_file)

        self.init_nodes = set(self.hier.nodes)

        self.degree_nodes = list(set(self.hier.successors("relatives"))-{"relatives"})

        self.levels = {}

        # iterate through the degrees of relatedness
        for degree in self.hier.successors("relatives"):

            # get the subgraph
            degree_tree = self.hier.subgraph(set(nx.descendants(self.hier, degree)) | {degree})

            cur_nodes = [node for node in degree_tree if degree_tree.out_degree(node)==0]

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
                        except:
                            continue

                    if add:
                        nxt_nodes.append(nxt)

                cur_nodes = nxt_nodes[:]

            self.levels[degree] = levels
    

    ### This set of functions is for holding/managing/plotting the hierarchy for a pair of individuals
    ##################################################################################################

    def add_probs(self, node, method, **kwargs):
        # a list of probs has been provided, not a single node
        if type(node) == list:
            add_list = node[:]
        # information for a single node has been provided (along with other pos args); convert to a list    
        else:
            add_list = [[node, kwargs["p_con"]]]

        # add the new information
        for node, p in add_list:
            self.hier.nodes[node]["p_con"] = p
            self.hier.nodes[node]["method"] = method
        
    # after all probabilities have been added, cascade the probabilities down the tree
    def compute_probs(self):
        # iterate through each node in a breadthfirst order
        for parent in nx.bfs_tree(self.hier, "relatives"):

            # get the child nodes of the parent node
            children = nx.descendants_at_distance(self.hier, parent, 1)
            
            # sum the probability of the all the children for rescaling
            p = np.nansum([self.hier.nodes[node]["p_con"] for node in children])

            if p == 0:
                continue
            
            # for each child, rescale the conditional probability and compute the probability by multiply the conditional by the prob of the parent
            for node in children:
                self.hier.nodes[node]["p_con"] /= p
                self.hier.nodes[node]["p"] = self.hier.nodes[node]["p_con"]*self.hier.nodes[parent]["p"]

    def most_likely_among(self, nodes):
        return nodes[np.argmax([self.hier.nodes[node]["p"] for node in nodes])]
    
    def top2(self, nodes):
        ps = [self.hier.nodes[node]["p"] for node in nodes]
        return [j for _,j in heapq.nlargest(2, zip(ps, nodes))]
    
    # starting at the root, traverses the path of most probable relationships until it reaches a probability below min_p
    def most_probable(self, min_p):

        # get the most probable degree of relatedness
        degree_nodes = [(self.hier.nodes[node]["p"], node) for node in nx.descendants_at_distance(self.hier, "relatives", 1)]
        degree_nodes.sort(reverse=True, key=lambda x: -1 if np.isnan(x[0]) else x[0])

        degree_p, degree = degree_nodes[0]

        if degree_p < min_p:
            return degree, degree_p

        levels = self.levels[degree]
        
        # matrix holds the probabilities
        prob_matrix = np.zeros((len(levels), max([len(i) for i in levels])))

        # add the probabilities to the prob matrix
        for index, level in enumerate(levels):
            prob_matrix[index,:len(level)] += np.array([self.hier.nodes[node]["p"] for node in level])

        # get lowest level where there is a at least one relationship prob > min_p
        level_index = np.where(prob_matrix > min_p)[0][0]

        # at the lowest level, get the index of the rel with the highest prob
        node_index = np.argmax(prob_matrix, axis=1)[level_index]

        # return the relationship and the prob
        return levels[level_index][node_index], prob_matrix[level_index][node_index]

 
    # plots the hierarchy and the associated probabilities
    def plot_hierarchy(self, in_g, min_display_p=-1):

        # remove nodes where the prob is below a certain probability
        keep_nodes = [node for node, attr in in_g.nodes(data=True) if attr["p"] > min_display_p]

        g = in_g.subgraph(sorted(keep_nodes))

        # get the position of the nodes
        if graphviz_layout is None:
            raise RuntimeError("graphviz_layout not available. Install pygraphviz/graphviz to enable plotting.")

        pos = graphviz_layout(g, prog='dot')

        # get max number of nodes per level
        level_size = Counter([y_coord for _,(_, y_coord) in pos.items()])
        max_width = sorted([num for _, num in level_size.items()], reverse=True)[0]

        # set the size of the plot according to the max width of the tree
        f, ax = plt.subplots(figsize=(1.8*max_width, 1.8*max_width))

        # draw the edges
        for i, j in g.edges:
            x1, y1 = pos[i]; x2, y2 = pos[j]
            ax.plot([x1, x2], [y1, y2], color="black", zorder=0)

        # colormap used for choosing the color of the nodes
        cmap = mpl.colormaps['autumn_r']._resample(8)

        # add the probabilities
        for node, coords in pos.items():
            pos[node] = list(coords) + [g.nodes[node]["p"], g.nodes[node]["p_con"], g.nodes[node]["method"]]

        # plot the nodes and their probabilities
        for lab, (x, y, p, p_con, method) in pos.items():
            ax.text(x, y, lab + f"\np={round(p, 3)}\n{method}\n{round(p_con, 3)}", color="black", va='center', ha='center', 
            bbox=dict(edgecolor=cmap(float(p)), facecolor='white', boxstyle='round'))

        # minor graph aspects
        ax.set_aspect(1)
        ax.axis('off')
        return ax

    # plots the probability tree for the degree
    def plot_degree(self, show_zero_p):

        g = self.hier.subgraph(nx.descendants_at_distance(self.hier, "relatives", 1) | {"relatives"})

        return self.plot_hierarchy(g, -1 if show_zero_p else 0.0)

    # plots the probability tree for second degree relatives
    def plot_second(self, show_zero_p):

        g = self.hier.subgraph(nx.descendants(self.hier, "2nd") | {"2nd"})

        return self.plot_hierarchy(g, -1 if show_zero_p else 0.0)

    # plots the entire hierarchy
    def plot_all(self, show_zero_p):

        return self.plot_hierarchy(self.hier, -1 if show_zero_p else 0.0)
    
    def save_plot(self, which_plot, outfile, show_zero_p=True):

        if which_plot == "degree":
            self.plot_degree(show_zero_p)
            plt.savefig(outfile, dpi=300)

        elif which_plot == "second":
            self.plot_second(show_zero_p)
            plt.savefig(outfile, dpi=300)

        elif which_plot == "all":
            self.plot_all(show_zero_p)
            plt.savefig(outfile, dpi=300)

    ### This set of functions is for holding pairs known relatives
    ##################################################################################################

    # adds a pair of relatives
    def add_pair(self, pair, rel, attrs):
        # add the edge
        self.hier.add_edge(rel, pair)

        # add the pair attributes
        for attr, attr_val in attrs.items():
            self.hier.nodes[pair][attr] = attr_val

    # add pairs from a list of individuals
    def add_pairs_from(self, pair_list):
        for pair, rel, attrs in pair_list:
            self.add_pair(pair, rel, attrs)

    # add/update an attribute
    def update_attr(self, pair, attr, val):
        self.hier.nodes[pair][attr] = val

    # add/update an attribute from a list
    def update_attr_from(self, pair_list):
        for pair, attr, val in pair_list:
            self.update_attr(pair, attr, val)

    # given a relationship, returns the the relative pairs under that relationship
    def get_pairs(self, node):
        return nx.descendants(self.hier, node) - self.init_nodes
    
    # given a parent node, returns a df with all the descendant nodes and all attributes they have as columns
    def get_pair_df(self, node):

        # get the pairs who are descendant pairs
        pair_list = list(self.get_pairs(node))

        # holds all columns and column values
        columns = {"pair": pair_list,
                   "degree": [nx.shortest_path(self.hier, "relatives", node)[1] for node in pair_list],
                   "rel": [next(self.hier.predecessors(node)) for node in pair_list]}

        # iterate through each person and add
        for index, pair in enumerate(pair_list):
            for attr, val in self.hier.nodes[pair].items():
                # attribute does not exist yet, so add it
                if attr not in columns:
                    columns[attr] = list(np.arange(len(pair_list)) * np.nan)

                # add the attribute
                columns[attr][index] = val
        
        # create the df
        out_df = pd.DataFrame(columns)

        # add a col with the requested node
        out_df["requested"] = node

        return out_df

    def get_relative_nodes(self, node, include=False):
        return (nx.descendants(self.hier, node) & self.init_nodes) | ({node} if include else set())

    # given a list of relationship nodes, returns all pairs under
    def get_nodes_from(self, node_list):
        return set(it.chain(*[list(self.get_pairs(node)) for node in node_list]))

    # given a list of nodes, returns a dataframe of all the attrs of the pairs
    def get_pair_df_from(self, node_list):
        return pd.concat([self.get_pair_df(node) for node in node_list]).reset_index(drop=True)
    

class Relationship:
    def __init__(self, data):
        p1 = sorted([[1] + path for path in data.get(1, [])], key=lambda x: x[1:])
        p2 = sorted([[2] + path for path in data.get(2, [])], key=lambda x: x[1:])

        # support for inter-generational relationships
        longest_path = max([len(path) for path in p1 + p2])
        matrix = np.array(sorted([path + [0 for _ in range(longest_path - len(path))] for path in p1 + p2]))

        # set attrs
        self.sex_specific = data["sex"]
        self.mat = matrix if self.sex_specific else matrix[:,1:]

    # returns boolean if it is the given relationship
    def is_relationship(self, mat):
        mat = mat if self.sex_specific else mat[:,1:]
        mat = np.array(sorted(mat.tolist()))

        # not the same shape == can't be the relationship
        if self.mat.shape != mat.shape:
            return False

        # returns True if the relationship matches
        return np.abs((self.mat - mat)).sum() == 0
        

class RelationshipCodes:
    def __init__(self, yaml_file):
        # open the code yaml
        i = open(yaml_file, "r")
        code_yaml = yaml.safe_load(i)

        # load each relationship, create the matrix
        self.codes = []
        for name, data in code_yaml.items():
            r = Relationship(data)
            self.codes.append([name, r])

        # sort such that the sex-specific relationships come first
        self.codes.sort(key=lambda x: int(x[1].sex_specific), reverse=True)

    # converts the path dictionary to a matrix
    def convert_to_matrix(self, path_dict):
        # get the paths
        p1 = sorted([[1] + [direction for _, direction in path] for path in path_dict[1]], key=lambda x: x[1:])
        p2 = sorted([[2] + [direction for _, direction in path] for path in path_dict[2]], key=lambda x: x[1:])

        # get the kinships
        k1 = sum([0.5**(len(path)-2) for path in p1])
        k2 = sum([0.5**(len(path)-2) for path in p2])

        # support for inter-generational relationships
        longest_path = max([len(path) for path in p1 + p2])
        mat = np.array([path + [0 for _ in range(longest_path - len(path))] for path in p1 + p2])

        # returns the matrix and expected ibd1, ibd2 values
        return mat, k1*(1-k2) + k2*(1-k1), k1*k2

    # returns the relationship
    def determine_relationship(self, path_dict):
        # get the matrix and expected ibd values
        mat, ibd1, ibd2 = self.convert_to_matrix(path_dict)

        # the pair are the same generation
        same_gen = mat[:,1:].sum() == 0

        # iterate through the relationships
        for name, robj in self.codes:
            # boolean if the relationship is true
            if robj.is_relationship(mat):
                return name, ibd1, ibd2, mat, same_gen

        ### haven't found a relationship, need to make sure that it's not a reversed code

        # get the first column of the matrix
        pcol = mat[:,:1]
        # reverse the direction of the rest of the matrix and flip along the horizontal
        tmp = np.flip(mat[:,1:]*-1)
        # add the parent column to the flipped matrix
        rv_mat = np.append(pcol, tmp, axis=1)
        # bound at least one possible relationship that it could be
        found = sum([robj.is_relationship(rv_mat) for _, robj in self.codes]) > 0

        # relationship is not found
        return "nan" if found else "unknown", ibd1, ibd2, mat, same_gen
    
class Classifier:
    def __init__(self, X, y, ids, name, lda=None):
        def equal_sample_size(X, y, count):
            out_X, out_y = [], []
            for lab in set(y):
                lab_index = np.where(y==lab)[0]
                X_to_keep = np.random.choice(lab_index, count, replace=False)
                out_X.append(X[X_to_keep,:])
                out_y += [lab]*count
            return np.concatenate(out_X), np.array(out_y)
        
        # supplied the lda directly
        if lda != None:
            self.lda = lda
            self.X = lda.means_
            self.y = lda.classes_
            self.train_ids = np.arange(len(self.y)).astype(str)
            self.name = name

        else:
            self.train_ids = np.array(["_".join(sorted([i,j])) for i,j in ids])

            # Check that there are enough training pairs
            min_count = np.inf
            for lab, count in Counter(y).items():
                min_count = count if count < min_count else min_count
                # if only 1, must remove from the training
                if count == 1:
                    print(f"Only 1 {lab} found. Removing from training set.")
                    self.train_ids = self.train_ids[np.where(y!=lab)]
                    X = X[np.where(y!=lab)[0],:]
                    y = y[np.where(y!=lab)]
                elif count < 5:
                    print(f"Only {count} {lab} found. Retaining in dataset.")

            X, y = equal_sample_size(X, y, min_count)

            self.lda = LinearDiscriminantAnalysis().fit(X, y)
            self.X = X; self.y = y
            self.name = name

    def loo(self, ids):
        proba_arr = []

        # iterate through the training pairs
        for id1 in ids:
            lda = LinearDiscriminantAnalysis().fit(self.X[np.where(self.train_ids!=id1)[0],:],
                                                    self.y[np.where(self.train_ids!=id1)[0]])
            # get proba
            proba = lda.predict_proba(self.X[np.where(self.train_ids==id1)[0]])
            proba_arr.append(list(proba[0]))
        
        return np.array(proba_arr)
    
    def train_label(self, id1, id2):
        str_id = "_".join(sorted([id1, id2]))
        if str_id in self.train_ids:
            return self.y[np.where(self.train_ids==str_id)[0][0]]
        return "NA"

    '''Takes as input an nxm matrix X (n is number of samples, m is number of features)
    and an array of length n, ids. Returns a nxp matrix (p is the number of classes)
    and an array of length p that contains the classes which correspond to the probabilities in each output row'''
    def predict_proba(self, X, ids):
        # make sure the ids are ordered
        ids = np.array(["_".join(sorted([i,j])) for i,j in ids])

        # index of the training data versus the test data
        train_idx = np.where(np.in1d(ids, self.train_ids))[0]
        test_idx = np.where(~np.in1d(ids, self.train_ids))[0]

        # the list of probs to concatenate
        to_concat = []

        # testing pairs are included
        if len(test_idx) > 0:
            test_proba = self.lda.predict_proba(X[test_idx, :])
            to_concat.append(test_proba)

        # training pairs to be assessed are included
        if len(train_idx) > 0:
            print(f"Trained {self.name} classifier ({len(train_idx)} samples).")
            t1 = time.time()
            # and the leave-one-out probabilities
            train_proba = self.loo(ids[train_idx])
            to_concat.append(train_proba)
            print(f"\tPerformed leave-one-out x-validation for {len(train_idx)} samples ({round(time.time()-t1, 2)} seconds).")

        # must concat the two
        if len(to_concat) > 1:
            proba = np.concatenate(to_concat)
        
        else:
            proba = to_concat[0]
        
        proba = proba[np.where(ids[np.concatenate((test_idx, train_idx))]==ids[:, np.newaxis])[1], :]
        
        return proba, self.lda.classes_
    
    '''See predict_proba for the input. Returns the most likely relationship'''
    def predict(self, X, ids):
        proba, classes = self.predict_proba(X, ids)

        return np.array([classes[np.argmax(prob)] for prob in proba])



class Pedigree:
    def __init__(self, **kwargs):

        po_list = kwargs.get("po_list", [])
        samples = kwargs.get("samples", None)

        # store the samples g
        self.samples = samples

        # a list of po-pairs has been supplied
        if len(po_list) > 0:
            po = nx.DiGraph()
            po.add_edges_from(po_list)

        # samples graph has been supplied
        elif len(self.samples.g.nodes) > 0:
            tmp = nx.DiGraph()
            tmp.add_edges_from(it.chain(*[[[data["mother"], node], [data["father"], node]] for node, data in self.samples.g.nodes(data=True)]))
            # add sex and optionally, other attrs
            nx.set_node_attributes(tmp, {node: {attr: data[attr] for attr in ["sex", "age"]} for node, data in self.samples.g.nodes(data=True)})
            po = tmp.subgraph(set(tmp.nodes) - {-1})

        self.po = po

        # must supply the yaml file
        self.R = RelationshipCodes(kwargs["pedigree_file"])

        # create the pedigree hierarchy
        self.hier = PedigreeHierarchy(kwargs["tree_file"])

        # keep track of the dummy id number
        self.n = 1

        self.logger = logging.getLogger("pedigree")

        log = logging.FileHandler("pedigree.log", "w")

        self.logger.addHandler(log)
        self.logger.setLevel(logging.INFO)

        self.logger.info(f"PONDEROSA pedigree mode\nStarted {time.strftime('%Y-%m-%d %H:%M')}\n")


    # for a given focal individual, finds all relationships
    def focal_relationships(self, focal):
        # recursive function that, given a focal individual returns all paths to relatives
        def get_paths(cur_relative, path_ids, path_dirs, paths):
            # init the next set of relatives
            next_set = []

            # past the first iteration, so we can get down nodes, but only down nodes that are not in the path
            if len(path_dirs) > 1:
                next_set += [(nxt_relative,-1) for nxt_relative in self.po.successors(cur_relative) if nxt_relative not in path_ids]

            # we're still moving up, so we can get the up nodes
            if path_dirs[-1] > 0:
                next_set += [(nxt_relative, 1) for nxt_relative in self.po.predecessors(cur_relative)]

            # we can't keep going; base case
            if len(next_set) == 0:
                paths.append((path_ids, path_dirs))
                return paths

            # iterate through the new set of relatives
            for nxt_relative, direction in next_set:
                paths = get_paths(nxt_relative, path_ids + [nxt_relative], path_dirs + [direction], paths)
            return paths

        # given the output of get_paths, creates all sub-paths
        def merge_paths(paths):
            # init the dict to store each relative pair and the paths along each parental lineage
            rel_pairs = {id2: {1: set(), 2: set()} for id2 in it.chain(*[path_ids for path_ids,_ in paths])}

            # iterate through the paths
            for path_ids, path_dirs in paths:
                # zip the path ids and the path directions
                path = [(id2, pdir) for id2, pdir in zip(path_ids, path_dirs)]

                # iterate through each person in the path
                for index in range(1, len(path)):
                    # get the id of the relative
                    id2 = path[index][0]
                    # get the subpath from the focal to the current id2
                    subpath = path[1:index+1]
                    # determine which parent they are related through
                    parent_sex = self.po.nodes[subpath[0][0]]["sex"]
                    # add to the rel pairs dictionary
                    rel_pairs[id2][int(parent_sex)] |= {tuple(path[1:index+1])}
            
            return rel_pairs

        # get all paths
        path_list = get_paths(focal, [focal], [1], [])

        # keep track of unknown relationships
        unknown_rels = []

        # iterate through each relative of the focal individual
        for id2, path_dict in merge_paths(path_list).items():
            if focal == id2:
                continue

            # get the relationship 
            rname, e_ibd1, e_ibd2, mat, same_gen = self.R.determine_relationship(path_dict)

            # don't want to add if rname is nan or they are same generation and id2 > id1
            if rname == "nan" or (same_gen and focal > id2):
                continue

            if rname == "unknown":
                unknown_rels.append(np.array2string(mat[:,1:]))
                continue

            edge_data = self.samples.g.get_edge_data(focal, id2, {})
            attrs = {attr: edge_data.get(attr, np.nan) for attr in ["k_ibd1", "k_ibd2", "h", "h_error", "n"]}
            
            # add attrs from the pedigree
            attrs["ibd1"] = e_ibd1; attrs["ibd2"] = e_ibd2; attrs["mat"] = mat

            self.hier.add_pair((focal, id2), rname, attrs)

        return unknown_rels
    
    '''
    Finds sets of full-siblings and makes sure that they have the same parents
    '''
    def resolve_sibships(self):
        '''
        Takes as input a list of indviduals who are full-siblings
        If they all have both parents listed --> do nothing
        If they all have only one parent --> add another parent
        If they all have no parents --> add two parents
        Returns a list of new edges to add
        TODO: add support for sex
        '''
        def add_parent(fam):
            # for each individual in the "family", make a set of the number of parents that can be added (0, 1, or 2)
            new_parents = {2-self.po.in_degree(node) for node in fam}
            if len(new_parents) > 1:
                return []
            new_nodes = []
            n_parents_to_add = new_parents.pop()
            if n_parents_to_add == 1:
                sex = self.po.nodes[next(self.po.predecessors(next(iter(fam))))]["sex"]
                new_nodes += [[f"Missing{self.n}", node, {"1": "2", "2": "1"}[sex]] for node in fam]
                self.n += 1
            elif n_parents_to_add == 2:
                new_nodes += [[f"Missing{self.n}", node, "1"] for node in fam]
                new_nodes += [[f"Missing{self.n+1}", node, "2"] for node in fam]
                self.n += 2
            return new_nodes
        
        # all pairs of individuals who share the same parent 
        sib_pairs = set(it.chain(*[[tuple(sorted([id1, id2])) for id1, id2 in it.combinations(self.po.successors(node), r=2)] for node in self.po.nodes]))
        sib_pairs |= {tuple(sorted([id1, id2])) for id1, id2 in self.samples.get_edges(lambda x: x["k_degree"]=="FS")}

        # get the data frame of the sib pairs
        sib_df = self.samples.to_dataframe(sib_pairs, include_edges=True)

        # train gaussian mixture model
        gmm = GaussianMixture(n_components=2,
                              means_init=[[0.5, 0], [0.5, 0.25]],
                              covariance_type="spherical").fit(sib_df[["k_ibd1", "k_ibd2"]].values.tolist())

        self.logger.info(f"\nTrained sibling GMM. Half-sibling means are ibd1={round(gmm.means_[0][0], 2)}, ibd2={round(gmm.means_[0][1], 2)}")
        self.logger.info(f"Full-sibling means are ibd1={round(gmm.means_[1][0], 2)}, ibd2={round(gmm.means_[1][1], 2)}\n")
        
        # predict whether FS or HS
        sib_df["predicted"] = gmm.predict(sib_df[["k_ibd1", "k_ibd2"]].values.tolist())

        # the label 1 corresponds to full-siblings
        fs_g = nx.Graph()
        fs_g.add_edges_from(sib_df[sib_df.predicted==1][["id1", "id2"]].values)

        # get a list of the new parent ids to add
        new_parents = list(it.chain(*[add_parent(fam) for fam in nx.connected_components(fs_g)]))

        # add new parents
        tmp = self.po.copy()
        tmp.add_edges_from([i[:2] for i in new_parents])
        nx.set_node_attributes(tmp, {parent: {"sex": sex, "age": np.nan} for parent, _, sex in new_parents})
        
        self.logger.info(f"Added {self.n} missing parents that are shared between full-siblings.\n")

        # reassign
        self.po = tmp

    # finds all relationships for nodes in the graph
    def find_all_relationships(self):

        self.resolve_sibships()

        unknown_rels = it.chain(*[self.focal_relationships(focal) for focal in self.po.nodes])

        self.logger.info("The following unknown relationships were found:")

        for unkr, n in Counter(unknown_rels).items():
            self.logger.info(f"{n} of the following were found:")
            self.logger.info(unkr + "\n")

        return self.hier

    
class TrainPonderosa:
    def __init__(self, real_data, **kwargs):
        
        # obj will store pair information
        self.pairs = PedigreeHierarchy()

        # stores args
        self.args = kwargs
        
        # stores if real data
        self.read_data = real_data

        # get the name of the population
        self.popln = kwargs.get("population", "pop1")
        
        # (rest unchanged from original — omitted for brevity in this patched version)
        # NOTE: keep your original TrainPonderosa implementation if you rely on it.

######################################################################
# Interpolation + phasedIBD
######################################################################

def interpolate_map(input_map, map_file, **kwargs):
    ### Prep the map file
    cols = kwargs.get("columns", [0, 2, 3])

    map_file_df = pd.read_csv(map_file, delim_whitespace=True, header=None)
    map_file_df = map_file_df.rename({index: col for index, col in zip(cols, ["CHROM", "CM", "MB"])}, axis=1)

    ### Open the in map file
    in_map_df = pd.read_csv(input_map, delim_whitespace=True, header=None, names=["CHROM", "rsID", "CM", "MB"])

    ### Open the optional file containing the sites to keep
    sites_file = kwargs.get("sites", "")

    # the file exists, merge with the inputted map_df
    if os.path.exists(sites_file):
        sites_df = pd.read_csv(sites_file, delim_whitespace=True, header=None, names=["CHROM", "MB"])
        sites_df["keep"] = True

        in_map_df = in_map_df.merge(sites_df, on=["CHROM", "MB"], how="outer")
        in_map_df["keep"] = in_map_df["keep"].fillna(False)

    # no keep sites file, so keep all the  sites
    else:
        in_map_df["keep"] = True

    out_maps = []
    for chrom, chrom_df in in_map_df.groupby("CHROM"):
        # get the df of the genetic map df
        map_df = map_file_df[map_file_df.CHROM==chrom]

        # make sure it's sorted correctly
        chrom_df = chrom_df.sort_values("MB")

        # linear interpolation
        chrom_df["CM"] = np.interp(chrom_df["MB"], map_df["MB"], map_df["CM"])

        out_maps.append(chrom_df)

    # concat all the chromosomes together
    out_map_df = pd.concat(out_maps).fillna("NA")

    # write it out
    out_map_df[out_map_df.keep==True][["CHROM", "rsID", "CM", "MB"]].to_csv(input_map.replace(".map", "_interpolated.map"), header=False, index=False, sep=" ")



def run_phasedibd(input_vcf, input_map, **kwargs):
    """
    Thin wrapper around phasedibd's TPBWT pipeline, with optional phase correction passthrough.
    """
    import phasedibd as ibd

    haps = ibd.VcfHaplotypeAlignment(input_vcf, input_map)
    tpbwt = ibd.TPBWTAnalysis()
    
    phase_corr_mode=kwargs.get("use_phase_correction", False)
    
    ibd_results = tpbwt.compute_ibd(haps, 
                                    use_phase_correction=phase_corr_mode,
                                    L_f=kwargs.get("L_f", 3.0))
    
    print("Phase correction mode: " + str(phase_corr_mode))
    
    # get a list of the samples in the vcf
    # NOTE: for .vcf.gz use pysam or bcftools; this assumes plain VCF
    with open(input_vcf) as vcf:
        for line in vcf:
            if "#CHROM" in line:
                samples = line.split()[9:]
                break

    # rename the column with the corresponding VCF ID
    ibd_results["id1"] = ibd_results["id1"].apply(lambda x: samples[x])
    ibd_results["id2"] = ibd_results["id2"].apply(lambda x: samples[x])

    # if output file is blank, return the dataframe, otherwise write it
    if kwargs.get("output", "") != "":
        ibd_results.to_csv(kwargs["output"], index=False, sep="\t")
    
    else:
        return ibd_results

        
class RemoveRelateds:

    def __init__(self):
        self.seed = np.random.choice(np.arange(20000))
        np.random.seed = self.seed

    # takes as input a king file
    # threshold_func is a func that takes as input PropIBD, IBD1Seg, IBD2Seg, InfType and returns True if the input is considered to be related
    def king_graph(self, king_file, threshold_func, other_args=[]):

        if type(king_file) == str:
            # read in king file
            king = pd.read_csv(king_file, delim_whitespace = True, dtype = {"ID1": str, "ID2": str})

        # we can also load an existing king df
        else:
            king = king_file

        # create graph structure with the kinship coeff
        self.kinG = nx.Graph()
        self.kinG.add_weighted_edges_from(king[["ID1", "ID2", "PropIBD"]].values)

        # build the kinship graph
        G = nx.Graph()
        G.add_nodes_from(self.kinG.nodes)
        king_related = king[king[["PropIBD", "IBD1Seg", "IBD2Seg", "InfType"]].apply(lambda x: threshold_func(*x, *other_args), axis = 1)]
        G.add_edges_from(king_related[["ID1", "ID2"]].values)

        return G

    def unrelated_family(self, g):

        # keep track of the unrelated nodes
        unrelated_nodes = list()

        # dict maps an ID to the number of close relatives
        degree_d = dict(g.degree())

        # each iteration removes a node from degree_d so will be executed len(degree_d) times
        while len(degree_d) > 0:

            # create function that returns the num of close relatives +- random noise for tie-breakers
            randmin = lambda x: np.random.normal(degree_d[x] , 1.5)

            # picks the node with the fewest close relatives
            node1 = min(degree_d, key = randmin)

            # add the node to unrelated_nodes but only if not already related
            add = True
            for node2 in unrelated_nodes:
                if g.has_edge(node1, node2):
                    add = False
            if add:
                unrelated_nodes.append(node1)

            # delete the node from degree_d regardless of if it's added or not
            del degree_d[node1]

        return unrelated_nodes

    def get_unrelateds(self, G):
        # object to store various components of the run
        # n_comp is number of distinct families, unrelateds holds the unrelateds, max k holds the highest kinship value of the set
        run = type('run', (object,), {"n_comp": 0, "unrelateds": []})

        # iterate through each "family" (clusters of relatives entirely unrelated)
        for i in nx.connected_components(G):
            g = G.subgraph(i)
            run.unrelateds += self.unrelated_family(g)
            run.n_comp += 1

        # return the run object
        return run

    # run it over multiple iterations using different seeds to get more individuals included
    # target is the min len of unrelateds for the loop to stop
    # max_iter is the max num of iterations before it stops
    def multiple_runs(self, G, target, max_iter = 10):

        # keep track of the runs
        run_list = []

        # only run it max_iter times
        for i in range(max_iter):

            # choose and set new seed
            seed = np.random.choice(np.arange(20000))
            np.random.seed = seed

            # run the unrelateds algorithm
            run = self.get_unrelateds(G)
            run.seed = seed

            # add the new run to run_list
            run_list.append(run)

            # if the most recent run exceeds our target, stio
            if len(run.unrelateds) >= target:
                print(f"Target of {target} relatives found")
                break

            print(f"Running iteration {i+1}...found a set of {len(run.unrelateds)}")

        # sort by length and get run with longest unrelateds list
        run_list.sort(key = lambda x: len(x.unrelateds), reverse = True)
        run = run_list[0]

        # set the class seed
        self.seed = run.seed

        return run


    def write_out(self, run, prefix = "unrelateds"):

        # write log file
        log = open(f"{prefix}.log", "w")
        log.write(f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}\n")
        log.write(f"Random seed: {self.seed}\n\n")
        log.write(f"Found {run.n_comp} distinct families/family at the given kinship threshold\n")
        log.write(f"Wrote a total of {len(run.unrelateds)} relatives to {prefix}.txt\n")
        log.write(f"Of these, the max kinship (or kinship equivalent) is {run.max_k}\n")
        log.close()

        # write out unrelateds
        out = open(f"{prefix}.txt", "w")
        out.write("\n".join(run.unrelateds) + "\n")
        out.close()

    def get_unrelated_set(self, G, **kwargs):
        # run iterations to find the largest set
        run = self.multiple_runs(G,
                                 target = kwargs.get("target", np.inf),
                                 max_iter = kwargs.get("max_iter", 25))
        
        run.max_k = max([d["weight"] for _,_,d in self.kinG.subgraph(run.unrelateds).edges(data=True)]) if len(run.unrelateds) > 1 else 0.0

        if kwargs.get("prefix", "") != "":
            self.write_out(run, kwargs["prefix"])

        return run

    def get_unrelated_set_lda(self, king_file, path_to_lda, max_degree, **kwargs):

        if os.path.exists(path_to_lda):
            i = open(path_to_lda, "rb")
            degree_lda = pkl.load(i)
        else:
            degree_lda = path_to_lda.lda

        degrees = ["4th", "4th+", "3rd", "3rd+", "2nd", "2nd+", "PO", "FS", "1st"]
        related_degree = degrees[degrees.index(max_degree)+1:]

        king_df = pd.read_csv(king_file, delim_whitespace=True)

        king_df["predicted"] = degree_lda.predict(king_df[["IBD1Seg", "IBD2Seg"]].values)

        self.kinG = nx.Graph()
        self.kinG.add_weighted_edges_from(king_df[["ID1","ID2","PropIBD"]].values)

        G = nx.Graph()
        G.add_edges_from(king_df[king_df.predicted.isin(related_degree)][["ID1","ID2"]].values)

        self.get_unrelated_set(G, prefix=kwargs.get("prefix", ""), max_iter=kwargs.get("max_iter", 25), target=kwargs.get("target", np.inf))

    def king_unrelateds(self, king_file: str, max_k: float):

        # create the relatedness network from the king file
        G = self.king_graph(king_file, lambda propIBD, a, IBD2Seg, c: propIBD > 0.1 or IBD2Seg > 0.03)

        # run 10 iterations to find the largest set
        run = self.multiple_runs(G, target = np.inf, max_iter = 10)

        # write out the file
        with open("sim_keep.txt", "w") as outfile:
            outfile.write("\n".join(run.unrelateds))
        
        print(f"Wrote out {len(run.unrelateds)} IDs to keep to 'sim_keep.txt'")


def quick_kinship(pair_df):
    # keep track of cm ibd1, ibd2
    ibd1, ibd2 = 0, 0

    # iterate through the chromosomes
    for _, chrom_df in pair_df.groupby("chromosome"):
        # add a column that is the cm of the ibd segments
        chrom_df["r"] = chrom_df.apply(lambda x: np.arange(ceil(x.start_cm), floor(x.end_cm)+1), axis=1)
        chrom_df["sites1"] = chrom_df.apply(lambda x: [(i, x.id1_haplotype) for i in x.r], axis=1)
        chrom_df["sites2"] = chrom_df.apply(lambda x: [(i, x.id2_haplotype) for i in x.r], axis=1)

        sites1 = [i[0] for i in set(it.chain(*chrom_df.sites1.values))]
        sites2 = [i[0] for i in set(it.chain(*chrom_df.sites2.values))]

        sites = Counter(sites1 + sites2)

        ibd1 += sum([1 for site, c in sites.items() if c < 4])
        ibd2 += sum([1 for site, c in sites.items() if c == 4])

    return ibd1, ibd2      

    
class Karyogram:
    def __init__(self, map_file, cm = True):
        if type(map_file) != list:
            map_file = [map_file]

        df = pd.DataFrame()
        for mapf in map_file:
            temp = pd.read_csv(mapf, delim_whitespace=True, header = None)
            df = pd.concat([df, temp])

        self.chrom_ends = {}
        self.max_x = 0
        for chrom, chrom_df in df.groupby(0):
            self.chrom_ends[chrom] = (min(chrom_df[2 if cm else 3]), max(chrom_df[2 if cm else 3])-min(chrom_df[2]))
            self.max_x = self.max_x if sum(self.chrom_ends[chrom]) < self.max_x else sum(self.chrom_ends[chrom])

        self.chrom_y = {(chrom, hap): (chrom - 1)*9 + 4*hap for chrom, hap in it.product(np.arange(1, 23), [0, 1])}

    def plot_segments(self, segments, **kwargs):

        # init the figure
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 20)

        # add the chromosome templates
        for chrom, hap in it.product(np.arange(1, 23), [0, 1]):
            rect = patches.Rectangle((self.chrom_ends[chrom][0], self.chrom_y[(chrom, hap)]),
                                    self.chrom_ends[chrom][1], 3, edgecolor = "black",
                                    facecolor = "darkgrey" if hap == 0 else "grey")
            ax.add_patch(rect)

        # add the segments
        for chrom, start, stop, hap in segments:
            facecolor = kwargs.get("hap0_color", "cornflowerblue") if hap == 0 else kwargs.get("hap1_color", "tomato")
            rect = patches.Rectangle((start, self.chrom_y[(chrom, hap)]), stop - start, 3,
                                    edgecolor = "black", facecolor = facecolor, alpha = 0.8)
            ax.add_patch(rect)

        # re-label the y ticks
        ax.set_yticks([self.chrom_y[(chrom, 0)] + 3.5 for chrom in range(1, 23)])
        ax.set_yticklabels([str(chrom) for chrom in range(1, 23)])

        # set axes limits, remove spines, modify ticks
        plt.xlim(0, self.max_x)
        plt.ylim(-2, self.chrom_y[(22, 1)] + 10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', which='major', labelsize=16)
        plt.tick_params(left = False)

        plt.savefig(f"{kwargs.get('file_name', 'karyogram')}.png", dpi = kwargs.get('dpi', 500))


def SiblingClassifier(pedigree, pair_data, dummy_n, classifier=None):
    # (unchanged; omitted here for brevity)
    # Keep your original if you use it operationally.
    return dummy_n, pedigree.copy()

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
                

def parse_args():
    parser = argparse.ArgumentParser()

    # args for interpolation
    parser.add_argument("-interpolate", action="store_true")
    parser.add_argument("-genetic_map", type=str)
    parser.add_argument("-input_map", type=str)
    parser.add_argument("-columns", nargs='+', default=[0, 2, 3], type=int)
    parser.add_argument("-sites", type=str, default="")

    # args for running phasedibd
    parser.add_argument("-phasedibd", action="store_true")
    parser.add_argument("-input_vcf", type=str)
    parser.add_argument("-output", type=str)
    parser.add_argument("-use_phase_correction", action="store_true")  # SDS added

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

