import numpy as np
import pandas as pd
import pickle as pkl
import networkx as nx
import argparse
import os
from collections import namedtuple
import itertools as it
import yaml
from sklearn.model_selection import LeaveOneOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from copy import deepcopy

from pedigree_tools import ProcessSegments, Pedigree, PedigreeHierarchy, introduce_phase_error, Classifier, RemoveRelateds
from ibd_tools import load_haproh_roh, annotate_segments_with_roh
''' Ponderosa can take as input a file (e.g., containing IBD segments) that has all chromosomes
represented, in which case it does not expect chr1 to be in the file name. Otherwise, if the files
are split by chromosome, the user should supply the path and file name for chromosome 1, in which
case the chromosome number should be denoted as chr1'''
def get_file_names(file_name):

    if "chr1" in file_name:
        return [file_name.replace("chr1", f"chr{chrom}") for chrom in range(1, 23)]
    else:
        return [file_name]
    
def load_fam(fam_file):
    ### load the fam file
    fam = pd.read_csv(fam_file, delim_whitespace=True, dtype=str, header=None)

    ### get new dataframe of the parents and fathers
    tmp_df_list = []
    for sex, col in zip(["1", "2"], [2, 3]):
        tmp = fam[fam[col]!="0"][[0, col]]
        tmp[4] = sex; tmp[5] = "-9"; tmp[1] = tmp[col]
        tmp_df_list.append(tmp[[0, 1, 4, 5]])

    parent_df = pd.concat(tmp_df_list)

    return pd.concat([fam, parent_df]).drop_duplicates(1, keep="first").fillna("0").reset_index(drop=True)
    


''' SampleData is a class that holds a networkx graph where nodes are individuals
and edges describe the relationship between those individuals. It can be queried easily
using the get_edges and get_nodes functions, which allow the user to supply functions
that act as conditionals for returning node pairs and nodes, resp.'''
class SampleData:

    def __init__(self, fam_file, **kwargs):

        ### load the fam file
        fam = load_fam(fam_file)

        ### the default information for adding a person
        self.default = {"sex": 0, "mother": -1, "father": -1, "children": [], "age": np.nan, "popl": "pop1"}

        ### init the graph and set defaults
        g = nx.Graph()
        g.add_nodes_from(fam.apply(lambda x: (x[1], {"sex": x[4],
                                                    "mother": x[3] if x[3] != "0" else -1,
                                                    "father": x[2] if x[2] != "0" else -1,
                                                    "children": [],
                                                    "age": np.nan,
                                                    "popl": "pop1"}), axis=1).values)

        ### get the population
        popln = kwargs.get("population", "pop1")

        # add the children
        for sex_col in [2, 3]:
            # assume that missing parent is coded as 0
            for parent, children_df in fam[fam[sex_col] != "0"].groupby(sex_col):
                # get a list of the children
                children = children_df[1].values.tolist()
                # add to attribute
                nx.set_node_attributes(g, {parent: children}, "children")

        # add the population data
        pop_file = kwargs.get("pop_file", "")
        if os.path.exists(pop_file):
            pops = pd.read_csv(pop_file, delim_whitespace=True, dtype=str, header=None)
            pop_attrs = {iid: pop for iid, pop in pops[[0,1]].values}
            nx.set_node_attributes(g, pop_attrs, "popl")
        else:
            print("No population file has been provided.")

        # add age data
        age_file = kwargs.get("age_file", "")
        if os.path.exists(age_file):
            ages = pd.read_csv(age_file, delim_whitespace=True, dtype={0: str, 1: float}, header=None)
            age_attrs = {iid: age for iid, age in ages[[0,1]].values}
            nx.set_node_attributes(g, age_attrs, "age")
        else:
            print("No age data has been provided.")

        ### get the genome len
        map_file = kwargs.get("map_file", "")
        if os.path.exists(map_file):
            map_files = get_file_names(map_file)

            map_df = pd.concat([pd.read_csv(filen, delim_whitespace=True, header=None) for filen in map_files])

            genome_len = 0
            for _, chrom_df in map_df.groupby([0]):
                genome_len += (chrom_df.iloc[-1][2] - chrom_df.iloc[0][2])
        # no map files provided, use default genome length
        else:
            print("No map files have been provided. Assuming genome length of 3545 cM.")
            # DELETE
            genome_len = 3542.241486670362#3545
                # store the genome_len

        self.genome_len = genome_len
        
        # --- aDNA additions: optional ROH inputs & thresholds ---
        self.roh_file = kwargs.get("roh_file", "")
        self.roh_min_cm = kwargs.get("roh_min_cm", 8.0)              # default 8 cM
        self.roh_overlap_frac = kwargs.get("roh_overlap_frac", 0.5)  # default 50%
        self.roh_dict = None
        if os.path.exists(self.roh_file):
            try:
                self.roh_dict = load_haproh_roh(self.roh_file)
                print(f"Loaded ROH tracts from {self.roh_file}")
            except Exception as e:
                print(f"Warning: failed to load ROH file {self.roh_file}: {e}")
  
        ### load the king file
        king_file = kwargs.get("king_file", "")
        mz_twins = set()
        if os.path.exists(king_file):
            master_hier = PedigreeHierarchy("tree_codes.yaml")

            king_df = pd.read_csv(king_file, delim_whitespace=True, dtype={"ID1": str, "ID2": str})

            # get set of MZ twins; remove the 2nd twin in the pair
            mz_twins = set(king_df[king_df.InfType=="Dup/MZ"]["ID2"].values.tolist())

            # create an IBD graph and add var attrs to it
            g.add_edges_from(king_df.apply(lambda x: [x.ID1, x.ID2, 
                                                {"k_prop": x.PropIBD,
                                                "k_ibd1": x.IBD1Seg,
                                                "k_ibd2": x.IBD2Seg,
                                                "k_degree": x.InfType,
                                                "probs": deepcopy(master_hier)}],
                                                axis=1).values)

        ### subset to only samples from the population and who are not part of the twin set
        popl_samples = {nodes for nodes, attrs in g.nodes(data=True) if attrs.get("popl", "")==popln}
        self.g = g.subgraph(popl_samples - mz_twins)


        ### load ibd, process ibd segs
        ibd_file = kwargs.get("ibd_file", "")
        if os.path.exists(ibd_file):
            print("Processing IBD segments...")
            ibd_files = get_file_names(ibd_file)
            # load ibd files
            ibd_df = pd.concat([pd.read_csv(filen, delim_whitespace=True, dtype={"id1": str, "id2": str})
                                for filen in ibd_files])
            # --- aDNA additions: annotate IBD segments that fall inside ROH ---
            if self.roh_dict is not None:
                if not {"start_cm","end_cm"}.issubset(set(ibd_df.columns)):
                    print("Warning: cannot annotate ROH (missing start_cm/end_cm in IBD). Skipping ROH annotation.")
                else:
                    ibd_df = annotate_segments_with_roh(
                        ibd_df,
                        self.roh_dict,
                        roh_min_cM=self.roh_min_cm,
                        overlap_frac=self.roh_overlap_frac
                    )
            ibd_df["l"] = ibd_df["end_cm"] - ibd_df["start_cm"]

            # iterate through the pairs of individuals, compute ibd stats
            for (id1, id2), pair_df in ibd_df.groupby(["id1", "id2"]):

                # only compute hsr, other stats for 3rd+ degree relatives
                k = self.g.get_edge_data(id1, id2, {}).get("k_prop", 0)

                # 2^-3.5 is the lower limit for 3rd degree relatives
                if k < 2**-3.5:
                    continue

                # compute the IBD data
                pair_ibd = ProcessSegments(pair_df)
                ibd_data = pair_ibd.ponderosa_data(genome_len, inter_phase=False)

                # add phase errors
                pair_ibd_pe = ProcessSegments(introduce_phase_error(pair_df, 50))
                ibd_data_pe = pair_ibd_pe.ponderosa_data(genome_len, inter_phase=False)

                # set up the pedigree hierarchy, which will store the probs and info on how the probs were computed
                hier = deepcopy(master_hier)
                # TODO implement this somewhere else
                # hier.update_attr_from([[node, "ordering", sorted([id1, id2])] for node in hier.init_nodes])
                # hier.update_attr_from([[node, "ordering_method", "sorted"] for node in hier.init_nodes])

                cur_edge_data = self.g.get_edge_data(id1, id2)


                cur_edge_data["ibd1"] = ibd_data.ibd1
                cur_edge_data["ibd2"] = ibd_data.ibd2
                cur_edge_data["h"] = {id1: ibd_data.h1, id2: ibd_data.h2}
                cur_edge_data["h_error"] = {id1: ibd_data_pe.h1, id2: ibd_data_pe.h2}
                cur_edge_data["n"] = ibd_data.n
                cur_edge_data["probs"] = hier


                # # add ibd1 data and initialze probs
                # self.g.add_edge(id1, id2, ibd1=ibd_data.ibd1,
                #                      ibd2=ibd_data.ibd2,
                #                      h={id1: ibd_data.h1, id2: ibd_data.h2},
                #                      h_pe={id1: ibd_data_pe.h1, id2: ibd_data_pe.h2},
                #                      n=ibd_data.n,
                #                      k=(ibd_data.ibd1/2 + ibd_data.ibd2),
                #                      probs=hier,
                #                      segments=pair_df)

        else:
            print("No IBD files have been provided.")


    # returns a list of node pair edges
    # optional func arg is a function that takes as input the data dict and returns true if wanted
    def get_edges(self, f=lambda x: True):
        return [(id1, id2) for id1, id2, data in self.g.edges(data=True) if f(data)]

    # returns a nested np array of [id1, id2, attr1, attr2, ...] for all edges that pass the function
    def get_edge_data(self, attr_list, f=lambda x: True):
        edges, out_attrs = [], []
        for id1, id2, data in self.g.edges(data=True):
            if f(data):
                edges.append([id1, id2])
                out_attrs.append([data[attr] for attr in attr_list])
        return np.array(edges, dtype=str), np.array(out_attrs)

    # updates a bunch of edges at once
    # edge_list looks like [(A, B), (B, C), (D, E)]
    # attr_list looks like [1, 2, 3]
    def update_edges(self, attr_list, attr_name):

        attr_dict = {(id1, id2): attr for (id1, id2), attr in attr_list}
        nx.set_edge_attributes(self.g, attr_dict, attr_name)

    # same as above but returns a set of nodes
    # e.g., get_nodes(lambda x: x["age"] > 90) returns all nodes that are >90 yo
    def get_nodes(self, f=lambda x: True):
        return [id1 for id1, data in self.g.nodes(data=True) if f(data)]
    
    def get_subset(self, f):
        # get the nodes based on the conditional f
        node_subset = self.get_nodes(f)
        # return the subset for which the conditional is true
        return self.g.subgraph(set(node_subset))
    
    # subsets or returns a subset of the g based on a iterable of edges; inplace=True means it will modify self.g
    def edge_subset(self, edges, inplace):
        if inplace:
            self.g = self.g.edge_subgraph(edges).copy()

        else:
            return self.g.edge_subgraph(edges).copy()

    
    def to_dataframe(self, edges, include_edges):
        if len(edges)==0 and include_edges==False:
            return nx.to_pandas_edgelist(self.g, source="id1", target="id2")
        # make copy of the graph
        tmp = self.g.copy()
        # subgraph of the edges
        if include_edges:
            tmp = tmp.edge_subgraph(edges)
        # subgraph of the complement of the edges
        else:
            tmp.remove_edges_from(edges)
        return nx.to_pandas_edgelist(tmp, source="id1", target="id2")

    # update the hier probabilities
    def update_probs(self, probs_list, prob_labs, method, prob_objs):
        # iterate through the probabilities
        for probs, prob_obj in zip(probs_list, prob_objs):
            prob_obj.set_attrs({i: p for i,p in zip(prob_labs, probs)}, "p_con")
            prob_obj.set_attrs({i: method for i in prob_labs}, "method")

    def node_attr(self, node, attr, nan=np.nan):
        try:
            return self.g.nodes[node][attr]
        except:
            return nan


class Classifiers:
    def __init__(self, pairs):
        # store the training data for each classifier; purpose is to allow leave one out training
        self.training = {}

        ### Train the degree classifier
        degree_train = pairs.get_pair_df("relatives").dropna(subset=["k_ibd1"])

        self.training["degree"] = [np.array(degree_train[["k_ibd1", "k_ibd2"]].values.tolist()),
                                   np.array(degree_train["degree"].values.tolist())]
        
        ### Train the hap classifier
        hap_train = pairs.get_pair_df_from(["HS", "GPAV"]).dropna(subset=["h"])

        # get the phase error classifier
        X_train = hap_train.apply(lambda x: [x.h_error[x.pair[0]], x.h_error[x.pair[1]]], axis=1).values.tolist()
        y_train = ["Phase error" for _ in X_train]

        # now add the actual haplotype scores
        X_train += hap_train.apply(lambda x: [x.h[x.pair[0]], x.h[x.pair[1]]], axis=1).values.tolist()
        y_train += hap_train["requested"].values.tolist()

        self.training["hap"] = [np.array(X_train), np.array(y_train)]

        ### Train the degree classifier
        n_train = pairs.get_pair_df_from(["MGP", "MHS", "PHS", "PGP", "AV"]).dropna(subset=["n"])

        # also train on the kinship coefficient
        n_train["ibd_cov"] = n_train.apply(lambda x: x.k_ibd2 + x.k_ibd1, axis=1)

        self.training["n"] = [np.array(n_train[["ibd_cov", "n"]].values.tolist()), np.array(n_train["requested"].values.tolist())]

        self.loo = LeaveOneOut()

    # if n samples in X_train, will train the classifier n times, leaving a different sample out each time
    def leaveOneOut(self, X_train, y_train, func, labels):
            return_probs = []; return_labels = []
            lda = LinearDiscriminantAnalysis()

            for train, test in self.loo.split(X_train):

                lda.fit(X_train[train], y_train[train])

                # predict
                return_probs.append(func(lda, X_train[test])[0])
                return_labels.append(lda.classes_)

            # just predicting the label; don't need the probabilities
            if labels:
                return return_probs, iter(return_labels)

            return return_probs

    def return_classifier(self, classif):

        X_train, y_train = self.training[classif]

        lda = LinearDiscriminantAnalysis()

        lda.fit(X_train, y_train)

        return lda
    
    # returns an array of the probabilities and an iterator of the classes that each probability describes
    def predict_proba(self, classif, X=[]):

        # get the training data for the classifier
        X_train, y_train = self.training[classif]

        # no training data supplied --> perform leave one out
        if len(X)==0:
            return self.leaveOneOut(X_train, y_train, lambda lda, X: lda.predict_proba(X), labels=True)

        lda = self.return_classifier(classif)
 
        return lda.predict_proba(X), it.cycle([lda.classes_])

    # predicts just the label
    def predict(self, classif, X=[]):
        lda = LinearDiscriminantAnalysis()
        X_train, y_train = self.training[classif]

        # perform leave-one-out
        if len(X)==0:
            return self.leaveOneOut(X_train, y_train, lda, lambda lda, X: lda.predict(X), labels=False)

        lda = self.return_classifier(classif)

        return lda.predict(X)
    
    def write_pkl(self, classif, output):

        lda = self.return_classifier(classif)

        i = open(output, "wb")
        pkl.dump(lda, i)
        i.close()

### converts the LDA loaded from the sims to the Classifier class. This is a TODO!!!!
def convert2Classifier(lda, name):
    classif = Classifier(None, None, None, name, lda)




def trainClassifiers(pairs, training):

    if os.path.exists(training):
        out_classf = []
        for classf_name in ["degree", "hap", "nsegs"]:
            i = open(training.replace("degree", classf_name), "rb")
            lda = pkl.load(i)
            name = {"nsegs": "no. of segments", "hap": "haplotype"}.get(classf_name, classf_name)
            out_classf.append(Classifier(None, None, None, name, lda))
        return out_classf

    degree_train = pairs.get_pair_df("relatives").dropna(subset=["k_ibd1"])

    ### Train the degree classifier
    degree_classif = Classifier(X=degree_train[["k_ibd1", "k_ibd2"]].values,
                                y=degree_train["degree"].values,
                                ids=degree_train["pair"].values,
                                name="degree")
    
    ### Train the hap classifier
    hap_train = pairs.get_pair_df_from(["HS", "GPAV"]).dropna(subset=["h"])

    # get the phase error classifier
    X_train = hap_train.apply(lambda x: [x.h_error[x.pair[0]], x.h_error[x.pair[1]]], axis=1).values.tolist()
    y_train = ["Phase error" for _ in X_train]

    # now add the actual haplotype scores
    X_train += hap_train.apply(lambda x: [x.h[x.pair[0]], x.h[x.pair[1]]], axis=1).values.tolist()
    y_train += hap_train["requested"].values.tolist()

    hap_classif = Classifier(X=np.array(X_train),
                             y=np.array(y_train),
                             ids=[("0","0") for _ in range(hap_train.shape[0])] + hap_train["pair"].values.tolist(),
                             name="haplotype")
    
    ### Train the nsegs classifier
    n_train = pairs.get_pair_df_from(["MGP", "MHS", "PHS", "PGP", "AV"]).dropna(subset=["n"])
    # also train on the kinship coefficient
    n_train["ibd_cov"] = n_train.apply(lambda x: x.k_ibd2 + x.k_ibd1, axis=1)

    n_classif = Classifier(X=n_train[["ibd_cov", "n"]].values,
                           y=n_train["requested"].values,
                           ids=n_train["pair"].values,
                           name="no. of segments")
    
    return degree_classif, hap_classif, n_classif




class ResultsData:
    def __init__(self, samples, pairs, classifiers, df=pd.DataFrame()):

        self.samples = samples
        self.pairs = pairs
        self.df = self.samples.to_dataframe([], include_edges=False) if df.shape[0]==0 else df
        self.classifiers = classifiers


    def get_classifier(self, name):
    # order of self.classifiers is [degree, hap, n]
    return self.classifiers[{"degree_class": 0, "hap_class": 1, "segments_class": 2}[name]]


    # writes out a human readable output; can specificy the columns
    def write_readable(self, output, **kwargs):

        cols = kwargs.get("columns",
                          ["id1", "id2", "k_ibd1", "k_ibd2", "most_probable", "probability", "degree"])
        
        if "h1" in cols:
            self.df["h1"], self.df["h2"] = zip(*self.df.apply(lambda x: [x.h[x.id1], x.h[x.id2]], axis=1))

        if "degree" in cols:
            self.df["degree"] = self.df["probs"].apply(lambda x: x.degree_nodes[np.argmax([x.hier.nodes[node]["p"] for node in x.degree_nodes])])

        self.df[cols].to_csv(output, index=False, sep="\t")

    # creates the dataframe
    def to_dataframe(self, edges, include_edges, inplace=False):

        tmp = self.samples.to_dataframe(edges, include_edges)

        if inplace:
            self.df = tmp
        else:
            return tmp

    # takes as input a function that works on the dataframe and subsets it
    def subset_dataframe(self, func, inplace=False):
        tmp = self.df[self.df.apply(lambda x: func(x), axis=1)]

        if inplace:
            self.df = tmp
        else:
            return tmp
        
    def subset_samples(self, func):

        self.subset_dataframe(func, inplace=True)

        self.samples.edge_subset(self.df[["id1", "id2"]].apply(tuple, axis=1).values, inplace=True)

    # sets the min prob for readable format; can be rerun with new probs
    def set_min_probability(self, min_p, update_attrs=False):
        self.df["most_probable"], self.df["probability"] = zip(*self.df["probs"].apply(lambda x: x.most_probable(min_p)))

        if update_attrs:
            edge_keys = [tuple(x) for x in self.df[["id1", "id2"]].values]
            mpairs = list(zip(edge_keys, self.df["most_probable"].values))
            ppairs = list(zip(edge_keys, self.df["probability"].values))
            self.samples.update_edges(mpairs, "most_probable")
            self.samples.update_edges(ppairs, "probability")


    # recomputes probs across all rows of df
    def compute_probs(self):
        self.df["probs"].apply(lambda x: x.compute_probs())

    # pickles the object
    def pickle_it(self, output):
        self.df = None
        # self.samples = None
        i = open(output, "wb")
        pkl.dump(self, i)
        i.close()

    def most_likely_among(self, df, nodes, update_attrs=False):
        if df.shape[0]==0:
            df = self.df

        likely_among = df["probs"].apply(lambda x: nodes[np.argmax([x.hier.nodes[node]["p"] for node in nodes])]).values

        if update_attrs:
            self.samples.update_edges(zip(self.df[["id1","id2"]].values, likely_among), "likely_among")

        else:
            return likely_among
        
    def get_evaluation_df(self, relationship_nodes):
        # get the df of pairs and their true relationship
        df = self.pairs.get_pair_df_from(relationship_nodes)[["pair", "requested"]].rename({"requested": "true"}, axis=1)

        tmp_df = self.to_dataframe(df["pair"].values, include_edges=True)
        tmp_df["predicted"] = self.most_likely_among(tmp_df, relationship_nodes)

        return df.merge(tmp_df[["pair", "predicted"]], on="pair")
    
    '''
    1. Mode 1: float
    2. Mode 2: degree of relatedness
    '''
    def get_unrelated_set(self, mode: str, use_king: bool, max_r, king_file, **kwargs):
        if mode not in ["float", "degree"]:
            raise ValueError("Only 'float' and 'degree' modes are allowed!")
        if mode == "float" and type(max_r) != float:
            raise TypeError("float mode specified: max_r must be a float!")
        if mode == "degree" and type(max_r) != str:
            raise TypeError("degree mode specified: max_r must be a string!")
        
        G = nx.Graph()
        G.add_weighted_edges_from(pd.read_csv(king_file, delim_whitespace=True).apply(lambda x: [x.ID1, x.ID2, {"IBD1":x.IBD1Seg, "IBD2":x.IBD2Seg, "k":x.PropIBD, "degree":x.InfType}], axis=1).values)
        
        if mode == "float":
            edges = [(i,j,d["k"]) for i,j,d in G.edges(data=True) if d["k"]>max_r]

        elif mode == "degree":
            deg_order = ["UN", "4th", "4th+", "3rd", "3rd+", "2nd", "2nd+", "PO", "FS", "1st"]
            related_deg = deg_order[deg_order.index(max_r)+1:]

            if use_king:
                edges = [(i,j,d["k"]) for i,j,d in G.edges(data=True) if d["degree"] in related_deg]

            else:
                degree_lda = self.get_classifier("degree_class")
                edges = [(i,j,d["k"]) for i,j,d in G.edges(data=True) if degree_lda.predict([[d["IBD1", d["IBD2"]]]])[0] in related_deg]

        g = nx.Graph()
        g.add_weighted_edges_from(edges)

        RR = RemoveRelateds()
        RR.get_unrelated_set(g, lambda x: True, "k", kwargs.get("",))
         
        





class Args:
    # init with the default arguments
    def __init__(self, **kwargs):
        self.ages = ""
        self.map = ""
        self.populations = ""
        self.yaml = ""
        self.pedigree_codes = "pedigree_codes.yaml"
        self.output = "Ponderosa"
        self.min_p = 0.50
        self.population = "pop1"
        self.training = ""
        self.interactive = False
                # --- aDNA / ROH defaults ---
        self.roh = ""                 # path to hapROH TSV
        self.roh_min_cM = 8.0
        self.roh_overlap_frac = 0.5
        self.phased_vcf = ""          # reserved for future hap-side work
        self.ibd_format = "phasedibd" # reserved; current loader expects PONDEROSA-style IBD
        self.ancibd_kind = "tsv"      # reserved
        self.min_seg_cM = 8.0         # reserved; enforce upstream when building IBD
        self.transversions_only = False
        self.phase_error_max = 0.95
        self.po_pairs = ""            # reserved; training hook


        self.update(kwargs)

    def update(self, arg_dict):
        for arg, val in arg_dict.items():
            setattr(self, arg, val)

    # update the arguments with kwargs
    def update_args(self, return_self, **kwargs):
        self.update(kwargs)

        if return_self:
            return self
        
    # given a yaml file with args and their vals, update the args
    def add_yaml_args(self, return_self, yamlf):
        i = open(yamlf)
        yaml_dict = yaml.safe_load(i)
        self.update(yaml_dict)

        if return_self:
            return self

    # add args from the command line
    def add_cli_args(self, return_self, args):
        self.update(vars(args))

        # yaml file exists, override any cur args
        if os.path.exists(self.yaml):
            self.add_yaml_args(return_self, self.yaml)

        if return_self:
            return self
            
def PONDEROSA(samples, args=Args()):

    pedigree = Pedigree(samples=samples, pedigree_file="pedigree_codes.yaml", tree_file="tree_codes.yaml")
    pairs = pedigree.find_all_relationships()

    # trains the classifiers
    degree_classif, hap_classif, n_classif = trainClassifiers(pairs, args.training)

    # get data frame of the unknown 
    unknown_df = samples.to_dataframe([], include_edges=False)
    unknown_df = unknown_df[unknown_df.k_degree!="UN"].reset_index(drop=True)
    unknown_df["ibd_cov"] = unknown_df.apply(lambda x: x.ibd1 + x.ibd2, axis=1)

    # get the degree probabilities
    probs, labels = degree_classif.predict_proba(X=unknown_df[["k_ibd1", "k_ibd2"]].values,
                                                ids=unknown_df[["id1", "id2"]].values)

    # add the probabilities to the tree
    for (_, row), prob in zip(unknown_df.iterrows(), probs):
        row["probs"].add_probs(list(zip(labels, prob)), "ibd")

    # subset to second degree relatives
    second_df = unknown_df[unknown_df["probs"].apply(lambda x: x.hier.nodes["2nd"]["p_con"] > 0.2)]

    # get the n_ibd segs classifier probabilities
    probs, labels = n_classif.predict_proba(X=second_df[["ibd_cov", "n"]].values,
                                            ids=second_df[["id1", "id2"]].values)
    # add the probabilities to the tree
    for (_, row), prob in zip(second_df.iterrows(), probs):
        row["probs"].add_probs(list(zip(labels, prob)), "nsegs")

        # for each of these nodes, take the sum of the two children
        for node in ["GP", "GPAV", "HS"]:
            # array of the children probabilities
            child_probs = [row["probs"].hier.nodes[i]["p_con"] for i in row["probs"].hier.successors(node)]
            # add the probability
            row["probs"].add_probs(node, p_con=sum(child_probs), method="nsegs")


    # get the probabilities from the hap score classifier
    probs, labels = hap_classif.predict_proba(X=np.array(second_df.apply(lambda x: sorted([h for _,h in x.h.items()], reverse=True), axis=1).values.tolist()),
                                                ids=second_df[["id1", "id2"]].values)

    for (_, row), prob in zip(second_df.iterrows(), probs):
        # get the index of the Phase error class
        classes = list(labels)
        pe_index = classes.index("Phase error"); del classes[pe_index]

        # the chance of high Phase error is high; do not update the HS or GPAV probabilities
        if prob[pe_index] < 0.2:
            row["probs"].add_probs(list(zip(classes, np.delete(prob, pe_index))), "hap")

    # merge the second degree back in with the unknowns
    unknown_df = pd.concat([unknown_df, second_df]).drop_duplicates(subset=["id1", "id2"], keep="last").reset_index(drop=True)

    samples.edge_subset(unknown_df[["id1", "id2"]].apply(lambda x: tuple(x), axis=1).values, inplace=True)

    relatives_obj = ResultsData(samples=samples,
                                pairs=pairs,
                                classifiers=[degree_classif, hap_classif, n_classif],
                                df=unknown_df)
    relatives_obj.compute_probs()
    relatives_obj.set_min_probability(args.min_p)

    if args.interactive:
        return relatives_obj
    
    relatives_obj.write_readable(f"{args.output}.txt")

    # relatives_obj.subset_samples(lambda x: x.probs.hier.nodes["relatives"]["p"] == 1)
    relatives_obj.pickle_it(f"{args.output}_results.pkl")

def parse_args():
    parser = argparse.ArgumentParser()

    # get the default args
    args = Args()

    # Required file arguments
    parser.add_argument("--ibd", help = "IBD segment file. If multiple files for each chromosome, this is the path for chromosome 1.")
    parser.add_argument("--fam", help="PLINK-formated .fam file")
    parser.add_argument("--king", help="KING .seg file.")

    # Optional file arguments
    parser.add_argument("--ages", help="Age file. First column is the IID, second column is the age", default=args.ages)
    parser.add_argument("--map", help = "PLINK-formatted .map file.", default=args.map)
    parser.add_argument("--populations", help="Path and file name of .txt file where col1 is the IID and col2 is their population.", default=args.populations)
    parser.add_argument("--yaml", help="YAML file containing all arguments (optional). Can be combined with CLI arguments.", default=args.yaml)
    parser.add_argument("--pedigree_codes", default=args.pedigree_codes)
    parser.add_argument("--roh", help="hapROH per-individual ROH TSV", default=args.roh)
    parser.add_argument("--phased_vcf", help="Phased VCF for haplotype-side inference (optional)", default=args.phased_vcf)

    # Other arguments
    parser.add_argument("--output", help = "Output prefix.", default="Ponderosa")
    parser.add_argument("--min_p", help="Minimum posterior probability to output the relationship.", default=args.min_p, type=float)
    parser.add_argument("--population", help="Population name to run Ponderosa on.", default=args.population)
    parser.add_argument("--assess", help="For assessing the performance of Ponderosa.", action="store_true")
    # aDNA / ROH options
    parser.add_argument("--roh_min_cM", type=float, default=args.roh_min_cM)
    parser.add_argument("--roh_overlap_frac", type=float, default=args.roh_overlap_frac)

    # reserved switches for future ancIBD/haplotype-side integrations
    parser.add_argument("--ibd_format", choices=["phasedibd","ancibd"], default=args.ibd_format)
    parser.add_argument("--ancibd_kind", choices=["tsv","hdf5"], default=args.ancibd_kind)
    parser.add_argument("--min_seg_cM", type=float, default=args.min_seg_cM)
    parser.add_argument("--transversions_only", action="store_true", default=args.transversions_only)
    parser.add_argument("--phase_error_max", type=float, default=args.phase_error_max)
    parser.add_argument("--po_pairs", default=args.po_pairs)

    parser.add_argument("--training", help = "Path and name of the 'degree' pkl file for training.", default=args.training)
    parser.add_argument("--debug", action="store_true")

    args.add_cli_args(return_self=False, args=parser.parse_args())
    
    return args

'''
Takes as input a yaml_file and returns an object whose attributes are various Ponderosa arguments
'''
def load_yaml_args(yaml_file, args):

    i = open(yaml_file)
    yaml_dict = yaml.safe_load(i)

    # iterate through the arguments and their values (possibly default)
    for argname, arg in vars(args).items():
        # if the arg is already in the yaml_dict, it will take it, otherwise take the default
        yaml_dict[argname] = yaml_dict.get(argname, arg)

    Args = namedtuple("Args", list(yaml_dict))
    args = Args(*[j for _,j in yaml_dict.items()])
    return args


if __name__ == "__main__":

    args = parse_args()

    # yaml file provided as the argument file
    if os.path.exists(args.yaml):
        args = load_yaml_args(args.yaml, args)

    if args.debug and os.path.exists(f"{args.output}_samples.pkl"):
        print("Samples pkl file provided.")
        i = open(f"{args.output}_samples.pkl", "rb")
        samples = pkl.load(i)
        i.close()

    # Get the samples for Ponderosa input
    else:
               else:
else:
    samples = SampleData(
        fam_file=args.fam,
        king_file=args.king,
        ibd_file=args.ibd,
        map_file=args.map,
        # aDNA / ROH passthrough
        roh_file=args.roh,
        roh_min_cm=args.roh_min_cM,
        roh_overlap_frac=args.roh_overlap_frac
    )

    if args.debug:
        i = open(f"{args.output}_samples.pkl", "wb")
        pkl.dump(samples, i)
        i.close()

    if args.debug:
        i = open(f"{args.output}_samples.pkl", "wb")
        pkl.dump(samples, i)
        i.close()

                )

        if args.debug:
            i = open(f"{args.output}_samples.pkl", "wb")
            pkl.dump(samples, i)
            i.close()

   PONDEROSA(samples=samples, args=args)



'''
Depcrecated
if __name__ == "__main__":

    args = parse_args()

    ### Samples is a SampleData obj that has all node and pairwise IBD information
    if args.debug:
        print("In debug mode.")
        i = open("Samples_debug.pkl", "rb")
        Samples = pkl.load(i)
        i.close()

    else:
        Samples = SampleData(fam_file=args.fam,
                            ibd_file=args.ibd,
                            age_file=args.ages,
                            map_file=args.map,
                            pop_file=args.populations,
                            population=args.population,
                            n_pairs=args.n_pairs)

    ### Training step
    
    # True if there is already trained data; load the pickled training objs
    if os.path.exists(args.training):
        print("Trained classifiers provided.")

        # load the degree classifier
        # proba goes: [2nd, 3rd, 4th, FS]
        infile = open(args.training, "rb")
        degree_classfr = pkl.load(infile)

        # load the n segs classifier
        # probs go ['AV', 'MGP', 'MHS', 'PGP', 'PHS']
        infile = open(args.training.replace("degree", "nsegs"), "rb")
        nsegs_classfr = pkl.load(infile)

        # load the hap classifier
        # probs go ['GP/AV', 'HS', 'PhaseError']
        infile = open(args.training.replace("degree", "hap"), "rb")
        hap_classfr = pkl.load(infile)

    # pedigree object to find existing relationships
    print("Finding pedigree relationships")
    if args.debug:
        i = open("Pedigree_debug.pkl", "rb")
        pedigree = pkl.load(i)
        i.close()
    else:
        try:
            pedigree = Pedigree(Samples.g, degree_classfr)
        except:
            pedigree = Pedigree(Samples.g, None)

        # find relationships
        pedigree.find_relationships()

    # no training provided; generate by self
    if not os.path.exists(args.training):
        print("Trained classifiers not provided. Training classifiers now.")

        # get the training data
        train = TrainPonderosa(real_data=True, samples=Samples, pedigree=pedigree)

        ### haplotype classifier
        hap_classfr = train.hap_classifier()

        ### n segs classifier
        nsegs_classfr = train.nsegs_classifier()

        ### degree classifier
        degree_classfr = train.degree_classifier()

    ### Classification step 1: degree of relatedness
    print("Beginning classification.")

    # get the input data; X has the following cols: id1, id2, ibd1, ibd2
    edges, X_degree = Samples.get_edge_data(["ibd1", "ibd2"], lambda x: (0 < x["k"] < 0.42) or (0.1 < x["ibd2"] < 0.5))
    degree_probs = degree_classfr.predict_proba(X_degree)

    # add the classification probs
    Samples.update_probs(edges, degree_probs, degree_classfr.classes_, "ibd")

    ### Classification step 2: n of segs
    edges, X_n = Samples.get_edge_data(["n", "k"], lambda x: x["probs"].hier.nodes["2nd"]["p"] > 0.80)
    n_probs = nsegs_classfr.predict_proba(X_n)

    # add the n probs
    Samples.update_probs(edges, n_probs, nsegs_classfr.classes_, "segs")

    ### Classification step 3: haps
    edges, X_hap = Samples.get_edge_data(["h"], lambda x: x["probs"].hier.nodes["2nd"]["p"] > 0.80)

    # retrieve h1 and h2 values and predict the probs
    haps = [[h[0][id1], h[0][id2]] for (id1, id2), h in zip(edges, X_hap)]
    h_probs = hap_classfr.predict_proba(haps)

    # update the hap probs
    # get the index of the GPAV and HS classes
    class_index = np.where(hap_classfr.classes_ != "PhaseError")[0]
    Samples.update_probs(edges, h_probs[:,class_index], hap_classfr.classes_[class_index] , "hap")

    ### Inference steps

    # hold the hier objects
    out_hiers = {}

    # iterate through the pairs
    for id1, id2, data in Samples.g.edges(data=True):

        # get the pair data
        pair_data = data.copy()

        # infer the rel if second
        if pair_data["probs"].hier.nodes["2nd"]["p"] > 0.80:
            infer_second(id1, id2, pair_data, Samples.g, 30, 30)

        else:
            pair_data["probs"].set_attrs({i: j.get("p" if i != "relatives" else "post", 1) for i,j in pair_data["probs"].hier.nodes(data=True)}, "post")

        hier = pair_data["probs"].hier

        # add the pair data as a node with attributes
        hier.add_node("pair_data",
                      h=pair_data["h"],
                      n=pair_data["n"],
                      ibd1=pair_data["ibd1"],
                      ibd2=pair_data["ibd2"])
        
        # add the ids as nodes so you can get the pair info
        hier.add_nodes_from(Samples.g.subgraph({id1, id2}).nodes(data=True))

        out_hiers[(id1, id2)] = hier

    # write out a pickle object of the results
    outf = open(f"{args.out}_results.pkl", "wb")
    pkl.dump(out_hiers, outf)
    outf.close()

    # get the readable results df
    results_df = readable_results(out_hiers, args.min_p)
    results_df.to_csv(f"{args.out}_results.txt", sep="\t", index=False)



def infer_second(id1, id2, pair_data, samples,
                mhs_gap, gp_gap):

    ### Re-compute probabilities based on phenotypic data (parents, age, etc)
    hier_obj = pair_data["probs"]
    hier = hier_obj.hier

    ### bool is True if we want to use the haplotype classifier
    use_hap = hier.nodes["HS"]["p"] + hier.nodes["GP/AV"]["p"] >= 0.80

    ### Use phenotype data to set certain relationships to 0

    # get the age gap
    age_gap = abs(samples.nodes[id1]["age"] - samples.nodes[id2]["age"])

    # too far in age to be MHS
    if age_gap > mhs_gap:
        hier.nodes["MHS"]["p"] = 0
        hier.nodes["MHS"]["method"] = "pheno"
        hier.nodes["PHS"]["method"] = "pheno"

    # too close in age to be GP
    if age_gap < gp_gap:
        hier_obj.set_attrs({i: 0 for i in ["PGP", "MGP", "GP"]}, "p")
        hier_obj.set_attrs({i: "pheno" for i in ["PGP", "MGP", "GP"]}, "method")

    # not a paternal relationship
    if samples.nodes[id1]["father"] != None or samples.nodes[id2]["father"] != None:
        # not PHS
        hier.nodes["PHS"]["p"] = 0
        hier.nodes["PHS"]["method"] = "pheno"
        hier.nodes["MHS"]["method"] = "pheno"

    # not MHS bc mother exists
    if samples.nodes[id1]["mother"] != None or samples.nodes[id2]["mother"] != None:
        hier.nodes["MHS"]["p"] = 0
        hier.nodes["MHS"]["method"] = "pheno"
        hier.nodes["PHS"]["method"] = "pheno"

    # re-compute probabilities
    second_nodes = ["AV", "MGP", "PGP", "PHS", "MHS"]
    prob_sum = sum([hier.nodes[i]["p"] for i in second_nodes])
    hier_obj.set_attrs({i: prob_sum and hier.nodes[i]["p"] / prob_sum or 0 for i in second_nodes}, "p")

    ### Done with pheno data

    ### Tier 2 probabilities

    # they are NOT hs; have to be GP/AV
    if hier.nodes["PHS"]["p"] + hier.nodes["MHS"]["p"] == 0:
        hier_obj.set_attrs({"HS": 0, "GP/AV": 1}, "p")
        hier_obj.set_attrs({"HS": "pheno"}, "method")

    # use haplotype data
    if use_hap:
        prob_sum = hier.nodes["HS"]["p"] + hier.nodes["GP/AV"]["p"]
        hier_obj.set_attrs({"HS": hier.nodes["HS"]["p"] / prob_sum if prob_sum > 0 else 0.50,
                            "GP/AV": hier.nodes["GP/AV"]["p"] / prob_sum if prob_sum > 0 else 0.50}, "p")

    # don't use the haplotype data
    else:
        hier_obj.set_attrs({"HS": hier.nodes["PHS"]["p"] + hier.nodes["MHS"]["p"],
                        "GP/AV": hier.nodes["MGP"]["p"] + hier.nodes["PGP"]["p"] + hier.nodes["AV"]["p"]}, "p")
        hier_obj.set_attrs({"HS": "segs", "GP/AV": "segs"}, "method")

    ### Tier 3 probabilities

    # for GP vs. AV
    prob_sum = hier.nodes["MGP"]["p"] + hier.nodes["PGP"]["p"] + hier.nodes["AV"]["p"]
    hier_obj.set_attrs({"GP": (hier.nodes["PGP"]["p"] + hier.nodes["MGP"]["p"])/prob_sum if prob_sum > 0 else 0.50,
                        "AV": hier.nodes["AV"]["p"]/prob_sum if prob_sum > 0 else 0.50}, "p")

    # for MHS versus PHS
    prob_sum = hier.nodes["MHS"]["p"] + hier.nodes["PHS"]["p"]
    hier_obj.set_attrs({"PHS": prob_sum and hier.nodes["PHS"]["p"]/prob_sum or 0.50,
                        "MHS": prob_sum and hier.nodes["MHS"]["p"]/prob_sum or 0.50}, "p")

    ### Tier 4 probabilities

    # for MGP vs PGP
    prob_sum = hier.nodes["MGP"]["p"] + hier.nodes["PGP"]["p"]
    hier_obj.set_attrs({"PGP": hier.nodes["PGP"]["p"]/prob_sum if prob_sum > 0 else 0.50,
                        "MGP": hier.nodes["MGP"]["p"]/prob_sum if prob_sum > 0 else  0.50}, "p")

    hier_obj.set_attrs({"relatives": 1, "2nd": hier.nodes["2nd"]["p"]}, "post")
    # iterate down the tree and compute the posterior prob
    for parent, child in nx.bfs_edges(hier, "2nd"):

        hier_obj.set_attrs({child: hier.nodes[child]["p"] * hier.nodes[parent]["post"]}, "post")

        # need to order according to genetically youngest --> oldest
        if child in ["GP", "MGP", "PGP", "AV", "GP/AV"]:

            # use age
            if samples.nodes[id1]["age"] > samples.nodes[id2]["age"]:
                hier.nodes[child]["ordering"] = (id2, id1)
                hier.nodes[child]["ordering_method"] = "age"
            elif samples.nodes[id2]["age"] > samples.nodes[id1]["age"]:
                    hier.nodes[child]["ordering"] = (id1, id2)
                    hier.nodes[child]["ordering_method"] = "age"  

            # use the haplotype score      
            elif use_hap:
                if samples.edges[id1, id2]["h"][id1] > samples.edges[id1, id2]["h"][id2]:
                    hier.nodes[child]["ordering"] = (id1, id2)
                    hier.nodes[child]["ordering_method"] = "hap"
                else:
                    hier.nodes[child]["ordering"] = (id2, id1)
                    hier.nodes[child]["ordering_method"] = "hap"

def readable_results(results, min_p):

    out_results = []
    for (id1, id2), hier in results.items():

        # compute lengths of all paths from relatives
        paths = nx.single_source_shortest_path_length(hier, "relatives")

        # highly probable rels
        probable_rels = sorted([[l and 1/l or np.inf, rel] for rel, l in paths.items() if hier.nodes[rel]["post"] > min_p])

        # get the deepest rel
        deepest_rel = probable_rels[0][1]
        
        # get the degree of relative
        degree = nx.shortest_path(hier, "relatives", deepest_rel)[1] if deepest_rel != "relatives" else "relatives"
        
        rel_data = hier.nodes[deepest_rel]

        row = list(rel_data["ordering"]) + [rel_data[col] for col in ["ordering_method", "post", "p", "method"]] + [deepest_rel, degree]
        row += [hier.nodes["pair_data"]["ibd1"], hier.nodes["pair_data"]["ibd2"]]
        out_results.append(row)

    results_df = pd.DataFrame(out_results, columns=["id1", "id2", "order_method", "posterior", "conditional", "method", "relationship", "degree", "ibd1", "ibd2"])
    
    return results_df.round(4)

'''
