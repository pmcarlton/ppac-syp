# functions for calculating various things with embeddings
# particularly, similarity bewteen proteins
# 20250524pmc

import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import ruptures as rpt
import umap


def get_protein_segments(emb_dict, max_bkps_per100aa=3):
    ## from A. Moses bioRχiv paper 2025-03
    """
    Define the boundaries of protein segments using change point analysis.

    Parameters:
    emb_dict: a dictionary of UniProt protein IDs as keys and ProtT5 per-residue embeddings as values
    max_bkps_per100aa: an integer that represents the maximum number of boundaries per 100 amino acids in the protein

    Returns:
    protein_segments: a dictionary of Uniprot protein IDs as keys and a list of boundaries as values
    """

    protein_segments = {}

    # for each protein and embedding
    for protein, emb in emb_dict.items():
        n_boundaries = max(int(emb.shape[0] * max_bkps_per100aa / 100), 1)

        # try to get the breakpoints from change point analysis
        try:
            alg = rpt.Window(width=30, model="rbf", jump=1).fit(emb)
            # n_bkps is the maximum number of breakpoints allowed in this protein
            boundaries = alg.predict(n_bkps=n_boundaries)
            # convert boundaries to segments
            my_segments = [[0, boundaries[0]]]
            my_segments.extend(
                [[boundaries[ii], boundaries[ii + 1]] for ii in range(len(boundaries) - 1)]
            )
        # this algorithm can fail if the maximum number of breakpoints is unreasonable
        except:
            print("Failed", protein, "number of boundaries", n_boundaries)
            my_segments = "Failed"
            raise

        protein_segments[protein] = my_segments

    return protein_segments


def get_protein_segment_embeddings(emb_dict, protein_segment_boundaries):
    ## from A. Moses bioRχiv paper 2025-03
    """
    Calculates the segment embedding for each segment defined by the change point analysis

    Parameters:
    emb_dict: a dictionary of UniProt protein IDs as keys and ProtT5 per-residue embeddings as values
    protein_segment_boundaries: a dictionary of Uniprot protein IDs as keys and a list of boundaries as values

    Returns:
    protein_segment_embeddings: a dictionary of "ID start-stop" as keys and segment embeddings as values
        IDs are UniProt protein IDs, start and stop corresponds to the positions of the segment in the
        original protein sequence, and segment embeddings are vectors of size 1x1024
    """

    protein_segment_embeddings = {}

    # for each protein with an embedding
    for protein, emb in emb_dict.items():
        # if the segmentation did not fail
        if protein_segment_boundaries[protein] != "Failed":
            # generate a "segment embedding" for each segment of the protein
            for segment in protein_segment_boundaries[protein]:
                seg_emb = np.mean(emb[segment[0] : segment[1], :], axis=0)
                key = protein + " " + str(segment[0]) + "-" + str(segment[1])

                protein_segment_embeddings[key] = seg_emb

    return protein_segment_embeddings


def plot_embed_segments(filepath):
    # filepath="iho1_minizoo_per_residue_embeddings.h5"
    with h5py.File(filepath, "r") as iho:
        els = {}
        for x in iho:
            els[x] = np.array(iho[x])

    ihosegs = get_protein_segments(els)
    ihosegembs = get_protein_segment_embeddings(els, ihosegs)

    pnl = []
    v = np.zeros((78, 1024))
    o = 0
    for pn, em in ihosegembs.items():
        pnl.append(pn)
        v[o] = em
        o = o + 1

    reducer = umap.UMAP(metric="cosine")
    vu = reducer.fit_transform(v)
    pnlab = []
    for i in pnl:
        pnlab.append(np.mean([int(x) for x in i.split(" ")[-1].split("-")]))
    px.scatter(x=vu[:, 0], y=vu[:, 1], hover_name=pnl, color=pnlab)


def cdist(x, y):
    # normalized dot product
    r = np.dot(x, y.T)
    r = r / np.sqrt(np.sum(x**2, axis=1))[:, np.newaxis]
    r = r / np.sqrt(np.sum(y**2, axis=1))[np.newaxis, :]
    return r


def load_embeds(filepath="iho1_minizoo_per_residue_embeddings.h5"):
    with h5py.File(filepath, "r") as iho:
        eli = []
        ell = []
        for x in iho:
            ell.append(x)
            eli.append(np.array(iho[x]))
    return (eli, ell)


def get_emb_by_match(uid, filepath):
    with h5py.File(filepath) as fh:
        kl = list(fh.keys())
        zk = [x for x in kl if uid in x]
        print(zk)
        emb = {}
        for i in zk:
            emb[i] = np.array(fh[i].astype("float32"))
    return emb


def get_embs_by_match(uids, filepath):
    # supply uids as list
    with h5py.File(filepath) as fh:
        kl = list(fh.keys())
        zk = []
        for theuid in uids:
            zk.append([x for x in kl if theuid in x])
        if len(zk) > 1:
            zk = [x for item in zk for x in item]  # flatten if needed
        emb = {}
        for i in zk:
            emb[i] = np.array(fh[i].astype("float32"))
    return emb


def do_one_mean_max_comparison(query, ref):
    # query and ref are both per-residue embeddings
    cdistvals = cdist(query, ref)
    mean0 = np.mean(np.max(cdistvals, axis=0))
    mean1 = np.mean(np.max(cdistvals, axis=1))
    return max(mean0, mean1)


def do_full_mean_max_comparison(filepath, emblist):
    # highly customized example, with a file of embeddings and assuming you already have embeddings in a list
    embl = len(emblist)
    embdict = [{} for x in range(embl)]
    with h5py.File(filepath) as fh:
        fk2 = list(fh.keys())
        for i in fk2:
            for e in range(embl):
                embdict[e][i] = do_one_mean_max_comparison(np.array(fh[i]), emblist[e])
    sypdfm = pd.DataFrame.from_dict({i: embdict[i] for i in range(len(embdict))}, orient="columns")
    return sypdfm
