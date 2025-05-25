# functions for calculating various things with embeddings
# particularly, similarity bewteen proteins
# 20250524pmc

import h5py
import numpy as np
import pandas as pd


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
                # embdict[e]=np.mean(np.max(cdist(np.array(fh[i]),emblist[e]),axis=0))
                embdict[e] = do_one_mean_max_comparison(np.array(fh[i]), emblist[e])
    sypdfm = pd.DataFrame.from_dict({i: embdict[i] for i in range(len(embdict))}, orient="columns")
    return sypdfm

# this version might work better, since the above gave an error
def do_full_mean_max_comparison(filepath,emblist):
    #highly customized example, with a file of embeddings and assuming you already have other embeddings in a list
    embl=len(emblist)
    scoredictlist=[{} for x in range(embl)]
    with h5py.File(filepath) as fh:
        fk2=list(fh.keys())
        for i in fk2:
            for e in range(embl):
                scoredictlist[e]=do_one_mean_max_comparison(np.array(fh[i]),emblist[e])
    return(scoredictlist)
    #sypdfm=pd.DataFrame.from_dict({i: scoredictlist[i] for i in range(len(scoredictlist))}, orient='columns')
    #return(sypdfm)
