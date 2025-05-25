## load a given embedding as dict by matching string in key

import h5py
import numpy as np

def get_emb_by_match(uid,filepath="/data6/bio/embeddings/p_pacificus.ann.c95.per_residue_embeddingsX.h5"):
    with h5py.File(filepath) as fh:
        kl=list(fh.keys())
        zk=[x for x in kl if uid in x]
        print(zk)
        emb={}
        for i in zk:
            emb[i]=np.array(fh[i].astype("float32"))
    return emb

