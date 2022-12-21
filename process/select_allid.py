import os 
import numpy as np

import sys 
import pdb
sys.path.append('.')
sys.path.append('..')
from utils.utils import compute_cosine,compute_graph,add_list


def work(iddict_path,save_path):

    iddict = np.load(iddict_path,allow_pickle=True).item()
    newdict = iddict.copy()
    for k,v in iddict.items():
        
        crop_path = v[0].replace('id','crop').replace('.npy','.png')
        if not os.path.exists(crop_path):
            del newdict[k]
    feature_matrix = np.array([np.load(v[0]) for v in newdict.values()])[:,0]
    keys = newdict.keys()

    i = 0
    unique_video_dict = {}
   

    # compute face id cos
    cos_dis = compute_cosine(feature_matrix)

    # build graph
    repeat_graph,index = compute_graph(cos_dis)

    repeat_keys = add_list([v for v in repeat_graph.values()])

    assert(len(set(index[0].tolist()+index[1].tolist())) == len(repeat_keys))

    # store the video in dict by same id
   
    for v in repeat_graph.values():
        vals = [newdict[k] for k in np.array(list(keys))[v]]
        unique_video_dict[i] = vals
        i += 1
        print('\r have done %06d'%i,end='',flush=True)
    
    for v in [v for v in range(len(feature_matrix)) if v not in repeat_keys]:
        vals = [newdict[np.array(list(keys))[v]]]
        unique_video_dict[i] = vals
        i += 1
        print('\r have done %06d'%i,end='',flush=True)   
    print()
    np.save(save_path,unique_video_dict)


if __name__ == "__main__":

    iddict_path = 'HeadSwap/wav2lip-headswap/info/single_id.npy'
    save_path = 'HeadSwap/wav2lip-headswap/info/all_id.npy' 
    work(iddict_path,save_path)
   