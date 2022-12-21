import numpy as np
import os 
import random 

def split_data(id_path,train_save_path,val_save_path):
    idinfo = np.load(id_path,allow_pickle=True).item()
    keys = list(idinfo.keys())
    random.shuffle(keys)
    train_id_info = {}
    val_id_info = {}

    for k in keys[:100]:
        val_id_info[k] = idinfo[k]

    for k in keys[100:]:
        train_id_info[k] = idinfo[k]

    np.save(train_save_path,train_id_info)
    np.save(val_save_path,val_id_info)

if __name__ == "__main__":
    id_path = '/HeadSwap/wav2lip-headswap/info/all_id.npy'
    train_path = '/HeadSwap/wav2lip-headswap/info/train_id.npy'
    val_path = 'HeadSwap/wav2lip-headswap/info/val_id.npy'
    split_data(id_path,train_path,val_path)
    