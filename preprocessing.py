import os
import pickle
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def load():
    """
    Legge i file pickle (dictionary) contenenti i dati
    """

    root_path = "metal_sites_data"

    phisio_file = "PHYSIO.pkl"
    removed_file = "ZN_NEGATIVES.pkl"
    #print(">>", os.getcwd())

    #root_path = os.path.join(os.getcwd(), root_path)
    #print(">>> 2", root_path)
    print(os.listdir(root_path))

    phisio_file = os.path.join(root_path, phisio_file)
    removed_file = os.path.join(root_path, removed_file)

    # print("CURR DIR: ", os.getcwd())
    print( phisio_file, removed_file )

    phisio, removed = None, None

    with open(phisio_file, 'rb') as f:
        phisio = pickle.load(f)

    with open(removed_file, 'rb') as f:
        removed = pickle.load(f)

    return phisio, removed

def toOneHot(x):
    assert x.dtype == torch.int or x.dtype == torch.int64 or x.dtype == torch.int32 or x.dtype == torch.int16
    assert x.dim() == 1
    n_attributes = x.max() + 1
    zeros = torch.zeros(x.size(0), n_attributes)
    return zeros.scatter(1, x.view(-1, 1), 1.)

def role_to_onehot1(ruolo, debug=False):
    # RUOLO TO ONE HOT OLD
    ruolo_one_hot  = toOneHot(ruolo)
    if debug:
        print('ruolo_one_hot: ', ruolo_one_hot.size())
    return ruolo_one_hot

def role_to_onehot2(ruolo, debug=False):
    # RUOLO TO ONE HOT NEW
    onehotencoder = OneHotEncoder()
    ruolo_one_hot2 = onehotencoder.fit_transform(ruolo.reshape(-1,1)).toarray()
    if debug:
        print('ruolo: ', ruolo.size() )
    return ruolo_one_hot2

def secondary_struct_onehot(secondary_struct, debug=False):
    # SECONDARY TO ONE HOT
    if debug:
        print(" # SECONDARY TO ONE HOT ")
        print(secondary_struct)

    le = LabelEncoder()
    le.fit(["E","H", "T","L"])
    second_struct_digit = le.transform(secondary_struct)
    # print(second_struct_digit)

    onehotencoder2 = OneHotEncoder(categories=[[0,1,2,3]])
    second_struct_one_hot = onehotencoder2.fit_transform(second_struct_digit.reshape(-1, 1)).toarray()
    second_struct_one_hot = torch.from_numpy(second_struct_one_hot)

    if debug:
        print(type(second_struct_one_hot), second_struct_one_hot.shape)
        print("$$$$$ ", onehotencoder2.categories_)
        print("$$$$$ ", le.classes_)
    #for i, a in enumerate(second_struct_one_hot):
    #    print(secondary_struct[i], second_struct_one_hot[i])
    return second_struct_one_hot

def data_to_tensor(dictiorarized_seq, debug=False):
    # 'HHM', 'name', 'fasta', 'accsolrel', 'AccSolAbs', 'bindigpar', 'secstru'
    aa_name = dictiorarized_seq['name'] # lista dei nomi di tutti gli aminoacidi
    seq = dictiorarized_seq['fasta']

    PSFM = dictiorarized_seq['HHM']['PSFM'] # hmm related stuff
    PSFM = torch.from_numpy(PSFM)

    acc_solv_ass = torch.tensor( dictiorarized_seq['AccSolAbs'] , dtype=torch.float )
    acc_solv_rel = torch.tensor( dictiorarized_seq['accsolrel'], dtype=torch.float )

    ruolo = torch.tensor(dictiorarized_seq['bindigpar'])
    secondary_struct = dictiorarized_seq['secstru']

    if debug:
        print('>> aa_name', aa_name)
        print('>> seq: ', seq)
        print('acc_solv_ass: ', acc_solv_ass.view(-1, 1).size() )
        print('acc_solv_rel: ', acc_solv_rel.view(-1, 1).size() )

    ruolo_onehot = role_to_onehot1(ruolo, debug)
    second_struct_onehot = secondary_struct_onehot(secondary_struct, debug)

    TT = torch.cat((
        PSFM, # 20
        acc_solv_ass.view(-1, 1), # 1
        acc_solv_rel.view(-1, 1), # 1
        ruolo_onehot, # 3
        second_struct_onehot.float() # 4
    ), dim=1) # ???
    #print(TT.size())
    return TT

def tensor_from_data(physio_file, removed_file):
    print(physio_file)
    print(removed_file)

    physio = torch.load(physio_file)
    removed = torch.load(removed_file)

    print("--- ok ---")

    y_phisio = torch.zeros(physio.size(0), 2)
    y_phisio[:, 0] = 1.

    y_removed = torch.zeros(removed.size(0), 2)
    y_removed[:, 1] = 1.

    X = torch.cat((physio, removed), dim=0)
    X = X.permute(0, 2, 1)

    Y = torch.cat((y_phisio, y_removed))

    print("physio: ", physio.size())
    print("removed: ", removed.size())

    return X, Y

########################################################################################


def main():
    """
    1) Load the two dictionary (one per class) containing the data
    2) For each of the two dictionary, for each example:
        2.1) check if the example fits the lengths constraints
        2.2) transform data to tensor
        2.3) Set the tensor to the size of the biggest one (related to the longest sequence)
    3) pack all the tensors of a single class in a single tensor
    :return: one tensor per class.
    """

    phisio, removed = load()

    PHYSIO, REMOVED = [], []
    PHYSIO_NAMES, REMOVED_NAMES = [], []
    DEBUG = False

    ok = 0
    tot = 0
    MAX_LEN = 500
    MIN_LEN = 300

    for i, key in enumerate(phisio):
        seqlen = len(phisio[key]['fasta'])
        tot += 1
        if seqlen > MIN_LEN and seqlen < MAX_LEN:
            ok += 1
            tt = data_to_tensor( phisio[key], debug=DEBUG )
            #print("////",tt.size())
            # FILLING DEL TENSORE
            delta = MAX_LEN - tt.size(0)
            fill_tensor = torch.zeros(delta, 29)
            #print(fill_tensor.size())
            newtensor = torch.cat( (tt, fill_tensor) )
            #print(newtensor.size())
            PHYSIO.append( newtensor )
            PHYSIO_NAMES.append(key+"_PHY")

    print(tot)
    print(ok)
    print("////////////////")
    ok = 0
    tot = 0
    for i, key in enumerate(removed):
        seqlen = len(removed[key]['fasta'])
        tot += 1
        if seqlen > MIN_LEN and seqlen < MAX_LEN:
            ok += 1
            tt = data_to_tensor( removed[key], debug=DEBUG )
            #print("////",tt.size())
            # FILLING DEL TENSORE
            delta = MAX_LEN - tt.size(0)
            fill_tensor = torch.zeros(delta, 29)
            #print(fill_tensor.size())
            newtensor = torch.cat( (tt, fill_tensor) )
            #print(newtensor.size())
            REMOVED.append( newtensor )
            REMOVED_NAMES.append(key+"_RND")

    print(tot)
    print(ok)

    PHYSIO = torch.stack( PHYSIO, dim= 0 )
    REMOVED = torch.stack( REMOVED, dim= 0 )

    print(PHYSIO.size())
    print(REMOVED.size())

    torch.save(PHYSIO, "metal_sites_data/big_physio300-500.pt")
    torch.save(REMOVED, "metal_sites_data/big_removed300-500.pt")
    #salvare pure le liste please - o aggiungerle direttamente allo oggetto salvato ??? ehhh!!!



if __name__ == "__main__":
    run()











