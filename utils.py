import sys
import torch
import os
import json
import pathlib
import pickle

from torchvision import transforms
#from importlib.resources import path
import pathlib


#from metalsiteprediction.data_loading import preprocessing
import preprocessing


class CustomTransform:

    def __call__(self, protein_file):
        return preprocessing.data_to_tensor(protein_file)
        # oppure ritorna una lista di tensori


#from metalsiteprediction.data_loading.dataloaders \
#    import getTrainTestLoadersByIdx, getZincoSites, getProteinLoader, getZincoPath, getIronPath, getIronSites


from dataloaders  import getTrainTestLoadersByIdx, getZincoSites, getProteinLoader, getZincoPath, getIronPath, getIronSites


zinco_path = getZincoPath()
#print("ZINC PATH ", zinco_path)
zinco_sites = getZincoSites(class_size=100000)
iron_path = getIronPath()
#print("IRON PATH ", iron_path)


folds_path = pathlib.Path(__file__).parent.parent.parent\
    .joinpath('data/10folds_5296points_20210615.json')

folds_path = '10folds_5296points_20210615.json'


#models_dir = pathlib.Path(__file__).parent.\
#    joinpath("trained/TrainWholeDataset88_7/models")

models_dir = pathlib.Path(__file__).parent.\
    joinpath("trained/TrainWholeDataset89_88/models")

models_dir = 'modelsTrainedOnWholeDataset89_88/models'


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

#with open('../../data/10folds_5296points_20210615.json') as json_folds:
with open(folds_path) as json_folds:
    json_folds = json.load(json_folds)


def get_device():
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


def get_models_path():
    models_dir = pathlib.Path(__file__).parent. \
        joinpath("trained/TrainWholeDataset89_88/models")
    return str(models_dir)

#from metalsiteprediction.ConvRecurrent.convRec import ConvRecurrentClassifier
from convRec import ConvRecurrentClassifier



def get_model(f_idx):
    config = {
        'n_random_init': 1,

        # architectural hyperparams:
        'input_dim': 29,
        'output_dim': 2,
        'rnn_hidden_size': 15,
        'rnn_n_layers': 3,
        'kernel_size': 15,

        # learning hyperparams
        'lr': 0.001,  # 0.0005
        'conv_dropout': 0.2,
        'rnn_dropout': 0.6,
        'batch_size': 16,
        'epochs': 200,
        'patience': 200,  # 15,
        'class_size': 10000,
    }

    #mm = torch.load(os.path.join(models_dir, f"F{f_idx}_R0"), map_location=device)

    #torch.save(mm.state_dict(), os.path.join(models_dir, f"F{f_idx}_R0_sd"))
    classifier = ConvRecurrentClassifier(config)
    mm = ConvRecurrentClassifier(config)
    mm.load_state_dict(torch.load(os.path.join(models_dir, f"F{f_idx}_R0_sd"), map_location=device))

    #return torch.load(os.path.join(models_dir, f"F{f_idx}_R0"), map_location=device)
    return mm


def getTestLoader(f_idx, label=None):
    test_names = json_folds[str(f_idx)]['test_names']
    #print(test_names)

    if label != None:
        test_data = zinco_sites[ (zinco_sites['site'].isin(test_names)) & (zinco_sites['label']==label) ]
    else:
        test_data = zinco_sites[zinco_sites['site'].isin(test_names)]
    #print(test_data)

    test_data.reset_index(drop=True, inplace=True)
    test_loader = getProteinLoader(test_data, batch_size=16, shuffle=False, data_path=zinco_path)
    return test_loader


def getValidationLoader(f_idx, label=None):
    test_names = json_folds[str(f_idx)]['val_names']
    #print(test_names)

    if label != None:
        test_data = zinco_sites[ (zinco_sites['site'].isin(test_names)) & (zinco_sites['label']==label) ]
    else:
        test_data = zinco_sites[zinco_sites['site'].isin(test_names)]
    #print(test_data)

    test_data.reset_index(drop=True, inplace=True)
    test_loader = getProteinLoader(test_data, batch_size=16, shuffle=False, data_path=zinco_path)
    return test_loader


def getTrainingLoader(f_idx, label=None):
    test_names = json_folds[str(f_idx)]['train_names']
    #print(test_names)

    if label != None:
        test_data = zinco_sites[ (zinco_sites['site'].isin(test_names)) & (zinco_sites['label']==label) ]
    else:
        test_data = zinco_sites[zinco_sites['site'].isin(test_names)]
    #print(test_data)

    test_data.reset_index(drop=True, inplace=True)
    test_loader = getProteinLoader(test_data, batch_size=16, shuffle=False, data_path=zinco_path)
    return test_loader


def getAsDataset():
    dataset = ProteinDataset(#data_path="./data/zinco",
                             data_path = getZincoPath(),
                             site_names= names_csv,
                             transform=transforms)


def getIronLoader(batch_size=16):
    iron_path = getIronPath()
    print(iron_path)
    iron_sites = getIronSites()
    print(iron_sites)

    #print(">>>>>>", iron_sites)
    loader = getProteinLoader(iron_sites,
                              batch_size=batch_size,
                              shuffle=True,
                              data_path=iron_path, num_workers=10)

    return loader


def getIronNIonesLoader():

    import sys
    import pandas as pd
    d = pathlib.Path("/Users/vincenzo/CERM/MetalSitePredictionV2/data/mapping_ferro.csv")

    mapping_df = pd.read_csv(d)
    mapping_df = mapping_df.set_index('OldName')
    print(mapping_df)
    iron_sites = getIronSites()
    iron_path = getIronPath()

    N_iones = []
    for site in iron_sites['site'].tolist():
        rowsite = readSitePickle(iron_path.joinpath(site + ".pkl"), to_transform=False)
        involved_iones = mapping_df.loc[site, 'Sites'].split(';')
        N_iones.append(len(involved_iones))
    iron_sites['n_iones'] = N_iones
    iron_sites = iron_sites[iron_sites['n_iones']==1]
    iron_sites.reset_index(inplace=True)
    print(iron_sites)
    #sys.exit()

    #print(">>>>>>", iron_sites)
    loader = getProteinLoader(iron_sites,
                              batch_size=14,
                              shuffle=True,
                              data_path=iron_path, num_workers=10)

    return loader


def readSitePickle(file_path, to_transform=True):
    transforms_ = transforms.Compose([CustomTransform()])

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    if to_transform:
        data = transforms_(data)

    return data

#getIronNIonesLoader()