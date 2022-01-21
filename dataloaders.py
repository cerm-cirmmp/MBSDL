"""
Tre data loaders, per train, val e test
"""
import pandas as pd
import pickle
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from importlib.resources import path
import pathlib
from sklearn.model_selection import StratifiedKFold

#from metalsiteprediction.data_loading import preprocessing
import preprocessing

class CustomTransform:

    def __call__(self, protein_file):
        return preprocessing.data_to_tensor(protein_file)
        # oppure ritorna una lista di tensori


class ProteinDataset(Dataset):

    def __init__(self, data_path, site_names, transform=None ):
        self.data_path = data_path
        # self.site_names = pd.read_csv(os.path.join(data_path+"/..", site_names_csv))
        # self.site_names = pd.read_csv(data_path.parent.joinpath(site_names_csv))
        self.site_names = site_names
        self.transform = transform
        print("Protein Dataser Loaded, ", len(self.site_names))

    def __len__(self):
        return len(self.site_names)

    def __load_protein_data(self, path, name):
        with open(path.joinpath(name), 'rb') as f:
        # with open(path+'/' + name, 'rb') as f:
            return pickle.load(f)

    def __getitem__(self, idx):

        site_name = "{}.pkl".format(self.site_names['site'][idx])

        name = self.site_names['site'][idx] # last added

        label = self.site_names['label'][idx]

        sample = self.__load_protein_data(self.data_path, site_name)
        site_length = len(sample['fasta'])

        if self.transform:
            sample = self.transform(sample)

        return sample, site_length, label, name


class ProteinDataLoader(DataLoader):

    def __batch_samples(self, samples):
        # print("Batch size: ", len(list_of_samples))

        z = torch.zeros(29)

        list_of_samples = [x[0] for x in samples]
        batch_labels = [x[1] for x in samples]

        longest = max([len(x) for x in list_of_samples])
        # print("Longest is ", longest)

        batch_tensor = []

        for protein, label in samples:
            rows_to_fill = longest - len(protein)
            zero_fill = z.repeat((rows_to_fill, 1))
            sample_filled = torch.cat((protein, zero_fill))
            batch_tensor.append(sample_filled)

        batch_tensor = torch.stack(batch_tensor, dim=0)
        batch_tensor = batch_tensor.permute(0, 2, 1)

        batch_labels = torch.stack(batch_labels, dim=0)

        return batch_tensor, batch_labels

    def __init__(self):
        super().__init__()
        self.transformations = transforms.Compose([CustomTransform()])
        self.dataset = ProteinDataset()
        self.protein_data_loader = torch.utils.data.DataLoader(self.dataset,
                                                               collate_fn=self.__batch_samples,
                                                               batch_size=4)



def batch_samples(samples):

    # print("Batch size: ", len(list_of_samples))
    z = torch.zeros(29)

    # le labels come le gestiamo?

    list_of_samples = [x[0] for x in samples]
    #batch_labels = [x[1] for x in samples] a sto giro non serve
    longest = max([len(x) for x in list_of_samples])

    #print("Longest is ", longest)

    batch_tensor = []
    prot_lengths = []
    prot_labels = []
    names = []

    for protein, prot_len, label, name in samples:
        #
        rows_to_fill = longest - len(protein)
        zero_fill = z.repeat((rows_to_fill, 1))
        sample_filled = torch.cat((protein, zero_fill))
        #
        batch_tensor.append(sample_filled)
        prot_lengths.append( torch.tensor(prot_len, dtype=torch.int) )
        prot_labels.append(label)
        names.append(name)

    batch_tensor = torch.stack(batch_tensor, dim=0)
    batch_tensor = batch_tensor.permute(0, 2, 1)

    batch_lenghts = torch.stack(prot_lengths)
    batch_labels = torch.tensor(prot_labels)

    return batch_tensor, batch_lenghts, batch_labels, names


def getZincoPath():
    rootpath = pathlib.Path(__file__).parent.parent.parent
    zincopath = rootpath.joinpath('data/zinco')
    zincopath = pathlib.Path('zinc')
    return zincopath


def getIronPath():
    rootpath = pathlib.Path(__file__).parent.parent.parent
    ironpath = rootpath.joinpath('data/ferro')
    ironpath = pathlib.Path('iron')
    return ironpath


def getProteinLoader(names_csv, batch_size=5, shuffle = True, data_path=None, num_workers=4):
    from torchvision import transforms
    transforms = transforms.Compose([CustomTransform()])

    rootpath = pathlib.Path(__file__).parent.parent.parent

    dataset = ProteinDataset(#data_path="./data/zinco",
                             data_path = data_path,
                             site_names= names_csv,
                             transform=transforms)

    loader = torch.utils.data.DataLoader(
        dataset, collate_fn=batch_samples, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader


def getZincoSites(zinco_pos_path=None, zinco_neg_path=None, seq_len_limit = 500, class_size = 1944):

    if zinco_pos_path == None:
        zinco_pos_path = "zinco_positive_sites.csv"

    if zinco_neg_path == None:
        zinco_neg_path = "zinco_negative_sitesV2.csv"

    #print(f"-- zinco_pos_path: {zinco_pos_path} --")
    #print(f"-- zinco_neg_path: {zinco_neg_path} --")

    MAX_SEQ_LEN = seq_len_limit

    #pos_path = getZincoPath().parent.joinpath("zinco_positive_sites.csv")
    pos_path = getZincoPath().parent.joinpath(zinco_pos_path)
    pos_zinco_sites = pd.read_csv(pos_path)
    # pos_zinco_sites.drop(pos_zinco_sites.columns[0], axis=1, inplace=True)

    #neg_path = getZincoPath().parent.joinpath("zinco_negative_sitesV2.csv")
    neg_path = getZincoPath().parent.joinpath(zinco_neg_path)
    neg_zinco_sites = pd.read_csv(neg_path)
    #neg_zinco_sites.drop(neg_zinco_sites.columns[0], axis=1, inplace=True)

    pos_zinco_sites = pos_zinco_sites[(pos_zinco_sites['length']<MAX_SEQ_LEN)][:class_size]
    pos_zinco_sites['label'] = [1]*len(pos_zinco_sites)

    neg_zinco_sites = neg_zinco_sites[(neg_zinco_sites['length']<MAX_SEQ_LEN)][:class_size]
    neg_zinco_sites['label'] = [0] * len(neg_zinco_sites)


    #print("LENGTHS", len(pos_zinco_sites), len(neg_zinco_sites))

    all_zinco_sites = pd.concat([pos_zinco_sites, neg_zinco_sites], ignore_index=True)
    # shuffling
    all_zinco_sites = all_zinco_sites.sample(frac=1).reset_index(drop=True)

    #print(all_zinco_sites)
    #loader = getProteinLoader(all_zinco_sites, batch_size=par['batch_size'])

    return all_zinco_sites







def getZincoSites_all(lim):

    pos_path = getZincoPath().parent.joinpath("zinco_positive_sites.csv")
    pos_zinco_sites = pd.read_csv(pos_path)
    pos_zinco_sites.drop(pos_zinco_sites.columns[0], axis=1, inplace=True)

    neg_path = getZincoPath().parent.joinpath("zinco_negative_sites.csv")
    neg_zinco_sites = pd.read_csv(neg_path)
    neg_zinco_sites.drop(neg_zinco_sites.columns[0], axis=1, inplace=True)

    pos_zinco_sites = pos_zinco_sites[(pos_zinco_sites['length']<lim)]
    pos_zinco_sites['label'] = [1]*len(pos_zinco_sites)

    neg_zinco_sites = neg_zinco_sites[(neg_zinco_sites['length']<lim)]
    neg_zinco_sites['label'] = [0] * len(neg_zinco_sites)

    #print(pos_zinco_sites)
    #print(neg_zinco_sites)

    #print(len(list(pos_zinco_sites.itertuples())))
    #print("--- ", len(pos_zinco_sites), len(neg_zinco_sites), "----")
    print("LENGTHS", len(pos_zinco_sites), len(neg_zinco_sites))

    all_zinco_sites = pd.concat([pos_zinco_sites, neg_zinco_sites], ignore_index=True)

    # shuffling
    all_zinco_sites = all_zinco_sites.sample(frac=1).reset_index(drop=True)

    #print(all_zinco_sites)
    #loader = getProteinLoader(all_zinco_sites, batch_size=par['batch_size'])

    return all_zinco_sites






def getIronSites():
    #pos_path = getIronPath().parent.joinpath("fe_positive_sites.csv")
    #pos_fe_sites = pd.read_csv(pos_path)
    pos_fe_sites = pd.read_csv("fe_positive_sites.csv")
    pos_fe_sites.drop(pos_fe_sites.columns[0], axis=1, inplace=True)

    #neg_path = getIronPath().parent.joinpath("fe_negative_sites.csv")
    #neg_fe_sites = pd.read_csv(neg_path)
    neg_fe_sites = pd.read_csv("fe_negative_sites.csv")
    neg_fe_sites.drop(neg_fe_sites.columns[0], axis=1, inplace=True)

    #pos_fe_sites = pos_fe_sites[(pos_fe_sites['length']<500)][:class_size]
    pos_fe_sites['label'] = [1]*len(pos_fe_sites)

    #neg_fe_sites = neg_fe_sites[(neg_fe_sites['length']<500)][:class_size]
    neg_fe_sites['label'] = [0] * len(neg_fe_sites)

    print("LENGTHS", len(pos_fe_sites), len(neg_fe_sites))

    all_fe_sites = pd.concat([pos_fe_sites, neg_fe_sites], ignore_index=True)

    # shuffling
    #all_zinco_sites = all_zinco_sites.sample(frac=1).reset_index(drop=True)

    #print(all_zinco_sites)
    #loader = getProteinLoader(all_zinco_sites, batch_size=par['batch_size'])

    return all_fe_sites


def getTrainTestLoaders(batch_size = 8, class_size=1944, shuffle=False):
    # data loading
    zinco_sites = getZincoSites(class_size=class_size)

    skf = StratifiedKFold(n_splits=5)

    for train_index, test_index in skf.split(zinco_sites.to_numpy(), zinco_sites['label'].to_numpy()):
        #print("TRAIN:", train_index, "TEST:", test_index)
        print("===> ",len(train_index)/batch_size, len(test_index)/batch_size)
        break

    train_data = zinco_sites.loc[train_index, :]
    test_data = zinco_sites.loc[test_index, :]

    print(train_data.reset_index(drop=True, inplace=True))
    print(test_data.reset_index(drop=True, inplace=True))
    print("==>", type(train_data), len(train_data))
    print("==>", type(test_data), len(test_data))

    train_loader = getProteinLoader(train_data, batch_size=batch_size, shuffle = shuffle)
    test_loader = getProteinLoader(test_data, batch_size=batch_size, shuffle = shuffle)

    return train_loader, test_loader


def getTrainTestLoadersByIdx(train_index, test_index, batch_size = 8, class_size=1944, shuffle=False):
    # data loading
    zinco_sites = getZincoSites(class_size=class_size)

    train_data = zinco_sites.loc[train_index, :]
    test_data = zinco_sites.loc[test_index, :]

    print(train_data.reset_index(drop=True, inplace=True))
    print(test_data.reset_index(drop=True, inplace=True))
    print("==>", type(train_data), len(train_data))
    print("==>", type(test_data), len(test_data))

    train_loader = getProteinLoader(train_data, batch_size=batch_size, shuffle = shuffle)
    test_loader = getProteinLoader(test_data, batch_size=batch_size, shuffle = shuffle)

    return train_loader, test_loader