import sys
import torch
import matplotlib.pyplot as plt
#from metalsiteprediction.ConvRecurrent.utils import getTestLoader, get_model
#from metalsiteprediction.ConvRecurrent.utils import getIronLoader, iron_path
#from metalsiteprediction.ConvRecurrent.trainer import predict, estimate
#from metalsiteprediction.ConvRecurrent.utils import readSitePickle


from utils import getTestLoader, get_model
from utils import getIronLoader, iron_path
from predict import predict, estimate
from utils import readSitePickle

import argparse
from datetime import datetime
import copy
import pandas as pd
import math
import pathlib

now = datetime.now().strftime("%Y%m%d%_H%M%S")
parser = argparse.ArgumentParser(description="Automatic peaks assignement tool")

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#device = "cpu"
torch.backends.cudnn.benchmark = True

parser.add_argument(
    "--metal",
    choices=['iron', 'zinc'],
    help="The metal data to classify",
    required=True)


def run_zinc_classification():
    OUTPUT = []
    LABELS = []
    SITE_NAMES = []
    accs = []

    for f_idx in range(10):
        print(f"FOLD {f_idx}", "=="*30)
        model = get_model(f_idx) # torch.load(os.path.join(models_dir, f"F{f_idx}_R0"), map_location=device)
        loader = getTestLoader(f_idx)
        output, labels, site_names = estimate(model, loader)

        acc = (output.argmax(dim=1) == labels).float().mean()
        accs.append(acc)
        print("Fold acc: ", acc)

        print(output.size())
        print(labels.size())
        OUTPUT.append(output)
        LABELS.append(labels)
        SITE_NAMES += site_names

    OUTPUT = torch.cat(OUTPUT, dim=0)
    LABELS = torch.cat(LABELS, dim=0)
    print(SITE_NAMES)
    print(len(SITE_NAMES))

    print(OUTPUT.size())
    print(LABELS.size())

    acc_mean = sum(accs) / len(accs)
    ACC = (OUTPUT.argmax(dim=1) == LABELS).float().mean()
    print("Mean acc: ", acc_mean, ACC)


    classifications = pd.DataFrame({'site':SITE_NAMES,
                  'P(0)':OUTPUT[:,0],
                  'P(1)':OUTPUT[:,1],
                  'LABEL': LABELS})

    classifications.to_csv(f"Zinc_Classifications{now}.csv", index=False)


def run_iron_classification():

    # dato che ho dieci modelli... faccio la media dei 10... ???

    d = pathlib.Path("mapping_iron.csv")
    mapping_df = pd.read_csv(d)
    mapping_df = mapping_df.set_index('OldName')
    print(mapping_df)

    ironLoader = getIronLoader(batch_size=32)

    classifications_df = None
    print("= ="*20)
    print("= =" * 20)

    acc0s = []
    acc1s = []

    # dovremmo mediare su tutti i fold
    for f_idx in range(1):
        print(f"FOLD {f_idx} Model", "=="*50)
        zincTrainedModel = get_model(f_idx)

        # fare la media delle classificazioni...  fare una funzion che lo fa
        # e metterla da qualche parte, riservirà
        print("Estimating...")
        output, labels, site_names = estimate(zincTrainedModel, ironLoader)

        N_iones = []
        Site_len = []
        for site in site_names:

            rowsite = readSitePickle(iron_path.joinpath(site+".pkl"), to_transform=False)
            involved_iones = mapping_df.loc[site, 'Sites'].split(';')
            N_iones.append(len(involved_iones))
            Site_len.append(len(rowsite['fasta']))
            print(site, " --> ", mapping_df.loc[site, 'Sites'], len(involved_iones))

        #if classifications_df != None:
        classifications = pd.DataFrame({'site': site_names,
                                        'P(0)': output[:, 0],
                                        'P(1)': output[:, 1],
                                        '#iones':N_iones,
                                        'Len':Site_len,
                                        'LABEL': labels})


        #print(classifications)
        accuracy = (output.argmax(dim=1) == labels).float().mean()
        C0_idxs = labels == 0
        C1_idxs = labels == 1

        print(len(C0_idxs.nonzero()), C0_idxs.sum())
        print(len(C1_idxs.nonzero()), C1_idxs.sum())

        print("Acc: ", accuracy)

        accuracy0 = ( output[C0_idxs].argmax(dim=1) == labels[C0_idxs]).float().mean()
        accuracy1 = (output[C1_idxs].argmax(dim=1) == labels[C1_idxs]).float().mean()


        print(f"#C0 {C0_idxs.sum()}, #C1 {C1_idxs.sum()}")

        print("ACC0", accuracy0, "ACC1", accuracy1)
        acc0s.append(accuracy0)
        acc1s.append(accuracy1)

        print(classifications)

    print(acc0s)
    print(acc1s)
    print("AVGs Results:")
    print(sum(acc0s)/len(acc0s))
    print(sum(acc1s) / len(acc1s))

    classifications.to_csv(f"Iron_Classifications{now}.csv", index=False)
    classifications1 = classifications[classifications['#iones'] == 1]


    classificationsL1 = classifications1[classifications1['LABEL'] == 1]
    output = classificationsL1[['P(0)', 'P(1)']].to_numpy()
    labels = classificationsL1['LABEL'].to_numpy()
    #print(output)
    #print(labels)
    accuracyL1 = (output.argmax(axis=1) == labels).mean()
    print(accuracyL1)



    classificationsL0 = classifications1[classifications1['LABEL'] == 0]
    output = classificationsL0[['P(0)', 'P(1)']].to_numpy()
    labels = classificationsL0['LABEL'].to_numpy()
    #print(output)
    #print(labels)
    accuracyL0 = (output.argmax(axis=1) == labels).mean()
    print(accuracyL0)

    return classifications




#run_zinc_classification()
#run_iron_classification()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    if args.metal == 'zinc':
        print("Running zinc data classification")
        run_zinc_classification()
    else:
        print("Running iron data classification")
        run_iron_classification()









