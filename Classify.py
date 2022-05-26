import torch
from utils import get_model
import pandas as pd
import sys
import pathlib
#from metalsiteprediction.data_loading.dataloaders import getProteinLoader
#from metalsiteprediction.ConvRecurrent.trainer import estimate,predict

from utils import getProteinLoader
from predict import estimate,predict
import numpy as np

from datetime import datetime
import argparse

now = datetime.now().strftime("%Y%m%d%_H%M%S")
parser = argparse.ArgumentParser(description="Automatic peaks assignement tool")

parser.add_argument(
    "--data_path",
    help="The folder containing the data to classify",
    required=True)


parser.add_argument(
    "--list_path",
    help="The path of the list file",
    required=True)


def run(args):
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    #device = "cpu"
    torch.backends.cudnn.benchmark = True

    test_data_path = args.data_path
    test_data = args.list_path

    test_data = pd.read_csv(test_data, index_col=False)
    test_data['label'] = [-1]*len(test_data)

    OUTPUT = torch.zeros(len(test_data), 2)
    OUTPUT2 = []
    LABELS = []
    SITE_NAMES = []
    accs = []


    for f_idx in range(10):
        print(f"FOLD {f_idx}", "==" * 30)
        model = get_model(f_idx)
        # torch.load(os.path.join(models_dir, f"F{f_idx}_R0"), map_location=device)
        test_loader = getProteinLoader(test_data, batch_size=1, shuffle=False, data_path=pathlib.Path(test_data_path))
        output, labels, site_names = estimate(model, test_loader)
        #print(output)
        #print(site_names)

        OUTPUT += output
        OUTPUT2.append(output[None, :, :])

    OUTPUT/=10
    #print(np.round(OUTPUT.numpy(), 3))
    #print(site_names)
    #print(np.round(OUTPUT.numpy(), 3)[:, 0].tolist())


    list0 = np.round(OUTPUT.numpy(), 3)[:, 0].tolist()
    list1 = np.round(OUTPUT.numpy(), 3)[:, 1].tolist()


    OUTPUT2 = torch.cat(OUTPUT2, dim=0)

    #print(OUTPUT2.size())

    #print(np.round(OUTPUT2.mean(dim=0).numpy(), 3))
    #print(np.round(OUTPUT2.std(dim=0).numpy(), 3))
    #print(OUTPUT2.round().sum(dim=0))

    dev0 = np.round(OUTPUT2.std(dim=0).numpy(), 3)[:, 0]
    dev1 = np.round(OUTPUT2.std(dim=0).numpy(), 3)[:, 1]

    sum0 = OUTPUT2.round().sum(dim=0).numpy()[:, 0]
    sum1 = OUTPUT2.round().sum(dim=0).numpy()[:, 1]

    preds = pd.DataFrame(
        {'names': site_names,
        'P(random)': list0,
        'P(physio)': list1,
         'std.dev_random':dev0,
        'std.dev_physio':dev1,
         '#random': sum0,
         '#physio':sum1
    })

    print(preds)
    preds.to_csv(f'predictions_{now}.csv')


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    run(args)