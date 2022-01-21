import torch

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def predict(model, l, prot):
    out = model(prot)[0]
    batch_idxs = torch.arange(out.size(1))
    idxs = l.long() - 1
    predictions = out[idxs, batch_idxs]
    return predictions


def estimate(model, loader):
    """
    Predict the class of the input data
    """

    output = []
    labels_list = []
    site_names_list = []

    model.eval()
    with torch.no_grad():
        for i, (prot, l, labels, site_names) in enumerate(loader):

            # making predictions
            predictions = predict(model, l, prot.to(device))
            output.append(predictions)
            labels_list.append(labels)
            site_names_list += site_names

    output = torch.cat(output, dim=0)
    labels = torch.cat(labels_list)
    return output, labels, site_names_list