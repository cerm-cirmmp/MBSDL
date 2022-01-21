import torch.nn as nn
#from metalsiteprediction.ConvRecurrent.conv import *
#from metalsiteprediction.ConvRecurrent.recurrent import *


"""
CONV BLOCK:
Prende in input la proteina come PSFM+altre features, fa una convoluzione 1D.
Estrae le informazioni "locali"

RECURRENT BLOCK:
Prende in input l'uscita della convoluzione e lo elabora con una sequenza
Estrae le informazioni "globali"

"""


base_config = {
    'input_dim': 29,
    'output_dim':2,
    'rnn_hidden_size':20,
    'rnn_n_layers':2,
}


class ConvRecConfig:
    rate = 1
    altricazzi = 1


class ConvNet(nn.Module):

    def __init__(self, config= None):
        super().__init__()

        def convblock(in_channels, out_channels, kernel_size, stride, dropout):
            block =  nn.Sequential(
                nn.Conv1d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=(int(kernel_size/2))),
                nn.BatchNorm1d(out_channels), # ??
                nn.ReLU(),
                nn.Dropout(p=dropout),
            )
            return block

        self.model = nn.Sequential(
            convblock(in_channels=29, out_channels=29, kernel_size=config['kernel_size'], stride=1, dropout=config['conv_dropout']),
            convblock(in_channels=29, out_channels=29, kernel_size=config['kernel_size'], stride=1, dropout=config['conv_dropout']), # added in last update
            #convblock(in_channels=29, out_channels=29, kernel_size=11, stride=1, dropout=0.2),
            #convblock(in_channels=29, out_channels=29, kernel_size=11, stride=1, dropout=0.5),
            #convblock(in_channels=29, out_channels=29, kernel_size=11, stride=1, dropout=0.5),
        )

    def forward(self, X):
        return self.model(X)


class RecurrentModel(nn.Module):

    def __init__(self, in_size, out_size, hidden_size, n_layers, dropout):
        super().__init__()
        self.rnn_model = nn.GRU(input_size=in_size, hidden_size=hidden_size, num_layers=n_layers, dropout=dropout)
        #self.rnn_model = nn.RNN(input_size=in_size, hidden_size=4, num_layers=4)

        self.final_layer = nn.Linear(in_features=hidden_size, out_features=out_size) #


    def forward(self, x):

        x, hiddens = self.rnn_model(x)
        #x = torch.relu(x)
        x = self.final_layer(x)
        #x = torch.sigmoid(x)
        return x, hiddens


class ConvRecurrentClassifier(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.conv = ConvNet(config)
        self.rnn = RecurrentModel(
            in_size=config['input_dim'], # corrisponde alla input dim
            out_size=config['output_dim'],
            hidden_size=config['rnn_hidden_size'],
            n_layers=config['rnn_n_layers'],
            dropout=config['rnn_dropout'])

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(2, 0, 1)
        x = self.rnn(x)
        return x