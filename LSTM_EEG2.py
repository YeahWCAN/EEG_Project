class LSTM_EEG2(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, hidden_dim2, n_class):
        super(LSTM_EEG2, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer,  batch_first=True)  
        self.outputlayer = nn.Linear(hidden_dim, hidden_dim2)
        self.classifier = nn.Linear(hidden_dim2,n_class)
        
    def forward(self, x):
        # h0 = Variable(torch.zeros(self.n_layer, x.size(1),
                                #   self.hidden_dim)).cuda()
        # c0 = Variable(torch.zeros(self.n_layer, x.size(1),
                                #   self.hidden_dim)).cuda()                     
        out, _ = self.lstm(x)
        out = out[:, -1, :]       
        out =  self.outputlayer(out)
        out = F.relu(out)       
        out = self.classifier(out)
        #out = F.softmax(out,dim=1)
        return out
