"""
Try different neural network setting for the contrastive learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLCPnet(nn.Module):
    """backbone + projection head"""
    def __init__(self, opt, head='mlp', feat_dim=1024):
        super(CLCPnet, self).__init__()
        dim_in = opt.model_out_dim
        feat_dim = opt.feat_dim
        self.encoder = activation3CLCPnet(opt) #simple3CLCPnet
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=0)
        return feat

class simple3CLCPnet(nn.Module):
    def __init__(self, opt):
        super(simple3CLCPnet, self).__init__()
        self.layer1 = nn.Linear(opt.model_in_dim, opt.model_n_hidden_1)
        self.layer2 = nn.Linear(opt.model_n_hidden_1, opt.model_n_hidden_2)
        self.layer3 = nn.Linear(opt.model_n_hidden_2, opt.model_out_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class simple2CLCPnet(nn.Module):
    def __init__(self, opt):
        super(simple2CLCPnet, self).__init__()
        self.layer1 = nn.Linear(opt.model_in_dim, opt.model_n_hidden_1)
        self.layer2 = nn.Linear(opt.model_n_hidden_1, opt.model_out_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class activation3CLCPnet(nn.Module):

    def __init__(self, opt):
        super(activation3CLCPnet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(opt.model_in_dim, opt.model_n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(opt.model_n_hidden_1, opt.model_n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(opt.model_n_hidden_2, opt.model_out_dim), nn.ReLU(True))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class Coxnnet2(nn.Module):

    def __init__(self, opt):
        super(Coxnnet2, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(opt.model_in_dim, opt.model_n_hidden_1), nn.ReLU(True),
                                    nn.Dropout(0.0775))
        self.layer2 = nn.Sequential(nn.Linear(opt.model_n_hidden_1, opt.model_n_hidden_2), nn.ReLU(True),
                                    nn.Dropout(0.0775))
        self.layer3 = nn.Sequential(nn.Linear(opt.model_n_hidden_2, opt.model_out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class Coxnnet1(nn.Module):

    def __init__(self, opt):
        super(Coxnnet1, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(opt.model_in_dim, opt.model_n_hidden_1), nn.ReLU(True),
                                    nn.Dropout(0.0775))
        self.layer2 = nn.Sequential(nn.Linear(opt.model_n_hidden_1, opt.model_out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class activation31mlpCLCPnet(nn.Module):

    def __init__(self, opt):
        super(activation31mlpCLCPnet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(opt.model_in_dim, opt.model_n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(opt.model_n_hidden_1, opt.model_n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(opt.model_n_hidden_2, opt.model_out_dim), nn.ReLU(True))
        self.head = nn.Sequential(
            nn.Linear(opt.model_out_dim, opt.model_out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(opt.model_out_dim, opt.feat_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.normalize(self.head(x), dim=0)
        return x

class activation21mlpCLCPnet(nn.Module):

    def __init__(self, opt):
        super(activation21mlpCLCPnet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(opt.model_in_dim, opt.model_n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(opt.model_n_hidden_1, opt.model_out_dim), nn.ReLU(True))
        self.head = nn.Sequential(
            nn.Linear(opt.model_out_dim, opt.model_out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(opt.model_out_dim, opt.feat_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = F.normalize(self.head(x), dim=0)
        return x

class activation21mlpl1CLCPnet(nn.Module):

    def __init__(self, opt):
        super(activation21mlpl1CLCPnet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(opt.model_in_dim, opt.model_n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(opt.model_n_hidden_1, opt.model_out_dim), nn.ReLU(True))
        self.head = nn.Sequential(
            nn.Linear(opt.model_out_dim, opt.model_out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(opt.model_out_dim, opt.feat_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = F.normalize(self.head(x), dim=0)
        return x

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

class activation21mlpl2CLCPnet(nn.Module):

    def __init__(self, opt):
        super(activation21mlpl2CLCPnet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(opt.model_in_dim, opt.model_n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(opt.model_n_hidden_1, opt.model_out_dim), nn.ReLU(True))
        self.head = nn.Sequential(
            nn.Linear(opt.model_out_dim, opt.model_out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(opt.model_out_dim, opt.feat_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = F.normalize(self.head(x), dim=0)
        return x

    def compute_l2_loss(self, w):
        return w.pow(2).sum()

class activation21mlpdropoutCLCPnet(nn.Module):

    def __init__(self, opt):
        super(activation21mlpdropoutCLCPnet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(opt.model_in_dim, opt.model_n_hidden_1), nn.ReLU(True), nn.Dropout(0.1))
        self.layer2 = nn.Sequential(nn.Linear(opt.model_n_hidden_1, opt.model_out_dim), nn.ReLU(True), nn.Dropout(0.1))
        self.head = nn.Sequential(
            nn.Linear(opt.model_out_dim, opt.model_out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(opt.model_out_dim, opt.feat_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = F.normalize(self.head(x), dim=0)
        return x

class activation11mlpCLCPnet(nn.Module):

    def __init__(self, opt):
        super(activation11mlpCLCPnet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(opt.model_in_dim, opt.model_out_dim), nn.ReLU(True))
        self.head = nn.Sequential(
            nn.Linear(opt.model_out_dim, opt.model_out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(opt.model_out_dim, opt.feat_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = F.normalize(self.head(x), dim=0)
        return x

class activation31linearCLCPnet(nn.Module):

    def __init__(self, opt):
        super(activation31linearCLCPnet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(opt.model_in_dim, opt.model_n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(opt.model_n_hidden_1, opt.model_n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(opt.model_n_hidden_2, opt.model_out_dim), nn.ReLU(True))
        self.head = nn.Sequential(nn.Linear(opt.model_out_dim, opt.feat_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.normalize(self.head(x), dim=0)
        return x

class activation21linearCLCPnet(nn.Module):

    def __init__(self, opt):
        super(activation21linearCLCPnet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(opt.model_in_dim, opt.model_n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(opt.model_n_hidden_1, opt.model_out_dim), nn.ReLU(True))
        self.head = nn.Sequential(nn.Linear(opt.model_out_dim, opt.feat_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = F.normalize(self.head(x), dim=0)
        return x

class activation2CLCPnet(nn.Module):

    def __init__(self, opt):
        super(activation2CLCPnet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(opt.model_in_dim, opt.model_n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(opt.model_n_hidden_1, opt.model_out_dim)) #, nn.ReLU(True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class batch3CLCPnet(nn.Module):
    """contrastive-learning-for-cancer-prognosis"""
    def __init__(self, opt):
        super(batch3CLCPnet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(opt.model_in_dim, opt.model_n_hidden_1), nn.BatchNorm1d(opt.model_n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.Linear(opt.model_n_hidden_1, opt.model_n_hidden_2), nn.BatchNorm1d(opt.model_n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.Linear(opt.model_n_hidden_2, opt.model_out_dim), nn.BatchNorm1d(opt.model_out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return (x)
