import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        
        V = args.embed_num
        D_u = 128
        D = args.embed_dim
        D = 400
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        V_u = args.embed_num_users
        # D_u = args.embed_dim_users
        
        self.embed = nn.Embedding(V, D)
        self.embed_u = nn.Embedding(V_u, D_u)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co + D_u, C)

        if args.pretrained_embed_words:
            # self.embed.weight.data.copy_(torch.tensor(args.custom_embed))
            # self.embed = nn.Embedding(V, 400)
            self.embed.weight.data.copy_(args.vectors)
            self.embed.weight.requires_grad = False
            # D = 400
        # else:
        #     D = args.embed_dim
        if self.args.static:
            self.embed.weight.requires_grad = False

    def forward(self, x, u):
        # print(x.size())
        # print(u.size())
        x = self.embed(x)  # (N, W, D)
        u = self.embed_u(u)
        # print(x.size())
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        # print(size(x))
        x = torch.cat(x, 1)
        # print(x.size())
        x = self.dropout(x)  # (N, len(Ks)*Co)
        # print(x.size())
        # print(u.size())
        # print(torch.cat((x, u), dim = 1).size())
        # print(x.size(),.size())
        x = torch.cat((x, u.squeeze(1)), dim = 1)
        # print(x.size())
        logit = self.fc1(x)  # (N, C)
        return logit
