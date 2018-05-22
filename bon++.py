from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
import time
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import datasets

plt.switch_backend('agg')

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--niter', type=int, default=1001, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for D, default=0.001') 
parser.add_argument('--lr_g', type=float, default=0.0001, help='learning rate for G, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--saving', type=int, default=10, help='after how many epochs do we save plot')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--w_g', type=float, default=2.0, help='weight on the fake G backprop relative to MAS')
parser.add_argument('--w_exp', type=float, default=1.025, help='exponent weight for MAS')
parser.add_argument('--w_init_std', type=float, default=0.02, help='standard deviation to initialize network parameters')
parser.add_argument('--nbG', type=int, default=1, help='number of generators')
parser.add_argument('--manualSeed', type=int, default=42, help='manual seed')
parser.add_argument('--use_bon_disable', action='store_false', help='disable bon, i.e. train only with a D(x)')
parser.add_argument('--clip_imp', action='store_true', help='if true: clip importance weights')
parser.add_argument('--w_imp', type=float, default=20.0, help='maximum value of importance weight, will only have effect if clip_imp==True')
parser.add_argument('--stoch', action='store_true', help='if true: use stochastic gradient descent')
parser.add_argument('--scale_real', action='store_true', help='if true: scale importance of real data points when backproping D')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

nb_G = opt.nbG
w_g = opt.w_g
w_exp = opt.w_exp
use_bon_disable = opt.use_bon_disable
stoch = opt.stoch

# import some data to play with
iris = datasets.load_iris()
x = iris.data[:, [0,1]]  # we only use the first two features.
y = iris.target

class ExampleDataset(Dataset):

    def __init__(self, x, y, batch_size=10):
        self.dataframe = pd.DataFrame(x)
        self.target = pd.DataFrame(y)
        self.batch_size = batch_size

    def __len__(self):
        return int(len(self.dataframe)/self.batch_size)

    def __getitem__(self, idx):
        x = self.dataframe.iloc[idx*self.batch_size:idx*self.batch_size+self.batch_size,:]
        y = self.target.iloc[idx*self.batch_size:idx*self.batch_size+self.batch_size]
        sample = {'x': np.array(x), 'y': np.array(y)}

        return sample

    def shuffle(self):
        temp_idx = np.random.permutation(len(self.dataframe))
        self.dataframe = self.dataframe.iloc[temp_idx]
        self.target = self.target.iloc[temp_idx]

trainloader = ExampleDataset(x, y)

ngpu = int(opt.ngpu)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, opt.w_init_std)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, opt.w_init_std)
        m.bias.data.fill_(0)


# Define D
class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(2,100), 
            nn.ReLU(True),
            nn.Linear(100,100),
            nn.ReLU(True),
            nn.Linear(100,50),
            nn.ReLU(True),
            nn.Linear(50,3),
        )

    def forward(self, input):
        return self.main(input)

# Define G
class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(2,50),
            nn.ReLU(True),
            nn.Linear(50,50),
            nn.ReLU(True),
            nn.Linear(50,50),
            nn.ReLU(True),
            nn.Linear(50,2),            
        )

    def forward(self, input):
        return self.main(input)

    # function to return the parameters, flattened, of G.
    # This is used for the Memory Aware Synapses
    def getParameters(self):
        params = []
        for m in self.parameters():
            # we do not get param of output module
            l = list(m)
            params.extend(l)

        one_dim = [p.view(p.numel()) for p in params]
        params = F.torch.cat(one_dim)
        return params        

    # function to return the absolute value of the gradients
    # This is also used for the Memory Aware Synapses
    def getGradients(self):
        grads = []
        for m in self.parameters():
            # we do not get param of output module
            l = list(m.grad)
            grads.extend(l)

        one_dim = [g.view(g.numel()) for g in grads]
        grads = F.torch.cat(one_dim)
        return F.torch.abs(grads)        

# Create nbG generators
G_lst = []
importance_weights = []
theta_star = []
for i in range(nb_G):
    G_lst.append(_netG(ngpu))
    G_lst[i].apply(weights_init)
    importance_weights.append(np.zeros((int(G_lst[i].getParameters().size()[0]))))
    theta_star.append(np.zeros((int(G_lst[i].getParameters().size()[0]))))

# Create D
netD = _netD(ngpu)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion_ce = nn.CrossEntropyLoss()

def mse_loss(input, target):
    return torch.sum((input - target)**2) / target.nelement()

if opt.cuda:
    netD = netD.cuda()
    criterions_ce = criterion_ce.cuda()

# setup optimizer for D
if stoch:
    optimizerD = optim.SGD(netD.parameters(), lr=opt.lr)
else:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99))

# setup optimizer for all G's
G_optimizers = []
for i in range(len(G_lst)):
    if stoch:
        G_optimizers.append(optim.SGD(G_lst[i].parameters(), opt.lr_g, momentum=0.9, weight_decay=1e-4))
    else:
        G_optimizers.append(optim.Adam(G_lst[i].parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.99)))
    
    if opt.cuda:
        G_lst[i] = G_lst[i].cuda()


# Run BON++
n = 0
for epoch in range(opt.niter):

    netD.train()

    fakes = []
    y_new = []

    print(epoch)

    trainloader.shuffle()

    for i in range(len(trainloader)):

        sample = trainloader[i]
        input = torch.FloatTensor(sample["x"])
        labels = torch.FloatTensor(sample["y"].tolist()).view(-1)
        
        #print(i)
        ############################
        # (1) Update D network
        ###########################
        # train with real
        for d_iter in range(1):

            netD.zero_grad()

            if opt.cuda:
                input = input.cuda()
                labels = labels.cuda()
                        
            inputv = Variable(input)
            output = netD(inputv)

            labels = labels.long()
            errD_real = criterion_ce(output, Variable(labels))

            if opt.scale_real:
                errD_real = nb_G * errD_real 
            
            errD_real.backward()
            D_x = output.data.mean()

            if use_bon_disable:

                # train with fake
                errD_fake = 0
                fake_lst = []
                for j, g in enumerate(G_lst):

                    fake_lst.append(g(inputv))
                    
                    labels = Variable(labels).data.float()
                    output = netD(fake_lst[j].detach())

                    if opt.cuda:
                        labels = labels.cuda()

                    errD_fake_temp = criterion_ce(output, Variable(labels.long()))
                    errD_fake += errD_fake_temp

                errD_fake.backward()
                D_G_z1 = output.data.mean()
                
                errD = errD_real + errD_fake
                optimizerD.step()

            else:

                errD = errD_real
                optimizerD.step()

        ############################
        # (2) Update G networks
        ###########################
        if use_bon_disable:

            for j, g in enumerate(G_lst):

                g.zero_grad()        

                output = netD(fake_lst[j])

                # calculate error of l2 of output w.r.t. weights
                l2 = torch.sum(output**2) / output.nelement()
                l2.backward(retain_graph=True)

                # we want to accumulate the gradients for every batch during an epoch
                # these will just be importance weights
                # we scale it to obtain a streaming way of importance weights
                n += 1
                if opt.cuda:
                    importance_weights[j] = (importance_weights[j]*n + g.getGradients().cpu().data.numpy()) / (n+1)
                else:
                    importance_weights[j] = (importance_weights[j]*n + g.getGradients().data.numpy()) / (n+1)

                if opt.clip_imp:
                    importance_weights[j] = np.clip(importance_weights[j], -opt.w_imp, opt.w_imp)

                g.zero_grad()


                # find the synthetic data points that were misclassified
                _, predicted = torch.max(output.data, 1)

                if opt.cuda:
                    ids = (predicted != labels.long())
                    ids = ids.long()
                else:
                    ids = (predicted != labels.type(torch.LongTensor))
                    ids = ids.type(torch.LongTensor)

                indices = []
                for k, index in enumerate(ids):
                    if index==1:
                        indices.append(k)

                indices = torch.LongTensor(indices)

                if indices.size() != ():

                    if opt.cuda:
                        indices = indices.cuda()

                    inputv = Variable(torch.FloatTensor(np.concatenate([inputv.cpu().data.numpy()])))

                    if opt.cuda:
                        inputv = inputv.cuda()

                    mse_l = w_g * mse_loss(inputv[indices,:], fake_lst[j][indices,:])

                    errG =  mse_l
             
                    errG.backward(retain_graph=True)
                    D_G_z2 = output.data.mean()

                    # catastrophic learning schedule
                    # additional loss term for MAS
                    if opt.cuda:
                        params = g.getParameters()
                        errMAS = w_exp**epoch * torch.sum(Variable(torch.FloatTensor(importance_weights[j]).cuda()) * (params-Variable(torch.FloatTensor(theta_star[j]).cuda()))**2)
                        errMAS.backward()

                    else:
                        params = g.getParameters()
                        errMAS = w_exp**epoch * torch.sum(Variable(torch.FloatTensor(importance_weights[j])) * (params-Variable(torch.FloatTensor(theta_star[j])))**2)
                        errMAS.backward()

                    G_optimizers[j].step()

                if opt.cuda:

                    fakes.extend(fake_lst[j].cpu().data.numpy().tolist())
                    y_new.extend(Variable(labels).cpu().data.numpy().tolist())
                
                else:

                    fakes.extend(fake_lst[j].data.numpy().tolist())
                    y_new.extend(Variable(labels).data.numpy().tolist())
    

    # fix the network weights, theta*
    if use_bon_disable:
        for l in range(len(theta_star)):
            if opt.cuda:
                theta_star[l] = G_lst[l].getParameters().cpu().data.numpy() 
            else:
                theta_star[l] = G_lst[l].getParameters().data.numpy() 
    
    try:
        _errG = errG.data[0]
    except:
        _errG=np.inf
        D_G_z2=np.inf


    # Create plots and save them
    if epoch % opt.saving == 0:

        netD.eval()

        nx = 300
        ny = 300
        y_grid = np.linspace(1.5, 5, nx)
        x_grid = np.linspace(3.8, 8.5, ny)
        x1 = []
        x2 = []
        pred = []
        for i in x_grid:
            for j in y_grid:
                x1.append(i)
                x2.append(j) 
                if opt.cuda:
                    pred.append(np.argmax(netD(Variable(torch.FloatTensor(np.array([i,j])).cuda())).cpu().data.numpy()))
                else:
                    pred.append(np.argmax(netD(Variable(torch.FloatTensor(np.array([i,j])))).data.numpy()))

        fakes = np.array(fakes)

        x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
        y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
        
        colors3 = ['brown', 'darkgreen', 'indigo']
        colors2 = ['brown', 'darkgreen', 'indigo']
        colors = ['brown','darkgreen', 'darkblue']
        fig = plt.figure(figsize=(8,8))
        plt.scatter(x1, x2, c=pred, cmap=mpl.colors.ListedColormap(colors), alpha=0.06, edgecolor='none')
        plt.scatter(x[:,0], x[:,1], c=y, cmap=mpl.colors.ListedColormap(colors2), s=100, zorder=1, edgecolor='none', label="Real data points", alpha=1.0)
        if use_bon_disable:
            plt.scatter(fakes[:,0], fakes[:,1], c=y_new, cmap=mpl.colors.ListedColormap(colors3), s=10, zorder=2, label="Synthetic data points", alpha=1.0)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.xlabel('Sepal length', size=15)
        plt.ylabel('Sepal width', size=15)

        plt.legend(loc="lower right")

        directory = "./visualization/bon++_nbG-" + str(nb_G) +"_w-g-" + str(w_g).split(".")[0] + "_w-exp-" + str(w_exp).split(".")[1] + "_lr-" + str(opt.lr) +"_lr-g-" + str(opt.lr_g) + "_w_init_std-" + str(opt.w_init_std) + "_clip-imp-" + str(opt.clip_imp) + "_stoch-" + str(stoch) + "_scale_real-" + str(opt.scale_real) + "/"

        if not os.path.exists(directory):
            os.makedirs(directory)

        fig.savefig(directory+str(epoch)+".png")

    try:
        D_G_z1
    except:
        D_G_z1 = 0.0

    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
          % (epoch, opt.niter, i, len(trainloader),
             errD.data[0], _errG, D_x, D_G_z1, D_G_z2))
    
