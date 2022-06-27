import os
import argparse
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_ubyte
import cv2


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')    
print(device)

def add_args(parser):
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs per prune iteration")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--root_dir", type=str, default='', help='Root Directory of Dataset')
    parser.add_argument("--classifier_root_dir", type=str, default='', help='Location of Classifier Weights')
    return parser

parser = argparse.ArgumentParser()
parser = add_args(parser)
args = parser.parse_args()
print("Arguments : " , args)






### Initialize Models
from Generator import Generator
from Classifier import Classifier
from Discriminator import Discriminator


# weight initialization
def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


# Initialize Models
netG = Generator().to(device)
netD = Discriminator().to(device)

netG.apply(weights_init)
netD.apply(weights_init)

classifier = Classifier().to(device)
classifier.load_state_dict(torch.load(args.classifier_root_dir, map_location = device))
classifier.eval()

for parameter in classifier.parameters():
    parameter.requires_grad = False

### Data

class SquashTransform:
    def __call__(self, inputs):
        return 2 * inputs - 1
     
BATCH_SIZE = args.batch_size
data_train = torchvision.datasets.ImageFolder(
    args.root_dir + '/Training',
    transform=torchvision.transforms.Compose([
         torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.ToTensor(),
        SquashTransform()
    ])
)


data_val = torchvision.datasets.ImageFolder(
    args.root_dir + '/Testing',
    transform=torchvision.transforms.Compose([
         torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.ToTensor(),
        SquashTransform()
    ])
)

print(len(data_train))
print(len(data_val))



train_loader = torch.utils.data.DataLoader(
    data_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0, 
    drop_last=True
)


val_loader = torch.utils.data.DataLoader(
    data_val,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


classes = data_train.classes
print(data_val.classes)
print(classes)


### Optimizer

optimizerD = torch.optim.Adam(
    netD.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)

optimizerG = torch.optim.Adam(
    netG.parameters(),
    lr=0.0002,
    betas=(0.5, 0.999)
)


## Global vars
nz1 = 1000
nz2 = 100
nl = 2

### Misc
num_examples = 10

fixed_noise = torch.randn(
    num_examples, nz1
).to(device)

fixed_noise_train = torch.randn(
    5, nz1
).to(device)


real_labels = torch.ones(BATCH_SIZE, 1).to(device)
fake_labels = torch.zeros(BATCH_SIZE, 1).to(device)

c1 = torch.nn.BCELoss()


### Categorical Cross-Entropy 
def c2(input, target):

    _, labels = target.max(dim=1)

    return torch.nn.CrossEntropyLoss()(input, labels)



### One Hot Encoding

def encodeOneHot(labels):
    ret = torch.FloatTensor(labels.shape[0], nl)
    ret.zero_()
    ret.scatter_(dim=1, index=labels.view(-1, 1), value=1)
    return ret


fixed_conditions = encodeOneHot(
    torch.randint(
        0,
        nl,
        (4, 1)
    )
).to(device)

### Discrete to Continuous Class Code

def condition_to_latent_vec(conditions):
    latent_vecs = torch.zeros((conditions.shape[0], nz2))
    
    for i in range (conditions.shape[0]):
        if conditions[i] == 0:
            latent_vecs[i,:]= torch.zeros((1, nz2))
        else: 
            latent_vecs[i,:] = torch.randn((1, nz2))
            
    latent_vecs = latent_vecs.to(device)
    
    return latent_vecs

conditions_ex = torch.randint(0,nl,(num_examples, 1))
fixed_conditions = condition_to_latent_vec(conditions_ex).to(device)
fixed_conditions_train_neg = torch.zeros((5,nz2)).to(device)
fixed_conditions_train_pos = torch.randn((5,nz2)).to(device)


### Train Discriminator Function
def trainD(images, labels):

    real_images = images.to(device)
    real_conditions = encodeOneHot(labels).to(device)

    
    
    fake_conditions_unencoded = torch.randint(
            0,
            nl,
            (BATCH_SIZE, 1)
        )
    
    fake_conditions = encodeOneHot(
        fake_conditions_unencoded
    ).to(device)
   
    
    fake_conditions_latent_vec =  condition_to_latent_vec(fake_conditions_unencoded)
    

    fake_images = netG(
        torch.randn(
            BATCH_SIZE, nz1
        ).to(device),
        fake_conditions_latent_vec     
    )
    
    

    optimizerD.zero_grad()

    real_valid = netD(real_images)
    fake_valid = netD(fake_images)
    
    l_s = c1(
        real_valid, real_labels
    ) + c1(
        fake_valid, fake_labels
    )


    d_loss = l_s

    d_loss.backward()

    optimizerD.step()

    return d_loss


### Train Generator Function
def trainG(labels):
    fake_conditions_latent_vec =  condition_to_latent_vec(labels)
    conditions = encodeOneHot(labels).to(device)

    z = torch.randn(
        BATCH_SIZE, nz1
    ).to(device)

    netG.zero_grad()

    sample = netG(z, fake_conditions_latent_vec)

    
    valid_outputs = netD(sample)
    clf_outputs,_ = Classifier(sample)

    ls = c1(valid_outputs, real_labels)
    lc = c2(clf_outputs, conditions)

    loss = 2*lc + ls
    

    loss.backward()

    optimizerG.step()

    return loss



### Training Loop

from math import sqrt

import warnings
warnings.filterwarnings("ignore")


for epoch in range(5000):

    d_loss = 0
    g_loss = 0
    
    for i, (images, labels) in enumerate(train_loader):

        if i == num_steps:
            break

        for k in range(1):

             d_loss += trainD(images, labels)

        g_loss = trainG(labels)

    
    if epoch % 1 == 0:
        print(
            "E:{}, G Loss:{}, D Loss:{}".format(
                epoch+1,
                g_loss / num_steps,
                d_loss / num_steps / 1
            )
        )

        

        generated_neg = netG(fixed_noise_train,fixed_conditions_train_neg).to(device)
        classifier_neg_results,_ = Classifier(generated_neg)
        classifier_neg_results = classifier_neg_results.detach().cpu().numpy()
        generated_neg = generated_neg.detach().cpu().view(-1,3,64,64)
        
        
        
        generated_pos = netG(fixed_noise_train,fixed_conditions_train_pos).to(device)
        classifier_pos_results,_ = Classifier(generated_pos)
        classifier_pos_results = classifier_pos_results.detach().cpu().numpy()
        generated_pos = generated_pos.detach().cpu().view(-1,3, 64, 64)

        fig=plt.figure(figsize=(15, 2))
        plt.title('Negative')
        plt.axis('off')
        for i in range(1,6):
            minifig= fig.add_subplot(1, 5, i)
            minifig.axis('off')
            #_, label = torch.max(fixed_conditions[i-1], dim = 0)
            #minifig.title.set_text('Label: {}'.format(label))
            image = np.transpose(generated_neg[i-1,:,:,:],(1,2,0))
            image = (image + 1)/2
            minifig.text(0,75, classifier_neg_results[i-1, :], size = 'small')
            minifig.imshow(image)
            
            
        fig=plt.figure(figsize=(15, 2))
        plt.title('Positive')
        plt.axis('off')
        for i in range(1,6):                 
            minifig= fig.add_subplot(1, 5, i)
            minifig.axis('off')
            #_, label = torch.max(fixed_conditions[i-1], dim = 0)
            #minifig.title.set_text('Label: {}'.format(label))
            image = np.transpose(generated_pos[i-1,:,:,:],(1,2,0))
            image = (image + 1)/2
            minifig.text(0,75, classifier_pos_results[i-1, :], size = 'small')
            minifig.imshow(image)
        
        plt.show()
        plt.savefig("brain_epoch"+str(epoch)+".jpg", bbox_inches='tight')
        
        
        # Save checkpoint
        torch.save(netG.state_dict(), './gen64.pytorch')
        torch.save(netD.state_dict(), './discr64.pytorch')
        torch.save(optimizerG.state_dict(), './optGen.pytorch')
        torch.save(optimizerD.state_dict(), './optDisc.pytorch')
