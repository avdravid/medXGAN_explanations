#!/usr/bin/env python
# coding: utf-8

# ### Device + Imports

# In[138]:


import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_ubyte
import cv2


device = torch.device('cpu')

# if torch.cuda.is_available():
#     device = torch.device('cuda:0')
    
print(device)


# ### Import Models + Load Weights

# In[139]:


from Generator import Generator
from Classifier import Classifier
from Discriminator import Discriminator


# In[140]:


gen = Generator().to(device)
discr = Discriminator().to(device)
classifier = Classifier().to(device)


gen.load_state_dict(torch.load('generator_weights.pytorch', map_location = device))
discr.load_state_dict(torch.load('discriminator_weights.pytorch', map_location = device))
classifier.load_state_dict(torch.load('classifier_weights.pytorch', map_location = device))
#gen.eval()
classifier.eval()


# ### Load Example Image and Reconstruct

# In[141]:


class SquashTransform:

    def __call__(self, inputs):
        return 2 * inputs - 1
    
data_test = torchvision.datasets.ImageFolder(
    './ex_image',
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((64, 64)),
        SquashTransform()
    ])
)
test_loader = torch.utils.data.DataLoader(
    data_test,
    batch_size=1,
    shuffle=False,
    num_workers=0
)


# In[142]:


image = next(iter(test_loader))


classifier.eval()
plt.figure(figsize=(3, 3))
plt.axis("off")
plt.title("Image")

plt.imshow(
    np.transpose(
        torchvision.utils.make_grid(
            image[0].to(device),
            padding=10,
            normalize=True,
            pad_value=1,
        ).cpu(),
        (1,2,0)
    )
);


clf, _ = classifier(image[0].to(device))
print("Classification:" + str(clf.detach().numpy()[0]))


# In[143]:


from torch.autograd import Variable

init_noisez1 = Variable(torch.randn(
    1, 1000
).to(device), requires_grad = True)

init_noisez2 = Variable(torch.randn(
    1, 100
).to(device), requires_grad = True)

optim = torch.optim.Adam([init_noisez1, init_noisez2], lr=0.001, betas=(0.5, 0.999))  


# ### 1

# In[144]:


for epoch in range(0,1000):
    image = next(iter(test_loader))
    original_image = image[0][0]
    optim.zero_grad()
    sample = gen(init_noisez1,init_noisez2 ).to(device)
    sample = (sample.reshape([1,3,64,64]))
    result,_ = classifier(sample) 
    prob = result[0,1]
    clf, _ = classifier(image[0].to(device))
    
    class_loss = torch.nn.BCELoss()(prob, clf[0,1])
    
    
    original_image =  (original_image.reshape([1,3,64,64]))
    

    loss = 10*torch.mean((original_image - sample)**2) + class_loss

    
    print("E:", epoch+1, "loss:", loss.item())
    loss.backward()
    optim.step()
    
    if (epoch+1) % 100 == 0:
        reconstructed_image = gen(
        init_noisez1, init_noisez2
        ).detach().cpu().view(-1, 3,64, 64)
        
        reconstructed_image = reconstructed_image[0,]
        
        print(result)
        fig=plt.figure(figsize=(5, 5))
        plt.title('Reconstruction')
        plt.axis('off')




        minifig= fig.add_subplot(1, 2, 1)
        minifig.axis('off')
        minifig.title.set_text('Original' )
        original_image = original_image.cpu().view(3, 64, 64)
        original_image = (np.transpose(original_image,(1,2,0))+1)/2
        original_image = (original_image)
        plt.imshow(original_image)


        minifig= fig.add_subplot(1, 2, 2)
        minifig.title.set_text('Reconstructed')
        minifig.axis('off')
        reconstructed_image = np.transpose(reconstructed_image,(1,2,0))
        reconstructed_image = (reconstructed_image + 1)/2
        plt.imshow(reconstructed_image)

        plt.show()


# In[145]:


data_test = torchvision.datasets.ImageFolder(
    './ex_image',
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.Resize((64, 64)),
        SquashTransform()
    ])
)
test_loader = torch.utils.data.DataLoader(
    data_test,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

image = next(iter(test_loader))


classifier.eval()
plt.figure(figsize=(3, 3))
plt.axis("off")
plt.title("Image")

plt.imshow(
    np.transpose(
        torchvision.utils.make_grid(
            image[0].to(device),
            padding=10,
            normalize=True,
            pad_value=1,
        ).cpu(),
        (1,2,0)
    )
);


clf, _ = classifier(image[0].to(device))
print("Classification:" + str(clf.detach().numpy()[0]))


# ### 2

# In[146]:


gen.train()
init_noisez1.detach()
init_noisez2.detach()
optim = torch.optim.Adam(gen.parameters(), lr=0.001, betas=(0.5, 0.999))  
for epoch in range(0,10000):
    image = next(iter(test_loader))
    original_image = image[0][0]
    optim.zero_grad()
    
    sample1 = gen(init_noisez1,init_noisez2 ).to(device)
    sample1 = (sample1.reshape([1,3,64,64]))
    #sample1 = torchvision.transforms.Grayscale()(sample1)
    #sample1 = torch.cat([sample1,sample1,sample1], 1)
    
    result1,_ = classifier(sample1) 
    clf1, _ = classifier(image[0].to(device))
    prob = result1[0,1]
    
    class_loss = torch.nn.BCELoss()(prob, clf1[0,1])
    
    original_image =  (original_image.reshape([1,3,64,64]))
    
    loss = 10*torch.mean((original_image - sample1)**2)#+class_loss
    
    loss.backward()
    optim.step()
    
    
#     #negative
    
    image = next(iter(test_loader))
    original_image = (image[0][0].to(device))
    
    
    mask = torch.zeros([3,64,64]).to(device)
    mask[:, 34:45, 15:26] = 1
    mask[:, 28:40, 34:38] = 1
    masked_im = (mask*original_image).cpu().numpy()#.view(3, 64, 64)

    masked_im[:, 28:40, 34:38] = np.where(masked_im[:, 28:40, 34:38]<-0.5, -0.3, masked_im[:, 28:40, 34:38])
    masked_im[:, 34:45, 15:26] = np.where(masked_im[:, 34:45, 15:26]>-0.2, -0.4, masked_im[:, 34:45, 15:26])

    masked_im = torch.from_numpy(masked_im)
    masked_im = masked_im.view(1,3,64,64)
    result,_ = classifier(masked_im) 
    masked_im = masked_im[0]
    #masked_im = (np.transpose(masked_im,(1,2,0))+1)/2
    ref_im = original_image
    ref_im[:, 28:40, 34:38] = torch.Tensor(masked_im[:, 28:40, 34:38])
    ref_im[:, 34:45, 15:26] = torch.Tensor(masked_im[:, 34:45, 15:26])
    
    
    
    
    
    
    
    
    
    

#     mask = torch.zeros([3,64,64]).to(device)
#     mask[:, 34:45, 15:26] = 1
#     masked_im = (mask*original_image).cpu().view(3, 64, 64)
#     masked_im = np.where(masked_im>-0.2, -0.4, masked_im)
#     #masked_im = (np.transpose(masked_im,(1,2,0))+1)/2
#     ref_im = original_image
#     ref_im[:, 34:45, 15:26] = torch.Tensor(masked_im[:, 34:45, 15:26])
    
    
    optim.zero_grad()
    sample2 = gen(init_noisez1,torch.zeros(1,100) ).to(device)
    sample2 = (sample2.reshape([1,3,64,64]))
    #sample2 = torchvision.transforms.Grayscale()(sample2)
    #sample2 = torch.cat([sample2,sample2,sample2], 1)
    
    result2,_ = classifier(sample2) 
    prob = result2[0,0]
    clf2,_ =  classifier(image[0]) 
    
    class_loss = torch.nn.BCELoss()(prob, clf2[0][1])

    loss =  10*torch.mean((ref_im - sample2)**2)# + class_loss

    loss.backward()
    optim.step()
    
    sample1 = gen(init_noisez1,init_noisez2 ).to(device)
    #sample1 = torchvision.transforms.Grayscale()(sample1)
    #sample1 = torch.cat([sample1,sample1,sample1], 1)
    
    sample2 = gen(init_noisez1,torch.zeros(1,100) ).to(device)
    #sample2 = torchvision.transforms.Grayscale()(sample2)
    #sample2 = torch.cat([sample2,sample2,sample2], 1)
    
    result1,_ = classifier(sample1) 
    result2,_ = classifier(sample2) 
    
     
    
    if epoch>200 and result1[0][0].item() <0.5 and result2[0][0].item()>0.5:
        print(result1)
        print(result2)
        reconstructed_image = gen(
        init_noisez1, init_noisez2
        ).detach().cpu().view(-1, 3,64, 64)
        
        reconstructed_image = reconstructed_image[0,]
        
        print(result1)
        fig=plt.figure(figsize=(5, 5))
        plt.title('Reconstruction')
        plt.axis('off')


        image = next(iter(test_loader))
        original_image = (image[0][0].to(device))

        minifig= fig.add_subplot(1, 3, 1)
        minifig.axis('off')
        minifig.title.set_text('Original' )
        original_image = original_image.cpu().view(3, 64, 64)
        original_image = (np.transpose(original_image,(1,2,0))+1)/2
        original_image = (original_image)
        plt.imshow(original_image)


        minifig= fig.add_subplot(1, 3, 2)
        minifig.title.set_text('Reconstructed')
        minifig.axis('off')
        reconstructed_image = np.transpose(reconstructed_image,(1,2,0))
        reconstructed_image = (reconstructed_image + 1)/2
        plt.imshow(reconstructed_image)

        
        
        reconstructed_image = gen(
        init_noisez1, torch.zeros(1,100)
        ).detach().cpu().view(-1, 3,64, 64)
        result,_ = classifier(reconstructed_image)
        print(result)
        reconstructed_image = reconstructed_image[0,]
        
        minifig= fig.add_subplot(1, 3, 3)
        minifig.title.set_text('Negative')
        minifig.axis('off')
        reconstructed_image = np.transpose(reconstructed_image,(1,2,0))
        reconstructed_image = (reconstructed_image + 1)/2
        plt.imshow(reconstructed_image)

        
        
        
        plt.show()
        
        break
    
    print("E:", epoch+1)#, "loss:", loss.item())
    
    
    
    if (epoch+1) % 1 == 0:
        reconstructed_image = gen(
        init_noisez1, init_noisez2
        ).detach().cpu().view(-1, 3,64, 64)
     
        #reconstructed_image = torchvision.transforms.Grayscale()(reconstructed_image)
        #reconstructed_image = torch.cat([reconstructed_image, reconstructed_image, reconstructed_image], 1)
        
        reconstructed_image = reconstructed_image[0,]
        
        print(result1)
        fig=plt.figure(figsize=(5, 5))
        plt.title('Reconstruction')
        plt.axis('off')


        image = next(iter(test_loader))
        original_image = (image[0][0].to(device))

        minifig= fig.add_subplot(1, 3, 1)
        minifig.axis('off')
        minifig.title.set_text('Original' )
        original_image = original_image.cpu().view(3, 64, 64)
        original_image = (np.transpose(original_image,(1,2,0))+1)/2
        original_image = (original_image)
        plt.imshow(original_image)


        minifig= fig.add_subplot(1, 3, 2)
        minifig.title.set_text('Reconstructed')
        minifig.axis('off')
        reconstructed_image = np.transpose(reconstructed_image,(1,2,0))
        reconstructed_image = (reconstructed_image + 1)/2
        plt.imshow(reconstructed_image)

        
        
        reconstructed_image = gen(
        init_noisez1, torch.zeros(1,100)
        ).detach().cpu().view(-1, 3,64, 64)
        
        #reconstructed_image = torchvision.transforms.Grayscale()(reconstructed_image)
        #reconstructed_image = torch.cat([reconstructed_image, reconstructed_image, reconstructed_image], 1)
        
        result,_ = classifier(reconstructed_image)
        print(result)
        reconstructed_image = reconstructed_image[0,]
        
        minifig= fig.add_subplot(1, 3, 3)
        minifig.title.set_text('Negative')
        minifig.axis('off')
        reconstructed_image = np.transpose(reconstructed_image,(1,2,0))
        reconstructed_image = (reconstructed_image + 1)/2
        plt.imshow(reconstructed_image)

        
        
        
        plt.show()


# In[147]:


# Reconstructed Pos
reconstructed_image = gen(init_noisez1.cpu(), init_noisez2.cpu()).detach().cpu().view(-1, 3, 64, 64)
clfrecpos, _ = classifier(reconstructed_image)
reconstructed_image = reconstructed_image[0,]
reconstructed_image = np.transpose(reconstructed_image,(1,2,0))

# Reconstructed Neg
reconstructed_image_neg = gen(init_noisez1.cpu(), torch.zeros(1,100)).detach().cpu().view(-1, 3, 64, 64)
clfrecneg, _ = classifier(reconstructed_image_neg)
reconstructed_image_neg = reconstructed_image_neg[0,]
reconstructed_image_neg = np.transpose(reconstructed_image_neg,(1,2,0))



#original
image = next(iter(test_loader))
original_image = image[0][0].view(-1,3,64,64).cpu()
clforiginal, _ = classifier(original_image)
original_image = (np.transpose(original_image[0,],(1,2,0))+1)/2





z2_features_reconstructed = abs(reconstructed_image - reconstructed_image_neg)#/reconstructed_image


fig=plt.figure(figsize=(15, 6))
plt.title('Reconstruction')
plt.axis('off')




minifig= fig.add_subplot(1, 4, 1)
minifig.axis('off')
minifig.title.set_text('Original Positive\n' + str(clforiginal.detach().cpu().numpy()[0]))
plt.imshow(original_image)




minifig= fig.add_subplot(1, 4, 2)
minifig.title.set_text('Reconstructed Positive\n' +str(clfrecpos.detach().cpu().numpy()[0] ))
minifig.axis('off')
reconstructed_image = (reconstructed_image + 1)/2
plt.imshow(reconstructed_image)







minifig= fig.add_subplot(1, 4, 3)
minifig.title.set_text('Reconstructed Negative\n'+str(clfrecneg.detach().cpu().numpy()[0]))
minifig.axis('off')
reconstructed_image_neg = (reconstructed_image_neg + 1)/2
plt.imshow(reconstructed_image_neg)



minifig= fig.add_subplot(1, 4, 4)
minifig.title.set_text('Absolute Difference')
minifig.axis('off')
z2_features_reconstructed = (z2_features_reconstructed) #+ 1)/2
#z2_features_reconstructed = z2_features_reconstructed/torch.max(z2_features_reconstructed)
#z2_features_reconstructed = np.where(z2_features_reconstructed<0,0, z2_features_reconstructed)
plt.imshow(z2_features_reconstructed)









plt.show()


# In[153]:


delta = 0.1

fig=plt.figure(figsize=(24, 10))
#plt.title('Latent Space Interpolation')
#fig.suptitle('Latent Space Interpolation', fontsize=20)
plt.axis('off')


for i in range(11):
    minifig= fig.add_subplot(2, 6, i+1)
    image = gen(init_noisez1.cpu(), delta*i*init_noisez2.cpu()).detach().cpu().view(-1, 3, 64, 64)
    class_result,_ = classifier(image)
    class_result = class_result.detach().numpy()
    minifig.axis('off')
    
    if (i==0):
        minifig.title.set_text("Negative:"+"\n"+"Class: "+str(class_result[0]))
    elif (i==10):
        minifig.title.set_text("Final Positive:"+"\n"+"Class: "+str(class_result[0]))
    else: 
        minifig.title.set_text("Class: "+str(class_result[0]))
    plt.imshow((np.transpose(image[0],(1,2,0))+1)/2)
   
  

 #difference image
minifig= fig.add_subplot(2, 6, 12)
minifig.title.set_text('Positive-Negative Difference')
minifig.axis('off')
pos = gen(init_noisez1.cpu(), init_noisez2.cpu()).detach().cpu().view(-1, 3, 64,64)
neg = gen(init_noisez1.cpu(), torch.zeros(1,100)).detach().cpu().view(-1, 3,64, 64)
diffimg = 3*abs(pos - neg)
diffimg = np.transpose(diffimg[0], (1,2,0))
diffimg = np.dot(diffimg[...,:3], [0.2989, 0.5870, 0.1140])
plt.imshow(diffimg, cmap = "gray");


# In[175]:


delta = 0.01

#fig=plt.figure(figsize=(24, 10))
#plt.title('Latent Space Interpolation')
#fig.suptitle('Latent Space Interpolation', fontsize=20)
#plt.axis('off')


for i in range(100):
    #minifig= fig.add_subplot(2, 6, i+1)
    image = gen(init_noisez1.cpu(), delta*i*init_noisez2.cpu()).detach().cpu().view(-1, 3, 64, 64)
    class_result,_ = classifier(image)
    class_result = class_result.detach().numpy()
    minifig.axis('off')
    
    #if (i==0):
    #    minifig.title.set_text("Negative:"+"\n"+"Class: "+str(class_result[0]))
    #elif (i==10):
    #    minifig.title.set_text("Final Positive:"+"\n"+"Class: "+str(class_result[0]))
    #else: 
    #    minifig.title.set_text("Class: "+str(class_result[0]))
    image =(image[0]+1)/2
    image = torchvision.transforms.Grayscale()(image)
    image = np.transpose(image, (1,2,0))
    plt.imshow(image, cmap = "gray")
    plt.axis("off")
    plt.savefig("brain_"+str(i)+".jpg", bbox_inches='tight')
    

 #difference image
minifig= fig.add_subplot(2, 6, 12)
minifig.title.set_text('Positive-Negative Difference')
minifig.axis('off')
pos = gen(init_noisez1.cpu(), init_noisez2.cpu()).detach().cpu().view(-1, 3, 64,64)
neg = gen(init_noisez1.cpu(), torch.zeros(1,100)).detach().cpu().view(-1, 3,64, 64)
diffimg = 3*abs(pos - neg)
diffimg = np.transpose(diffimg[0], (1,2,0))
diffimg = np.dot(diffimg[...,:3], [0.2989, 0.5870, 0.1140])
plt.imshow(diffimg, cmap = "gray");
plt.axis("off")
plt.savefig("brain_diff.jpg", bbox_inches='tight')


# In[166]:


delta = 0.1

#fig=plt.figure(figsize=(24, 10))
#plt.title('Latent Space Interpolation')
#fig.suptitle('Latent Space Interpolation', fontsize=20)
#plt.axis('off')


for i in range(11):
    
    image = gen(init_noisez1.cpu(), delta*i*init_noisez2.cpu()).detach().cpu().view(-1, 3, 64, 64)
    class_result,_ = classifier(image)
    class_result = class_result.detach().numpy()
    plt.imshow((np.transpose(image[0],(1,2,0))+1)/2)
    plt.axis("off")
    plt.savefig("brain_"+str(i)+".jpg", bbox_inches='tight')
    
pos = gen(init_noisez1.cpu(), init_noisez2.cpu()).detach().cpu().view(-1, 3, 64,64)
neg = gen(init_noisez1.cpu(), torch.zeros(1,100)).detach().cpu().view(-1, 3,64, 64)
diffimg = 3*abs(pos - neg)
diffimg = np.transpose(diffimg[0], (1,2,0))
diffimg = np.dot(diffimg[...,:3], [0.2989, 0.5870, 0.1140])
plt.imshow(diffimg, cmap = "gray");
plt.savefig("brain_diff.jpg", bbox_inches='tight')


# In[167]:


import cv2
from skimage import img_as_ubyte

diff = img_as_ubyte(diffimg/np.max(diffimg))

heatmap_img = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
superimposed_img = heatmap_img * 0.002 + original_image.numpy()

final_img = superimposed_img[:,:,::-1]
final_img = final_img/np.max(final_img)
plt.imshow(final_img);
plt.axis("off")
plt.savefig("brain_heatmap.jpg", bbox_inches='tight')


# ### Latent Integrated Gradients

# In[159]:


neg = gen(init_noisez1.cpu(), torch.zeros(1,100).cpu()).view(-1, 3, 64, 64).clone().detach().requires_grad_(True)
output,_ = classifier(neg)
target_label_idx = torch.argmax(output, 1).item()
index = np.ones((output.size()[0], 1)) * target_label_idx
index = torch.tensor(index, dtype=torch.int64)
output = output.gather(1, index)


# In[164]:


delta = 0.01

m = 101

mask = torch.zeros((3,64,64))

for i in range(m):
    image = gen(init_noisez1.cpu(), delta*i*init_noisez2.cpu()).detach().cpu().view(-1, 3, 64, 64).clone().detach().requires_grad_(True)
    output,_ = classifier(image)
    
    target_label_idx = torch.argmax(output, 1).item()
    index = np.ones((output.size()[0], 1)) * target_label_idx
    index = torch.tensor(index, dtype=torch.int64)
    output = output.gather(1, index)
    
    classifier.zero_grad()
    output.backward()
    mask = (mask+image.grad[0])
    minifig.axis('off')
    
diffimg = pos - neg
final_mask = diffimg[0] * (mask/m)
final_mask = final_mask/torch.max(final_mask).item()
final_mask = final_mask.detach().numpy()
final_mask = (np.transpose(final_mask,(1,2,0)))
final_mask = np.dot(final_mask[...,:3], [0.2989, 0.5870, 0.1140])
plt.imshow(5*abs(final_mask), cmap = "gray");
plt.axis("off")
plt.savefig("brain_LIG.jpg", bbox_inches='tight')


# In[55]:


classifier.eval()
image = next(iter(test_loader))
original_image = (image[0][0].to(device))

mask = torch.zeros([3,64,64]).to(device)
mask[:, 34:45, 15:26] = 1
masked_im = (mask*original_image).cpu().view(3, 64, 64)
masked_im = np.where(masked_im>-0.2, -0.4, masked_im)
masked_im = torch.from_numpy(masked_im)
masked_im = masked_im.view(1,3,64,64)
result,_ = classifier(masked_im) 
print(result)
masked_im = masked_im[0]
#masked_im = (np.transpose(masked_im,(1,2,0))+1)/2
ref_im = original_image
ref_im[:, 34:45, 15:26] = torch.Tensor(masked_im[:, 34:45, 15:26])
plt.imshow((np.transpose(ref_im, (1,2,0))+1)/2)


# In[109]:


classifier.eval()
image = next(iter(test_loader))
original_image = (image[0][0].to(device))

mask = torch.zeros([3,64,64]).to(device)
mask[:, 34:45, 15:26] = 1
mask[:, 28:40, 34:38] = 1
masked_im = (mask*original_image).cpu().numpy()#.view(3, 64, 64)

masked_im[:, 28:40, 34:38] = np.where(masked_im[:, 28:40, 34:38]<-0.5, -0.3, masked_im[:, 28:40, 34:38])
masked_im[:, 34:45, 15:26] = np.where(masked_im[:, 34:45, 15:26]>-0.2, -0.4, masked_im[:, 34:45, 15:26])

masked_im = torch.from_numpy(masked_im)
masked_im = masked_im.view(1,3,64,64)
result,_ = classifier(masked_im) 
masked_im = masked_im[0]
#masked_im = (np.transpose(masked_im,(1,2,0))+1)/2
ref_im = original_image
ref_im[:, 28:40, 34:38] = torch.Tensor(masked_im[:, 28:40, 34:38])
ref_im[:, 34:45, 15:26] = torch.Tensor(masked_im[:, 34:45, 15:26])
plt.imshow((np.transpose(ref_im, (1,2,0))+1)/2)


# In[ ]:




