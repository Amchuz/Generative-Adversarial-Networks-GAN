# Generative Adversarial Networks (GAN)
  
## General Introduction on GAN
  
GAN consist of 2 networks. \
A **Generator** and a **Discriminator.** \
**Generator** generates new images from the images fed and make fake samples. **Discriminator** finds the fake and real samples. With the loss function, we can update both discriminator and generator network until we get a good result.
    
<img src="https://github.com/Amchuz/Generative-Adversarial-Networks-GAN/blob/master/GAN.png">
  
## GAN Optimization objective
  
Generator and Discriminator are actually playing a Mini-Max game. Generator is trying to fool discriminator. Discriminator is always trying to be right. Discriminator is trying to distiguish real images from generated ones. 
  
<img src="https://github.com/Amchuz/Generative-Adversarial-Networks-GAN/blob/master/Mini-max%20game.png">

That is, we are **minizing** the loss fuction w.r.t the generator parameters and **maximizing** the loss function w.r.t the discriminator parameters.\
Ok, let's go little bit more into details. 
  
<img src="https://github.com/Amchuz/Generative-Adversarial-Networks-GAN/blob/master/Discriminator.png">
  
Discriminator network outputs a single scaled value, **D(X)** per image which indicates how likely it is the image **x**  is infact a real images coming from the dataset. \
And it is same for the Generator images.
  
<img src="https://github.com/Amchuz/Generative-Adversarial-Networks-GAN/blob/master/Generator%20in%20Discriminator.png">
  
The Generator images are **G(z), z** here is the noise vector and **G** is the Generator network. And hence, Discriminator outputs the score for G(z) as D(G(z)).
While training, we want the discriminator to recognize real image x as real. So we want to output a high value close to 1. 
  
<img src="https://github.com/Amchuz/Generative-Adversarial-Networks-GAN/blob/master/realimage.png">
  
At the same time, we also want to recognize fake images G(z) as fake and therefore, outputting a low value close to zero.
  
<img src="https://github.com/Amchuz/Generative-Adversarial-Networks-GAN/blob/master/fakeimage.png">
  
<img src="https://github.com/Amchuz/Generative-Adversarial-Networks-GAN/blob/master/filoss.png">
  
The first part (Ex⇠pdata(x)[logD(x)]) does not depend on the parameters of the Generator. So when optimizing for the Generator, we only have to do it for the second part (Ez⇠pz(z)[log(1-D(G(z))]).
  
Got it ? Now we can use this idea to generate the algorithm.
  
<img src="https://github.com/Amchuz/Generative-Adversarial-Networks-GAN/blob/master/algorithm.png" width=600 height=450>
  
It continues till the iteration ends. The algorithm starts with two things, sample a batch of noise vectors and a batch of images from the dataset. Then it is going to use the objective function from the above discussion in order to update the parameters of the discriminator by doing gradient descent w.r.t it's parameters. During the updation of Discriminator, the generator part is fixed. Once we have updated the discriminator for a couple of time steps, we will freeze the weights of discriminator, and goes to another part where the generator is trained. Prior to that, we resample the batch of noise vectors. We generate images with them and then, apply gradient descent on the second part of the objective function in order to update the parameters of the generator. Well, that's the algorithm. \
In practical, there are some additional steps that can be used to converge it smoothly because it tends to be a little unstable. Now a days, there are a wide variety of objective functions that is used to train these GANs but all of them are build on the same core idea that we have discussed here. To know more about the various objective fuctions, check out <a href="http://hunterheidenreich.com/blog/gan-objective-functions/">the blog.</a> Now, let us look at two final ideas that we are going to be using in our generative model.
  
## 1.Progressive Growing
  
<img src="https://github.com/Amchuz/Generative-Adversarial-Networks-GAN/blob/master/progressive%20growing.gif" width=600 height=300>
  
This was an idea published by NVIDIA. We basically start from a generative model that generates very small images with low resolution and at the same time, the discriminator also gets to discriminate very low resolution images. This make the entire process simple. So this network is very stable and it converges quickly. Once that networ has stabilized, we then simply add an additional layer to both the generator and discriminator architecture which works at a slightly higher resolution and we keep on training. Instead of just adding this layers one shot, we basically do it gradually by blending the previous layer towards the higher resolution one.

## 2.Style-GAN architecture
  
In **traditional generator architecture**, it gets a random noise sample as an input and it is fed into a bunch of upsampling and convoutional layers until we get an image. **StyleGAN generator architecture** is slightly different. It has a mapping network. This mapping network takes the noise vector **z** and transforms it into a different vector called **w**. w vector doesn't have to be gaussian anymore. The distributions of w can be whatever generator want it to be. After that, the generator architecture doesn't start from noise vector anymore. It starts from a constant vector. This constant vector is optimized during training. The output of the mapping layer **w** is plugged into multiple layers of the generative architecture using a blending layer called **AdaIN.** During training also we add noise to these parameters. 
  
<img src="https://github.com/Amchuz/Generative-Adversarial-Networks-GAN/blob/master/traditional.png"> <img align="right" src="https://github.com/Amchuz/Generative-Adversarial-Networks-GAN/blob/master/style-based%20generator.png">
  
With there two tricks combined, StyleGAN is created. 
  
<img src="https://github.com/Amchuz/Generative-Adversarial-Networks-GAN/blob/master/styleGAN.gif">
  
## Playing with StyleGAN
  
- StyleGAN's latent space has Structure.
- This structure is learned fully **unsupervised** during the adversarial training processes.
- **Core Idea** : Instead of manipulation images in the pixel domain, let's manipulate them in the latent space.
- To do this, we first need to find a query image inside styleGAN's latent space of the generator. 
  
### How to find latent vector z for a query image ?
  
We could try randomly sampling the whole unch of these latent vectors and see which one is closest but, that will take a long time. \
One of the approach we could use is to randomly start from any vector z and generate and image and then compare that image to the query image. We could define a simple loss function L2 loss (pixel by pixel difference) and gradient descent to the pixel loss backpropagate generator model into the latent code and update i
