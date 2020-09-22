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
  
<img src="https://github.com/Amchuz/Generative-Adversarial-Networks-GAN/blob/master/algorithm.png">
  
It continues till the iteration ends. The algorithm starts with two things, sample a batch of noise vectors and a batch of images from the dataset. Then it is going to use the objective function from the above discussion in order to update the parameters of the discriminator by doing gradient descent w.r.t it's parameters. During the updation of Discriminator, the generator part is fixed. Once we have updated the discriminator for a couple of time steps, we will freeze the weights of discriminator, and goes to another part where the generator is trained. Prior to that, we resample the batch of noise vectors. We generate images with them and then, apply gradient descent on the second part of the objective function in order to update the parameters of the generator. Well, that's the algorithm.

