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


