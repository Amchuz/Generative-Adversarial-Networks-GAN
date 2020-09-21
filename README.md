# Generative Adversarial Networks (GAN)
  
## General Introduction on GAN
  
GAN consist of 2 networks. \
A **Generator** and a **Discriminator.** \
**Generator** generates new images from the images fed and make fake samples. **Discriminator** finds the fake and real samples. With the loss function, we can update both discriminator and generator network until we get a good result.
  
  
<img src="https://github.com/Amchuz/Generative-Adversarial-Networks-GAN/blob/master/GAN.png">
  
## GAN Optimization objective
  
Generator and Discriminator are actually playing a Mini-Max game. Generator is trying to fool discriminator. Discriminator is always trying to be right. Discriminator is trying to distiguish real images from generated ones. 
  
<img src="https://github.com/Amchuz/Generative-Adversarial-Networks-GAN/blob/master/Mini-max%20game.png">

