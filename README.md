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
  
## 1. Progressive Growing
  
<img src="https://github.com/Amchuz/Generative-Adversarial-Networks-GAN/blob/master/progressive%20growing.gif" width=600 height=300>
  
This was an idea published by NVIDIA. We basically start from a generative model that generates very small images with low resolution and at the same time, the discriminator also gets to discriminate very low resolution images. This make the entire process simple. So this network is very stable and it converges quickly. Once that networ has stabilized, we then simply add an additional layer to both the generator and discriminator architecture which works at a slightly higher resolution and we keep on training. Instead of just adding this layers one shot, we basically do it gradually by blending the previous layer towards the higher resolution one.

## 2. Style-GAN Architecture
  
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
  
We could try randomly sampling the whole bunch of these latent vectors and see which one is closest but, that will take a long time. \
One of the approach we could use is to randomly start from any vector z and generate and image and then compare that image to the query image. We could define a simple loss function L2 loss (pixel by pixel difference) and gradient descent to the pixel loss backpropagate generator model into the latent code and update the vector. And that way, finally we can find optinal latent vector.
  
Unfortunately, this doesn't really work ! Sad :(
  
The L2 optimization objective start going in te direction of the image but far before it gets there it is going to get stuck in a very bad local minimum of an image that doesn't look like the query image.
  
We can think about a different apporoach. It is the idea of using a pretrained image classifier as a lens to look at the pixels. Rather than optimizing the L2 loss directly in the pixel space we are going to send both the output of our generator and the query image through a **pre-trained VGG network** that was trained to classify ImageNet images. But instead of actually going all the way to the last layer onto the classification, we are going to cut off the head of that network and extract a feature vector somewhere inside the last fully connected layers of that classifier. 
  
<img src="https://github.com/Amchuz/Generative-Adversarial-Networks-GAN/blob/master/VGG-16.png">
  
We send these images through the pretrained classifier, we extract a feature vector at some of the last fully connected layers in the network and this gives us a high-level semantic representation of what is in the image. It turns out that if we actually do gradient descent on this feature vector rather than on the pixels of the image, our approach does work.
  
<img src="https://github.com/Amchuz/Generative-Adversarial-Networks-GAN/blob/master/vector2query.png">
  
But, this apporach is really **slow.** What if there is a way to make a really good guess of the starting point where we start our search in the latent space. With that idea, we can make a dataset. First we sample a whole bunch of random vectors and send them through the generator and will generate faces. Once we have that dataset we can train a ResNet to go from the images to their respective latent code.
  
<img src="https://github.com/Amchuz/Generative-Adversarial-Networks-GAN/blob/master/ResNet.png">
  
So, let's go through the whole pipeline now. We take a query image and we send it through the ResNet and this network gives us an initial estimate of the latent space vector in the StyleGAN network. We then take that latent vector and send it through the generator which gives us an image. On this image, we apply a pretrained VGG network in order to extract features from it and we do the same for our query image. In that feature space we start doing gradient descent. We minimize the L2 distance in this feature space and send those gradients through the generator model all the way back into our latent code. During this optimization process the generator weights itself are completely fixed and only thing we are updating is the latent code at the input of our generator. 
  
<img src="https://github.com/Amchuz/Generative-Adversarial-Networks-GAN/blob/master/pipeline1.png" width=450 height=250> <img src="https://github.com/Amchuz/Generative-Adversarial-Networks-GAN/blob/master/pipeline2.png" width=450 height=250> 
  
#### Additional Tweaks :
- Which specific layer of the VGG network are you using as that semantic feature vector. 
- Possibility of apply a face- mask before computing losses.
- Add a L1-penalty to the latent code to keep it close to StyleGAN's concept of a face.
- Learning rate decay during optimization.
  
It is simple, right ? Now, it's time for a little bit of complication.
  
In order to play with the latent space we need another dataset. Because now, we want to start messing with specific attributes of those faces like age gender smiling etc. We ae going to randomly sample a whole bunch of these latent vectors and send them through the generator and get our faces. Then we are going to apply a pretrained classifier that was trained to recognize a bunch of these attributes. We can hand label these images with any kind of attribute we care about. StyleGAN latent space is actually a 512 dimensional space. It is really complicated. What we really care about is how a certain direction in that latent space changes the face that comes out of the generative model.
  
<img src="https://github.com/Amchuz/Generative-Adversarial-Networks-GAN/blob/master/direction.png" width=450 height=300>
  
With the data set that we just created we can basically put all of those faces at their respective locations and we can start looking at all of the attributes that we have collected. All of those attributes are quite well separable by a relatively simple linear hyperplane in that latent space. 
  
<img src="https://github.com/Amchuz/Generative-Adversarial-Networks-GAN/blob/master/hyperplane.png" width=450 height=300>
  
Once we found that hyperplane if we take the normal with respect to that hyperplane, this direction in the latent space basically tells us how can I make a face more female (in the given picture). 
  
<img src="https://github.com/Amchuz/Generative-Adversarial-Networks-GAN/blob/master/normal.png" width=450 height=300>
  
