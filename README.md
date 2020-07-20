# ProgressiveGAN-SR
Progressive GAN Super Resolution model inspired by the SRGAN and Progressive GAN from https://arxiv.org/bs/1609.04802 and https://arxiv.org/abs/1710.10196 respectively

Utlilizes:

*  Minibatch Standard Deviation
*  Fade in layers to smoothen transition between dimensions

## Normalization in Generator and Discriminator
*  Need to implement equalized learning rate
*  PixelWise Normalization after every conv 3x3 in the Generator
*  implements WGAN loss -> need to transition into GP ( Gradient Penalty )
*  Generator loss includes both critic loss and perceptual loss (VGG loss)