## Autoencoding beyond pixels using a learned similarity metric

### VAE-GAN

* ENCODE - GENERATE - COMPARE
* vae encoder - decode/generator - discriminator



Generative model은 reconstruction error로 MSE 같은 element-wise error를 사용

이는 Human perception하고는 차이가 있음. 그래서 image generation에 적합하지 않음

그래서 element-wise error 대신에 좀 더 high level인 feature-wise error를 사용해보자.

이때, High level error를 하나의 metric으로 정하기 보다는, similarity를 학습하도록 해보자

=> HOW?

GAN의 discriminator가 학습한다.

그래서 VAE의 decoder와 GAN의 generator를 하나로 합치고, 마지막에 Discriminator를 붙인다. 





#### 학습을 더 잘하고자 쓴 방법

* Limiting error signals to relevant networks
  * Using the loss function "L = L_prior + L_dis_llike+L_gan", we train both a VAE and a GAN simultaneously.
  * This is possible because we do not update all network parameters wrt. the combined loss.
  * In particular, Dis should not try to minimize L_dis_llikeas this would collapse the discriminator to 0.
  * We also observe better results by not backpropagating the error signal from L_GAN to Enc.
* Weighting VAE vs GAN
  * As Dec receives an error signal from L_dis_llike and L_gan, we use a parameter r to weight the ability to reconstruct vs fooling the discriminator.
  * This can also be interpreted as weighting style and content.
  * Rather than applying r to the entire model, we perform the weighting only when updating the parameter of Dec



