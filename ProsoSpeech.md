# ProsoSpeech: Enhancing Prosody with Quantized Vector Pretraining in Text-To-Speech

## Overview

[논문링크](https://arxiv.org/pdf/2202.07816.pdf)

* Pitch extractor의 고질적인 error를 없애고

  * Errors: voiced, unvoiced decision/inaccuracy of F0 values

* Prosody attributes 간의 연관성을 반영하는 

* Prosody modeling 방법을 제시

  => Prosody encoder, Latent Prosody Vector Predictor를 제시

* 특히, LPV predictor에서 text로부터 prosody를 modeling하는 방법을 제시
  * **Reference speech 없이 expressive speech 를 합성할 수 있게 함** 
  * 이때 부족한 학습 데이터를 보완하기 위해 LPV predictor를 pretrain->finetune하는 방법 제시
* Prosody encoder에서는 word-level vector quantization bottleneck을 이용하여 speech로부터 prosody를 분리한다. 
  * GST 의 reference encoder와 같은 역할
* LPV predictor self-attention based autoregressive 구조를 이용하여 LPV sequence를 통해 prosody 를 모델링



## Abstract

* 현재 Prosody modeling의 문제점을 지적하고 이에 대한 해결 방안을 제시한다.

* 문제점
  1. 현재 사용하는 Pitch 추출 방법은 error가 생길 수 밖에 없다. 그렇기 때문에 prosody modeling의 성능을 저해한다.
  2. Prosody attributes(energy, pitch, duration ..)들은 서로 dependent하고 prosody를 형성할 때도 서로 상호작용한다. 
  3. prosody는 너무나 다양하다. 그런데 이러한 prosody를 다 modeling하기에 현재 데이터들은 너무 부족하다. 데이터가 부족해서 full distribution을 shaping하기 어렵다.
* 제안하는 해결책: **"Enhance prosody with latent prosody vector"**
  1. word-level prosody encoder : Speech의 low-frequency band를 quantize하고, Prosody attributes를 LPV에 압축한다.
  2. LPV predictor: word sequence를 받아서 LPV를 예측함. (pretrain with large scale text and low-quality speech data, and finetune with high-quality tts dataset)



> Vector Quantization ([참고링크1](https://hyunlee103.tistory.com/30), [참고링크2](https://datacrew.tech/vector-quantization))
>
> - 벡터 양자화는 프로토 타입 벡터의 분포에 의해 확률 밀도 함수를 모델링하는 양자화 방법
> - Quantization이 되면, 연속값이 이산화되어 counting을 통해 pmf로 모델링이 가능해진다. 
> - 이 과정은 codebook, codeword, clustering algorithm, distance metric으로 구성된다. 
>   - codebook : vocabulary를 구성하는 symbol들의 set
>   - codeword(Prototype vector): codebook 안에 있는 각각의 symbol 
>     - V = {v1, v2, ... vn } 에서 v1, v2 ... 각각이 codeword
>   - clustering algorithm : codebook을 만드는 알고리즘. 
>     - 학습set내의 모든 자질 벡터들을 256개의 class로 clustering.
>     - 그리고 나서 이 cluster에서 대표적인 자질 벡터를 고르고, 이를 그 cluster에 대한 codeword로 삼는다.
>     - 주로, K-means clustering이 자주 사용된다.
>     - distrance metric : codebook을 만들고 나면, 각각의 Input 자질 벡터들을 256개의 codeword와 비교하고, distance metric에 의해 가장 가까운 codeword 하나를 선택한다.
> - 장점: 정확도가 높고, 간단하다.
> - 단점 : 
>   - class수 가 한정적이고, codebook  사이즈가 고정되어 있다. 때문에 flexibility가 부족하다.
>   - 내부 구조가 불투명하다. class를 형성하는 과정에서 각 데이터의 특성이 반영되지 않고, 단순하게 distance metric을 기반으로 clustering한다.
> - 즉, Quantization은 적은 양의 정보를 가지고 데이터를 표현하는 방법 => bottleneck으로 작용할 수 있음



## Introduction

* 기존 prosody modeling 방법은 크게 2가지로 나뉜다.

  1. GST 계열 : auto encoder-like 구조를 이용하여 latent disentangle representation을 학습한다. 이 방법은 speaker identity와 prosody를 성공적으로 factorize한다.
  2. Prosody-prediction method : 먼저 prosody attributes를 추출하고, Predictor로 예측한다. 이때 linguistic feature를 이용한다.

* 기존 방식의 문제점

  1. pitch extractor의 문제점 : voiced, unvoiced decision, F0 values 등 (이 문제는 [RADTTS++](https://arxiv.org/pdf/2203.01786.pdf)에서 해결책을 제시함)

     - 이러한 오류는 pitch prediction(부분) 뿐만 아니라 TTS model(전체) 최적화에도 영향을 끼친다.

  2. Prosody-prediction method의 경우, Prosody attributed를 각각 추출하고, 각각 모델링하기 때문에 attritbutes간의 유기적인 관계를 modeling하지 못한다. 

     => word-level prosody encoder를 제안. pitch extractor의 error를 피하고, dependency를 반영하는 prosody 추출 방법

  3. prosody는 굉장히 다양하다. 단어마다도 달라지고 사람마다도 달라진다. 그런데 그에 비해 데이터는 너무 부족하다.

     => LPV predictor와 LPV predictor를 pretrain->finetune하는 방법을 제안



## Proposed Method

1. Overall architecture

   <img src="../image/prosoArchi.png" width=70%>

#### Prosody encoder

* 목적 : speech로부터 prosody를 분리

* 방법 : word-level vector quantization bottleneck을 이용한다.

  The first stack ) Mel-spectrogram을 word boundary에 따라 word-level hidden state로 압축한다. 

  * 구조: Conv - Relu - LayerNorm 

  The second stack) word-level hidden state를 post-process

  그리고, EMA(Exponential Moving Averages)-based Vector Quantization layer 

  => word-level LPV sequence (disentangled prosody from speech, speaker&content-independent prosody information)

* 여기서 input으로 Low-frequency band(first 20bins)만 사용한다.

  * 이는 Disentangle을 쉽게 하기 위해서다.
  * full band에 비해 Low-frequency band에 almost complete prosody를 담고 있고, content, timbre information이 적다. 
    * content -> linguistic feature, timbre -> speaker embedding으로 들어옴
  
* 훈련 시 생길 수 있는 문제점과 그 해결책

  * Index collapse: some embedding vectors are close to a lot of encoder outputs and the model uses only a limited number of vectors from e. 
  
  > Index collapse is **a problem with discrete latent variables that occurs when q(z|x) is supported only on a single small subset of the discrete latent space across all x**. In this case, the discrete VAE only learns to use a small portion of the latents to compute p(x|z), and most of the latent space is meaningless.
  
  * severely limits the expression ability of our prosody encoder
  * 해결책: warm-up strategy and k-means cluster-based centroid initialization
    1. remove the vector quantization layer in the first 20k steps => the prosody encoder extracts the prosody information freely without any bottleneck
    2. After the first 20k steps, we initialize the codebook of the vector quantization layer with k-means cluster centers
    3. after initialization, we add the vector quantization layer as the prosody bottleneck of later training
  
* 궁금Point : vector quantization이 왜 bottleneck이지?

  * Vector quantization은 continous 값이 codebook에 의해 discrete한 값으로 바뀜
  * 결과적으로 codebook에 있는 codeword들로 값들이 다 변환됨 => bottleneck
  

#### LPV predictor

* 목적 : LPV sequence를 모델링함으로 prosody를 모델링한다. 즉, text input에서부터 word-levle LPV를 predict한다.
* 구조: self-attention based autoregressive architecture 
  * train: teacher forcing mode
  * inference: predicts LPV auto regressively
  * text input -> content encoder -> word-level context feature

#### Pre train - finetuning 

* 목적 : 데이터 부족 문제를 보완하기 위해서 LPV 내의 모듈을 

* 데이터 부족으로 인해 생기는 문제들

  * context encoder에서 context understanding 성능이 떨어짐
  * prosody와 text간의 연결성을 파악하는게 어려움
  * prosody distribution estimation이 부정확함

* pretrain

  * text: context encoder in LPV predictor를 a BERT-list mask prediction으로 훈

  * 전체 : Low-quality audio를 이용해서 noisy audio로부터 LPV sequence를 encode함.

  * pretrain 과정에서 사용되는 Text와 audio는 쌍을 이루지는 않는다. "Unpaired"

    > * 그럼 의문 : unpaired인데 어떻게 connectivity 문제를 해결하는 것인가? 
    > * pretrain으로 prosody modeling, text modeling을 각각 해결한 후, finetune으로 connection을 학습하겠다는 것인가?
    > * Fig1을 보면, Pretrain 과정에 noisy audio가 prosody encoder로 들어가고, 이때 linguistic feature와 speaker embedding이 prosody encoder에 같이 들어간다. 이런 것으로 보아, noisy audio에 상응하는 text가 사용된 것 같긴 하다. 하지만 그 text와 content encoder 훈련에 사용한 text가 같지는 않다는 것이 unpaired의 의미로 추론할 수 있는 것 같다.

* Fine-tune 

  * 전체 훈련 frame work에서  ground truth mel-spectrogram에서 prosody encoder가 뽑은 LPV로부터 학습 





## Experiments & Results

* Dataset : 중국어 데이터셋

  * 62,586개 (약 30시간) 

  * Train : 61,000 / valid : 586 / test : 1,000

  * Open source Mandarin G2P
* Sample rate : 22050
* Window size : 1024, hop size : 256
* Vocoder : Hifi GAN
* Subjective evaluation : MOS
  * Speech의 내용은 무시하고 audio quality와 prosody만을 평가하기 위해 중국어 음성을 English native에게 들려줌
* Objective evaluation
  * Average pitch dynamic time warping (DTW) distance
    * Ground Truth와 합성음의 Pitch contour의 차이
  * Duration KL divergence 
    * Ground Truth와 합성음의 Duration 분포 차이
  

### Result

<img src ="../image/prosoResult.png" width=70%>



* FastSpeech2 joint 
  * pretrain에서 사용한 low-quality dataset (ASR dataset) 을 FastSpeech2 훈련에 high quality dataset(TTS dataset)과 함께 사용
  * 결과적으로 Low-quality data를 tts 훈련에 직접 사용하면 더 안 좋아짐. Module pretrain에 적합함. 
    * ~~당연한 거 아닌가?~~

* Ablation study 결과는 생략

