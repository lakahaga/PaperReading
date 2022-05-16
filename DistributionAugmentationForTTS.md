# Distribution Augmentation For Low-Resource Expressive Text-To-Speech

* ICASSP 2022 accepted
* Alexa AI, Amazon, Cambridge
* [논문링크](https://arxiv.org/pdf/2202.06409.pdf)


## Overview

데이터의 다양성을 높일 수 있는 음성 합성 데이터 증강 방법을 제안한다. 이때 증강의 대상이 되는 것은 text다. (음성 자체를 증강하지는 않음)

기존의 text를 재조합하고, 이에 따라 audio도 fragment로 나눠서 재조합한다. 이때 증강된 데이터의 품질 문제로 훈련을 방해할 수 있는데, 이를 해결하기 위해 증강의 정도를 제약하고, 증강된 데이터를 따로 표시하는 방법을 제시하였다.

증강된 text가 문장 구조적으로, 문법적으로 올바른 text가 될 수 있도록 constituency parse based tree substitution을 제안한다. Subtree만을 대체하는 방법으로 증강의 정도를 제약하는 것이다. 특히 이 방법은 기존 문장에서 바뀌는 부분이 상대적으로 짧기 때문에 기존 음성의 prosody를 학습하는데 적합하다.

이러한 방법으로 TTS 모델을 훈련하고 평가한 결과를 통해 제안한 데이터 증강 방법이 overfitting을 방지하고, robustness를 향상한 것을 확인하였다. 

동일 모델을 증강하지 않은 데이터셋과, 증강한 데이터셋으로 훈련했을 때 test loss가 줄어든 것을 확인하였고, (Overfitting 방지) 

WER과 PER 모두 줄어든 것을 확인할 수 있었다.

## Abstract 

* Data Augmentation for Neural TTS
* 훈련 텍스트 데이터의 다양성을 높이고자 함
  * 그 방법으로는 문장 구조적으로는 correctness를 유지하면서 text와 audio fragment를 바꾸는 방식을 택했다.
* 그 결과, 많은 dataset, 화자, TTS model의 합성음 품질이 향상 되었다. 특히 어텐션 기반 TTS 모델의 Robustness가 항샹되었다.

## Introduction

* TTS는 generative model이기 때문에 데이터 증강은 아직 연구된바가 별로 없다. 이전 연구의 대부분이 in-distribution data augmentation이기 때문에 증강 후에 다양성이 높아지지 않았다. 
* 본 논문의 inspiration이 된 선행 연구로는
  * [Distribution Augmentation for Generative Modeling](http://proceedings.mlr.press/v119/jun20a/jun20a.pdf): ICML 2020
  * [Substructure substitution: Structured data augmentation for NLP](https://arxiv.org/pdf/2101.00411.pdf): ACL/IJCNLP 2021

## Related work

* TTS에서는 데이터 증강 방법에 대한 연구가 거의 없다. 
* low-resource 환경에서의 기존 훈련 방법은 데이터를 증강하기 보다는 transfer learning방식을 사용하였다.
* TTS에서 기존 데이터 증강 방법
  * 기존 연구들은 거의 in-distribution 데이터 증강이다. 따라서 증강을 해도 다양성이 증가하지는 않는다.
  * 대표적인 방법으로는 voice conversion 방법이다.
    * 다른 화자의 목소리로 녹음된 음성을 훈련하고자 하는 화자의 목소리로 바꾸는 방법이다.
    * 이 방법은 데이터 양을 늘릴 수 있지만, 일단 다른 화자의 목소리로 녹음된 음성 데이터가 존재해야한다. 뿐만 아니라, voice conversion 모델을 훈련하기 위한 데이터도 필요하고, 최종 합성음의 품질이 voice conversion 성능의 영향을 많이 받는다.
* 그래서 본 논문에서 제안하는 방식은 추가로 데이터나 훈련 과정이 필요하지 않고, 기존 모델에 바로 적용할 수 있는 방식을 제안한다.
* 특히, 데이터 부족 문제가 자주 발생하는 감정 음성 데이터에 대해서 집중적으로 다룬다.

## Proposed Method

### 1. Data augmentation through word permutations

* 문장을 분해하고, 재조합해서 새 문장을 만든다.

* 여기서 발생할 수 있는 문제를 지적하고 이에 대한 해결책을 제시한다. 

* 발생할 수 있는 문제
  1. 증강된 sample이 unsound 할 수 있다. 문장 구조가 안 맞거나, 의미적으로 틀린 문장을 만들어 낼 수 있다.
     * 이를 수학적으로 나타내면 text conditioninig의 marginal distribution이 증강에 의해 변한다는 것이다.
  2. 오디오 fragment를 조합하는 과정에서 joint되는 부분의 소리가 unnatural할 수 있다. 
     - 이를 out-of-distribution local structure를 가질 수 있는 문제라고 칭한다.
  3. 새로 조합된 audio의 전체적인 prosody가 기존 데이터의 것과 달라질 수 있다.
     - 이를 Out-of-distribution global structure를 가질 수 있는 문제라고 칭한다.

* 해결 방법
  * 1번 문제를 해결하는 방법
  
    * 증강된 데이터셋이 기존 data distribution을 따르는 data sample을 충분히 포함하도록 한다.
      * in-distribution sample을 fitting하는데 문제가 없는 수준으로 
  
  * 1,3번 문제를 해결하는 방법
  
    * 증강된 데이터의 분포가 기존 distribution에서 벗어나는 정도를 제약으로 추가한다.
      * 어느 정도로 벗어나는 것을 허용해야 할까? : Syntax, grammer 규칙을 따르는 정도로 
  
  * 2,3번 문제를 해결하는 방법
  
    <img src="../image/distAugTTSFig3.png" alt="distAugTTSFig3" width="25%" height ="25%" align="left"/>
  
    
  
    *  augmentation tag를 추가적인 훈련 데이터를 제공한다. 이 tag는 증강 type을 identify한다. 
    * 그림에서와 같이 증강된 부분과 증강되지 않은 부분이 만나는 audio joint를 표시하기 위해서 augmentation tag가 1인 것을 확인할 수 있다. 
      * 논문에 나오진 않았지만, 이를 통해 audio joint가 이상할 경우 이를 학습에 덜 반영한다던가 하는 방법을 쓰는 게 아닐까?

### 2. Constituency parse based tree substitutions

> Constituency parse : 문장을 문장성분 단위로 나누는 과정 ([참고링크](https://web.stanford.edu/~jurafsky/slp3/13.pdf))

* main idea는 증강된 text가 문장 구조적으로, 문법적으로 올바른 text가 되도록하는 augmenation에 대한 constraint 정의해야한다는 것이다.

* HOW?

  <img src ="../image/constsubstiute.png" width=50% align=left>

  * non-terminal Node를 다시 쓰는 방식이다.
  * 구체적으로, 같은 코퍼스에 있는 다른 문장에 있는 subtree로 대체하는 것이다. 
  * 두 문장을 constituency parsing을 한 다음, *같은 문장 성분*의 subtree를 서로 바꾼다. 그리고, 기존 alignment 정보를 활용하여 audio fragment를 재조합한다.

* 이 방법의 장점은

  * 실제 문법을 예측하지 않아도 된다.
  * subtree substitution을 한번 하기 때문에, audio joint(서로 다른 original sample에서 가져온 audio fragment가 만나는 부분)이 sparse하고,
  * subtree로 바뀐 분은 대체로 짧고, 문장의 대부분은 원래 문장과 같기 때문에 원래의 prosody를 학습하기에 충분하다.

## Experiments

* 평가를 위해 학습한 모델 2가지
  * attention-based model : Tacotron2
  * externally provided durations를 사용하는 모델 : Non-attentive tacotron
    * non-attentive tacotron을 좀 변형해서 사용
      * TTS model에서 attention mechanism을 upsampling from phoneme to frame level로 대체하는데, 이때 Gaussian upsampling을 upsampling->bi-LSTM으로 변형하였다.
      * external duration model은 a stack of 1D convolutions + dense layer로 변형하였다. 이때 ground truth로는 oracle duration을 사용하고, L2 loss로 학습했다.
  * 본 논문에서 제안한 데이터 증강 방법을 사용하기 위해 추가한 것은
    * phoneme level conditioning을 추가하였다. (원래 문장 부분에서 새로 들어온 부분으로 넘어가는 첫 phoneme을 1로 표시, 나머지를 0으로 표시)
    * phoneme에 binary tag가 concate되어서 encoder에 들어감

* 사용한 데이터셋
  *  D1 : 여성 화자 영어 감정 음성 데이터셋
  * D2 : 여성 화자 한명, 남성 화자 한명 ([HiFi TTS dataset](https://arxiv.org/pdf/2104.01497.pdf))
  * 여기서 데이터셋 양을 조절하기 위해서 D1을 10h, 5h, 2h 세개로 Random하게 나눈다. 
  * D2는 화자의 성별에 따라 D2f, D2m으로 나눈다. D2f,m 모두 10h 분량의 데이터를 사용한다.

### Result

* baseline : 증강 없이 Orignal data만 훈련 / proposed model : 증강된 데이터를 포함해서 훈련

1. Attentive model

   <img src="../image/attenWER.png" width=50% align="left">

   * 10h, 5h 모두 baseline에 비해 WER,PER가 내려감. 
   * Robustness가 올라간 것을 확인 할 수 있음.
   * 음성인식 모델을 뭘 썼는지는 확인할 수 없음. 10h에 비해 5h가 더 낮은데, 이건 stochasticity of training the models, selecting the data, and running the robustness analyses라고 함.

   <img src="../image/attenTestLoss.png" width=50% align="left">

   * baseline에 비해 test loss가 적음 ->제안한 데이터 증강 방식이 overfitting을 방지하는 것을 알 수 있음
     * ~~데이터가 많아 졌으니까 당연한거 아냐? 라고 할 수 있지만, 기존 데이터의 성격을 유지하면서도 다양성을 높였기 때문에 test loss도 줄였다는 건가?~~
   
1. Non-attentive model

   <img src="../image/nonResult.png" width=50% align="left">

   <img src="../image/nonAblation.png" width=50% align="left">
   
   

