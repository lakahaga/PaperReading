## Style Tokens

### *Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis*

[github_link](https://github.com/KinglittleQ/GST-Tacotron)

* unsupervised -> The embeddings are trained with no explicit labels, yet learn to model a large range of acoustic expressiveness.
* The soft interpretable "labels" they generate can be used to control synthesis in novel ways, such as varing speed and speaking style **-independently of the text context.**
* **Style transfer**
  * replicate the speaking style of a single audio clip across an entire long-form text corpus.
* **Prosody**
  * the confluence of a number of phenomena in speech, such as paralinguistic information, intonation, stress and style.

* the goal of this study
  * to provide models the capability to choose a speaking style appropriate for the given context.
* Style modeling challenges.
  * no objective measure of *correct* prosodic style.
  * acquiring annotations for large datasets can be costly and similarly problematic
    * since human raters often disagree.
  * the high dynamic range in expressive vocies is difficult to model.
    * only learn an averaged prosodic distirbution over their input data
  * they often lack the ability to control the expression with which speech is synthesized.



### GST

* trained without any prosodic labels
  * internal architecture itself produces soft interpretable "labels" that can be used to perform various style control and transfer tasks
  * leading to significant improvements for expressive long-form synthesis.
* can be directly applied to noisy, unlabled found data, providing a path towards highly scalable but robust speech synthesis.



### Model Architecture

* 기본 tts pipeline
  * grapheme or phoneme input -> mel spectogram -> waveform

* **For tacotron, the choice of  vocoder does not affect prosody, which is modeled by the seq2seq model**
  * 인터레스팅...
  * 스타일 임베딩 없이 보코더 훈련시킨 거 결과 확인해보기
  * 꽤나 재현성이 좋다. 보코더는 음질만 잡아주면 되는 듯



<img src="/Users/choiyelin/Library/Application Support/typora-user-images/스크린샷 2021-12-27 오후 3.10.57.png" alt="스크린샷 2021-12-27 오후 3.10.57" style="zoom:50%;" />



* training

  * reference encoder -> style token layer

#### `reference encoder` 

 * compresses the prosody of a varaible length audio signal into a fixed-length vector
 * input : reference signal == ground-truth audio 
 * output : reference embedding (a fixed-length vector) 
 * detailed model architecture
     * made up of a convolutional stack, followed by an RNN
     * input : a log-mel spectogram
     * output :  3 dimension tensor 
     * the output tensor then is fed to a single layer 128-unit unidirectional GRU.
     * The last GRU state serves as the reference embedding
  * the reference embedding is input of the style token layer

~~~python
class Encoder(nn.Module):
    '''
    input:
        inputs: [N, T_x, E]
    output:
        outputs: [N, T_x, E]
        hidden: [2, N, E//2]
    '''

    def __init__(self):
        super().__init__()
        self.prenet = PreNet(in_features=hp.E)  # [N, T, E//2]

        self.conv1d_bank = Conv1dBank(K=hp.K, in_channels=hp.E // 2, out_channels=hp.E // 2)  # [N, T, E//2 * K]

        self.conv1d_1 = Conv1d(in_channels=hp.K * hp.E // 2, out_channels=hp.E // 2, kernel_size=3)  # [N, T, E//2]
        self.conv1d_2 = Conv1d(in_channels=hp.E // 2, out_channels=hp.E // 2, kernel_size=3)  # [N, T, E//2]
        self.bn1 = BatchNorm1d(num_features=hp.E // 2)
        self.bn2 = BatchNorm1d(num_features=hp.E // 2)

        self.highways = nn.ModuleList()
        for i in range(hp.num_highways):
            self.highways.append(Highway(in_features=hp.E // 2, out_features=hp.E // 2))

        self.gru = nn.GRU(input_size=hp.E // 2, hidden_size=hp.E // 2, num_layers=2, bidirectional=True, batch_first=True)

    def forward(self, inputs, prev_hidden=None):
        # prenet
        inputs = self.prenet(inputs)  # [N, T, E//2]

        # CBHG
        # conv1d bank
        outputs = self.conv1d_bank(inputs)  # [N, T, E//2 * K]
        outputs = max_pool1d(outputs, kernel_size=2)  # [N, T, E//2 * K]

        # conv1d projections
        outputs = self.conv1d_1(outputs)  # [N, T, E//2]
        outputs = self.bn1(outputs)
        outputs = nn.functional.relu(outputs)  # [N, T, E//2]
        outputs = self.conv1d_2(outputs)  # [N, T, E//2]
        outputs = self.bn2(outputs)

        outputs = outputs + inputs  # residual connect

        # highway
        for layer in self.highways:
            outputs = layer(outputs)
            # outputs = nn.functional.relu(outputs)  # [N, T, E//2]

        # outputs = torch.transpose(outputs, 0, 1)  # [T, N, E//2]

        self.gru.flatten_parameters()
        outputs, hidden = self.gru(outputs, prev_hidden)  # outputs [N, T, E]

        return outputs, hidden
~~~

* GRU
  * Gated Recurrent unit
  * LSTM의 long term dependency 문제 해결을 그대로 가져가면서 Hidden state를 update하는 계산을 줄였다. 

#### `style token layer`

* reference embedding is used as query vector to an attention module
* attention module learns a similarity measure between the reference embedding and each token in a bank of randomly initialized embeddings. (rather than learning an alignment)
 * a bank of randomly initialized embeddings == global style tokens (GSTs)
 * they are shared across all training sequences.
* Output = a set of combination weights that represent the contribution of each style token to the encoded reference embedding.
* *style embedding* = weighted sum of the GSTs
 * is passed to the text encoder for conditioning at every timestep.
   * -> 무슨 말 이해 엑스

* Style token layer is jointly trained with the rest of the model, driven only by the reconstruction loss from the Tacotron decoder. GSTs thus do not require any explicit style of prosody labels.

 * -> 무슨 말 이해 엑스

* Inference
  * inference 하는 2가지 방법이 있다. 
    1. "Conditioned on Token B"
       - directly condition the text encoder on certain tokens
       - This allows for style control and manipulation *without a reference signal*
    2. "Condtitioned on audio signal"
       - feed a different audio signal (whose transcript does not need to match the text to be synthesized) 



