### Cho et. al. 2014, Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation

- **Encoder-decoder with GRUs for translation:**

  - The **encoder** processes a variable-length input sentence into a
    **fixed-length** vector representation.
  - The **decoder** generates a variable-length output (translation) from this encoded message.
  - The network is used within a older-fashioned SMT framework.

- **Key Results:**

- Created GRU!

- Invented Encoder-decoder with RNNs.

- **Details:**

- 1-layer GRU of size (N) 1000 for encoder and for decoder.

- encoded message has size 1000.

- 100 word embedding (D).

- Only a few M params.

- Too many tricks, confusing. Decoder uses 500 maxout units.

- Network initialized with gaussian distribution, except recurrent weights which had some tricks.

- Adadelta and SGD, trained for 3 days, batch size 64

### Sutskever et. al. 2014, Sequence to Sequence Learning with Neural Networks

- **Encoder-decoder with LSTMs for translation:**
  - Like Cho, produces fixed-length encoding.
- Simple network only uses LSTMs and embeddings, disregards older tricks.
- **Key Result:**
- Achieved SOTA using 5 ensemble with left-to-right beam search decoder.
- Small beam sizes worked well (1 or 2).
- **Details:**
- Dataset: 12M sentences, 160k input vocab, 80k output vocab (+ UNK EOS).
- 4-layer LSTM of size (N) 1000 for encoder and for decoder.
- 1000 word embedding (D).
- 8000 floats for encoded message.
- 384M params total, 64M in LSTMs.
- params in LSTM layer = 4x(NxD + NxN) = 8M
- params in embedding = vocab x D = 240M
- Computation is 2xLSTM params. Embedding computation is O(1).
- Network initialized with uniform distribution.
- SGD with momentum, 7.5 epochs, batch size 128.
- 8 GPUs. 4 GPUs processed a single layer each. (struggling to parallelize)
- **Notes:**
- They were surprised LSTM worked in long sentences despite small memory.
- One of their "key" innovations is inverting the source sentence (lol).
- Really, without ensemble or reversed sentence, it was not SOTA.
- Cites Kalchbrenner+, 2013 as original encoder-decoder.
- Cites Cho+, 2014 as also doing encoder-decoder LSTM.
- Cites Bahdanau+ for its new attention mechanism.

### Bahdanau+ 2014, Neural Machine Translation by Jointly Learning to Align and Translate

- **Introduces attention**, removing the**fixed-length bottleneck**.
- Uses**bidirectional GRUs**to encode input.
- The attention mechanism searches in encoded input and produces a context vector (the relevant information).
- decoder is a GRU that uses the context and the previous output as intput.
- Search ranking is done using a trained network (a).
- **Details:**
- Bidirectional GRU encoder:
- two separate GRU networks, result is concatenated.
- Both directions use the same embedding matrix.
- Alignment (ranking or attention) term is $e\_{ij} = v^T \\tanh(W s\_{i-1} + U h_j)$ .
- It is calculated for every input and output word combination (expensive)
- The U h term can be calculated before decoding and reused in every step.
- Comparison of key and query is done through awkward addition.
- Technically, v could learn to do cosine similarity.
- Model size:
- All GRUs used hidden size 1000, O(1M) params
- Embedding dimension 620, O(vocab x 620) params
- Alignment size 1000, O(1M) params
- Computational cost is dominated by attention O(1M _ input words _ output words)

* Adadelta, batch size 80
* Used single mid grade GPU.

### Vaswani et al. (2017) – Attention Is All You Need

- **Transformer: A Fully Attention-Based Model for Sequence Processing**
  - **Self-attention** allows each token to attend to all others in the sequence, replacing recurrence and convolution.
  - Uses **positional encoding** to retain order information since self-attention is inherently order-agnostic.
- **Key Result:**
  - Got rid of the RNNs and CNNs, increasing parallelism and thus faster training.
  - Outperformed RNN-based models on machine translation (**WMT’14 En→De, En→Fr**).
- **Details:**
  - **Dataset:** WMT’14 English-German & English-French.
  - **Architecture:**
    - **Encoder-Decoder model** with **stacked self-attention layers (6 each)**.
    - **Multi-Head Self-Attention** (h=8) projects input into multiple attention spaces.
    - **Feedforward layers (ReLU activations) per Transformer block**.
    - **Positional Encoding** (sinusoidal functions).
  - **Parameters:**
    - **d_model = 512**, with **64 per attention head**.
    - **FFN hidden size = 2048**.
    - **Total parameters ~65M for base model, ~213M for large model**.
  - **Computation:**
    - **Self-attention is O(n²) in sequence length**, making it expensive for long inputs.
    - **Much faster training** than RNNs due to full parallelization.
  - **Training Details:**
    - **Optimizer:** Adam with learning rate warm-up.
    - **Batch size:** 25k tokens.
    - Trained 8 P100 GPUs.
- **Extras:**
  - Used byte-pair encoding (grouped words that frequently go together as a word)
  - Used label smoothing.
  - Used Dropout.
  - Used beam search with size 4.
  - They average the last N checkpoint weights.
