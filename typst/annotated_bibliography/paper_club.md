# Summary of Paper Club items

## Notes For Lectures

Based on the lectures for CS 285 from Barkeley on youtube.

[link to youtube playlist](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps)

### Lecture 8 - Deep RL with Q-Functions

???

### Lecture 9 - Advanced Policy Gradients

* Demonstrated why policy gradients are reasonable - used policy iteration methods as starting point
* Showed policy gradient works when policy doesn't change too quickly. That's why we need the constraint
* Gradient descent vs "natural gradient descent" - it is better to constrain changes in policy NOT model parameters
   
### Lecture 10 - Optimal Control & Planning

* Model based learning
* If you know the dynamics, you don't even need reinforcement learning
* Discrete planning as a search problem (Monte Carlo tree search), but it is exponentially expensive, so you need a strategy for how to explore. Useful algorithm for when you use trial and error and don't know the derivatives.
* LQR - Linear Quadratic Regulator - for when you know the derivatives:  linear dynamics with a quadratic cost function. It was shown that it works for stochastic and nonlinear cases
* Example paper: Synthesis and stabilization of complex behaviors through online trajectory optimization

### Lecture 11 - Model-Based Learning

* How to learn model dynamics from data so that we can use model-based planning methods we learned in the last lecture.
* Distribution shift is a problem - we want to know the dynamics near the policy we plan to use. Otherwise you can "fall off a cliff".
* Model Uncertainty - Bayesian methods, Bootstrapping - Train several models and compare results.
* Latent space modeling - e.g. for images - the Observations may differ from actual States. 4 key parts: Observation model, dynamics model, rewards model, encoder.

### Lecture 12 - Model-Based Policy Learning

* Model free RL using a model for easy to-get data, and then they apply to LQR
* We pick some points out of "real" trajectories and then continue using the model that approximates the dynamics for some number of steps.
* You cannot do real model-based learning because of exploding/vanishing gradients because as time goes on the actions/states aren't as impactful. Similar problem as in RNNs, but you can't borrow their solutions because you can't control the dynamics.
* Two buffers: replay buffer and the augmented buffer with the generated data.
* Distillation: The strategy of combining multiple local policies (trained for specific initial conditions or sub-tasks) into a single, more general policy through supervised learning.
* From newer version of lecture: instead of value functions, you can produce a vector that when multiplied with the rewards gives you the value/reward (you can also represent it using different base functions that you would choose).

### Lecture 13 - Exploration (Part 1)

* Exploitation vs exploration: More difficult to explore more when rewards are sparse. Models tend to exploit suboptimal strategies when rewards are hard to get or too far in the future
* Three ways to encourage exploration: 
  1. Count-based: bonus for new states. You can use simple counts or other methods (e.g. classifiers) that help quantify the uniqueness of a state
  2. Thompson sampling (aka posterior sampling): Produce a pool of q-functions using bootstrapping, sample q and assume it is correct, then execute policy for a whole episode, then update your q pool. Example of a good application was the diver Atari game where you have to resurface for air at the end, so oscillating between policies won't likely let you discover this strategy.
  3. Information gain: Add a bonus for actions that allow you to gain information about the environment. For example, they use KL-divergence to rewards actions that lead to large changes in policy parameters. You can also quantify it by trying to predict something (e.g. the dynamics) and measuring the error (and reward discovery of states with high error).

### Lecture 14 - Exploration (Part 2)

* ???
  
## Notes for papers 

### Paper 1 - 1990 Le Cun -  Handwritten Digit Recognition with a Back-Propagation Network (LeNet)
[neurips link](https://proceedings.neurips.cc/paper/1989/file/53c3bce66e43be4f209556518c2fcb54-Paper.pdf)

* First use of backdrop on CNN. Classified handwritten digits.
	* Local features, weight sharing and backprop were not innovations of this paper.
	* Before this paper they used to hand code the first few layers. 
* ~10k images, size 28x28, active part is only 16x16, 
	* Used printed digits to enrich the dataset of handwritten digits.
* 4 CNN layers, 1 fully connected, all sigmoid
	* The network omits some connections.
* Scores are unfair since digits were cleaned up from marks.
* Interesting metric post training: Thresholds of rejection to reach a specific error rate.
* It took 3 days on a tiny dataset and network! only 30 epochs!
* Implemented on float32. chip had 12 mflops (~1e7) Nowadays we get 1 pflop (~1e15). 8 orders of magnitude.

### Paper 2 - 2012 Krizhevsky - ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)
[neurips link](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

* Achieves SOTA on image classification using CNNs.
* Weird normalization prioritizes large weights.
* There are two GPUs but they communicate only on layer 3. 
* Test predictions are using a mixture of results over multiple patches!
* Weird PCA preprocessing
* Interesting: he tried fine tuning (first train on entire dataset, then refine on subset)
* The results on ImageNet 2009 seem terrible!
* He’s already considering auto encoders for image retrieval.
* 90 epochs, 6M parameters, ~500M FLOP

### Paper 3 - 2015 Simolyan - Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)
https://arxiv.org/abs/1409.1556

* Achieves **state-of-the-art performance** on the ImageNet challenge.
* Demonstrates that **deeper networks improve accuracy** while keeping computational efficiency.
* Uses **small (3x3) convolutional filters** stacked deep in the network.
* Highlights the importance of **pretrained models** for transfer learning.
* Tests models of varying depth, 
* Networks and training:
	* 11 to 19 weight layers. Last 3 layers fully connected.
	* Batch size of 256, 72 epochs (370k iterations) (fewer than alexnet)
	* Momentum, dropout reg, l2 penalty, lr decrease when stops improving.
	* Pre-initialized weights. Multiple jittered crops.
	* 138M parameters, 20B FLOP 
* Interesting: Train filters on small image size, inference on larger image size.
	* The fully connected layers are converted to CNNs. The resulting predictions are sum pooled.
* They didn’t find useful to use local response normalization (weird normalization from AlexNet)
* Some of the pixels in the last conv layer have a receptive field that doesn't cover the entire image.

### Paper 4 - 2015 Ioffe - Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
https://arxiv.org/abs/1502.03167
Summarized by GPT4o, I didn't verify the numbers.

* Introduces **Batch Normalization (BN)** to address **internal covariate shift**, stabilizing deep network training.
* Normalizes activations **within each mini-batch**, ensuring consistent distributions across layers.
* BN applies a **learnable scale (γ) and shift (β)** after normalization to preserve representational power.
* Significantly **accelerates training**, allowing for **higher learning rates** and reducing sensitivity to initialization.
* Acts as a **regularizer**, reducing the need for dropout in some cases.
* Reduces **vanishing/exploding gradients**, making deep networks easier to optimize.
  
### Paper 5 - 2015 He - Deep Residual Learning for Image Recognition
https://arxiv.org/abs/1512.03385
Summarized by GPT4o, I didn't verify the numbers.

* Introduces **Residual Networks (ResNet)** to tackle the **vanishing gradient problem**, enabling effective training of very deep networks.
* Uses **residual learning** with **skip (shortcut) connections**, allowing identity mappings to bypass one or more layers, making optimization easier.
* **ResNet-152**: **60M parameters**, **11B FLOPs** per forward pass
* Some variants use **bottleneck layers** (1x1, 3x3, 1x1 convolutions) to reduce computation while maintaining depth.
* Incorporates **Batch Normalization (BN)** after each convolution for stable training.
* Trained on **ImageNet** with **SGD optimizer**, **learning rate = 0.1**, **batch size = 256**, **momentum = 0.9**, and **weight decay = 0.0001**.
* Achieves **3.57% top-5 error** on ImageNet, setting a **new state-of-the-art** at the time.
  
### Paper 6 - 1992 Watkins - Q-Learning
[springer link](https://link.springer.com/article/10.1007/BF00992698)

* Produces the Q function Q(x,a), which in this case is a table, and which provides the expected discounted rewards from applying action a on state x.
* The procedure runs a step in the simulation and updates Q at the current x and selected a according to the following rule
	* $Q[xn,an] = (1-k)Q[xn,an] + k(rn+g Vn[yn])$
	* Where k is the learning factor, g is the discount rate, yn is next state, and $Vn[y] = max_b {Q(y,b)}$ is the estimated reward for the best possible next action.
* Questions
	* Seems discrete only. Why do they describe Q as Q(x,a) instead of Q_xa?
	* Seems difficult to translate to deep learning. What would I do to get there?

### Paper 7 - 2017 Schulman - TRPO
https://arxiv.org/pdf/1502.05477

### Paper 8 - 2017 Schulman - PPO
https://arxiv.org/pdf/1707.06347

### Paper 9 - 1999 Sutton - Policy Gradient
[neurips link](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)

### Paper 10 - 2013 Mnih - Playing Atari with Deep Reinforcement Learning
https://arxiv.org/pdf/1312.5602

### Paper 11 - 2015 Hasselt - Double Q-Learning
https://arxiv.org/pdf/1509.06461

### Paper 12 - 2015 Watter - Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images 
https://arxiv.org/pdf/1506.07365

* Model-based learning for images (high dimensional input) that uses latent space representation (low dimensional).
* Main contributions: trains the encoder, decoder, and transition model simultaneously, which allows them to set a restriction for local linearity for the transition dynamics.
* The resulting encoder produces good representation spaces. They use KL divergence to make sure model stays in relevant space.

### Paper 13 - 2021 Janner - When to Trust Your Model: Model-Based Policy Optimization
https://arxiv.org/pdf/1906.08253

* Introduces technique where they use the dynamics-model in short runs, branching frequently from runs using "real life" data. They keep separate replay buffers for model-generated data and the real environment data.
* It improves the sample efficiency, and avoids overfitting to data in the "real data" replay buffer which may have few samples.
* They were comparing usefulness of off-policy real data versus on-policy model-generated data. They found that branches of 0 length were best in theory, but in practice, branches of length 1 were most helpful.

### Paper 14 - 2017 Tang - # Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning
https://arxiv.org/pdf/1611.04717

* Explores **count-based exploration** methods for deep reinforcement learning (RL).
* The counts is over hashes produced by an autoencoder. 
* The space of the hashes is smaller and thus generates more collisions.
* Worked well but didn't seem robust.

### Paper 15 - 2018 Eysenbach - Diversity is All You Need: Learning Skills without a Reward Function
https://arxiv.org/abs/1802.06070

* Exploration "without" rewards using information gain/entropy to train multiple policies that depend on a "skill" parameter
* Three terms: (1) maximizes spread across different skills, (2) ability to infer skill from current state, (3) a term that maximizes possible actions for a given state and skill
	* Point 2 requires you to train a separate discriminator classifier
* Result: a policy `π(a|s,z)` that produces actions for learned skills z
* They show different ways to use the learned policy in tasks with rewards: 
	* Choose the skill that gives the most reward and then fine-tune
	* Train a model to select the skill for the next n steps
	* Train a model that mimics a human expert
