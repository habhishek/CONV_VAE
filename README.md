## The Model

Implemented as tf.keras.Model subclass, the VAE model consists of 2 main parts - namely the encoder and decoder - both
smaller ConvNets. In the literature these are reffered to as inference/recongnition and generative models respectively.

It's important to note the output layers of each of these ConvNets:-


*   **Encoder Output Layer** - `tf.keras.layers.Dense` that outputs concatenated means and standard deviations
*   **Decoder Output Layer** - `tf.layers.Conv2DTranpose` that outputs the logits of the reconstructed image produced 
    by the decoder.

## The Loss Function
The goal of our generative model is to essentially maximize the log probability of the datapoint x or $\log p(x)$. Using the variational principal and inference techniques we define the VAE model loss function as that equal to maximizing the Evidence Lower Bound (ELBO) on the marginal log-likelihood.

In other words, the marginal log-likehood of the data defined as:-

<a href="https://www.codecogs.com/eqnedit.php?latex=\log&space;p(x)&space;=&space;\log\int{p(x,z)}dz" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\log&space;p(x)&space;=&space;\log\int{p(x,z)}dz" title="\log p(x) = \log\int{p(x,z)}dz" /></a>

is intractable. The encoder-decoder combination is essentially a method of applying the variational technique to find this intractable $\log p(x)$. During this application, since we cannot directly compute the posterior probability of the latent space $p(z|x)$, for similar reasons as stated above, we try to approximate this density function using another probability density function $q(z|x)$. This introduction of an aproximate posterior then converts the problem into an optimization problem. Instead of trying to maximize our log probability of the data, we try to improve our approximation of the posterior. We decrease the measure of farness (or divergence) of the aproximate posterior and the true posterior.

This application helps us to define an ELBO over the marginal log likelihood as defined below:-

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{equation}\label{eblo_1}&space;ELBO&space;=&space;\mathbb{E}_{z\sim{q(z|x)}}[\log{p(x|z)}]-&space;D_{KL}((q|z)||p(z))&space;\end{equation}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{equation}\label{eblo_1}&space;ELBO&space;=&space;\mathbb{E}_{z\sim{q(z|x)}}[\log{p(x|z)}]-&space;D_{KL}((q|z)||p(z))&space;\end{equation}" title="\begin{equation}\label{eblo_1} ELBO = \mathbb{E}_{z\sim{q(z|x)}}[\log{p(x|z)}]- D_{KL}((q|z)||p(z)) \end{equation}" /></a>

Can also be defined in terms of <a href="https://www.codecogs.com/eqnedit.php?latex=\mathbb{E}_{z\sim{q(z|x)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathbb{E}_{z\sim{q(z|x)}}" title="\mathbb{E}_{z\sim{q(z|x)}}" /></a> as below:-

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{equation}\label{elbo_2}&space;ELBO&space;=&space;\mathbb{E}_{z\sim{q(z|x)}}[\log{p(x|z)}-&space;\log{(q|z)}&space;&plus;&space;\log{p(z))}]&space;\end{equation}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{equation}\label{elbo_2}&space;ELBO&space;=&space;\mathbb{E}_{z\sim{q(z|x)}}[\log{p(x|z)}-&space;\log{(q|z)}&space;&plus;&space;\log{p(z))}]&space;\end{equation}" title="\begin{equation}\label{elbo_2} ELBO = \mathbb{E}_{z\sim{q(z|x)}}[\log{p(x|z)}- \log{(q|z)} + \log{p(z))}] \end{equation}" /></a>
Using the Monte Carlo Sampling technique, for simplicity we can define this for a single sample:-

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{equation}\label{elbo_3}&space;ELBO&space;=&space;\log{p(x|z)}-&space;\log{(q|z)}&space;&plus;&space;\log{p(z))}&space;\end{equation}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{equation}\label{elbo_3}&space;ELBO&space;=&space;\log{p(x|z)}-&space;\log{(q|z)}&space;&plus;&space;\log{p(z))}&space;\end{equation}" title="\begin{equation}\label{elbo_3} ELBO = \log{p(x|z)}- \log{(q|z)} + \log{p(z))} \end{equation}" /></a>
We try to maximum the ELBO or minimize the -ELBO.

The first part of this ELBO is called the Reconstruction term and the second term is called the Regularizer.

1. **Reconstruction Loss**:- $\log{p(x|z)}$ essentially measures the quality of reconstructed image from the decoder. Is implemented here using the `tf.nn.sigmoid_cross_entropy_with_logits`. Reconstruction loss also read as negative log likelihood of the data or probability of x, given the latent sample z. We take the sigmoid cross entropy loss of each pixel values of the input x and reconstructed image of the logits from the decoder. We then take sum of all these values(using `tf.math.reduce_sum`) and the sum thereby represents the log likelihood of the data.

2. **Regularizer**:- KL Divergence term is also known as the Regularizer. Since we cannot directly measure the quality of approximation, i.e., $q(z|x)$, we try to instead keep it closer to a known, simple (and tractable) prior distribution $p(x)$- in this case a Gaussian. This KL term helps in acheiving this objective and is a part of the ELBO. The generic implementation for a single Monte Carlo sampling has the terms - $\log{p(z))}-\log{(q|z)}$. The Analytical KL divergence can also be computed for a special case where we assume that the true $p(z)$ has a diagonal covariance matrix.

We define a function to calculate the neccessary log normal pdfs in the loss function