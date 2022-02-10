
### Generative Model

<br>

- [ ] **Deep Generative Modelling: A Comparative Review of VAEs, GANs, Normalizing Flows, Energy-Based and Autoregressive Models [TPAMI, 2021]**[[Paper]](pdfs/2103.04922.pdf)

<br>

- [ ] **Deep Generative Models: Survey [ISCV, 2018]** [[Paper]](pdfs/Deep_generative_models_Survey.pdf)

<br>

#### GAN

<br>

- [ ] **A Review on Generative Adversarial Networks: Algorithms, Theory, and Applications [TKDE, 2021]** [[Paper]](pdfs/A_Review_on_Generative_Adversarial_Networks_Algorithms_Theory_and_Applications.pdf)


<br>

- [ ] **GAN-Control: Explicitly Controllable GANs [arxiv, 2021]** Amazon One [[Paper]](pdfs/gan-control.pdf)

<br>

- [ ] **On GANs and GMMs [NIPS, 2018]**
  - Training a extreme-scale dimensional GMM

<br>

#### Normalizing Flow

<br>

- [ ] **Normalizing Flows for Probabilistic Modeling and Inference [JMLR, 2021]** [[Paper]](pdfs/19-1028.pdf)

<br>

- [ ] **Normalizing Flows: An Introduction and Review of Current Methods [IPAMI, 2020]** [[Paper]](pdfs/1908.09257.pdf)

<br>

#### Energy-Based Models

<br>

- [ ] **How to Train Your Energy-Based Models [arxiv, 2020]** [[Paper]](pdfs/2101.03288.pdf)

<br>

#### VAE

<br>

- [ ] **A survey on Variational Autoencoders from a GreenAI perspective [arxiv, 2021]** [[Paper]](pdfs/2103.01071.pdf)

<br>

- [ ] **Dynamical Variational Autoencoders: A Comprehensive Review [arxiv, 2020]** [[Paper]](pdfs/2008.12595.pdf)
  - Temporal VAE: handles sequence dependency

<br>

- [ ] **An Introduction to Variantional Autoencoders [Foundations and Trends in ML, 2019]** [[Paper]](pdfs/1906.02691.pdf)

<br>


- [ ] **Mixture Density Network [Classics, 1994]** Christopher M. Bishop [[Paper]](pdfs/NCRG_94_004.pdf)
  - Question:
    - Difference between EM on GMM?
      - MDN provides conditional pdf: $P(y|x)=\sum_if_\alpha(x) N(f_\mu(x),f_\sigma(x))$
    - softmax and exponential term theoretical interpretation
      - softmax for component prior: sum to one
      - sigma = exp(z): un-informative Bayesian prior, assuming z has uniform distribution; avoid variance goes to 0

- **Question: GMM vs KDE?**
  - is GMM a simplified KDE with fewer components?
  - No:
    - KDE: each data point represent a distribution
    - GMM: data points conform to a predefined mixture of components

<br>

- **Review: EM algorithm**
  - Problem:
    - Assumethere is an unknown statistical model with some parameters $\theta$, from which some observations $X=(x_1,...,x_n)$ are generated. Want to find the model by maximizing the likehood of the parameter given observations $P(\theta;X) = \prod_iP(\theta;x_i)$    - 
      - max this is equivalent to max the log form, which is by setting the derivative to 0:
      - $logP(\theta;X) = \sum_ilogP(\theta;x_i)$
        - **Now: No analytical solution. How to maximize?**
          - introduce hidden indicator variable $Z$ mapping $x_i$ to its membered component. Then can, for each member, get MLE of parameters
          - $Z$ unknown, and EM goes in and estimate E, then use E to update param, then estimate E again, then ...... until convergence
  - Expectation-Maximization Algorithm for GMM
    - GMM pdf: $p(x)=\sum_i^k\alpha_iN(x;\mu_i,\sigma_i)$
    - Expectation: approximate $Z$ (i.e. $\alpha$s) with observation and current parameters:
      - $P(\alpha_i|x_j)=\alpha_{norm}P(x_j|\alpha_i)P(\alpha_i)$
        - $P(x_j|\alpha_i)$ is $N(\mu_i,\sigma_i)$ in GMM optimization
        - under perfect indicator assumption $P(\alpha_i|x_j)$ should be 1, but we can only approximate
    - Maximization (MLE with $P(x_j|\alpha_i)$):
      - Define $N_i=\sum_jP(\alpha_i|x_j)$
        - intuitively for an indicator function Z (which we can only appximate), it's the number of observations in this component
      - $\mu_i=\frac{1}{N_i}\sum_jx_jP(\alpha_i|x_j)$
      - $\sigma_i^2=\frac{1}{N_i}\sum_j(x_j-\mu_i)^2P(\alpha_i|x_j)$
      - $\alpha_i=\frac{N_i}{\sum_jN_j}$
    - Goes back to Expectation and update our approximated $Z$ and then update parameters with new Z, iteratively.

<br>
