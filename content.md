## Review



### What Is Bayesian Inference?

1. **Observe** the phenomenon, gather $\mathbf{X} = \left\\{\mathbf{x}\_i\right\\}\_{i=1}^N$, $\mathbf{x}\_i \in \mathbb{R}^D$.
2. **Build** a model, $p(\mathbf{X}, \mathbf{z})$ with latent variables $z\_1, \ldots, z\_d$.
3. **Infer** the posterior, $p(\mathbf{z}| \mathbf{X}) = p(\mathbf{X}, \mathbf{z}) / \int p(\mathbf{X}, \mathbf{z}) \, \mathrm{d}\mathbf{z}$, in order to **reason** about the phenomenon.
4. **Criticize** the model, revise it (->2), or collect additional data (->1).
5. **Apply** the model, i.e. calculate integrals over $p(\mathbf{z}| \mathbf{X})$: expectations of $f(\mathbf{z})$, posterior predictive $p(\mathbf{x}^\*|\mathbf{X})$, etc.



### Why Is Bayesian Inference Hard?

Most posteriors $p(\mathbf{z}| \mathbf{X})$ are not analytically tractable.

A possible solution is numerical estimation via MCMC, e.g. via:
* Metropolis Hastings Sampling,
* Gibbs Sampling,
* Hamiltonian Monte Carlo Sampling,
* No-U-Turn Hamiltonian Monte Carlo Sampling (NUTS).

NUTS is close to exact, but also slow and sequential.



### What Is Variational Inference (VI)?

VI is a class of algorithms which cast posterior inference as optimization:
<ol start="3">
	<li><span style="text-decoration: line-through">**Infer**</span> **Approximate** the posterior, $p(\mathbf{z}| \mathbf{X})$:
		<ol style="list-style-type: lower-alpha;">
		  <li>
		  	**Build** a variational model, $q(\mathbf{z}; \mathbf{\lambda})$, over $\mathbf{z}$ with parameters $\mathbf{\lambda}$.
		  </li>
		  <li>
		  	**Match** $q(\mathbf{z}; \mathbf{\lambda})$ to $p(\mathbf{z}| \mathbf{X})$ by optimizing over $\mathbf{\lambda}$,
	      $$
	      	\begin{aligned}
	      		\mathbf{\lambda}^\* & = \, \mathrm{argmin}\_{\mathbf{\lambda}} \; \mathrm{divergence}\left(p(\mathbf{z}| \mathbf{X}), q(\mathbf{z}; \mathbf{\lambda})\right).
	      	\end{aligned}
			  $$
		  </li>
		  <li>
		  	**Use** $q(\mathbf{z}; {\mathbf{\lambda}^\*})$ instead of $p(\mathbf{z}| \mathbf{X})$.
		  </li>
		  <li>
		  	**Criticize** the variational model, revise it (->a).
		  </li>
		</ol>
	</li>
</ol>

Effectively, VI is an additional layer of approximation that facilitates convenient model iteration.



### Matching And Optimizing

The **Kullback-Leibler divergence** from $q$ to $p$ is a good measure for closeness between $p$ and $q$,
$$
	\mathrm{KL}\left(q(\mathbf{z}; \mathbf{\lambda})\\| p(\mathbf{z}\|\mathbf{X})\right) \triangleq \, \mathbb{E}_{q(\mathbf{z}; \mathbf{\lambda})} \left[\log \frac{q(\mathbf{z}; \mathbf{\lambda})}{p(\mathbf{z}\|\mathbf{X})}\right].
$$

**Minimization** of this with respect to $\mathbf{\lambda}$ is intractable, though, because it directly depends on  $p(\mathbf{z}\|\mathbf{X})$.

**Maximize** the Evidence Lower BOund (ELBO) instead,
$$
	\begin{aligned}
		\mathcal{L}(\mathbf{\lambda}) & \triangleq \log p(\mathbf{X}) - \mathrm{KL}\left(q(\mathbf{z}; \mathbf{\lambda})\\| p(\mathbf{z}\|\mathbf{X})\right) \\\\[.5em]
		& = \,\\! \mathbb{E}\_{q(\mathbf{z}; \mathbf{\lambda})} \left[\log p(\mathbf{X}, \mathbf{z})\right] - \,\\! \mathbb{E}\_{q(\mathbf{z}; \mathbf{\lambda})} \left[\log q(\mathbf{z}; \mathbf{\lambda})\right].
	\end{aligned}
$$



### Conventional Variational Modelling

Two conflicting demands:
<ol style="list-style-type:lower-roman;">
  <li>
  	Make $q$ **simpler** than $p$, e.g. choose a factorized multivariate (mean-field) normal distribution,
	  $$
	    q(\mathbf{z}; \mathbf{\lambda}) = \prod\_{i=1}^d \mathcal{N}\left(z\_i; \mu\_{i}, \sigma\_{i}^2\right).
	  $$
	</li>
	<li>
	  Make $q$ more **expressive** so that it can give good results, e.g. choose a full-rank multivariate normal distribution,
  	$$
    	q(\mathbf{z}; \mathbf{\lambda}) = \mathcal{N}\left(\mathbf{z}; \mathbf{\mu}, \mathbf{\Sigma}\right).
  	$$
  </li>
</ol>



### Hierarchical Variational Modelling

<ol start="3" style="list-style-type:lower-roman;">
	<li>
		Use a mean-field distribution, $\prod\_i q(z\_i|\mathbf{\lambda}\_i)$, but softly constrain it by putting a prior $q(\mathbf{\lambda}; \mathbf{\theta})$ on it,
		$$
			q\_{\,\mathrm{HVM}}^{\;}(\mathbf{z}; \mathbf{\theta}) = \int \left[\prod\_{i=1}^d q(z\_i|\mathbf{\lambda}\_i)\right] q(\mathbf{\lambda}; \mathbf{\theta}) \, \mathrm{d}\mathbf{\lambda} \, .
		$$
	</li>
</ol>

Hierarchy captures dependencies between latent variables, $\mathbf{z}$.

More computationally tractable than a variational model with full dependence structure.

Expressiveness is determined by the complexity of $q(\mathbf{\lambda}; \mathbf{\theta})$.



### Variational Gaussian Process

Let the mean-field parameters be ${\lambda\_{i}} = f\_i(\mathbf{\xi}) \in \mathbb{R}$, $i = 1, \ldots, d$, where
* the latent input $\mathbf{\xi} \in {\mathbb{R}^{c}}$ is normally distributed, $\mathbf{\xi} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$,

* the functions $f\_{i} : {\mathbb{R}^{c}} \to \mathbb{R}$ are distributed according to a **Gaussian process**,
$$
  f\_{i} \sim \left.\mathcal{GP}\left(\mathbf{0}, {\mathbf{K}_{\mathbf{\xi}\mathbf{\xi}}}\right) \,\\! \right| \mathcal{D},
$$
conditioned on a fake data set, $\mathcal{D}$.


decomposing the vector-valued function into scalar-valued functions

Gaussian process is conditioned on a hypothetical data set.

we have latent inputs into this Gaussian process

the inputs are into the GP draws

the output of the GP draws are the mean field parameters

the variational Gaussian process is just an ensemble of mean-field distributions, the weights of the individual distributions given by a Bayesian nonparametric prior

evaluation of these GP draws at the same input $\mathbf{\xi}$ induces correlation between the outputs.

the parameters of this VGP are the data itself, because rather than observing data in variable space we are going to imagine a bunch of hypothetical data and we want to learn where to situate these data points, so that they anchor the random nonlinear mappings of the GP draws at certain input-output pairs.

we can also think of this in terms of a generative process:
1. Draw the latent inputs $\xi$ from a standard normal.
2. Draw the non-linear mapping conditioned on that fake data set.
3. Draw mean-field samples, i.e. approximate posterior samples $\mathbf{z} \in \mathrm{supp}(p)$, conditioned on the output of this GP draw. 


### Gaussian Process

given the data $\mathcal{D}$, $p(f| \mathcal{D})$ forms a distribution over mappings $f$ which interpolate between input-output pairs in $\mathcal{D}$.
