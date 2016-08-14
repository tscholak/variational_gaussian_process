## Review



### What Is Bayesian Inference?

Given:

* Observed data set $x$
* Joint probability model $p(\mathbf{x}, \mathbf{z})$ with latent variables $z\_1, \ldots, z\_d$

Objective

I like $f$ math $a = b$:
$$
\\begin{aligned}
	a &= b \\\\
	&= c
\\end{aligned}
$$



\begin{aligned}
\mathcal{L} & = \log p(x) - \mathrm{KL}\left(q(z;\lambda)\middle\| p\left(z\middle| x\right)\right) \\
& = \mathbb{E}_{q(z;\lambda)}\left[\log p(x) - \log \frac{q(z;\lambda)}{p\left(z\middle| x\right)}\right] \\
& = \mathbb{E}_{q(z;\lambda)} \log \frac{p\left(x, z\right)}{q(z;\lambda)} \\
& = \mathbb{E}_{q(z;\lambda)}\left[\log p\left(x\middle| z\right) - \log \frac{q(z;\lambda)}{p\left(z\right)}\right] \\
& = \mathbb{E}_{q(z;\lambda)}\left[\log p\left(x\middle| z\right)\right] - \mathrm{KL}\left(q(z;\lambda)\middle\| p\left(z\right)\right)
\end{aligned}
