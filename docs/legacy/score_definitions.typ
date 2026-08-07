#set page(paper: "us-letter", margin: 1in)
#set text(size: 11pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 18pt, weight: "bold")[The Three Implemented Detection Scores]
]

At round $t$, let $q_t(o) = L_1^(t)(o)$ be the credulous listener's current distribution over observations and let $k_t(u | o) = S_1^(t)(u | o, "inf")$ be its informative-speaker model.

The code constructs the prior predictive distribution and observation posterior as

$
  p_t(u) & = sum_o k_t(u | o) q_t(o), \\
           q_t(o | u) & = frac(k_t(u | o) q_t(o), p_t(u)).
$

For the posterior-predictive score it additionally constructs $r_t(v | u) = sum_o k_t(v | o) q_t(o | u)$. Write $H(a) = -sum_x a(x) log a(x)$. The implementations in `rsa/detection/scores.py` evaluate the following quantities for the heard utterance $u_t$.

= Prior-predictive surprisal (`surp2`)

$ "surp"_2^(t)(u_t) = -log p_t(u_t) - H(p_t). $

This is centered self-information under the listener's prediction before hearing the utterance. Its code-level variance estimate is the varentropy of that prediction:

$ sigma_("surp"_2,t)^2 = op("Var")_(U ~ p_t)[-log p_t(U)]. $

Under the informative null, $U ~ p_t$, so $E["surp"_2^(t)(U)] = 0$.

= Posterior-predictive surprisal (`surp1`)

$ "surp"_1^(t)(u_t) = -log r_t(u_t | u_t) - H(r_t(dot | u_t)). $

Here the same utterance is used twice: first to update the observation distribution from $q_t(o)$ to $q_t(o | u_t)$, then to score itself under the resulting posterior predictive distribution. The implemented variance is the corresponding conditional varentropy:

$ sigma_("surp"_1,t)^2 = op("Var")_(V ~ r_t(dot | u_t))[-log r_t(V | u_t)]. $

Unlike `surp2`, this is a data-dependent conditional reference distribution; the code does not assert an unconditional zero-mean identity for this score.

= Observation-level suspicion (`sus`)

For each possible observation, define the conditional excess surprise $b_t(o, u) = -log k_t(u | o) - H(k_t(dot | o))$. The suspicion score averages this quantity using the observation posterior:

$ "sus"^(t)(u_t) = sum_o q_t(o | u_t) b_t(o, u_t). $

Thus `sus` takes the log before marginalizing over observations, whereas `surp2` marginalizes first. Its original implemented variance is the posterior-weighted conditional varentropy:

$ sigma_("sus",t)^2 = sum_o q_t(o | u_t) op("Var")_(U ~ k_t(dot | o))[-log k_t(U | o)]. $

For every fixed $o$, $E_(U ~ k_t(dot | o))[b_t(o, U)] = 0$; therefore `sus` has mean zero under the informative null. The later full-sweep pipeline uses `sus_1`, which applies a law-of-total-variance correction to this variance; see `docs/detection_math.tex` for that extension.

= Sequential use

For any score $s_t$ with variance estimate $sigma_t^2$, the comparison experiments accumulate

$
  overline(s)_t & = frac(1, t) sum_(i=1)^t s_i, \\
                  overline(sigma)_t^2 & = frac(1, t) sum_(i=1)^t sigma_i^2,
$

and flag when $overline(s)_t > c frac(overline(sigma)_t, sqrt(t))$.
