#set page(paper: "us-letter", margin: 1in, numbering: "1")
#set text(size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 17pt, weight: "bold")[
    Argumentative Awareness
  ]
  #v(0.2em)
  #text(size: 12pt)[
    Sequential detection of persuasive speakers by a credulous RSA listener
  ]
  #v(0.4em)
  #text(size: 10pt, style: "italic")[Project reference: model, scores, and experiments]
]

#v(1em)

= Overview

This project starts from the Rational Speech Act (RSA) opinion-dynamics model of
Fang (2025), _A Computational Account of Epistemic Vigilance_, and pushes it in a
different direction.

Fang's model contrasts two listeners. A *credulous* listener ($omega = "coop"$)
assumes every speaker is informative and is therefore vulnerable to selective
truth-telling. A *vigilant* listener ($omega = "strat"$) carries a hypothesis
space over speaker goals $psi$ and can discount persuasive framing. Vigilance is
shown to be nearly free: it costs little against an honest speaker and protects a
lot against a persuasive one.

That result presumes the listener is _already_ vigilant. This project asks the
prior question:

#block(inset: (left: 1em), stroke: (left: 2pt + luma(200)))[
  Can a listener who begins *credulous* — with no $psi$ hypothesis space at all —
  notice from the utterance stream alone that its informative-speaker model is
  wrong, and do so with calibrated error control?
]

The reframing turns epistemic vigilance from a fixed disposition into a
*sequential hypothesis test*. The listener keeps its cheap credulous model,
accumulates a per-round statistic measuring how poorly that model predicts what
it hears, and raises an alarm when the accumulated evidence crosses a threshold.
Only then would it pay for the expensive vigilant machinery.

Two questions organize the work: which per-round *score* best separates an
informative speaker from a persuasive one, and does the sequential test built on
that score actually control its false-alarm rate. The short answers, developed in
@experiments, are `sus_1` with a law-of-total-variance–corrected variance, and
*no* — false alarms run at roughly 21–25% against a nominal 2.3%, for reasons
that turn out to lie in the shape of the stopping boundary rather than in the
score or its variance.

= World and communication setup <setup>

The world model is taken unchanged from Fang (2025), so notation is shared
throughout.

== Latent state and observations

A population of patients shares a latent per-session improvement rate
$theta$, drawn from a discrete grid:

$ theta in Theta = {0.1, 0.2, dots.c, 0.9}, quad o ~ "Bernoulli"(theta). $

An observation collects $n$ patients each undergoing $m$ sessions. Because
neither patient order nor session order matters, an observation is sufficiently
summarized by the histogram

$ O equiv chevron.l n_0, n_1, dots.c, n_m chevron.r, quad
  n_k = sum_(i=1)^n bb(1){sum_(j=1)^m o_(i,j) = k}, quad
  sum_(k=0)^m n_k = n. $

The total improvement count is $S = sum_(i,j) o_(i,j) ~ "Binomial"(n dot m, theta)$.
All detection experiments in this repository use $n = 1$, $m = 7$, so $O$ reduces
to a single patient's success count in $\{0, dots.c, 7\}$.

The observation likelihood, implemented in `rsa/environment.py` as
`get_obs_prob`, treats each patient as an independent $"Binomial"(m, theta)$ draw:

$ P(O | theta) = binom(n, n_0 med n_1 med dots.c med n_m)
  product_(k=0)^m [binom(m, k) theta^k (1 - theta)^(m - k)]^(n_k). $

== Utterances and truth

Speakers do not report counts. They report a quantifier–predicate statement drawn
from

$ Q = {"none", "some", "most", "all"}, quad P = {"ineffective", "effective"}. $

For $n = 1$ the utterance space is $cal(U) = {(q, p) : q in Q, p in P}$, realized
as _"The patient had $q$ sessions $p$."_ For $n > 1$ utterances nest two
quantifiers, $(q_1, q_2, p)$, realized as _"$q_1$ patients had $q_2$ sessions
$p$."_

Truth conditions are the standard quantifier semantics on the count $k$ of items
satisfying the predicate out of $t$ total:

$ "none": k = 0, quad "some": k >= 1, quad "most": k > t\/2, quad "all": k = t. $

This yields the indicator $"Truth"(u; O) in {0, 1}$, with
$[|u|]$ denoting the set of observations for which $u$ is semantically true.

This coarse vocabulary is the *linguistic bottleneck* that makes the whole
problem interesting. Many distinct observations share the same true utterance,
and many distinct utterances are simultaneously true of one observation, so a
speaker can select among truths to steer a listener without ever lying. With 3 of
7 sessions effective, both _"most sessions ineffective"_ and _"some sessions
effective"_ hold; they invite opposite conclusions.

= RSA agents <agents>

All agents share one sequential-Bayes template: $P^((0))$ is a specified prior,
and $P^((t))(dot) = P^((t-1))(dot | "signal at round" t-1)$. Beliefs carry across
rounds; the world state $theta$ is fixed within a run.

== Literal speaker $S_0$ and literal listener $L_0$

$S_0$ samples uniformly among literally true utterances:

$ P_(S_0)(u | O) prop "Truth"(u; O). $

$L_0$ inverts $S_0$, marginalizing over observations:

$ P_(L_0)^((t))(theta | u^((t))) prop P_(L_0)^((t))(theta)
  sum_(O') P_(S_0)(u^((t)) | O') P(O' | theta). $

== Pragmatic speaker $S_1$

$S_1$ trades informativeness against persuasion. The world type
$omega in {"coop", "strat"}$ fixes the available goals,

$ Psi(omega) = cases(
  {"inf"} & "if" omega = "coop",
  {"pers"^-, "inf", "pers"^+} quad & "if" omega = "strat",
) $

and the utterance policy is a softmax over a truth-gated utility with inverse
temperature $alpha > 0$:

$ P_(S_1)^((t))(u | O^((t)), psi, alpha) prop
  "Truth"(u; O^((t))) dot
  "Inf"_(S_1)^((t))(u; O^((t)))^(alpha beta) dot
  "PersStr"_(S_1)^((t))(u; psi)^(alpha (1 - beta)), $

where $beta = bb(1){psi = "inf"}$ switches the speaker between a purely
informative and a purely persuasive objective. The two utility terms are

$ "Inf"_(S_1)^((t))(u; O) = P_(L_0)^((t))(O | u), quad
  "PersStr"_(S_1)^((t))(u; psi) = cases(
    bb(E)_(L_0)^((t))[theta | u] & "if" psi = "pers"^+,
    1 - bb(E)_(L_0)^((t))[theta | u] quad & "if" psi = "pers"^-,
    1 & "if" psi = "inf".
  ) $

Informativeness is the probability a literal listener recovers the true
observation; persuasiveness is the direction in which the utterance moves a
literal listener's posterior mean.

#block(fill: luma(247), inset: 8pt, radius: 3pt)[
  *Code naming.* The implementation labels the goals `"inf"`, `"high"`
  ($=psi = "pers"^+$), and `"low"` ($= psi = "pers"^-$). The sweep's
  `run_config.json` records this under `psi_label_map`. Where this document says
  $"pers"^+$ the parquet columns say `high`.
]

== Pragmatic listener $L_1$

$L_1$ inverts $S_1$ while maintaining a joint posterior over the world state and
the speaker's hidden parameters:

$ P_(L_1)^((t))(theta, psi, alpha | u^((t))) prop
  P_(L_1)^((t))(theta, psi, alpha)
  sum_(O') P_(S_1)^((t))(u^((t)) | O', psi, alpha) P(O' | theta). $

The prior on $psi$ is exactly what separates the two listener types:

$ P_(L_1)^((0))(psi | omega = "coop") = bb(1){psi = "inf"}, quad
  P_(L_1)^((0))(psi | omega = "strat") = 1/3 " for each " psi in Psi("strat"). $

Marginals follow by summing the joint over the nuisance coordinates, e.g.
$P_(L_1)^((t))(theta | u) = sum_(psi', alpha') P_(L_1)^((t))(theta, psi', alpha' | u)$.

== Higher levels

$S_2$ has the same functional form as $S_1$ with its internal $L_0$ replaced by
$L_1$, so that $"Inf"_(S_2)^((t))(u; O) = P_(L_1|omega)^((t))(O | u)$ and
$"PersStr"_(S_2)$ is defined against $L_1$'s posterior mean. Past level 2 the
recursion is structurally fixed: every $S_n$ for $n >= 2$ reuses the $S_2$
machinery with index $n-1$ on its internal listener, and every $L_n$ for
$n >= 1$ reuses the $L_1$ machinery with index $n$ on its internal speaker.
`rsa/speaker2.py` implements this; the detection experiments stay at $S_1$.

= The detection problem <detection>

== Setting

Detection fixes the listener to the *credulous* instance

$ L_1^"inf" := L_1 (omega = "coop"), $

whose prior places all mass on an informative speaker. This is Fang's
$L_1$-coop, and in code it is the internal `naive` listener inside
`DetectionListener` (`rsa/detection/listener.py`).

The data-generating speaker is an $S_1$ with a true goal $psi^star$ and
rationality $alpha$. The listener's internal speaker model is always
$S_1 (dot | dot, "inf", alpha)$. Whenever $psi^star != "inf"$ the listener's
generative model is *misspecified*, and detection asks whether that
misspecification is visible in the utterance stream. Formally,

$ H_0 : psi^star = "inf", quad quad H_A : psi^star in {"pers"^+, "pers"^-}. $

Note the listener is given the correct $alpha$. Detection is a test of the goal
only, not of the rationality parameter.

== Round-$t$ belief state

Everything a score needs at round $t$ is built from four objects, bundled by
`ScoreContext` in `rsa/detection/scores.py` so that all scores see one consistent
snapshot. Writing $q_t$ for the listener's push-forward onto observations and
$k_t$ for its internal speaker model:

$ q_t (O) = P_(L_1^"inf")^((t))(O) = sum_(theta') P(O | theta') P_(L_1^"inf")^((t))(theta'),
  quad quad
  k_t (u | O) = P_(S_1)^((t))(u | O, "inf", alpha). $

From these come the *prior predictive* over utterances and, by Bayes, the
listener's *posterior over observations* given the utterance actually heard:

$ p_t (u) = sum_O k_t (u | O) q_t (O), quad quad
  q_t (O | u) = frac(k_t (u | O) med q_t (O), p_t (u)). $

The posterior predictive $r_t (v | u) = sum_O k_t (v | O) q_t (O | u)$ is also
available and is used by one score. Throughout,
$H(a) = -sum_x a(x) log a(x)$ denotes entropy, and logs are clamped below at
$epsilon = 10^(-12)$ so that $0 log 0 = 0$.

= Detection scores <scores>

Every score is a per-round statistic of the heard utterance $u^((t))$, centered so
that it has *mean zero under $H_0$*. Positive values are evidence against the
informative-speaker null. The scores differ along two axes: which distribution
over $O$ they use, and whether the log is taken inside or outside the
marginalization over $O$.

== Prior-predictive surprisal (`surp2`)

Centered self-information of the heard utterance under the listener's prediction
made _before_ hearing it:

$ s_"surp2"^((t)) = -log p_t (u^((t))) - H(p_t). $

Under $H_0$ the heard utterance is genuinely a draw from $p_t$, so the score is
the deviation of a log-loss from its own expectation and
$bb(E)[s_"surp2"^((t))] = 0$ by definition of entropy. Its null variance is the
*varentropy* of the prior predictive:

$ sigma_"surp2"^(2,(t)) = op("Var")_(u ~ p_t)[-log p_t (u)]. $

This score marginalizes over $O$ first, then takes the log. It therefore sees
only the utterance distribution, not the observation structure behind it.

== Observation-level suspicion (`sus`)

Suspicion reverses that order. Define the per-observation *excess surprise*

$ b_t (O, u) = -log k_t (u | O) - H(k_t (dot | O)), $

which for each fixed $O$ has mean zero under $k_t (dot | O)$. The score averages it
over the listener's posterior about which observation the speaker actually saw:

$ s_"sus"^((t)) = sum_O q_t (O | u^((t))) med b_t (O, u^((t))). $

Because the inner quantity is mean-zero for every $O$ separately,
$bb(E)[s_"sus"^((t))] = 0$ under $H_0$ as well. Taking the log inside the
marginalization is what lets the score notice that an utterance is unlikely
*given the observations that would make it worth saying* — the signature of
strategic vagueness — rather than merely globally unlikely.

== Variance of `sus` and the total-variance correction <ltv>

The natural variance estimate weights the per-$O$ varentropy by the same
posterior:

$ sigma_"naive"^(2,(t)) = sum_O q_t (O | u^((t)))
  op("Var")_(u ~ k_t (dot | O))[-log k_t (u | O)]. $

This *overstates* the variance. Since $O$ is never observed, the law of total
variance splits the true variance into a within-$O$ part and a between-$O$ part,
and only the within-$O$ part is genuine noise in the score; the between-$O$
spread of the conditional mean is already integrated out by the posterior
average. The corrected variance subtracts it:

$ sigma_"corr"^(2,(t)) = sigma_"naive"^(2,(t)) - K_t, quad
  K_t = sum_(u') p_t (u') op("Var")_(O ~ q_t (dot | u'))[b_t (O, u')], $

clipped at zero. This is an *exact identity*, not an approximation: writing
$pi(O, u) = q_t (O) k_t (u | O)$ for the null joint, $EE[b_t | O] = 0$ by definition of
entropy, so $op("Var")_pi [b_t] = EE_u [sigma_"naive"^2 (u)]$, and the law of total
variance gives

$ op("Var")_(u ~ p_t)[s_"sus"(u)] = EE_u [sigma_"naive"^(2)(u)] - K_t. $

This was verified numerically to machine precision at 120 frozen listener states
(`notebooks/variance_diagnostics.ipynb`). Empirically the correction is the difference
between a per-round variance ratio of $0.88$ and $0.99$.

#block(fill: luma(245), inset: 8pt, radius: 3pt, width: 100%)[
  *Implementation note (fixed 2026-08-07).* The identity holds for
  $EE_u [sigma_"corr"^2 (u)]$, so the clip must not be applied term by term. An earlier
  version of `scores.py` used $max(sigma_"naive"^2 (u) - K_t, 0)$ *per utterance*, while
  $K_t$ is a single constant for the state; every utterance with
  $sigma_"naive"^2 (u) < K_t$ was clipped upward, adding the clipped mass back and
  inflating the effective $sigma^2$ by up to 9% at $alpha = 1$, $theta^* = 0.1$ (where
  those utterances carry ~60% of the probability mass).

  `make_sus_variant` now returns the *unclipped* difference — an individual round may be
  negative — and `SequentialTest` clips the running average
  $max(sum_i sigma_"corr"^(2,(i)) \/ t, 0)$ before the square root, which is the correct
  place to enforce non-negativity. Note that `results/full_sweep/` predates the fix and
  still carries the inflated values at $alpha <= 2$.
]

== Variant family and why only variant 1 survives

The suspicion score generalizes: replace the posterior weighting $q_t (O | u)$
with any column-normalized weighting $W_v (O | u)$, giving
$s_"sus_v" = sum_O W_v (O | u) b_t (O, u)$. Four were implemented:

#table(
  columns: (auto, 1fr, auto),
  align: (left, left, left),
  stroke: (x, y) => if y == 0 { (bottom: 0.7pt) } else { (bottom: 0.3pt + luma(200)) },
  table.header([*Variant*], [*Unnormalized weight $W_v (O | u) prop$*], [*Status*]),
  [`sus_1`], [$k_t (u | O) med q_t (O)$ — the true posterior], [*adopted*],
  [`sus_3`], [$"Truth"(u; O) med q_t (O)$ — truth-restricted $L_1$ prior], [dropped],
  [`sus_4`], [$"Truth"(u; O) med L_0^((t))(O)$ — truth-restricted $L_0$ prior], [dropped],
  [`sus_4b`], [$"Truth"(u; O) med L_0^((t))(O) \/ T(O)$, with $T(O) = |{u : "Truth"(u;O)}|$], [dropped],
)

Only variant 1 is theoretically sound. The correction in @ltv is a
law-of-total-variance identity, and it holds only when the implied joint
$p_"pred,v" (u) dot W_v (O | u)$ equals the true null joint $q_t (O) k_t (u | O)$.
That is an identity for $W_1$, which _is_ the real posterior. For variants 3, 4,
and 4b the weighting is an ad-hoc reweighting with the $k_t (u | O)$ factor
stripped out, so the joint is fictitious and the identity fails. In practice
their naive variance understates the truth by $1.6$–$1.7 times$ per round (giving
95–97% false-alarm rates), while their "corrected" variance overshoots and clamps
to zero (giving no detections at all). They are dominated by `sus_1` on every
axis and add no detection power; all three are retired.

#block(fill: luma(247), inset: 8pt, radius: 3pt)[
  *On `surp1`.* A third core score, the posterior-predictive surprisal
  $s_"surp1"^((t)) = -log r_t (u^((t)) | u^((t))) - H(r_t (dot | u^((t))))$, is
  implemented and recorded in the raw parquet data. It uses the heard utterance
  twice — once to form the posterior, once to score itself against it — and its
  variance formula runs about $2 times$ too large (empirical/theoretical ratio
  $approx 0.46$), which makes its test silently over-conservative. It is
  *collected but excluded from all presented figures and results*, and is
  documented here only so the extra parquet columns are not mistaken for live
  results.
]

= Sequential test <test>

Given any per-round score $s^((i))$ with null variance $sigma^(2,(i))$, the test
in `rsa/detection/sequential_test.py` accumulates a running mean and a running
average variance,

$ "Sus"^((t)) = 1/t sum_(i=1)^t s^((i)), quad quad
  overline(sigma)^(2,(t)) = 1/t sum_(i=1)^t sigma^(2,(i)), $

and fires the first time the running mean exceeds a $z$-scaled boundary:

$ "Sus"^((t)) > c dot overline(sigma)^((t)) / sqrt(t). $

The first crossing time is $tau$. Setting $c = +infinity$ disables crossing and
makes the test a pure observer, which is how the full sweep is run so that
downstream analyses can recompute error rates for any $c$ without re-simulating.

Because `sus_1` reports two variances, the test tracks *both in parallel* —
two running sigmas, two thresholds, two crossing times ($tau_"naive"$ and
$tau_"corrected"$) — from a single pass. A score function may return either
$(s, sigma^2)$ or $(s, sigma_"naive"^2, sigma_"corr"^2)$.

The test is a passive observer by default. `DetectionListener` can optionally let
a crossing drive a switch from the credulous listener to a full vigilant $L_1$,
either *hard* (vigilant starts from a uniform prior) or *soft* (vigilant inherits
the credulous $theta$-marginal, spread uniformly over $psi$). Switching is
disabled in all reported experiments — the studies below characterize the
detector itself before wiring it to a consequence.

== What the null should look like

Under $H_0$ the generative model matches the listener's internal model, giving
three checkable predictions: $bb(E)[s^((t))] = 0$ so $"Sus"^((t)) -> 0$;
$op("Var")[s^((t))] \/ bb(E)[sigma^(2,(t))] approx 1$; and
$op("Var")["Sus"^((t))] ~ 1\/t$. Under $H_A$ both scores acquire a strictly
positive mean — a KL-like gap between the listener's informative prior predictive
and the true persuasive utterance distribution — so $"Sus"^((t))$ drifts upward.
The empirical question is how fast, and for which $(theta^star, alpha)$.

= Experiments and findings <experiments>

Four studies live in `experiments/`, each writing to a matching directory under
`results/`.

== Group 1 — is the FPR inflation a variance bug?

`experiments/group1/` · 200 sims $times$ 150 rounds, $alpha = 3$, $c = 2$,
null sweep over $theta^star in {0.1, 0.3, 0.5, 0.7, 0.9}$.

A one-sided $c = 2$ boundary should give roughly 2.3% false alarms. Observed
false-positive rates were 23.1% (`surp2`) and 12.6% (`sus`). Four sub-experiments
localized the cause:

- *1.1* — crossings are spread across all $t$, not bunched in an early transient.
- *1.2* — a warm-up gate barely helps: $t_"warmup" = 10$ only moves FPR to
  18.1% and 9.3%, at no TPR cost. Cheap insurance, not a fix.
- *1.3* — the per-round variance formula is well calibrated (`surp2` ratio
  $1.00$, `sus` $0.89$).
- *1.4* — the $1\/t$ shrinkage of $op("Var")["Sus"^((t))]$ holds to within about
  10%, so the CLT scaling is intact.

*Conclusion:* the inflation is neither a miscomputed $sigma^(2,(t))$ nor a broken
CLT. It is *multiple-testing inflation* — a fixed pointwise $c$ is compared
against the boundary at all 150 rounds, and first-crossing probability against a
fixed-$z$ boundary grows with the horizon.

== sus variants — which weighting to keep

`experiments/sus_variants/` · same configuration, five scores attached as passive
observers, both variance formulas logged per score.

This study produced the verdict in @scores: `sus_1` + corrected variance is the
only calibration-consistent combination (per-round ratio $0.88 -> 0.99$,
running-mean $0.84 -> 0.97$), while variants 3/4/4b are dominated and
theoretically unsound. Notably the correction moves `sus_1`'s FPR the *wrong*
way, 14.2% $->$ 21.1%, because the naive formula had been accidentally
conservative. The correction is a free honesty fix for the per-round test; it is
not an FPR fix.

== Full sweep — the main dataset

`experiments/full_sweep/` · the headline artifact. The grid factorizes as

$ underbrace({0.1, dots.c, 0.9}, theta^star med (9)) times
  underbrace({"inf", "pers"^+, "pers"^-}, psi^star med (3)) times
  underbrace({1, 1.5, 2, 2.5, 3, 4, 5, 7, 10, 15, 20}, alpha med (11))
  = 297 "cells", $

with 200 sims $times$ 150 rounds per cell, for 8,910,000 trajectory rows. Sanity
checks all pass, including null $"Sus"(t=150)$ means at $-0.0007$ (`surp2`) and
$+0.0000$ (`sus_1`).

Sweeping the cutoff $c$ over the recorded trajectories gives the operating
characteristics:

#table(
  columns: 6,
  align: (left, center, center, center, center, center),
  stroke: (x, y) => if y == 0 { (bottom: 0.7pt) } else { (bottom: 0.3pt + luma(200)) },
  table.header([*Score*], [*$c$*], [*FPR*], [*TPR $"pers"^+$*], [*TPR $"pers"^-$*], [*TPR $minus$ FPR*]),
  [`surp2`], [2.0], [0.234], [0.743], [0.740], [0.508],
  [`surp2`], [3.0], [0.069], [0.667], [0.669], [0.598],
  [`surp2`], [3.5], [0.034], [0.635], [0.636], [0.601],
  [`sus_1`], [2.0], [0.251], [0.997], [0.996], [0.746],
  [`sus_1`], [3.0], [0.065], [0.980], [0.978], [0.914],
  [`sus_1`], [3.5], [0.035], [0.962], [0.960], [0.926],
)

`sus_1` dominates `surp2` everywhere: at matched FPR it detects persuasion
roughly 30 percentage points more often, and at $c = 3.5$ it holds 96% TPR
against a 3.5% false-alarm rate. The observation-level score's advantage is
exactly the structural sensitivity noted in @scores — taking the log inside the
$O$-marginalization.

Raising $c$ from 2.0 to 3.5 is also the pragmatic answer to Group 1's finding:
it is an empirically calibrated horizon-level correction, trading a little power
for honest error control.

== Detection comparison

`experiments/run_detection_comparison.py` and `build_report.py` · an earlier
head-to-head of the score family, producing trajectory, latency, power-curve and
correlation plots into `results/detection_comparison/`. Superseded by the full
sweep for headline numbers but retained for its latency and power-curve views.

== Where this leaves the project

The detector works: `sus_1` with the corrected variance separates persuasive from
informative speakers with high power, and the score's calibration is verified at
both the per-round and running-mean level. The open problem is entirely the
*stopping boundary*. A fixed-$z$ rule cannot control error over a 150-round
horizon. Three candidate fixes, in rough order of appeal:

+ a law-of-iterated-logarithm boundary,
  $c med overline(sigma)^((t)) sqrt(2 log log t \/ t)$;
+ time-uniform confidence sequences (e.g. Howard–Ramdas betting bounds);
+ an empirically calibrated $c$ from null-simulation quantiles, valid for a
  committed horizon $T$ — effectively what the $c = 3.5$ row above does.

Beyond that, the switching machinery in `DetectionListener` is implemented but
unexercised: the natural next study is whether a detector-triggered switch to a
vigilant $L_1$ recovers Fang's vigilance benefit while paying credulous-listener
costs up to $tau$.

= Repository map

#table(
  columns: (auto, 1fr),
  align: (left, left),
  stroke: (x, y) => if y == 0 { (bottom: 0.7pt) } else { (bottom: 0.3pt + luma(200)) },
  table.header([*Path*], [*Contents*]),
  [`rsa/`], [Core library: `core.py` (Belief, Semantics, World), `environment.py` (observation model, truth conditions), `speaker0–2.py`, `listener0–1.py`, `game.py`, `setup.py`],
  [`rsa/detection/`], [The detection extension: `scores.py`, `sequential_test.py`, `listener.py`],
  [`rsa/experimental/`], [Switching listeners and $S_1$ variants; not used in reported results],
  [`experiments/`], [One package per study: `group1/`, `sus_variants/`, `full_sweep/`, plus the comparison scripts],
  [`results/`], [Outputs mirroring `experiments/`; each has `run_config.json` and a `summary.md` or `sanity.md`],
  [`docs/`], [This document; `TYPST.md` for the toolchain; `legacy/` for the superseded LaTeX sources],
  [`notebooks/`], [Analysis notebooks: `reproduce_paper`, `suspicion_analysis`, `sandbox`],
  [`tests/`], [`test_detection.py`, `test_sus_variants.py`],
  [`archive/`], [Pre-refactor exploratory notebooks, figures, pickles, and rendered reports],
)

== Notation and code cross-reference

#table(
  columns: (auto, auto, 1fr),
  align: (left, left, left),
  stroke: (x, y) => if y == 0 { (bottom: 0.7pt) } else { (bottom: 0.3pt + luma(200)) },
  table.header([*Symbol*], [*Code / data column*], [*Meaning*]),
  [$q_t (O)$], [`ScoreContext.L1_O`], [Listener belief pushed onto observations],
  [$k_t (u | O)$], [`ScoreContext.S1_table`], [Internal informative-speaker model],
  [$p_t (u)$], [`ScoreContext.P_prior`], [Prior predictive over utterances],
  [$q_t (O | u)$], [`ScoreContext.L1_O_given_u`], [Posterior over observations],
  [$b_t (O, u)$], [`ScoreContext.B_matrix`], [Per-observation excess surprise],
  [$s_"surp2"^((t))$], [`surp2_score`], [Prior-predictive surprisal],
  [$s_"sus"^((t))$], [`sus1_score`], [Observation-level suspicion],
  [$sigma_"corr"^(2,(t))$], [`sus1_sigma2_corrected`], [Total-variance–corrected null variance],
  [$"Sus"^((t))$], [`sus1_Sus`, `surp2_Sus`], [Running mean of the score],
  [$overline(sigma)^(2,(t))$], [`*_sigma_bar2*`], [Running average variance],
  [$tau$], [`SequentialTest.tau_naive` / `.tau_corrected`], [First crossing time; per-score `tau_*` columns in `tau_summary.parquet`],
)

#v(1em)
#line(length: 100%, stroke: 0.5pt + luma(180))
#v(0.3em)
#text(size: 9pt)[
  Base model: Ke Fang, _A Computational Account of Epistemic Vigilance: Learning
  from Selective Truths through Bayesian Reasoning_, Stanford University. The
  world setup, agent definitions, and notation in @setup and @agents follow that
  paper; @detection onward is this project's extension.
]
