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
@experiments, are `sus_1` with its exact state-only null variance, and
*no* — at the nominal $c = 2$ false alarms run at roughly 22% against a nominal
2.3%, for reasons that lie in the shape of the stopping boundary rather than in
the score or its variance. The inflation is at least tunable: raising the cutoff
to $c = 3.5$ brings it to 3.1% while `sus_1` still detects 95% of persuasive
speakers.

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
`rsa/speaker2.py` implements this as the $S_1$ template with the internal
listener swapped, and nothing else: informativeness is the observation-level
$P_(L_1)(O | u)$, persuasiveness is $L_1$'s posterior mean, and the speaker's
own belief over $theta$ is carried but kept out of the policy. (Earlier versions
deviated on all three counts — a state-level cross-entropy informativeness that
read $S_2$'s private belief, a mean-centred and truncated persuasiveness, and an
$exp(alpha log_2 dot)$ softmax. Those are gone; see the switching report.) The
core detection studies stay at $S_1$; the switching studies of @switching use
$S_2$.

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

== Null variance of `sus` <ltv>

The sequential test needs the variance of the score under the listener's own
predictive distribution. With $|cal(U)| = |cal(O)| = 8$ this is an exact finite
sum and is computed directly:

$ V_t = sum_(u) p_t (u) med s_"sus" (u)^2, quad
  s_"sus"(u) = sum_O q_t (O | u) med b_t (O, u). $

The square needs no centring because the score is mean-zero under the listener's
own joint $pi(O, u) = q_t (O) k_t (u | O)$: summing $p_t (u) s_"sus"(u)$ over $u$
collapses to $sum_O q_t (O) (H_O - H_O) = 0$, by definition of the entropy $H_O$.

Two properties matter. $V_t$ depends only on the listener state, not on the
utterance actually heard — it is the same number whichever $u$ arrives — and it
is a probability-weighted sum of squares, so it is non-negative by construction.
Nothing is clipped anywhere.

The law of total variance gives the same quantity a second form,

$ V_t = sum_O q_t (O) med v_t (O) - K_t, quad
  K_t = sum_(u') p_t (u') op("Var")_(O ~ q_t (dot | u'))[b_t (O, u')], $

with $v_t (O)$ the varentropy of row $O$. The two forms are checked against each
other to machine precision at twelve frozen listener states in
`tests/test_sus_variants.py`, together with a Monte-Carlo check that sampling
$u ~ p_t$ reproduces $V_t$.

#block(fill: luma(245), inset: 8pt, radius: 3pt, width: 100%)[
  *Implementation note.* Earlier versions returned the per-round proxy
  $sigma_"naive"^2 (u^((t))) - K_t$, using the *posterior*-weighted varentropy
  $sum_O q_t (O | u^((t))) v_t (O)$ in place of the first term of the LTV form.
  Its expectation over $u$ is $V_t$, so pooled calibration looked correct — the
  per-round ratio sat at $0.99$ — but round by round it is a different number,
  and one that moves with the score it is meant to scale.

  On the old null trajectories that proxy correlates with its own numerator at
  $rho = -0.27$ at $alpha = 1.5$, and comes out non-positive on 35% of rounds.
  A threshold that shrinks precisely when the score spikes manufactures
  crossings, which is what the false-alarm rate at low $alpha$ was recording.
  Replacing it with the exact $V_t$ removes the $alpha <= 2$ blow-up entirely.
  See `results/full_sweep/variance_fix_diff.md` for the before/after tables.
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

Only variant 1 is theoretically sound. The exact variance of @ltv rests on the
score being mean-zero under the true null joint $q_t (O) k_t (u | O)$, which
requires the implied joint $p_"pred,v" (u) dot W_v (O | u)$ to equal it. That is
an identity for $W_1$, which _is_ the real posterior. For variants 3, 4 and 4b
the weighting is an ad-hoc reweighting with the $k_t (u | O)$ factor stripped
out, so the joint is fictitious, the score is not mean-zero, and no analogue of
$V_t$ exists. They were also dominated by `sus_1` on every empirical axis. All
three are retired: `make_sus_variant` raises `NotImplementedError` for them.

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

A score function returns $(s, sigma^2)$. Because $sigma^2$ is exact and
non-negative by construction, the test accumulates it directly: there is one
running sigma, one threshold and one crossing time $tau$, and no clipping
anywhere. A negative variance would be a bug in the score function and is
raised rather than absorbed.

The test is a passive observer by default. `DetectionListener` can optionally let
a crossing drive a switch from the credulous listener to a full vigilant $L_1$.
The tests score $u_tau$ against the credulous belief as it stands; if one
crosses, the switch happens before any belief update and $u_tau$ is then
absorbed exactly once, by the vigilant listener. The credulous listener freezes
at $tau - 1$. Two switch types:

#table(
  columns: (auto, 1fr),
  align: (left, left),
  stroke: (x, y) => if y == 0 { (bottom: 0.7pt) } else { (bottom: 0.3pt + luma(200)) },
  table.header([*`switch_type`*], [*Belief the vigilant $L_1$ starts from at $tau$*]),
  [`"hard"`], [*Retrospective.* A vigilant $L_1$ that has listened since round 1: the listener keeps a shadow vigilant $L_1$ updated in parallel with the credulous one, and at $tau$ the shadow becomes the active listener. Identical to an always-vigilant $L_1$ on the same stream, with no replay.],
  [`"soft"`], [Inherits the credulous $theta$-marginal from *before* $u_tau$, spread uniformly over $psi$ — the pre-$tau$ credulous belief, direction-blind at $tau$ — then applies $u_tau$ vigilantly. At $tau = 1$ it coincides with `"hard"`.],
)

Every sub-listener is updated from the same per-round snapshot of the speaker's
tables $P_(S_1)^((i))(u | O, psi)$, which `DetectionListener.update` also
records into `table_history`; `Listener1.update_with_tables` updates from an
explicit table triple instead of consulting the speaker. That is what makes any
listener replayable offline on a stored utterance stream
(`rsa/detection/replay.py`).

`DetectionListener.peek(u)` returns the $theta$-marginal the listener *would*
hold after hearing $u$ — detector update and switch included — without mutating
anything. This is what lets an $S_2$ put the consequence of tripping the
detector inside its own utility. Switching is disabled in the four studies of
@experiments, which characterize the detector alone; the studies of @switching
wire it to a consequence.

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
false-positive rates are 22.8% (`surp2`) and 18.3% (`sus`). Four sub-experiments
localized the cause:

- *1.1* — crossings are spread across all $t$, not bunched in an early transient.
- *1.2* — a warm-up gate helps but does not fix: $t_"warmup" = 10$ moves FPR to
  17.3% and 14.0%, at no TPR cost.
- *1.3* — the per-round variance is well calibrated (`surp2` ratio $0.99$,
  `sus` $1.00$).
- *1.4* — the $1\/t$ shrinkage of $op("Var")["Sus"^((t))]$ holds to within about
  4%, so the CLT scaling is intact.

(These are the re-run figures using the exact variance of @ltv. The study
originally reported 12.6% for `sus` at $c = 2$ and a per-round ratio of $0.89$;
both came from the retired posterior-weighted variance, which overstated
$sigma^2$ and so under-fired. The conclusion below is unchanged, and 1.3 is now
a cleaner result than it was.)

*Conclusion:* the inflation is neither a miscomputed $sigma^(2,(t))$ nor a broken
CLT. It is *multiple-testing inflation* — a fixed pointwise $c$ is compared
against the boundary at all 150 rounds, and first-crossing probability against a
fixed-$z$ boundary grows with the horizon.

== sus variants — which weighting to keep

`experiments/sus_variants/` · same configuration, with `surp2` and `sus_1`
attached as passive observers.

This study produced the verdict in @scores: `sus_1` is the only
calibration-consistent variant, while 3/4/4b are dominated and theoretically
unsound. Re-run against the exact variance of @ltv, its per-round calibration
ratio is $1.000$ for `sus_1` and $0.994$ for `surp2`, and the running-mean ratios
are $1.010$ and $1.008$ — so both scores are honest at both levels.

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
  [`surp2`], [5.0], [0.005], [0.556], [0.559], [0.553],
  [`sus_1`], [2.0], [0.219], [0.996], [0.995], [0.777],
  [`sus_1`], [3.0], [0.065], [0.974], [0.970], [0.907],
  [`sus_1`], [3.5], [0.031], [0.947], [0.947], [0.916],
  [`sus_1`], [5.0], [0.011], [0.830], [0.826], [0.817],
  [`sus_1`], [7.0], [0.003], [0.681], [0.683], [0.679],
)

Rates are pooled over $theta^star$ and over $alpha >= 1.5$; the $alpha = 1$ cells
are simulated and stored but excluded here, since at $alpha = 1$ the pragmatic
speaker is nearly literal and there is almost no persuasion to detect.

`sus_1` dominates `surp2` throughout. At $c = 3.5$ it holds 94.7% power against a
3.1% false-alarm rate, where `surp2` at the same cutoff manages 63.5% against
3.4%; at every matched error rate the gap is 25–30 percentage points. The
observation-level score's advantage is exactly the structural sensitivity noted
in @scores — taking the log inside the $O$-marginalization.

Both scores now behave as a fixed-$z$ rule should: the false-alarm rate falls
monotonically with $c$, reaching 0.3–0.5% by $c = 7$. Raising $c$ from 2.0 to 3.5
is therefore a usable horizon-level correction for Group 1's finding, trading
about 5 points of power for a sevenfold cut in false alarms.

#block(fill: luma(245), inset: 8pt, radius: 3pt, width: 100%)[
  *These numbers supersede an earlier version of this table.* Read with the
  retired per-round variance proxy (@ltv), `sus_1` appeared to have an
  irreducible false-alarm floor: 9.1% at $c = 3.5$, falling only to 5.8% at
  $c = 7$ while power drained away, and a false-alarm rate that was
  *non-monotone* in $alpha$ — 0.374 at $alpha = 1.5$, 0.013 at $alpha = 3$,
  rising again above $alpha = 7$. Both effects were artefacts of the proxy.
  With the exact $V_t$ the floor is gone and the $alpha$-profile is monotone
  (0.002 at $alpha = 1.5$ rising smoothly to 0.063 at $alpha = 20$). The
  before/after tables are in `results/full_sweep/variance_fix_diff.md`.
]

== Detection comparison

`experiments/run_detection_comparison.py` and `build_report.py` · an earlier
head-to-head of the score family, producing trajectory, latency, power-curve and
correlation plots into `results/detection_comparison/`. Superseded by the full
sweep for headline numbers but retained for its latency and power-curve views.

== Where this leaves the project <where>

The detector works: `sus_1` with its exact null variance separates persuasive from
informative speakers with high power, and the score's calibration is verified at
both the per-round and running-mean level — the per-round ratio sits within 1% of
1.0 at every $alpha$. The open problem is the *stopping boundary*. At the nominal
$c = 2$ the test fires on 22% of null runs against a nominal 2.3%, a tenfold
inflation that Group 1 traced to multiple testing across the horizon rather than
to the score or its variance.

What has changed is that the problem is now the horizon effect and nothing else.
With the retired variance proxy the false-alarm rate was also badly
$alpha$-dependent and could not be tuned below about 6%; with the exact $V_t$ it
falls monotonically with $c$ and varies only mildly across the grid (0.002 to
0.066 at $c = 3.5$). Three candidate fixes, in rough order of appeal:

+ a law-of-iterated-logarithm boundary,
  $c med overline(sigma)^((t)) sqrt(2 log log t \/ t)$;
+ time-uniform confidence sequences (e.g. Howard–Ramdas betting bounds);
+ an empirically calibrated $c$ from null-simulation quantiles, valid for a
  committed horizon $T$.

The third is now a serious contender rather than a fallback. A single scalar
cutoff calibrated against pooled null quantiles is only as good as the uniformity
of the null crossing rate across the grid, and that rate is now monotone in
$alpha$ and small everywhere; calibrating to the worst cell costs little. The
first two remain preferable in principle because their boundaries adapt to the
observed $overline(sigma)^((t))$ and stay valid at a horizon not committed to in
advance.

All three are evaluable offline: the sweep ran every test as a pure observer with
$c = infinity$ and recorded $"Sus"^((t))$ and $overline(sigma)^((t))$ at every
round, so any stopping rule can be scored against the stored trajectories without
re-simulating. `results/full_sweep_v2/analyses/stopping_rules.ipynb` does exactly
this for five rules.

== Switching: wiring the detector to a consequence <switching>

The switching machinery is now exercised. `experiments/switching/` runs a
detector-triggered switch to a vigilant $L_1$ against both $S_1$ and $S_2$
speakers, and `results/switching/report.md` is its report; the spec is
`switching_experiments_spec.md`. The headline is that the *retrospective* hard
switch recovers Fang's vigilance benefit in full and immediately — after $tau$ its
belief equals the always-vigilant listener's to machine precision, so the entire
cost of having started credulous is confined to the rounds before $tau$ — while
the cheaper soft and amnesic switches do not. Under an honest speaker a false
alarm costs a retrospective switcher essentially nothing, which inverts the usual
tuning advice: the boundary should be set for power, not for calibration.

Three results from the $S_2$ studies qualify that, and all three are negative in
a useful way.

+ *Level mismatch dominates everything else.* An $L_1$ detector facing an
  _honest_ $S_2^"inf"$ alarms on 0.7% of runs at $alpha = 1.5$ but *98–100%* at
  $alpha = 10$, because the statistic tests "is my $S_1^"inf"$ model wrong?" and
  one extra level of speaker reasoning is enough to make the answer yes. At
  $alpha = 10$ the honest and persuasive rates are both $approx 1$, so the test
  has no discriminating power at all. Giving the detector the right level (an
  $L_2$ inverting the true $S_2$, experiment C2) collapses the false-alarm rate
  to *0.2%* at $c = 3.5$ and makes it flat in $alpha$ — but costs 22–38 points
  of power, since a level-2 persuader is genuinely harder to catch.

+ *The arms race is about recursion depth, not feedback.* An $S_2$ modelling a
  _fixed_ vigilant $L_1$ (C1) degrades that listener exactly as much as one
  modelling the switching listener (B): $|bb(E)[theta] - theta^star| = 0.043$
  against $0.039$ at $alpha = 10$. What defeats vigilance is the extra level of
  reasoning, not the closed loop.

+ *The speaker does not evade the detector.* Its $"Sus"(t)$ appears to ride just
  under the boundary, but that plot averages over runs that have not yet
  crossed, and the conditioning forces the survivor mean below the boundary
  whatever the speaker does. The selection-free comparison — unconditional
  crossing rates, B against C1 at matched cells — is flat ($-0.003 plus.minus
  0.007$). `peek` shows the speaker the _consequence_ of tripping the detector,
  but the utility is one-round myopic, so it can never trade a loss now for
  staying undetected later.

= Repository map

#table(
  columns: (auto, 1fr),
  align: (left, left),
  stroke: (x, y) => if y == 0 { (bottom: 0.7pt) } else { (bottom: 0.3pt + luma(200)) },
  table.header([*Path*], [*Contents*]),
  [`rsa/`], [Core library: `core.py` (Belief, Semantics, World), `environment.py` (observation model, truth conditions), `speaker0–2.py`, `listener0–1.py`, `game.py`, `setup.py`],
  [`rsa/detection/`], [The detection extension: `scores.py`, `sequential_test.py`, `listener.py` (detector + switching), `replay.py` (offline replay of any listener on a stored stream)],
  [`rsa/experimental/`], [Superseded switching listeners and $S_1$ variants; not used in reported results],
  [`experiments/`], [One package per study: `group1/`, `sus_variants/`, `full_sweep/`, `switching/` (one runner, one JSON grid per study), plus the comparison scripts],
  [`results/`], [Outputs mirroring `experiments/`; each has `run_config.json` and a `summary.md`, `sanity.md` or `report.md`],
  [`docs/`], [This document; `TYPST.md` for the toolchain; `legacy/` for the superseded LaTeX sources],
  [`notebooks/`], [Analysis notebooks: `reproduce_paper`, `suspicion_analysis`, `sandbox`],
  [`tests/`], [`test_detection.py`, `test_sus_variants.py`, `test_switching.py`, `test_speaker2.py`, `test_replay.py`, `test_switching_runner.py`],
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
  [$V_t$], [`sus1_sigma2`], [Exact state-only null variance of the score],
  [$"Sus"^((t))$], [`sus1_Sus`, `surp2_Sus`], [Running mean of the score],
  [$overline(sigma)^(2,(t))$], [`*_sigma_bar2*`], [Running average variance],
  [$tau$], [`SequentialTest.tau`], [First crossing time; per-score `tau_*` columns in `tau_summary.parquet`],
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
