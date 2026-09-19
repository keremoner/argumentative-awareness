#set page(paper: "us-letter", margin: 1in, numbering: "1")
#set text(size: 11pt)
#set par(justify: true)
#set heading(numbering: "1.")
#set figure(numbering: "1")
#show figure.caption: set text(size: 9pt)
#show figure: set block(breakable: false)

#let swfig(name, cap) = figure(
  image("/results/switching/figures/" + name, width: 100%),
  caption: cap,
)
#let sqfig(name, cap) = figure(
  image("/results/full_sweep_v2/figures/" + name, width: 100%),
  caption: cap,
)
#let hrule = table.hline(stroke: 0.7pt)
#let tstroke = (x, y) => if y == 0 { (bottom: 0.7pt) } else { (bottom: 0.3pt + luma(200)) }

#align(center)[
  #text(size: 17pt, weight: "bold")[Argumentative Awareness]
  #v(0.2em)
  #text(size: 12pt)[Model, detection scores, experiments and results]
  #v(0.4em)
  #text(size: 10pt, style: "italic")[Project reference document]
]

#v(1em)

= Scope

This project extends the Rational Speech Act (RSA) opinion-dynamics model of
Fang (2025), _A Computational Account of Epistemic Vigilance_, to a setting in
which the listener does *not* know from the start that it may be facing a
persuasive speaker. In the base model the listener is either credulous
($omega = "coop"$, no hypothesis space over speaker goals) or vigilant
($omega = "strat"$, a fixed prior over goals), and which one it is is given.
Here the listener begins credulous, accumulates a per-round statistic measuring
how badly its informative-speaker model predicts what it hears, and may switch
to a vigilant listener when that statistic crosses a boundary.

The document records the model, the detection machinery, the experiments that
were run, and the results. @setup and @agents define the world and the agents;
@detection and @test define the scores, their null variances and the sequential
test; @inventory lists every dataset; @sweepres, @switchres and @combined
present the results. Figures are reproduced from
`notebooks/sweep_analysis.ipynb` and `notebooks/switching_analysis.ipynb`, and
each caption names the quantity plotted.

= World and communication setup <setup>

The world model is taken unchanged from Fang (2025); notation is shared
throughout.

== Latent state and observations

A population of patients shares a latent per-session improvement rate $theta$
on a discrete grid:

$ theta in Theta = {0.1, 0.2, dots.c, 0.9}, quad o ~ "Bernoulli"(theta). $

An observation collects $n$ patients each undergoing $m$ sessions. Because
neither patient order nor session order matters, an observation is sufficiently
summarized by the histogram

$ O equiv chevron.l n_0, n_1, dots.c, n_m chevron.r, quad
  n_k = sum_(i=1)^n bb(1){sum_(j=1)^m o_(i,j) = k}, quad
  sum_(k=0)^m n_k = n. $

The total improvement count is
$S = sum_(i,j) o_(i,j) ~ "Binomial"(n dot m, theta)$. Every experiment in this
repository uses $n = 1$, $m = 7$, so $O$ reduces to one patient's success count
in ${0, dots.c, 7}$ and $|cal(O)| = 8$.

The observation likelihood, implemented in `rsa/environment.py` as
`get_obs_prob`, treats each patient as an independent $"Binomial"(m, theta)$
draw:

$ P(O | theta) = binom(n, n_0 med n_1 med dots.c med n_m)
  product_(k=0)^m [binom(m, k) theta^k (1 - theta)^(m - k)]^(n_k). $

The listener's grid over $theta$ is not always the same as the set of simulated
true values. The switching studies use the 9-point grid
${0.1, dots.c, 0.9}$ for both; the full sweep gives its listeners an 11-point
grid ${0.0, 0.1, dots.c, 1.0}$ that includes the two endpoints, while still
simulating $theta^star in {0.1, dots.c, 0.9}$.

== Utterances and truth

Speakers do not report counts. They report a quantifier--predicate statement
drawn from

$ Q = {"none", "some", "most", "all"}, quad P = {"ineffective", "effective"}. $

For $n = 1$ the utterance space is $cal(U) = {(q, p) : q in Q, p in P}$,
realized as _"The patient had $q$ sessions $p$"_, so $|cal(U)| = 8$. For
$n > 1$ utterances nest two quantifiers, $(q_1, q_2, p)$, realized as
_"$q_1$ patients had $q_2$ sessions $p$"_.

Truth conditions are the standard quantifier semantics on the count $k$ of
items satisfying the predicate out of $t$ total:

$ "none": k = 0, quad "some": k >= 1, quad "most": k > t\/2, quad "all": k = t. $

This yields the indicator $"Truth"(u; O) in {0, 1}$, with $[|u|]$ denoting the
set of observations for which $u$ is semantically true. Many distinct
observations share a true utterance and many distinct utterances are
simultaneously true of one observation, so a speaker can select among truths
without lying: with 3 of 7 sessions effective, both _"most sessions
ineffective"_ and _"some sessions effective"_ hold.

= RSA agents <agents>

All agents share one sequential-Bayes template: $P^((0))$ is a specified prior
and $P^((t))(dot) = P^((t-1))(dot | "signal at round" t-1)$. Beliefs carry
across rounds; the world state $theta$ is fixed within a run.

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
  ($= psi = "pers"^+$) and `"low"` ($= psi = "pers"^-$), recorded under
  `psi_label_map` in every `run_config.json`. Where this document writes
  $"pers"^+$ the parquet columns say `high`.
]

== Pragmatic listener $L_1$

$L_1$ inverts $S_1$ while maintaining a joint posterior over the world state
and the speaker's hidden parameters:

$ P_(L_1)^((t))(theta, psi, alpha | u^((t))) prop
  P_(L_1)^((t))(theta, psi, alpha)
  sum_(O') P_(S_1)^((t))(u^((t)) | O', psi, alpha) P(O' | theta). $

The prior on $psi$ separates the two listener types:

$ P_(L_1)^((0))(psi | omega = "coop") = bb(1){psi = "inf"}, quad
  P_(L_1)^((0))(psi | omega = "strat") = 1/3 " for each " psi in Psi("strat"). $

Marginals follow by summing the joint over the nuisance coordinates, e.g.
$P_(L_1)^((t))(theta | u) = sum_(psi', alpha') P_(L_1)^((t))(theta, psi', alpha' | u)$.
In the implementation $alpha$ is fixed and known rather than inferred, so the
joint is over $(theta, psi)$ only: a credulous listener carries
$|Theta|$ states and a vigilant one $3 |Theta|$.

== Pragmatic speaker $S_2$ <s2>

$S_2$ is the $S_1$ template with the internal listener swapped from $L_0$ to
$L_1$:

$ P_(S_2)^((t))(u | O, psi, alpha) prop
  "Truth"(u; O) dot "Inf"_(S_2)^((t))(u; O)^(alpha beta) dot
  "PersStr"_(S_2)^((t))(u; psi)^(alpha (1 - beta)), $
$ "Inf"_(S_2)^((t))(u; O) = P_(L_1)^((t))(O | u), quad
  "PersStr"_(S_2)^((t))(u; psi) = cases(
    bb(E)_(L_1)^((t))[theta | u] & "if" psi = "pers"^+,
    1 - bb(E)_(L_1)^((t))[theta | u] quad & "if" psi = "pers"^-,
    1 & "if" psi = "inf".
  ) $

Both terms are one-step-ahead quantities of the *same* internal listener. The
speaker's own belief over $theta$ is carried but kept out of the policy. Past
level 2 the recursion is structurally fixed: every $S_n$ for $n >= 2$ reuses
the $S_2$ machinery with index $n-1$ on its internal listener, and every $L_n$
for $n >= 1$ reuses the $L_1$ machinery with index $n$ on its internal speaker.

When the internal listener is a switching listener (@switching), the two terms
are read through `peek` and `peek_obs`, which return the $theta$-marginal and
the observation posterior the listener *would* hold after hearing $u$ --
detector update and any switch included -- without mutating it. Both terms
therefore score against the same hypothetical listener.

Every agent exposes a nested `version` fingerprint
$(text("own updates"), text("dependency.version"))$; each cached table is
stamped with it and dropped when the agent below moves, so a policy can never
be served from a listener state that has advanced.

= The detection problem <detection>

== Setting

Detection fixes the listener to the *credulous* instance

$ L_1^"inf" := L_1 (omega = "coop"), $

whose prior places all mass on an informative speaker. In code this is the
internal `naive` listener inside `DetectionListener`
(`rsa/detection/listener.py`).

The data-generating speaker has a true goal $psi^star$ and rationality
$alpha$. The listener's internal speaker model is always
$S_1 (dot | dot, "inf", alpha)$. Whenever $psi^star != "inf"$ the listener's
generative model is misspecified, and detection asks whether that is visible in
the utterance stream:

$ H_0 : psi^star = "inf", quad quad H_A : psi^star in {"pers"^+, "pers"^-}. $

The listener is given the correct $alpha$, so this is a test of the goal only.
A second form of misspecification appears in @switchres, where the speaker is
an $S_2$ and the listener's model is still $S_1^"inf"$; there $H_0$ is false in
level as well as goal.

== Round-$t$ belief state

Every score at round $t$ is built from four objects, bundled by `ScoreContext`
in `rsa/detection/scores.py` so that all scores see one consistent snapshot.
Writing $q_t$ for the listener's push-forward onto observations and $k_t$ for
its internal speaker model:

$ q_t (O) = P_(L_1^"inf")^((t))(O) = sum_(theta') P(O | theta') P_(L_1^"inf")^((t))(theta'),
  quad quad
  k_t (u | O) = P_(S_1)^((t))(u | O, "inf", alpha). $

From these come the *prior predictive* over utterances and, by Bayes, the
listener's *posterior over observations* given the utterance actually heard:

$ p_t (u) = sum_O k_t (u | O) q_t (O), quad quad
  q_t (O | u) = frac(k_t (u | O) med q_t (O), p_t (u)). $

The posterior predictive $r_t (v | u) = sum_O k_t (v | O) q_t (O | u)$ is also
available and used by one score. Throughout,
$H(a) = -sum_x a(x) log a(x)$ denotes entropy, and logs are clamped below at
$epsilon = 10^(-12)$ so that $0 log 0 = 0$.

= Detection scores <scores>

Every score is a per-round statistic of the heard utterance $u^((t))$, centered
so that it has *mean zero under $H_0$*. Positive values are evidence against
the informative-speaker null. The scores differ along two axes: which
distribution over $O$ they use, and whether the log is taken inside or outside
the marginalization over $O$.

== Prior-predictive surprisal (`surp2`)

Centered self-information of the heard utterance under the listener's
prediction made _before_ hearing it:

$ s_"surp2"^((t)) = -log p_t (u^((t))) - H(p_t). $

Under $H_0$ the heard utterance is a draw from $p_t$, so the score is the
deviation of a log-loss from its own expectation and
$bb(E)[s_"surp2"^((t))] = 0$ by definition of entropy. Its null variance is the
*varentropy* of the prior predictive:

$ sigma_"surp2"^(2,(t)) = op("Var")_(u ~ p_t)[-log p_t (u)]. $

This score marginalizes over $O$ first, then takes the log, so it sees only the
utterance distribution and not the observation structure behind it.

== Observation-level suspicion (`sus`)

Suspicion reverses that order. Define the per-observation *excess surprise*

$ b_t (O, u) = -log k_t (u | O) - H(k_t (dot | O)), $

which for each fixed $O$ has mean zero under $k_t (dot | O)$. The score
averages it over the listener's posterior about which observation the speaker
saw:

$ s_"sus"^((t)) = sum_O q_t (O | u^((t))) med b_t (O, u^((t))). $

Because the inner quantity is mean-zero for every $O$ separately,
$bb(E)[s_"sus"^((t))] = 0$ under $H_0$ as well. Taking the log inside the
marginalization lets the score register that an utterance is unlikely *given
the observations that would make it worth saying*, rather than merely globally
unlikely.

== Null variance of `sus` <ltv>

The sequential test needs the variance of the score under the listener's own
predictive distribution. With $|cal(U)| = |cal(O)| = 8$ this is an exact finite
sum, computed directly:

$ V_t = sum_(u) p_t (u) med s_"sus" (u)^2, quad
  s_"sus"(u) = sum_O q_t (O | u) med b_t (O, u). $

The square needs no centring because the score is mean-zero under the
listener's own joint $pi(O, u) = q_t (O) k_t (u | O)$: summing
$p_t (u) s_"sus"(u)$ over $u$ collapses to $sum_O q_t (O) (H_O - H_O) = 0$.

Two properties matter. $V_t$ depends only on the listener state, not on the
utterance actually heard -- it is the same number whichever $u$ arrives -- and
it is a probability-weighted sum of squares, so it is non-negative by
construction. Nothing is clipped anywhere.

The law of total variance gives the same quantity a second form,

$ V_t = sum_O q_t (O) med v_t (O) - K_t, quad
  K_t = sum_(u') p_t (u') op("Var")_(O ~ q_t (dot | u'))[b_t (O, u')], $

with $v_t (O)$ the varentropy of row $O$. The two forms are checked against
each other to machine precision at twelve frozen listener states in
`tests/test_sus_variants.py`, together with a Monte-Carlo check that sampling
$u ~ p_t$ reproduces $V_t$.

#block(fill: luma(245), inset: 8pt, radius: 3pt, width: 100%)[
  *Superseded variance.* Earlier versions returned the per-round proxy
  $sigma_"naive"^2 (u^((t))) - K_t$, using the *posterior*-weighted varentropy
  $sum_O q_t (O | u^((t))) v_t (O)$ in place of the first term of the LTV form.
  Its expectation over $u$ is $V_t$, so pooled calibration looked correct, but
  round by round it is a different number that moves with the score it scales:
  on the old null trajectories it correlates with its own numerator at
  $rho = -0.27$ at $alpha = 1.5$ and is non-positive on 35% of rounds. All
  results in this document use the exact $V_t$; `results/full_sweep/` holds the
  pre-fix record and `variance_fix_diff.md` compares the two.
]

== Variant family

The suspicion score generalizes: replace the posterior weighting $q_t (O | u)$
with any column-normalized weighting $W_v (O | u)$, giving
$s_"sus_v" = sum_O W_v (O | u) b_t (O, u)$. Four were implemented:

#table(
  columns: (auto, 1fr, auto),
  align: (left, left, left),
  stroke: tstroke,
  table.header([*Variant*], [*Unnormalized weight $W_v (O | u) prop$*], [*Status*]),
  [`sus_1`], [$k_t (u | O) med q_t (O)$ --- the true posterior], [*adopted*],
  [`sus_3`], [$"Truth"(u; O) med q_t (O)$ --- truth-restricted $L_1$ prior], [retired],
  [`sus_4`], [$"Truth"(u; O) med L_0^((t))(O)$ --- truth-restricted $L_0$ prior], [retired],
  [`sus_4b`], [$"Truth"(u; O) med L_0^((t))(O) \/ T(O)$, $T(O) = |{u : "Truth"(u;O)}|$], [retired],
)

The exact variance of @ltv rests on the score being mean-zero under the true
null joint $q_t (O) k_t (u | O)$, which requires the implied joint
$p_"pred,v" (u) dot W_v (O | u)$ to equal it. That is an identity for $W_1$,
which _is_ the real posterior. For variants 3, 4 and 4b the weighting strips
out the $k_t (u | O)$ factor, so the joint is fictitious, the score is not
mean-zero, and no analogue of $V_t$ exists. `make_sus_variant` raises
`NotImplementedError` for them.

#block(fill: luma(247), inset: 8pt, radius: 3pt)[
  *On `surp1`.* A third score, the posterior-predictive surprisal
  $s_"surp1"^((t)) = -log r_t (u^((t)) | u^((t))) - H(r_t (dot | u^((t))))$, is
  implemented and recorded in the raw parquet data. It uses the heard utterance
  twice -- once to form the posterior, once to score itself against it -- and
  its variance formula runs about $2 times$ too large (empirical/theoretical
  ratio $approx 0.46$). It is collected but excluded from every figure and
  table here.
]

= Sequential test and switching listener <test>

== The test

Given a per-round score $s^((i))$ with null variance $sigma^(2,(i))$, the test
in `rsa/detection/sequential_test.py` accumulates a running mean and a running
average variance,

$ "Sus"^((t)) = 1/t sum_(i=1)^t s^((i)), quad quad
  overline(sigma)^(2,(t)) = 1/t sum_(i=1)^t sigma^(2,(i)), $

and fires the first time the running mean exceeds a $z$-scaled boundary:

$ "Sus"^((t)) > c dot overline(sigma)^((t)) / sqrt(t). $

The first crossing time is $tau$. Setting $c = +infinity$ disables crossing and
makes the test a pure observer, which is how the full sweep is run so that any
stopping rule can be scored offline. Writing
$S_t = sum_(i <= t) s^((i)) = t dot "Sus"^((t))$ and
$V_t^"cum" = sum_(i <= t) sigma^(2,(i)) = t dot overline(sigma)^(2,(t))$, the
rule is equivalently $S_t > c sqrt(V_t^"cum")$, so every alternative boundary
in @stopping is a function $B(t, V_t^"cum")$ compared against the same
accumulated matrix.

== Switching listener <switching>

`DetectionListener` can let a crossing drive a switch from the credulous
listener to a full vigilant $L_1$. The tests score $u_tau$ against the
credulous belief as it stands; if one crosses, the switch happens before any
belief update and $u_tau$ is then absorbed exactly once, by the vigilant
listener. The credulous listener freezes at $tau - 1$. Two switch types:

#table(
  columns: (auto, 1fr),
  align: (left, left),
  stroke: tstroke,
  table.header([*`switch_type`*], [*Belief the vigilant $L_1$ starts from at $tau$*]),
  [`"hard"`], [*Retrospective.* A vigilant $L_1$ that has listened since round 1: the listener keeps a shadow vigilant $L_1$ updated in parallel with the credulous one, and at $tau$ the shadow becomes the active listener. Identical to an always-vigilant $L_1$ on the same stream, with no replay.],
  [`"soft"`], [Inherits the credulous $theta$-marginal from *before* $u_tau$, spread uniformly over $psi$ --- the pre-$tau$ credulous belief, direction-blind at $tau$ --- then applies $u_tau$ vigilantly. At $tau = 1$ it coincides with `"hard"`.],
)

Every sub-listener is updated from the same per-round snapshot of the speaker's
tables $P_(S_1)^((i))(u | O, psi)$, which `DetectionListener.update` also
records into `table_history`; `Listener1.update_with_tables` updates from an
explicit table triple instead of consulting the speaker, which makes any
listener replayable offline on a stored stream (`rsa/detection/replay.py`).

== What the null should look like

Under $H_0$ the generative model matches the listener's internal model, giving
three checkable predictions:
$bb(E)[s^((t))] = 0$ so $"Sus"^((t)) -> 0$;
$op("Var")[s^((t))] \/ bb(E)[sigma^(2,(t))] approx 1$; and
$op("Var")["Sus"^((t))] prop 1\/t$. Under $H_A$ the score acquires a strictly
positive mean, so $"Sus"^((t))$ drifts upward. @sweepdiag checks all three.

= Experiment inventory <inventory>

Five datasets are current. All use $n = 1$, $m = 7$ and 150 rounds per
simulation.

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  align: (left, left, center, center, center, right),
  stroke: tstroke,
  table.header([*Dataset*], [*Speaker / audience*], [*Cells*], [*Sims*], [*Rows*], [*Wall*]),
  [`full_sweep_v2`], [$S_1$ / credulous $L_1$, detector passive], [297], [200], [8.9 M], [8.4 min],
  [`switching_A`], [$S_1$ / switching $L_1$ (offline)], [216], [100], [32.4 M], [32 min],
  [`switching_B`], [$S_2$ modelling the switching $L_1$ (feedback)], [2160], [100], [32.4 M], [459 min],
  [`switching_Fvig`], [$S_2$ modelling a vigilant $L_1$ / vigilant $L_1$], [216], [100], [32.4 M], [36 min],
  [`switching_Fcred`], [$S_2$ modelling a credulous $L_1$ / credulous $L_1$], [216], [100], [32.4 M], [46 min],
)

The grids are:

#table(
  columns: (auto, 1fr),
  align: (left, left),
  stroke: tstroke,
  table.header([*Dataset*], [*Grid*]),
  [`full_sweep_v2`],
  [$theta^star in {0.1, dots.c, 0.9}$ (9) $times$ $psi^star in {"inf", "pers"^+, "pers"^-}$ (3) $times$ $alpha in {1, 1.5, 2, 2.5, 3, 4, 5, 7, 10, 15, 20}$ (11). Listener $theta$-grid has 11 points including 0 and 1. Both scores recorded; test passive ($c = infinity$); full $L_1$ and $L_0$ posteriors stored every round.],
  [switching A/B/Fvig/Fcred],
  [$theta^star$ (9) $times$ $psi^star$ (3) $times$ $alpha in {1, 1.5, 2, 3, 4, 5, 7, 10}$ (8) $times$ $c in {2, 3, 3.5, 4, 5}$ (5) $times$ switch $in$ {hard, soft} (2). `sus_1` only. Every round records the credulous, always-vigilant and switching listeners.],
)

In the offline studies (A, Fvig, Fcred) the speaker cannot see the listener, so
one utterance stream per simulation serves all ten $(c, "switch")$ conditions.
In the feedback study B the $S_2$ speaker's internal listener is a private
replica of the actual switching listener, built with identical parameters and
fed the same public utterances; the two are asserted equal every round
(observed maximum divergence $0.0$ across all 216{,}000 runs), so each
$(c, "switch")$ condition needs its own simulations.

All four switching studies share `seed_base`, and the world's observation RNG
is seeded by simulation index alone, so simulation $i$ sees the *same*
observation stream in every cell of every study at a given $theta^star$. Every
between-condition and between-study comparison below is therefore paired.

#block(fill: luma(247), inset: 8pt, radius: 3pt)[
  *Aggregation.* The raw trajectories total about 5 GB. Two scripts,
  `experiments/full_sweep/aggregate.py` and
  `experiments/switching/aggregate.py`, rebuild small per-round, per-run and
  panel tables that every figure reads. Both carry self-checks that reproduce
  the previously published tables exactly. Paired differences between listeners
  are formed per simulation *before* averaging, so their confidence intervals
  are valid.
]

= Results --- full sweep, $S_1$ speaker <sweepres>

Source: `notebooks/sweep_analysis.ipynb`. The speaker is an $S_1$, the listener
a credulous $L_1$, and the detector is a passive observer recording both
scores.

== Diagnostics <sweepdiag>

#sqfig("D1_world_check.png")[
  *Observation model.* Empirical distribution of the number of effective
  sessions out of 7 (solid, from the simulated streams at $alpha = 3$, honest
  speaker) against the $"Binomial"(7, theta^star)$ probability mass function
  (dashed). One colour per $theta^star$.
]

#sqfig("D2_null_convergence.png")[
  *Null convergence of the running mean.* Mean $"Sus"^((t))$ against round
  under an honest $S_1$, pooled over $theta^star$, one line per $alpha$. Left:
  `surp2`. Right: `sus_1`. The null prediction is convergence to 0.
]

#sqfig("D2b_null_convergence_theta.png")[
  *Null convergence per $theta^star$.* The same quantity at $alpha = 3$ with
  one line per $theta^star$ and a 95% confidence interval of the mean over 200
  simulations.
]

#sqfig("D3_variance_calibration.png")[
  *Per-round variance calibration.* Ratio of the empirical variance of the
  per-round score to the analytic $sigma^(2,(t))$, under the honest speaker,
  one line per $alpha$. The null prediction is 1. Pooled values: $0.999$
  (`surp2`), $1.001$ (`sus_1`).
]

#sqfig("D3b_running_variance.png")[
  *Running-mean variance calibration.* Ratio
  $op("Var")["Sus"^((t))] \/ (overline(sigma)^(2,(t)) \/ t)$ against round. The
  null prediction is 1. Pooled values: $0.963$ (`surp2`), $1.030$ (`sus_1`).
]

#sqfig("D4_t_var_scaling.png")[
  *Variance scaling.* $t dot op("Var")["Sus"^((t))]$ against round on a log
  $x$-axis, one line per $alpha$. A flat line indicates the $1\/t$ law holds.
]

Mean $"Sus"(t = 150)$ under the null is $-0.00066$ for `surp2` and $+0.00003$
for `sus_1`.

== Score behaviour

#sqfig("S1_separation.png")[
  *Separation at the horizon.* Distribution of $"Sus"(150)$ at $alpha = 3$,
  pooled over $theta^star$, for the honest, $"pers"^+$ and $"pers"^-$ speakers.
  Left: `surp2`. Right: `sus_1`.
]

#sqfig("S2_dprime.png")[
  *Separation over the grid.* $d' = (bb(E)["Sus"(150) | H_A] - bb(E)["Sus"(150) | H_0]) \/ op("sd")("Sus"(150) | H_0)$
  at each $(theta^star, alpha)$ cell. Median $d'$: 6.24 (`surp2`), 9.34
  (`sus_1`).
]

#sqfig("S3_score_correlation.png")[
  *Per-round score correlation.* Hex-binned joint distribution of the two
  per-round scores over all rounds, simulations and $theta^star$ at
  $alpha = 3$, split by true speaker goal. Pearson $rho$ in each panel title.
]

== Detection with the fixed-$z$ rule

Rates are computed from each run's $tau$. The table pools over $theta^star$ and
over $alpha >= 1.5$; the $alpha = 1$ cells are simulated and stored but
excluded, since at $alpha = 1$ the pragmatic speaker is nearly literal.

#figure(
  table(
    columns: 7,
    align: (left, center, center, center, center, center, center),
    stroke: tstroke,
    table.header([*Score*], [*$c$*], [*FPR*], [*TPR $"pers"^+$*], [*TPR $"pers"^-$*], [*TPR $minus$ FPR*], [*median $tau$*]),
    [`surp2`], [2.0], [0.234], [0.743], [0.740], [0.508], [3],
    [`surp2`], [2.5], [0.121], [0.699], [0.700], [0.578], [4],
    [`surp2`], [3.0], [0.069], [0.667], [0.669], [0.598], [4],
    [`surp2`], [3.5], [0.034], [0.635], [0.636], [0.601], [5],
    [`surp2`], [4.0], [0.019], [0.606], [0.608], [0.588], [5],
    [`surp2`], [5.0], [0.005], [0.556], [0.559], [0.553], [6],
    [`sus_1`], [2.0], [0.219], [0.996], [0.995], [0.777], [2],
    [`sus_1`], [2.5], [0.128], [0.989], [0.988], [0.860], [3],
    [`sus_1`], [3.0], [0.065], [0.974], [0.970], [0.907], [4],
    [`sus_1`], [3.5], [0.031], [0.947], [0.947], [0.916], [5],
    [`sus_1`], [4.0], [0.019], [0.910], [0.911], [0.892], [6],
    [`sus_1`], [5.0], [0.011], [0.830], [0.826], [0.817], [7],
  ),
  caption: [*Fixed-$z$ operating characteristics on the full sweep*, pooled
    over $theta^star$ and $alpha >= 1.5$. FPR is the crossing rate under an
    honest $S_1$; TPR under each persuasive goal; median $tau$ over detected
    persuasive runs.],
)

Including $alpha = 1$ lowers `sus_1` power at $c = 3.5$ from 0.947 to 0.912 and
leaves FPR at 0.028.

#sqfig("S4_rates_vs_alpha.png")[
  *Rates against $alpha$.* False-alarm rate (solid) and power (dashed) by
  round 150, pooled over $theta^star$, one line per $c$. Left: `surp2`. Right:
  `sus_1`.
]

#sqfig("S5_roc.png")[
  *Operating characteristics.* False-alarm rate against power at round 150;
  each line is one $alpha$ traced through $c in {2, 2.5, 3, 3.5, 4, 5}$.
]

#sqfig("S6_fa_vs_theta.png")[
  *False alarms against $theta^star$.* Crossing rate under the honest $S_1$ at
  $c = 3.5$, one line per $alpha$.
]

#sqfig("S7_power_heatmaps.png")[
  *Power over the grid.* Detection rate of a persuasive $S_1$ at $c = 3.5$ for
  every $(theta^star, alpha)$ cell.
]

#sqfig("S8_pers_asymmetry.png")[
  *$"pers"^+$ against $"pers"^-$.* Detection rate against $theta^star$ at
  $c = 3.5$ for the two persuasive goals (solid and dashed) at five values of
  $alpha$.
]

At $c = 3.5$ the `sus_1` false-alarm rate by $alpha$ is
$0.002, 0.002, 0.002, 0.003, 0.012, 0.018, 0.027, 0.066, 0.065, 0.051, 0.063$
for $alpha = 1, 1.5, 2, 2.5, 3, 4, 5, 7, 10, 15, 20$, and power rises from
0.57 at $alpha = 1$ to 1.00 from $alpha = 10$ upward.

#sqfig("S9_median_tau.png")[
  *Detection latency.* Median $tau$ over detected persuasive runs for every
  $(alpha, c)$ cell, pooled over $theta^star$.
]

#sqfig("S9b_latency_box.png")[
  *Latency distributions.* Distribution of $tau$ over detected persuasive runs
  at $c = 3.5$, one box per $alpha$ (whiskers exclude outliers).
]

#sqfig("S10_horizon.png")[
  *Rates against horizon.* False-alarm rate (solid) and power (dashed) by
  round $t in {10, 25, 50, 100, 150}$ at $alpha = 3$, one line per $c$.
]

== Stopping rules <stopping>

Every rule is a boundary $B(t, V_t^"cum")$ compared against $S_t$, scored
offline on the stored panels. $alpha = 1$ is excluded. The empirical rule is
calibrated on half the null runs and scored on the held-out half.

#table(
  columns: (auto, 1fr, auto),
  align: (left, left, left),
  stroke: tstroke,
  table.header([*Rule*], [*Boundary*], [*Property*]),
  [`fixed_z(c)`], [$c sqrt(V_t^"cum")$], [the production rule],
  [`warmup(c, t_0)`], [$c sqrt(V_t^"cum")$ for $t >= t_0$, else $infinity$], [gates the early rounds],
  [`lil(c)`], [$c sqrt(2 V_t^"cum" log log t)$], [law of the iterated logarithm],
  [`robbins(delta, rho)`], [$sqrt((V_t^"cum" + rho) log frac(V_t^"cum" + rho, rho delta^2))$], [anytime-valid; $P("ever cross") <= delta$],
  [`empirical(delta)`], [$c_delta sqrt(V_t^"cum")$, $c_delta$ a null quantile], [horizon-committed calibration],
)

#figure(
  table(
    columns: 8,
    align: (left, center, center, center, center, center, center, center),
    stroke: tstroke,
    table.header(
      table.cell(colspan: 2)[], table.cell(colspan: 3, align: center)[*`surp2`*], table.cell(colspan: 3, align: center)[*`sus_1`*],
    ),
    [*Rule*], [*setting*], [FPR], [power], [med $tau$], [FPR], [power], [med $tau$],
    [`fixed_z`], [$c = 2$], [0.234], [0.741], [3], [0.219], [0.995], [2],
    [`fixed_z`], [$c = 3.5$], [0.034], [0.635], [5], [0.031], [0.947], [5],
    [`warmup`], [$c = 3.5, t_0 = 10$], [0.015], [0.629], [10], [0.016], [0.946], [10],
    [`lil`], [$c = 2$], [0.331], [0.728], [1], [0.268], [0.973], [1],
    [`robbins`], [$delta = 0.05$], [0.002], [0.591], [17], [0.002], [0.820], [17],
    [`empirical`], [$delta = 0.05$], [0.042], [0.646], [4], [0.049], [0.959], [4],
  ),
  caption: [*Stopping rules at their nominal settings.* Rates pooled over
    $theta^star$ and $alpha >= 1.5$; the `empirical` row is scored on held-out
    null runs with $c_delta = 3.34$ (`surp2`) and $3.27$ (`sus_1`).],
)

#figure(
  table(
    columns: 6,
    align: (left, left, center, center, center, center),
    stroke: tstroke,
    table.header([*Score*], [*Rule*], [*param*], [*FPR (held-out)*], [*power*], [*$alpha$ spread*]),
    [`surp2`], [`fixed_z`], [3.341], [0.042], [0.646], [0.041],
    [`surp2`], [`warm-up` $t_0 = 10$], [2.868], [0.051], [0.665], [0.026],
    [`surp2`], [`lil`], [4.912], [0.046], [0.516], [0.047],
    [`surp2`], [`robbins`], [0.409], [0.053], [0.669], [0.058],
    [`sus_1`], [`fixed_z`], [3.275], [0.049], [0.959], [0.106],
    [`sus_1`], [`warm-up` $t_0 = 10$], [2.790], [0.052], [0.979], [0.058],
    [`sus_1`], [`lil`], [5.916], [0.046], [0.647], [0.148],
    [`sus_1`], [`robbins`], [0.421], [0.051], [0.930], [0.056],
  ),
  caption: [*Rules calibrated to the same achieved 5% false-alarm rate.* Each
    parameter is bisected on a training half of the null runs; FPR, power and
    the across-$alpha$ spread of the false-alarm rate are measured on the
    held-out half. All eight reach the target.],
)

#sqfig("R2_operating_curves.png")[
  *Operating curves.* False-alarm rate (log axis) against power as each rule's
  parameter is swept; Robbins and the held-out empirical rule are single
  points at $delta in {0.01, 0.05, 0.10}$. Dotted line marks the 5% target.
]

#sqfig("R3_fpr_vs_horizon.png")[
  *False alarms against horizon.* Cumulative fraction of null runs that have
  crossed by round $t$, one line per rule at its nominal setting.
]

#sqfig("R4_fpr_per_alpha.png")[
  *False alarms per $alpha$.* Null crossing rate by $alpha$ for each rule at
  its nominal setting; dashed line is the 5% target.
]

#sqfig("R5_latency.png")[
  *Latency by rule.* Distribution of $tau$ over detected persuasive runs, one
  box per rule at its nominal setting.
]

#sqfig("R6_matched_fpr.png")[
  *Power and uniformity at matched false-alarm rate.* Left: power of each rule
  once calibrated to an achieved 5% false-alarm rate. Right: the spread
  (max $minus$ min) of that rule's false-alarm rate across $alpha$.
]

== Belief dynamics of the credulous listener

The sweep stores the full posterior every round, so the following quantities
are available beyond $bb(E)[theta]$: the posterior standard deviation, the
entropy, the mass on the true grid point, and whether $theta^star$ lies inside
the listener's own $plus.minus 2$ sd interval. The literal listener $L_0$ is
stored alongside.

#sqfig("B1_bias_traj.png")[
  *Signed bias by round.* $bb(E)[theta] - theta^star$ of the credulous $L_1$
  against round (log axis) at $alpha = 3$, one line per $theta^star$ with a 95%
  confidence band, split by true speaker goal.
]

#sqfig("B2_abias_heat.png")[
  *Time-averaged $|"bias"|$.* Mean $|bb(E)[theta] - theta^star|$ over the 150
  rounds for every $(theta^star, alpha)$ cell, split by speaker goal.
]

#sqfig("B3_L1_vs_L0.png")[
  *Pragmatic against literal listener.* Difference in mean $|"bias"|$ between
  $L_1$ and $L_0$ hearing identical utterances, by round, one line per $alpha$.
  Negative means the pragmatic listener is more accurate.
]

#sqfig("B4_entropy.png")[
  *Posterior uncertainty.* Left: mean posterior entropy of the credulous $L_1$
  by round. Right: mean posterior mass on the true grid point. Colour is the
  speaker goal, opacity is $alpha$.
]

#sqfig("B5_coverage.png")[
  *Calibration.* Fraction of runs in which $theta^star$ lies within
  $bb(E)[theta] plus.minus 2 "sd"$ of the listener's own posterior, by round.
  Left: $L_1$. Right: $L_0$. Dashed line at 0.95.
]

#sqfig("B6_run_spread.png")[
  *Run-level spread.* Distribution over 200 simulations of the credulous
  listener's signed bias at rounds 25, 50, 100 and 150, at
  $theta^star = 0.3$, $alpha = 3$.
]

= Results --- switching studies <switchres>

Source: `notebooks/switching_analysis.ipynb`. Study A has an $S_1$ speaker;
studies B, Fvig and Fcred have $S_2$ speakers. In all four the detector uses
`sus_1` and its crossing drives a switch.

== Diagnostics

All four studies report `status=complete` with every cell present and 100
simulations per cell. Simulation $i$ draws the identical observation stream in
A, B, Fvig and Fcred at a given $theta^star$, and is invariant to $psi^star$
and $alpha$. In study A the hard and soft conditions share $tau$ exactly (the
speaker cannot see the listener); in study B they differ in 64.3% of runs. The
private replica inside the $S_2$ of study B never diverged from the actual
listener (maximum $0.0$).

#swfig("D2_null_calibration.png")[
  *Detector null under a correctly specified speaker (study A).* Left: mean
  per-round `sus_1` score against round under an honest $S_1$, one line per
  $alpha$; the null prediction is 0. Right: ratio of empirical to analytic
  variance; the null prediction is 1. Pooled ratio $1.005$.
]

== Detection against a level-2 speaker

Study B pairs an $S_2$ speaker with the switching $L_1$ whose internal model is
$S_1^"inf"$, so an "honest-speaker alarm" records a mismatch in *level* rather
than in goal.

#figure(
  table(
    columns: 6,
    align: (left, center, center, center, center, center),
    stroke: tstroke,
    table.header([*$alpha$*], [*$c = 2$*], [*$c = 3$*], [*$c = 3.5$*], [*$c = 4$*], [*$c = 5$*]),
    [1.0], [0.166], [0.009], [0.000], [0.000], [0.000],
    [1.5], [0.247], [0.034], [0.009], [0.002], [0.000],
    [2.0], [0.418], [0.107], [0.051], [0.017], [0.003],
    [3.0], [0.801], [0.598], [0.453], [0.317], [0.122],
    [4.0], [0.909], [0.820], [0.769], [0.706], [0.591],
    [5.0], [0.956], [0.912], [0.871], [0.839], [0.736],
    [7.0], [0.972], [0.962], [0.951], [0.927], [0.894],
    [10.0], [0.917], [0.911], [0.901], [0.905], [0.899],
  ),
  caption: [*Study B, honest $S_2$: fraction of runs that switched by round
    150*, pooled over $theta^star$ and switch type (1800 runs per cell).],
)

#figure(
  table(
    columns: 6,
    align: (left, center, center, center, center, center),
    stroke: tstroke,
    table.header([*$alpha$*], [*$c = 2$*], [*$c = 3$*], [*$c = 3.5$*], [*$c = 4$*], [*$c = 5$*]),
    [1.0], [0.969], [0.769], [0.566], [0.341], [0.067],
    [1.5], [0.991], [0.904], [0.814], [0.701], [0.398],
    [2.0], [0.987], [0.918], [0.851], [0.776], [0.588],
    [3.0], [0.984], [0.941], [0.899], [0.864], [0.734],
    [4.0], [0.998], [0.981], [0.957], [0.933], [0.868],
    [5.0], [0.999], [0.993], [0.988], [0.973], [0.934],
    [7.0], [1.000], [1.000], [0.999], [0.999], [0.992],
    [10.0], [1.000], [1.000], [1.000], [1.000], [0.999],
  ),
  caption: [*Study B, $"pers"^+$ $S_2$: fraction of runs that switched by round
    150*, pooled over $theta^star$ and switch type. The $"pers"^-$ table is
    within 0.02 of this one in every cell.],
)

Median $tau$ for the $"pers"^+$ $S_2$ at $c = 3.5$ falls from 99 rounds at
$alpha = 1$ to 1 round at $alpha >= 5$.

#swfig("F1_roc.png")[
  *Operating characteristics by speaker level.* False-alarm rate against power
  at round 150; each line is one $alpha$ traced through $c in {2, 3, 3.5, 4, 5}$
  (labels are $c$). Left: study B ($S_2$). Right: study A ($S_1$).
]

#swfig("F2_rates_vs_alpha.png")[
  *Rates against $alpha$.* False-alarm rate (solid) and power (dashed) by
  round 150, one line per $c$. Left: study B. Right: study A.
]

#swfig("F3_fa_vs_theta.png")[
  *Honest-$S_2$ false alarms against $theta^star$* (study B), one panel per
  boundary $c$, one line per $alpha$.
]

#swfig("F4_rate_heatmaps.png")[
  *Rate surfaces at $c = 3.5$* (study B). Left: false alarm under the honest
  $S_2$. Middle: power against the persuasive $S_2$. Right: their difference.
]

#swfig("F5_pers_asymmetry.png")[
  *$"pers"^+$ against $"pers"^-$ in study B*, detection rate against
  $theta^star$ at $c = 3.5$, for five values of $alpha$.
]

#swfig("F6_survival.png")[
  *Survival of the credulous state.* $P(tau > t)$ against round at
  $c = 3.5$, hard switch, for $theta^star in {0.1, 0.5, 0.9}$; solid is the
  honest $S_2$, dashed the $"pers"^+$ $S_2$.
]

#swfig("F7_median_tau.png")[
  *Time to alarm in study B.* Median $tau$ over runs that switched, for every
  $(alpha, c)$ cell, pooled over $theta^star$. Left: $"pers"^+$ speaker.
  Right: honest speaker.
]

#swfig("F8_horizon.png")[
  *Rates against horizon in study B.* False-alarm rate (solid) and power
  (dashed) by round $t in {25, 50, 100, 150}$ at $alpha = 3$, one line per $c$.
]

== Where the level mismatch comes from

#swfig("F9_score_by_study.png")[
  *Per-round score under honest speakers, by study.* Mean `sus_1` score against
  round for the honest speaker of each of the four switching studies at three
  values of $alpha$, at $theta^star = 0.3$ (left) and $0.5$ (right). Study A is
  the $S_1$ null.
]

#swfig("F10_sus_vs_boundary.png")[
  *Running statistic against its own boundary.* $"Sus"(t)$ (solid) and the
  boundary $c overline(sigma)^((t)) \/ sqrt(t)$ (dotted) against round on a log
  axis, at $theta^star = 0.3$, $c = 3.5$, one line per $alpha$. Left: honest
  $S_2$. Right: $"pers"^+$ $S_2$.
]

#swfig("F11_policy_gap.png")[
  *Empirical utterance policies of two honest speakers.* $P(u | O)$ estimated
  from the simulated streams at $theta^star = 0.3$, $alpha = 3$. Left:
  informative $S_1$ (study A). Middle: informative $S_2$ (study Fcred). Right:
  their difference.
]

#swfig("F12_expected_score.png")[
  *Expected per-round score of an honest speaker under the $S_1$ null.*
  Computed directly from the model at round 1 with a uniform listener prior,
  not from the simulations: $bb(E)[s]$ when the utterance is drawn from
  $S_1^"inf"$ (the null) and when it is drawn from $S_2^"inf"$.
]

#swfig("F13_utterance_scores.png")[
  *Score contribution by utterance.* Average policy mass each honest speaker
  places on each utterance (bars, left axis) against the score $s(u)$ that
  utterance earns under the $S_1$ null (diamonds, right axis), at $alpha = 3$.
]

== Belief accuracy

#swfig("F14_bias_traj.png")[
  *Signed bias of the three listeners* (credulous, always-vigilant, switching)
  against round on a log axis in study B at $alpha = 3$, $c = 3.5$, hard
  switch; one line per $theta^star$ (all nine). Left: $"pers"^+$ $S_2$. Right:
  honest $S_2$.
]

#swfig("F15_bias_theta0.1.png")[
  *Signed bias at $theta^star = 0.1$* for the three listeners, by round, with
  95% confidence bands, split by speaker goal. Dotted vertical line marks the
  median $tau$. Study B, $alpha = 3$, $c = 3.5$, hard switch.
]

#swfig("F15_bias_theta0.5.png")[
  *Signed bias at $theta^star = 0.5$*, otherwise as the previous figure.
]

#swfig("F15_bias_theta0.9.png")[
  *Signed bias at $theta^star = 0.9$*, otherwise as the previous figure.
]

#swfig("F16_paired_vs_cred.png")[
  *Paired difference against the credulous listener.*
  $|"bias"|_"switch" - |"bias"|_"cred"$ computed per simulation and then
  averaged, by round, with 95% confidence bands; one line per $theta^star$,
  split by speaker goal.
]

#swfig("F17_paired_vs_vig.png")[
  *Paired difference against the always-vigilant listener.*
  $|"bias"|_"switch" - |"bias"|_"vig"$, otherwise as the previous figure.
]

By round 150 every listener has converged under any speaker, so terminal bias
is near zero; the following figures use the time-averaged $|"bias"|$ over the
150 rounds.

#figure(
  table(
    columns: 7,
    align: (left, center, center, center, center, center, center),
    stroke: tstroke,
    table.header(
      table.cell(rowspan: 2)[*$alpha$*],
      table.cell(colspan: 3, align: center)[*honest $S_2$*],
      table.cell(colspan: 3, align: center)[*$"pers"^+$ $S_2$*],
      [cred], [vig], [switch], [cred], [vig], [switch],
    ),
    [1.0], [0.018], [0.019], [0.018], [0.035], [0.029], [0.033],
    [1.5], [0.017], [0.019], [0.017], [0.042], [0.030], [0.035],
    [2.0], [0.016], [0.019], [0.016], [0.050], [0.031], [0.035],
    [3.0], [0.015], [0.018], [0.016], [0.061], [0.033], [0.035],
    [4.0], [0.014], [0.017], [0.015], [0.072], [0.039], [0.040],
    [5.0], [0.013], [0.016], [0.015], [0.082], [0.044], [0.045],
    [7.0], [0.012], [0.015], [0.014], [0.099], [0.061], [0.062],
    [10.0], [0.012], [0.014], [0.013], [0.101], [0.093], [0.094],
  ),
  caption: [*Time-averaged $|bb(E)[theta] - theta^star|$ over 150 rounds*,
    study B at $c = 3.5$, hard switch, pooled over $theta^star$, for the
    credulous, always-vigilant and switching listeners.],
)

#swfig("F18_vig_vs_cred_B.png")[
  *Vigilant minus credulous, study B.* Difference in time-averaged
  $|"bias"|$ between the always-vigilant and the credulous listener, for every
  $(theta^star, alpha)$ cell at $c = 3.5$, split by speaker goal. Negative
  (blue) means the vigilant listener is more accurate.
]

#swfig("F18_vig_vs_cred_A.png")[
  *Vigilant minus credulous, study A* ($S_1$ speaker), otherwise as the
  previous figure.
]

#swfig("F19_vig_flip.png")[
  *The same difference as curves.* Time-averaged $|"bias"|$, vigilant minus
  credulous, against $alpha$ at three values of $theta^star$; solid is the
  level-2 speaker (study B), dashed the level-1 speaker (study A).
]

#swfig("F20_regret_high.png")[
  *Regret against a persuasive $S_2$.* Time-averaged $|"bias"|$ of the
  switching listener minus the better of the two fixed strategies, for every
  $(alpha, c)$ cell, at three values of $theta^star$.
]

#swfig("F20_regret_inf.png")[
  *Regret against an honest $S_2$*, otherwise as the previous figure.
]

#swfig("F21_cost_benefit.png")[
  *Cost--benefit plane.* Regret against an honest $S_2$ ($x$) against regret
  against a persuasive $S_2$ ($y$), both time-averaged, for every $(alpha, c)$
  cell; point labels are $c$.
]

#swfig("F21_coverage.png")[
  *Calibration of the three listeners.* Fraction of runs in which $theta^star$
  lies within $bb(E)[theta] plus.minus 2 "sd"$ of the listener's own posterior,
  by round, at $alpha = 3$, $c = 3.5$, pooled over $theta^star$.
]

#swfig("F22_run_spread.png")[
  *Run-level spread.* Distribution over 100 simulations of each listener's
  signed bias at rounds 25, 50, 100 and 150, at $theta^star = 0.3$,
  $alpha = 3$, $c = 3.5$.
]

== Hard against soft switching

#swfig("F23_aligned_recovery.png")[
  *Recovery after the alarm.* Signed bias against $k = t - tau$ (the switch is
  at $k = 0$) for the hard and soft switching listeners, with the
  always-vigilant and always-credulous listeners for reference, at
  $theta^star = 0.3$, $alpha = 3$, $c = 3.5$.
]

#swfig("F24_psi_identification.png")[
  *Speaker-type identification after the switch.* $P(psi = psi^star)$ held by
  the switched listener against $k = t - tau$, for the hard and soft switches,
  with the always-vigilant listener for reference. Dotted line is the uniform
  prior $1\/3$.
]

#swfig("F25_hard_minus_soft.png")[
  *Honest-speaker false alarms, hard minus soft*, at $c = 3.5$ for every
  $(theta^star, alpha)$ cell. Left: study B, where the speaker can see which
  switch type it faces. Right: study A, where it cannot; the maximum absolute
  difference there is $0.0$.
]

At $c = 3.5$ the honest-$S_2$ alarm rate in study B is 0.994 (hard) against
0.807 (soft) at $alpha = 10$, and within 0.02 of itself for $alpha <= 5$.

== Speaker behaviour near the boundary

Study B records, every pre-switch round, the margin $"Sus"(t) - $ boundary at
choice time (always negative before a crossing) and whether the speaker took
the utterance a purely informative or purely persuasive $S_2$ would have
chosen.

#swfig("F26_evasion_margin.png")[
  *Choices against distance to the boundary.* Probability that the persuasive
  $S_2$ took the informative reference utterance (left) and the persuasive
  reference utterance (right), by margin bin, at $c = 3.5$, hard switch; one
  line per $alpha$. The rightmost bin is the round before a crossing.
]

#swfig("F27_evasion_rounds.png")[
  *The same quantities against round.* Left: probability of taking the
  informative utterance. Right: policy mass on the chosen utterance (solid)
  and on a pure persuader's pick (dashed). Pre-switch rounds only.
]

#swfig("F29_B_minus_A.png")[
  *Alarm rate, study B minus study A*, paired by simulation on identical
  observation streams, against $alpha$, with 95% confidence intervals; one
  line per true speaker goal.
]

== Fang dyads at level 2

Studies Fvig and Fcred pair an $S_2$ with the audience it actually models:
Fcred is the cooperative dyad ($S_2 <-> $ credulous $L_1$), Fvig the strategic
one ($S_2 <-> $ vigilant $L_1$). Study A on the same streams is the level-1
reference.

#swfig("F30_fcred_inf.png")[
  *Cooperative dyad learning curves.* $bb(E)[theta]$ of the credulous $L_1$
  against round when hearing an honest $S_2$ that models it (study Fcred), at
  $alpha = 3$; one line per $theta^star$ with dotted lines at the true values.
]

#swfig("F31_fvig_high.png")[
  *Strategic dyad learning curves.* $bb(E)[theta]$ of the vigilant $L_1$
  against round when hearing a $"pers"^+$ $S_2$ that models it (study Fvig), at
  $alpha = 3$.
]

#swfig("F32_fang_contrast.png")[
  *Credulous against vigilant.* Mean $|"bias"|$ by round, pooled over
  $theta^star$, at $alpha = 3$; solid lines are the matched level-2 dyads
  (Fcred, Fvig), dashed are the level-1 speaker of study A.
]

#swfig("F33_psi_learning_fvig.png")[
  *Speaker-goal identification.* $P(psi = psi^star)$ held by the vigilant $L_1$
  against round when hearing a level-2 speaker (study Fvig) at $alpha = 3$; one
  line per $theta^star$, dotted line at the uniform prior.
]

#swfig("F34_speaker_policies.png")[
  *Utterance choice by speaker type.* Empirical $P(u | O)$ at three
  observations (0, 2 and 4 of 7 sessions effective) for the informative $S_1$
  and the informative, $"pers"^+$ and $"pers"^-$ $S_2$, at $theta^star = 0.3$,
  $alpha = 3$.
]

At $alpha = 3$ the time-averaged $|"bias"|$ is, for the credulous listener,
0.0145 (honest) and 0.0586 ($"pers"^+$) in Fcred; for the vigilant listener,
0.0154 (honest) and 0.0354 ($"pers"^+$) in Fvig; and for study A's level-1
speaker, 0.0130/0.0624 (credulous) and 0.0133/0.0342 (vigilant).

= Results --- sweep and switching combined <combined>

Source: `notebooks/sweep_analysis.ipynb`, sections 6--8.

== Cross-check: sweep against study A

Both have an $S_1$ speaker, the `sus_1` score, the same fixed-$z$ rule and the
same world. They differ in the listener's $theta$ grid (11 points including the
endpoints against 9), in the number of simulations (200 against 100), and in
their seeds, so this is a comparison of independent samples. Overlapping
$alpha in {1, 1.5, 2, 3, 4, 5, 7, 10}$ and $c in {2, 3, 3.5, 4, 5}$.

#sqfig("X1_sweep_vs_A.png")[
  *Cell-by-cell agreement.* One point per $(theta^star, alpha, c)$ cell,
  sweep on the $x$-axis against study A on the $y$-axis; colour is $c$. Left:
  false-alarm rate under the honest $S_1$. Right: power against the persuasive
  $S_1$. The dashed line is the identity.
]

#sqfig("X2_rates_vs_alpha.png")[
  *Rates against $alpha$.* Solid is the sweep (which extends to $alpha = 20$),
  dashed is study A; one line per $c$. Left: false alarm. Right: power.
]

#sqfig("X3_residual_heat.png")[
  *Residual by cell.* Sweep minus study A at $c = 3.5$ for every
  $(theta^star, alpha)$ cell. Left: false-alarm rate. Right: power.
]

#sqfig("X4_latency_null.png")[
  *Latency and null score.* Left: median $tau$ over detected persuasive runs
  against $alpha$; solid the sweep, dashed study A, one line per $c$. Right:
  mean per-round `sus_1` score under the honest $S_1$ at $theta^star = 0.3$ in
  both datasets, at three values of $alpha$.
]

#sqfig("X5_theta_space_bias.png")[
  *Effect of the $theta$ grid on the credulous listener.* Time-averaged
  $|"bias"|$ of the credulous $L_1$, 11-point grid minus 9-point grid, for
  every $(theta^star, alpha)$ cell, split by speaker goal.
]

== The speaker-level ladder

#sqfig("L1_fa_ladder.png")[
  *False-alarm rate by speaker level.* Crossing rate under an honest speaker
  against $alpha$ at three boundaries; the $S_1$ sweep, the $S_1$ of study A
  and the $S_2$ of study B.
]

#sqfig("L2_power_ladder.png")[
  *Power by speaker level.* Detection rate of a persuasive speaker against
  $alpha$ at three boundaries, for the same three datasets.
]

#sqfig("L3_level_residual.png")[
  *Level residual.* Study B ($S_2$) minus the sweep ($S_1$) at $c = 3.5$ for
  every $(theta^star, alpha)$ cell. Left: false-alarm rate. Right: power.
]

#sqfig("L4_null_score_by_level.png")[
  *Honest per-round score by audience.* Mean `sus_1` score against round at
  $theta^star = 0.3$ for the honest speaker of the sweep and of each switching
  study, at three values of $alpha$.
]

At $c = 3.5$ the honest-speaker crossing rate is flat in $alpha$ for the $S_1$
sweep ($0.002$ at $alpha = 1.5$, $0.063$ at $alpha = 20$) and rises to
$0.901$ at $alpha = 10$ for the $S_2$ of study B.

== Damage before detection

The quantity is the credulous listener's bias at $tau - 1$: in the sweep from
the stored posteriors, in study B from the counterfactual credulous listener on
the $S_2$ stream.

#sqfig("G1_damage_vs_theta.png")[
  *Credulous bias at $tau - 1$ against $theta^star$*, at $c = 3.5$ for a
  $"pers"^+$ speaker, one line per $alpha$. Left: $S_1$ persuader (sweep).
  Right: $S_2$ persuader (study B).
]

#sqfig("G1b_excess_damage.png")[
  *Excess damage.* The same quantity minus the mean credulous bias an *honest*
  speaker produces at the same round, same $theta^star$ and $alpha$, which
  removes the contribution of the listener's prior still converging.
]

#sqfig("G2_damage_vs_tau.png")[
  *Credulous bias at $tau - 1$ against $tau$.* One point per run, log $x$-axis,
  colour is $alpha$; $c = 3.5$, $"pers"^+$ speaker.
]

= Earlier studies

Two smaller studies preceded the sweep and are retained for their diagnostic
value. Both were re-run against the exact $V_t$ of @ltv.

*Group 1* (`experiments/group1/`) --- 200 simulations $times$ 150 rounds,
$alpha = 3$, $c = 2$, null sweep over
$theta^star in {0.1, 0.3, 0.5, 0.7, 0.9}$. Observed false-positive rates 22.8%
(`surp2`) and 18.3% (`sus`) against a nominal 2.3%. Four sub-experiments
localized the cause: crossings are spread across all $t$ rather than bunched
early (1.1); a warm-up gate at $t_0 = 10$ moves the rates to 17.3% and 14.0% at
no power cost (1.2); the per-round variance is calibrated (ratios $0.99$ and
$1.00$, 1.3); and the $1\/t$ shrinkage of $op("Var")["Sus"^((t))]$ holds to
within about 4% (1.4). The inflation is therefore multiple testing across the
horizon rather than a miscomputed variance or a broken CLT.

*sus variants* (`experiments/sus_variants/`) --- same configuration, with
`surp2` and `sus_1` attached as passive observers. Produced the verdict in
@scores. Per-round calibration ratios are $1.000$ (`sus_1`) and $0.994$
(`surp2`); running-mean ratios $1.010$ and $1.008$.

= Repository map

#table(
  columns: (auto, 1fr),
  align: (left, left),
  stroke: tstroke,
  table.header([*Path*], [*Contents*]),
  [`rsa/`], [Core library: `core.py` (Belief, Semantics, World; the cache/version protocol), `environment.py` (observation model, truth conditions), `speaker0-2.py`, `listener0-1.py`, `game.py`, `setup.py`],
  [`rsa/detection/`], [`scores.py`, `sequential_test.py`, `listener.py` (detector, switching, `peek`/`peek_obs`, `make_switching_listener`), `replay.py` (offline replay of any listener on a stored stream)],
  [`rsa/experimental/`], [Superseded $S_1$ variants; not used in reported results],
  [`experiments/full_sweep/`], [`run_sweep.py` (the sweep runner), `aggregate.py` (builds `agg/` for the notebook), `recompute_variance.py`],
  [`experiments/switching/`], [`run.py` (one runner), `grids/*.json` (one grid per study), `aggregate.py`, `run_all.sh`],
  [`experiments/group1/`, `sus_variants/`], [The earlier studies and their analysis scripts],
  [`notebooks/`], [`sweep_analysis.ipynb`, `switching_analysis.ipynb` (the two analysis notebooks behind this document), `reproduce_paper.ipynb`, `sandbox.ipynb`],
  [`results/`], [One directory per dataset, each with `run_config.json`, `trajectories/`, `agg/` and `figures/`; parquet is gitignored and regenerable, figures and configs are tracked],
  [`docs/`], [This document; `TYPST.md` for the toolchain; `legacy/` for superseded LaTeX sources],
  [`tests/`], [`test_detection.py`, `test_sus_variants.py`, `test_switching.py`, `test_speaker2.py`, `test_replay.py`, `test_switching_runner.py`, `test_cache_versions.py`, `test_s2_model.py`, `test_game.py` --- 156 tests],
)

== Notation and code cross-reference

#table(
  columns: (auto, auto, 1fr),
  align: (left, left, left),
  stroke: tstroke,
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
  [$overline(sigma)^(2,(t))$], [`*_sigma_bar2`], [Running average variance],
  [$tau$], [`SequentialTest.tau`], [First crossing time; `tau_*` columns in `tau_summary.parquet` and `runs.parquet`],
  [$k = t - tau$], [`aligned.parquet` column `k`], [Rounds since the switch],
)

== Reproducing

#block(fill: luma(247), inset: 8pt, radius: 3pt, width: 100%)[
```
# simulations (raw parquet, regenerable; resumable, rerun to continue)
python experiments/full_sweep/run_sweep.py --out_dir results/full_sweep_v2
bash   experiments/switching/run_all.sh            # A, B, Fvig, Fcred in order

# aggregates the notebooks read (about one minute each)
python experiments/full_sweep/aggregate.py
python experiments/switching/aggregate.py

# figures used in this document (both notebooks have SAVE = True)
jupyter nbconvert --to notebook --execute --inplace notebooks/sweep_analysis.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/switching_analysis.ipynb

# this document
typst compile --root . docs/project.typ build/typst/project.pdf
```
]

#v(1em)
#line(length: 100%, stroke: 0.5pt + luma(180))
#v(0.3em)
#text(size: 9pt)[
  Base model: Ke Fang, _A Computational Account of Epistemic Vigilance: Learning
  from Selective Truths through Bayesian Reasoning_, Stanford University. The
  world setup, agent definitions and notation in @setup and @agents follow that
  paper; @detection onward is this project's extension.
]
