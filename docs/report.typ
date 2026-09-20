// Report for Michael Franke.  Build:
//   typst compile --root . docs/report.typ build/typst/report.pdf
// Figures and tables for the simulation section: notebooks/analysis.ipynb
//   -> results/report/figures, results/report/tables

#set page(paper: "a4", margin: (x: 2.3cm, y: 2.4cm), numbering: "1")
#set text(size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)", supplement: "Eq.")
#show heading.where(level: 1): it => {
  v(0.6em)
  it
  v(0.25em)
}
#show heading.where(level: 2): it => {
  v(0.35em)
  it
  v(0.15em)
}
#show figure.caption: set text(size: 9pt)
#show figure: set block(breakable: false)
#show figure.where(kind: table): set figure.caption(position: top)
#set table(stroke: (x, y) => if y == 0 { (bottom: 0.6pt) } else { (bottom: 0.25pt + luma(210)) })
#show table.cell.where(y: 0): set text(weight: "semibold")
#show link: set text(fill: rgb("#1c5cab"))

// Figures stay in reading order (no floating): the simulation section has
// many full-width panels, and letting them float pushes them past the text.
#let fig(name, cap, width: 85%) = figure(
  image("/results/report/figures/" + name, width: width),
  caption: cap,
  placement: none,
)
#let inf = $"inf"$
#let pp = $"pers"^+$
#let pm = $"pers"^-$
#let Lc = $L_1^"cred"$
#let Ls = $L_1^"vig"$
#let Sinf = $S_1^inf$
#let sus = $"sus"$
#let surp(i) = $"surp"_#i$
// single-letter identifiers resolve to the literal letter in math mode, so
// the expectation operator needs a multi-character name.
#let EE = $bb(E)$
#let one = $bb(1)$

// ---------------------------------------------------------------- title --
#align(center)[
  #text(size: 17pt, weight: "bold")[Argumentative Awareness]
  #v(0.3em)
  #text(size: 12.5pt)[How a credulous RSA listener comes to suspect persuasion]
  #v(0.6em)
  // TODO: full name
  #text(size: 10pt)[Kerem #h(0.6em) $dot$ #h(0.6em) ILLC, University of Amsterdam]
  #v(0.2em)
  #text(size: 9.5pt, style: "italic")[Progress report for Michael Franke, September 2026]
]
#v(1.2em)

// ============================================================= 1 ==========
= Introduction

Much of what people know about the world is acquired through testimony, and
the sources of that testimony frequently have an interest in what the
recipient comes to believe. A reader who learns about a clinical trial from a
company's press releases, or about a policy from the statements of its
advocates, is rarely informed in advance whether the source is impartial. If
the reader comes to doubt the source, the doubt is typically prompted by the
communicative behaviour itself: a description that is less specific than the
evidence permitted, or a pattern of emphasis that consistently favours one
conclusion. The present report examines this transition in a computational
model of pragmatic language use. Specifically, we ask under what conditions a
listener who initially accepts a speaker's statements at face value can come
to infer, from the speaker's choice of utterances alone, that the speaker is
persuading rather than informing.

== Persuasion in the Rational Speech Act framework

The Rational Speech Act (RSA) framework treats communication as recursive
probabilistic inference @goodman2016 @degen2023. A literal listener
interprets an utterance by its semantics alone; a pragmatic speaker chooses
an utterance by reasoning about how that listener will interpret it; a
pragmatic listener recovers the speaker's meaning by inverting that choice.
Because the speaker's choice is governed by an explicit utility, the
framework extends readily to speakers whose aims are not purely informative.
A persuasive speaker, one who seeks to move the listener's belief in a
particular direction rather than to reduce the listener's uncertainty, is
obtained by modifying that utility and leaving the rest of the machinery
intact @cummins2021 @barnett2022. The same asymmetry of interest is the
subject of a long literature on strategic communication in economics
@crawford1982 @kamenica2011; what RSA adds is a listener who can, in
principle, reason about the speaker's aim.

@fang2026 develop this idea for the case of _selective truth-telling_
@rogers2017: a speaker who never says anything false but exploits the
vagueness of quantifiers to frame the same evidence favourably or
unfavourably. Their speaker has one of three goals: an informative goal, or
one of two persuasive goals that aim to raise or to lower the listener's
estimate of the treatment's effectiveness. Given its goal, the speaker
chooses among the literally true quantifier statements accordingly. The model
has two listeners. A _credulous_ listener inverts the informative speaker
only; a _vigilant_ listener inverts the full family of speakers and thereby
infers the speaker's goal jointly with the state of the world. In simulation
the credulous listener is systematically misled by a persuasive speaker,
whereas the vigilant listener recovers the true state under a persuasive
speaker and, under an informative speaker, loses almost nothing relative to
the credulous listener. Human speakers behave as the model's persuasive
speaker predicts; human listeners, as we discuss below, largely do not behave
as the model's vigilant listener predicts.

== Vigilance as a prior, and why that is not enough

In the model of @fang2026, credulity and vigilance are not two strategies
between which a listener can move. They are two priors over the speaker's
goal, and which of the two a listener holds is fixed exogenously by a world
type $omega in {"coop", "strat"}$ that both agents are assumed to know. The
credulous listener places all of its prior mass on the informative goal, and
Bayesian conditioning cannot move mass onto a hypothesis that begins at zero.
Consequently, no sequence of utterances, however suspicious, can lead the
credulous listener to revise its assumption that the speaker is informative.
This is not a technical inconvenience but the heart of the matter: the transition of
interest, from taking a speaker at their word to asking why they chose those
words, is precisely the one the model cannot represent.

The transition is also where the empirical picture is least settled.
In the listener experiment of @fang2026, participants who were explicitly
warned that the speaker might be a promoter or a sceptic estimated the
treatment's effectiveness much as participants who had been told the speaker
was honest, and their speaker-type judgements were best fitted by a vigilant
model with a strong prior on the informative speaker and a near-zero
rationality parameter. The authors relate this to the _truth default_ of
@levine2022: belief in what one is told is the resting state, and deception
is entertained only once something triggers the consideration. What such a
trigger can consist of, when the listener has no independent access to the
facts and no history with the source, is left open. Related computational
work on lying shows that listeners can reason their way back to the truth
from a known lie once they are told the speaker's motive @oey2023 @oey2024;
the question here is how a listener could come to entertain a motive it was
never told about.

== The proposal: model criticism inside the listener

Our proposal is that the credulous listener can find the trigger in its own
predictions. Interpreting an utterance in RSA requires the listener to hold a
generative model of the speaker, and a generative model makes predictions:
before each utterance the credulous listener has a distribution over what an
informative speaker would say next, and after each utterance it has a
posterior over what the speaker must have observed to say it. If the speaker
is in fact persuasive, the utterances it produces are systematically more
surprising, relative to these predictions, than the predictions themselves
say they should be. Averaging that excess surprisal across rounds gives a
statistic, the _suspicion score_, whose null distribution the listener can
compute exactly; when the score exceeds a boundary the listener discards the
informative-speaker model and adopts the vigilant listener's prior over
goals. We call the resulting listener _switching_.

The mechanism adds very little to the listener. Every quantity it uses is one
the RSA listener already computes in the course of interpretation, so the
only new machinery is a running average of the per-round surprisal scores and
a boundary that this average has to cross. The boundary is a single
dispositional parameter, the amount of unexplained framing a listener
tolerates before it begins to consider persuasive goals, and it is the
natural place for individual differences in the strength of the truth default
to live. The two ways of scoring surprise that we compare, one based on the
prior predictive over utterances and one on the posterior over the speaker's
observation, are the sequential counterparts of prior and posterior
predictive checks in Bayesian model criticism @box1980 @gelman1996, with the
accumulation and the boundary borrowed from sequential analysis @wald1945.

The rest of this report defines the setting (@setting) and the mechanism
with its two scores (@mechanism), and then presents simulation results in
two parts: how the two scores behave when a level-1 pragmatic speaker talks
to a credulous listener (@sweep), and how the switching listener fares in
dyadic interaction, including against a level-2 speaker that models the
switching listener itself (@switching).


// ============================================================= 2 ==========
= Setting <setting>

The world, utterance space and agents are those of @fang2026; we repeat them
in the notation used there, restricted to what the simulations use.

== World and observations

A treatment has a latent per-session improvement rate
$theta in Theta = {0.1, 0.2, dots, 0.9}$, and each session outcome is
$o tilde "Bernoulli"(theta)$. At every round $t$ the speaker privately
observes $O^((t))$, the outcomes of $n$ patients over $m$ sessions each. All
simulations below use $n = 1$ and $m = 7$, so $O^((t))$ is the number of
effective sessions of one patient, $O in cal(O) = {0, dots, 7}$, with
$P(O | theta) = binom(7, O) theta^O (1 - theta)^(7 - O)$. The world state
$theta^star$ is fixed within a run.

== Utterances and truth

Utterances combine a quantifier and a predicate,
$cal(U) = Q times P$ with $Q = {"none", "some", "most", "all"}$ and
$P = {"ineffective", "effective"}$, realised as _"The patient had $q$
sessions $p$"_. Quantifiers have the standard
semantics on the number $k$ of the $m$ sessions satisfying the predicate:
"none" iff $k = 0$, "some" iff $k >= 1$, "most" iff $k > m\/2$, "all" iff
$k = m$. $"Truth"(u; O) in {0, 1}$ records whether $u$ is literally true of
$O$. Several utterances are true of most observations; with 3 of 7 sessions
effective, both "most sessions ineffective" and "some sessions effective" are
true, and the speaker's choice among them is what carries the persuasive
signal.

== Agents

All agents update sequentially: $P^((0))$ is a prior and
$P^((t))(dot) = P^((t - 1))(dot | "signal at round" t - 1)$. Speakers only
ever produce true utterances.

*Literal speaker and listener.* $S_0$ samples uniformly among the true
utterances and $L_0$ inverts it:
$
  P_(S_0)(u | O) prop "Truth"(u; O), quad quad
  P_(L_0)^((t))(theta | u) prop P_(L_0)^((t))(theta)
  sum_(O') P_(S_0)(u | O') P(O' | theta).
$

*Pragmatic speaker $S_1$.* The speaker's goal $psi$ is one of
$Psi = {pm, inf, pp}$, and its policy is a softmax with rationality $alpha$
over a truth-gated utility,
$
  P_(S_1)^((t))(u | O, psi) prop "Truth"(u; O) dot
  "Inf"^((t))(u; O)^(alpha beta) dot "PersStr"^((t))(u; psi)^(alpha (1 - beta)),
  quad beta = one{psi = inf}.
$ <s1>
Informativeness is the probability that a literal listener recovers the
observation, and persuasiveness is the direction in which $u$ moves the
literal listener's posterior mean:
$
  "Inf"^((t))(u; O) = P_(L_0)^((t))(O | u), quad quad
  "PersStr"^((t))(u; psi) = cases(
    EE_(L_0)^((t))[theta | u] & quad psi = pp,
    1 - EE_(L_0)^((t))[theta | u] & quad psi = pm,
    1 & quad psi = inf.
  )
$
Since $beta$ is either 1 or 0, exactly one of the two terms in @s1 carries
the exponent $alpha$ and the other is raised to the power zero: an
informative speaker is thus purely informative and a persuasive speaker
purely persuasive, subject only to truth.

*Pragmatic listener $L_1$.* The listener inverts $S_1$ with a joint
posterior over the world state and the goal,
$
  P_(L_1)^((t))(theta, psi | u) prop P_(L_1)^((t))(theta, psi)
  sum_(O') P_(S_1)^((t))(u | O', psi) P(O' | theta),
$ <l1>
and reports the marginal $P_(L_1)^((t))(theta | u)$. The rationality
parameter $alpha$ is treated as known. The prior over goals is set by the
world type:
$
  "credulous" Lc: quad P^((0))(psi) = one{psi = inf}, quad quad
  "vigilant" Ls: quad P^((0))(psi) = 1\/3 "  for every " psi in Psi.
$
For $Lc$ the joint in @l1 collapses to a posterior over $theta$ alone under
the informative speaker model $Sinf := P_(S_1)(dot | dot, inf)$.

*Pragmatic speaker $S_2$.* $S_2$ has the form of @s1 with $L_0$ replaced by
an internal $L_1$: $"Inf"^((t))(u; O) = P_(L_1)^((t))(O | u)$ and
$"PersStr"$ defined from $EE_(L_1)^((t))[theta | u]$, where the internal
$L_1$ is whichever listener the speaker models (credulous, vigilant, or the
switching listener of @mechanism). Higher levels repeat this pattern.

// ============================================================= 3 ==========
= The switching mechanism <mechanism>

The switching listener is a credulous $Lc$ equipped with a sequential
test of its own generative model. At each round it scores the utterance it
hears against what its model of an informative speaker predicted, accumulates
the scores, and switches to $Ls$ when the accumulated evidence against that
model crosses a boundary.

It is worth naming the hypothesis being tested once and for all, since
everything below is computed under it. The _null_ is that the credulous
listener's internal model is correct, that is, that the utterances it hears
are produced by the informative speaker $Sinf$ that it inverts. Two features
of this null matter for what follows. First, it is a hypothesis about the
speaker only; the world state $theta$ is not part of it. The listener does
not know $theta$, so under the null the observation the speaker describes is,
from the listener's point of view, drawn from its own current belief
$P_(Lc)^((t))(theta)$ rather than from the true rate $theta^star$. All
expectations "under the null" below are therefore expectations under the
listener's own model at round $t$, in the sense of a prior or posterior
predictive check @box1980 @gelman1996, and not under the actual
data-generating process of a run; we return to the relation between the two
in @seqtest. Second, the listener needs no model of the alternative: it only
has to notice that the data are too surprising to have come from the null.

== Predictive distributions of the credulous listener

At the start of round $t$ the credulous listener holds $P_(Lc)^((t))(theta)$.
Pushing this belief onto observations and composing it with the informative
speaker's current policy gives the listener's _prior predictive_ over
utterances, and Bayes' rule gives its _posterior over the speaker's
observation_ once $u$ is heard:
$
  P_(Lc)^((t))(O) = sum_theta P(O | theta) P_(Lc)^((t))(theta), quad quad
  P_(Lc)^((t))(u) = sum_O P_(Sinf)^((t))(u | O) P_(Lc)^((t))(O), \
  P_(Lc)^((t))(O | u) = frac(P_(Sinf)^((t))(u | O) P_(Lc)^((t))(O), P_(Lc)^((t))(u)).
$ <pred>
The policy $P_(Sinf)^((t))(u | O)$ is the informative speaker's policy at
round $t$. It depends on the round only through $P_(L_0)^((t))$, which is a
function of the public utterance history, so it is available to the listener
without further assumptions. All three distributions in @pred are computed
by $Lc$ in the course of ordinary interpretation.

== Two surprisal scores

Write $H(p) = -sum_x p(x) log p(x)$ for the entropy of a distribution $p$. We
consider two ways of scoring the utterance $u^((t))$ heard at round $t$. Each
compares the surprisal of $u^((t))$ under one of the predictive distributions
of @pred with the surprisal that the same distribution leads the listener to
expect, so that under the null each score has mean zero.

Under the null, the listener's model of round $t$ generates the data by
drawing an observation $O tilde P_(Lc)^((t))$ from the listener's current
belief and then an utterance $u tilde P_(Sinf)^((t))(dot | O)$ from the
informative speaker. We write $EE_"null"$ and $op("Var")_"null"$ for
expectations and variances under this joint, whose $u$-marginal is the prior
predictive $P_(Lc)^((t))(u)$ of @pred. Both scores are centred against this
joint, so "mean zero under the null" means mean zero as judged by the
listener's model going into round $t$.

*Prior predictive surprisal* ($surp(1)$). The surprisal of $u^((t))$ under the
prior predictive, minus its expected value:
$ surp(1)^((t)) = -log P_(Lc)^((t))(u^((t))) - H(P_(Lc)^((t))(dot)). $ <surp1>
The subtracted term is by definition the mean surprisal under
$P_(Lc)^((t))$, so $EE_"null" [surp(1)^((t))] = 0$.

*Posterior predictive surprisal* ($surp(2)$). Define the excess surprisal of
$u$ under the informative speaker's policy at a fixed observation,
$ xi^((t))(u, O) = -log P_(Sinf)^((t))(u | O) - H(P_(Sinf)^((t))(dot | O)), $ <xi>
which, by the same argument, has mean zero under $P_(Sinf)^((t))(dot | O)$
for every $O$. The score averages it over the listener's posterior about
which observation the speaker saw:
$ surp(2)^((t)) = sum_O P_(Lc)^((t))(O | u^((t))) med xi^((t))(u^((t)), O). $ <surp2>
This score also has mean zero under the null. Averaging @surp2 over the prior
predictive, substituting the definition of $P_(Lc)^((t))(O | u)$ from @pred
to clear the denominator, and exchanging the order of summation,
$
  EE_"null" [surp(2)^((t))] & = sum_u P_(Lc)^((t))(u) sum_O P_(Lc)^((t))(O | u) med xi^((t))(u, O) \
                            & = sum_u sum_O P_(Lc)^((t))(O) P_(Sinf)^((t))(u | O) med xi^((t))(u, O) \
                            & = sum_O P_(Lc)^((t))(O) sum_u P_(Sinf)^((t))(u | O) med xi^((t))(u, O) \
                            & = sum_O P_(Lc)^((t))(O) [H(P_(Sinf)^((t))(dot | O)) - H(P_(Sinf)^((t))(dot | O))] = 0,
$ <surp2mean>
the inner sum vanishing because $xi^((t))(dot, O)$ is centred at the entropy
of the very policy it is averaged against.

The two scores differ in where the marginalisation over $O$ sits relative to
the logarithm. $surp(1)$ marginalises first and asks how unexpected the
utterance is overall; $surp(2)$ takes the logarithm inside and asks how
unexpected the utterance is _given the observations that would have led an
informative speaker to say it_. Since
$surp(2)^((t)) = EE_"null" [xi^((t))(u, O) | u = u^((t))]$, it is the
conditional expectation of the per-observation excess surprisal given the
utterance. Persuasive utterances such as "some sessions effective" are common
under the prior predictive, because they are true of many observations, but
they are poor descriptions of the observations that most plausibly accompany
them; $surp(2)$ is designed to register exactly this.

== Variances under the null

To standardise the scores the listener also needs their variances under the
null. Both are available in closed form, and both are sums over the utterance
space, so the listener can evaluate them exactly rather than by sampling.

For $surp(1)$ the variance is the varentropy of the prior predictive,
$
  sigma_1^(2,(t)) = op("Var")_"null" [-log P_(Lc)^((t))(u)]
  = sum_u P_(Lc)^((t))(u) (log P_(Lc)^((t))(u))^2 - H(P_(Lc)^((t))(dot))^2.
$ <var1>

For $surp(2)$, write
$surp(2)^((t))(u) = sum_O P_(Lc)^((t))(O | u) med xi^((t))(u, O)$ for the
score the listener would assign to an arbitrary utterance $u$, so that the
realised score of @surp2 is $surp(2)^((t))(u^((t)))$. The variance is the
second moment minus the squared mean, and by @surp2mean the squared mean is
zero, which leaves the second moment alone:
$
  sigma_2^(2,(t)) & = sum_u P_(Lc)^((t))(u) med surp(2)^((t))(u)^2
                    - (sum_u P_(Lc)^((t))(u) med surp(2)^((t))(u))^2 \
                  & = sum_u P_(Lc)^((t))(u) med surp(2)^((t))(u)^2 - EE_"null" [surp(2)^((t))]^2
                    = sum_u P_(Lc)^((t))(u) med surp(2)^((t))(u)^2.
$ <var2>
Both variances depend only on the listener's state at round $t$, not on the
utterance actually heard. The law of total variance gives @var2 a second
reading. Under the null the excess surprisal $xi^((t))$ has mean zero within
every $O$, so its total variance is the average of the per-observation
varentropies $v^((t))(O) = op("Var")_(u tilde P_(Sinf)^((t))(dot | O)) [-log P_(Sinf)^((t))(u | O)]$
weighted by $P_(Lc)^((t))(O)$. Decomposing that same variance along $u$
instead,
$
  sum_O P_(Lc)^((t))(O) med v^((t))(O) = op("Var")_"null" [xi^((t))]
  &= op("Var")_u [EE_"null" [xi^((t)) | u]] + EE_u [op("Var")_"null" [xi^((t)) | u]] \
  &= sigma_2^(2,(t)) + sum_u P_(Lc)^((t))(u) op("Var")_(O tilde P_(Lc)^((t))(dot | u)) [xi^((t))(u, O)].
$ <ltv>
So $sigma_2^(2,(t))$ equals the average varentropy of the speaker's policy
minus the variance of the excess surprisal that remains once the utterance is
known, that is, the part attributable to the listener's residual uncertainty
about the observation.

== Suspicion score and sequential test <seqtest>

Let $s^((t))$ denote either score at round $t$ and $sigma^(2,(t))$ its
variance. After $t$ rounds the listener holds the running mean of the scores,
the _suspicion score_, together with the running mean of the variances,
$
  sus^((t)) = 1/t sum_(i = 1)^t s^((i)), quad quad
  macron(sigma)^(2,(t)) = 1/t sum_(i = 1)^t sigma^(2,(i)),
$ <sus>
and raises an alarm the first time the suspicion score exceeds a
$z$-scaled boundary,
$ tau = min { t : sus^((t)) > c dot macron(sigma)^((t)) \/ sqrt(t) }. $ <test>
Under the null the scores are mean-zero with the stated variances, so
$sus^((t))$ hovers around zero with standard deviation of order
$macron(sigma)^((t)) \/ sqrt(t)$. The scale treats the per-round scores as
uncorrelated, which is what the listener's model implies, since each score is
centred on the belief formed from the earlier rounds; whether the resulting
variance is accurate in a run is checked against simulation in @sweep. Under
a persuasive speaker the per-round scores no longer average to zero, so
$sus^((t))$ settles at a nonzero level while the boundary keeps shrinking
like $1 \/ sqrt(t)$, and any level that stays positive is eventually crossed.
The two scores differ in whether that level stays positive. For $surp(2)$ it
does: a persuasive utterance remains a poor description of the observations
that would explain it, whatever belief about $theta$ the listener has settled
on. For $surp(1)$ it need not: the credulous listener's belief adapts to the
persuasive utterances, and once it has, those utterances are no longer rare
under its prior predictive, which is all $surp(1)$ measures. Whether
$surp(1)$ still detects then depends on where the credulous belief settles,
and it need not detect at all (@sweep). Both scores at
round $t$ are computed from the belief $P_(Lc)^((t))$ that the listener
carries into the round, which was formed from $u^((1)), dots, u^((t - 1))$;
the round's own utterance enters only as the argument being scored. The
boundary $c$ is the only free parameter of the switching mechanism.

*The null and the world.* The statement that the scores are mean-zero is a
statement about the listener's model, not about the run it is in. In a run
the observations are drawn from $P(O | theta^star)$ at the fixed true rate,
whereas the null draws them from the listener's belief $P_(Lc)^((t))(O)$,
and the two coincide only once that belief has concentrated on
$theta^star$. Under an informative $S_1$ the listener's model of the speaker
is correct, and we expect exactly this concentration: the credulous
listener's estimate of $theta$ converges to the true rate as the rounds
accumulate (@fang2026, Fig. 1B). Hence, at a fixed
$theta^star$, the scores carry a bias in the early rounds, while the belief
is still diffuse, and are mean-zero in the limit; the running mean
$sus^((T))$ inherits that early bias and averages it away. Whether the bias
is small enough for the test of @test to be calibrated at every $theta^star$
over a finite horizon is an empirical question, which @sweep answers.
Conversely, if the listener's speaker model is wrong, whether because the
speaker is persuasive or because it reasons at a deeper level than $Sinf$
(@switching), the belief need not concentrate on $theta^star$, the scores
retain a nonzero mean, and $sus^((T))$ does not return to zero; that is what
the test detects.

== The switch

At $tau$ the listener replaces $Lc$ by $Ls$: it replaces the degenerate prior
$P^((0))(psi | omega = "coop")$ by the uniform prior
$P^((0))(psi | omega = "strat")$ and from then on interprets by @l1 with the
full joint over $(theta, psi)$. Two ways of initialising the joint at $tau$
were run.

- *Hard switch.* The listener adopts the belief a vigilant $Ls$ would hold
  after hearing $u^((1)), dots, u^((tau - 1))$, and then interprets $u^((tau))$
  vigilantly. After $tau$ the listener is indistinguishable from an
  always-vigilant listener on the same stream. This is the retrospective
  reading of the trigger: the listener re-evaluates the whole history in
  the light of its suspicion.
- *Soft switch.* The listener keeps its credulous marginal over $theta$
  from before $u^((tau))$, spreads it uniformly over $psi$,
  $P_(L_1)^((tau))(theta, psi) = P_(Lc)^((tau))(theta) dot 1\/3$, and then
  interprets $u^((tau))$ vigilantly. Nothing is re-evaluated; only the future
  is interpreted with vigilance.

// ============================================================= 4 ==========
= Simulations I: scoring an $S_1$ speaker <sweep>

This section reports the simulation sweep that isolates the two scores. Every
figure and table in it is produced by `notebooks/analysis.ipynb` from the
stored runs of the sweep; nothing is re-simulated there.

== The sweep

*Design.* An $S_1$ speaker with true goal $psi^star$ and rationality $alpha$
talks for 150 rounds to a credulous $Lc$ that computes $surp(1)$ and
$surp(2)$, their null variances, and the two running means of @sus every
round, but never switches. The grid is $theta^star in {0.1, 0.2, dots, 0.9}$,
$psi^star in {inf, pp, pm}$ and
$alpha in {1, 1.5, 2, 2.5, 3, 4, 5, 7, 10, 15, 20}$, with 200 independent
simulations per cell: 297 cells, 59,400 runs and 8.91 million
listener-rounds in all. The world is that of @setting, the listener's
$theta$ grid has eleven points ${0, 0.1, dots, 1}$, and the listener is given
the speaker's $alpha$.

*What is recorded and how the test is scored.* Every round stores the
utterance heard, both per-round scores $s^((t))$ with their null variances
$sigma^(2,(t))$, the running means $sus^((t))$ and $macron(sigma)^(2,(t))$,
and the listener's belief over $theta$. Because the listener never switches,
the test of @test can be evaluated offline on the saved runs for any
boundary $c$: a run is _detected_ at $c$ if
$sus^((t)) > c thin macron(sigma)^((t)) \/ sqrt(t)$ for some $t <= 150$, and
$tau$ is the first such $t$. Crossing times recomputed this way from the
stored running means reproduce the crossing times written during the sweep
exactly, for both scores and every $c in {2, 2.5, 3, 3.5, 4, 5}$.

*Conventions.* $alpha = 3$ and $c = 3.5$ are the reference settings wherever
a single value is shown. Rates pooled over the grid exclude $alpha = 1$,
where the pragmatic speaker is nearly literal and the three goals are hard to
tell apart by any means. _Power_ is the fraction of persuasive runs detected
by round 150 and the _false-alarm rate_ the fraction of informative runs
detected by round 150. Bands in the figures are $plus.minus 1$ standard
error of a mean over the 200 runs of a cell unless the caption says
otherwise.

== The suspicion score

#fig("S17_runmean_inf.png")[
  *Suspicion score under an informative $S_1$*, $alpha = 3$. Mean of
  $sus^((t))$ over 200 runs against round $t$, one panel per $theta^star$,
  for $surp(1)$ (pink) and $surp(2)$ (blue), with $plus.minus 1$ SE bands.
  Dashed: the boundary $c thin macron(sigma)^((t)) \/ sqrt(t)$ at $c = 1$,
  averaged over runs, for scale.
] <f-null-mean>

#fig("S18_runmean_pers_up.png")[
  *Suspicion score under a persuade-up $S_1$*, $alpha = 3$; layout as in
  @f-null-mean.
] <f-pers-mean>

== The variance

#fig("S9_variance_ratio_inf.png")[
  *Null calibration of the variance*, informative $S_1$, $alpha = 3$.
  $op("Var")[sus^((t))]$ across the 200 runs of each $theta^star$ divided by
  the analytic $macron(sigma)^(2,(t)) \/ t$; the prediction is the dashed
  line at 1.
] <f-null-var>

#fig("S9_variance_levels_scaled_inf.png", width: 78%)[
  *Empirical against analytic variance*, informative $S_1$, $alpha = 3$,
  both multiplied by $t$. Solid: $t thin op("Var")[sus^((t))]$ across runs
  with its two-sided 95% $chi^2$ interval (shaded). Dashed: the analytic
  $macron(sigma)^(2,(t))$. Top block $surp(1)$, bottom block $surp(2)$;
  each panel has its own vertical axis.
] <f-null-var-levels>

#fig("S9_variance_ratio_pers_up.png")[
  *Variance against the analytic prediction under a persuade-up $S_1$*,
  $alpha = 3$; layout as in @f-null-var. The persuade-down figure is the
  same, mirrored in $theta^star$.
] <f-pers-var>

== False alarms against the horizon

#fig("S13_false_alarm_vs_T.png")[
  *False-alarm rate against the horizon*, informative $S_1$. Top: fraction
  of runs that have crossed by round $t$, one line per $c$ (light to dark),
  pooled over $theta^star$ and $alpha >= 1.5$; dotted lines are the
  fixed-sample levels $Phi(-c)$. Bottom: the same at $c = 3.5$, one line per
  $theta^star$, pooled over $alpha >= 1.5$.
] <f-fa>

== Absorption of $surp(1)$

#fig("S8_surp1_absorption.png")[
  *Absorption of $surp(1)$ under a persuade-up $S_1$*, $alpha = 3$. Tall
  panels: realised surprisal $-log P_(Lc)^((t))(u^((t)))$ (pink) and entropy
  $H(P_(Lc)^((t)))$ (grey) of the credulous listener's prior predictive,
  rebuilt from the stored beliefs; the shaded gap is $surp(1)^((t))$. Short
  panels: the credulous posterior mean $EE[theta]$ against $theta^star$
  (dotted). Mean $plus.minus 1$ SE over 200 runs.
] <f-absorb>

== Operating characteristics

#figure(
  table(
    columns: 8,
    align: (left, center, center, center, center, center, center, center),
    [score], [$c$], [false alarm], [power, $pp$], [power, $pm$],
    [median $tau$], [$tau$ quartiles], [by round 10],
    [$surp(1)$], [2.0], [0.234], [0.743], [0.740], [3], [1, 11], [0.75],
    [$surp(1)$], [2.5], [0.121], [0.699], [0.700], [4], [1, 14], [0.71],
    [$surp(1)$], [3.0], [0.069], [0.667], [0.669], [4], [1, 17], [0.67],
    [$surp(1)$], [3.5], [0.034], [0.635], [0.636], [5], [2, 21], [0.64],
    [$surp(1)$], [4.0], [0.019], [0.606], [0.608], [5], [1, 23], [0.62],
    [$surp(1)$], [5.0], [0.005], [0.556], [0.559], [6], [1, 28], [0.58],
    [$surp(2)$], [2.0], [0.219], [0.996], [0.995], [2], [1, 7], [0.80],
    [$surp(2)$], [2.5], [0.128], [0.989], [0.988], [3], [1, 12], [0.73],
    [$surp(2)$], [3.0], [0.065], [0.974], [0.970], [4], [1, 20], [0.66],
    [$surp(2)$], [3.5], [0.031], [0.947], [0.947], [5], [2, 27], [0.61],
    [$surp(2)$], [4.0], [0.019], [0.911], [0.911], [6], [2, 34], [0.58],
    [$surp(2)$], [5.0], [0.011], [0.830], [0.826], [7], [1, 43], [0.55],
  ),
  caption: [*Operating characteristics of the sequential test by round
    150*, pooled over $theta^star$ and $alpha >= 1.5$. False alarm is the
    crossing rate under an informative $S_1$ and power the crossing rate
    under each persuasive goal. Median, quartiles and the fraction crossing
    by round 10 are taken over detected persuasive runs.],
) <t-rates>

#figure(
  table(
    columns: 9,
    align: (left, center, center, center, center, center, center, center, center),
    table.header(
      table.cell(rowspan: 2)[$theta^star$],
      table.cell(colspan: 2, align: center)[false alarm],
      table.cell(colspan: 2, align: center)[power],
      table.cell(colspan: 2, align: center)[median $tau$],
      table.cell(colspan: 2, align: center)[$sus^((150))$, persuasive],
      [$surp(1)$], [$surp(2)$], [$surp(1)$], [$surp(2)$],
      [$surp(1)$], [$surp(2)$], [$surp(1)$], [$surp(2)$],
    ),
    [0.1], [0.015], [0.010], [0.85], [1.00], [51], [18.5], [0.35], [0.46],
    [0.2], [0.030], [0.015], [0.14], [0.90], [13], [18.5], [$-0.04$], [0.11],
    [0.3], [0.020], [0.005], [0.19], [0.80], [22], [10], [$-0.04$], [0.13],
    [0.4], [0.015], [0.015], [0.50], [0.99], [37], [16], [0.20], [0.40],
    [0.5], [0.030], [0.015], [0.94], [1.00], [49], [13.5], [0.46], [0.73],
    [0.6], [0.035], [0.000], [0.49], [0.99], [36], [15], [0.19], [0.39],
    [0.7], [0.035], [0.010], [0.18], [0.79], [20], [10], [$-0.05$], [0.12],
    [0.8], [0.015], [0.015], [0.13], [0.88], [12], [19.5], [$-0.04$], [0.11],
    [0.9], [0.035], [0.020], [0.86], [1.00], [46], [17], [0.36], [0.47],
  ),
  caption: [*The reference setting by $theta^star$*, $c = 3.5$,
    $alpha = 3$. Power and the mean of $sus^((150))$ are over both persuasive
    goals (400 runs); median $tau$ is over detected persuasive runs; the
    false-alarm rate is over 200 informative runs.],
) <t-theta>

== Detection time

#fig("S10_tau_violins_pers_up.png")[
  *Crossing round $tau$ under a persuade-up $S_1$*, $c = 3.5$, over
  detected runs. Left: against $alpha$, pooled over $theta^star$. Right:
  against $theta^star$ at $alpha = 3$. Violin width is proportional to the
  power printed above it; the tick is the median.
] <f-tau-up>

#fig("S10_tau_violins_pers_down.png")[
  *Crossing round $tau$ under a persuade-down $S_1$*; layout as in
  @f-tau-up.
] <f-tau-down>

#fig("S11_tau_histogram.png")[
  *When detection happens*, $alpha = 3$, $c = 3.5$, both persuasive goals
  and all $theta^star$ (3,600 runs per score). Fraction of runs whose $tau$
  falls in each 10-round bin; the hatched bar is the fraction never
  detected.
] <f-tau-hist>

== Dependence on speaker rationality, and the two goals

#fig("S16_power_vs_alpha_per_theta.png")[
  *Power against $alpha$ per $theta^star$*, $c = 3.5$, both persuasive
  goals pooled (400 runs per point), one line per $theta^star$ (light to
  dark).
] <f-power-alpha>

// ============================================================= 5 ==========
= Simulations II: the switching listener and the level-2 speaker <switching>

This section reports the studies in which the switch is on, against the $S_1$
speaker of @sweep and against an $S_2$ speaker that models the switching
listener. Every figure and table in it is produced by the second part of
`notebooks/analysis.ipynb` from the aggregated tables in
`results/switching/agg/`, which `experiments/switching/aggregate.py` builds
from the raw trajectories; nothing is re-simulated there.

== The studies

*Design.* Three studies share the grid $theta^star in {0.1, dots, 0.9}$,
$psi^star in {inf, pp, pm}$, $alpha in {1, 1.5, 2, 3, 4, 5, 7, 10}$,
$c in {2, 3, 3.5, 4, 5}$ and both switch types of @mechanism (hard, soft),
with 100 simulations per cell over 150 rounds: 216 $(theta^star, psi^star,
alpha)$ cells per study, each scored under the ten $(c, "switch type")$
conditions. The $theta$ grid has nine points ${0.1, dots, 0.9}$ on both
sides, the listener is given the speaker's $alpha$, and the score is
$surp(2)$ throughout.

- *Study A: $S_1$ speaker.* The speaker of @sweep. It does not model the
  listener, so one utterance stream per cell serves every condition
  (_offline_: the switch cannot feed back into the stream).
- *Study B: $S_2$ speaker modelling the switching listener.* The speaker's
  internal listener is a replica of the actual switching listener, with the
  same $c$ and switch type, fed the same utterances. Its informativeness and
  persuasiveness terms are computed against the belief the switching listener
  _would_ hold after hearing $u$, including any switch that $u$ would trigger.
  Each condition is a fresh simulation (_feedback_).
- *Study Fcred: $S_2$ speaker modelling the credulous listener.* An $S_2$
  whose internal listener is $Lc$, offline. Before any switch this is exactly
  the speaker of study B; it is used where the score must be followed for
  all 150 rounds, because in study B the detector stops at the switch.

*What is recorded.* Every simulation records three listeners on the same
utterance stream, the always-credulous $Lc$, the always-vigilant $Ls$ and the
switching listener, together with the detector's state, and, in study B, the
speaker's distance to the boundary before each choice and whether it chose
the utterance an informative or a persuasive $S_2$ would have chosen. The
observation stream of simulation $i$ at a given $theta^star$ is identical
across studies and conditions, so every comparison between listeners, and
between studies A and B, is paired.

*Conventions.* $alpha = 3$, $c = 3.5$ and the hard switch are the reference
settings wherever a single value is shown. _Switched_ means the switching
listener crossed the boundary at some $t <= 150$, and $tau$ is that round;
under the informative speaker every switch is a false alarm in the usual
sense. Curves are means over the 100 runs of a cell, or over the 900 runs of
a $theta^star$-pooled panel; the aggregated tables carry no spread for the
running mean or for the two reference listeners' error, so these curves are
drawn without bands.

== The informative level-2 speaker under the score

#fig("W7_sus_runmean_inf_S1_vs_S2.png")[
  *Suspicion score under an informative speaker of level 1 and of level 2*,
  $alpha = 3$. Mean of $sus^((t))$ of $surp(2)$ over 100 runs against round
  $t$, one panel per $theta^star$, for the informative $S_1$ of study A and
  the informative $S_2$ of study Fcred, whose internal listener is the $Lc$
  the score is built on. Dashed: the boundary at $c = 3.5$. Both studies are
  offline, so the score is followed for all 150 rounds.
] <f-w-sus>

== Switch rates

#fig("W8_switch_rate_vs_t.png")[
  *Switch rate against the horizon*, $c = 3.5$, hard switch. Fraction of runs
  in which the switching listener has switched by round $t$, one line per
  $alpha$ (light to dark), pooled over $theta^star$. Columns: the $S_1$
  speaker of study A and the $S_2$ speaker of study B. Rows: informative
  speaker (900 runs per line) and persuasive speaker, both goals (1,800 runs
  per line).
] <f-w-rate>

#figure(
  table(
    columns: 9,
    align: (left, center, center, center, center, center, center, center, center),
    table.header(
      table.cell(rowspan: 2)[$alpha$],
      table.cell(colspan: 4, align: center)[$c = 3.5$],
      table.cell(colspan: 4, align: center)[$c = 5$],
      [inf., $S_1$], [inf., $S_2$], [pers., $S_1$], [pers., $S_2$],
      [inf., $S_1$], [inf., $S_2$], [pers., $S_1$], [pers., $S_2$],
    ),
    [1], [0.000], [0.000], [0.566], [0.559], [0.000], [0.000], [0.065], [0.069],
    [1.5], [0.002], [0.011], [0.821], [0.818], [0.000], [0.000], [0.426], [0.389],
    [2], [0.006], [0.046], [0.856], [0.852], [0.000], [0.003], [0.583], [0.573],
    [3], [0.007], [0.461], [0.916], [0.909], [0.000], [0.126], [0.758], [0.729],
    [4], [0.018], [0.764], [0.964], [0.960], [0.000], [0.599], [0.867], [0.856],
    [5], [0.082], [0.878], [0.990], [0.989], [0.006], [0.734], [0.939], [0.938],
    [7], [0.036], [0.973], [1.000], [0.998], [0.014], [0.906], [0.989], [0.993],
    [10], [0.053], [0.994], [1.000], [1.000], [0.010], [0.992], [1.000], [0.999],
  ),
  caption: [*Switch rate by round 150 by speaker level*, hard switch, pooled
    over $theta^star$. $S_1$ is study A and $S_2$ study B; _inf._ is the
    informative speaker (900 runs per entry) and _pers._ the two persuasive
    goals together (1,800 runs).],
) <t-w-rates>

== Accuracy of the three listeners

#fig("W9_listener_accuracy.png")[
  *Accuracy of the three listeners on the same runs*, $alpha = 3$,
  $c = 3.5$, hard switch. Mean $|EE[theta] - theta^star|$ against round for
  the always-credulous, always-vigilant and switching listeners, pooled over
  $theta^star$ (900 runs per panel). Rows: study A ($S_1$) and study B
  ($S_2$ modelling the switching listener). Columns: true speaker goal.
] <f-w-acc>

#fig("W10_accuracy_per_theta_B_pers_up.png")[
  *Accuracy of the three listeners per $theta^star$* under the persuade-up
  $S_2$ of study B, $alpha = 3$, $c = 3.5$, hard switch; 100 runs per panel.
  The dotted vertical line is the median switch round of the panel, and the
  fraction of runs that switched is printed with it.
] <f-w-acc-theta>

#figure(
  table(
    columns: 10,
    align: (left, center, center, center, center, center, center, center, center, center),
    table.header(
      table.cell(rowspan: 2)[$alpha$],
      table.cell(colspan: 3, align: center)[informative $S_2$],
      table.cell(colspan: 3, align: center)[persuade-up $S_2$],
      table.cell(colspan: 3, align: center)[persuade-down $S_2$],
      [cred.], [vig.], [switch], [cred.], [vig.], [switch], [cred.], [vig.], [switch],
    ),
    [1], [0.018], [0.019], [0.018], [0.035], [0.029], [0.033], [0.035], [0.029], [0.034],
    [1.5], [0.017], [0.019], [0.017], [0.042], [0.029], [0.035], [0.041], [0.030], [0.035],
    [2], [0.016], [0.019], [0.016], [0.050], [0.031], [0.035], [0.051], [0.032], [0.036],
    [3], [0.015], [0.018], [0.016], [0.061], [0.033], [0.035], [0.062], [0.035], [0.037],
    [4], [0.014], [0.017], [0.015], [0.072], [0.038], [0.040], [0.073], [0.040], [0.041],
    [5], [0.013], [0.016], [0.014], [0.082], [0.044], [0.045], [0.084], [0.046], [0.047],
    [7], [0.012], [0.014], [0.014], [0.099], [0.061], [0.062], [0.104], [0.068], [0.068],
    [10], [0.012], [0.014], [0.013], [0.101], [0.093], [0.093], [0.100], [0.100], [0.101],
  ),
  caption: [*Time-averaged $|EE[theta] - theta^star|$ over the 150 rounds*
    in study B at $c = 3.5$, hard switch, pooled over $theta^star$, for the
    credulous, vigilant and switching listeners.],
) <t-w-acc>

== Recovery after the switch: hard against soft

#fig("W11_switch_recovery.png")[
  *Recovery around the switch* in study B at $alpha = 3$, $c = 3.5$, over
  the runs that switched, aligned at the switch round and pooled over
  $theta^star$ (weighted by the number of switched runs per cell). Top: mean
  $|EE[theta] - theta^star|$ of the switching listener against
  $k = t - tau$ for the hard and the soft switch, with the always-vigilant
  and always-credulous listeners on the hard-switch runs. Bottom: the
  switching listener's posterior probability of the true goal, with the
  always-vigilant listener's for reference; dotted line at the uniform prior
  $1\/3$. Columns: true speaker goal. In study B the two switch types are
  separate simulations, so their streams differ after $tau$.
] <f-w-rec>

== The persuasive $S_2$ near the boundary

#fig("W12_speaker_evasion.png")[
  *The persuasive $S_2$ near the boundary* (study B, both persuasive goals,
  $c = 3.5$, hard switch, pooled over $theta^star$, pre-switch rounds only).
  Left: probability that the speaker chose the utterance an informative
  $S_2$ would have chosen, against the margin
  $sus^((t)) - c thin macron(sigma)^((t)) \/ sqrt(t)$ at the time of the
  choice, binned; the rightmost bin is the round before a crossing. Right:
  the same probability against round, over pre-switch rounds with at least
  300 observations. One line per $alpha$ (light to dark).
] <f-w-eva>

// ============================================================= 6 ==========
= Discussion

#text(style: "italic", fill: luma(110))[To be written.]

// Candidate points, not rendered:
// - The mechanism as a formalisation of truth-default theory's trigger:
//   credulity is the default, suspicion is accumulated from the form of the
//   utterances alone, and c is a dispositional parameter. Relation to the
//   Exp. 2 finding that human listeners do not reciprocate.
// - surp2 vs surp1: why the log-inside score works; its reading as a
//   conditional expectation of the per-observation excess surprisal; the
//   flat false-alarm rate in alpha as evidence that the centring is right.
// - Level mismatch: the test is a test of S1^inf, and an informative S2 is
//   rejected. Is that a bug or a feature? Argument: any listener at level n
//   will treat a level n+1 informative speaker as strategic; the cost is small
//   because the switched listener is still a good listener.
// - Hard vs soft: what the retrospective re-evaluation buys, and whether a
//   soft switch is the more plausible cognitive story.
// - Speaker evasion is weak in this utterance space; richer spaces, a cost
//   term, or a speaker that models c explicitly.
// - Next steps: human data (Exp. 2 sequences), fitting c, listener-side
//   uncertainty about alpha, larger n (nested quantifiers).

#v(1em)
#bibliography("report.bib", style: "apa", title: "References")
