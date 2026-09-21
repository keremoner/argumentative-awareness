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
  #text(size: 17pt, weight: "bold")[Detection of Persuasion]
  #v(0.3em)
  #text(size: 12.5pt)[How a listener comes to suspect persuasion]
  #v(0.6em)
  #text(size: 10pt)[Kerem Yoner #h(0.6em) $dot$ #h(0.6em) ILLC, University of Amsterdam]
  #v(0.2em)
  #text(size: 9.5pt)[Supervisor: Michael Franke]
  #v(0.2em)
  #text(size: 9.5pt)[September 2026]
]
#v(1.2em)
// ============================================================= 1 ==========
= Introduction
We often learn about the world from what people say, and most of the time, those people have a stake in what we believe. A company reporting the results of a clinical trial may want the
treatment to look effective; an advocate describing a policy may want us to
support it. Neither has to lie to influence our judgement. Choosing which
facts to mention, and how to describe them, can be enough. A listener who
initially trusts such a source may eventually notice a pattern in its
choices of words and begin to question its motives. This report asks how that change
could arise in a computational model: how a speaker's utterances could
lead a listener to suspect persuasion without independent access to the facts?

== Persuasion in the Rational Speech Act framework

The Rational Speech Act (RSA) framework models language use through
recursive probabilistic reasoning (TO CITE) /* @goodman2016 @degen2023 */. A literal listener
interprets an utterance using its semantics. A pragmatic speaker chooses an
utterance by considering how that listener will interpret it, and a pragmatic
listener reasons backwards from the choice to what the speaker meant.
The speaker's utility determines what counts as a useful utterance. Changing
that utility allows the model to represent persuasion: the speaker can aim
to shift the listener's belief in a preferred direction (TO CITE) /* @barnett2022 */.
Related work on numerical expressions examines how a speaker's argumentative
aims shape the choice of a description (TO CITE) /* @cummins2021 */. Strategic communication
is also studied in economics, where receivers reason about senders with
different interests (TO CITE) /* @crawford1982 @kamenica2011 */. RSA connects this reasoning
about motives to the meanings of particular utterances.

Fang et al. study _selective truth-telling_ in this framework @fang2026.
Their speakers describe treatment outcomes using quantifiers such as
"some" and "most". Several descriptions can be true of the same evidence,
leaving room to present it favourably or unfavourably. The speaker has one
of three goals: to inform, to raise the listener's estimate of the treatment's
effectiveness, or to lower it. A _credulous_ listener interprets every
utterance as the choice of an informative speaker. A _vigilant_ listener
considers all three goals and jointly infers the speaker's goal and the
state of the world. In their simulations, persuasive speakers systematically
mislead the credulous listener, whereas the vigilant listener recovers the
true state and loses little under an informative speaker. Human speakers'
choices follow the model's predictions more closely than human listeners'
responses follow those of the vigilant listener.

== Vigilance as a prior

Credulity and vigilance in this model are fixed starting assumptions. A world
type $omega in {"coop", "strat"}$, known to both agents, determines the
listener's prior over speaker goals @fang2026. The credulous listener assigns
all prior probability to the informative goal. Since Bayesian updating cannot
give positive probability to a hypothesis whose prior probability is zero,
no sequence of utterances can make this listener consider persuasion.
The model therefore leaves open how a listener could move from trusting a
speaker to questioning the reasons behind its choices.

The listener experiment makes this question particularly relevant. Participants
warned that a speaker might be a promoter or a sceptic estimated the
treatment's effectiveness much like participants told that the speaker was
informative. Their judgements of speaker type were best fitted by a vigilant model
with a strong informative prior and a near-zero rationality parameter
@fang2026. The authors connect these findings to the _truth default_: people
ordinarily accept what they are told, with the possibility of deception
entering consideration only when something triggers it (TO CITE) /* @levine2022 */.
Computational work on lying shows how recursive social reasoning can support
both lie design and detection (TO CITE) /* @oey2023 */. Listeners can also correct directional
bias in false messages, although with reduced precision, when they expect
directional deception and greater costs for larger lies (TO CITE) /* @oey2024 */. Here the
question comes one step earlier: how might a listener begin to consider a
persuasive motive that it initially ruled out?

== The proposal: model criticism inside the listener

We propose that the listener can use its own predictions to recognise a
problem with its assumption. Before each utterance, the credulous listener's
model predicts what an informative speaker might say. After hearing the
utterance, the listener can assess how surprising it was under that model.
A persuasive speaker may repeatedly choose descriptions that are more
surprising than the model expects. The listener accumulates this excess
surprisal in a _suspicion score_. If the score crosses a boundary, it begins
to consider persuasive goals by adopting the vigilant listener's prior.
We call this a _switching_ listener.

The scores use distributions that the RSA listener already computes during
interpretation. The additional mechanism keeps a running average, computes
its scale from the model's per-round variances, and compares it with a
boundary. The threshold parameter controls how much accumulated excess
surprisal prompts a switch. It could therefore represent individual
differences in how readily people move beyond the truth default.

This approach draws on Bayesian model criticism: the listener evaluates its
own model using the observations that model is meant to explain
(TO CITE) /* @box1980 @gelman1996 */. We compare two scores, one based on the utterance's
one-step predictive probability and one averaging conditional log scores
over the posterior distribution of the speaker's observation. Both are
centred under the listener's current model, and their per-round variances
can be computed exactly. Evaluating successive predictions follows the
prequential perspective (TO CITE) /* @dawid1984 */. Checking the standardised cumulative
score after every utterance is related to repeated significance testing
(TO CITE) /* @armitage1969 */: a threshold checked repeatedly has a different false-alarm
rate from one checked once. We evaluate the finite-horizon false-alarm rate
by simulation rather than treating the threshold as a calibrated significance
level.

The report first defines the setting (@setting) and the switching mechanism
(@mechanism). The simulations then examine the two scores under a level-1
speaker (@sweep), followed by the switching listener's performance against
both a level-1 speaker and a level-2 speaker that models the switching
listener itself (@switching).


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

The listener initially assumes that the speaker is informative. We use
_null model_ to refer to the generative model behind this assumption: the
informative speaker $Sinf$, together with the listener's current uncertainty
about $theta$. The listener does not know the true rate $theta^star$, so it
predicts the speaker's observations by averaging $P(O | theta)$ over its
belief $P_(Lc)^((t))(theta)$. Expectations "under the null" below refer to
this internal predictive model. They need not equal expectations in a
simulation with a particular fixed $theta^star$. Using the model's own
predictions as a reference is motivated by Bayesian model criticism
(TO CITE) /* @box1980 @gelman1996 */; the centred scores below implement that comparison.
To raise an alarm, the listener needs only this informative reference model,
without specifying a persuasive alternative. We return to the distinction
between its model and the simulation's data-generating process in @seqtest.

== Predictive distributions of the credulous listener

At the start of round $t$ the credulous listener holds $P_(Lc)^((t))(theta)$.
Pushing this belief onto observations and composing it with the informative
speaker's current policy gives the listener's _one-step predictive_ over
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
by $Lc$ in the course of ordinary interpretation. The one-step predictive is
conditional on the preceding utterance history, before the current utterance
is heard.

== Two surprisal scores

Write $H(p) = -sum_x p(x) log p(x)$ for the entropy of a distribution $p$. We
consider two ways of scoring the utterance $u^((t))$ heard at round $t$. Each
measures excess surprisal relative to the listener's model: the first uses
the marginal probability of the utterance, and the second averages centered
conditional log scores over the posterior observation distribution. Under
the null each score has mean zero.

Under the null, the listener's model of round $t$ generates the data by
drawing an observation $O tilde P_(Lc)^((t))$ from the listener's current
belief and then an utterance $u tilde P_(Sinf)^((t))(dot | O)$ from the
informative speaker. We write $EE_"null"$ and $op("Var")_"null"$ for
expectations and variances under this joint. Its $u$-marginal is the one-step
predictive distribution from @pred:
#math.equation(block: true, numbering: none, $P_(Lc)^((t))(u).$)
Both scores are centred against this
joint, so "mean zero under the null" means mean zero as judged by the
listener's model going into round $t$.

*One-step predictive surprisal* ($surp(1)$). The surprisal of $u^((t))$ under the
one-step predictive, minus its expected value:
$ surp(1)^((t)) = -log P_(Lc)^((t))(u^((t))) - H(P_(Lc)^((t))(dot)). $ <surp1>
The subtracted term is by definition the mean surprisal under
$P_(Lc)^((t))$, so $EE_"null" [surp(1)^((t))] = 0$.

*Posterior-averaged conditional surprisal* ($surp(2)$). Define the excess surprisal of
$u$ under the informative speaker's policy at a fixed observation,
$ xi^((t))(u, O) = -log P_(Sinf)^((t))(u | O) - H(P_(Sinf)^((t))(dot | O)), $ <xi>
which, by the same argument, has mean zero under $P_(Sinf)^((t))(dot | O)$
for every $O$. The score averages it over the listener's posterior about
which observation the speaker saw:
$ surp(2)^((t)) = sum_O P_(Lc)^((t))(O | u^((t))) med xi^((t))(u^((t)), O). $ <surp2>
This is a posterior average of conditional log scores, rather than the
negative log probability of a replicated utterance under a posterior
predictive distribution. This score also has mean zero under the null.
Averaging @surp2 over the one-step
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
under the one-step predictive, because they are true of many observations, but
they are poor descriptions of the observations that most plausibly accompany
them; $surp(2)$ is designed to register exactly this.

== Variances under the null

To standardise the scores the listener also needs their variances under the
null. Both are available in closed form, and both are sums over the utterance
space, so the listener can evaluate them exactly rather than by sampling.

For $surp(1)$ the variance is the varentropy of the one-step predictive,
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
When the accumulated variance is positive, the same rule can be written
on the standardised cumulative-score scale:
$
  Z_t = frac(sum_(i = 1)^t s^((i)), sqrt(sum_(i = 1)^t sigma^(2,(i)))),
  quad tau = min { t : Z_t > c }.
$ <standardised-test>
Thus the listener checks a constant upper threshold after every utterance.
This structure is related to repeated significance testing (TO CITE) /* @armitage1969 */,
which studies the overall rejection probability when fixed-level tests are
repeated on accumulating data. A fixed-sample normal tail probability
$Phi(-c)$ does not give the probability of crossing at any time up to a
horizon $T$. The relevant false-alarm rate is
$P_("informative speaker")(tau <= T)$, evaluated here at $T = 150$ for each
threshold. Choosing $c$ to attain a desired overall rate would require
calibration for that horizon and inspection schedule. The present study
compares a grid of thresholds and reports their empirical operating
characteristics; it does not impose a prespecified overall significance level.
Because our scores and their model-based variances adapt to the utterance
history, thresholds derived for independent observations do not transfer
automatically. The stopping rule requires only the informative model,
without a likelihood ratio against a specified persuasive alternative.
/*
The centring gives a precise, but model-dependent, reason for adding the
per-round variances. Let $cal(H)_(t - 1)$ denote the preceding utterance
history. If each utterance is generated from the listener's conditional
predictive distribution, then
$ EE_"null" [s^((t)) | cal(H)_(t - 1)] = 0. $
Thus the scores form a martingale-difference sequence: for $i < t$,
$EE_"null" [s^((i)) s^((t))] = 0$, so scores at different rounds are
uncorrelated under that process. Their conditional variances can still
depend on the history. The sum $sum_(i = 1)^t sigma^(2,(i))$ is therefore
an accumulated conditional-variance scale, not a guarantee that $Z_t$ is
normally distributed. Nor does the uncorrelatedness claim automatically
hold in simulations at a fixed $theta^star$, where the listener's predictive
distribution may differ from the actual one. We check the model-based scale
against simulation in @sweep.
*/

Under a persuasive speaker the scores can have a positive running mean.
If that mean stays bounded away from zero while the average model variance
remains bounded, the boundary shrinks towards zero and is eventually crossed.
The two scores differ in whether that level stays positive. For $surp(2)$ it
does: a persuasive utterance remains a poor description of the observations
that would explain it, whatever belief about $theta$ the listener has settled
on. For $surp(1)$ it need not: the credulous listener's belief adapts to the
persuasive utterances, and once it has, those utterances are no longer rare
under its one-step predictive, which is all $surp(1)$ measures. Whether
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
$sus^((t))$ inherits that early bias and averages it away. Whether the bias
is small enough for the test of @test to be calibrated at every $theta^star$
over a finite horizon is an empirical question, which @sweep answers.
Conversely, if the listener's speaker model is wrong, whether because the
speaker is persuasive or because it reasons at a deeper level than $Sinf$
(@switching), the belief need not concentrate on $theta^star$, the scores
retain a nonzero mean, and $sus^((t))$ does not return to zero; that is what
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

This section reports the simulation sweep that isolates the two scores.
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
  $H(P_(Lc)^((t)))$ (grey) of the credulous listener's one-step predictive,
  rebuilt from the stored beliefs; the shaded gap is $surp(1)^((t))$. Short
  panels: the credulous posterior mean $EE[theta]$ against $theta^star$
  (dotted). Mean $plus.minus 1$ SE over 200 runs.
] <f-absorb>

== Operating characteristics

#figure(
  table(
    columns: 8,
    align: (left, center, center, center, center, center, center, center),
    [score], [$c$], [false alarm], [power, $pp$], [power, $pm$], [median $tau$], [$tau$ quartiles], [by round 10],
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
speaker of @sweep and against $S_2$ speakers that model the listener they
talk to: the switching listener, the credulous $Lc$ or the vigilant $Ls$.

== The studies

*Design.* Four studies share the grid $theta^star in {0.1, dots, 0.9}$,
$psi^star in {inf, pp, pm}$, $alpha in {1, 1.5, 2, 3, 4, 5, 7, 10}$,
$c in {2, 3, 3.5, 4, 5}$ and both switch types of @mechanism (hard, soft),
with 100 simulations per cell over 150 rounds: 216 $(theta^star, psi^star,
  alpha)$ cells per study, each scored under the ten $(c, "switch type")$
conditions. The $theta$ grid has nine points ${0.1, dots, 0.9}$ on both
sides, the listener is given the speaker's $alpha$, and the score is
$surp(2)$ throughout. Each study pairs a speaker with the listener it
models, so that listener's beliefs are the ones that enter the speaker's
informativeness and persuasiveness terms in @s1.

- *Study A: $S_1$ speaker.* The speaker of @sweep. It models no listener, so
  each simulated utterance stream serves every condition (_offline_: the
  switch cannot feed back into the stream), and the credulous, vigilant and
  switching listeners on that stream are each a genuine interaction with it.
- *Study B: $S_2$ modelling the switching listener.* The speaker's internal
  listener is a replica of the actual switching listener, with the same $c$
  and switch type, fed the same utterances. Its informativeness and
  persuasiveness terms are computed against the belief the switching listener
  _would_ hold after hearing $u$, including any switch that $u$ would trigger.
  Each condition is a fresh simulation (_feedback_).
- *Study C: $S_2$ modelling the credulous $Lc$.* The speaker's internal
  listener is the credulous $Lc$ that hears it. It never switches, so the
  stream does not depend on $c$ or the switch type (offline). Up to the
  switch, this speaker coincides with the speaker of study B.
- *Study D: $S_2$ modelling the vigilant $Ls$.* As study C, with the
  vigilant $Ls$ as the speaker's internal listener and audience.

*What is compared.* Every listener is evaluated in the study whose speaker
models it: the credulous $Lc$ in study C, the vigilant $Ls$ in study D and
the switching listener in study B. Each simulation also carries the two
listeners the speaker is not modelling, interpreting the same stream; those
shadow trajectories are not used anywhere in this section. Across studies
B, C and D a comparison at a given $(theta^star, alpha)$ therefore sets
three separate interactions side by side, one per listener type, in which
the $S_2$ adapts to that listener. Comparisons with study A set the same
listeners against a speaker that models none of them.

We also record the detector's state and, in study B, its distance to the
boundary before each choice and whether the chosen utterance matches the
most probable choice under an informative reference policy (defined in
@speaker-boundary). The observation stream of simulation $i$ at a given
$theta^star$ is shared across studies and conditions. Comparisons across
studies and conditions therefore share observations but have different
utterance streams.

*Conventions.* $alpha = 3$, $c = 3.5$ and the hard switch are the reference
settings wherever a single value is shown. _Switched_ means the switching
listener crossed the boundary at some $t <= 150$, and $tau$ is that round;
under the informative speaker every switch is a false alarm in the usual
sense. Curves are means over the 100 runs of a cell, or over the 900 runs of
a $theta^star$-pooled panel; the aggregated tables carry no spread for the
running mean or for the credulous and vigilant listeners' error, so these
curves are drawn without bands.

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

== Accuracy of each listener against the speaker that models it

#fig("W9_listener_accuracy.png")[
  *Accuracy of each listener in its own interaction*, $alpha = 3$,
  $c = 3.5$, hard switch. Mean $|EE[theta] - theta^star|$ against round,
  pooled over $theta^star$ (900 runs per curve). Top row: the credulous,
  vigilant and switching listeners on the stream of the $S_1$ of study A.
  Bottom row: each listener against the $S_2$ that models it, the credulous
  $Lc$ from study C, the vigilant $Ls$ from study D and the switching
  listener from study B; the three curves of a panel come from three
  separate simulations with shared observations. Columns: true speaker goal.
] <f-w-acc>

#fig("W10_accuracy_per_theta_matched_pers_up.png")[
  *Accuracy of each listener per $theta^star$ against the persuade-up $S_2$
  that models it* (studies C, D, B), $alpha = 3$, $c = 3.5$, hard switch;
  100 runs per curve. The dotted vertical line is the median switch round in
  study B, and the fraction of study-B runs that switched is printed with it.
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
    [1], [0.018], [0.019], [0.018], [0.034], [0.029], [0.033], [0.036], [0.031], [0.034],
    [1.5], [0.017], [0.017], [0.017], [0.042], [0.031], [0.035], [0.043], [0.031], [0.035],
    [2], [0.016], [0.017], [0.016], [0.049], [0.031], [0.035], [0.049], [0.032], [0.036],
    [3], [0.014], [0.015], [0.016], [0.059], [0.035], [0.035], [0.060], [0.036], [0.037],
    [4], [0.014], [0.014], [0.015], [0.064], [0.040], [0.040], [0.064], [0.041], [0.041],
    [5], [0.013], [0.014], [0.014], [0.068], [0.043], [0.045], [0.069], [0.048], [0.047],
    [7], [0.013], [0.014], [0.014], [0.073], [0.061], [0.062], [0.074], [0.066], [0.068],
    [10], [0.012], [0.013], [0.013], [0.080], [0.091], [0.093], [0.081], [0.102], [0.101],
  ),
  caption: [*Time-averaged $|EE[theta] - theta^star|$ over the 150 rounds*
    at $c = 3.5$, hard switch, pooled over $theta^star$, each listener
    against the $S_2$ that models it: _cred._ is the credulous $Lc$ of
    study C, _vig._ the vigilant $Ls$ of study D and _switch_ the switching
    listener of study B.],
) <t-w-acc>

Under the informative $S_2$ the three listeners are within 0.002 of one
another at every $alpha$: the switching listener's false alarms (@t-w-rates)
cost it little against an honest speaker. Under the persuasive $S_2$ at
$alpha = 3$ the switching listener's time-averaged error, 0.035, equals the
vigilant listener's and is well below the credulous listener's 0.059, and
@f-w-acc-theta shows the switching curve leaving the credulous one around
the median switch round in each $theta^star$ panel. This ordering holds up
to $alpha = 7$ and reverses at $alpha = 10$, where the persuasive $S_2$ that
models a vigilant or switching listener drives its error above that of the
credulous dyad (0.091 and 0.093 against 0.080 for persuade-up).

== Recovery after the switch: hard against soft

#fig("W11_switch_recovery.png")[
  *Recovery around the switch* in study B at $alpha = 3$, $c = 3.5$, over
  the runs that switched, aligned at the switch round and pooled over
  $theta^star$ (weighted by the number of switched runs per cell). Top: mean
  $|EE[theta] - theta^star|$ of the switching listener against
  $k = t - tau$ for the hard and the soft switch. Bottom: the switching
  listener's posterior probability of the true goal; dotted line at the
  uniform prior $1\/3$. Columns: true speaker goal. In study B the two switch types are
  separate simulations with shared observations but potentially different
  utterance streams, including before $tau$.
] <f-w-rec>

== The persuasive $S_2$ near the boundary <speaker-boundary>

For each choice, we evaluate an informative $S_2$ policy using the same
observation, rationality parameter and internal switching-listener state as
the actual persuasive speaker, but with the goal set to $inf$. Its most
probable utterance is
$ u_"inf"^((t)) = arg max_u P_(S_2)^((t))(u | O^((t)), inf). $
Ties are resolved by the fixed utterance ordering. We then record whether
the persuasive speaker's sampled utterance equals $u_"inf"^((t))$.
The quantity plotted in @f-w-eva is the empirical fraction of such matches within a
margin bin or round, pooled over observations and runs. It measures how
often the persuasive speaker selects the informative policy's most probable
utterance; it is not the probability that policy assigns to the choice, or
a comparison with a separately sampled informative utterance.

#fig("W12_speaker_evasion.png")[
  *The persuasive $S_2$ near the boundary* (study B, both persuasive goals,
  $c = 3.5$, hard switch, pooled over $theta^star$). Only choices made before
  the listener has switched are included; this includes the choice that
  triggers a switch. Left: empirical fraction matching the informative
  policy's most probable utterance, against the pre-choice margin
  $sus^((t - 1)) - c thin macron(sigma)^((t - 1)) \/ sqrt(t - 1)$,
  binned. Round 1 has no preceding score and is excluded from both panels.
  The rightmost bin contains choices closest to the boundary from below,
  whether or not they subsequently trigger a switch. Right: the same
  fraction against round, shown through round 40 where at least 300
  observations remain. One line per $alpha$ (light to dark).
] <f-w-eva>

We can also evaluate this choice directly from the persuasive speaker's
policy. For the same informative reference utterance, define
$ q^((t)) = P_(S_2)^((t))(u_"inf"^((t)) | O^((t)), psi^star). $
This is the probability that the actual persuasive speaker assigns to
choosing the informative policy's most probable utterance. Conditional on
the current observation and agent states, it is the expected value of the
binary match indicator used in @f-w-eva. Averaging these probabilities over
the same choices removes the sampling noise from the current utterance draw,
while retaining variation in observations and interaction histories.

#fig("W13_informative_policy_probability.png")[
  *Policy probability of the informative reference utterance.* Same study,
  conditions, margin bins and inclusion rules as @f-w-eva. Solid lines:
  mean $q^((t))$ under the actual persuasive policy. Dashed lines: empirical
  match fractions, for comparison. Both quantities are pooled using the
  number of choices in each cell as weights. Left: by pre-choice margin;
  right: by round through round 40, with at least 300 observations per point.
  Colours identify $alpha$.
] <f-w-prob>

The policy probabilities broadly agree with the empirical frequencies.
In the closest-to-boundary bin, $(-0.01, 0]$, their means are 0.341 at
$alpha = 3$, 0.405 at $alpha = 7$ and 0.523 at $alpha = 10$; the
corresponding empirical fractions are 0.347, 0.390 and 0.542, based on
4,922, 459 and 240 choices respectively. The increase near the boundary
at high rationality is therefore also present in the policies themselves.
Some irregularities in the empirical curves are less pronounced: at
$alpha = 10$ in the adjacent bin $(-0.02, -0.01]$, the mean probability
is 0.450, compared with an empirical fraction of 0.374 over 211 choices.
These are descriptive comparisons across different observations and
histories, conditional on the listener not having switched before the
choice. They do not isolate the effect of proximity to the boundary or
by themselves establish deliberate evasion.

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
