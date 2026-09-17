**Finding.** A retrospective switch recovers Fang's vigilance benefit in full, and it
recovers it instantly. Once the detector fires at τ, the retrospective listener's belief
*is* the always-vigilant listener's belief — not approximately, but to machine precision,
because replaying ψ-agnostic likelihoods over the stored history is the same computation
the always-vigilant listener performed live. Averaged over the rounds from τ on, its mean
|E[θ]−θ*| equals the always-vigilant listener's to three decimals at every α (0.014 vs
0.014 at α=1.5, 0.033 vs 0.033 at α=10), while the credulous listener it replaced sits at
0.030 and 0.111. The whole cost of starting credulous is therefore confined to the rounds
before τ; nothing is paid after it.

**What the two cheaper switches lose.** Neither alternative reaches that. The soft switch
inherits the credulous θ-marginal, which is exactly the belief the persuasive speaker has
spent τ rounds bending, and it inherits it without the ψ-uncertainty that would justify
discounting it; its mean post-τ |bias| runs 0.024–0.046, a quarter to a third worse than
retrospective at moderate α, and only 88–97% of its runs ever come within 0.02 of the
vigilant listener's bias. The amnesic switch throws the history away instead and pays the
opposite price: it restarts from a uniform prior and needs a few rounds to climb back
(median 5 rounds at α=1.5, 3 at α=3). Retrospective is the best of the three at every α,
but the ranking of the other two flips: soft beats amnesic at α=1.5, where detection is
late and there is a long credulous history worth inheriting, and amnesic beats soft from
α=3 up, where τ arrives early and that history is mostly distortion. At α=1.5 the amnesic
switch is in fact slightly *worse than not switching at all* (mean |bias| 0.044 against
the credulous 0.042), the only place in the study where a switch is a net loss under a
persuasive speaker.

**The cost of starting credulous grows with α, for two compounding reasons.** The credulous
bias at the moment of the alarm rises from 0.045 at α=1.5 to 0.177 at α=10. That is not
because detection gets slower — it gets much faster, median τ falling from 66 rounds to 1 —
but because a sharper speaker distorts the belief faster than the detector can flag it, and
because at low α there is so little distortion that the runs which trip the boundary at all
are the unrepresentative ones. The practical reading is that the detector's latency only
matters in the α range where the damage is small anyway.

**False alarms are nearly free, and that is the asymmetry that makes the design work.**
Under an honest speaker at c=3.5 the boundary fires on 3.3% of runs, and on those runs the
retrospective and soft listeners end at |bias| 0.000 at round 150 — indistinguishable from
the credulous listener they replaced, because a vigilant listener facing a genuinely
informative speaker simply identifies it as informative and converges anyway. Only the
amnesic variant leaves a visible scar (mean post-τ |bias| 0.018–0.028 against the credulous
0.006–0.016). Combined with the c=2.0 column, where the false-alarm rate is 21% and the
cost of those alarms is still ~0.000 at round 150, this says the usual worry about
trigger-happy detection does not apply here: for a retrospective switch, a false alarm
costs essentially nothing, so the boundary should be tuned for power, not for calibration.

**Sanity.** First-crossing rates reproduce the full sweep on the shared (θ*, α) cells:
0.033 vs 0.027 false alarms and 0.928/0.933 vs 0.943/0.940 power at c=3.5, 0.209 vs 0.233
and 0.992 vs 0.994 at c=2.0. The residual gaps are the 9-point versus 11-point θ grid and
100 versus 200 sims, not a change in behaviour.

**Front-loading.** Persuasive speakers lean on `some` from the first round and ease off
slightly as the listener's belief firms up (68% → 59% of rounds at α=10, rounds 1–10 versus
51–150), whereas informative speakers do the opposite (29% → 34%). The vague quantifier is
thus front-loaded by the speakers that are trying to exploit it, which is the behaviour the
detector is picking up — and a hint that a detector weighted toward early rounds would have
better latency than a flat running mean.
