**Finding.** Moving to an S2 speaker changes the conclusion of Experiment A. The
retrospective switch still recovers *exactly* the always-vigilant belief from τ on — the
identity is about a listener and a stream, not about how the stream was produced, so it
survives the speaker adapting (checked at 0.0 across all 10,800 runs). What no longer
survives is the claim that vigilance is cheap. At α=10 the S2 drives the always-vigilant
listener to |E[θ]−θ*| = 0.039 at round 150, almost six times the 0.007 it managed against
the S1 speaker of Experiment A. And at round 50 the ordering has collapsed altogether: the
switching listener sits at 0.112 and the always-vigilant one at 0.113, both *worse* than
the credulous listener they were supposed to improve on (0.106).

**But the feedback loop is not what does this**, and that is worth stating before anything
else, because it is the natural misreading of this study. Experiment C1 runs the same S2
against the same detector with the loop *open* — its internal listener is a fixed
always-vigilant L1 that knows nothing about any detector — and it degrades the vigilant
listener just as much, to 0.043 at α=10 against B's 0.039, if anything slightly more. The
damage comes from the speaker having one more level of recursion, not from its modelling the
switch. Everything below that looks like strategic adaptation has to survive that control,
and mostly does not.

**Retro versus soft is the headline, and retro wins everywhere.** At round 150 and c=3.5,
retrospective sits at 0.008/0.009/0.039 (α = 1.5/3/10) against soft's 0.016/0.015/0.045 —
a factor of two at low α, narrowing at high α where both are swamped by the speaker's
adaptation. Retrospective also matches the always-vigilant listener to the third decimal at
every α (0.007/0.009/0.039), while soft does not. The gap is wider here than in Experiment
A at the same c, which is the feedback effect working against soft: the credulous
θ-marginal that soft inherits was shaped by a speaker optimising against the very listener
that will inherit it.

**The trade-off is front-loaded, but not for the reason the spec expected.** The persuasive
S2 does lean persuasive early and inform later: the probability its policy puts on the
utterance a credulous-modelling S2 would choose falls from 0.50 in rounds 1–5 to 0.37 by
rounds 51–150 at α=10, while the informative reference holds near 0.29, and at α=3 the lean
(0.42 versus 0.29) disappears entirely by round 50. But this is not boundary avoidance. At
α=1.5 the lean has already collapsed by round 25, when only 5% of runs have switched and the
boundary is nowhere near. The mechanism is the listener's belief sharpening: once a listener
is confident, E_L1[θ|u] barely moves with u, the persuasiveness term flattens, and the
policy decays toward uniform over the true utterances. The tell is that the probability on
the *chosen* utterance converges to 0.315 ≈ 1/3 at α ≤ 3, which is what a uniform choice
among about three true utterances gives. The speaker goes quiet because it has run out of
leverage, not because it is hiding.

**Boundary-hugging is survivorship, not strategy.** Sus(t) averaged over the runs that have
not yet switched does track the boundary from below and converge onto it, exactly as the
spec predicted. That figure is not evidence of anything: conditioning on "has not crossed
yet" selects the runs whose statistic stayed under the boundary, so the survivor mean must
lie below it whatever the speaker does. The selection-free test is the *unconditional*
first-crossing rate against C1's non-adaptive speaker at matched cells, and it comes out
flat — a mean difference of −0.003 ± 0.007 over 120 matched persuasive cells, with median τ
indistinguishable too (44.4 versus 46.2 rounds at α=1.5, 2.3 versus 2.6 at α=10). The
speaker that models the detector is detected neither less often nor later than the speaker
that does not. `experiments/switching/compare_hugging.py` runs this test.

The reason is structural, and it is a limitation of the design rather than a result about
persuasion: `peek` shows the speaker the *consequence* of tripping the detector — a
candidate that would trigger returns the discounted vigilant marginal, which scores lower —
but the speaker maximises a one-round utility, so it can never trade a small loss now for
staying undetected later. Modelling the detector one step ahead is not enough to produce
evasion; that would need a speaker that plans over the horizon.

**Under an honest S2 the detector fails outright at high α, and this is the most important
caveat in the study.** The listener models S1-inf, so an honest S2-inf is already
misspecified for it, and the resulting alarm rate is a level-mismatch rate rather than a
false-alarm rate. At c=3.5 it runs at 0.3% at α=1.5, 40% at α=3 and **100%** at α=10, with
median τ of 6 rounds. At α=10 the detector therefore fires on every run whether the speaker
is persuasive or scrupulously honest: measured power of 92% against pers+ and pers− is not
discrimination at that α, because the honest rate is 100% too. The statistic answers "is my
S1-inf model wrong?", and one extra level of speaker reasoning is enough to make the answer
yes. Any deployment would have to either model the speaker's depth or accept that it cannot
tell a persuader from a deeper cooperator.
