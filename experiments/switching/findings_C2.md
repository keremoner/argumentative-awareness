**Finding.** C2 is the properly specified null, and it settles what the level mismatch was
costing. Here the data-generating speaker is an S2 modelling a credulous L1, and the
listeners are L2s whose internal model is that same S2, so the detector's null is exactly
right. The false-alarm rate collapses from the 42–47% that B and C1 recorded at c=3.5 to
**0.2%**, and it stops depending on α altogether (0.000, 0.003, 0.003 at α = 1.5, 3, 10,
against C1's 0.007, 0.277, 0.980). Essentially the entire alarm rate those studies measured
under an honest speaker was misspecification, not false positives. Get the level right and
the test is calibrated — arguably better calibrated than the S1 case, where c=3.5 still
gives 3.3%.

**The power is the price.** Matched at the same cutoff, and pooling over θ* and α:

| | FPR (ψ*=inf) | TPR (pers+) | TPR − FPR |
|---|---|---|---|
| A — S1 speaker, L1 detector, c=2.0 | 0.208 | 0.992 | 0.784 |
| A — S1 speaker, L1 detector, c=3.5 | 0.033 | 0.928 | 0.895 |
| C2 — S2 speaker, L2 detector, c=2.0 | 0.110 | 0.771 | 0.661 |
| C2 — S2 speaker, L2 detector, c=3.5 | 0.002 | 0.547 | 0.545 |

C2 is dominated, and not by a little: at c=2.0 it buys a *lower* false-alarm rate than A
(0.110 against 0.208) and still gives up 22 points of power. Detection is simply harder one
level up. That is the prediction Fang's paper makes for the arms race — a speaker reasoning
about a smarter listener chooses less conspicuous utterances — and it shows up here as a
26–38 point drop in the separation the same statistic can achieve. Note also that C2's power
*rises* with α at c=3.5 (0.50, 0.51, 0.63) where its false-alarm rate stays flat, the
opposite of the pattern in the misspecified studies, where both rise together and meet at 1.

**Switching still pays, and the ordering still holds.** Among the runs that do alarm, the
retrospective switch again lands exactly on the always-vigilant listener's post-τ bias at
every cell (0.011/0.024/0.034 against 0.011/0.024/0.034), while soft runs 0.046–0.065 and
amnesic 0.039–0.055. The credulous bias at the moment of the alarm is larger than anywhere
else in the study (0.072–0.228, against 0.045–0.177 in Experiment A), because the alarm
comes later and a level-2 persuader is more effective in the meantime. Unconditional
round-150 numbers look weaker here (retrospective 0.020–0.031 against the always-vigilant
0.006–0.010) purely because only about half the runs ever alarm at c=3.5, so the average
is diluted by runs that stayed credulous — not because the switch works less well when it
fires.
