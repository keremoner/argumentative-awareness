# Finalization plan for the report

Written 27 September 2026 after reading `docs/report.typ`, `docs/project.typ`,
`docs/citation-audit.md`, the detection code, the report figures, and the
released code of Fang et al. (`KeFangPsych/cogsci26_selective_truth`; the PDF
itself was not reachable from this session, so paper-text claims below are
marked "verify against PDF").

The plan has six parts: (1) an overall assessment, (2) the story I recommend
the report tell, (3) concrete defects in the current draft, (4) citations and
literature claims, (5) mathematical additions ranked by cost, (6) an ordered
task list.

---

## 1. Assessment

**What is strong.** The project has a clean, well-posed question (how does a
credulous RSA listener come to suspect persuasion from utterance form alone),
a mechanism that uses only quantities the listener already computes, exact
closed-form null moments for both scores, a large and carefully reproducible
simulation base (seeded, resumable, self-checking), and two results that are
genuinely interesting and not obvious in advance:

- the log-inside score (report `surp2`) works and the log-outside score
  (report `surp1`) is *absorbed* by the listener's own belief adaptation
  (fig. S8 is the best figure in the report);
- the detector is a test of a *specific* speaker model, so an honest
  level-2 speaker is flagged at high α (table of switch rates, informative
  S2 column: 0.46 at α=3, 0.99 at α=10). This is the "level mismatch" and it
  is the most discussable finding in the whole project.

**What is weak, in order of importance.**

1. *Sections 4 and 5 have no argument.* They are a sequence of figures and
   tables with captions but almost no prose. A reader cannot tell which
   result matters, what each figure is evidence *for*, or why it is there.
   This is the single biggest gap, larger than citations or maths.
2. *There is no discussion.* The commented-out bullet list at the end is a
   good skeleton, but nothing is written.
3. *Naming is inconsistent across report, code and figures.* The report's
   `surp1`/`surp2` are the code's `surp2`/`sus_1`, and the code's `surp1` is
   a retired third score. The figure legends still say "posterior
   predictive" for the report's `surp2`, which the citation audit explicitly
   said is wrong (and the report text was fixed accordingly, but the figures
   were not). Anyone who opens the repo after reading the report will be
   confused within a minute.
4. *The results are under-explained where they are most interesting.* The
   per-θ\* power table (`t-theta`) shows `surp1` power going
   0.85, 0.14, 0.19, 0.50, 0.94, 0.49, 0.18, 0.13, 0.86 across θ\*, and
   `surp2` dipping at 0.2/0.3/0.7/0.8. That pattern has a clean analytic
   explanation (section 5, item B) and the report currently says nothing
   about it.
5. *Robustness is thin on the one dimension that matters most.* The listener
   is always given the speaker's true α. Since the null is "my model of an
   informative speaker", a listener with the wrong α has a wrong null, and
   the level-mismatch result already shows what a wrong null does: false
   alarms. This is not addressed anywhere. It is the one *new simulation* I
   would run before finalizing.
6. *Citations are placeholders.* Every literature citation except
   `@fang2026` is currently a `(TO CITE) /* @key */` comment, so the compiled
   PDF literally prints "(TO CITE)" and the bibliography has one entry.

**On "story vs. lab report".** Tell a story. The audience you name (PhD
committees, Michael) reads for a claim and its evidence, not for coverage.
Keep the main text to roughly 12 figures/tables that each carry one claim,
and move everything else to an appendix (the material in `project.typ` is a
ready-made appendix). A master-level report gets credit for the appendix
existing; it gets read for the main text.

---

## 2. The story

One sentence: *A credulous listener can detect persuasion from the form of
utterances alone by running a mean-zero sequential check of its own
informative-speaker model; the check that works is the one that scores the
utterance against the observations that would have made an informative
speaker say it, and its price is that it also rejects honest speakers whose
reasoning is deeper than the listener's model.*

Six claims, each with the evidence that should sit next to it in the main
text:

| # | Claim | Evidence (keep in main text) |
|---|---|---|
| C1 | Both scores are exactly mean-zero and correctly scaled under the listener's own model, and this holds empirically under an honest S1 at every θ\* and α. | One compact null figure: running mean of both scores under honest S1 (a 3-panel reduction of S17: θ\*=0.1, 0.5, 0.9) plus one sentence with the pooled variance ratios (0.999 / 1.001). Move the four S9 variance figures to the appendix. |
| C2 | Under a persuasive S1 the running mean of `surp2` stays positive at every θ\*, but `surp1` returns to zero or goes negative at intermediate θ\* because the credulous belief absorbs the persuasive utterances. | S18 (3-panel reduction) and S8 (3-panel reduction: θ\*=0.2, 0.5, 0.9). This is the mechanistic heart of the report; write a full page on it, with the drift formula of section 5 item B. |
| C3 | Detection is a repeated significance test; the false-alarm rate at a horizon is well above Φ(−c), flat in θ\*, and nearly flat in α, so c can be chosen from a horizon-specific table. | S13 top row and the operating-characteristics table (`t-rates`). Add one row of the Robbins/anytime-valid boundary from `project.typ` to show the principled alternative (section 5 item A). |
| C4 | With `surp2` and c=3.5 the switching listener reaches the vigilant listener's accuracy under persuasion and pays nothing measurable under honesty, and the switch is fast when α is high. | W9 (accuracy of each listener against the speaker that models it), `t-w-acc`, and the τ histogram (S11). Add "damage before detection" (`project.typ` G1/G2) as the honest cost measure: how far the credulous belief had been pushed by the time the alarm fired. |
| C5 | The test is a test of S1^inf, not of honesty: an honest S2 is flagged at high α, and a persuasive S2 that models the detector barely evades it in this utterance space. | W8 (switch rate vs t, S1 vs S2 columns), `t-w-rates`, and W13 (merge W12 into it). Frame the honest-S2 alarm as the mechanism's defining limitation and also as its point: what is detected is *non-conformity with the listener's model of cooperative speech*. |
| C6 | Hard vs soft switch: retrospective re-evaluation recovers faster and identifies the goal better; the soft switch is the cheaper cognitive story. | W11. |

Then a discussion (section 6 below) that connects C5 to truth-default theory
and to Fang et al.'s listener experiment, and that lists the extensions.

---

## 3. Defects in the current draft

Concrete things to fix, roughly in reading order. Line numbers are in
`docs/report.typ`.

**Front matter and setting**

- Lines 74–155: turn every `(TO CITE) /* @key */` into a real `@key`
  citation (the `.bib` entries already exist and were verified by the audit).
- Line 96: "the vigilant listener recovers the true state" → "reduces the
  bias". (Audit finding; verify against PDF fig. 1B.)
- Lines 112–116: Fang's released fits use a "truth-default" prior of
  P(inf)=0.50 and α floors near 0.10 on the persuasive sequences. Say
  "P(inf)=0.5" rather than "strong". Also say the sequences were **5 rounds
  long**; this matters for the discussion (section 6).
- Line 170: "we repeat them in the notation used there, restricted to what
  the simulations use" is not accurate. State the deviations explicitly:
  Fang's released simulations use m=5, T=60, an 11-point θ grid including
  0 and 1; this project uses m=7, T=150, and *two different* listener grids
  (11-point in the sweep, 9-point in the switching studies). Either justify
  the grid difference or note it as an inconsistency; `project.typ` X5 shows
  its effect on bias is small, which is enough to cite.
- Lines 213–219: the released Fang code computes the utility as
  β·log Inf + (1−β)·log PersStr and applies a softmax with α, which gives
  exactly the E[θ|u]^α form used here. So this repo matches the *released
  implementation*. The audit says the *typeset* paper uses
  exp(α·E[θ|u]). Verify against the PDF; if so, add one sentence: "we follow
  the released implementation, in which the persuasive utility is
  log E[θ|u]; the paper's displayed equation differs in this respect".
- Line 245: "credulous" prior is written as an indicator over ψ but Fang's
  code implements it as a collapsed ψ-grid `['inf']`. Same thing; fine.

**Mechanism section**

- Lines 435–450: **un-comment the martingale block**. It is the strongest
  mathematical statement in the report and it is currently invisible. See
  section 5 item A for how to tighten it.
- Line 466: "The boundary c is the only free parameter" is only true given
  the fixed-z rule. Say "the only free parameter of the fixed-z rule".
- Line 452–462: the paragraph on why `surp1` need not detect is correct but
  hand-waving. Replace with the drift identity (section 5 item B), which
  makes it a one-line argument.

**Simulation sections**

- Every figure needs a paragraph of prose *before* it stating the claim it
  supports, and a sentence after it with the caveat. Right now sections 4
  and 5 are figure dumps. Suggested pattern per figure: claim → what the
  figure shows → number(s) that matter → caveat.
- Rename the scores consistently. Simplest fix that avoids re-running
  anything: keep the report's `surp1`/`surp2`, but (a) change `SC_LAB` in
  `experiments/report_figures.py` to "one-step predictive" / "posterior-
  averaged conditional" and regenerate the PNGs; (b) add a footnote in the
  report and a line in `README.md` mapping report names to parquet prefixes
  (`surp1 → surp2_*`, `surp2 → sus1_*`) and noting that the parquet `surp1_*`
  columns are a retired score not used anywhere in the report. Better fix
  if you have a day: rename in the report to something non-colliding
  (e.g. `s_pred` and `s_cond`, or `surp` and `sus`, matching the code).
- Table `t-theta`: add the drift explanation (section 5 B) as the paragraph
  under the table. Without it the numbers look like noise.
- Line 536: say *why* α=3, c=3.5 are the reference settings (α=3 is Fang's
  value; c=3.5 gives ≈3% false alarms at T=150 for both scores).
- Study B design (line 713): say plainly that an S2 that contains an exact
  replica of the switching listener, *including its c and switch type*, is
  an omniscient adversary, so its evasion results are an upper bound on what
  a realistic persuader could do in this utterance space.
- Section 5 has no prose at all until the accuracy table. The honest-S2
  alarm rate (0.994 at α=10) is a headline result buried in a table.
- W12 and W13 show the same quantity; W13 already overlays the empirical
  fraction. Drop W12.
- Section 5's "damage before detection" and "regret / cost–benefit plane"
  analyses exist in `project.typ` (G1, G2, F20, F21) and are exactly what a
  reader wants after C4. Bring a reduced version in.

**Missing robustness (one new simulation, the only one I would run)**

- *Listener α ≠ speaker α.* Give the listener a fixed assumed α (3) and
  sweep the speaker's true α over the existing grid, honest speaker only.
  Report the false-alarm rate as a function of the mismatch. Expectation: a
  speaker more rational than assumed looks like the level-2 case (false
  alarms rise); a speaker less rational than assumed produces
  higher-entropy utterances and the running mean goes negative (no alarms,
  but the null is still wrong). Either way this is one figure, reuses
  `run_sweep.py` with a listener-α argument, and closes the most obvious
  objection. If you cannot run it, at least state the limitation in the
  discussion with the level-mismatch result as the analogy.

---

## 4. Citations and literature claims

The audit (`docs/citation-audit.md`) is good and I agree with it. Its
findings 2–6 are already reflected in the report text. Remaining work:

**Mechanical**

- Restore the six commented citations (goodman2016, degen2023, barnett2022,
  cummins2021, crawford1982, kamenica2011, levine2022, oey2023, oey2024,
  box1980, gelman1996, dawid1984, armitage1969).
- `sperber2010` should be cited at the first use of "vigilance" (line 92 or
  101). It is the term's origin and the report uses the term throughout.
- `wald1945` is no longer cited; either drop it from the `.bib` or cite it
  once as history in the sequential-test paragraph. `itti2009` and
  `grice1975` can be dropped.
- Add DOIs to every entry (audit note).

**Claims to check yourself (these are where "inspired by" was written as
"is an instance of")**

| Claim in report | Status | What to do |
|---|---|---|
| "This approach draws on Bayesian model criticism (Box; Gelman et al.)" | Fine as *motivation*. The scores are not posterior predictive checks: `surp2` is a posterior expectation of a centred conditional log score, not the log of a posterior predictive. | Keep the wording as is (it already says "draws on"). Do not call either score a "predictive check" anywhere, including figure legends. |
| "Evaluating successive predictions follows the prequential perspective (Dawid 1984)" | Correct, and there is a *much* more direct correspondence you are not using: Seillier-Moiseiwitsch & Dawid (1993), *On testing the validity of sequential probability forecasts*, JASA 88(421):355–359, propose exactly a standardized cumulative score Σ(s_i)/√(Σσ_i²) built from proper-scoring-rule residuals of one-step forecasts, and show it is asymptotically N(0,1) under the forecaster's own model. Your `surp1` with the fixed-z rule is that test with the log score. | Read that paper (it is short). If it matches as I expect, cite it as the direct precedent for `surp1` and say `surp2` is your extension to a latent-observation setting. This upgrades the maths section from "inspired by" to "is an instance of, plus a novel extension". |
| "related to repeated significance testing (Armitage et al. 1969)" | Correct. The repeated-test inflation of Φ(−c) is precisely their topic. | Keep. Consider adding Siegmund (1985), *Sequential Analysis*, as the textbook reference if you want one. |
| The Robbins boundary in `project.typ` ("anytime-valid; P(ever cross) ≤ δ") | This is a normal-mixture boundary; the anytime guarantee follows from Ville's inequality applied to a mixture martingale. The exact form you used (√((V+ρ) log((V+ρ)/(ρδ²)))) is the one in Howard, Ramdas, McAuliffe & Sekhon (2021), *Time-uniform, nonparametric, nonasymptotic confidence sequences*, Ann. Statist., which traces it to Robbins (1970). | If you bring the Robbins row into the report (recommended, section 5 A), cite Howard et al. 2021 and Robbins 1970. Note the guarantee is for a martingale with *known* conditional variances; yours has model-based variances, which is the same assumption you already make. |
| "Bayesian surprise" (Itti & Baldi) | Not the same notion (belief change vs. centred surprisal). | Do not cite unless contrasting. |
| Truth-default theory (Levine) | Fine. The mapping of c to individual differences is your proposal; say so. | Keep. Consider Mercier (2020), *Not Born Yesterday*, for the opposing view that people are not gullible, which your C5 result speaks to. |

---

## 5. Mathematical additions, ranked by cost

I agree that deep new theory is out of scope. But three additions are cheap
(hours, not weeks), use only what is already in the report, and change how
grounded the whole thing reads. A fourth is a nice "theory predicts
simulation" check. Items E are out of scope and listed only so you can name
them as future work.

**A. State the martingale structure and use it (½ day).** Under the
listener's own model, E[s^(t) | H_{t−1}] = 0, so (s^(t)) is a
martingale-difference sequence with predictable conditional variances
σ^{2,(t)} (both are functions of H_{t−1} only; your `project.typ` already
stresses the variance is state-only). Hence S_t = Σs^(i) is a martingale
with predictable quadratic variation V_t = Σσ^{2,(i)}, and Z_t = S_t/√V_t is
the standardized martingale. Consequences you can state in three sentences:
(i) the scores are uncorrelated across rounds under the null, which is why
the 1/t law of Var[sus^(t)] holds (fig. D4 in `project.typ`); (ii) the
fixed-z rule is a repeated significance test on a martingale, so its
false-alarm rate grows with the horizon (S13) and Φ(−c) is a lower bound;
(iii) a mixture boundary (Robbins) gives an anytime-valid guarantee by
Ville's inequality, at a cost in power and latency that you have already
measured (`project.typ` stopping-rule tables: at δ=0.05, FPR 0.002 vs 0.031,
power 0.82 vs 0.95, median τ 17 vs 5). Bring two rows of that table into the
report. This turns "we evaluate the false-alarm rate by simulation" into
"we evaluate it by simulation, and here is the principled boundary if one
wants a guarantee".

**B. The drift identity (½ day, and it explains the θ\* pattern).** For any
alternative policy k′(·|O) at fixed O,

  E_{u∼k′(·|O)}[ξ(u,O)] = KL(k′(·|O) ‖ k(·|O)) + H(k′(·|O)) − H(k(·|O)),

and likewise for `surp1` with the marginals p′ and p:

  E_{u∼p′}[surp1] = KL(p′ ‖ p) + H(p′) − H(p).

So the expected per-round score under a persuader is *divergence plus
entropy gap*, and the entropy gap can be negative: a persuasive speaker is a
sharper speaker (it says "some effective" almost always), so its utterance
distribution has lower entropy than the informative one. When the credulous
belief has drifted to the point where p′ ≈ p on the persuasive utterances,
the KL term is small and the negative entropy gap dominates, and `surp1`
goes negative. That is the absorption of fig. S8 in one line. For `surp2`
the same identity applies inside the posterior average, but the KL term is
taken at each O separately, so it cannot be absorbed by the marginal
adapting. Then compute the drift *from the model tables* (no simulation) at
the converged credulous belief for each θ\*, and plot it against θ\* next to
table `t-theta`. You already do this kind of table-based computation for
the level mismatch (`project.typ` F12). The θ\* pattern in `t-theta` should
fall out.

**C. Predicted crossing time (½ day, one figure).** If the drift is μ>0 and
the average variance σ̄² is roughly constant, the fixed-z rule crosses when
t·μ > c·σ̄·√t, i.e. at t ≈ (c σ̄ / μ)². Using the drift from B and σ̄ from the
stored variances, plot predicted τ against the observed median τ per
(θ\*, α). Where they agree, the mechanism is understood analytically; where
they disagree (early rounds, where the belief is still moving) you have a
precise statement of what the simulation adds.

**D. Reading of `surp2` (one paragraph, no computation).** You already note
`surp2` = E_null[ξ | u]. Say what that means: it is the listener's best
estimate, under its own model, of the excess surprisal the utterance would
have had *if the listener knew what the speaker saw*. The law-of-total-
variance decomposition then says its null variance is the average per-
observation varentropy minus the variance that remains because the listener
does not see O. That is the whole conceptual difference between the two
scores, and it belongs in the mechanism section, not only in the equations.

**E. Out of scope (name as future work only).** Exact finite-horizon
crossing probabilities for the fixed-z rule (Armitage-type recursions);
e-process / test-martingale constructions that would give a likelihood-free
anytime-valid test with better power than the Robbins mixture; asymptotic
optimality questions; treating the credulous belief's fixed point under
persuasion analytically.

---

## 6. Discussion outline

Write these in this order; each is one to three paragraphs.

1. **What was shown** (C1–C6 in four sentences).
2. **Why the log-inside score works** (item B in words). The mechanism
   detects *non-informativity relative to the listener's model*, not
   persuasion as such; persuasion is one way to be non-informative.
3. **The level-mismatch result as the mechanism's defining property.** A
   listener at level n will flag an honest speaker at level n+1 whose
   choices are too sharp for the level-n model. Argue this is a feature: the
   switched listener remains a good listener (accuracy table under the
   honest S2 is within 0.002), so the cost of a false alarm is small, and
   what is being detected is "this speaker is not the speaker I assumed",
   which is the right trigger for re-evaluation. Connect to Mercier's point
   that vigilance is cheap when it is model-based.
4. **Truth-default theory and Fang's listener experiment.** Their listeners
   saw 5-round sequences and their fitted α on persuasive sequences floors
   near 0.1. In your sweep, at α≤1.5 and c=3.5, the switch rate within 5
   rounds is essentially zero (read it off the switch-rate-vs-t figure and
   quote the number). So the mechanism *predicts* that those participants
   should have stayed credulous, which is what was observed, and it predicts
   that a longer sequence with a sharper speaker would flip them. That is a
   concrete, falsifiable claim and the natural next experiment. Say it.
5. **Hard vs soft switch.** Retrospective re-evaluation buys faster recovery
   and better goal identification; the soft switch is cognitively cheaper
   (no stored history). Which is more plausible is an empirical question;
   the difference in accuracy is a prediction.
6. **Speaker evasion.** Weak in an 8-utterance space; an omniscient S2 gains
   little. Richer utterance spaces, a cost term, and a speaker that models c
   without knowing it are the obvious extensions.
7. **Limitations.** Known α (and the mismatch result if you run it); n=1;
   tiny utterance space; a single switch with no way back; c is a free
   parameter chosen by horizon; the null is the listener's model, not the
   world, so calibration at fixed θ\* is empirical.
8. **Extensions to other RSA settings.** The mechanism needs only an
   internal speaker model and a way to score heard utterances against it,
   so it applies wherever RSA has a cooperative baseline and a deviating
   speaker: detecting a socially motivated speaker (politeness / white lies,
   Yoon, Tessler, Goodman & Frank 2020), detecting a speaker whose lexicon
   or α differs from assumed, detecting a *change* of goal mid-conversation
   (then the right tool is a CUSUM-type statistic rather than a running
   mean, and the change-detection literature becomes relevant), and
   learning a prior over speaker types across many speakers (hierarchical
   version, where c becomes a learned quantity rather than a disposition).

---

## 7. Ordered task list

Effort estimates are rough; "day" means a focused working day.

**Must do before the report is usable as a writing sample**

1. Restore all citations; fix the four factual qualifications from the audit
   that are still in the text (lines 96, 112–116, 170, 213–219). *½ day.*
2. Fix the score naming in figure legends (`SC_LAB`), regenerate PNGs, add
   the name-mapping footnote. *½ day.*
3. Write the prose for sections 4 and 5 following the C1–C6 table: one
   claim paragraph per figure, and cut the figures listed under "appendix"
   below. *2 days.*
4. Un-comment and tighten the martingale paragraph; add the drift identity
   (items A, B, D). *1 day.*
5. Write the discussion from the outline in section 6. *1 day.*

**Should do**

6. Compute the model-based drift vs θ\* and the predicted τ (items B, C);
   one figure each. *1 day.*
7. Bring the Robbins boundary rows and the damage-before-detection figure
   into the report from `project.typ`. *½ day.*
8. Run the listener-α-mismatch sweep (section 3, last item). *½ day compute
   + ½ day writing.*
9. Read Seillier-Moiseiwitsch & Dawid (1993) and Howard et al. (2021) and
   cite them precisely; verify the Fang PDF's speaker equation. *½ day.*

**Nice to have**

10. Rename scores non-collidingly across report and code.
11. A one-figure comparison with Fang's world (m=5, T=60) to show nothing
    depends on m=7. Cheap with the existing runner.
12. Move `project.typ` results that do not serve C1–C6 into an appendix of
    the report so there is one document.

**Figures: main text vs appendix**

| Keep in main text (reduced where noted) | Move to appendix | Drop |
|---|---|---|
| S17 (3 panels), S18 (3 panels), S8 (3 panels), S13 (top row), `t-rates`, `t-theta` + drift plot, S11, S16, W8, `t-w-rates`, W9, `t-w-acc`, W11, W13, one damage-before-detection panel, one Robbins-vs-fixed-z row | S9 ×4, S10 ×2, W10, the full `project.typ` diagnostics (D1–D4), stopping-rule tables, X1–X5, L1–L4, F14–F22 | W12 (duplicate of W13), S9 "levels" variants |
