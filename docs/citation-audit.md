# Citation audit of Argumentative Awareness

Checked 20 September 2026 against `docs/report.typ` and `docs/report.bib`.

**Revision status:** Findings 2–6 have now been addressed in the report.
The sequential-testing revision incorporates the subsequent discussion:
Dawid (1984) motivates sequential predictive assessment, and Armitage,
McPherson & Rowe (1969) provide the repeated-significance-testing connection.
The report distinguishes empirical finite-horizon false-alarm evaluation
from choosing a threshold to enforce a prespecified overall significance
level. Finding 1 was explicitly excluded from the requested revision; its
associated report passages and model equations were left unchanged.
The audit below records the original review, so its line numbers and citation
counts refer to the pre-revision report.

**Result: no fabricated publications found.** All 17 bibliography entries identify real works. The report contains 20 citation occurrences referring to 14 distinct works; three bibliography entries are unused. Several real sources are attached to claims that need qualification, and the attribution of the implemented model needs correction.

This is an audit, not an edited report. Source and bibliography files were left unchanged. Locations below are source-file line numbers at the time of review. Internal equation/section/figure references are not counted as literature citations. Archived reports and `docs/project.typ` are outside this audit.

## Findings requiring attention

### 1. High priority: the model is not identical to the cited published model

**Locations: report lines 182 and 222–243.** The published Fang paper, p. 625, uses `exp(alpha * E[theta|u])` for persuade-up; the report uses `E[theta|u]^alpha`. The corresponding persuade-down difference also applies. These are different policies. The report matches `rsa/speaker1.py:170`, so this is not just a typesetting error. Identify this as a variant or cite the exact earlier specification used; do not silently change the equation without checking the simulations. [Published paper](https://charleywu.github.io/downloads/fang2026selective.pdf).

The following additional qualifications concern the same source:

| Report location | Assessment and correction |
|---|---|
| 95–110 | Relevant, but replace “recovers the true state” with “reduces bias”: Fig. 1B retains bias under persuasion. |
| 114–119 | Fixed credulous/vigilant assumptions are supported; the common-knowledge world variable `omega` is not specified in this published formalization. Attribute that construction separately. |
| 127–131 | Qualify the fit: informative prior probability is 0.50; near-floor rationality concerns persuasive sequences. Table 2 has informative-sequence exceptions. |
| 182–193 | Label the one-patient/seven-session setup as an adaptation; published simulations use five patients per round. |
| 450 | Fig. 1B supports finite-run behavior at theta = 0.2, not a general convergence theorem. Say “consistent with” and identify convergence as an expectation. |

### 2. High priority: Bayesian model-checking citations do not establish equivalence to these scores

**Locations: 165–169, 288–290; terminology at 299, 330 and 336.** Box and Gelman, Meng & Stern are appropriate background for comparing observations with model predictions. However, “the sequential counterparts of prior and posterior predictive checks” asserts a stronger correspondence than the report establishes. Gelman et al. define checks using replicated data and discrepancies under a posterior predictive distribution. [Original article, especially Section 2](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A6n41.pdf).

My assessment of the report's equations: `surp2` is a posterior expectation over a latent observation of a centered conditional log score. It is not the negative log probability of a replicated utterance under a posterior predictive distribution. Posterior averaging of log probabilities and taking the log of a posterior predictive mixture are generally different operations. Also, the “prior predictive” at round t is already conditioned on preceding utterances: “one-step predictive” would be less ambiguous.

Suggested replacement:

> Our scores are motivated by Bayesian model criticism (Box, 1980; Gelman et al., 1996). We compare a centered one-step predictive log score with a posterior-averaged conditional log score; both are centered under the listener's current generative model.

Use the same distinction at the second citation pair on line 290. Retain these references as conceptual motivation, and describe the score derivations as this report's contribution. [Box's article and discussion](https://www.cs.princeton.edu/courses/archive/fall09/cos597A/papers/Box1980.pdf).

### 3. High priority: Wald is historical background, not a justification for this boundary

**Location: 169; relevant definition at 418.** Wald (1945) is a real, foundational sequential-testing paper. The claim that “the accumulation and the boundary” are borrowed from it implies a specific derivation that the report does not provide. Its stopping rule uses centered scores and a repeatedly inspected z-scaled boundary; it does not present a sequential probability ratio test. A fixed-sample normal tail probability is not automatically the probability of ever crossing this boundary.

Suggested replacement:

> Following the sequential-testing idea of accumulating observations until a stopping rule is met (Wald, 1945), we use a z-scaled score boundary whose finite-horizon false-alarm rate is evaluated by simulation.

This is an assessment of the report's procedure, not a claim to have checked every theorem in Wald. The journal issue verifies the citation; full text was not accessible through the attempted publisher/JSTOR routes. [Original journal issue](https://www.jstor.org/stable/i312742), [paper identifier](https://doi.org/10.1214/aoms/1177731118).

### 4. Medium priority: split the two Oey claims and restore the conditions

**Location: 136–138.** Oey, Schachner & Vul (2023) concern designing and detecting lies through recursive social reasoning. Oey & Vul (2024) directly address recovering information from false messages, conditional on expectations about directional motives **and greater costs for larger lies**. The current joint citation makes both papers support the latter result and drops the cost condition. It also makes truth recovery sound more exact than the reported loss of precision permits. [2023 article abstract](https://pubmed.ncbi.nlm.nih.gov/35980710/), [2024 publisher abstract](https://link.springer.com/article/10.1007/s42113-023-00187-0).

Suggested replacement:

> Related work models lie design and detection through recursive social reasoning (Oey et al., 2023). Listeners can also correct directional bias in false messages, with reduced precision, when they expect directional deception and greater costs for larger lies (Oey & Vul, 2024).

### 5. Medium priority: the economics comparison misstates what RSA adds

**Location: 90–93.** Both economics references support the strategic-communication connection. The continuation claiming that RSA adds a listener able to reason about a speaker's aim misleadingly suggests that strategic receivers are missing from those models. Crawford & Sobel explicitly model rational responses to a sender's strategy; Kamenica & Gentzkow model persuasion of a Bayesian receiver through signal design. [Crawford & Sobel, Sections 1–2](https://www.haverford.edu/sites/default/files/CrawfordSobel1982.pdf), [Kamenica & Gentzkow](https://www.aeaweb.org/articles?id=10.1257/aer.101.6.2590).

Suggested replacement:

> Strategic communication has a long history in economics (Crawford & Sobel, 1982; Kamenica & Gentzkow, 2011). RSA provides a framework for connecting strategic utterance choice to linguistic meaning and recursive pragmatic interpretation.

### 6. Low priority: distinguish the roles of Cummins & Franke and Barnett et al.

**Location: 90.** Both are relevant. Barnett et al., especially equations 4–6, directly support introducing persuasive goals through the speaker's utility. Cummins & Franke support the wider point about interpreting numerical expressions used for argumentative purposes. Move the latter to a sentence about argumentative framing if you want the methodological attribution to be precise. It is not a fabricated or wholly misplaced citation. [Barnett et al.](https://cocosci.princeton.edu/papers/barnett2022pragmatic.pdf), [Cummins & Franke](https://www.frontiersin.org/journals/communication/articles/10.3389/fcomm.2021.662027/full).

## Complete citation inventory

“Metadata matches” means the supplied author list, title, year, venue, volume/issue where applicable, and page/article number identify the verified work; formatting details and publication-date conventions are noted below. Source links provide the verification trail. Repeated citations are all included in the location column.

| BibTeX key / work | Lines in report | Existence and metadata | Relevance and placement |
|---|---:|---|---|
| `fang2026` — Fang et al., Selective Truths | 95, 114, 127, 182, 450 | Real; 2026 proceedings, pp. 624–630 match. [Author-hosted published PDF](https://charleywu.github.io/downloads/fang2026selective.pdf). | Central source; see finding 1 for all five placements. |
| `goodman2016` — Goodman & Frank | 81 | Real; metadata matches. [Published PDF](https://langcog.stanford.edu/papers_new/goodman-2016-tics.pdf). | Correct support and placement for recursive probabilistic pragmatics. |
| `degen2023` — Degen | 81 | Real; metadata matches. [Publisher](https://www.annualreviews.org/content/journals/10.1146/annurev-linguistics-031220-010811). | Correct support and placement for the RSA overview. |
| `cummins2021` — Cummins & Franke | 90 | Real; metadata matches. [Publisher](https://www.frontiersin.org/journals/communication/articles/10.3389/fcomm.2021.662027/full). | Relevant context; less direct methodological support than Barnett. Finding 6. |
| `barnett2022` — Barnett, Griffiths & Hawkins | 90 | Real; metadata matches. [Published PDF](https://cocosci.princeton.edu/papers/barnett2022pragmatic.pdf). | Correct support for persuasive utility extensions. |
| `crawford1982` — Crawford & Sobel | 92 | Real; metadata matches. [Published paper](https://www.haverford.edu/sites/default/files/CrawfordSobel1982.pdf). | Correct background; repair the following contrast with RSA. Finding 5. |
| `kamenica2011` — Kamenica & Gentzkow | 92 | Real; metadata matches. [Publisher](https://www.aeaweb.org/articles?id=10.1257/aer.101.6.2590). | Correct background; same contrast problem. Finding 5. |
| `rogers2017` — Rogers et al. | 96 | Real; metadata matches. [Coauthor's publication record](https://rzeckhauser.scholars.harvard.edu/publications/artful-paltering-risks-and-rewards-using-truthful-statements-mislead-others). | Correctly supports using true statements to mislead. The specific quantifier model should remain attributed to Fang. |
| `levine2022` — Levine | 133 | Real; metadata matches. [Publisher abstract](https://www.sciencedirect.com/science/article/pii/S2352250X22000999). | Correct support for default acceptance and suspicion triggers. Treat the mapping from the threshold c to individual differences as your proposal, not a result established by Levine. |
| `oey2023` — Oey, Schachner & Vul | 138 | Real; metadata matches. [Article record and abstract](https://pubmed.ncbi.nlm.nih.gov/35980710/). | Relevant but attached to an overly specific shared claim. Finding 4. |
| `oey2024` — Oey & Vul | 138 | Real; metadata matches. [Publisher](https://link.springer.com/article/10.1007/s42113-023-00187-0). | Directly relevant, but restore lying costs and reduced precision. Finding 4. |
| `box1980` — Box | 168, 290 | Real; page range includes discussion. [Original scan](https://www.cs.princeton.edu/courses/archive/fall09/cos597A/papers/Box1980.pdf). | Relevant model-criticism motivation at both placements; avoid claiming score equivalence. Finding 2. |
| `gelman1996` — Gelman, Meng & Stern | 168, 290 | Real; main-article page range matches. [Publisher PDF](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A6n41.pdf). | Relevant background at both placements; posterior predictive terminology needs correction. Finding 2. |
| `wald1945` — Wald | 169 | Real; metadata matches journal contents. [Journal issue](https://www.jstor.org/stable/i312742). | Valid historical citation; insufficient attribution for the specific boundary. Finding 3. |
| `sperber2010` — Sperber et al. | Unused | Real; metadata matches. [Author's record](https://www.dan.sperber.fr/?p=833). | Relevant foundational reference for epistemic vigilance; optional placement at its first conceptual introduction. No current placement to assess. |
| `itti2009` — Itti & Baldi | Unused | Real; metadata matches. [Publisher](https://www.sciencedirect.com/science/article/pii/S0042698908004380). | Relevant only if contrasting notions of surprise. Their Bayesian surprise concerns belief change; do not cite it as the definition of this report's centered surprisal scores. |
| `grice1975` — Grice | Unused | Real; chapter details match. [Chapter scan](https://web.stanford.edu/class/psych205/papers/Grice-1975.pdf), [publisher chapter record](https://brill.com/display/book/edcoll/9789004368811/BP000003.xml). | Relevant historical pragmatics background; optional, since modern RSA sources already support the existing overview. |

## Bibliographic details that are not hallucinations

- **Oey & Vul (2024):** first published online in November 2023, assigned to volume 7 in 2024. The publisher's recommended citation uses 2024. Keep the report's year.
- **Oey et al. (2023):** online publication in 2022 does not invalidate the 2023 volume/issue citation.
- **Degen (2023):** the 2022 advance-publication date does not invalidate the 2023 volume citation.
- **Box:** 383–430 includes discussion; the publisher's article-only record gives 383–404. The longer range is defensible, not invented. Consider adding “with discussion” and using the historical journal designation, Series A (General). [Publisher metadata](https://academic.oup.com/jrsssa/article/143/4/383/7105478).
- **Gelman et al.:** 733–760 is the main paper; 733–807 includes discussion and rejoinder. Do not change the current range simply because a downloaded combined PDF runs longer.
- DOI/URL fields are missing from the bibliography. Adding verified persistent links would improve traceability; their absence is not evidence of fabrication.
- The three unused entries are not citation-placement errors. The bibliography command does not request all entries, so their presence in the `.bib` file should not be confused with their appearance in rendered references.

## Verification limits

This audit checked publication identity, supplied bibliographic details, and each citation's relationship to its surrounding claims. It is not a replication of the cited experiments or of this project's results. Full texts were used for the principal model/equation comparisons and methodological distinctions; publisher/author records and original abstracts suffice for several narrower background claims. Wald's full text was inaccessible in the attempted routes, so no complete theorem-level validation is claimed. Oey et al. (2023) was assessed from its original abstract and bibliographic record, not a full experimental-methods audit.

The highest-priority revision is to distinguish the implemented model from the published specification, followed by narrowing the Bayesian-checking and sequential-boundary attributions. The remaining references can largely be retained with more precise surrounding prose.
