# Phase 6 — Prior-art matrix for the new candidate directions

This file adds to `records.md`, which holds the 64 phase-2 records. It also adds to the works already cited in `04_mathematical_framework.md` and `05_revised_framework.md`. Together that is more than 100 publications reviewed across phases 2–6.

## Evidence levels (as in `records.md`)

| Level | Meaning |
|---|---|
| **F** | Full text read. Here that means the HTML version was read by a fetch agent, and the parts read are stated. |
| **A** | Abstract, official page or bibliographic record read in this session. |
| **K** | Well-known foundational work, described from established knowledge. It must be re-read before it is cited in a manuscript. |

## Rules applied

- Nothing below is inferred beyond what the evidence level allows.
- Where a theorem is quoted, it comes from text that was actually read.
- Columns:
  - **Diff-A** is the exact difference from Contribution A (covert mission capacity, `06_*.md` §5).
  - **Diff-B** is the exact difference from the PS-VTS detector (§6). That detector is a candidate that failed its pre-registered superiority test.
  - Other candidates are referenced by their number C1–C12.

## 1. Covert communication and steganography: square-root laws

| ID | Work | Ev. | Problem / formulation | Assumptions | Guarantee / result | Evidence | Limitations | Diff-A |
|---|---|---|---|---|---|---|---|---|
| P1 | Bash, Goeckel, Towsley, "Limits of reliable communication with low probability of detection on AWGN channels", IEEE JSAC 2013 (tech report UM-CS-2012-003; arXiv 1202.6423) | A | Covert bits in n channel uses under a KL/TV covertness constraint | AWGN; warden knows the noise law; shared secret | O(√n) covert bits; a converse beyond that | Theory | Information is measured in bits; i.i.d. noise | A maximises a **linear mean-displacement functional** (mission progress), not information. A also covers general covariance. |
| P2 | Bash, Goeckel, Towsley et al., "Covert communication gains from adversary's ignorance of transmission time" (arXiv 1403.1013; UM-CS-2014-001) | A | Covert bits when the warden does not know the start time | AWGN; T(n) possible start slots | ~√(n log T(n)) | Theory | Bits; i.i.d. | This is the mechanism behind A4's log price for unknown onset. A4 applies it to mean-displacement under general covariance. |
| P3 | Bouette, Luzzi, Wang, "Covert communication over two types of additive noise channels", arXiv 2303.11697 | **F** (Thm 1 read verbatim) | Covert nats over Gaussian noise with arbitrary invertible covariance Σ_n | Σ_n known, invertible | "L=1 irrespectively of μ_n and Σ_n": the SRL constant **does not depend on covariance** | Theory | Information functional | **Key contrast.** For information the covariance is irrelevant. For mean displacement, A shows capacity = (z_{1−α}+z_β)·SD_benign(wᵀX), which **depends entirely on covariance** and scales as T^H under long-range dependence (LRD). |
| P4 | Ker, "A capacity result for batch steganography", IEEE SPL 2007 | A | Secure payload against pooled steganalysis | Uniform cover capacity; quantitative steganalyser | Payload ∝ √(number of covers) | Theory + experiments | Cover model | Same √ structure as A1 for i.i.d. benign data. A generalises the covariance and the functional. |
| P5 | Filler, Ker, Fridrich, SRL for Markov covers (SPIE 2009) | A | SRL for stationary Markov covers | Markov cover; independent embedding changes | √ law holds | Theory | Short memory only | A shows the √ law **fails** under LRD (T^H) and under a finite reference (linear). |
| P6 | Soltani, Goeckel, Towsley, Houmansadr, "Fundamental limits of covert packet insertion", IEEE TCOM 2020 (doi 10.1109/TCOMM.2020.2980788; arXiv 1903.11640) | A | Covert packet insertion into a Poisson/renewal flow | Poisson or renewal timings of rate λ | ~√(λT) packets undetectable | Theory | Single flow; i.i.d. renewal | This is A's Poisson special case (count progress). A adds general covariance, the reference/nuisance law and the panel residual. |

## 2. Quickest change detection with covert or stealthy adversaries

| ID | Work | Ev. | Problem / formulation | Assumptions | Guarantee / result | Evidence | Limitations | Diff-A |
|---|---|---|---|---|---|---|---|---|
| P7 | Ramtin, Nain, Towsley, "Quickest change detection in continuous-time in presence of a covert adversary", arXiv 2509.17778 | **F** | Adversary picks the post-change drift μ(γ) to maximise damage D(γ) = μ(γ)·ADD | Pre-change is standard Brownian motion; the reference is fixed; one stream | Maximal damage when μ = Θ(1/√γ), giving D = Θ(√γ) | Theory + numerical example | No dependence, no adaptive reference, no panels, no covariance | **Closest prior art.** It establishes the √ law for a cumulative-damage functional when pre-change data are i.i.d. A is the extension to: general Gaussian covariance (A1), LRD T^H (A2), finite reference with unknown offset giving a linear law (A3), and peer residuals (A5). A1's exact finite-T saddle point is also not in P7. |
| P8 | Ramtin, Nain, Towsley, "Quickest change detection in discrete-time in presence of a covert adversary", arXiv 2601.20022 | **F** (first ~40k characters) | CuSum; adversary picks the post-change law q_γ; damage d(γ) = n(γ)·g(KL) | i.i.d. pre-change law q, fixed | ADD Θ(γ) under covertness; damage scaling (theorem not read verbatim) | Theory | i.i.d. only | Same as P7. |
| P9 | Lo & Bloch, "Covert Bayesian quickest change detection", arXiv 2605.16140 | A | The *detector* must hide its probing from an adversary | DMC; Bayesian | Square-root-order gain in log false-alarm probability | Theory | Roles reversed | Different problem (covert sensing). Cited only to show that √ phenomena in QCD are known. |
| P10 | Bai, Pasqualetti, Gupta, "Data-injection attacks in stochastic control systems: detectability and performance tradeoffs", Automatica 82:251–260, 2017 | **F** (Def. 1, Thm 1) | ε-stealthiness defined through the KL rate; attacker maximises Kalman-filter error covariance | LTI Gaussian; ergodic | Stealthy iff the KL rate ≤ ε; log-type degradation bounds (summary only) | Theory | Estimation-error functional; KL *rate* (asymptotic) | Shares the mathematical style (KL-budgeted attack). A uses a linear progress functional with a finite-horizon exact NP value and the T^H, reference and panel laws. The read text has no √(2ε)·SD statement. |
| P11 | Kung, Dey, Shi, "The performance and limitations of ε-stealthy attacks on higher order systems", IEEE TAC 62(2):941–947, 2017 | A | ε-stealthy attacks on vector systems | LTI | Damage can be arbitrarily large in vector systems | Theory | Control damage | It parallels A3's message: some structures give unbounded covert damage. The mechanism differs (system dynamics, not reference length). |
| P12 | Mo & Sinopoli, stealthy integrity attacks on CPS (IEEE TAC 2016) | K | Reachable set under a χ² detector | LTI Gaussian | Ellipsoidal reachable set | Theory | Detector-specific | A is detector-independent (NP converse). |
| P13 | Kloft & Laskov, "Security analysis of online centroid anomaly detection", JMLR 13:3681–3724, 2012 | A | Poisoning an online centroid detector with infinite or finite window | Hypersphere detector; attacker controls a fraction of traffic | Bounds on centroid displacement per attack point; finite-window case | Theory + HTTP experiments | One detector family; geometric | Qualitatively related to A3 (a finite window allows progressive displacement). A3 is a **detector-independent** statistical lower bound (any test, NP converse) with an explicit rate z·√(m/N_ref). |
| P14 | Caulfield, "A simple data exfiltration game", arXiv 2509.06571, 2025 (sole author per a third-party record; to verify) | A | Defender sets per-link thresholds at multiples of SD; attacker chooses duration | Per-link variance | Lower variance forces longer exfiltration | Game model | Fixed threshold rule; no dependence | A gives the detector-independent optimum and the dependence laws. P14 is the closest *security-domain* statement of "attackers hide in variance". |

## 3. Network-security detectors used as baselines or comparators

| ID | Work | Ev. | Problem / formulation | Guarantee / evidence | Limitations | Diff-A / Diff-B |
|---|---|---|---|---|---|---|
| P15 | Jung, Paxson, Berger, Balakrishnan, "Fast portscan detection using sequential hypothesis testing", IEEE S&P 2004, doi 10.1109/SECPRI.2004.1301325 | A | Threshold random walk (SPRT) on connection success/failure | Error bounds; detects after 4–5 attempts | i.i.d. Bernoulli model | A explains why scan-rate limiting succeeds only up to √(λT) for benign-rare events. PS-VTS is not a scan detector. |
| P16 | Zou, Gao, Gong, Towsley, "Monitoring and early warning for Internet worms", ACM CCS 2003, pp. 190–199 | A | Kalman filter on illegitimate scan counts; "detect the trend, not the rate" | Simulations (Code Red, Slammer) | Epidemic model | Closest work for C10 (precursors). |
| P17 | Ho et al., "Hopper: modeling and detecting lateral movement", USENIX Security 2021, pp. 3093–3110 | A | Login-graph path inference + rules + anomaly score | 94.5% detection over 300+ simulated scenarios, < 9 alerts/day, 15-month enterprise data | Login data | Closest work for C3 (propagation direction). |
| P18 | King & Huang, "Euler: detecting network lateral movement via scalable temporal link prediction", NDSS 2022 (extended ACM TOPS 2023) | A | GNN + sequence model; anomalous edges | Precision 0.243 → 0.986 with an alert layer (journal abstract) | Learned model; no FA guarantee | C3 comparator. |
| P19 | Soule et al., "Combining filtering and statistical methods for anomaly detection", IMC 2005 | A | Kalman residual traffic + GLRT and other alerting rules | Empirical | Traffic-matrix level | Residual + GLRT is the established idea behind "median-resid+VTS". |
| P20 | Luzio & Antonelli Ponti, "Real-time anomaly detection with synthetic anomaly monitoring (SAM)", arXiv 2501.18417 | A | Synthetic control per feature; residual score | IoMT traffic | Claims to be the first SC-based anomaly detection; no FA guarantee | Peer/synthetic-control residualisation is **established**. PS-VTS does not claim it. |
| P21 | Bolton & Hand, peer group analysis (Credit Scoring & Credit Control VII, 2001); Weston, Hand et al. (plastic card fraud) | A | Compare each account with its most similar peers over time | Fraud applications | Heuristic thresholds | Cross-sectional peer comparison is **established**. That is the core of PS-VTS's studentisation. |

## 4. Statistics underpinning A and B

| ID | Work | Ev. | Content used | Diff |
|---|---|---|---|---|
| P22 | Leland, Taqqu, Willinger, Wilson, "On the self-similar nature of Ethernet traffic (extended version)", IEEE/ACM ToN 2(1):1–15, 1994, doi 10.1109/90.282603 | A | Benign LAN traffic is statistically self-similar, and aggregation intensifies it | Empirical premise of A2. It does not concern detection limits. |
| P23 | Horváth & Kokoszka, CUSUM under long-range dependence (J. Statist. Plann. Inference 1997); Kokoszka & Leipus 1998 | A (via later papers) | Limit laws and estimator consistency for change points under LRD | Asymptotic estimation, not adversarial capacity. |
| P24 | Berkes, Horváth, Kokoszka, Shao, "On discriminating between long-range dependence and changes in mean", Ann. Statist. 34(3), 2006, doi 10.1214/009053606000000254 | A | LRD and mean changes are hard to tell apart with CUSUM | **Supports A2/A3**: it is the statistical root of why LRD helps attackers. A turns it into a capacity law. |
| P25 | Dehling, Rooch, Taqqu, power of change-point tests for LRD data (arXiv 1303.4917) | A | ARE of CUSUM against Wilcoxon under LRD | Power analysis for fixed tests. A is a detector-independent converse. |
| P26 | Arias-Castro, Candès, Durand, "Detection of an anomalous cluster in a network", Ann. Statist. 39(1):278–304, 2011, doi 10.1214/10-AOS839 | A | Minimax detection rate for clusters; scan statistics are rate-optimal | Basis of the scan price in A4 and B. i.i.d. N(0,1) null. |
| P27 | Neyman & Pearson 1933; Huber 1965 (robust testing); Le Cam | K | NP lemma; least-favourable pairs | A1's converse is a direct consequence of NP (stated honestly in §5). |
| P28 | Gao, Xie, Xie, Xu, "Robust hypothesis testing using Wasserstein uncertainty sets", NeurIPS 2018, pp. 7902–7912 | A | Minimax tests over Wasserstein balls | C5 comparator. |
| P29 | Orlitsky, Suresh, Wu, "Optimal prediction of the number of unseen species", PNAS 113(47):13283–13288, 2016 | A | Smoothed Good–Toulmin is minimax-optimal up to t ∝ log n | C4 comparator. |
| P30 | Anagnostopoulos, Kumar, Mahdian, "Influence and correlation in social networks", KDD 2008, pp. 7–15 | A | Shuffle test and edge-reversal test separating influence from correlation | C3 is an **adaptation** of the edge-reversal test to initiator direction in lateral movement. |
| P31 | Shalizi & Thomas, "Homophily and contagion are generically confounded…", Sociol. Methods Res. 2011 | K | Contagion and latent homophily are not identifiable in observational network data | Identifiability limit for C3. |
| P32 | Gomez-Rodriguez, Leskovec, Krause, NetInf, KDD 2010 | K | Inferring diffusion networks from cascades | C3 comparator. |
| P33 | Peters, Bühlmann, Meinshausen, invariant causal prediction, JRSSB 2016 | K | Invariance across environments | C6 is an **adaptation**. |
| P34 | Chizat, Peyré, Schmitzer, Vialard, unbalanced optimal transport, Math. Comp. 2018 | K | Transport plus mass creation and destruction | C5 basis. |
| P35 | Scheffer et al., "Early-warning signals for critical transitions", Nature 2009 | K | Rising variance and autocorrelation before tipping | C10 basis. |

## 5. Other representations surveyed for candidates C7–C12

| ID | Work | Ev. | Finding | Relevance |
|---|---|---|---|---|
| P36 | Koopman-mode attack identification in power grids (arXiv 2007.11717) and EV charging (arXiv 2406.16780) | A | Koopman/DMD used for CPS attacks; no IT-NIDS work found in one search | C7 would be an **adaptation** to NIDS. Absence from one search is not evidence of originality. |
| P37 | "A review of topological data analysis for cybersecurity", arXiv 2202.08037; Bruillard, Nowak, Purvine 2016 (cited therein) | A | Persistent homology already used for scans and DDoS | C9 is **established / adaptation**. |
| P38 | Patel et al., "On the evaluation of spiking neural network configurations for network intrusion detection", arXiv 2606.01442, 2026 | A | 27 SNN configurations; encoding matters more than neuron model | C12 is **established** as a model family. Only efficiency could be new. |
| P39 | Hierarchical D-SNN for lifelong NIDS, arXiv 2508.04610 | A | Two-stage SNN | C12 comparator. |
| P40 | Self-similarity / Hurst-based DDoS and anomaly detectors (e.g. ResearchGate 309222950; CEUR-WS Vol-4223 paper 6; J. Netw. Syst. Manag. 2026, doi 10.1007/s10922-026-10069-y) | A | Hurst estimates used *as features* for detection | These use H as a signal. A uses H to bound what **any** detector can see. The two are distinct. |
| P41 | Clegg et al., "A critical look at power law modelling of the Internet", arXiv 0910.2140 | A | High Ĥ can come from non-stationarity | **Limitation for A2**: H must be estimated robustly. A3 shows non-stationarity (finite reference) causes linear capacity anyway. |

## 6. Searches that found nothing

These are recorded so nobody reads them as originality claims.

- No paper was found applying the square-root law, or any covert-capacity notion, to **intrusion missions under long-range dependent benign traffic**. Two extended searches and three standard searches were run.
- No paper was found on **IT-network** Koopman/DMD intrusion detection (one search).

Absence from a search is **not** evidence of originality. Before any claim reaches a manuscript, the following are still needed:
- full-text reading of the remainder of P8 (the final ~37k characters were unread);
- P10's Section 4 onwards;
- P13's finite-window theorems;
- the covert-communication-with-memory literature for *linear* functionals;
- the SPC literature on estimated-parameter effects. This includes self-starting charts and Phase-I/Phase-II analysis, e.g. Jones, Champ & Rigdon 2001 (K, to verify). It is the closest statistical prior art to A3.
