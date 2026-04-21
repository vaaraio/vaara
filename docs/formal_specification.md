# Vaara Adaptive Risk Scoring — Formal Specification

**Version**: 0.1.0
**Date**: 2026-04-10

## 1. Problem Statement

Let an AI agent **A** operate in an environment **E** by executing a sequence of actions **a₁, a₂, ..., aₜ** drawn from an action space **A**. Each action **aₜ** has a true but unknown risk **r*(aₜ) ∈ [0, 1]**, where 0 denotes no harm and 1 denotes catastrophic harm.

**The compositional safety problem**: Individual actions may be safe — r*(aᵢ) ≈ 0 for all i — yet the *sequence* (a₁, ..., aₖ) produces compound risk R*(a₁, ..., aₖ) ≫ max{r*(aᵢ)}. Example: {read_data, export_data, delete_data} is a data exfiltration pattern where each step alone is benign.

**Goal**: Construct an adaptive risk scorer **f: A × H → [0, 1]** and a conformal prediction set **C(aₜ) ⊆ [0, 1]** such that:

1. The scorer learns from outcomes: f improves over time as action consequences are observed.
2. The prediction set provides a distribution-free coverage guarantee: P(r*(aₜ) ∈ C(aₜ)) ≥ 1 − α for any α ∈ (0, 1), regardless of the underlying distribution of risks.
3. Temporal sequences are scored compositionally, not independently.

## 2. Action Space and Taxonomy

### 2.1 Action Classification

An action type **τ ∈ T** is characterized by a tuple:

    τ = (c, v, b, u, R)

where:
- **c ∈ {financial, data, comm, infra, identity, governance, physical}** — category
- **v ∈ {fully, partially, irreversible}** — reversibility
- **b ∈ {self, local, shared, global}** — blast radius
- **u ∈ {deferrable, timely, immediate, irrevocable}** — urgency class
- **R ⊂ {EU_AI_Act, GDPR, MiFID2, DORA, NIS2, ...}** — regulatory domains

### 2.2 Base Risk Score

The base risk score **β(τ)** is a deterministic function of static metadata:

    β(τ) = min(1, (s_v(v) + s_b(b) + s_u(u)) / Z)

where Z = 1.6 is a normalization constant, and:

    s_v(fully) = 0.1,  s_v(partially) = 0.4,  s_v(irreversible) = 0.8
    s_b(self) = 0.0,    s_b(local) = 0.1,      s_b(shared) = 0.3,      s_b(global) = 0.5
    s_u(deferrable) = 0.0, s_u(timely) = 0.1, s_u(immediate) = 0.2, s_u(irrevocable) = 0.3

**Proposition 2.1**: β(τ) ∈ [0.0625, 1.0] for all valid action types. The minimum is achieved by (fully, self, deferrable) and the maximum by (irreversible, global, irrevocable).

## 3. Expert Risk Signals

The scorer maintains **K** expert signals, each producing a risk estimate sₖ(aₜ, hₜ) ∈ [0, 1], where hₜ is the observable history up to time t.

### 3.1 Default Expert Set (K = 5)

1. **Taxonomy base** (s₁): s₁(aₜ) = β(τ(aₜ)) — the static base risk from §2.2.

2. **Agent history** (s₂): Blends the agent's denial rate and bad outcome rate:

       s₂(aₜ) = 0.3 · (denied/total) + 0.7 · (bad_outcomes/total)

   with a new-agent risk premium that decays as d(n) = 0.2 · (1 − min(1, n/100)), where n = total actions.

3. **Sequence pattern** (s₃): For each dangerous pattern **p = (p₁, ..., pₘ)** with risk boost **δₚ**, if the last w actions contain p as an ordered subsequence:

       s₃(aₜ) = max{δₚ : p matches in window}

4. **Action frequency / burst detection** (s₄): Let n_w be the count of actions by this agent in the last W seconds (default W = 60s). With threshold T = 10:

       s₄ = 0                           if n_w ≤ T/2
       s₄ = 0.5 · (n_w − T/2) / (T/2)  if T/2 < n_w ≤ T
       s₄ = min(0.9, 0.5 + 0.1·(n_w−T)) if n_w > T

5. **Confidence gap** (s₅): Given agent-reported confidence c ∈ [0, 1]:

       s₅ = max(0, c − (1 − β(τ)))

   A high-confidence claim on a high-base-risk action is suspicious.

## 4. Multiplicative Weight Update (MWU)

### 4.1 Formulation

We maintain a weight vector **w ∈ Δ_K** (the K-simplex) over experts. The combined risk score at time t is:

    f(aₜ) = Σₖ wₖ · sₖ(aₜ)

After observing outcome r*(aₜ), each expert incurs loss:

    ℓₖ(t) = |sₖ(aₜ) − r*(aₜ)|

Weights update multiplicatively:

    w̃ₖ(t+1) = wₖ(t) · exp(−η · ℓₖ(t))
    wₖ(t+1) = max(w_min, w̃ₖ(t+1)) / Σⱼ max(w_min, w̃ⱼ(t+1))

where η > 0 is the learning rate and w_min > 0 is the minimum weight floor preventing expert death.

**Theorem 4.1** (Regret bound, Arora et al. 2012): For any expert k, the cumulative regret after T rounds satisfies:

    Σₜ ℓ_MWU(t) − Σₜ ℓₖ(t) ≤ (ln K)/η + η·T/2

Setting η = √(2 ln K / T) gives O(√(T ln K)) regret, meaning MWU converges to the best expert at rate O(√(ln K / T)).

### 4.2 Cold Start

Before any outcomes are observed (t = 0), weights are uniform: wₖ = 1/K. The scorer operates in "rule-based" mode using the taxonomy base score as the dominant signal. As outcomes arrive, MWU shifts weight toward experts that actually predict bad outcomes.

## 5. Conformal Prediction

### 5.1 Split Conformal Intervals

Given a calibration set D_cal = {(f(aᵢ), r*(aᵢ))}ᵢ₌₁ⁿ of (prediction, outcome) pairs, define the nonconformity scores:

    eᵢ = |f(aᵢ) − r*(aᵢ)|

Sort them: e₍₁₎ ≤ e₍₂₎ ≤ ... ≤ e₍ₙ₎. The conformal quantile at level 1 − α is:

    q̂ = e₍⌈(1−α)(n+1)⌉₎

**Theorem 5.1** (Vovk et al. 2005): If the calibration points are exchangeable with the test point, then:

    P(r*(a_{n+1}) ∈ [f(a_{n+1}) − q̂, f(a_{n+1}) + q̂]) ≥ 1 − α

No distributional assumptions. No model retraining.

### 5.2 FACI Adaptive Alpha

Under distribution shift (non-exchangeability), we use Fully Adaptive Conformal Inference (Gibbs & Candès 2021). The miscoverage rate α_t adapts online:

    err_t = 𝟙{r*(aₜ) ∉ C(aₜ)}       (coverage error indicator)
    α_{t+1} = α_t + γ · (α − err_t)   (gamma-step update)

where γ > 0 is the step size. This provides long-run approximate coverage:

    (1/T) Σₜ err_t → α   as T → ∞

even under arbitrary distribution shift, at the cost of wider intervals when the distribution is adversarial.

### 5.3 Decision Rule

Let C(aₜ) = [f(aₜ) − q̂, f(aₜ) + q̂] be the conformal interval. The decision uses the **upper bound**:

    decision(aₜ) = ALLOW      if upper(C(aₜ)) < θ_allow
                 = ESCALATE   if θ_allow ≤ upper(C(aₜ)) ≤ θ_deny
                 = DENY       if upper(C(aₜ)) > θ_deny

This is **conservative by construction**: the decision is based on the worst-case risk within the 1 − α confidence set. If even the worst case is safe, allow. If even the best case is dangerous, deny. Between: escalate for human judgment.

**Proposition 5.2**: Let θ_allow = 0.3 and θ_deny = 0.7 (defaults). Before calibration, the interval is [f(a) − 0.3, f(a) + 0.3]. A raw score of f(a) ≤ 0.0 is needed for auto-allow, and f(a) ≥ 0.4 for auto-deny. This means the pre-calibration scorer is maximally cautious — most actions will be escalated, which is the correct cold-start behavior.

## 6. Temporal Sequence Scoring

### 6.1 Subsequence Pattern Matching

A dangerous pattern **p = (p₁, p₂, ..., pₘ)** is an ordered tuple of action names. Pattern p *matches* in history h = (h₁, ..., hₙ) if there exist indices i₁ < i₂ < ... < iₘ ≤ n such that h_{iⱼ} = pⱼ for all j.

The matching is **ordered but not contiguous** — interleaving benign actions does not break the match. This catches evasion attempts where an attacker inserts harmless actions between the dangerous ones.

**Complexity**: Subsequence matching is O(n·m) per pattern per action, where n = window size and m = pattern length. With W patterns each of length ≤ M, total cost per action is O(W·n·M). With default values (W = 7, n = 10, M = 3), this is O(210) — negligible.

### 6.2 Compositional Risk

The sequence risk boost δₚ is additive to the base score, not multiplicative:

    f_seq(aₜ) = f(aₜ) + max{δₚ : p matches}

This ensures that a benign action (f(a) = 0.1) in a dangerous sequence (δ = 0.5) is scored as 0.6 — elevated but not automatically blocked. The conformal interval then widens to [0.3, 0.9], which triggers escalation.

## 7. Regulatory Compliance

### 7.1 Evidence Mapping

Each audit record **r** is tagged with a set of regulatory articles **A(r) ⊆ Articles**. The compliance engine evaluates coverage:

    Coverage(article) = |{r : article ∈ A(r)}| / min_evidence

An article has **sufficient evidence** if Coverage ≥ 1 and evidence is not stale (age < staleness_hours). **Partial** if some evidence exists but insufficient. **Insufficient** if no evidence. This is an evidence-status classification, not a compliance verdict — the deployer (and, where required, a Notified Body) makes the conformity determination.

### 7.2 Hash Chain Integrity

Records form a hash chain:

    h₀ = ""
    hₜ = SHA256(canonical_json(rₜ) || hₜ₋₁)

where canonical_json uses sorted keys and no whitespace. **Proposition 7.1**: Tampering with any record rᵢ (i < t) changes hᵢ, which cascades to hₜ. Chain verification is O(n).

This satisfies EU AI Act Article 12(1) requirement for tamper-evident logging and DORA Article 12(1) requirement for ICT incident detection with automated alert mechanisms.

## 8. Convergence Properties

### 8.1 MWU Convergence Rate

With K = 5 experts and η = 0.1:
- After T = 100 outcomes: regret ≤ √(2 · ln(5) · 100) / 100 ≈ 0.18 per round
- After T = 1000 outcomes: regret ≤ 0.057 per round
- The scorer converges to within 6% of the best expert within 1000 observations.

### 8.2 Conformal Coverage Guarantee

With n = 30 calibration points (minimum) and α = 0.10:
- Finite-sample coverage: ≥ 1 − α − 1/(n+1) = 0.868 (worst case with 30 points)
- With n = 100: ≥ 0.890
- With n = 1000: ≥ 0.899 (approaching 0.90 asymptotically)

The FACI adaptive alpha further tightens this under stationarity and maintains approximate coverage under shift.

### 8.3 Cold Start Duration

The system transitions from rule-based to calibrated mode after 30 outcomes (min_calibration). At 10 actions/hour (moderate agent activity), calibration takes ~3 hours. During cold start:
- MWU weights are uniform → f(a) ≈ (s₁ + s₂ + s₃ + s₄ + s₅) / 5
- Conformal interval is [f(a) − 0.3, f(a) + 0.3] → wide, conservative
- Most actions route through ESCALATE → human review
- This is **correct behavior**: new systems should have high human oversight

## 9. Security Properties

### 9.1 Evasion Resistance

**Claim**: An adversarial agent cannot reliably evade the scorer by manipulating observable signals.

- s₁ (taxonomy) is immutable per action type — the agent cannot change what "tx.transfer" means.
- s₂ (history) is computed from the agent's own track record — the only way to improve it is to behave well.
- s₃ (sequence) uses non-contiguous subsequence matching — interleaving benign actions doesn't help.
- s₄ (burst) counts actions in a time window — cannot be gamed without slowing down.
- s₅ (confidence) penalizes overcalibrated confidence claims — lying about confidence increases risk.

The remaining attack vector is **outcome manipulation**: if the agent can influence the reported outcome severity, it can bias MWU weights. Defense: outcomes should be reported by an independent monitoring system, not by the agent itself.

### 9.2 Fail-Closed Design

If any component fails:
- Registry returns UNKNOWN_ACTION (medium risk) for unclassified tools
- Scorer returns score based on available signals (never returns "allow by default")
- Conformal interval widens when calibration is insufficient → more escalations
- Audit trail detects its own corruption via hash chain

The system never silently degrades to "allow everything."

## References

1. Arora, S., Hazan, E., & Kale, S. (2012). The multiplicative weights update method: a meta-algorithm and applications. *Theory of Computing*, 8(1), 121-164.

2. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.

3. Angelopoulos, A. N., & Bates, S. (2022). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning*, 16(4), 494-591.

4. Gibbs, I., & Candès, E. (2021). Adaptive conformal inference under distribution shift. *NeurIPS 2021*.

5. European Parliament. (2024). Regulation (EU) 2024/1689 (EU AI Act). *Official Journal of the European Union*.

6. European Parliament. (2022). Regulation (EU) 2022/2554 (DORA). *Official Journal of the European Union*.

7. Microsoft. (2026). Agent Governance Toolkit. https://github.com/microsoft/agent-governance-toolkit.
