# PNC P&C Subrogation Payments Strategy
## Version 2

**PNC Bank · Treasury Management | Data Science**

---

## What changed in Version 2

Version 1 established the strategic thesis and a strong casualty operational primer. Version 2 keeps that spine intact and closes the gaps between the *strategy as written* and the *capability as it exists today*, grounded in what the Payment Knowledge Graph can and cannot currently see.

| Area | V1 | V2 change |
|---|---|---|
| **Detection feasibility** | Assumed complete | New Section 3 — the graph is PNC↔PNC today; detection is real but partial, and unlocks in tiers as counterparty data lands |
| **The Match layer** | Named as central, not specified | New Section 5 — specifies the actual matching problem, data sources, and why Arbitration Forums' structured demand data is the key that unlocks it |
| **Detection method** | Per-line "signatures" only | New Section 6 — a concrete detection methodology tying the signatures to graph algorithms already prototyped |
| **Arbitration Forums** | Treated as partner | New Section 10 — AF is *also* an overlap (its Settlement Exchange already does electronic settlement); sharpens where PNC is complementary vs. competitive |
| **Measurement** | Absent | New Section 8 — KPIs per phase, and an explicit validation dependency (no ground truth today) |
| **Governance / risk** | Absent | New Section 9 — data-commercialization, prospect-intelligence, and AML-overlap governance |
| **Sequencing** | Phases read as parallel | New Section 11 — phases mapped to the data-availability timeline, with a concrete first deliverable |
| **Operational primer** | — | Preserved (Section 7), lightly tightened |

---

## 1. Executive recommendation (revised)

PNC should treat casualty subrogation as a specialized **receivables, settlement, and liquidity vertical** — not as a new claims-administration business.

The strategic sequence remains:

- **Detect** — identify subrogation payment corridors, entities, and operational friction in the B2B Payment Knowledge Graph.
- **Match** — become the cash-application control plane that associates every recovery payment with the correct claim, demand, insured deductible, reinsurer share, and accounting entry.
- **Orchestrate** — add verified settlement instructions, payment routing, virtual accounts, and bilateral settlement.
- **Finance** — extend credit against acknowledged or adjudicated recovery receivables — not speculative, contested tort claims.
- **Benchmark** — commercialize privacy-safe recovery-velocity and payment-efficiency analytics.

The initial product remains a **Subrogation Receivables Control Tower** layered on PNC's existing integrated receivables, lockbox, virtual-account, claims-payment, API, and real-time-payment capabilities.

**What V2 makes explicit:** these five phases are a *dependency chain, not a parallel program.* Match depends on data integration that does not exist today. Finance depends on Match. Detection itself is currently partial and widens in stages as counterparty data lands. Sections 3, 5, and 11 make this sequencing concrete so leadership funds it in the right order and does not expect corridor-level completeness on day one.

The central strategic principle is unchanged and correct:

> **Own the cash truth layer, not the liability-adjudication layer.**

---

## 2. The strategic principle

Adjudication — who was at fault, in what percentage, subject to what coverage — is already served by claims platforms, recovery vendors, and Arbitration Forums. Rebuilding it is a losing proposition.

What no single party owns end-to-end is the **join between a recovery payment and everything it means**: which paid loss it reduces, which deductible it returns, which reinsurer shares it, which contingency fee it triggers, which arbitration award it satisfies, and whether it closes the underlying claim. The claims system knows the claim but not the bank-ledger truth. The bank sees the credit but not the claim. Arbitration Forums sees its own network's settlements but not the large volume of subrogation that flows outside it.

PNC is uniquely positioned to own that join because it already holds the account, the rails, and — for its carrier customers — the settlement relationship. That is the cash truth layer. Everything else in this strategy is built on it.

---

## 3. Data reality: what we can detect today *(new)*

The strategy is only as strong as the "Detect" phase that seeds it, so leadership must understand precisely what the graph sees now.

### 3.1 The current graph is PNC-customer-to-PNC-customer

The Payment Knowledge Graph today encodes **PAYS** edges between PNC customers only. Counterparty (outside-PNC) data is arriving in stages. This has direct consequences for subrogation detection:

- The strongest structural signal — **reciprocity between two carriers** (A pays B *and* B pays A across the window) — is only fully observable when **both** carriers bank at PNC. That is a minority of the carrier-to-carrier market.
- A recovery paid by a PNC carrier customer to a non-PNC carrier is currently **invisible** (that is outbound counterparty flow, not yet ingested).

Framed honestly: today's structural detection surfaces the *subset* of subrogation where both sides are PNC customers, plus — importantly — the inbound flows described next.

### 3.2 The asymmetry, and its silver lining

Counterparty data is asymmetric in availability, and the asymmetry favors the financing thesis:

| Direction | Data asset | Status | What it reveals for subrogation |
|---|---|---|---|
| Counterparty → PNC customer (inbound) | `CPTY_PAYS` | ✅ Ingested | External carriers paying **recoveries into** a PNC carrier customer |
| PNC customer → counterparty (outbound) | `PAYS_CPTY` | ⏳ Upcoming | Recoveries **paid out** by PNC customers to external carriers |
| Counterparty ↔ its bank | `CptyFinEntity` | 🔭 Future | Which external institutions hold the carrier counterparties (interbank view) |

The inbound direction is available **now**. Inbound recoveries are exactly the receivable the **Finance** phase intends to lend against. So the financing case has near-term data support *before* full bidirectional corridor visibility exists — a stronger near-term story than V1 implied, provided we do not overstate corridor completeness.

### 3.3 Detection unlocks in three tiers

- **Tier 0 (now — PAYS + CPTY_PAYS):** PNC↔PNC reciprocal carrier corridors + inbound external recoveries. Sufficient to stand up a candidate-corridor registry and the MVP (Section 11).
- **Tier 1 (PAYS_CPTY lands):** full bidirectional corridors, including outbound recoveries — complete carrier-to-carrier maps.
- **Tier 2 (CptyFinEntity lands):** interbank flow intelligence — which external banks sit behind the carrier counterparties, informing correspondent and settlement strategy.

**Recommendation:** present detection to leadership as a *staged* capability with these three tiers, not as a finished corridor map. Overselling completeness at Tier 0 is the fastest way to lose credibility with the fraud, liquidity, and correspondent teams who know the counterparty data is still arriving.

---

## 4. The five-phase sequence, mapped to data availability *(revised)*

| Phase | Core question | Data prerequisite | Earliest start |
|---|---|---|---|
| **Detect** | Where is subrogation happening, and between whom? | PAYS + CPTY_PAYS (have) → PAYS_CPTY (widens) | **Now** (Tier 0) |
| **Match** | Which claim/demand does this recovery payment belong to? | Carrier claims integration **and/or** AF structured demand data | Parallel workstream, gated on integration |
| **Orchestrate** | Route and settle through PNC accounts/VAM/RTP | Match maturity + customer onboarding | After Match pilots |
| **Finance** | Lend against acknowledged recovery receivables | Match (to know a receivable is *acknowledged*) | After Match maturity |
| **Benchmark** | Sell privacy-safe recovery analytics | Governance + anonymization sign-off | After governance workstream (Section 9) |

The key correction over V1: **Match is not phase two in time — it is a parallel workstream that everything downstream depends on, and it is gated on data integration, not on detection.** Detection can proceed on graph data alone; Match cannot.

---

## 5. The Match layer is the moat *(new / expanded)*

V1 correctly identified that "ordinary invoice-based cash application performs poorly in this domain" but did not specify how PNC would do better. This section does.

### 5.1 The matching problem, precisely

A recovery credit landing in a PNC carrier customer's account typically carries:

- A **payer name** — often a TPA, recovery vendor, or law-firm trust account, frequently *not* the ultimate responsible carrier.
- An **amount** and a **date**.
- At most a short, unstructured **memo / addenda** line.

It does **not** carry the claim number, demand ID, deductible split, reinsurer share, LAE component, contingency-fee flag, or accounting classification. As Section 1.1 of the primer establishes, a single recovery payment can simultaneously be a paid-loss reduction, a deductible return, a reinsurer's share, and a partial settlement. None of that is legible from the bank credit alone. That is exactly why invoice-based cash application fails, and exactly the gap the Control Tower fills.

### 5.2 The two data joins that solve it

To associate a recovery credit with its claim, PNC needs one or both of:

1. **The carrier's own claims/recovery data** — the demand sent, the expected amount, the claim ID, the insured, the deductible owed. Requires the carrier customer to share claims/recovery-ledger data with its bank. (This is the central open question in Section 12.)
2. **Structured remittance metadata traveling *with* the payment** — richness varies sharply by channel:

| Channel | Remittance richness |
|---|---|
| ACH CCD+ | One 80-character addenda record — minimal |
| ACH CTX | Can carry structured ANSI X12 820 remittance — usable |
| Wire (OBI/BBI) | Free-text, unstructured |
| ISO 20022 (pain/pacs + remittance) | Structured remittance — the strategic target state |
| Check / lockbox | Paper stub, OCR — lossy |

### 5.3 Arbitration Forums is the key that unlocks Match

The single most valuable observation folded into V2: **the draft already lists AF's structured fields** — demander and responder claim numbers, policy numbers, AF demand IDs, loss details, party data, damages, liability, evidence, and remittance information. AF's Settlement Exchange System reportedly moves electronic payments carrying this metadata.

If PNC positions itself as the **settlement bank beneath AF's Settlement Exchange**, those structured identifiers arrive *with* the payment. Matching **AF demand ID → PNC bank credit** becomes a tractable, high-precision join — and it is a join neither the carrier's claims system (no bank-ledger truth) nor AF (no cross-network cash picture across non-AF flows) holds end-to-end. This is the moat, and it is reachable through partnership rather than rebuild.

### 5.4 Reuse the matching engine we already have

Where structured IDs are absent, association is probabilistic: payer-name fuzzy matching (Jaro-Winkler — already implemented in the self-payment detection pipeline, with first-token blocking and jellyfish/rapidfuzz scoring), amount tolerance, date windows, and exact claim-ID match when present. The Match engine is an extension of existing internal tooling, not a green-field build.

---

## 6. Detection methodology *(new)*

This bridges the strategy to the operational primer's per-line "payment-graph signatures," and reflects what has already been prototyped against the graph.

### 6.1 Entity layer — who is a subrogation participant

- **NAICS filtering (524x — Insurance Carriers & Related Activities):** 524126 (direct P&C carriers), 524114 (health), 524291 (claims adjusting), 524292 (third-party administration), 524298 (all other insurance-related — where subrogation clearinghouses tend to land).
- **EDA-driven code discovery:** rather than trusting a hardcoded list, surface the actual `(code, description)` pairs present in the data whose descriptions match `insur|casualt|reinsur|subrogat|claim|underwrit`, and pick from what is really there.
- **Name-pattern matching** for clearinghouses and recovery vendors (e.g. arbitration/subrogation-recovery entities) that a NAICS code alone may miss.

### 6.2 Structural layer — the signature on the graph

- **Reciprocity on the multi-month aggregated graph** — the primary signal. Bidirectional carrier-to-carrier flow is unusual for ordinary vendor relationships and characteristic of two carriers each subrogating against the other across a portfolio.
- **Clearinghouse hub detection** — a node with many small reciprocal edges to most carriers at once is a near-certain marker (the Arbitration-Forums pattern). These entities should be added to the hub/exclusion registry alongside PNC-internal accounts, processors, and payroll — not treated as actionable customers.
- **Degree / strength / persistence profiles** — including `n_rels` (how many months a pair recurs), a recurrence signal distinct from volume.

### 6.3 Discriminators — cutting the false positives

- **Average amount per transaction** — subrogation settlements cluster at *many moderate* amounts; **reinsurance treaty flows** are *few and large* and look structurally similar (carrier-to-carrier) but are a different business. This ratio is the primary discriminator between them.
- **Volume and frequency**, and **temporal persistence** across the window.

### 6.4 The temporal correction

Recovery lags the original claim payment by weeks to months. Reciprocity must therefore be evaluated on the **multi-month aggregate**, never per-month — same-month reciprocity would badly undercount real relationships. (This shaped the aggregated-window design of the detection pipeline.)

### 6.5 Detectability by line — sequence accordingly

Not all four casualty lines are equally detectable from payment structure. This directly informs sequencing:

| Line | Structural detectability | Why |
|---|---|---|
| **Commercial Auto** | **High — the wedge** | High transaction count, repeating carrier corridors, repeated demand IDs, heavy arbitration-platform use, bidirectional carrier relationships |
| Commercial Property | Medium | Lower frequency, higher severity, more wires and manual payments, more one-payment-many-components |
| Workers' Comp | Low — FP-prone | "Lien" descriptors, plaintiff-firm trust accounts, long durations; **must not be inferred from "claim"/"recovery" alone** |
| General Liability | Lowest | Law firms, confidential settlements, coverage-allocation matters mislabeled as subrogation |

**Sequence detection Auto → Property → WC/GL**, and treat WC/GL candidates as requiring corroboration (Match data or memo text), never structural inference alone.

### 6.6 The validation gap

There is **no transaction-level ground truth today** — no confirmed subrogation labels against which to measure precision. All Tier-0 detection is structural *inference* with unknown precision. Ground truth arrives from AF integration (labeled demands), carrier claims integration, or — if accessible upstream of the aggregated snapshot — ACH addenda / wire OBI keyword search. Establishing at least one label source is a prerequisite for trusting detection output, not a nice-to-have (Section 8).

---

## 7. Casualty subrogation operational primer *(preserved from V1, tightened)*

### 7.1 The three events that must remain separate

Subrogation is the substitution of one party — typically an insurer that has paid a loss — into the legal rights of the insured against a responsible third party. Data Science and Product must keep three events distinct:

- **Claim payment** — who paid the insured, claimant, provider, repairer, or other loss payee?
- **Liability determination** — who was legally responsible, in what percentage, subject to what coverage, contract, limit, waiver, or defense?
- **Recovery** — who ultimately reimbursed the paying carrier, self-insured, employer, or other party?

These can be separated by months or years, involve different legal entities, different claim numbers, and different bank accounts. The claim system may recognize the recovery at a very different time from the bank ledger.

A recovery payment is therefore not merely an accounts-receivable payment. It can simultaneously be: a reduction in paid loss; a return of the insured's deductible; a reimbursement of allocated loss-adjustment expense; a recovery belonging partly to a reinsurer; a payment subject to a recovery-vendor contingency fee; satisfaction of an arbitration award or negotiated settlement; or a partial settlement that does not close the underlying claim. That is why ordinary invoice-based cash application performs poorly — and why the Match layer (Section 5) is the product.

### 7.2 Commercial Auto

**Workflow.** (1) *First-party payment* — the carrier pays its policyholder under collision or other first-party physical-damage coverage: vehicle repair or ACV total loss, towing and storage, rental/loss-of-use, appraisal/inspection, less the deductible. Medical payments and some no-fault benefits may create separate recovery rights by coverage and jurisdiction. (2) *Liability investigation* — identify the adverse driver, owner, employer, and insurer; allocate fault under the jurisdiction's negligence rules; confirm coverage and limits; assess course-and-scope of employment; identify any additional responsible entity (leasing company, maintenance contractor, shipper, manufacturer). (3) *Demand creation* — demander/responder claim numbers, policy and insured information, loss date and location, liability narrative and evidence, damage documentation, amount demanded (including deductible and permitted expenses), and remittance/settlement instructions. AF workflows use these fields plus internal references and AF demand IDs. (4) *Negotiation or arbitration* — accept, deny, accept a comparative-negligence percentage, dispute damages, request evidence, negotiate, or proceed to intercompany arbitration under AF's Auto Agreement. (5) *Settlement and payment* — payment may go to the primary carrier, a subsidiary, a TPA, a recovery vendor, a law firm, or a designated lockbox; once received it must be associated with the proper claim and split with the insured where a deductible reimbursement is due.

**Payment-graph signature.** Commercial auto is the best initial line for graph detection: high transaction count; repeating carrier-to-carrier corridors; repeated claim and demand identifiers; standard damage categories; frequent arbitration-platform use; many modest settlements rather than only large litigation proceeds; and bidirectional payment relationships between major carriers (a large carrier is a net receiver against some counterparties and a net payer against others).

### 7.3 Commercial Property

**Workflow.** (1) *First-party indemnification* — covered physical damage plus potentially business interruption, extra expense, debris removal, equipment breakdown, inventory/contents, and mitigation/emergency services. (2) *Cause-and-origin investigation* — preserve evidence and determine whether the loss arose from defective work or product, contractor negligence, utility failure, fire origin, sprinkler/plumbing failure, landlord/tenant conduct, vehicle impact, or equipment malfunction. Unlike many auto losses, the existence and identity of a responsible third party may not be apparent when the first-party payment issues. (3) *Contract and rights analysis* — contractual indemnity, waivers of subrogation, additional-insured provisions, risk-transfer clauses, lease terms, construction contracts, product warranties, economic-loss doctrines, and spoliation. Waivers can prevent or narrow recovery even where another party appears factually responsible. (4) *Recovery* — against a liability carrier, contractor/subcontractor, manufacturer, landlord/tenant/property manager, utility, another insurer, or multiple tortfeasors, with AF's Property Agreement available for qualifying disputes.

**Payment-graph signature.** Lower transaction frequency than auto; higher average severity; more law-firm and expert involvement; more wires and manually authorized payments; more settlement agreements, releases, and allocations; greater probability of one payment covering numerous claim components; and longer intervals between loss, payment, demand, settlement, and cash receipt. A bank can identify the settlement payment but, without claims integration, often lacks the original paid-loss denominator and cannot infer true recovery performance — the Match gap made concrete.

### 7.4 Workers' Compensation

**Workflow.** (1) *Statutory benefit payment* — a primarily no-fault system paying medical expense, temporary/permanent disability, wage replacement, rehabilitation, and death benefits. (2) *Third-party action* — a separate recovery opportunity where someone other than the employer or coemployee is responsible (negligent driver, premises owner, equipment manufacturer, general contractor/subcontractor, other vendor). (3) *Lien or statutory recovery right* — WC statutes commonly grant the employer or carrier a lien, reimbursement right, or subrogation interest against third-party proceeds; details are highly state-specific and affect control of the action, notice/consent, attorney-fee allocation, recovery expenses, the employee's net recovery, the carrier's lien, future-credit rights, and settlement approval. (4) *Settlement distribution* — among injured employee, employee's counsel/trust account, WC carrier, employer/self-insured, medical lienholders, recovery counsel, and other statutory claimants. AF's Special Arbitration program includes certain WC subrogation disputes.

**Payment-graph signature.** Payments from plaintiff law-firm trust accounts; "lien"/"WC lien"/"comp lien" descriptors; multiparty distribution; long durations; irregular amounts; a mix of reimbursement and future-credit accounting; and TPA accounts acting for self-insured employers. **This line must not be inferred solely from "claim" or "recovery"** — those generate many false positives.

### 7.5 General Liability

**Workflow.** GL differs because the carrier ordinarily pays defense and indemnity on behalf of its insured against a third-party claim, rather than first-party damage to its own property. Post-payment recovery may arise through contribution from a joint tortfeasor, contractual indemnity from a contractor/vendor, additional-insured coverage, equitable subrogation against another insurer, allocation among primary/excess/umbrella carriers, other-insurance disputes, or recovery from a party that assumed the risk contractually. The operational label "subrogation" may thus include matters legal teams characterize as contribution, indemnification, or coverage allocation.

**Payment-graph signature.** Law firms or settlement administrators; large and infrequent wires; confidential settlement references; multiple carriers; primary-versus-excess allocations; defense-cost contribution; payments to or from attorney trust accounts; and separate coverage and liability disputes. AF's Special Arbitration program addresses multiple-tortfeasor, concurrent-coverage, and excess-versus-primary allocation questions.

---

## 8. Measurement & validation *(new)*

The program cannot be managed without metrics, and — critically — detection precision **cannot be measured today** for lack of ground truth. An early validation workstream is therefore a prerequisite, not an afterthought.

**Per-phase KPIs:**

- **Detect** — number of candidate corridors and entities identified; share of insurance-segment payment volume explained by the subrogation signature; *once labels exist*, precision and recall by line.
- **Match** — straight-through cash-application rate (% of recovery credits auto-associated to a claim/demand); exception rate; time-to-application.
- **Orchestrate** — verified settlement instructions; share of settlement volume routed through PNC accounts; VAM adoption among carrier customers.
- **Finance** — recovery receivables financed ($); advance rate; dilution/loss rate (contested or reversed recoveries); reduction in days-to-recovery.
- **Benchmark** — analytics-product adoption; anonymization-threshold compliance; governance sign-off status.

**Establishing ground truth (pick at least one before scaling detection):**
1. AF integration — labeled demands and settlements.
2. Carrier claims/recovery-ledger integration — direct labels.
3. Transaction memo / addenda keyword search, *if* accessible upstream of the aggregated snapshot.

Until one exists, all detection output ships with an explicit **confidence tier** and is described as structural inference, not confirmed subrogation.

---

## 9. Governance, compliance & risk *(new)*

The strategy touches sensitive territory that V1 did not address. Recommend a **parallel governance workstream owned outside Data Science** (legal, compliance, privacy).

- **Data-use boundaries (Benchmark phase).** Payment data collected to service one customer, then used to (a) build prospect intelligence on non-customers via counterparty nodes, and (b) sell benchmarking analytics, carries information-barrier, privacy, and potential contractual implications. This needs legal/compliance clearance *before* the Benchmark phase, and anonymization thresholds defined up front.
- **Prospect-intelligence sensitivity.** The counterparty graph is a powerful shadow prospect asset, but must be governed — who may access it, and how it may be used in sales — before it drives outreach.
- **AML overlap.** Reciprocal high-frequency carrier flows structurally resemble layering. Detection output is adjacent to AML monitoring. Coordinate with Fraud/AML so that formalizing subrogation patterns *reduces* false positives (by adding clearinghouses to the hub/exclusion registry) rather than creating a parallel, ungoverned monitoring capability.
- **Financing prudence.** Lend only against **acknowledged or adjudicated** receivables — but "acknowledged" is knowable only via Match (claims or AF integration). Credit policy is therefore gated on data-integration maturity; do **not** finance on structural inference alone.

---

## 10. Arbitration Forums: integration vs. overlap *(new)*

V1 treats AF purely as a partner. The relationship is sharper than that, and getting it right is central to the Match layer.

**Where AF already competes with "Orchestrate."** AF reportedly operates a Settlement Exchange System *with electronic payment capability* and broad carrier adoption. That overlaps PNC's ambition to route and settle. This is not green field — assume AF is already moving settlement dollars for its signatory network.

**Where PNC is genuinely complementary:**
- **Financing** — AF does not extend credit against recovery receivables. Clear white space.
- **Cash truth across *all* subrogation** — AF covers signatory carriers and qualifying auto/property/special matters. Large volume sits **outside** AF: non-signatories, direct negotiation, WC, GL, self-insureds, and law-firm settlements. PNC sees all of it in the payment graph regardless of AF membership.
- **The banking layer beneath AF** — AF's settlement payments still move through bank accounts. PNC can be the account / VAM / RTP layer under AF settlement, which is *also* how the structured demand IDs (Section 5.3) reach PNC for matching.

**Strategic options:** (a) integrate as the settlement bank beneath AF — structured demand data flows to PNC, unlocking Match; (b) serve the non-AF long tail directly; (c) both. **Recommend opening the AF integration conversation early**, because AF's structured data is the fastest path to solving Match — while independently building the non-AF cash-truth capability the graph already supports.

**Action:** verify AF's current Settlement Exchange payment and API capabilities and partnership posture directly with AF; the market claims in this document should be confirmed before they anchor product decisions.

---

## 11. Sequenced roadmap & first deliverable *(new)*

**Data-gated sequence:**

- **Now (PAYS + CPTY_PAYS):** stand up **Detect** at Tier 0 — PNC↔PNC reciprocal corridors + inbound external recoveries; build the candidate corridor/entity registry with confidence tiers and the avg-amount reinsurance discriminator.
- **When PAYS_CPTY lands:** Detect Tier 1 — full bidirectional corridor maps.
- **When CptyFinEntity lands:** Detect Tier 2 — interbank flow intelligence.
- **In parallel, gated on integration:** the **Match** workstream — pursue AF integration and/or a carrier claims-data pilot; extend the existing Jaro-Winkler matching engine.
- **After Match maturity:** Orchestrate, then Finance.
- **After governance sign-off:** Benchmark.

**First deliverable (≈90 days): Subrogation Corridor Detection module.** On current data, produce a registry of candidate carrier corridors, clearinghouse entities, and inbound-recovery flows for the insurance segment, each with a confidence tier. It is largely built already. It immediately serves two live use cases — prospect intelligence (heavy external subrogation counterparties to existing clients) and AML false-positive reduction (formalized subrogation patterns feeding the hub/exclusion registry) — and it is the credibility-builder that earns the mandate for the full Control Tower.

---

## 12. Open dependencies & questions for leadership *(new)*

1. **Will carrier customers share claims/recovery-ledger data with their bank?** The entire Match layer — and therefore Finance — depends on this. It is the single largest open question.
2. **What is Arbitration Forums' actual partnership posture and settlement API?** (Verify the claims in Section 10.)
3. **Is transaction-level remittance / addenda text accessible upstream of the aggregated snapshot?** If so, it enables both ground truth (Section 8) and far better matching (Section 5).
4. **Who owns the governance/compliance workstream** for data commercialization and prospect intelligence (Section 9)?
5. **What is the regulatory read** on ML-driven prospect targeting built from payment data?
6. **What is the rough TAM** — subrogation volume flowing through PNC's insurance customers? This can be estimated directly from the Tier-0 detection registry once the first deliverable ships.

---

*Document prepared for internal use — PNC Treasury Management, Data Science.*
