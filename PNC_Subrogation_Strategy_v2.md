# PNC P&C Subrogation Payments Strategy — Version 2

**PNC Bank · Treasury Management | Data Science**
*Version 2.0 — supersedes v1.0*

---

## 0. Changelog — What Changed From v1 and Why

v1 established the domain framing and the operational primer, both of which are retained. Five material revisions follow:

| # | Change | Type | Rationale |
|---|---|---|---|
| C1 | **Competitive position rewritten.** Arbitration Forums' Settlement Exchange System already performs the "Match" and "Orchestrate" functions v1 proposed as PNC's product — at no charge to members. | **Correction** | v1 treated AF as a workflow-only incumbent. AF is also a payments and cash-application incumbent. This changes where the whitespace is. |
| C2 | **Detection methodology added (§6) and v1's payment-graph signatures reality-checked against the actual PKN schema.** | **Correction + Addition** | v1 asserts detection signatures that the analytical store cannot currently observe. Roughly half depend on transaction-level remittance, memo, or rail data that does not exist in the monthly snapshots. |
| C3 | **Target segment (ICP) specified (§4); focus shifts from large carriers toward self-insureds, TPAs, recovery vendors, and municipalities.** | **Addition** | v1 never named the buyer. The AF finding makes the carrier-to-carrier corridor the *least* commercially open segment. |
| C4 | **Economics, regulatory/compliance, and phased gating added (§§3, 9, 10, 11).** | **Addition** | v1 committed to a product before establishing that the volume exists among PNC clients. Sequencing now gated. |
| C5 | **Scope, definitions, and terminology standardized (§1); health subrogation explicitly addressed.** | **Change** | v1 mixes "B2B Payment Knowledge Graph" with the internal PKN naming, and silently excludes health-plan recovery. |

**The one-sentence version of what changed:** *the intercompany carrier-to-carrier corridor that v1 targets is the most visible part of the graph and the least available part of the market; the commercially open segments are the ones hardest to see in our data.*

---

## 1. Scope, Definitions, and Terminology

### 1.1 Terminology standardization

This document uses **Payment Knowledge Network (PKN)** throughout, consistent with the Treasury Management roadmap. v1's "B2B Payment Knowledge Graph" refers to the same asset.

"Subrogation" is used here in the **operational** sense — any post-loss recovery of paid amounts from a responsible third party or its insurer — which is broader than the strict legal doctrine. Where legal precision matters (§5.4), contribution, indemnity, and coverage allocation are distinguished, because they behave differently in the payment graph even when claims teams file them under the same heading.

### 1.2 Lines in scope

| Line | In scope | Note |
|---|---|---|
| Commercial auto | ✅ Primary | Shortest recovery cycle; only line fully observable in a 23-month window |
| Commercial property | ✅ Secondary | Long tail; partial observability |
| Workers' compensation | ✅ Secondary | Statutory lien recovery; hardest to detect, commercially open |
| General liability | ✅ Tertiary | Often contribution/allocation rather than true subrogation |
| **Health plan subrogation** | ⚠️ **Decision required** | Excluded in v1 without explanation. See §1.3. |

### 1.3 Health subrogation — explicit scope decision needed

v1 is titled "P&C" but includes workers' compensation, which straddles the P&C/health boundary. Health-plan subrogation — recovery of paid medical expense against auto and liability settlements — is excluded entirely and without comment.

This is a large, cash-intensive, vendor-concentrated business, and the major recovery vendors in it are exactly the kind of receivables-heavy commercial client TM targets. They are also, unlike large carriers, unlikely to have specialized recovery treasury infrastructure.

**Recommendation:** make the exclusion explicit and time-boxed rather than silent. Either scope health in as a Phase 2 extension, or state the reason for exclusion (most likely: PHI handling burden — see §10.4). Do not leave it ambiguous in a document that goes to Product and Sales.

---

## 2. Revised Executive Recommendation

### 2.1 What survives from v1

The core framing holds: **PNC should treat casualty subrogation as a specialized receivables, settlement, and liquidity vertical, not as a new claims-administration business.** The governing principle also holds, and is in fact stronger than v1 argued — it is not only a strategic boundary but a regulatory one (§10.2):

> **Own the cash truth layer, not the liability-adjudication layer.**

### 2.2 What must be amended

v1 proposed a **Subrogation Receivables Control Tower** whose primary functions are matching recovery payments to claims and orchestrating settlement. For the intercompany carrier-to-carrier segment, both functions are already delivered by Arbitration Forums' Settlement Exchange System, at no charge, to a membership base that covers the overwhelming majority of the P&C market (§3).

This does not invalidate the strategy. It relocates it. The amended thesis:

> **PNC's opportunity is in the recovery flows that AF does not mediate, the balance-sheet and liquidity layer that AF does not provide, and the financing that no incumbent offers — not in the intercompany carrier-to-carrier settlement corridor.**

### 2.3 Revised strategic sequence, with gates

v1's five-step sequence is retained but re-ordered and gated. Each step now has an explicit pass condition; failing a gate stops the program rather than escalating it.

| Step | Revised description | Gate to proceed |
|---|---|---|
| **1. Detect** | Establish whether material subrogation-related flow exists *among PNC clients*, in the segments AF does not serve | ≥ N identifiable client relationships with material recurring recovery flow (N to be set with Sales; §11) |
| **2. Qualify** | Banker-led validation of detected candidates; convert structural inference into confirmed relationships | ≥ 60% of top-ranked candidates confirmed as genuine recovery flow by RM interview |
| **3. Bank** | Capture the cash: DDA/VAM/lockbox for recovery receipts, disbursement for outbound settlements | Signed pilot with ≥ 2 clients across ≥ 2 segments |
| **4. Match** | Cash-application layer — **only where AF/SES does not already provide allocation data** | Pilot clients demonstrate unmet matching need post-SES |
| **5. Finance** | Credit against **adjudicated or settled** recovery receivables only | Credit and Legal sign-off on collateral perfection (§10.3) |
| **6. Benchmark** | Privacy-safe recovery-velocity analytics | k-anonymity design approved by Privacy (§10.5) |

**Kill criteria.** v1 states no conditions under which the strategy should be abandoned. Proposed: if Phase 0 detection finds fewer than the agreed threshold of qualifying client relationships, or if RM validation confirms under 40% of top candidates, the vertical thesis is not supported and the work should be folded back into general receivables rather than continued as a standalone program.

---

## 3. Competitive Reality — The Arbitration Forums Position

**This section is new and is the most consequential revision in v2.**

### 3.1 What AF already does

v1 correctly identifies AF as the incumbent in demand management and intercompany arbitration, and correctly recommends against recreating it. It understates the position considerably.

- AF is a not-for-profit, membership-driven organization founded in 1943, serving over 5,400 members, and is the largest arbitration and subrogation services provider in the US.
- Its **E-Subro Hub** platform handles electronic subrogation demand exchange, document attachment, claim management, and escalation to arbitration. AF has stated participation exceeding 91% of the P&C market.
- E-Subro Hub is **free to members** — no start-up and no transactional fees.
- Its **Settlement Exchange System (SES)** automates EFT issuance and processing of incoming subrogation and arbitration payments once settlement is reached or an award is rendered. Claim resolution triggers the payment automatically.
- **AF supplies claim and coverage allocation data to the receiving party to automate processing of the incoming EFT.** This is cash application.
- SES includes a **payment aggregation and netting option**, reducing the number of payments exchanged between participating parties and letting them set their own settlement cadence.
- SES is available to members at no charge beyond modest implementation cost.
- Participating carriers include Nationwide, Enterprise, Allstate, GEICO, Travelers, American Family, and Farmers.
- AF has built integration into Guidewire ClaimCenter, allowing carriers to file and negotiate demands without leaving their claims system.

Annual volume through AF's member base runs to roughly two million subrogation demands and, per AF's current public materials, over $27 billion in claims.

### 3.2 What this means

Three consequences follow, and they should be stated plainly to leadership.

**First, "Match" and "Orchestrate" are occupied for the segment v1 targets.** SES does automated settlement payment plus allocation data delivery. That is the Control Tower's core function, already shipped, already adopted by the largest carriers, already integrated with the dominant claims system.

**Second, PNC cannot compete on price in that segment.** A not-for-profit giving the capability away at cost is not a margin structure a bank can undercut. Any PNC offering aimed at AF-mediated flow must win on something other than the workflow itself.

**Third — and this is the detection consequence — SES netting actively suppresses the signal v1's detection approach depends on.** v1 identifies bidirectional carrier-to-carrier payment relationships as the primary graph signature. SES netting collapses many bilateral settlements into fewer, larger net payments on a participant-chosen cadence. As SES adoption grows, bilateral reciprocity between major carriers *decreases* in the payment data even though the underlying subrogation activity is unchanged or growing. Detection built on bilateral reciprocity is measuring a shrinking and increasingly unrepresentative population.

### 3.3 The adjacent competitive set

v1 names no competitor other than AF. The claims-payments orchestration space also includes commercial vendors — One Inc's ClaimsPay platform, for instance, serves over 310 insurance companies with multi-party disbursement workflows, direct API integration into claims systems, and digital payment rails. On the health side, Zelis operates comparable payment and remittance infrastructure across a large payer network.

The implication is the same as the AF finding: outbound claims disbursement is a contested, well-served market. PNC entering it as a generalist bank has no obvious edge.

### 3.4 Where the whitespace actually is

AF's agreements cover intercompany disputes between signatory members, principally auto and property, involving carriers and self-insureds who have signed on. Outside that boundary:

| Flow | AF-mediated? | Commercial opening |
|---|---|---|
| Carrier ↔ carrier auto/property intercompany | **Yes** — E-Subro Hub + SES | Low. Do not target. |
| **Workers' comp lien recovery** (law firm trust → carrier/employer) | Largely no | **High** |
| **General liability contribution and coverage allocation** | Partially (Special Arbitration) | **Medium-high** |
| **Property recovery against non-insurer defendants** (contractors, manufacturers, utilities) | No | **High** |
| **Deductible reimbursement leg** (carrier → insured) | No | **Medium** |
| **Self-insured and TPA recovery cash operations** | Variable | **High** |
| **Financing against recovery receivables** | No — AF does not extend credit | **High, uncontested** |
| **Holding the balances** | No — AF moves money, does not hold it | **High** |

The last two are the durable positions. AF is a payment *network*; it is not a bank. Every dollar SES moves settles into a deposit account somewhere, and no incumbent lends against recovery receivables.

---

## 4. Segment Prioritization — Who Is the Buyer

v1 does not name a target customer. Given §3, the ranking inverts from the intuitive one.

| Rank | Segment | Rationale | Difficulty |
|---|---|---|---|
| **1** | **Self-insured corporates** (large fleets, retail, health systems, manufacturers, municipalities) | Already TM clients or prospects. Run real recovery operations. Rarely have specialized recovery treasury infrastructure. Often outside or peripheral to AF agreements. | Medium — must be identified in the graph, not by SIC/NAICS alone |
| **2** | **TPAs and recovery vendors** | Handle cash on behalf of many principals — a natural virtual-account and segregated-ledger use case. Fee-sensitive, operationally strained. | Low — cleanly identifiable via NAICS 524292 / 524291 |
| **3** | **Plaintiff and defense firms handling recovery distributions** | Trust-account operations with multiparty distribution. Genuine escrow and reconciliation need. | High — trust vs. operating accounts indistinguishable in current data |
| **4** | **Mid-market and regional carriers** | Less likely to have full SES integration or in-house treasury sophistication than the top ten. | Medium |
| **5** | **Top-tier national carriers** | Entrenched bank relationships, sophisticated treasury, already on SES. | Very high — deprioritize |

**Note on segment 1:** self-insured employers are the strategically interesting group because they sit at the intersection of an existing TM relationship and an unserved recovery-cash problem. They are also the group AF's carrier-centric infrastructure serves least well.

---

## 5. Casualty Subrogation Operational Primer

*Retained from v1 with detectability annotations added. The operational content is unchanged; the added column states whether each stated payment-graph signature is observable in the PKN as currently constituted.*

### 5.0 The three events that must remain separate

Subrogation is the substitution of one party — typically an insurer that has paid a loss — into the legal rights of its insured against a responsible third party. Three distinct events must not be conflated:

1. **Claim payment** — who paid the insured, claimant, provider, repairer, or other loss payee.
2. **Liability determination** — who was legally responsible, in what percentage, subject to what coverage, contract, limit, waiver, or defense.
3. **Recovery** — who ultimately reimbursed the paying carrier, self-insured, employer, or other party.

These can be separated by months or years, involve different legal entities and claim numbers, and settle through different bank accounts. The claim system may recognize a recovery at a materially different time from the bank ledger.

A recovery payment is therefore not an ordinary receivable. It can simultaneously be a reduction in paid loss, a return of the insured's deductible, reimbursement of allocated loss-adjustment expense, a reinsurer's share, a payment net of a recovery vendor's contingency fee, satisfaction of an arbitration award, and a partial settlement that leaves the underlying claim open.

This is why ordinary invoice-based cash application performs poorly here — and it remains the strongest argument for the strategy.

**Critical addition to v1:** PNC observes only event 3, and only the leg that touches a PNC account. Without claims integration the bank sees a recovery payment with no paid-loss denominator, no liability context, and no claim linkage. Every downstream analytic must be honest about this. It is the difference between *detecting a payment* and *understanding a recovery*.

### 5.1 Commercial auto

**Workflow.** The carrier pays its policyholder under first-party physical damage coverage — repair or ACV total loss, towing and storage, rental or loss of use, appraisal expense, less deductible. It then investigates liability: adverse driver, owner, employer and insurer; fault allocation under the jurisdiction's negligence rules; coverage status and limits; course and scope of employment; and whether a leasing company, maintenance contractor, shipper, or manufacturer shares responsibility. A demand follows, carrying demander and responder claim numbers, policy and insured detail, loss date and location, liability narrative, evidence, damage documentation, amount demanded, and remittance instructions. The adverse insurer may accept, deny, accept a comparative-negligence percentage, dispute damages, request evidence, negotiate, or proceed to intercompany arbitration. Settlement may be paid to the primary carrier, a subsidiary, a TPA, a recovery vendor, a law firm, or a designated lockbox — after which the recovery must be associated with the correct claim and split with the insured where deductible reimbursement is due.

**Payment-graph signature — detectability check:**

| v1 signature | Observable in PKN today? | Note |
|---|---|---|
| High transaction count | ✅ | `volume` field |
| Repeating carrier-to-carrier corridors | ✅ | Aggregate edges + month-persistence count |
| Repeated claim and demand identifiers | ❌ | No remittance or memo field in snapshots |
| Standard damage categories | ❌ | Not in payment data at all |
| Frequent use of arbitration platforms | ✅ | If AF appears as a node — see §6.5 |
| Many modest settlements vs. few large | ⚠️ Partial | Monthly pre-aggregation gives mean only, not distribution |
| Bidirectional carrier relationships | ✅ | But suppressed by SES netting — see §3.2 |

Commercial auto remains the right starting line, though for a reason v1 does not give: it is the **only line whose recovery cycle is short enough to be validated inside our 23-month snapshot window.**

### 5.2 Commercial property

**Workflow.** The carrier indemnifies its insured for covered physical damage and potentially business interruption, extra expense, debris removal, equipment breakdown, contents, and mitigation. Cause-and-origin investigation must preserve evidence and establish whether the loss arose from defective work or product, contractor negligence, utility failure, fire origin, sprinkler or plumbing failure, landlord or tenant conduct, vehicle impact, or equipment malfunction. Unlike most auto losses, a responsible third party may not be apparent when the first-party payment issues. Recovery turns heavily on contractual indemnity, waivers of subrogation, additional-insured provisions, risk-transfer clauses, lease and construction terms, product warranties, economic-loss doctrines, and evidence preservation. Waivers can defeat recovery even where another party is factually responsible.

**Payment-graph signature — detectability check:**

| v1 signature | Observable in PKN today? | Note |
|---|---|---|
| Lower transaction frequency than auto | ✅ | `volume` |
| Higher average severity | ✅ | `amount` ÷ `volume` |
| More law-firm and expert involvement | ⚠️ Partial | NAICS 5411 identifiable; trust vs. operating account is not |
| More wires and manual payments | ❌ | **No rail field in PKN snapshots** — see §7 |
| Settlement agreements and releases | ❌ | Documents, not payments |
| One payment covering many claim components | ❌ | Requires claim-level data |
| Long interval loss → payment → recovery | ❌ | Only the recovery leg is visible; 23-month window insufficient |

### 5.3 Workers' compensation

**Workflow.** Primarily no-fault: the employer or carrier pays statutory benefits — medical, temporary or permanent disability, wage replacement, rehabilitation, death benefits. A separate recovery opportunity arises where a party other than the employer or a coemployee is responsible: a negligent driver, premises owner, equipment manufacturer, general contractor, subcontractor, or vendor. State statutes then grant the employer or carrier a lien, reimbursement right, or subrogation interest against third-party proceeds, with highly state-specific rules governing control of the action, notice and consent, attorney-fee allocation, recovery expenses, the employee's net recovery, future-credit rights, and settlement approval. Distribution may run to the injured employee, employee's counsel or trust account, the comp carrier, the employer or self-insured, medical lienholders, recovery counsel, and other statutory claimants.

**Payment-graph signature — detectability check:**

| v1 signature | Observable in PKN today? | Note |
|---|---|---|
| Payments from plaintiff law-firm trust accounts | ⚠️ Weak | Firm identifiable; trust account is not |
| "Lien," "WC lien," "comp lien" descriptors | ❌ | **No memo or addenda field exists in the snapshots** |
| Multiparty distribution | ⚠️ Weak | Inferable as same-month fan-out from a firm node |
| Long durations | ❌ | Exceeds available window |
| Irregular payment amounts | ❌ | Monthly aggregation destroys per-transaction variance |
| Reimbursement vs. future-credit accounting | ❌ | Accounting treatment, not a payment attribute |
| TPA accounts for self-insureds | ✅ | NAICS 524292 |

v1's own caution is correct and worth reinforcing: this line **must not** be inferred from the words "claim" or "recovery," which generate heavy false positives. Note the strategic tension this creates — WC is simultaneously the **most commercially open** segment (§3.4) and the **least detectable** in current data. That tension should drive the data-acquisition priorities in §7 rather than being resolved by quietly retargeting to easier segments.

### 5.4 General liability

**Workflow.** Structurally different: the carrier pays defense and indemnity on behalf of its insured against a third-party claim rather than first-party damage to its insured's property. Post-payment recovery may arise through contribution from a joint tortfeasor, contractual indemnity, additional-insured coverage, equitable subrogation against another insurer, primary/excess/umbrella allocation, other-insurance disputes, or recovery from a party that assumed the risk contractually. The operational label "subrogation" therefore covers matters legal teams would characterize more precisely as contribution, indemnification, or coverage allocation.

**Payment-graph signature — detectability check:**

| v1 signature | Observable in PKN today? | Note |
|---|---|---|
| Law firms or settlement administrators | ⚠️ Partial | NAICS + name |
| Large infrequent wires | ⚠️ Partial | Amount yes; rail no |
| Confidential settlement references | ❌ | No memo field |
| Multiple carriers involved | ✅ | Structural — multi-party motif |
| Primary vs. excess allocation | ❌ | Requires coverage data |
| Defense-cost contribution | ❌ | Indistinguishable from other carrier-to-carrier flow |

---

## 6. Detection Methodology

**This section is new.** v1's opening pillar is "Detect: identify subrogation payment corridors, entities, and operational friction in the Payment Knowledge Graph," but supplies no method. This section supplies one, and states its limits.

### 6.1 What the PKN can actually see

The monthly analytical snapshots carry exactly these fields:

```
source, source_name, source_naics, amount, volume, dest, dest_name, dest_naics
```

at monthly grain, covering January 2024 through November 2025 (23 snapshots).

There is **no memo field, no remittance or addenda text, no rail indicator, no claim identifier, no account-type flag, and no transaction-level detail.** Every pair of counterparties is pre-aggregated to one row per month.

### 6.2 The aggregation problem

Monthly pre-aggregation is the most under-appreciated constraint in v1, and it degrades several stated signatures directly. Because `amount` and `volume` are both monthly sums, only the **mean** transaction size is recoverable — the distribution is gone.

A relationship showing 100 transactions totalling $500K is indistinguishable between:
- 100 settlements of $5,000 each (a classic auto subrogation portfolio), and
- 99 payments of $1,000 plus one $401,000 wire (a property settlement inside routine vendor traffic).

v1 relies on exactly this distinction to separate commercial auto ("many relatively modest settlements") from property and GL ("large and infrequent wires"). **That separation cannot be made at monthly grain.** It requires transaction-level data.

### 6.3 The scope constraint — PNC-to-PNC only

The PKN today contains PNC-customer-to-PNC-customer flows. Counterparty ingestion (PAYS_CPTY) has not landed.

For subrogation this bites hard, because the defining structure is a payment **between two institutions**. Today we can only observe a recovery corridor when *both* sides bank at PNC. Any Phase 0 volume estimate is therefore a floor, not an estimate, and its relationship to the true population depends on PNC's share of the insurance and self-insured segment — which is itself unknown.

The corollary is the genuine prize: once counterparty data lands, the non-PNC institutions on the other side of confirmed recovery corridors are among the best-qualified prospects the counterparty node set will ever produce. They are demonstrably transacting recovery volume with an existing PNC client, at observable frequency and size. That is a materially stronger prospecting signal than industry-code targeting.

### 6.4 Seed identification

Two complementary approaches, run as a union rather than an intersection:

**NAICS-based.** The 524 family (Insurance Carriers and Related Activities) is the anchor — direct P&C carriers (524126), TPAs (524292), claims adjusters (524291), and the residual 524298, where clearinghouse and intercompany-arbitration entities tend to land. Legal services (5411) supports the WC and GL work.

Rather than hardcoding a code list, the working notebook surfaces the distinct `(code, description)` pairs actually present in our data matching insurance- and recovery-related terms, and selects from observed values. This matters because the NAICS field carries known dirt (`-1|UNKNOWN`, `******`) documented in the PKN roadmap.

**Name-based.** Regex over `source_name` / `dest_name` for carrier, TPA, recovery-vendor, and arbitration-entity naming patterns. This catches entities mis-coded in NAICS, which is common for subsidiaries and captives.

### 6.5 The Arbitration Forums anchor

The most valuable single detection asset available today, and it costs nothing.

**Any node with a payment relationship to Arbitration Forums, Inc. is, by construction, an entity engaged in intercompany subrogation.** No inference required. AF's business is subrogation demand exchange and arbitration; it has no other reason to be exchanging payments with a carrier or self-insured.

This gives us:
- A **labeled seed set** in a problem that otherwise has no ground truth
- A **membership proxy** — AF-connected nodes are AF members, a population otherwise not visible to us
- A **discriminator** — corridors between two AF-connected nodes are far more likely to be genuine recovery flow than corridors between two arbitrary 524-coded nodes

Caveats: this only works if AF (or its SES settlement accounts) appears in our data at all, which depends on AF's own banking relationships and may require counterparty ingestion. And per §3.2, it biases the labeled set toward AF-mediated flow — precisely the segment §3.4 tells us not to target commercially. **Use AF as a training and validation anchor; do not use it as a targeting list.**

### 6.6 Structural discriminators

Applied on the multi-month aggregate, not per-month — recovery lags the original claim payment by weeks to months, so same-month reciprocity would badly undercount genuine relationships.

| Discriminator | Computation | Expected subrogation behaviour |
|---|---|---|
| **Reciprocity** | Directed edge exists both ways on the aggregate | Present — the core signal |
| **Balance ratio** | min(amount fwd, rev) ÷ max(amount fwd, rev) | Moderate to high; portfolio-scale symmetry between comparably-sized carriers |
| **Persistence** | Count of distinct months the pair is active | High — recovery is continuous, not episodic |
| **Mean transaction size** | amount ÷ volume | Moderate — distinguishes from reinsurance treaty flow |
| **Directional asymmetry** | Net position across all counterparties | Mixed — a carrier is a net receiver against some and net payer against others (v1's own observation, correctly elevated to a discriminator) |

That last row is the sharpest available discriminator against reinsurance, which is strongly and persistently directional in a way subrogation is not.

### 6.7 Confusables — what else produces this pattern

v1 lists no false-positive sources. Reciprocal insurer-to-insurer flow can be any of the following, and each needs an explicit rule-out:

| Confusable | Discriminator |
|---|---|
| Reinsurance (treaty and facultative) | Strongly directional; large and infrequent; premium leg is periodic |
| Coinsurance and fronting arrangements | Typically fixed proportional splits; stable ratios |
| Pooling among affiliated carriers | Same ultimate parent |
| **Intercompany affiliate transfers** | **Name similarity — reuse the existing Jaro-Winkler self-payment detection logic** |
| Agency and broker commission | Directional; counterparty NAICS 5242x brokers |
| Premium finance | Directional; regular amortizing pattern |
| Claims-fund replenishment (carrier ↔ TPA) | Directional; TPA is a net receiver of funding, net payer of claims |
| **Salvage proceeds** | Counterparty NAICS is salvage auction/buyer, not insurance |

Two notes. First, the intercompany-affiliate case is directly addressable with infrastructure we already have — the self-payment detection pipeline's first-token blocking and Jaro-Winkler scoring were built for exactly this name-matching problem and should be reused rather than rebuilt.

Second, on salvage: industry benchmarks including the NAIC's own reporting combine **salvage and subrogation** into a single figure. Any sizing exercise that cites those figures inherits the blend, and any detection work must separate them — they are different corridors with different counterparties and different treasury implications.

### 6.8 Validation strategy

Structural inference without ground truth is a hypothesis, not a finding. Four validation routes, in ascending cost:

1. **AF anchoring** (§6.5) — free, immediate, biased toward AF-mediated flow
2. **RM and banker confirmation** — take the top 20 ranked candidate relationships to the covering relationship managers and ask directly. Cheapest real validation available; should happen before any further engineering investment.
3. **Named-entity manual review** — analyst review of top corridors against public knowledge of the entities involved
4. **Pilot client claims-data share** — a single cooperating client providing recovery records under NDA would convert the entire exercise from inference to supervised learning

**Route 2 should be executed before Phase 1 begins.** It is a two-week exercise that can invalidate the entire premise cheaply, which is precisely what makes it worth doing first.

---

## 7. Data Requirements and Gaps

Mapping each proposed capability to the data it actually requires. Status reflects the analytical store as of this revision.

| Capability | Required data | Status | Blocking? |
|---|---|---|---|
| Carrier-to-carrier corridor detection | Aggregated PKN snapshots | ✅ Available | No |
| Entity typing (carrier / TPA / adjuster / firm) | NAICS + name | ⚠️ NAICS dirt documented | Degrades quality |
| Reciprocity and persistence analysis | Multi-month aggregate | ✅ Available | No |
| **Counterparty-side corridors** | **PAYS_CPTY** | ⏳ Pending ingestion | **Yes — caps addressable scope** |
| **Transaction-size distribution** | **Transaction-level detail** | ❌ Not in snapshots | **Yes — blocks auto/property separation (§6.2)** |
| **Rail mix (ACH / wire / RTP / check)** | **Rail indicator** | ❌ Not in PKN; exists in self-payment pipeline source | **Yes — blocks all rail-based signatures** |
| **Memo / addenda keyword detection** | **ACH addenda, wire OBI/BBI** | ❌ Not in analytical store | **Yes — would beat structural inference outright** |
| Claim-level matching | Claims system integration | ❌ Requires client partnership | Yes — for Match pillar |
| Trust vs. operating account typing | Account master attributes | ❌ Not joined | Yes — for WC and GL |
| Recovery cycle timing | Multi-year window | ⚠️ 23 months | Limits property, WC, GL |

**Highest-leverage single data acquisition:** remittance/addenda text. Free-text memo lines on ACH and wire commonly carry claim numbers, demand references, and literal terms like "subrogation," "subro," and "lien." A keyword search on that field would outperform every structural method in this document combined. It is worth establishing whether that data exists anywhere accessible upstream of the aggregated snapshots before investing further in structural inference.

**Second-highest:** rail indicator. Available in the source feeding the self-payment detection pipeline, therefore likely obtainable without a new data acquisition.

---

## 8. Product Architecture

v1 describes the Control Tower in a single paragraph. Given §3, it should be decomposed so that the AF-contested components can be dropped without collapsing the whole.

| Layer | Function | AF/SES overlap | Build? |
|---|---|---|---|
| **Deposit and liquidity** | DDA, VAM, segregated ledgers for recovery receipts | None | ✅ **Core** |
| **Collection** | Lockbox, RTP/ACH receipt, remittance capture | None | ✅ **Core** |
| **Cash application** | Match receipts to claims, demands, deductibles | **High for AF-mediated flow** | ⚠️ Only for non-AF flow |
| **Disbursement** | Outbound settlement payments | High — SES and commercial vendors | ❌ Deprioritize |
| **Netting** | Bilateral net settlement | **Direct — SES has this** | ❌ Do not build |
| **Financing** | Credit against adjudicated receivables | **None — uncontested** | ✅ **Differentiator** |
| **Analytics** | Recovery velocity, corridor benchmarking | Partial — AF has data, limited commercialization | ✅ Differentiator |

The two ✅ **Core** rows plus **Financing** constitute a defensible product. The Match and Orchestrate layers v1 centered the strategy on are the ones to scope carefully or drop.

**Action item:** v1 asserts existing PNC capability in integrated receivables, automated cash application, virtual account management, RTP connectivity, and insurance claim payment/remittance services. These should be confirmed against the current product catalog with Product Management, with exact product names, before this document is shared outside Data Science. v2 does not independently verify them.

---

## 9. Economics

v1 contains no revenue model or sizing. A first-pass frame follows; all PNC-specific figures require Phase 0 data.

### 9.1 Market anchors

- NAIC reporting indicates carriers recovered approximately **$51.6 billion** through salvage and subrogation in 2021 across auto physical damage, commercial auto liability, and personal auto liability. *(Note the salvage blend — §6.7.)*
- The ratio of salvage and subrogation recovery to claims paid rose from roughly 11% in 1996 to approximately 20% in 2021.
- Industry estimates cited in NAIC's Journal of Insurance Regulation place **missed** subrogation opportunity at roughly **$15 billion annually**, with other sources citing $15–20B.
- AF's member base files on the order of **two million subrogation demands annually**, representing **$27B+ in claims** per AF's current materials.

### 9.2 Revenue lines

| Line | Basis | Notes |
|---|---|---|
| Deposit balances | Recovery receipts held pre-application | Likely the largest single component; float on funds awaiting claim matching |
| Lockbox and receivables fees | Per-item | Standard TM pricing |
| VAM account fees | Per-account | Strong fit for TPAs holding funds for multiple principals |
| Payment origination | Per-transaction | Contested (§8) — low expectation |
| **Financing spread** | Advance rate against adjudicated receivables | **Uncontested; highest margin; highest legal complexity (§10.3)** |
| Analytics subscription | Per-seat or per-client | Requires §10.5 privacy design |

### 9.3 The sizing gate

None of the above can be converted to a PNC revenue estimate without knowing how much qualifying flow exists among PNC clients. That is exactly what Phase 0 detection is for, and it is why §2.3 gates product commitment behind it. **v1's error is committing to build the Control Tower before establishing that the market exists inside our client base.**

---

## 10. Risk, Compliance, and Regulatory

Absent from v1 entirely. Six issues, at least three of which need sign-off before Phase 1.

### 10.1 Data-use purpose limitation ⚠️ *raise early*

The PKN was built substantially for fraud and AML analytics. Using the same asset to drive commercial prospecting and product targeting raises purpose-limitation questions that should be settled with Legal and Privacy **before** a prospect list derived from it reaches Sales. This is a governance question, not a technical one, and it applies to the whole prospect-intelligence thesis, not just subrogation.

### 10.2 Proximity to claims adjudication

The governing principle — own the cash layer, not the liability layer — is also a regulatory boundary. Activities that constitute claims adjusting or administration are state-licensed. Any product feature that could be characterized as making or influencing coverage or liability determinations pulls PNC toward an activity it is not licensed for. Worth stating explicitly in the document so the boundary is understood as a constraint rather than a preference.

### 10.3 Financing collateral perfection

v1 correctly recommends lending only against acknowledged or adjudicated recoveries. Reinforcing, with additions: a subrogation recovery is a contingent claim, not a conventional receivable. Perfection of a security interest is legally non-obvious; state workers' compensation lien statutes vary and may subordinate or limit the interest; anti-assignment provisions may apply; recovery vendors commonly take contingency fees off the top, ahead of the carrier. **Recommend restricting Phase 1 credit exposure to post-award or post-executed-settlement receivables only, with the award or settlement agreement as the collateral document, and obtaining Credit and Legal sign-off before any term sheet.**

### 10.4 PII and PHI

Workers' compensation liens and any health-plan subrogation involve medical information. If PNC touches claim-level detail to perform matching, business-associate obligations and HIPAA-adjacent handling requirements arise. This is the most likely legitimate reason to scope health subrogation out (§1.3) — but it should be the stated reason, decided deliberately.

### 10.5 Privacy-safe benchmarking

v1 proposes commercializing recovery-velocity and payment-efficiency analytics as "privacy-safe" without designing the safety. Required minimum: aggregation thresholds preventing single-client identifiability, k-anonymity guarantees, and explicit contractual permission for use of client-derived data in benchmarks. Not optional, and cheaper to design now than retrofit.

### 10.6 Model risk and fair lending

If detection scores influence credit decisions or client targeting, SR 11-7 model risk management applies — documentation, validation, ongoing monitoring. Fair-lending review is required for any graph-derived feature entering a credit decision. Both are known-cost, known-process, but need to be in the plan rather than discovered during Phase 2.

### 10.7 Escheat

Settlement funds that go stale are subject to state unclaimed-property law. If PNC holds recovery funds pending application, the escheat obligation and its operational handling should be scoped into the product from the start.

---

## 11. Phased Plan with Decision Gates

Aligned to the PKN roadmap's data-availability milestones.

### Phase 0 — Detect and Validate *(now, ~1 quarter)*

**Goal:** establish whether qualifying flow exists among PNC clients. Cheap, fast, falsifiable.

| Deliverable | Owner | Notes |
|---|---|---|
| Aggregated 2025 network; NAICS-parsed seed set | Data Science | Built |
| AF anchor analysis | Data Science | §6.5 |
| Reciprocity, persistence, and confusable rule-out | Data Science | §§6.6–6.7 |
| Ranked candidate relationship list | Data Science | Top ~20 for validation |
| **RM validation interviews** | **Sales / TM** | **§6.8 route 2 — the gate** |
| Data-gap assessment: does addenda text exist upstream? | Data Science / Data Eng | §7 — highest leverage |
| Purpose-limitation review initiated | Legal / Privacy | §10.1 |

**Gate:** ≥ 60% of top candidates confirmed as genuine recovery flow. Below 40%, stop and reconsider per §2.3 kill criteria.

### Phase 1 — Qualify and Bank *(post-gate, Q4 2026 – Q1 2027)*

Segment-targeted outreach to §4 ranks 1–2. Pilot deposit, lockbox, and VAM structures. Rail data acquisition. Credit and Legal work on §10.3 in parallel.

### Phase 2 — Counterparty Expansion *(post PAYS_CPTY)*

Re-run detection with counterparty data. Convert non-PNC recovery counterparties into a qualified prospect list — subject to §10.1 clearance. Selective cash-application build for non-AF-mediated flow only.

### Phase 3 — Finance and Benchmark *(2027+)*

Recovery receivables financing on adjudicated claims. Privacy-safe benchmarking, subject to §10.5 design.

---

## 12. Open Questions for Leadership

1. **Does addenda or memo text exist anywhere accessible upstream of the aggregated snapshots?** Single highest-leverage question in this document (§7).
2. **Is health subrogation in or out**, and on what stated basis (§1.3)?
3. **What is N** — the minimum number of qualifying client relationships that justifies continuing (§2.3)?
4. **Has purpose-limitation clearance been obtained** for commercial use of the PKN (§10.1)?
5. **Is there appetite for a partnership conversation with AF**, given they own the workflow layer and PNC owns the banking layer? Complementary rather than competitive — but only if approached that way.
6. **Which existing PNC products actually deliver the §8 core layers today**, under what names (§8 action item)?
7. **Is one pilot client willing to share recovery records under NDA?** Converts the whole exercise from inference to supervised learning (§6.8 route 4).

---

## Appendix A — Sources for External Claims

External assertions in §§3 and 9 draw on the following. Internal claims about PKN schema, data availability, and existing pipelines derive from the PKN roadmap and current Data Science work.

- Arbitration Forums, Inc. — corporate site, E-Subro Hub and Total Recovery Solution product pages (`home.arbfile.org`)
- Arbitration Forums SES adoption announcements — Travelers (2020), American Family (2021), Farmers (2021), payment aggregation offering (2018)
- Guidewire press release, AF Subrogation Accelerator for ClaimCenter (December 2024)
- NAIC, *Journal of Insurance Regulation* — "How's the Recovery? Salvage and Subrogation in the Property & Casualty Industry" (2023)
- One Inc — company and ClaimsPay product materials
- Zelis — provider payments materials

**Verification note:** AF adoption and volume figures are drawn from AF's own published materials and press releases spanning 2018–2026 and reflect different reporting dates; the member counts and claim values differ across sources accordingly. They should be re-confirmed at a single point in time before external use.

---

*Prepared for internal use — PNC Treasury Management, Data Science. Version 2.0.*
