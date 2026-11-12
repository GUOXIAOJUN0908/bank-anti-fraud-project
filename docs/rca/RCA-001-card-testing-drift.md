# RCA-001: Fraud recall dropped from 92% to 22% after a change in fraud tactics

| | |
|---|---|
| Severity | 2 (wrong decisions at scale; financial exposure) |
| Detected | Weekly performance check with chargeback labels (`make drift-batch`) |
| Period affected | New period, 2026-05-01 to 2026-05-30 (simulated) |
| Model | `hgb-20261112T081747Z` |
| Status | Root cause found; corrective action validated; preventive actions open |

> This is a simulated incident on synthetic data, written to exercise the monitoring and RCA process end to end. Every
> number comes from running the code in this repository (`make drift-batch`, `make rca`).

## 1. Summary

In May the share of fraud the model caught at the 1% review threshold fell from **92.4%** (test period) to **21.6%**.
Fraudsters moved from large online purchases to **card testing**: bursts of 5-15 payments of a few dollars at online
merchants close to the cardholder, used to check a stolen card works before selling it. The model had never seen this
pattern. Feature drift monitoring (PSI) **did not fire** because these payments look like normal small spend and are
only 0.6% of traffic. Retraining with 10 days of new labels restored recall to **85.2%**.

## 2. Timeline

| When | Event |
|---|---|
| May 1 | New fraud pattern starts. Live service keeps approving it. |
| Daily | PSI report: all features OK (max PSI 0.022, `amt`). No alert. |
| End of May | Chargeback labels arrive. Weekly performance check: recall 21.6% vs 92.4%, ALERT. |
| Investigation | "Caught vs missed" analysis (below) points to small, high-velocity, local fraud. |
| Fix | Challenger trained on Jan-Apr + May 1-10, evaluated on May 16-30. |

## 3. Impact

| Metric | Test period (Apr) | New period (May) |
|---|---|---|
| Fraud rate | 0.35% | 0.58% |
| PR-AUC | 0.893 | 0.207 |
| Recall at review threshold | 92.4% | 21.6% |
| Precision at review threshold | 33.5% | 13.6% |
| Share of fraud **amount** caught | 95.8% | 93.3% |

The money directly lost stayed small (93% of fraud value still caught), because each test payment is tiny. The real risk
is **downstream**: cards that pass testing are sold and used for large fraud later, and a falling precision wastes
analyst time.

## 4. Analysis

### 4.1 What changed? Caught vs missed fraud (May)

| | Caught | Missed |
|---|---|---|
| Count | 103 | 373 |
| Median amount | $329.01 | **$8.20** |
| Median transactions on card in last hour | 2 | **6** |
| Median distance from home | 4,972 km | **6.9 km** |
| Top categories | shopping_net, misc_net, travel | misc_net, shopping_net, food_dining |

Missed fraud is the opposite of what the model learnt: small amounts near home instead of large amounts far away.
The one strong signal it still carries is **velocity** (6 payments an hour vs 1-2 normally).

### 4.2 Why did the model not use velocity?

In training, velocity barely mattered (permutation importance `txn_count_1h` 0.006) because amount and distance
already separated the old fraud patterns. Trees put their splits on the features that worked then, so a burst of tiny
payments still scored low.

### 4.3 Why did drift monitoring not catch it?

PSI measures the shape of the **whole population**. Card testing adds ~400 small payments to ~80,000 genuine ones, so
`amt`, `txn_count_1h` and the score distribution hardly move (all PSI < 0.03). The change is in the relationship
between features and the label (**concept drift**), which only labels can show.

### 4.4 Ruled out

| Hypothesis | Evidence | Verdict |
|---|---|---|
| Training/serving skew (bug in online features) | Scores logged by the live API equal the offline batch scores for the same 5,000 May transactions (max difference 0.0); feature parity test passes | Ruled out |
| Threshold misconfiguration | Alert rate 0.58% (live sample) and 0.93% (full May batch), in line with the 1% policy | Ruled out |
| Upstream data error | No schema errors, PSI normal, fraud rate higher not lower | Ruled out |

### 4.5 Root cause

The model was trained only on fraud patterns that were large-value and/or distant, and monitoring relied on population
drift, which cannot see a new pattern hidden inside normal-looking small transactions.

## 5. Corrective action (validated)

Retrain on Jan-Apr plus the first 10 days of labelled May data; re-select thresholds on May 10-15; compare on
May 16-30 (40,985 transactions, 230 frauds), unseen by both models (`make rca`):

| Metric | Champion (Jan-Apr) | Challenger (+ May 1-10) |
|---|---|---|
| PR-AUC | 0.184 | **0.837** |
| Alert rate | 0.89% | 0.92% |
| Recall at review threshold | 19.1% | **85.2%** |
| Precision at review threshold | 12.1% | **52.0%** |
| Share of fraud amount caught | 91.3% | 96.9% |

Same analyst workload (~0.9% of traffic), more than 4 times the recall and precision.

## 6. Preventive actions

| # | Action | Owner | Status |
|---|---|---|---|
| 1 | Weekly label-based recall check is a release-blocking alert, not just PSI | ML engineering | Done (`drift.py`, `--fail-on-alert`) |
| 2 | Add segment monitoring: share of cards with >= 5 payments under $20 in an hour | ML engineering | Open |
| 3 | Add card-testing features: count of small payments in 1h, distinct merchants in 1h | Data science | Open |
| 4 | Monthly retrain with the latest labelled data, champion/challenger before release | ML engineering | Open |
| 5 | Rule-based safety net for extreme velocity while a retrain is in progress | Fraud operations | Open |

## 7. Lessons learned

1. **PSI is necessary but not sufficient.** A model can fail badly with every PSI "OK". Monitor outcomes as well.
2. **Look at what the model misses, not just the averages.** The caught-vs-missed table found the cause in minutes.
3. **Prove there is no pipeline bug first.** Matching live and offline results ruled out training/serving skew early.
