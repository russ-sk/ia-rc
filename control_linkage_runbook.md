# Control Linkage Pipeline — Technical Runbook

**Notebook:** `V2_API_RGL_Linkage_with_ROB_Control_Integration_V4.x_Prompt_v2.ipynb`
**Owner:** Russel (russel-s-k) · Impact Assessment GenAI
**Audience:** Engineers running / maintaining the control-linkage pipeline
**Scope:** Lane A "controls-first" — given a regulatory alert already linked to regulation summaries (RSMs), decide which **L1 Controls (1C)** the alert genuinely bears on, roll up to **L1 Library Controls (L1C)**, and score the result honestly against the golden dataset.

---

## 0. How to read this document

The notebook is organised into lettered sections. Run them **in order**, top to bottom:

| Section | Purpose | Produces |
|---------|---------|----------|
| **[B2]** | Build ID parentage maps + funnel / reachability report | `parentage`, `reachable_controls` |
| **[C]**  | Build candidates, run the LLM judge concurrently | `verdicts` parquet |
| **[D]**  | Score verdicts vs golden (gate taxonomy, addressable recall) | `perf` frame (`control` df), scorecard |
| **[E]**  | Diagnose *where* the recall misses go (fixable vs not) | printed diagnosis |
| **[F]**  | LLM-as-judge audit of the model's own verdicts | audit categories / drill-down |

Everything downstream depends on the two upstream inputs being present in the kernel:
- `results_with_truth` — the previous (RSM-linkage) run, carrying `linked_rsm_ids` per record.
- `golden` — the hand-verified truth table (see §2).

**Golden rule while running:** never run another cell while a judge cell (`[C]`) is still executing — the notebook uses `nest_asyncio` on a single event loop, and a second cell fighting that loop throws `cannot enter context ... already entered` and "Task was destroyed but it is pending". Wait for the `[*]` to become a number.

---

## 1. Environment & prerequisites

These must already be defined in the kernel (from the base pipeline cells run earlier):

| Name | What it is |
|------|-----------|
| `oai` / `openai_client` | The gateway chat client (OpenAI-compatible). |
| `tok` / `trust_token` | Trust-token provider. `await tok.get()` returns a bearer token; `tok.invalidate()` forces refresh on 401. |
| `MODEL` / `CHAT_MODEL` | Model id string (e.g. a GPT or Gemini model). Drives the `judge_call_kwargs()` family branch. |
| `USE_CASE` | **Mandatory** on every gateway chat/embed call as `user=USE_CASE`. Omitting it → `400 use case not found`. |
| `CTRL_USAGE_LOG` | List that accumulates token-usage dicts for cost tracking. |
| `embed_cache` / `embed_with_cache` | Embedding function with on-disk cache (returns vectors for a list of texts). |
| `LANE` / `LANE_HP` | Hyper-parameter dict: `concurrency`, `chunk`, `max_retries`, `max_tokens`, `top_k`, `reasoning`, etc. |
| `_norm_id` | ID normaliser — strips/upper-cases IDs so RegMap / Helios / golden join cleanly. Used **everywhere** IDs are compared. |
| `nest_asyncio` | Applied so `asyncio.run_until_complete` works inside Jupyter. |

**Prompt files on disk:**
- `CONTROL_PROMPT_PATH` → `CONTROL_SYSTEM_PROMPT` (the judge instructions).
- `AUDIT_SYSTEM_PROMPT` → built inline in [F].

**Data directories:** `DATA` / `DATA_DIR` (RegMap, Helios, Rapid2, golden), `OUTPUT_DIR` / `OUTP` (parquet outputs).

---

## 2. Data sources & schemas

| Source | Grain | Key columns used |
|--------|-------|------------------|
| **Rapid2** | alert (record) | `RECORD_ID`, alert title + summary → concatenated into **alert text**. |
| **RegMap** | reg → summary → risk → control | `REG_ID`, `SUM_ID` (RSM), `L1_CTRL` (1C), `L1_LIB_CTRL` (L1C), status/version join keys. Provides the **parentage** (which 1C sits under which RSM, which L1C is the parent). |
| **Helios** | control | `L1_CTRL`, `L1_LIB_CTRL`, **control text** (the 1C description the judge reads). A 1C with **no Helios text cannot be judged**. |
| **`results_with_truth`** | record × RSM | `linked_rsm_ids` — the RSMs the upstream run linked to each alert (LINKED only). This is the **entry point** for candidate generation. |
| **`golden`** | record × (RSM/L1C/1C) | `RECORD_ID`, `REG_ID`, `SUM_ID`, `L1_LIB_CTRL`, `L1_CTRL`, `Regulation Summary Control Mapping Approach` (Detailed/Standard). The hand-verified truth. |

> **Golden is HAND-PICKED and NON-EXHAUSTIVE** (business-confirmed). Two consequences, and they drive the whole scoring philosophy in [D]:
> 1. A golden link is verified truth → we **can** measure recall (caught / golden).
> 2. A predicted link **not** in golden is **not** a false positive — the golden is incomplete. We call these **discoveries**, not errors. So we do **not** compute precision.

---

## 3. Section [B2] — Parentage & Funnel

### 3.1 `build_parentage()` → `parentage`

Builds clean lookup dictionaries used by every downstream step. All IDs pass through `_norm_id`.

```
rsm_to_reg  : RSM  -> RGL        (regmap_rsm)      # each summary's parent regulation
ctrl_to_rsm : 1C   -> set(RSM)   (regmap_ctrl)     # a control can sit under >1 RSM
ctrl_to_lib : 1C   -> L1C        (helios)          # control's parent library control
lib_to_ctrls: L1C  -> [1C,...]   (helios)          # parent -> children, used for roll-up
```

Returned as a dict:
```python
parentage = {"rsm_to_reg": ..., "ctrl_to_rsm": ..., "ctrl_to_lib": ..., "lib_to_ctrls": ...}
```

**Why it matters:** `ctrl_to_rsm` is the reachability gate (a control is only reachable if RegMap maps it to an RSM), and `ctrl_to_lib` / `lib_to_ctrls` drive the 1C→L1C roll-up in scoring.

### 3.2 `print_funnel_report(parentage)`

Prints a "child-simple" picture of level sizes per source and how much they overlap, to prove the pipeline sees what it should.

Key locals:
- `regmap_rgls / regmap_rsms / regmap_ctrls` — distinct ID sets from RegMap at each level (via `_norm_id`).
- `helios_ctrls / helios_libs` — distinct control / library-control sets from Helios.
- `golden_recs / golden_rgls / golden_ctrls / golden_libs` — distinct sets from golden.
- `_ov(name, a, b)` — overlap helper: prints `|a ∩ b| / |a|` so you can see whether two sources agree on IDs (e.g. "golden 1C present in Helios").

**The critical block — REACHABILITY:**
```python
reachable = {c for c in golden_ctrls if c in ctrl_to_rsm and c in helios_ctrls}
```
A golden 1C is **reachable** only if **both**:
1. RegMap maps it to an RSM (`c in ctrl_to_rsm`) — so it can be reached from the alert→RSM path, **and**
2. Helios has its text (`c in helios_ctrls`) — so the judge has something to read.

`reachable` is the **upper bound on control recall** — no prompt can recover a control the pipeline can't even surface. `reachable_controls` is returned for use by the scorecard in [D].

> **Why apparent recall looks low:** the GOLDEN path is `Record → RGL → L1C → 1C`, but the PIPELINE path is `Record → RSM → 1C`. Golden links that have no RSM parent can never be reached by the pipeline, even though they're valid. That gap — not the model — is the main driver of low whole-golden recall.

---

## 4. Section [C] — Model Execution (concurrent)

### 4.1 `CONTROL_SCHEMA`

Strict `json_schema` response format. Each item requires:

| Field | Type | Meaning |
|-------|------|---------|
| `l1_control_id` | string | Echo of the candidate 1C id (verbatim; never invented). |
| `decision` | enum | `LINKED` / `NOT_LINKED` / `INSUFFICIENT_EVIDENCE`. |
| `relevance_score` | integer | 0–10, how directly the alert bears on the control. |
| `reasoning` | string | Written before deciding. |

`CONTROL_SYSTEM_PROMPT` is read from `CONTROL_PROMPT_PATH`.

> ⚠️ **Field-name discipline.** The reader must read the **same** key the model emits. If the prompt/schema say `relevance_score`, the row-builder must read `it.get("relevance_score")`. A mismatch (e.g. reading `relevance_score` while the model emits `link_score`) silently makes every score default to −1/0 — recall collapses to 0 with no error. Keep **prompt field = schema property = reader key** identical.

### 4.2 `judge_call_kwargs()`

Returns the per-model-family kwargs for the chat call:
- Always: `model=MODEL`, `user=USE_CASE`, `response_format=CONTROL_SCHEMA`.
- **GPT family:** `temperature=0.0`, `seed=42`, `max_tokens=LANE["max_tokens"]`; if `JUDGE_LOGPROBS`, also `logprobs=True`, `top_logprobs=5`.
- **Gemini / else:** `reasoning_effort=LANE["reasoning"]`, `max_completion_tokens=LANE["max_tokens"]` (no `seed`).

### 4.3 `judge_one_chunk(alert_text, control_chunk)`  *(async)*

Judges one alert against a small **chunk** of controls in a single API call.
- `control_chunk`: list of `(l1_control_id, control_text)`.
- `payload`: `{"alert_text": ..., "controls": [{"l1_control_id":cid, "control_text":txt}, ...]}`.
- `messages`: system = `CONTROL_SYSTEM_PROMPT`, user = `json.dumps(payload)`.
- Retry loop up to `LANE["max_retries"]`:
  - `token = await tok.get()`; call `oai.chat.completions.create(..., extra_headers={"Authorization": f"Bearer {token}"}, **judge_call_kwargs())`.
  - Log usage into `CTRL_USAGE_LOG` (`in`=prompt tokens, `out`=completion tokens, `ts`).
  - Parse `r.choices[0].message.content` → `["items"]`.
  - On `401`/`unauth`: `tok.invalidate()` then back off (`1.5 ** attempt`).
- **Fallback** (all retries failed): returns one dict per control with `decision="ERROR"`, `relevance_score=0`, so the batch **never crashes** the whole run.

### 4.4 `run_control_linkage(candidates_by_record, alert_text_by_id, control_text_by_id)`  *(async)*

Judges every record's candidates concurrently and in small chunks.
- **Inputs:**
  - `candidates_by_record`: `{record_id: [l1_control_id, ...]}`
  - `alert_text_by_id`: `{record_id: alert_text}`
  - `control_text_by_id`: `{l1_control_id: control_text}`
- **Output:** `DataFrame[RECORD_ID, L1_CTRL, decision, relevance_score, reasoning]`.
- **Concurrency:** `sem = asyncio.Semaphore(LANE["concurrency"])`; `chunk = LANE["chunk"]`.
- **Work units:** flat list of `(record_id, chunk_of_controls)` — for each record, its control ids are sliced into chunks of `chunk`.
- `_one(rid, control_chunk)`: acquires the semaphore, calls `judge_one_chunk`, then builds rows with **defensive `.get(...)`** (so a malformed item degrades to one row instead of a `KeyError`):
  ```python
  "RECORD_ID": rid,
  "L1_CTRL":  str(it.get("l1_control_id","")).strip(),
  "decision": it.get("decision","ERROR"),
  "relevance_score": it.get("relevance_score", 0),
  "reasoning": it.get("reasoning"),
  ```
- Tasks gathered with `asyncio.as_completed` under a `tqdm` progress bar (`judge chunks`).

> ⚠️ **Never read `it["decision"]` directly.** With a non-strict backend any single item can omit a field; a bare key access takes down the whole (multi-thousand-batch) run. Read every field with `.get(...)` and a default.

### 4.5 `build_control_candidates(records, top_k=None)`

The candidate builder. Produces `(candidates_by_record, alert_text_by_id, control_text_by_id)`.

Steps:
1. **record → linked RSMs** (from the previous run `results_with_truth`, `final_decision == "LINKED"`): explode `linked_rsm_ids` into `rsms_by_record[rid] = set(RSMs)`.
2. **RSM → child controls** (from RegMap): `ctrls_by_rsm[rsm] = set(1C)`.
3. **control text (Helios) + alert text (Rapid2):** `control_text_by_id[cid]`, `alert_text_by_id[rid]`. Controls with no text are dropped later (`c in cpos`).
4. **For each record:** gather child controls of its linked RSMs, dedup, keep only those with text/vectors:
   ```python
   child = []
   for rsm in rsms_by_record.get(rid, set()):
       child.extend(ctrls_by_rsm.get(rsm, []))
   child = [c for c in dict.fromkeys(child) if c in cpos]   # dedup + must have text
   ```
   - `cpos` / `apos`: position maps into `control_vecs` / `alert_vecs` (the embedding matrices) for controls and alerts respectively.
   - **`top_k is None` → judge ALL** controls under the record's linked RSMs (recommended default). Still ordered by cosine so the judge sees the most-relevant first (helps chunking / early-stop reasoning), but nothing is dropped:
     ```python
     order = np.argsort(-(control_vecs[[cpos[c] for c in child]] @ alert_vecs[apos[rid]]))
     ```
   - **`top_k` set → cosine shortlist** the top_k per record: same `order` sliced `[:top_k]`.
   - `candidates_by_record[rid] = [child[i] for i in order]`.

> **Design intent (read the in-code comment):** because the candidate set is already **scoped to the controls under the record's linked RSMs**, it is bounded. With `top_k=None` there is no retrieval cap, so a `NOT_JUDGED` outcome can only be a **data gap** (control had no RSM parent / no Helios text), never a "top_k too small" artefact. This removes retrieval as a confounder when interpreting recall.

### 4.6 Execute (cell [21])

```python
scope_records = set(_norm_id(golden["RECORD_ID"]))          # only score records present in golden
cands, alert_txt, control_txt = build_control_candidates(scope_records)
verdicts = await run_control_linkage(cands, alert_txt, control_txt)
verdicts.to_parquet(OUTPUT_DIR / f"control_verdicts_{MODEL}.parquet", index=False)
```
Prints candidates built, avg controls/record, and linked count.

**Output columns of `verdicts`:** `RECORD_ID, L1_CTRL, decision, relevance_score, reasoning`.

---

## 5. Section [D] — Model Performance / `control_scorecard`

`control_scorecard(golden_df, verdicts_df, reachable_controls)` — the **honest** three-number scorecard. Returns the per-pair frame `m` (this is the `control` df with columns `[REC, CTRL, reachable, DEC, gate]`).

### 5.1 Denominators
- `golden truth` = distinct (record, control) pairs in golden.
- `addressable golden` = golden pairs whose control **is reachable** (has Helios text) → the **recall base**.

### 5.2 Building the per-pair frame
- `g` = normalised golden `(REC, CTRL)` pairs (`_norm_id`, dropna, drop_duplicates).
- `v` = verdicts collapsed to the **strongest decision per (record, control)**: rank `LINKED(3) > INSUFFICIENT(2) > NOT_LINKED(1) > ERROR(0)`, keep the max.
- `m = g.merge(v, how="left")` — every golden pair gets the model's decision (or `NaN` if never judged).
- `m["reachable"]` = whether the control is in `reachable_controls`.

### 5.3 Gate taxonomy (`m["gate"]`)
Each golden pair is assigned exactly one **gate**:

| Gate | Meaning | Fix owner |
|------|---------|-----------|
| `CAUGHT` | Addressable **and** model said LINKED → **validated (TP)**. | — |
| `NOT_JUDGED` | Control has no Helios text / never judged → **data gap**. | Data reconciliation |
| `JUDGED_NOT_LINKED` | Judge saw it, said NOT_LINKED. | Prompt (if text-derivable) |
| `JUDGED_INSUFFICIENT` | Judge said INSUFFICIENT_EVIDENCE. | Enrich control text |
| `JUDGED_ERROR` | API/parse error on that pair. | Re-run |

Only **addressable** pairs are eligible for `CAUGHT`; recall is measured on addressable only.

### 5.4 The three honest numbers
```
addressable_recall = validated / max(golden_addressable, 1)     # A) ADDRESSABLE RECALL
data_gap           = golden_total - golden_addressable          # B) DATA GAP (no Helios text)
discoveries        = |predicted_LINKED pairs NOT in golden|     # C) DISCOVERY VOLUME
```
- **A) Addressable Recall** — of golden links the pipeline *could* judge, how many it caught. This is the number to quote, not the inflated whole-RGL recall.
- **B) Data Gap** — golden links with no control text; unjudgeable → reconciliation, not model fault.
- **C) Discovery Volume** — predicted LINKED not in golden; candidate **new** links for SME review (NOT false positives, because golden is non-exhaustive).

**Reference run (example figures):** known-true pairs = 64,114; `CAUGHT` 2,318 (3.6%), `NOT_JUDGED` 47,655 (74.3%), `JUDGED_NOT_LINKED` 13,592 (21.2%), `JUDGED_INSUFFICIENT` 549 (0.9%), `JUDGED_ERROR` 0. Whole-golden recall 3.6% is misleading; addressable recall (§6) is the real figure.

---

## 6. Section [E] — Control Diagnostic / `diagnose(perf_df)`

Explains **where the addressable-recall misses go**, so you know which lever to pull. Recall is measured only on reachable golden controls (those with Helios text); unreachable controls are excluded (reported in [D] as the data gap).

Locals (counts over `addr = perf_df[perf_df.reachable]`):
- `caught` = `gate == CAUGHT`.
- `not_judged` = `gate == NOT_JUDGED`, further split into:
  - **UNREACHABLE (data)** — no RSM parent / no Helios text → **UNFIXABLE by prompt**.
  - **reachable-but-missed** — retrieval/top_k gap → **raise top_k**.
- `judge_miss` = `JUDGED_NOT_LINKED` → **prompt-fixable if the link is text-derivable**.
- `insuff` = `JUDGED_INSUFFICIENT` → **enrich control text**.

**Reference diagnosis:** addressable golden 58,770 → `CAUGHT` 5,518 (9.4%), `NOT_JUDGED` 9,486 (16.1%), `JUDGED_NOT_LINKED` 42,627 (72.5%), `JUDGED_INSUFFICIENT` 1,139 (1.9%). Verdict: most misses are reachable controls the judge saw and said NOT_LINKED → the lever is the **prompt** (and the [F] audit tells you whether those rejections are correct).

> Note in code: with `top_k=None` (judge everything), reachable-but-missed should be ≈0 — if it isn't, a control's parent RSM was in the golden's RGL but not in the record's linked-RSM list (upstream RSM-recall gap or RegMap RSM→control mapping gap), i.e. a **candidate-build / upstream** issue, not the judge.

---

## 7. Section [F] — LLM as a Judge (audit layer)

A **second** LLM pass that audits the model's own verdicts and classifies *why* a link was missed or rejected — turns raw NOT_LINKED counts into a management-friendly story (is it our fault, a data/label limitation, or genuine SME judgement?).

- Runs on a **selectable slice**: a sampled number, or `None` = the whole set (sampled = fast for iteration, whole = defensible for the final deck).
- **`AUDIT_BUCKETS`** (level-1 categories):
  - `TRUE_LINK_MISSED` — link present in text but model failed → **our fault**.
  - `CORRECT_REJECTION` — genuinely different domain/product/process; no text-derivable link → **model right, golden questionable**.
  - `SME_JUDGEMENT_REQUIRED` — link may exist at policy level but is **not** derivable from the supplied texts; needs expert/institutional knowledge.
  - `STRUCTURAL_OVERMAPPING` — the control is unrelated; it only appears because it sits under a linked RGL that broadly touches the alert (whole-RGL / sibling over-mapping in the golden). A data/label artefact.
  - `TEXT_TOO_THIN` — control or alert text too sparse/generic to judge either way.
- **`SUB_REASONS`** — level-2 sub-reasons under each bucket (e.g. `MISS_paraphrase`, `MISS_terminology_gap`, `CR_different_regulator_domain`, `OVER_whole_rgl_swept`, `THIN_control_text`, …), used for the drill-down slides.
- **`ALL_SUB_REASONS`** = flatten of `SUB_REASONS.values()`.
- **`AUDIT_SYSTEM_PROMPT`** — instructs the auditor as an independent Regulatory Model Auditor: given the alert text, control text, the model's decision + reasoning, do **NOT** re-judge the link; instead explain **why** the model's decision differs from (or matches) the golden by choosing (1) a top category and (2) a specific sub-reason.

**Output:** per-verdict audit category → aggregated to show the composition of misses (how much is fixable by us vs data/SME limitations).

---

## 8. End-to-end run order (quick reference)

1. Run the **base pipeline** cells so `oai/tok/MODEL/USE_CASE/embed_cache/LANE/_norm_id` and the inputs `results_with_truth`, `golden` exist.
2. **[B2]** → `parentage = build_parentage()`; `reachable_controls = print_funnel_report(parentage)`.
3. **[C]** → define schema + judge funcs; run cell [21] to build candidates and produce `verdicts` (saved to `control_verdicts_{MODEL}.parquet`). **Do not touch other cells while it runs.**
4. **[D]** → `perf = control_scorecard(golden, verdicts, reachable_controls)` → read addressable recall / data gap / discovery volume; `perf` is your per-pair `control` df.
5. **[E]** → `diagnose(perf)` → see which bucket the misses fall in.
6. **[F]** → run the audit on a sample (or whole) to classify the misses for the deck.

---

## 9. Common errors & fixes (seen in practice)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `400 use case not found` (all rows ERROR) | `user=USE_CASE` missing on the gateway call. | Always pass `user=USE_CASE` in `judge_call_kwargs()`. |
| `name 'tok'/'oai'/'MODEL' is not defined` | Base-pipeline aliases not run. | Run the setup cell that defines `oai=openai_client; tok=trust_token; MODEL=CHAT_MODEL; USAGE=...`. |
| Every score 0 / recall collapses to 0, no error | Reader key ≠ emitted key (`relevance_score` vs `link_score`). | Make prompt field = schema property = reader key identical; or read tolerantly `it.get("relevance_score", it.get("link_score", -1))`. |
| `KeyError: 'decision'` mid-run, whole run dies | Bare `it["decision"]` on an item missing that field. | Read every field with `.get(..., default)`; make the fallback dict carry all keys. |
| `cannot enter context ... already entered` / "Task destroyed but pending" | A second cell run while the judge cell was still executing. | Wait for `[*]`→number; run [C] alone. Restart kernel if it got wedged. |
| JSON `Unterminated string` at large chunk | Response truncated at high `chunk`. | Lower `LANE["chunk"]`; retry errored candidates at a smaller chunk and merge. |
| `data type 'dbdate' not understood` (parquet) | BigQuery date extension type. | Read with a safe reader that casts unmappable Arrow types to string. |
| Recall 0% but scores exist | Linked-prediction filter empty (wrong gate/decision value). | Confirm `verdicts.decision.value_counts()`; build predicted set from the value that actually exists (`DEC == "LINKED"`). |
| New/Review = 0 | Candidate universe == golden (no non-golden candidates to discover). | Expected in golden-only mode; to test discovery, judge the full reachable pool, not just golden pairs. |

---

## 10. Glossary of key variables

| Variable | Section | Meaning |
|----------|---------|---------|
| `parentage` | B2 | Dict of `rsm_to_reg`, `ctrl_to_rsm`, `ctrl_to_lib`, `lib_to_ctrls`. |
| `reachable_controls` | B2 | Set of golden 1C that have both an RSM parent and Helios text — the recall ceiling. |
| `cands` / `candidates_by_record` | C | `{record: [1C candidate ids]}`. |
| `alert_txt` / `alert_text_by_id` | C | `{record: alert text}` (Rapid2 title+summary). |
| `control_txt` / `control_text_by_id` | C | `{1C: control text}` (Helios). |
| `cpos` / `apos` | C | Row positions of controls / alerts in the embedding matrices. |
| `control_vecs` / `alert_vecs` | C | Embedding matrices for controls / alerts. |
| `verdicts` | C | Judge output: `RECORD_ID, L1_CTRL, decision, relevance_score, reasoning`. |
| `perf` / `control` (`m`) | D | Per-pair labelled frame `REC, CTRL, reachable, DEC, gate`. |
| `gate` | D | Outcome class per golden pair (CAUGHT / NOT_JUDGED / JUDGED_NOT_LINKED / JUDGED_INSUFFICIENT / JUDGED_ERROR). |
| `addressable_recall` | D | validated / addressable golden — the headline recall. |
| `data_gap` | D | Golden pairs with no Helios text (unjudgeable). |
| `discoveries` | D | Predicted LINKED not in golden — candidate new links. |
| `AUDIT_BUCKETS` / `SUB_REASONS` | F | Level-1 / level-2 taxonomy for auditing misses. |

---

## 11. Output files

| File | From | Contents |
|------|------|----------|
| `control_verdicts_{MODEL}.parquet` | [C] | Raw judge verdicts per (record, control). |
| Scorecard / per-pair frame | [D] | `perf` (`control`) df; export as needed for the deck. |
| Audit results | [F] | Per-verdict audit category + sub-reason for the drill-down. |

---

*Reachability first, addressable recall second, discovery volume third — and the [F] audit turns the NOT_LINKED pile into a defensible story of what is our fault vs a data/SME limitation.*
