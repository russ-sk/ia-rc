# =============================================================================
# LANE A — alert -> regulation -> summary -> library risk -> library control
#          -> control instance
#
# Clean rebuild. Replaces every earlier A-cell. Paste cell by cell; each ends
# with a CHECK block. Run in order.
#
# Fixes folded in vs the earlier version:
#   - library risks are sourced independently of whether they have a control
#   - library control TEXT comes from Helios_rtcl (library grain), not from the
#     instance table, so a control with zero instances still has a title
#   - control INSTANCES come from RegMap Extract 5 (regulation-scoped), not from
#     the global Helios instance inventory
#   - every enrichment join is LEFT; nothing is silently dropped
#   - routing on Control Mapping Approach taken from E3 (E2 as fallback)
#   - judge: reasoning first, nullable evidence, seed varies per retry
#   - risks RegMap references but Helios cannot describe are carried as UNRESOLVED
#   - '-' placeholders stripped at every level; statuses compared casefolded
#
# Needs from RGL_Linkage_V1: DATA, OUT, clean, embed_cache, oai, tok, MODEL,
# USE_CASE, USAGE, load_alerts, nest_asyncio, and `pairs`.
# =============================================================================


# %% A1. Config and helpers ---------------------------------------------------
import re, json, ast, time, asyncio
import numpy as np
import pandas as pd
from pathlib import Path

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

REGMAP_DIR = Path(DATA) / "REGMAP"
HELIOS_DIR = Path(DATA) / "Helios"
OUTP       = Path(OUT) if isinstance(OUT, str) else OUT
OUTP.mkdir(parents=True, exist_ok=True)

STATUS_KEEP        = "published"          # compared casefolded
DIRECT_APPROACHES  = {"Detailed", "Enhanced"}
UNKNOWN_APPROACH   = "judged"             # blank / unrecognised -> judge, never inherit
KEEP_RECORDS       = {"139559", "141013", "142845"}   # None = every alert in `pairs`

HP = {"top_k": 15, "max_tokens": 6000, "concurrency": 10,
      "max_retries": 3, "min_score": 5}

C = {
    "JUR":          "Regulation Jurisdiction",
    "REG_ID":       "Regulation ID",
    "REG_STATUS":   "Regulation Status",
    "REG_VER":      "Regulation Version",
    "SUM_ID":       "Regulation Summary ID",
    "SUM_STATUS":   "Regulation Summary Status",
    "SUM_VER":      "Regulation Summary Version",
    "SUM_APPROACH": "Regulation Summary Control Mapping Approach",
    "SUM_TITLE":    "Regulation Summary Title",
    "SUM_TEXT":     "Regulation Summary Detail Text",
    "SUM_STEWARD":  "Regulation Summary Risk Steward Area",
    "SUM_TAX_L1":   "Regulation Summary Risk Taxonomy L1",
    "SUM_TAX_L2":   "Regulation Summary Risk Taxonomy L2",
    "SUM_TAX_L3":   "Regulation Summary Risk Taxonomy L3",
    "APPLIC_KEY":   "Regulation Summary Applicability Unique Key",
    "LIB_RISK_ID":  "Regulation Summary Library Risk ID Number",
    "L1_LIB_CTRL":  "Regulation Summary L1 Library Control ID Number",
    "RISK_ID":      "Risk ID",
    "RISK_APPLIC":  "Risk Instance Applicable?",
    "L1_CTRL":      "L1 Control ID",
    "L1_CTRL_APPL": "L1 Control Instance Applicable?",
}

SUMMARY_KEY       = ["JUR", "REG_ID", "REG_STATUS", "REG_VER",
                     "SUM_ID", "SUM_STATUS", "SUM_VER"]
APPLICABILITY_KEY = SUMMARY_KEY + ["SUM_TAX_L1", "SUM_TAX_L2", "SUM_TAX_L3", "APPLIC_KEY"]

PLACEHOLDER = {"nan", "none", "nat", "", "-", "null", "<na>"}


def _k(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def resolve(df, label):
    exact = [c for c in df.columns if _k(c) == _k(label)]
    if exact:
        return exact[0]
    loose = [c for c in df.columns if _k(label) in _k(c)]
    if len(loose) == 1:
        return loose[0]
    raise KeyError(f"cannot resolve {label!r}\n  candidates: {loose or 'none'}"
                   f"\n  available : {list(df.columns)}")


def standardise(df, shorts):
    return df.rename(columns={resolve(df, C[s]): s for s in shorts})[shorts].copy()


def strip_ids(df, cols):
    for c in cols:
        if c in df.columns:
            s = df[c].astype(str).str.strip()
            df[c] = s.where(~s.str.lower().isin(PLACEHOLDER), np.nan)
    return df


def norm_approach(s):
    a = s.astype(str).str.strip().str.title()
    return a.where(~a.str.lower().isin(PLACEHOLDER), np.nan)


def published(df):
    n = len(df)
    for c in ("REG_STATUS", "SUM_STATUS"):
        if c in df.columns:
            df = df[df[c].astype(str).str.strip().str.casefold() == STATUS_KEEP]
    if n and not len(df):
        raise ValueError(f"published filter removed every row — check {STATUS_KEEP!r}")
    return df.reset_index(drop=True)


def read_regmap(n):
    hits = [p for p in REGMAP_DIR.iterdir()
            if p.suffix.lower() in {".xlsx", ".xls", ".csv", ".parquet"}
            and re.search(rf"extract\s*_?{n}\b", p.name, flags=re.I)]
    if not hits:
        raise FileNotFoundError(f"no Extract {n} in {REGMAP_DIR}: "
                                f"{[p.name for p in REGMAP_DIR.iterdir()]}")
    p = sorted(hits)[0]
    df = (pd.read_parquet(p) if p.suffix.lower() == ".parquet" else
          pd.read_csv(p, dtype=str) if p.suffix.lower() == ".csv" else
          pd.read_excel(p, dtype=str))
    df.columns = [str(c).strip() for c in df.columns]
    print(f"[load] {p.name}: rows={len(df):,} cols={len(df.columns)}")
    return df


def read_helios(stem):
    hits = [p for p in HELIOS_DIR.iterdir()
            if p.suffix.lower() in {".parquet", ".csv", ".xlsx"}
            and stem.lower() in p.name.lower()]
    if not hits:
        raise FileNotFoundError(f"no {stem!r} in {HELIOS_DIR}: "
                                f"{[p.name for p in HELIOS_DIR.iterdir()]}")
    p = sorted(hits)[0]
    df = (pd.read_parquet(p) if p.suffix.lower() == ".parquet" else
          pd.read_csv(p, dtype=str) if p.suffix.lower() == ".csv" else
          pd.read_excel(p, dtype=str))
    df.columns = [str(c).strip() for c in df.columns]
    print(f"[load] {p.name}: rows={len(df):,} cols={len(df.columns)}")
    return df


def joinset(s):
    return " | ".join(sorted({str(x) for x in s if pd.notna(x)}))


print("A1 CHECK  regmap:", REGMAP_DIR.exists(), " helios:", HELIOS_DIR.exists(),
      " out:", OUTP)


# %% A2. Load RegMap E2, E3, E5 ----------------------------------------------
raw2, raw3, raw5 = read_regmap(2), read_regmap(3), read_regmap(5)

e2 = standardise(raw2, SUMMARY_KEY + ["SUM_APPROACH", "SUM_TITLE", "SUM_TEXT",
                                      "SUM_STEWARD", "SUM_TAX_L1", "SUM_TAX_L2",
                                      "SUM_TAX_L3", "LIB_RISK_ID"])
e2 = strip_ids(e2, ["REG_ID", "SUM_ID", "LIB_RISK_ID"])
e2["SUM_APPROACH"] = norm_approach(e2.SUM_APPROACH)
e2p = published(e2)

e3 = standardise(raw3, APPLICABILITY_KEY + ["SUM_APPROACH", "LIB_RISK_ID", "L1_LIB_CTRL"])
e3 = strip_ids(e3, ["REG_ID", "SUM_ID", "APPLIC_KEY", "LIB_RISK_ID", "L1_LIB_CTRL"])
e3["SUM_APPROACH"] = norm_approach(e3.SUM_APPROACH)
e3p = published(e3)

e5 = standardise(raw5, APPLICABILITY_KEY + ["SUM_APPROACH", "RISK_ID", "RISK_APPLIC",
                                            "L1_CTRL", "L1_CTRL_APPL"])
e5 = strip_ids(e5, ["REG_ID", "SUM_ID", "APPLIC_KEY", "RISK_ID", "L1_CTRL"])
e5p = published(e5)

print("\nA2 CHECK")
for nm, d in [("E2", e2p), ("E3", e3p), ("E5", e5p)]:
    print(f"  {nm}: rows={len(d):,}  regs={d.REG_ID.nunique():,}  "
          f"summaries={d.SUM_ID.nunique():,}")
print(f"  E3 library risks={e3p.LIB_RISK_ID.nunique()}  "
      f"library controls={e3p.L1_LIB_CTRL.nunique()}")
print(f"  E5 risk instances={e5p.RISK_ID.nunique()}  "
      f"control instances={e5p.L1_CTRL.nunique()}")


# %% A3. Load Helios ----------------------------------------------------------
hrisk_raw = read_helios("risk")
l1_raw    = read_helios("L1_Control")
rtcl      = read_helios("rtcl")

# library risks — library-grain text
hrisks = (hrisk_raw[["library_risk_id", "library_risk_title",
                     "library_risk_description", "risk_status"]]
          .dropna(subset=["library_risk_id"]).drop_duplicates("library_risk_id")
          .rename(columns={"library_risk_id": "LIB_RISK_ID",
                           "library_risk_title": "risk_title",
                           "library_risk_description": "risk_description"}))
hrisks["LIB_RISK_ID"] = hrisks.LIB_RISK_ID.astype(str).str.strip()
hrisks["risk_text"] = (hrisks.risk_title.fillna("") + ". " +
                       hrisks.risk_description.fillna("")).map(clean)

# library controls — text from rtcl (library grain), US_L1_Control as fallback
ctrl_lib = (rtcl[["l1_control_library_id", "l1_control_library_title",
                  "l1_control_library_description", "l1_control_library_status"]]
            .dropna(subset=["l1_control_library_id"])
            .drop_duplicates("l1_control_library_id")
            .rename(columns={"l1_control_library_id": "L1_LIB_CTRL",
                             "l1_control_library_title": "control_title",
                             "l1_control_library_description": "control_description",
                             "l1_control_library_status": "control_status"}))
ctrl_fb = (l1_raw[["l1_library_control_id", "l1_library_control_title",
                   "l1_library_control_desc"]]
           .dropna(subset=["l1_library_control_id"])
           .drop_duplicates("l1_library_control_id")
           .rename(columns={"l1_library_control_id": "L1_LIB_CTRL",
                            "l1_library_control_title": "control_title",
                            "l1_library_control_desc": "control_description"})
           .assign(control_status=np.nan))
for d in (ctrl_lib, ctrl_fb):
    d["L1_LIB_CTRL"] = d.L1_LIB_CTRL.astype(str).str.strip()
hcontrols = pd.concat([ctrl_lib,
                       ctrl_fb[~ctrl_fb.L1_LIB_CTRL.isin(set(ctrl_lib.L1_LIB_CTRL))]],
                      ignore_index=True)

# identity maps: risk instance -> library risk, control instance -> library control
r2lr = (hrisk_raw[["name", "library_risk_id"]].dropna().drop_duplicates()
        .rename(columns={"name": "RISK_ID", "library_risk_id": "LIB_RISK_ID"}))
c2lc = (l1_raw[["name", "l1_library_control_id"]].dropna().drop_duplicates()
        .rename(columns={"name": "L1_CTRL", "l1_library_control_id": "L1_LIB_CTRL"}))
for d in (r2lr, c2lc):
    for c in d.columns:
        d[c] = d[c].astype(str).str.strip()

print("\nA3 CHECK")
print(f"  helios library risks   : {len(hrisks):,}  "
      f"median text len={hrisks.risk_text.str.len().median():.0f}")
print(f"  library control text   : {len(ctrl_lib):,} rtcl + "
      f"{len(hcontrols)-len(ctrl_lib):,} fallback = {len(hcontrols):,}")
print(f"  risk instance map      : {len(r2lr):,}")
print(f"  control instance map   : {len(c2lc):,}")
print(f"  RegMap risks in Helios : "
      f"{len(set(e3p.LIB_RISK_ID.dropna()) & set(hrisks.LIB_RISK_ID))}"
      f" / {e3p.LIB_RISK_ID.nunique()}")
print(f"  RegMap ctrls with text : "
      f"{len(set(e3p.L1_LIB_CTRL.dropna()) & set(hcontrols.L1_LIB_CTRL))}"
      f" / {e3p.L1_LIB_CTRL.nunique()}")


# %% A4. Approach profile — is the routing key sound? -------------------------
print("A4 CHECK")
print("approach by distinct summary (E2):")
print(e2p.drop_duplicates("SUM_ID").SUM_APPROACH.value_counts(dropna=False).to_string())

chk = (e2p[["SUM_ID", "SUM_APPROACH"]].drop_duplicates("SUM_ID")
       .merge(e3p[["SUM_ID", "SUM_APPROACH"]].drop_duplicates("SUM_ID"),
              on="SUM_ID", suffixes=("_e2", "_e3")))
print(f"\nE2 vs E3 approach agrees on "
      f"{(chk.SUM_APPROACH_e2 == chk.SUM_APPROACH_e3).mean():.1%} of {len(chk):,} summaries")
print(f"summaries with >1 approach in E2: "
      f"{(e2p.groupby('SUM_ID').SUM_APPROACH.nunique(dropna=False) > 1).sum()}")

for nm, col in [("controls", "L1_LIB_CTRL"), ("risks", "LIB_RISK_ID")]:
    s = e3p.dropna(subset=[col]).groupby(["SUM_APPROACH", "SUM_ID"])[col].nunique()
    print(f"\n{nm} per summary, by approach:")
    print(s.groupby("SUM_APPROACH").agg(summaries="count", median="median",
                                        p90=lambda x: x.quantile(.9), max="max").to_string())


# %% A5. Build the four base tables -------------------------------------------
# summaries : one row per SUM_ID, with the routing approach
# rr        : risk rows        (REG_ID, SUM_ID, LIB_RISK_ID)  -- NO control required
# rc        : risk-control     (REG_ID, SUM_ID, LIB_RISK_ID, L1_LIB_CTRL)
# ri        : scoped instances (REG_ID, SUM_ID, LIB_RISK_ID, L1_LIB_CTRL, l1_control_id)

summaries = e2p[SUMMARY_KEY + ["SUM_TITLE", "SUM_TEXT", "SUM_STEWARD"]].drop_duplicates()
assert summaries.SUM_ID.is_unique, "SUM_ID not unique — inspect before routing"

appr = (e3p[["SUM_ID", "SUM_APPROACH"]].drop_duplicates("SUM_ID")
        .merge(e2p[["SUM_ID", "SUM_APPROACH"]].drop_duplicates("SUM_ID"),
               on="SUM_ID", how="outer", suffixes=("_e3", "_e2")))
appr["SUM_APPROACH"] = appr.SUM_APPROACH_e3.fillna(appr.SUM_APPROACH_e2)
summaries = summaries.merge(appr[["SUM_ID", "SUM_APPROACH"]], on="SUM_ID", how="left")

e3_all = e3p[e3p.LIB_RISK_ID.astype(str).str.startswith("LR-")]      # no control filter
rr = pd.concat([
    e3_all[["REG_ID", "SUM_ID", "LIB_RISK_ID"]],
    e2p.loc[e2p.LIB_RISK_ID.astype(str).str.startswith("LR-"),
            ["REG_ID", "SUM_ID", "LIB_RISK_ID"]],
], ignore_index=True).drop_duplicates()

rc = (e3p.loc[e3p.LIB_RISK_ID.astype(str).str.startswith("LR-")
              & e3p.L1_LIB_CTRL.astype(str).str.startswith("L1C-"),
              ["REG_ID", "SUM_ID", "LIB_RISK_ID", "L1_LIB_CTRL"]].drop_duplicates())

ri = (e5p.merge(r2lr, on="RISK_ID", how="left").merge(c2lc, on="L1_CTRL", how="left"))
ri = (ri.loc[ri.LIB_RISK_ID.notna() & ri.L1_LIB_CTRL.notna(),
             ["REG_ID", "SUM_ID", "LIB_RISK_ID", "L1_LIB_CTRL", "L1_CTRL"]]
        .rename(columns={"L1_CTRL": "l1_control_id"}).drop_duplicates())

with_ctrl = set(rc.LIB_RISK_ID)
print("A5 CHECK")
print(f"  summaries            : {len(summaries):,}")
print(f"  risk rows (rr)       : {len(rr):,}  risks={rr.LIB_RISK_ID.nunique()}")
print(f"     with a control    : {len(set(rr.LIB_RISK_ID) & with_ctrl)}")
print(f"     WITHOUT a control : {len(set(rr.LIB_RISK_ID) - with_ctrl)}  "
      f"{sorted(set(rr.LIB_RISK_ID) - with_ctrl)[:8]}")
print(f"  risk-control (rc)    : {len(rc):,}  controls={rc.L1_LIB_CTRL.nunique()}")
print(f"  scoped instances (ri): {len(ri):,}  instances={ri.l1_control_id.nunique()}")
g = l1_raw.groupby("l1_library_control_id").name.nunique()
s = ri.groupby("L1_LIB_CTRL").l1_control_id.nunique()
both_ = pd.concat([s.rename("scoped"), g.rename("global")], axis=1).dropna()
if len(both_):
    print(f"  instances/control scoped vs global: "
          f"median {both_.scoped.median():.0f} vs {both_['global'].median():.0f} "
          f"({(both_['global']/both_.scoped).median():.1f}x)")


# %% A6. Routing --------------------------------------------------------------
def route(a):
    a = str(a).strip().title()
    if a in DIRECT_APPROACHES: return "direct"
    if a == "Standard":        return "judged"
    return UNKNOWN_APPROACH

linked = pairs.loc[pairs.decision == "LINKED", ["RECORD_ID", "REG_ID", "link_score"]].copy()
linked["RECORD_ID"] = linked.RECORD_ID.astype(str).str.strip()
linked["REG_ID"]    = linked.REG_ID.astype(str).str.strip()
if KEEP_RECORDS is not None:
    linked = linked[linked.RECORD_ID.isin({str(r) for r in KEEP_RECORDS})]

alerts_all = load_alerts(set(linked.RECORD_ID))
alerts_all["RECORD_ID"] = alerts_all.RECORD_ID.astype(str).str.strip()

a_sum = linked.merge(summaries, on="REG_ID", how="left")
orph = a_sum.loc[a_sum.SUM_ID.isna(), "REG_ID"].nunique()
a_sum = a_sum.dropna(subset=["SUM_ID"])
a_sum["branch"] = a_sum.SUM_APPROACH.map(route)
a_sum = a_sum.merge(alerts_all[["RECORD_ID", "alert_text"]], on="RECORD_ID", how="left")

print("A6 CHECK")
print(f"  linked pairs={len(linked):,}  alerts={a_sum.RECORD_ID.nunique()}  "
      f"alert-summary rows={len(a_sum):,}")
print(f"  linked regs with no published summary: {orph}")
print(f"  rows by branch: {a_sum.branch.value_counts().to_dict()}")
print(f"  approach in scope: {a_sum.SUM_APPROACH.value_counts(dropna=False).to_dict()}")
print(f"  alerts missing alert_text: {a_sum.alert_text.isna().sum()}")
print(a_sum.groupby(["RECORD_ID", "branch"]).SUM_ID.nunique().to_string())


# %% A7. Direct branch — Detailed / Enhanced ----------------------------------
direct = (a_sum[a_sum.branch == "direct"]
          .merge(rr, on=["REG_ID", "SUM_ID"], how="inner")
          .merge(rc, on=["REG_ID", "SUM_ID", "LIB_RISK_ID"], how="left"))
direct["branch"]     = "direct"
direct["decision"]   = "INHERITED"
direct["risk_score"] = 10
direct["reasoning"]  = ("Control Mapping Approach is " + direct.SUM_APPROACH.astype(str)
                        + "; risk and control mapping inherited without filtering.")

print("A7 CHECK")
print(f"  rows={len(direct):,}  alerts={direct.RECORD_ID.nunique()}  "
      f"risks={direct.LIB_RISK_ID.nunique()}  controls={direct.L1_LIB_CTRL.nunique()}")
print(f"  risks with no control: "
      f"{direct.loc[direct.L1_LIB_CTRL.isna(),'LIB_RISK_ID'].nunique()}")
if len(direct):
    print(direct.groupby("RECORD_ID").agg(summaries=("SUM_ID","nunique"),
          risks=("LIB_RISK_ID","nunique"), controls=("L1_LIB_CTRL","nunique")).to_string())


# %% A8. Judged branch — candidates, embeddings, shortlist --------------------
cand = (a_sum[a_sum.branch == "judged"]
        [["RECORD_ID", "REG_ID", "SUM_ID", "SUM_APPROACH", "link_score", "alert_text"]]
        .merge(rr, on=["REG_ID", "SUM_ID"], how="inner"))

need    = sorted(set(cand.LIB_RISK_ID) & set(hrisks.LIB_RISK_ID))
missing = sorted(set(cand.LIB_RISK_ID) - set(hrisks.LIB_RISK_ID))
rtext   = hrisks.set_index("LIB_RISK_ID").risk_text.to_dict()
per_alert, cos, atext = {}, {}, {}

if need:
    risk_vecs = embed_cache(need, [rtext[i] for i in need], "helios_library_risks")
    rpos = {i: n for n, i in enumerate(need)}
    a_ids = sorted(cand.RECORD_ID.unique())
    atext = cand.drop_duplicates("RECORD_ID").set_index("RECORD_ID").alert_text.to_dict()
    a_vecs = embed_cache(a_ids, [atext[i] for i in a_ids], "alerts")
    apos = {i: n for n, i in enumerate(a_ids)}
    for rid in a_ids:
        pool = [c for c in cand.loc[cand.RECORD_ID == rid, "LIB_RISK_ID"].unique()
                if c in rpos]
        if not pool:
            continue
        sims  = risk_vecs[[rpos[c] for c in pool]] @ a_vecs[apos[rid]]
        order = np.argsort(-sims)[:HP["top_k"]]
        per_alert[rid] = [pool[i] for i in order]
        for i in order:
            cos[(rid, pool[i])] = float(sims[i])

print("A8 CHECK")
print(f"  candidate rows={len(cand):,}  risks={cand.LIB_RISK_ID.nunique()}  "
      f"alerts={cand.RECORD_ID.nunique()}")
print(f"  risks with Helios text={len(need)}  without={len(missing)} {missing[:5]}")
if per_alert:
    pool_sz  = pd.Series({r: cand.loc[cand.RECORD_ID == r, "LIB_RISK_ID"].nunique()
                          for r in per_alert})
    short_sz = pd.Series({k: len(v) for k, v in per_alert.items()})
    print(f"  pool per alert : median={pool_sz.median():.0f} max={pool_sz.max()}")
    print(f"  shortlist      : median={short_sz.median():.0f} max={short_sz.max()}")
    print(f"  top_k bites on : {(pool_sz > HP['top_k']).sum()} of {len(pool_sz)} alerts")
    sp = [max(cos[(r,c)] for c in per_alert[r]) - min(cos[(r,c)] for c in per_alert[r])
          for r in per_alert if len(per_alert[r]) > 1]
    if sp: print(f"  cosine spread  : median={np.median(sp):.3f}")
else:
    print("  no judgeable candidates — LLM will be skipped")


# %% A9. Judge ----------------------------------------------------------------
RISK_SYSTEM_PROMPT = """ROLE:
You are a Lead Operational Risk Analyst at a Tier-1 Global Bank, expert in mapping
external regulatory developments onto an internal risk taxonomy.

CONTEXT:
An incoming regulatory alert has already been linked to one or more regulations. Those
regulations carry summaries whose control mapping was done at a STANDARD level of rigour —
a lighter-touch mapping than the Detailed and Enhanced summaries elsewhere in the estate.
The risks attached to them are therefore less reliable: some are right, some were attached
without close analysis. Confirm the ones the alert genuinely bears on.

Rules of Execution:
1. Populate "reasoning" BEFORE deciding. Reason first, then commit.
2. Evaluate EACH candidate library risk independently against the alert.
3. Assign a "decision", exactly one of:
   - "LINKED": the alert changes, constrains, or creates obligations for how this risk
     must be managed or controlled.
   - "NOT_LINKED": the risk concerns a materially different subject.
   - "INSUFFICIENT_EVIDENCE": the risk description is too generic or sparse to decide.
4. Judge on subject-matter overlap, not vocabulary overlap. A risk written in internal
   control language may still be the right risk for an alert written in legal language.
   Shared generic words ("reporting", "compliance", "customer") are not a link.
5. Judge each candidate on its own merits. The candidate list is short and was not
   exhaustively curated, so do not assume a fixed proportion should be accepted or rejected.
6. "evidence_from_alert" must be an exact verbatim substring of the alert text, or null
   if no single span justifies the decision. Never paraphrase into this field.
7. "link_score" is an integer 0-10 for how directly the alert bears on the risk.
8. Return exactly one verdict per supplied library_risk_id. Never invent an ID.
"""

RISK_SCHEMA = {"type": "json_schema", "json_schema": {
    "name": "risk_verdicts", "strict": True, "schema": {
        "type": "object", "additionalProperties": False, "required": ["items"],
        "properties": {"items": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["library_risk_id", "reasoning", "decision",
                         "link_score", "evidence_from_alert"],
            "properties": {
                "library_risk_id":     {"type": "string"},
                "reasoning":           {"type": "string"},
                "decision":            {"type": "string",
                                        "enum": ["LINKED", "NOT_LINKED",
                                                 "INSUFFICIENT_EVIDENCE"]},
                "link_score":          {"type": "integer"},
                "evidence_from_alert": {"type": ["string", "null"]}}}}}}}}


def _nrm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


async def judge_risks(alert_text, cands):
    payload = {"alert_text": alert_text,
               "library_risks": [{"library_risk_id": i, "risk_text": t} for i, t in cands]}
    msgs = [{"role": "system", "content": RISK_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    last = None
    for att in range(HP["max_retries"]):
        try:
            t = await tok.get()
            r = await oai.chat.completions.create(
                model=MODEL, messages=msgs, user=USE_CASE, temperature=0.0,
                max_tokens=HP["max_tokens"], seed=42 + att,      # vary: identical retry
                response_format=RISK_SCHEMA,                     # would re-truncate
                extra_headers={"X-HSBC-E2E-Trust-Token": t})
            u = r.usage
            USAGE.append({"kind": "chat", "in": u.prompt_tokens,
                          "out": u.completion_tokens, "ts": time.time()})
            return json.loads(r.choices[0].message.content)["items"]
        except Exception as e:
            last = e
            await asyncio.sleep(1.5 ** (att + 1))
    return [{"library_risk_id": i, "decision": "ERROR", "link_score": 0,
             "reasoning": f"n/a: {last}", "evidence_from_alert": None} for i, _ in cands]


async def run_judge():
    sem = asyncio.Semaphore(HP["concurrency"])
    async def one(rid):
        async with sem:
            items = await judge_risks(atext[rid], [(c, rtext[c]) for c in per_alert[rid]])
            rank = {c: i + 1 for i, c in enumerate(per_alert[rid])}
            out = []
            for it in items:
                ev = it.get("evidence_from_alert")
                out.append({"RECORD_ID": rid, "LIB_RISK_ID": it["library_risk_id"],
                            "decision": it["decision"],
                            "risk_score": it.get("link_score", 0),
                            "reasoning": it.get("reasoning"),
                            "evidence_from_alert": ev,
                            "evidence_grounded": bool(ev) and len(_nrm(ev)) >= 8
                                                 and _nrm(ev) in _nrm(atext[rid]),
                            "cosine": cos.get((rid, it["library_risk_id"])),
                            "cosine_rank": rank.get(it["library_risk_id"])})
            return out
    rows = []
    for f in asyncio.as_completed([asyncio.create_task(one(r)) for r in per_alert]):
        rows += await f
    return pd.DataFrame(rows)


JCOLS = ["RECORD_ID", "REG_ID", "SUM_ID", "SUM_APPROACH", "LIB_RISK_ID", "L1_LIB_CTRL",
         "link_score", "branch", "decision", "risk_score", "reasoning"]

nest_asyncio.apply()
if per_alert:
    verdicts = asyncio.get_event_loop().run_until_complete(run_judge())
    accepted = verdicts[(verdicts.decision == "LINKED") &
                        (verdicts.risk_score >= HP["min_score"])]
    judged = (cand.merge(accepted[["RECORD_ID", "LIB_RISK_ID", "decision",
                                   "risk_score", "reasoning"]],
                         on=["RECORD_ID", "LIB_RISK_ID"], how="inner")
                  .merge(rc, on=["REG_ID", "SUM_ID", "LIB_RISK_ID"], how="left"))
    judged["branch"] = "judged"
else:
    verdicts, accepted = pd.DataFrame(), pd.DataFrame()
    judged = pd.DataFrame(columns=JCOLS)

if per_alert and missing:
    unres = (cand[cand.LIB_RISK_ID.isin(set(missing))]
             .merge(rc, on=["REG_ID", "SUM_ID", "LIB_RISK_ID"], how="left"))
    unres["decision"], unres["risk_score"], unres["branch"] = "UNRESOLVED", 0, "judged"
    unres["reasoning"] = ("Library risk referenced by RegMap but absent from Helios; "
                          "no text to judge. Carried for review.")
    judged = pd.concat([judged, unres], ignore_index=True)

print("A9 CHECK")
if len(verdicts):
    print(f"  verdicts: {verdicts.decision.value_counts().to_dict()}")
    lk = verdicts[verdicts.decision == "LINKED"]
    if len(lk):
        print(f"  grounded on LINKED: {lk.evidence_grounded.mean():.1%}")
        print(f"  cosine rank of LINKED: {sorted(lk.cosine_rank.dropna().tolist())}")
    print(f"  accepted {len(accepted)} of {len(verdicts)}")
print(f"  judged rows={len(judged):,}  "
      f"risks={judged.LIB_RISK_ID.nunique() if len(judged) else 0}  "
      f"controls={judged.L1_LIB_CTRL.nunique() if len(judged) else 0}")


# %% A10. Assemble — risk grain, then control grain ---------------------------
both = pd.concat([d.reindex(columns=JCOLS) for d in (direct, judged) if len(d)],
                 ignore_index=True)

risk_grain = (both.groupby(["RECORD_ID", "LIB_RISK_ID"])
    .agg(branches=("branch", joinset), decisions=("decision", joinset),
         risk_score=("risk_score", "max"), max_link_score=("link_score", "max"),
         via_regulations=("REG_ID", joinset), via_summaries=("SUM_ID", joinset),
         via_approaches=("SUM_APPROACH", joinset), reasoning=("reasoning", "first"))
    .reset_index())
risk_grain["branch"] = np.where(risk_grain.branches.str.contains("direct"),
                         np.where(risk_grain.branches.str.contains("judged"),
                                  "both", "direct"), "judged")
risk_grain["confidence"] = np.select(
    [risk_grain.decisions.eq("UNRESOLVED"),
     risk_grain.branch.isin(["direct", "both"]),
     risk_grain.risk_score >= 8, risk_grain.risk_score >= 5],
    ["REVIEW", "HIGH", "HIGH", "MEDIUM"], default="LOW")
risk_grain = risk_grain.merge(hrisks[["LIB_RISK_ID", "risk_title", "risk_description"]],
                              on="LIB_RISK_ID", how="left")

CONF = {"REVIEW": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
ctrl_rows = both[both.L1_LIB_CTRL.notna()]
final = (ctrl_rows.groupby(["RECORD_ID", "L1_LIB_CTRL"])
    .agg(branches=("branch", joinset), best_risk_score=("risk_score", "max"),
         n_risks=("LIB_RISK_ID", "nunique"), n_summaries=("SUM_ID", "nunique"),
         via_risks=("LIB_RISK_ID", joinset), via_regulations=("REG_ID", joinset),
         via_approaches=("SUM_APPROACH", joinset)).reset_index())
final["branch"] = np.where(final.branches.str.contains("direct"),
                    np.where(final.branches.str.contains("judged"), "both", "direct"),
                    "judged")
best = (ctrl_rows[["RECORD_ID", "LIB_RISK_ID", "L1_LIB_CTRL"]].drop_duplicates()
        .merge(risk_grain[["RECORD_ID", "LIB_RISK_ID", "confidence"]],
               on=["RECORD_ID", "LIB_RISK_ID"], how="left"))
best["o"] = best.confidence.map(CONF)
best = best.groupby(["RECORD_ID", "L1_LIB_CTRL"]).o.max().reset_index()
best["confidence"] = best.o.map({v: k for k, v in CONF.items()})
final = (final.merge(best[["RECORD_ID", "L1_LIB_CTRL", "confidence"]],
                     on=["RECORD_ID", "L1_LIB_CTRL"], how="left")
              .merge(hcontrols, on="L1_LIB_CTRL", how="left"))
final["source"] = "regmap_laneA"

print("A10 CHECK")
print(f"  risks recommended   : {len(risk_grain):,}")
print(f"  controls recommended: {len(final):,}")
print(f"  risks by confidence : {risk_grain.confidence.value_counts().to_dict()}")
print(f"  controls by branch  : {final.branch.value_counts().to_dict()}")
print(f"  controls missing text: {final.control_title.isna().sum()}")


# %% A11. Explode -------------------------------------------------------------
links = both[["RECORD_ID", "LIB_RISK_ID", "L1_LIB_CTRL", "REG_ID", "SUM_ID"]].drop_duplicates()
links["has_ctrl"] = links.L1_LIB_CTRL.notna()
keep = links.groupby(["RECORD_ID", "LIB_RISK_ID"]).has_ctrl.transform("any")
links = links[links.has_ctrl | ~keep].drop(columns="has_ctrl")

laneA_exploded = (risk_grain
    .merge(links, on=["RECORD_ID", "LIB_RISK_ID"], how="left")
    .merge(hcontrols, on="L1_LIB_CTRL", how="left")
    .merge(ri, on=["REG_ID", "SUM_ID", "LIB_RISK_ID", "L1_LIB_CTRL"], how="left"))

laneA_exploded["has_control"]  = laneA_exploded.L1_LIB_CTRL.notna()
laneA_exploded["has_instance"] = laneA_exploded.l1_control_id.notna()
laneA_exploded["retired_flag"] = (
    laneA_exploded.control_title.fillna("").str.contains("RETIRE", case=False) |
    laneA_exploded.control_status.fillna("").astype(str).str.contains("RETIRE", case=False))
laneA_exploded["source"] = "regmap_laneA"

COLS = ["RECORD_ID", "LIB_RISK_ID", "risk_title", "L1_LIB_CTRL", "control_title",
        "l1_control_id", "has_control", "has_instance", "retired_flag",
        "branch", "decisions", "confidence", "risk_score", "max_link_score",
        "via_regulations", "via_summaries", "via_approaches",
        "risk_description", "control_description", "control_status", "reasoning", "source"]
laneA_exploded = laneA_exploded[[c for c in COLS if c in laneA_exploded.columns]]
laneA_exploded = laneA_exploded.sort_values(
    ["RECORD_ID", "confidence", "LIB_RISK_ID", "L1_LIB_CTRL", "l1_control_id"])

laneA_risks = (laneA_exploded.groupby(["RECORD_ID", "LIB_RISK_ID"])
    .agg(risk_title=("risk_title", "first"), branch=("branch", "first"),
         confidence=("confidence", "first"), n_controls=("L1_LIB_CTRL", "nunique"),
         n_instances=("l1_control_id", "nunique")).reset_index())

print("A11 CHECK")
print(f"  exploded rows: {len(laneA_exploded):,}")
print(laneA_exploded.groupby("RECORD_ID").agg(
        library_risks=("LIB_RISK_ID", "nunique"),
        library_controls=("L1_LIB_CTRL", "nunique"),
        instances=("l1_control_id", "nunique"),
        rows=("LIB_RISK_ID", "size")).to_string())
print(f"  risks with no control : {(laneA_risks.n_controls == 0).sum()} of {len(laneA_risks)}")
print(f"  controls with no text : "
      f"{laneA_exploded.loc[laneA_exploded.has_control & laneA_exploded.control_title.isna(),'L1_LIB_CTRL'].nunique()}")
print(f"  controls with no instance: "
      f"{laneA_exploded.loc[laneA_exploded.has_control & ~laneA_exploded.has_instance,'L1_LIB_CTRL'].nunique()}")

laneA_exploded.to_parquet(OUTP / "laneA_exploded.parquet", index=False)
laneA_risks.to_parquet(OUTP / "laneA_risks.parquet", index=False)
final.to_parquet(OUTP / "laneA_control_recommendations.parquet", index=False)
laneA_exploded.to_excel(OUTP / "laneA_exploded.xlsx", index=False)


# %% A12. Golden -> flat triples ----------------------------------------------
GOLDEN_RC_PATH = "../data/golden_risk_control.parquet"     # <-- adjust
G_RECORD, G_REG, G_DICT = "Record ID", "Regulation ID", "Risk_Control_dict"
GOLDEN_NONE_RISK_IS_POSITIVE = True    # flip if A12 shows the key sets are identical

def _id(x):
    if x is None: return None
    s = str(x).strip()
    return None if s.lower() in PLACEHOLDER else s

def parse_rc(v):
    if isinstance(v, dict): return v
    if v is None or (isinstance(v, float) and pd.isna(v)): return {}
    s = re.sub(r"array\(\s*(\[.*?\])\s*(?:,\s*dtype=[^)]*)?\)", r"\1", str(v), flags=re.S)
    try:
        d = ast.literal_eval(s.replace("nan", "None"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}

def flatten_rc(d):
    rows = []
    for risk, ctrls in (d or {}).items():
        r = _id(risk)
        if r is None: continue
        if not isinstance(ctrls, dict) or not ctrls:
            if GOLDEN_NONE_RISK_IS_POSITIVE: rows.append((r, None, None))
            continue
        for ctrl, insts in ctrls.items():
            c = _id(ctrl)
            if c is None:
                if GOLDEN_NONE_RISK_IS_POSITIVE: rows.append((r, None, None))
                continue
            seq = [] if insts is None else (
                list(insts) if hasattr(insts, "__iter__") and not isinstance(insts, str)
                else [insts])
            seq = [i for i in (_id(x) for x in seq) if i is not None]
            rows += [(r, c, i) for i in seq] if seq else [(r, c, None)]
    return rows

gc_raw = pd.read_parquet(GOLDEN_RC_PATH)
gc_raw[G_RECORD] = gc_raw[G_RECORD].astype(str).str.strip()
gold = pd.DataFrame([{"RECORD_ID": row[G_RECORD], "REG_ID": _id(row.get(G_REG)),
                      "LIB_RISK_ID": r, "L1_LIB_CTRL": c, "L1_CTRL": i}
                     for _, row in gc_raw.iterrows()
                     for r, c, i in flatten_rc(parse_rc(row[G_DICT]))]).drop_duplicates()

keysets = {}
for _, r in gc_raw.iterrows():
    keysets.setdefault(str(r[G_RECORD]).strip(), set()).update(
        {_id(k) for k in parse_rc(r[G_DICT])} - {None})

print("A12 CHECK")
print(f"  golden rows={len(gc_raw):,}  records={gold.RECORD_ID.nunique():,}  "
      f"triples={len(gold):,}")
print(f"  risks={gold.LIB_RISK_ID.nunique()}  controls={gold.L1_LIB_CTRL.nunique()}  "
      f"instances={gold.L1_CTRL.nunique()}")
print(f"  distinct risk-key sets across records: "
      f"{len({frozenset(s) for s in keysets.values()})}   (1 => pure enumeration, "
      "set GOLDEN_NONE_RISK_IS_POSITIVE = False and re-run)")


# %% A13. Precision / recall — flat, per record -------------------------------
RECORDS = sorted({str(r) for r in (KEEP_RECORDS or gold.RECORD_ID.unique())}
                 & set(gold.RECORD_ID))
pred = laneA_exploded.copy()
pred["RECORD_ID"] = pred.RECORD_ID.astype(str).str.strip()

def sets_for(df, rid, col):
    s = df.loc[df.RECORD_ID.astype(str).str.strip() == rid, col]
    return {x for x in s.dropna().astype(str).str.strip() if x.lower() not in PLACEHOLDER}

def score(p, g):
    tp, fp, fn = len(p & g), len(p - g), len(g - p)
    pr  = tp / (tp + fp) if (tp + fp) else np.nan
    rc_ = tp / (tp + fn) if (tp + fn) else np.nan
    f1  = 2 * pr * rc_ / (pr + rc_) if (pr and rc_ and pr + rc_) else np.nan
    return tp, fp, fn, pr, rc_, f1

rows = []
for level, col in [("library_risk", "LIB_RISK_ID"), ("l1_library_control", "L1_LIB_CTRL")]:
    ap, ag = set(), set()
    for rid in RECORDS:
        g, p = sets_for(gold, rid, col), sets_for(pred, rid, col)
        ag |= {(rid, x) for x in g}; ap |= {(rid, x) for x in p}
        tp, fp, fn, pr, rc_, f1 = score(p, g)
        rows.append(dict(RECORD_ID=rid, level=level, n_gold=len(g), n_laneA=len(p),
                         matched=tp, extra_fp=fp, missed_fn=fn,
                         precision=pr, recall=rc_, f1=f1))
    tp, fp, fn, pr, rc_, f1 = score(ap, ag)
    rows.append(dict(RECORD_ID="ALL", level=level, n_gold=len(ag), n_laneA=len(ap),
                     matched=tp, extra_fp=fp, missed_fn=fn,
                     precision=pr, recall=rc_, f1=f1))

res = pd.DataFrame(rows)
print("A13 CHECK")
print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
res.to_csv(OUTP / "laneA_eval_flat.csv", index=False)

for level, col in [("LIBRARY RISK", "LIB_RISK_ID"), ("L1 LIBRARY CONTROL", "L1_LIB_CTRL")]:
    print(f"\n{'='*74}\n{level}\n{'='*74}")
    for rid in RECORDS:
        g, p = sets_for(gold, rid, col), sets_for(pred, rid, col)
        print(f"\n--- {rid} ---  golden={len(g)} laneA={len(p)}")
        print(f"  matched ({len(g & p)}): {sorted(g & p)}")
        print(f"  missed  ({len(g - p)}): {sorted(g - p)}")
        print(f"  extra   ({len(p - g)}): {sorted(p - g)}")
