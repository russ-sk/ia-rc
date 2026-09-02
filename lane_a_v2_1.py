# =============================================================================
# LANE A v2.1 — CONTROLS-FIRST
#   entry:  results_with_truth.linked_rsm_ids   (RSM-level upstream)
#   target: recommend L1 Control (1C-), roll up to L1 Library Control (L1C-)
#   judge:  embed + LLM on the 1C instance text, for BOTH Detailed and Standard
#           (no direct routing for now)
#   risk layer: dropped
#
# Needs from RGL_Linkage: DATA, OUT, clean, embed_cache, _emb_batch, oai, tok,
#   MODEL, USE_CASE, USAGE, load_alerts, nest_asyncio, and `results_with_truth`.
# =============================================================================


# %% A1. Config + helpers -----------------------------------------------------
import re, json, time, asyncio, ast
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 60)

REGMAP_DIR = Path(DATA) / "REGMAP"
HELIOS_DIR = Path(DATA) / "Helios"
OUTP       = Path(OUT) if isinstance(OUT, str) else OUT
OUTP.mkdir(parents=True, exist_ok=True)

STATUS_KEEP  = "published"
KEEP_RECORDS = None                     # None = all records
PLACEHOLDER  = {"nan","none","nat","","-","null","<na>"}

LANE_HP = {"top_k": 200, "max_tokens": 6000, "concurrency": 10,
           "max_retries": 3, "min_score": 5}

C = {
    "JUR":"Regulation Jurisdiction","REG_ID":"Regulation ID",
    "REG_STATUS":"Regulation Status","REG_VER":"Regulation Version",
    "SUM_ID":"Regulation Summary ID","SUM_STATUS":"Regulation Summary Status",
    "SUM_VER":"Regulation Summary Version",
    "SUM_APPROACH":"Regulation Summary Control Mapping Approach",
    "SUM_TITLE":"Regulation Summary Title","SUM_TEXT":"Regulation Summary Detail Text",
    "SUM_TAX_L1":"Regulation Summary Risk Taxonomy L1",
    "SUM_TAX_L2":"Regulation Summary Risk Taxonomy L2",
    "SUM_TAX_L3":"Regulation Summary Risk Taxonomy L3",
    "APPLIC_KEY":"Regulation Summary Applicability Unique Key",
    "RISK_ID":"Risk ID","L1_CTRL":"L1 Control ID",
}
SUMMARY_KEY       = ["JUR","REG_ID","REG_STATUS","REG_VER","SUM_ID","SUM_STATUS","SUM_VER"]
APPLICABILITY_KEY = SUMMARY_KEY + ["SUM_TAX_L1","SUM_TAX_L2","SUM_TAX_L3","APPLIC_KEY"]

def _k(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())
def resolve(df, label):
    ex=[c for c in df.columns if _k(c)==_k(label)]
    if ex: return ex[0]
    lo=[c for c in df.columns if _k(label) in _k(c)]
    if len(lo)==1: return lo[0]
    raise KeyError(f"cannot resolve {label!r}\n  in: {list(df.columns)}")
def standardise(df, shorts):
    return df.rename(columns={resolve(df, C[s]): s for s in shorts})[shorts].copy()
def strip_ids(df, cols):
    for c in cols:
        if c in df.columns:
            s=df[c].astype(str).str.strip()
            df[c]=s.where(~s.str.lower().isin(PLACEHOLDER), np.nan)
    return df
def norm_approach(s):
    a=s.astype(str).str.strip().str.title()
    return a.where(~a.str.lower().isin(PLACEHOLDER), np.nan)
def published(df):
    n=len(df)
    for c in ("REG_STATUS","SUM_STATUS"):
        if c in df.columns:
            df=df[df[c].astype(str).str.strip().str.casefold()==STATUS_KEEP]
    if n and not len(df): raise ValueError(f"published filter emptied — check {STATUS_KEEP!r}")
    return df.reset_index(drop=True)

def _safe_parquet(path, columns=None):
    import pyarrow as pa, pyarrow.parquet as pq
    names=pq.ParquetFile(path).schema.names
    cols=[c for c in columns if c in names] if columns else names
    tbl=pq.read_table(path, columns=cols)
    fixed=[]
    for f in tbl.schema:
        col=tbl.column(f.name)
        try:
            pd.api.types.pandas_dtype(f.type.to_pandas_dtype()); fixed.append(col)
        except Exception:
            fixed.append(pa.compute.cast(col, pa.string()))
    return pa.table(fixed, names=tbl.schema.names).to_pandas()

def _read_any(path, columns=None):
    if path.suffix.lower()==".parquet":
        try: df=_safe_parquet(path, columns)
        except Exception:
            import pyarrow as pa, pyarrow.parquet as pq
            tbl=pq.read_table(path)
            df=pa.table([pa.compute.cast(tbl.column(n),pa.string()) for n in tbl.schema.names],
                        names=tbl.schema.names).to_pandas()
            if columns: df=df[[c for c in columns if c in df.columns]]
    else:
        df=(pd.read_csv(path,dtype=str) if path.suffix.lower()==".csv"
            else pd.read_excel(path,dtype=str))
        if columns: df=df[[c for c in columns if c in df.columns]]
    df.columns=[str(c).strip() for c in df.columns]
    print(f"[load] {path.name}: rows={len(df):,} cols={len(df.columns)}")
    return df

def _find(directory, stem):
    hits=[p for p in Path(directory).iterdir()
          if p.suffix.lower() in {".xlsx",".xls",".csv",".parquet"}
          and stem.lower() in p.name.lower()]
    if not hits: raise FileNotFoundError(f"no {stem!r} in {directory}")
    return sorted(hits)[0]

def read_regmap(n):
    hits=[x for x in REGMAP_DIR.iterdir()
          if x.suffix.lower() in {".xlsx",".xls",".csv",".parquet"}
          and re.search(rf"extract\s*_?{n}\b", x.name, flags=re.I)]
    return _read_any(sorted(hits)[0])
def read_helios(stem, usecols=None): return _read_any(_find(HELIOS_DIR, stem), columns=usecols)
def joinset(s): return " | ".join(sorted({str(x) for x in s if pd.notna(x)}))

def parse_id_list(v):
    """linked_rsm_ids may be a python-list string, a pipe/comma string, or a real list."""
    if isinstance(v,(list,tuple,set)): items=list(v)
    elif v is None or (isinstance(v,float) and pd.isna(v)): return []
    else:
        s=str(v).strip()
        if s[:1] in "[(" :
            try: items=list(ast.literal_eval(s))
            except Exception: items=re.split(r"[|,;]", s.strip("[]() "))
        else:
            items=re.split(r"[|,;]", s)
    out=[]
    for x in items:
        x=str(x).strip().strip("'\"")
        if x and x.lower() not in PLACEHOLDER: out.append(x)
    return out

print("A1 CHECK  regmap:", REGMAP_DIR.exists(), " helios:", HELIOS_DIR.exists())


# %% A2. RegMap E2 (summary + approach) and E5 (RSM -> 1C candidates) ---------
e2 = standardise(read_regmap(2), SUMMARY_KEY + ["SUM_APPROACH","SUM_TITLE","SUM_TEXT"])
e2 = strip_ids(e2, ["REG_ID","SUM_ID"]); e2["SUM_APPROACH"]=norm_approach(e2.SUM_APPROACH)
e2p = published(e2)

e5 = standardise(read_regmap(5), APPLICABILITY_KEY + ["SUM_APPROACH","RISK_ID","L1_CTRL"])
e5 = strip_ids(e5, ["REG_ID","SUM_ID","APPLIC_KEY","RISK_ID","L1_CTRL"])
e5p = published(e5)

print("A2 CHECK")
print(f"  E2 summaries={e2p.SUM_ID.nunique():,}")
print(f"  E5 rows={len(e5p):,}  summaries={e5p.SUM_ID.nunique():,}  "
      f"1C instances={e5p.L1_CTRL.nunique():,}")


# %% A3. Helios L1 control text + instance->library parentage ----------------
l1_raw = read_helios("L1_Control", usecols=[
    "l1_library_control_id","l1_control_library_id",
    "l1_library_control_title","l1_control_library_title",
    "l1_library_control_desc","l1_library_control_description","l1_control_library_description",
    "name","l1_control_title_alias","title","description","is_active","l1_control_status","status"])

def pick(df,*cands,required=True):
    for lab in cands:
        ex=[c for c in df.columns if _k(c)==_k(lab)]
        if ex: return ex[0]
    for lab in cands:
        lo=[c for c in df.columns if _k(lab) in _k(c) or _k(c) in _k(lab)]
        if len(lo)==1: return lo[0]
    if required: raise KeyError(f"none of {cands}")
    return None

L1_LIB_ID = pick(l1_raw,"l1_library_control_id","l1_control_library_id")
L1_NAME   = pick(l1_raw,"name")                                    # 1C-
L1_ITITLE = pick(l1_raw,"l1_control_title_alias","title",required=False)
L1_IDESC  = pick(l1_raw,"description",required=False)
L1_ACTIVE = pick(l1_raw,"is_active",required=False)
L1_STATUS = pick(l1_raw,"l1_control_status","status",required=False)
L1_LTITLE = pick(l1_raw,"l1_library_control_title","l1_control_library_title",required=False)

# instance-grain text for the 1C controls we will judge
ren={L1_NAME:"L1_CTRL", L1_LIB_ID:"L1_LIB_CTRL"}
if L1_ITITLE: ren[L1_ITITLE]="ctrl_title"
if L1_IDESC:  ren[L1_IDESC]="ctrl_description"
if L1_ACTIVE: ren[L1_ACTIVE]="is_active"
if L1_STATUS: ren[L1_STATUS]="ctrl_status"
inst = l1_raw[list(ren)].rename(columns=ren)
for c in ["L1_CTRL","L1_LIB_CTRL"]: inst[c]=inst[c].astype(str).str.strip()
inst = inst[~inst.L1_CTRL.str.lower().isin(PLACEHOLDER)].drop_duplicates("L1_CTRL")
inst["ctrl_text"] = ((inst.get("ctrl_title","").fillna("") if "ctrl_title" in inst else "")
                     + ". " +
                     (inst.get("ctrl_description","").fillna("") if "ctrl_description" in inst else "")
                    ).map(clean)

# 1C -> L1C parentage, and library title
c2lc = inst[["L1_CTRL","L1_LIB_CTRL"]].dropna().drop_duplicates()
lib_title = (l1_raw[[L1_LIB_ID]+([L1_LTITLE] if L1_LTITLE else [])]
             .rename(columns={L1_LIB_ID:"L1_LIB_CTRL",
                              **({L1_LTITLE:"lib_control_title"} if L1_LTITLE else {})})
             .drop_duplicates("L1_LIB_CTRL"))
lib_title["L1_LIB_CTRL"]=lib_title.L1_LIB_CTRL.astype(str).str.strip()

print("A3 CHECK")
print(f"  1C with text={len(inst):,}  ctrl_text populated="
      f"{(inst.ctrl_text.str.len()>0).mean():.1%}")
print(f"  E5 1C with Helios text: "
      f"{len(set(e5p.L1_CTRL.dropna()) & set(inst.L1_CTRL))} / {e5p.L1_CTRL.nunique()}")


# %% A4. Base tables ----------------------------------------------------------
summaries = e2p[["REG_ID","SUM_ID","SUM_APPROACH","SUM_TITLE","SUM_TEXT"]].drop_duplicates("SUM_ID")

# candidate 1C per (REG_ID, SUM_ID) from E5
cand_ctrl = (e5p.loc[e5p.L1_CTRL.astype(str).str.startswith("1C-"),
                     ["REG_ID","SUM_ID","L1_CTRL"]].drop_duplicates())

print("A4 CHECK")
print(f"  summaries={len(summaries):,}  "
      f"candidate (reg,sum,1C) rows={len(cand_ctrl):,}  1C={cand_ctrl.L1_CTRL.nunique():,}")
print(f"  1C per summary: mean={cand_ctrl.groupby('SUM_ID').L1_CTRL.nunique().mean():.1f} "
      f"max={cand_ctrl.groupby('SUM_ID').L1_CTRL.nunique().max()}")


# %% A5. Entry — linked RSMs from results_with_truth --------------------------
rwt = results_with_truth.copy()
rwt.columns=[str(c).strip() for c in rwt.columns]
R_REC = resolve(rwt,"RECORD_ID"); R_REG = resolve(rwt,"REG_ID")
R_DEC = resolve(rwt,"final_decision")
R_RSM = resolve(rwt,"linked_rsm_ids")
rwt[R_REC]=rwt[R_REC].astype(str).str.strip(); rwt[R_REG]=rwt[R_REG].astype(str).str.strip()

# drop stage-1 errors
is_err = rwt[R_DEC].astype(str).str.upper().eq("ERROR")
rwt_clean = rwt[~is_err]

linked = rwt_clean[rwt_clean[R_DEC].astype(str).str.upper()=="LINKED"].copy()
if KEEP_RECORDS is not None:
    linked = linked[linked[R_REC].isin({str(r) for r in KEEP_RECORDS})]

# explode linked_rsm_ids -> one row per (record, reg, summary)
rows=[]
for _,r in linked.iterrows():
    for sid in parse_id_list(r[R_RSM]):
        rows.append({"RECORD_ID":r[R_REC], "REG_ID":r[R_REG], "SUM_ID":sid})
a_sum = pd.DataFrame(rows).drop_duplicates()
a_sum = a_sum.merge(summaries, on=["REG_ID","SUM_ID"], how="left")

alerts_all = load_alerts(set(a_sum.RECORD_ID)); alerts_all["RECORD_ID"]=alerts_all.RECORD_ID.astype(str).str.strip()
a_sum = a_sum.merge(alerts_all[["RECORD_ID","alert_text"]], on="RECORD_ID", how="left")

print("A5 CHECK")
print(f"  linked (rec,reg) rows        : {len(linked):,}")
print(f"  exploded (rec,reg,summary)   : {len(a_sum):,}")
print(f"  alerts                       : {a_sum.RECORD_ID.nunique():,}")
print(f"  summaries not found in E2     : {a_sum.SUM_TITLE.isna().sum():,}")
print(f"  alert_text missing           : {a_sum.alert_text.isna().sum():,}")
print(f"  approach mix                  : {a_sum.SUM_APPROACH.value_counts(dropna=False).to_dict()}")


# %% A6. Candidates + embeddings (both approaches judged; no direct routing) --
if not getattr(_emb_batch,"_has_bar",False):
    _emb_batch_orig=_emb_batch
    async def _emb_batch(texts,sem):
        out=await _emb_batch_orig(texts,sem)
        b=globals().get("_EMB_BAR")
        if b is not None: b.update(len(texts))
        return out
    _emb_batch._has_bar=True
_EMB_BAR=None
def embed_bar(ids,texts,tag,desc):
    global _EMB_BAR
    _EMB_BAR=tqdm(total=len(ids),desc=desc,unit="txt",leave=False)
    try:
        v=embed_cache(ids,texts,tag)
        if _EMB_BAR.n==0: print(f"  [{desc}] cache hit ({len(ids):,})")
        return v
    finally:
        _EMB_BAR.close(); _EMB_BAR=None

# candidate 1C for the linked summaries
cand = (a_sum[["RECORD_ID","REG_ID","SUM_ID","SUM_APPROACH","alert_text"]]
        .merge(cand_ctrl, on=["REG_ID","SUM_ID"], how="inner"))

need    = sorted(set(cand.L1_CTRL) & set(inst.L1_CTRL))
missing = sorted(set(cand.L1_CTRL) - set(inst.L1_CTRL))
ctext   = inst.set_index("L1_CTRL").ctrl_text.to_dict()
per_alert, cos, atext = {}, {}, {}

if need:
    ctrl_vecs = embed_bar(need,[ctext[i] for i in need],"helios_l1_controls","embed: 1C controls")
    cpos={i:n for n,i in enumerate(need)}
    a_ids=sorted(cand.RECORD_ID.unique())
    atext=cand.drop_duplicates("RECORD_ID").set_index("RECORD_ID").alert_text.to_dict()
    a_vecs=embed_bar(a_ids,[atext[i] for i in a_ids],"alerts","embed: alerts")
    apos={i:n for n,i in enumerate(a_ids)}
    for rid in tqdm(a_ids,desc="shortlist",unit="alert",leave=False):
        pool=[c for c in cand.loc[cand.RECORD_ID==rid,"L1_CTRL"].unique() if c in cpos]
        if not pool: continue
        sims=ctrl_vecs[[cpos[c] for c in pool]]@a_vecs[apos[rid]]
        order=np.argsort(-sims)[:LANE_HP["top_k"]]
        per_alert[rid]=[pool[i] for i in order]
        for i in order: cos[(rid,pool[i])]=float(sims[i])

print("A6 CHECK")
print(f"  candidate (rec,sum,1C) rows={len(cand):,}  1C={cand.L1_CTRL.nunique()}  "
      f"alerts={cand.RECORD_ID.nunique()}")
print(f"  1C with Helios text={len(need)}  without={len(missing)}")
if per_alert:
    ps=pd.Series({r:cand.loc[cand.RECORD_ID==r,'L1_CTRL'].nunique() for r in per_alert})
    print(f"  pool/alert median={ps.median():.0f} max={ps.max()}  "
          f"top_k bites on {(ps>LANE_HP['top_k']).sum()} alerts")


# %% A7. Judge the 1C controls ------------------------------------------------
CTRL_SYSTEM_PROMPT="""ROLE:
You are a Lead Operational Risk & Controls Analyst at a Tier-1 Global Bank, expert in
deciding whether an internal control is affected by an external regulatory change.

CONTEXT:
An incoming regulatory alert has already been linked to one or more regulation summaries.
Each summary carries candidate L1 controls. Decide which of these controls the alert
genuinely bears on — i.e. controls whose design or operation the alert would change,
constrain, or create obligations for.

Rules of Execution:
1. Populate "reasoning" BEFORE deciding.
2. Evaluate EACH candidate control independently against the alert.
3. "decision" is exactly one of LINKED / NOT_LINKED / INSUFFICIENT_EVIDENCE.
4. Judge on whether the alert affects what this control must do — not on shared vocabulary.
   A control written in procedural language may still be affected by an alert in legal language.
5. "evidence_from_alert" is an exact verbatim substring of the alert, or null.
6. "link_score" is an integer 0-10 for how directly the alert bears on the control.
7. Return exactly one verdict per supplied l1_control_id. Never invent an ID.
"""
CTRL_SCHEMA={"type":"json_schema","json_schema":{"name":"ctrl_verdicts","strict":True,"schema":{
    "type":"object","additionalProperties":False,"required":["items"],
    "properties":{"items":{"type":"array","items":{
        "type":"object","additionalProperties":False,
        "required":["l1_control_id","reasoning","decision","link_score","evidence_from_alert"],
        "properties":{"l1_control_id":{"type":"string"},"reasoning":{"type":"string"},
            "decision":{"type":"string","enum":["LINKED","NOT_LINKED","INSUFFICIENT_EVIDENCE"]},
            "link_score":{"type":"integer"},"evidence_from_alert":{"type":["string","null"]}}}}}}}}
def _nrm(s): return re.sub(r"\s+"," ",str(s)).strip().lower()

async def judge_ctrls(alert_text,cands):
    payload={"alert_text":alert_text,"controls":[{"l1_control_id":i,"control_text":t} for i,t in cands]}
    msgs=[{"role":"system","content":CTRL_SYSTEM_PROMPT},
          {"role":"user","content":json.dumps(payload,ensure_ascii=False)}]
    last=None
    for att in range(LANE_HP["max_retries"]):
        try:
            t=await tok.get()
            r=await oai.chat.completions.create(model=MODEL,messages=msgs,user=USE_CASE,
                temperature=0.0,max_tokens=LANE_HP["max_tokens"],seed=42+att,
                response_format=CTRL_SCHEMA,extra_headers={"X-HSBC-E2E-Trust-Token":t})
            u=r.usage; USAGE.append({"kind":"chat","in":u.prompt_tokens,"out":u.completion_tokens,"ts":time.time()})
            return json.loads(r.choices[0].message.content)["items"]
        except Exception as e:
            last=e; await asyncio.sleep(1.5**(att+1))
    return [{"l1_control_id":i,"decision":"ERROR","link_score":0,
             "reasoning":f"n/a: {last}","evidence_from_alert":None} for i,_ in cands]

async def run_judge():
    sem=asyncio.Semaphore(LANE_HP["concurrency"])
    bar=tqdm(total=len(per_alert),desc="judge: alerts",unit="alert")
    tally={"LINKED":0,"NOT_LINKED":0,"INSUFFICIENT_EVIDENCE":0,"ERROR":0}
    async def one(rid):
        async with sem:
            items=await judge_ctrls(atext[rid],[(c,ctext[c]) for c in per_alert[rid]])
            rank={c:i+1 for i,c in enumerate(per_alert[rid])}
            out=[]
            for it in items:
                ev=it.get("evidence_from_alert"); tally[it["decision"]]=tally.get(it["decision"],0)+1
                out.append({"RECORD_ID":rid,"L1_CTRL":it["l1_control_id"],"decision":it["decision"],
                    "ctrl_score":it.get("link_score",0),"reasoning":it.get("reasoning"),
                    "evidence_from_alert":ev,
                    "evidence_grounded":bool(ev) and len(_nrm(ev))>=8 and _nrm(ev) in _nrm(atext[rid]),
                    "cosine":cos.get((rid,it["l1_control_id"])),"cosine_rank":rank.get(it["l1_control_id"])})
            bar.update(1); bar.set_postfix(linked=tally["LINKED"],rej=tally["NOT_LINKED"],err=tally["ERROR"])
            return out
    rows=[]
    try:
        for f in asyncio.as_completed([asyncio.create_task(one(r)) for r in per_alert]):
            rows+=await f
    finally: bar.close()
    return pd.DataFrame(rows)

nest_asyncio.apply()
if per_alert:
    verdicts=asyncio.get_event_loop().run_until_complete(run_judge())
    accepted=verdicts[(verdicts.decision=="LINKED")&(verdicts.ctrl_score>=LANE_HP["min_score"])]
    verdicts.to_parquet(OUTP/"laneA_v21_all_verdicts.parquet",index=False)
else:
    verdicts,accepted=pd.DataFrame(),pd.DataFrame()

print("A7 CHECK")
if len(verdicts):
    print(f"  verdicts: {verdicts.decision.value_counts().to_dict()}")
    lk=verdicts[verdicts.decision=='LINKED']
    if len(lk): print(f"  grounded on LINKED: {lk.evidence_grounded.mean():.1%}")
    print(f"  accepted {len(accepted)} of {len(verdicts)}  ->  1C={accepted.L1_CTRL.nunique() if len(accepted) else 0}")


# %% A8. Assemble + roll up to L1 Library Control -----------------------------
# recommended 1C, joined back to the (record, reg, summary, approach) that surfaced them
rec_1c = (accepted.merge(
            cand[["RECORD_ID","REG_ID","SUM_ID","SUM_APPROACH","L1_CTRL"]].drop_duplicates(),
            on=["RECORD_ID","L1_CTRL"], how="left")
          if len(accepted) else pd.DataFrame(
            columns=["RECORD_ID","L1_CTRL","ctrl_score","reasoning","REG_ID","SUM_ID","SUM_APPROACH"]))

# roll up: parent L1C from Helios, any recommended instance -> parent recommended
rec_1c = rec_1c.merge(c2lc, on="L1_CTRL", how="left")

# 1C-grain output
laneA_1c = (rec_1c.merge(inst[["L1_CTRL","ctrl_title","ctrl_status","is_active"]]
                         if set(["ctrl_title","ctrl_status","is_active"]).issubset(inst.columns)
                         else inst[["L1_CTRL"]], on="L1_CTRL", how="left"))
laneA_1c["source"]="regmap_laneA_v21"

# L1C-grain output (rolled up)
laneA_l1c = (rec_1c.dropna(subset=["L1_LIB_CTRL"])
             .groupby(["RECORD_ID","L1_LIB_CTRL"])
             .agg(n_instances=("L1_CTRL","nunique"),
                  best_score=("ctrl_score","max"),
                  via_summaries=("SUM_ID",joinset),
                  via_regulations=("REG_ID",joinset),
                  approaches=("SUM_APPROACH",joinset),
                  instances=("L1_CTRL",joinset)).reset_index()
             .merge(lib_title, on="L1_LIB_CTRL", how="left"))
laneA_l1c["source"]="regmap_laneA_v21"

print("A8 CHECK")
print(f"  recommended 1C rows={len(laneA_1c):,}  distinct 1C={laneA_1c.L1_CTRL.nunique()}")
print(f"  rolled-up L1C       ={laneA_l1c.L1_LIB_CTRL.nunique()}")
print(f"  1C with no parent L1C in Helios: "
      f"{rec_1c.L1_LIB_CTRL.isna().sum()}  (cannot roll up)")
per = laneA_1c.groupby("RECORD_ID").L1_CTRL.nunique()
if len(per): print(f"  1C per record: median={per.median():.0f} p90={per.quantile(.9):.0f} max={per.max()}")

laneA_1c.to_parquet(OUTP/"laneA_v21_controls.parquet",index=False)
laneA_l1c.to_parquet(OUTP/"laneA_v21_libcontrols.parquet",index=False)


# %% A9. Golden ---------------------------------------------------------------
GOLDEN_PATH="../data/golden_dataset_US_exploded_31aug26.parquet"
G={"RECORD_ID":"Record ID","REG_ID":"Regulation ID",
   "SUM_ID":"RegMap Regulation Summary ID",
   "L1_LIB_CTRL":"Regulation Summary L1 Library Control ID Number",
   "L1_CTRL":"L1 Control ID"}
g_raw=pd.read_parquet(GOLDEN_PATH); g_raw.columns=[str(c).strip() for c in g_raw.columns]
gren={}
for short,label in G.items():
    try: gren[resolve(g_raw,label)]=short
    except KeyError: print(f"[!] golden missing {label!r}")
gold=g_raw.rename(columns=gren)[[s for s in G if s in gren.values()]].copy()
for c in gold.columns:
    s=gold[c].astype(str).str.strip(); gold[c]=s.where(~s.str.lower().isin(PLACEHOLDER),np.nan)
gold=gold.drop_duplicates()
print("A9 CHECK")
print(f"  rows={len(gold):,} records={gold.RECORD_ID.nunique()} "
      f"1C={gold.L1_CTRL.nunique()} L1C={gold.L1_LIB_CTRL.nunique()}")


# %% A10. Precision / recall — 1C and L1C, all golden records -----------------
pred_1c  = laneA_1c.copy();  pred_1c["RECORD_ID"]=pred_1c.RECORD_ID.astype(str).str.strip()
pred_l1c = laneA_l1c.copy(); pred_l1c["RECORD_ID"]=pred_l1c.RECORD_ID.astype(str).str.strip()
gold["RECORD_ID"]=gold.RECORD_ID.astype(str).str.strip()

def sets_for(df,rid,col):
    if col not in df.columns: return set()
    s=df.loc[df.RECORD_ID.astype(str).str.strip()==rid,col]
    return {x for x in s.dropna().astype(str).str.strip() if x.lower() not in PLACEHOLDER}
def score(p,g):
    tp,fp,fn=len(p&g),len(p-g),len(g-p)
    pr=tp/(tp+fp) if (tp+fp) else np.nan; rc=tp/(tp+fn) if (tp+fn) else np.nan
    f1=2*pr*rc/(pr+rc) if (pr and rc and pr+rc) else np.nan
    return tp,fp,fn,pr,rc,f1

ALL_GOLD=sorted(set(gold.RECORD_ID))
LEVELS=[("L1 Control","L1_CTRL",pred_1c,"L1_CTRL"),
        ("L1 Library Control","L1_LIB_CTRL",pred_l1c,"L1_LIB_CTRL")]

micro,per_rec=[],[]
for level,gcol,pdf,pcol in LEVELS:
    ap,ag=set(),set()
    for rid in ALL_GOLD:
        g=sets_for(gold,rid,gcol); p=sets_for(pdf,rid,pcol)
        ap|={(rid,x) for x in p}; ag|={(rid,x) for x in g}
        tp,fp,fn,pr,rc,f1=score(p,g)
        per_rec.append(dict(level=level,RECORD_ID=rid,precision=pr,recall=rc,
                            recall0=(rc if not np.isnan(rc) else (0.0 if g else np.nan)),
                            n_gold=len(g),n_pred=len(p)))
    tp,fp,fn,pr,rc,f1=score(ap,ag)
    micro.append(dict(level=level,tp=tp,fp=fp,fn=fn,precision=pr,recall=rc,f1=f1))
per_record=pd.DataFrame(per_rec)

print("A10 CHECK")
print("\n== MICRO (all golden records) ==")
print(pd.DataFrame(micro).to_string(index=False,float_format=lambda x:f"{x:.3f}"))
print("\n== MACRO (per-record mean; recall_all folds empties as 0) ==")
print(per_record.groupby("level").agg(precision=("precision","mean"),
      recall_scored=("recall","mean"),recall_all=("recall0","mean"))
      .to_string(float_format=lambda x:f"{x:.3f}"))
pd.DataFrame(micro).to_csv(OUTP/"laneA_v21_eval_micro.csv",index=False)
per_record.to_csv(OUTP/"laneA_v21_eval_per_record.csv",index=False)


# %% A11. Scorecard -----------------------------------------------------------
rows=[]
for level,gcol,pdf,pcol in LEVELS:
    gp={(r,x) for r in ALL_GOLD for x in sets_for(gold,r,gcol)}
    pp={(r,x) for r in set(pdf.RECORD_ID) for x in sets_for(pdf,r,pcol)}
    val=len(gp&pp); miss=len(gp-pp); new=len(pp-gp); gn=len(gp)
    rg={r for r,_ in gp}; rh={r for r,_ in (gp&pp)}
    rows.append({"Level":level,"Golden links":gn,"Validated (TP)":val,"Missed (FN)":miss,
                 "Recall":f"{val/gn*100:.0f}%" if gn else "-","New/Review":new,
                 "Alert coverage":f"{len(rh)}/{len(rg)}"})
scorecard=pd.DataFrame(rows)
print("="*90); print("A11 — LANE A v2.1 SCORECARD (controls-first)"); print("="*90)
print(scorecard.to_string(index=False))

# roll-up consistency: every recommended L1C must have >=1 recommended 1C under it
orphan_l1c = set(pred_l1c[["RECORD_ID","L1_LIB_CTRL"]].itertuples(index=False,name=None))
child_ok   = {(r,c) for r,c in rec_1c[["RECORD_ID","L1_LIB_CTRL"]].dropna().itertuples(index=False,name=None)}
print(f"\nroll-up consistency: L1C with no recommended 1C child = {len(orphan_l1c - child_ok)} "
      "(should be 0)")
scorecard.to_csv(OUTP/"laneA_v21_scorecard.csv",index=False)
