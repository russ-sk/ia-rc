# =============================================================================
# LANE A — FULL RUN over every record in `pairs`
# A1..A13, progress bars on embed + LLM, exploded-golden eval.
# Replaces all earlier A-cells. Paste cell by cell; each ends with a CHECK.
# Needs from RGL_Linkage_V1: DATA, OUT, clean, embed_cache, _emb_batch, oai, tok,
#   MODEL, USE_CASE, USAGE, load_alerts, nest_asyncio, and `pairs`.
# =============================================================================


# %% A1. Config + helpers -----------------------------------------------------
import re, json, time, asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 60)

REGMAP_DIR = Path(DATA) / "REGMAP"
HELIOS_DIR = Path(DATA) / "Helios"
OUTP       = Path(OUT) if isinstance(OUT, str) else OUT
OUTP.mkdir(parents=True, exist_ok=True)

STATUS_KEEP       = "published"
DIRECT_APPROACHES = {"Detailed", "Enhanced"}
UNKNOWN_APPROACH  = "judged"
KEEP_RECORDS      = None            # None = every record in `pairs`

LANE_HP = {"top_k": 200, "max_tokens": 6000, "concurrency": 10,
           "max_retries": 3, "min_score": 5}
PLACEHOLDER = {"nan", "none", "nat", "", "-", "null", "<na>"}

C = {
    "JUR": "Regulation Jurisdiction", "REG_ID": "Regulation ID",
    "REG_STATUS": "Regulation Status", "REG_VER": "Regulation Version",
    "SUM_ID": "Regulation Summary ID", "SUM_STATUS": "Regulation Summary Status",
    "SUM_VER": "Regulation Summary Version",
    "SUM_APPROACH": "Regulation Summary Control Mapping Approach",
    "SUM_TITLE": "Regulation Summary Title", "SUM_TEXT": "Regulation Summary Detail Text",
    "SUM_STEWARD": "Regulation Summary Risk Steward Area",
    "SUM_TAX_L1": "Regulation Summary Risk Taxonomy L1",
    "SUM_TAX_L2": "Regulation Summary Risk Taxonomy L2",
    "SUM_TAX_L3": "Regulation Summary Risk Taxonomy L3",
    "APPLIC_KEY": "Regulation Summary Applicability Unique Key",
    "LIB_RISK_ID": "Regulation Summary Library Risk ID Number",
    "L1_LIB_CTRL": "Regulation Summary L1 Library Control ID Number",
    "RISK_ID": "Risk ID", "RISK_APPLIC": "Risk Instance Applicable?",
    "L1_CTRL": "L1 Control ID", "L1_CTRL_APPL": "L1 Control Instance Applicable?",
}
SUMMARY_KEY       = ["JUR","REG_ID","REG_STATUS","REG_VER","SUM_ID","SUM_STATUS","SUM_VER"]
APPLICABILITY_KEY = SUMMARY_KEY + ["SUM_TAX_L1","SUM_TAX_L2","SUM_TAX_L3","APPLIC_KEY"]

def _k(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())

def resolve(df, label):
    ex = [c for c in df.columns if _k(c) == _k(label)]
    if ex: return ex[0]
    lo = [c for c in df.columns if _k(label) in _k(c)]
    if len(lo) == 1: return lo[0]
    raise KeyError(f"cannot resolve {label!r}\n  candidates: {lo or 'none'}\n  in: {list(df.columns)}")

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
    if n and not len(df): raise ValueError(f"published filter emptied — check {STATUS_KEEP!r}")
    return df.reset_index(drop=True)

def _read(directory, pat):
    hits = [p for p in Path(directory).iterdir()
            if p.suffix.lower() in {".xlsx",".xls",".csv",".parquet"}
            and re.search(pat, p.name, flags=re.I)]
    if not hits: raise FileNotFoundError(f"no {pat!r} in {directory}: "
                                         f"{[p.name for p in Path(directory).iterdir()]}")
    p = sorted(hits)[0]
    df = (pd.read_parquet(p) if p.suffix.lower()==".parquet" else
          pd.read_csv(p, dtype=str) if p.suffix.lower()==".csv" else
          pd.read_excel(p, dtype=str))
    df.columns = [str(c).strip() for c in df.columns]
    print(f"[load] {p.name}: rows={len(df):,} cols={len(df.columns)}")
    return df

read_regmap = lambda n: _read(REGMAP_DIR, rf"extract\s*_?{n}\b")
read_helios = lambda stem: _read(HELIOS_DIR, re.escape(stem))
def joinset(s): return " | ".join(sorted({str(x) for x in s if pd.notna(x)}))

print("A1 CHECK  regmap:", REGMAP_DIR.exists(), " helios:", HELIOS_DIR.exists())


# %% A1b. Drop stage-1 error records from `pairs` -----------------------------
p = pairs.copy()
p["RECORD_ID"] = p.RECORD_ID.astype(str).str.strip()

is_err = (p.decision.astype(str).str.upper().eq("ERROR")
          | p.get("reasoning", pd.Series("", index=p.index))
             .astype(str).str.strip().str.lower().str.startswith("n/a"))
err_records = set(p.loc[is_err, "RECORD_ID"])

pairs_clean = p[~p.RECORD_ID.isin(err_records)].copy()
print("A1b CHECK")
print(f"  records in pairs        : {p.RECORD_ID.nunique():,}")
print(f"  records with an ERROR   : {len(err_records)}  {sorted(err_records)[:10]}")
print(f"  records kept            : {pairs_clean.RECORD_ID.nunique():,}")
print(f"  LINKED pairs kept       : {(pairs_clean.decision=='LINKED').sum():,}")


# %% A2. RegMap E2, E3, E5 ----------------------------------------------------
e2 = standardise(read_regmap(2), SUMMARY_KEY + ["SUM_APPROACH","SUM_TITLE","SUM_TEXT",
                 "SUM_STEWARD","SUM_TAX_L1","SUM_TAX_L2","SUM_TAX_L3","LIB_RISK_ID"])
e2 = strip_ids(e2, ["REG_ID","SUM_ID","LIB_RISK_ID"]); e2["SUM_APPROACH"] = norm_approach(e2.SUM_APPROACH)
e2p = published(e2)

e3 = standardise(read_regmap(3), APPLICABILITY_KEY + ["SUM_APPROACH","LIB_RISK_ID","L1_LIB_CTRL"])
e3 = strip_ids(e3, ["REG_ID","SUM_ID","APPLIC_KEY","LIB_RISK_ID","L1_LIB_CTRL"]); e3["SUM_APPROACH"] = norm_approach(e3.SUM_APPROACH)
e3p = published(e3)

e5 = standardise(read_regmap(5), APPLICABILITY_KEY + ["SUM_APPROACH","RISK_ID","RISK_APPLIC","L1_CTRL","L1_CTRL_APPL"])
e5 = strip_ids(e5, ["REG_ID","SUM_ID","APPLIC_KEY","RISK_ID","L1_CTRL"])
e5p = published(e5)

print("A2 CHECK")
for nm,d in [("E2",e2p),("E3",e3p),("E5",e5p)]:
    print(f"  {nm}: rows={len(d):,} regs={d.REG_ID.nunique():,} summaries={d.SUM_ID.nunique():,}")


# %% A3. Helios ---------------------------------------------------------------
hrisk_raw = read_helios("risk")
l1_raw    = read_helios("L1_Control")
rtcl      = read_helios("rtcl")

hrisks = (hrisk_raw[["library_risk_id","library_risk_title","library_risk_description","risk_status"]]
          .dropna(subset=["library_risk_id"]).drop_duplicates("library_risk_id")
          .rename(columns={"library_risk_id":"LIB_RISK_ID","library_risk_title":"risk_title",
                           "library_risk_description":"risk_description"}))
hrisks["LIB_RISK_ID"] = hrisks.LIB_RISK_ID.astype(str).str.strip()
hrisks["risk_text"] = (hrisks.risk_title.fillna("")+". "+hrisks.risk_description.fillna("")).map(clean)

def pick(df,*cands,required=True):
    for lab in cands:
        ex=[c for c in df.columns if _k(c)==_k(lab)]
        if ex: return ex[0]
    for lab in cands:
        lo=[c for c in df.columns if _k(lab) in _k(c) or _k(c) in _k(lab)]
        if len(lo)==1: return lo[0]
    if required: raise KeyError(f"none of {cands} in {list(df.columns)}")
    return None

L1_LIB_ID=pick(l1_raw,"l1_library_control_id","l1_control_library_id")
L1_LIB_TITLE=pick(l1_raw,"l1_library_control_title","l1_control_library_title")
L1_LIB_DESC=pick(l1_raw,"l1_library_control_desc","l1_library_control_description","l1_control_library_description",required=False)
L1_NAME=pick(l1_raw,"name"); L1_ALIAS=pick(l1_raw,"l1_control_title_alias","title",required=False)
L1_DESC=pick(l1_raw,"description",required=False); L1_ACTIVE=pick(l1_raw,"is_active",required=False)
L1_STATUS=pick(l1_raw,"l1_control_status","status",required=False)

hcontrols=(l1_raw[[c for c in [L1_LIB_ID,L1_LIB_TITLE,L1_LIB_DESC] if c]]
           .dropna(subset=[L1_LIB_ID]).drop_duplicates(L1_LIB_ID)
           .rename(columns={L1_LIB_ID:"L1_LIB_CTRL",L1_LIB_TITLE:"control_title",
                            **({L1_LIB_DESC:"control_description"} if L1_LIB_DESC else {})}))
if "control_description" not in hcontrols.columns: hcontrols["control_description"]=np.nan
hcontrols["L1_LIB_CTRL"]=hcontrols.L1_LIB_CTRL.astype(str).str.strip()

_rt=(rtcl[["l1_control_library_id","l1_control_library_title","l1_control_library_description"]]
     .dropna(subset=["l1_control_library_id"]).drop_duplicates("l1_control_library_id")
     .rename(columns={"l1_control_library_id":"L1_LIB_CTRL","l1_control_library_title":"control_title",
                      "l1_control_library_description":"control_description"}))
_rt["L1_LIB_CTRL"]=_rt.L1_LIB_CTRL.astype(str).str.strip()
_gap=_rt[~_rt.L1_LIB_CTRL.isin(set(hcontrols.L1_LIB_CTRL))]
hcontrols=pd.concat([hcontrols,_gap],ignore_index=True)

ren={L1_LIB_ID:"L1_LIB_CTRL",L1_NAME:"l1_control_id"}
if L1_ALIAS: ren[L1_ALIAS]="instance_title"
if L1_DESC: ren[L1_DESC]="instance_description"
if L1_ACTIVE: ren[L1_ACTIVE]="is_active"
if L1_STATUS: ren[L1_STATUS]="instance_status"
hinst_text=(l1_raw[list(ren)].dropna(subset=[L1_LIB_ID,L1_NAME]).drop_duplicates().rename(columns=ren))
for c in ["L1_LIB_CTRL","l1_control_id"]: hinst_text[c]=hinst_text[c].astype(str).str.strip()

r2lr=(hrisk_raw[["name","library_risk_id"]].dropna().drop_duplicates()
      .rename(columns={"name":"RISK_ID","library_risk_id":"LIB_RISK_ID"}))
c2lc=(l1_raw[[L1_NAME,L1_LIB_ID]].dropna().drop_duplicates()
      .rename(columns={L1_NAME:"L1_CTRL",L1_LIB_ID:"L1_LIB_CTRL"}))
for d in (r2lr,c2lc):
    for c in d.columns: d[c]=d[c].astype(str).str.strip()

print("A3 CHECK")
print(f"  library risks={len(hrisks):,}  controls with text={len(hcontrols):,} "
      f"(rtcl fill={len(_gap):,})  instance rows={len(hinst_text):,}")
print(f"  RegMap ctrls with text: {len(set(e3p.L1_LIB_CTRL.dropna()) & set(hcontrols.L1_LIB_CTRL))}"
      f" / {e3p.L1_LIB_CTRL.nunique()}")


# %% A4. Base tables ----------------------------------------------------------
summaries=e2p[SUMMARY_KEY+["SUM_TITLE","SUM_TEXT","SUM_STEWARD"]].drop_duplicates()
assert summaries.SUM_ID.is_unique, "SUM_ID not unique"
appr=(e3p[["SUM_ID","SUM_APPROACH"]].drop_duplicates("SUM_ID")
      .merge(e2p[["SUM_ID","SUM_APPROACH"]].drop_duplicates("SUM_ID"),
             on="SUM_ID",how="outer",suffixes=("_e3","_e2")))
appr["SUM_APPROACH"]=appr.SUM_APPROACH_e3.fillna(appr.SUM_APPROACH_e2)
summaries=summaries.merge(appr[["SUM_ID","SUM_APPROACH"]],on="SUM_ID",how="left")

e3_all=e3p[e3p.LIB_RISK_ID.astype(str).str.startswith("LR-")]
rr=pd.concat([e3_all[["REG_ID","SUM_ID","LIB_RISK_ID"]],
              e2p.loc[e2p.LIB_RISK_ID.astype(str).str.startswith("LR-"),["REG_ID","SUM_ID","LIB_RISK_ID"]]],
             ignore_index=True).drop_duplicates()
rc=(e3p.loc[e3p.LIB_RISK_ID.astype(str).str.startswith("LR-")
            & e3p.L1_LIB_CTRL.astype(str).str.startswith("L1C-"),
            ["REG_ID","SUM_ID","LIB_RISK_ID","L1_LIB_CTRL"]].drop_duplicates())
ri=(e5p.merge(r2lr,on="RISK_ID",how="left").merge(c2lc,on="L1_CTRL",how="left"))
ri=(ri.loc[ri.LIB_RISK_ID.notna() & ri.L1_LIB_CTRL.notna(),
           ["REG_ID","SUM_ID","LIB_RISK_ID","L1_LIB_CTRL","L1_CTRL"]]
     .rename(columns={"L1_CTRL":"l1_control_id"}).drop_duplicates())

print("A4 CHECK")
print(f"  summaries={len(summaries):,}  rr={len(rr):,} (risks={rr.LIB_RISK_ID.nunique()})  "
      f"rc={len(rc):,} (ctrls={rc.L1_LIB_CTRL.nunique()})  ri={len(ri):,}")


# %% A5. Routing --------------------------------------------------------------
def route(a):
    a=str(a).strip().title()
    return "direct" if a in DIRECT_APPROACHES else "judged" if a=="Standard" else UNKNOWN_APPROACH

linked=pairs_clean.loc[pairs_clean.decision=="LINKED",["RECORD_ID","REG_ID","link_score"]].copy()
linked["REG_ID"]=linked.REG_ID.astype(str).str.strip()
if KEEP_RECORDS is not None:
    linked=linked[linked.RECORD_ID.isin({str(r) for r in KEEP_RECORDS})]

alerts_all=load_alerts(set(linked.RECORD_ID)); alerts_all["RECORD_ID"]=alerts_all.RECORD_ID.astype(str).str.strip()
a_sum=linked.merge(summaries,on="REG_ID",how="left")
orph=a_sum.loc[a_sum.SUM_ID.isna(),"REG_ID"].nunique()
a_sum=a_sum.dropna(subset=["SUM_ID"]); a_sum["branch"]=a_sum.SUM_APPROACH.map(route)
a_sum=a_sum.merge(alerts_all[["RECORD_ID","alert_text"]],on="RECORD_ID",how="left")

print("A5 CHECK")
print(f"  alerts={a_sum.RECORD_ID.nunique()}  alert-summary rows={len(a_sum):,}  "
      f"linked regs w/o summary={orph}")
print(f"  rows by branch: {a_sum.branch.value_counts().to_dict()}")
print(f"  alerts missing alert_text: {a_sum.alert_text.isna().sum()}")


# %% A6. Direct branch --------------------------------------------------------
direct=(a_sum[a_sum.branch=="direct"].merge(rr,on=["REG_ID","SUM_ID"],how="inner")
        .merge(rc,on=["REG_ID","SUM_ID","LIB_RISK_ID"],how="left"))
direct["branch"]="direct"; direct["decision"]="INHERITED"; direct["risk_score"]=10
direct["reasoning"]=("Control Mapping Approach is "+direct.SUM_APPROACH.astype(str)
                     +"; inherited without filtering.")
print("A6 CHECK")
print(f"  rows={len(direct):,}  alerts={direct.RECORD_ID.nunique()}  "
      f"risks={direct.LIB_RISK_ID.nunique()}  controls={direct.L1_LIB_CTRL.nunique()}  "
      f"risks w/o control={direct.loc[direct.L1_LIB_CTRL.isna(),'LIB_RISK_ID'].nunique()}")


# %% A7. Judged candidates + embeddings (progress) ----------------------------
if not getattr(_emb_batch, "_has_bar", False):
    _emb_batch_orig = _emb_batch
    async def _emb_batch(texts, sem):                       # noqa: F811
        out = await _emb_batch_orig(texts, sem)
        b = globals().get("_EMB_BAR")
        if b is not None: b.update(len(texts))
        return out
    _emb_batch._has_bar = True
_EMB_BAR=None
def embed_bar(ids,texts,tag,desc):
    global _EMB_BAR
    _EMB_BAR=tqdm(total=len(ids),desc=desc,unit="txt",leave=False)
    try:
        v=embed_cache(ids,texts,tag)
        if _EMB_BAR.n==0: print(f"  [{desc}] cache hit ({len(ids):,} vectors)")
        return v
    finally:
        _EMB_BAR.close(); _EMB_BAR=None

cand=(a_sum[a_sum.branch=="judged"]
      [["RECORD_ID","REG_ID","SUM_ID","SUM_APPROACH","link_score","alert_text"]]
      .merge(rr,on=["REG_ID","SUM_ID"],how="inner"))
need=sorted(set(cand.LIB_RISK_ID)&set(hrisks.LIB_RISK_ID))
missing=sorted(set(cand.LIB_RISK_ID)-set(hrisks.LIB_RISK_ID))
rtext=hrisks.set_index("LIB_RISK_ID").risk_text.to_dict()
per_alert,cos,atext={},{},{}

if need:
    risk_vecs=embed_bar(need,[rtext[i] for i in need],"helios_library_risks","embed: risks")
    rpos={i:n for n,i in enumerate(need)}
    a_ids=sorted(cand.RECORD_ID.unique())
    atext=cand.drop_duplicates("RECORD_ID").set_index("RECORD_ID").alert_text.to_dict()
    a_vecs=embed_bar(a_ids,[atext[i] for i in a_ids],"alerts","embed: alerts")
    apos={i:n for n,i in enumerate(a_ids)}
    for rid in tqdm(a_ids,desc="shortlist",unit="alert",leave=False):
        pool=[c for c in cand.loc[cand.RECORD_ID==rid,"LIB_RISK_ID"].unique() if c in rpos]
        if not pool: continue
        sims=risk_vecs[[rpos[c] for c in pool]]@a_vecs[apos[rid]]
        order=np.argsort(-sims)[:LANE_HP["top_k"]]
        per_alert[rid]=[pool[i] for i in order]
        for i in order: cos[(rid,pool[i])]=float(sims[i])

print("A7 CHECK")
print(f"  candidate rows={len(cand):,} risks={cand.LIB_RISK_ID.nunique()} "
      f"alerts={cand.RECORD_ID.nunique()}  no-text risks={len(missing)}")
if per_alert:
    ps=pd.Series({r:cand.loc[cand.RECORD_ID==r,'LIB_RISK_ID'].nunique() for r in per_alert})
    print(f"  pool/alert median={ps.median():.0f} max={ps.max()}  "
          f"top_k bites on {(ps>LANE_HP['top_k']).sum()} alerts")


# %% A8. Judge (progress) -----------------------------------------------------
RISK_SYSTEM_PROMPT="""ROLE:
You are a Lead Operational Risk Analyst at a Tier-1 Global Bank, expert in mapping
external regulatory developments onto an internal risk taxonomy.

CONTEXT:
An incoming regulatory alert has already been linked to one or more regulations. Those
regulations carry summaries whose control mapping was done at a STANDARD level of rigour —
a lighter-touch mapping than the Detailed and Enhanced summaries elsewhere. The risks
attached to them are less reliable: some are right, some were attached without close
analysis. Confirm the ones the alert genuinely bears on.

Rules of Execution:
1. Populate "reasoning" BEFORE deciding.
2. Evaluate EACH candidate library risk independently against the alert.
3. "decision" is exactly one of LINKED / NOT_LINKED / INSUFFICIENT_EVIDENCE.
4. Judge on subject-matter overlap, not shared generic words.
5. Judge each candidate on its own merits; do not assume a fixed accept/reject proportion.
6. "evidence_from_alert" is an exact verbatim substring of the alert, or null.
7. "link_score" is an integer 0-10.
8. Return exactly one verdict per supplied library_risk_id. Never invent an ID.
"""
RISK_SCHEMA={"type":"json_schema","json_schema":{"name":"risk_verdicts","strict":True,"schema":{
    "type":"object","additionalProperties":False,"required":["items"],
    "properties":{"items":{"type":"array","items":{
        "type":"object","additionalProperties":False,
        "required":["library_risk_id","reasoning","decision","link_score","evidence_from_alert"],
        "properties":{"library_risk_id":{"type":"string"},"reasoning":{"type":"string"},
            "decision":{"type":"string","enum":["LINKED","NOT_LINKED","INSUFFICIENT_EVIDENCE"]},
            "link_score":{"type":"integer"},"evidence_from_alert":{"type":["string","null"]}}}}}}}}
def _nrm(s): return re.sub(r"\s+"," ",str(s)).strip().lower()

async def judge_risks(alert_text,cands):
    payload={"alert_text":alert_text,"library_risks":[{"library_risk_id":i,"risk_text":t} for i,t in cands]}
    msgs=[{"role":"system","content":RISK_SYSTEM_PROMPT},
          {"role":"user","content":json.dumps(payload,ensure_ascii=False)}]
    last=None
    for att in range(LANE_HP["max_retries"]):
        try:
            t=await tok.get()
            r=await oai.chat.completions.create(model=MODEL,messages=msgs,user=USE_CASE,
                temperature=0.0,max_tokens=LANE_HP["max_tokens"],seed=42+att,
                response_format=RISK_SCHEMA,extra_headers={"X-HSBC-E2E-Trust-Token":t})
            u=r.usage; USAGE.append({"kind":"chat","in":u.prompt_tokens,"out":u.completion_tokens,"ts":time.time()})
            return json.loads(r.choices[0].message.content)["items"]
        except Exception as e:
            last=e; await asyncio.sleep(1.5**(att+1))
    return [{"library_risk_id":i,"decision":"ERROR","link_score":0,
             "reasoning":f"n/a: {last}","evidence_from_alert":None} for i,_ in cands]

async def run_judge():
    sem=asyncio.Semaphore(LANE_HP["concurrency"])
    bar=tqdm(total=len(per_alert),desc="judge: alerts",unit="alert")
    tally={"LINKED":0,"NOT_LINKED":0,"INSUFFICIENT_EVIDENCE":0,"ERROR":0}
    async def one(rid):
        async with sem:
            items=await judge_risks(atext[rid],[(c,rtext[c]) for c in per_alert[rid]])
            rank={c:i+1 for i,c in enumerate(per_alert[rid])}
            out=[]
            for it in items:
                ev=it.get("evidence_from_alert"); tally[it["decision"]]=tally.get(it["decision"],0)+1
                out.append({"RECORD_ID":rid,"LIB_RISK_ID":it["library_risk_id"],"decision":it["decision"],
                    "risk_score":it.get("link_score",0),"reasoning":it.get("reasoning"),
                    "evidence_from_alert":ev,
                    "evidence_grounded":bool(ev) and len(_nrm(ev))>=8 and _nrm(ev) in _nrm(atext[rid]),
                    "cosine":cos.get((rid,it["library_risk_id"])),"cosine_rank":rank.get(it["library_risk_id"])})
            bar.update(1); bar.set_postfix(linked=tally["LINKED"],rej=tally["NOT_LINKED"],err=tally["ERROR"])
            return out
    rows=[]
    try:
        for f in asyncio.as_completed([asyncio.create_task(one(r)) for r in per_alert]):
            rows+=await f
    finally: bar.close()
    return pd.DataFrame(rows)

JCOLS=["RECORD_ID","REG_ID","SUM_ID","SUM_APPROACH","LIB_RISK_ID","L1_LIB_CTRL",
       "link_score","branch","decision","risk_score","reasoning"]
nest_asyncio.apply()
if per_alert:
    verdicts=asyncio.get_event_loop().run_until_complete(run_judge())
    accepted=verdicts[(verdicts.decision=="LINKED")&(verdicts.risk_score>=LANE_HP["min_score"])]
    judged=(cand.merge(accepted[["RECORD_ID","LIB_RISK_ID","decision","risk_score","reasoning"]],
                       on=["RECORD_ID","LIB_RISK_ID"],how="inner")
                .merge(rc,on=["REG_ID","SUM_ID","LIB_RISK_ID"],how="left"))
    judged["branch"]="judged"
else:
    verdicts,accepted=pd.DataFrame(),pd.DataFrame(); judged=pd.DataFrame(columns=JCOLS)
if per_alert and missing:
    unres=(cand[cand.LIB_RISK_ID.isin(set(missing))].merge(rc,on=["REG_ID","SUM_ID","LIB_RISK_ID"],how="left"))
    unres["decision"],unres["risk_score"],unres["branch"]="UNRESOLVED",0,"judged"
    unres["reasoning"]="Referenced by RegMap, absent from Helios; carried for review."
    judged=pd.concat([judged,unres],ignore_index=True)

print("A8 CHECK")
if len(verdicts):
    print(f"  verdicts: {verdicts.decision.value_counts().to_dict()}")
    lk=verdicts[verdicts.decision=='LINKED']
    if len(lk): print(f"  grounded on LINKED: {lk.evidence_grounded.mean():.1%}")
    print(f"  accepted {len(accepted)} of {len(verdicts)}")
    verdicts.to_parquet(OUTP/"laneA_all_verdicts.parquet",index=False)
print(f"  judged rows={len(judged):,} controls={judged.L1_LIB_CTRL.nunique() if len(judged) else 0}")


# %% A9. Assemble -------------------------------------------------------------
both=pd.concat([d.reindex(columns=JCOLS) for d in (direct,judged) if len(d)],ignore_index=True)
risk_grain=(both.groupby(["RECORD_ID","LIB_RISK_ID"])
    .agg(branches=("branch",joinset),decisions=("decision",joinset),risk_score=("risk_score","max"),
         max_link_score=("link_score","max"),via_regulations=("REG_ID",joinset),
         via_summaries=("SUM_ID",joinset),via_approaches=("SUM_APPROACH",joinset),
         reasoning=("reasoning","first")).reset_index())
risk_grain["branch"]=np.where(risk_grain.branches.str.contains("direct"),
    np.where(risk_grain.branches.str.contains("judged"),"both","direct"),"judged")
risk_grain["confidence"]=np.select(
    [risk_grain.decisions.eq("UNRESOLVED"),risk_grain.branch.isin(["direct","both"]),
     risk_grain.risk_score>=8,risk_grain.risk_score>=5],
    ["REVIEW","HIGH","HIGH","MEDIUM"],default="LOW")
risk_grain=risk_grain.merge(hrisks[["LIB_RISK_ID","risk_title","risk_description"]],on="LIB_RISK_ID",how="left")

CONF={"REVIEW":0,"LOW":1,"MEDIUM":2,"HIGH":3}
ctrl_rows=both[both.L1_LIB_CTRL.notna()]
final=(ctrl_rows.groupby(["RECORD_ID","L1_LIB_CTRL"])
    .agg(branches=("branch",joinset),best_risk_score=("risk_score","max"),
         n_risks=("LIB_RISK_ID","nunique"),via_risks=("LIB_RISK_ID",joinset),
         via_regulations=("REG_ID",joinset)).reset_index())
final["branch"]=np.where(final.branches.str.contains("direct"),
    np.where(final.branches.str.contains("judged"),"both","direct"),"judged")
best=(ctrl_rows[["RECORD_ID","LIB_RISK_ID","L1_LIB_CTRL"]].drop_duplicates()
      .merge(risk_grain[["RECORD_ID","LIB_RISK_ID","confidence"]],on=["RECORD_ID","LIB_RISK_ID"],how="left"))
best["o"]=best.confidence.map(CONF)
best=best.groupby(["RECORD_ID","L1_LIB_CTRL"]).o.max().reset_index()
best["confidence"]=best.o.map({v:k for k,v in CONF.items()})
final=(final.merge(best[["RECORD_ID","L1_LIB_CTRL","confidence"]],on=["RECORD_ID","L1_LIB_CTRL"],how="left")
            .merge(hcontrols,on="L1_LIB_CTRL",how="left"))
final["source"]="regmap_laneA"
print("A9 CHECK")
print(f"  risks={len(risk_grain):,}  controls={len(final):,}  "
      f"conf={risk_grain.confidence.value_counts().to_dict()}")


# %% A10. Explode + save ------------------------------------------------------
def _col(df,n): return df[n].fillna("").astype(str) if n in df.columns else pd.Series("",index=df.index)
links=both[["RECORD_ID","LIB_RISK_ID","L1_LIB_CTRL","REG_ID","SUM_ID"]].drop_duplicates()
links["has_ctrl"]=links.L1_LIB_CTRL.notna()
keepm=links.groupby(["RECORD_ID","LIB_RISK_ID"]).has_ctrl.transform("any")
links=links[links.has_ctrl|~keepm].drop(columns="has_ctrl")

laneA_exploded=(risk_grain.merge(links,on=["RECORD_ID","LIB_RISK_ID"],how="left")
    .merge(hcontrols,on="L1_LIB_CTRL",how="left")
    .merge(ri,on=["REG_ID","SUM_ID","LIB_RISK_ID","L1_LIB_CTRL"],how="left")
    .merge(hinst_text,on=["L1_LIB_CTRL","l1_control_id"],how="left"))
laneA_exploded["has_control"]=laneA_exploded.L1_LIB_CTRL.notna()
laneA_exploded["has_instance"]=laneA_exploded.l1_control_id.notna() if "l1_control_id" in laneA_exploded.columns else False
laneA_exploded["retired_flag"]=(_col(laneA_exploded,"control_title").str.contains("RETIRE",case=False)
    |_col(laneA_exploded,"instance_title").str.contains("RETIRE",case=False)
    |_col(laneA_exploded,"instance_status").str.contains("RETIRE",case=False))
laneA_exploded["inactive_flag"]=_col(laneA_exploded,"is_active").str.strip().str.lower().isin({"false","f","n","no","0"})
laneA_exploded["source"]="regmap_laneA"
COLS=["RECORD_ID","LIB_RISK_ID","risk_title","L1_LIB_CTRL","control_title","l1_control_id",
      "instance_title","is_active","instance_status","has_control","has_instance",
      "retired_flag","inactive_flag","branch","decisions","confidence","risk_score",
      "max_link_score","via_regulations","via_summaries","via_approaches",
      "risk_description","control_description","instance_description","reasoning","source"]
laneA_exploded=laneA_exploded[[c for c in COLS if c in laneA_exploded.columns]]
laneA_exploded=laneA_exploded.sort_values([c for c in ["RECORD_ID","confidence","LIB_RISK_ID","L1_LIB_CTRL","l1_control_id"] if c in laneA_exploded.columns])
laneA_risks=(laneA_exploded.groupby(["RECORD_ID","LIB_RISK_ID"])
    .agg(risk_title=("risk_title","first"),branch=("branch","first"),confidence=("confidence","first"),
         n_controls=("L1_LIB_CTRL","nunique"),n_instances=("l1_control_id","nunique")).reset_index())
print("A10 CHECK")
print(f"  exploded rows={len(laneA_exploded):,}  records={laneA_exploded.RECORD_ID.nunique()}")
print(laneA_exploded.groupby("RECORD_ID").agg(risks=("LIB_RISK_ID","nunique"),
      controls=("L1_LIB_CTRL","nunique"),instances=("l1_control_id","nunique")).describe().to_string())
laneA_exploded.to_parquet(OUTP/"laneA_exploded_full.parquet",index=False)
laneA_risks.to_parquet(OUTP/"laneA_risks_full.parquet",index=False)
final.to_parquet(OUTP/"laneA_control_recs_full.parquet",index=False)


# %% A11. Golden (exploded) ---------------------------------------------------
GOLDEN_PATH="../data/golden_dataset_US_exploded.pkl"
G={"RECORD_ID":"Record ID","REG_ID":"Regulation ID","SUM_ID":"RegMap Regulation Summary ID",
   "OBL_ID":"RegMap Obligation ID","LIB_RISK_ID":"Regulation Summary Library Risk ID Number",
   "L1_LIB_CTRL":"Regulation Summary L1 Library Control ID Number","L1_CTRL":"L1 Control ID"}
g_raw=pd.read_pickle(GOLDEN_PATH); g_raw.columns=[str(c).strip() for c in g_raw.columns]
gren={}
for short,label in G.items():
    try: gren[resolve(g_raw,label)]=short
    except KeyError: print(f"[!] golden missing {label!r}")
gold=g_raw.rename(columns=gren)[[s for s in G if s in gren.values()]].copy()
for c in ["RECORD_ID","REG_ID","SUM_ID","OBL_ID","LIB_RISK_ID","L1_LIB_CTRL","L1_CTRL"]:
    if c in gold.columns:
        s=gold[c].astype(str).str.strip(); gold[c]=s.where(~s.str.lower().isin(PLACEHOLDER),np.nan)
gold=gold.drop_duplicates()
print("A11 CHECK")
print(f"  rows={len(gold):,} records={gold.RECORD_ID.nunique()} risks={gold.LIB_RISK_ID.nunique()} "
      f"controls={gold.L1_LIB_CTRL.nunique()} instances={gold.L1_CTRL.nunique()}")


# %% A12. Precision / recall over ALL scored records --------------------------
pred=laneA_exploded.copy(); pred["RECORD_ID"]=pred.RECORD_ID.astype(str).str.strip()
SCORED=sorted(set(pred.RECORD_ID)&set(gold.RECORD_ID))
only_gold=sorted(set(gold.RECORD_ID)-set(pred.RECORD_ID))
print(f"records scored (in both): {len(SCORED)}")
print(f"in golden but Lane A produced nothing: {len(only_gold)}")

def sets_for(df,rid,col):
    if col not in df.columns: return set()
    s=df.loc[df.RECORD_ID.astype(str).str.strip()==rid,col]
    return {x for x in s.dropna().astype(str).str.strip() if x.lower() not in PLACEHOLDER}
def score(p,g):
    tp,fp,fn=len(p&g),len(p-g),len(g-p)
    pr=tp/(tp+fp) if (tp+fp) else np.nan; rc_=tp/(tp+fn) if (tp+fn) else np.nan
    f1=2*pr*rc_/(pr+rc_) if (pr and rc_ and pr+rc_) else np.nan
    jac=tp/(tp+fp+fn) if (tp+fp+fn) else np.nan
    return tp,fp,fn,pr,rc_,f1,jac
LEVELS=[("library_risk","LIB_RISK_ID","LIB_RISK_ID"),
        ("l1_library_control","L1_LIB_CTRL","L1_LIB_CTRL"),
        ("l1_control_instance","L1_CTRL","l1_control_id")]

# MICRO (pooled over scored records) and MACRO (mean of per-record)
micro,macro_rows=[],[]
for level,gcol,pcol in LEVELS:
    ap,ag=set(),set()
    for rid in SCORED:
        g,pp=sets_for(gold,rid,gcol),sets_for(pred,rid,pcol)
        ag|={(rid,x) for x in g}; ap|={(rid,x) for x in pp}
        tp,fp,fn,pr,rc_,f1,jac=score(pp,g)
        macro_rows.append(dict(level=level,RECORD_ID=rid,precision=pr,recall=rc_,f1=f1,
                               n_gold=len(g),n_pred=len(pp)))
    tp,fp,fn,pr,rc_,f1,jac=score(ap,ag)
    micro.append(dict(level=level,tp=tp,fp=fp,fn=fn,precision=pr,recall=rc_,f1=f1,jaccard=jac))
per_record=pd.DataFrame(macro_rows)

print("\n==== MICRO (pooled over all scored records) ====")
print(pd.DataFrame(micro).to_string(index=False,float_format=lambda x:f"{x:.3f}"))
print("\n==== MACRO (per-record mean) ====")
print(per_record.groupby("level")[["precision","recall","f1"]].mean()
      .to_string(float_format=lambda x:f"{x:.3f}"))
print("\n==== recall spread across records ====")
print(per_record.groupby("level").recall.describe()[["mean","min","25%","50%","75%","max"]]
      .to_string(float_format=lambda x:f"{x:.3f}"))

pd.DataFrame(micro).to_csv(OUTP/"laneA_eval_micro_full.csv",index=False)
per_record.to_csv(OUTP/"laneA_eval_per_record_full.csv",index=False)

# worst records by control recall — where to look first
worst=(per_record[per_record.level=="l1_library_control"]
       .sort_values("recall").head(15))
print("\nlowest control recall records:")
print(worst[["RECORD_ID","n_gold","n_pred","precision","recall"]]
      .to_string(index=False,float_format=lambda x:f"{x:.3f}"))
