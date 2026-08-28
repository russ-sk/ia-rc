# %% ---------------------------------------------------------------------------
# Lane A evaluation — precision / recall / F1 against the golden Risk_Control_dict
#
# Golden shape (one row per Record ID x Regulation ID x Obligation ID):
#   Risk_Control_dict = {LIB_RISK_ID: {L1_LIB_CTRL: [1C ids...]} | None, ...}
#   '-' appears as a placeholder at every level and is discarded.
#   A risk mapped to None is a genuine positive AT RISK LEVEL with no control mapping —
#   it must count for risk recall and must NOT be treated as a control-level negative.
#
# Scored at three levels, independently:
#   1. library risk        LR-
#   2. library control     L1C-
#   3. control instance    1C-
#
# Requires: laneA_exploded (A10i), a_sum, rc, hrisks, hcontrols.
# ---------------------------------------------------------------------------

import ast, re
import numpy as np
import pandas as pd


# %% 1. Load and parse the golden -------------------------------------------

GOLDEN_RC_PATH = "../data/golden_risk_control.parquet"   # <-- adjust

G_RECORD = "Record ID"
G_REG    = "Regulation ID"
G_SUM    = "RegMap Regulation Summary ID"
G_OBL    = "RegMap Obligation ID"
G_DICT   = "Risk_Control_dict"

BAD = {"-", "", "nan", "none", "null", "<na>"}


def _id(x):
    """Normalise an id, returning None for placeholders."""
    if x is None:
        return None
    s = str(x).strip()
    return None if s.lower() in BAD else s


def parse_rc(v):
    """Risk_Control_dict may already be a dict, or a string repr containing numpy arrays."""
    if isinstance(v, dict):
        return v
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return {}
    s = str(v)
    # array([...], dtype=object) -> [...]
    s = re.sub(r"array\(\s*(\[.*?\])\s*(?:,\s*dtype=[^)]*)?\)", r"\1", s, flags=re.S)
    s = s.replace("nan", "None")
    try:
        d = ast.literal_eval(s)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def flatten_rc(d):
    """-> list of (risk, control, instance); control/instance may be None."""
    rows = []
    for risk, ctrls in (d or {}).items():
        r = _id(risk)
        if r is None:
            continue
        if not isinstance(ctrls, dict) or not ctrls:
            rows.append((r, None, None))          # risk is golden, no control mapping
            continue
        for ctrl, insts in ctrls.items():
            c = _id(ctrl)
            if c is None:
                rows.append((r, None, None))
                continue
            seq = [] if insts is None else (list(insts) if hasattr(insts, "__iter__")
                                            and not isinstance(insts, str) else [insts])
            seq = [i for i in (_id(x) for x in seq) if i is not None]
            if not seq:
                rows.append((r, c, None))         # control is golden, no instance recorded
            else:
                rows += [(r, c, i) for i in seq]
    return rows


gc = pd.read_parquet(GOLDEN_RC_PATH)
gc[G_RECORD] = gc[G_RECORD].astype(str).str.strip()

grows = []
for _, row in gc.iterrows():
    for r, c, i in flatten_rc(parse_rc(row[G_DICT])):
        grows.append({"RECORD_ID": row[G_RECORD],
                      "REG_ID": _id(row.get(G_REG)),
                      "LIB_RISK_ID": r, "L1_LIB_CTRL": c, "L1_CTRL": i})

gold = pd.DataFrame(grows).drop_duplicates()

print(f"golden rows in file        : {len(gc):,}")
print(f"golden records             : {gold.RECORD_ID.nunique():,}")
print(f"flattened triples          : {len(gold):,}")
print(f"  distinct library risks   : {gold.LIB_RISK_ID.nunique():,}")
print(f"  distinct library controls: {gold.L1_LIB_CTRL.nunique():,}")
print(f"  distinct instances       : {gold.L1_CTRL.nunique():,}")
print(f"  risks with NO control    : {gold.loc[gold.L1_LIB_CTRL.isna(),'LIB_RISK_ID'].nunique():,}")
print(f"  controls with NO instance: "
      f"{gold.loc[gold.L1_LIB_CTRL.notna() & gold.L1_CTRL.isna(),'L1_LIB_CTRL'].nunique():,}")


# %% 2. Align the two sides --------------------------------------------------

pred = laneA_exploded.copy()
pred["RECORD_ID"] = pred.RECORD_ID.astype(str).str.strip()
pred = pred.rename(columns={"l1_control_id": "L1_CTRL"})

EVAL_RECORDS = sorted(set(gold.RECORD_ID) & set(pred.RECORD_ID))
print(f"\ngolden records          : {gold.RECORD_ID.nunique():,}")
print(f"Lane A records          : {pred.RECORD_ID.nunique():,}")
print(f"scored (in both)        : {len(EVAL_RECORDS):,}")
missed = sorted(set(gold.RECORD_ID) - set(pred.RECORD_ID))
if missed:
    print(f"[!] golden records Lane A produced nothing for: {len(missed)} {missed[:10]}")
    print("    these are recall-zero records — excluding them flatters the result,")
    print("    so both scored-only and all-golden recall are reported below")

gold_e = gold[gold.RECORD_ID.isin(EVAL_RECORDS)]
pred_e = pred[pred.RECORD_ID.isin(EVAL_RECORDS)]


def pairs(df, col, records=None):
    d = df if records is None else df[df.RECORD_ID.isin(records)]
    d = d[d[col].notna()]
    return set(zip(d.RECORD_ID, d[col]))


# %% 3. Metrics --------------------------------------------------------------

def prf(p, g):
    tp, fp, fn = len(p & g), len(p - g), len(g - p)
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec  = tp / (tp + fn) if (tp + fn) else np.nan
    f1   = 2 * prec * rec / (prec + rec) if (prec and rec and (prec + rec)) else np.nan
    jac  = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1, "jaccard": jac}


LEVELS = [("library risk", "LIB_RISK_ID"), ("library control", "L1_LIB_CTRL"),
          ("control instance", "L1_CTRL")]

print("\n" + "=" * 72)
print("MICRO — pooled over every (record, id) pair, scored records only")
print("=" * 72)
micro = []
for name, col in LEVELS:
    m = prf(pairs(pred_e, col), pairs(gold_e, col))
    m["level"] = name
    micro.append(m)
micro = pd.DataFrame(micro)[["level", "tp", "fp", "fn",
                             "precision", "recall", "f1", "jaccard"]]
print(micro.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\n" + "=" * 72)
print("MICRO — all golden records (Lane A misses count as recall zero)")
print("=" * 72)
micro_all = []
for name, col in LEVELS:
    m = prf(pairs(pred, col), pairs(gold, col))
    m["level"] = name
    micro_all.append(m)
print(pd.DataFrame(micro_all)[["level", "tp", "fp", "fn", "precision", "recall", "f1"]]
      .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\n" + "=" * 72)
print("MACRO — per record, then averaged")
print("=" * 72)
rows = []
for rid in EVAL_RECORDS:
    for name, col in LEVELS:
        p = {x for r, x in pairs(pred_e, col) if r == rid}
        g = {x for r, x in pairs(gold_e, col) if r == rid}
        m = prf(p, g)
        m.update(RECORD_ID=rid, level=name, n_gold=len(g), n_pred=len(p))
        rows.append(m)
per_record = pd.DataFrame(rows)
print(per_record.groupby("level")[["precision", "recall", "f1", "jaccard"]]
      .mean().to_string(float_format=lambda x: f"{x:.3f}"))

print("\nper record:")
print(per_record[["RECORD_ID", "level", "n_gold", "n_pred", "tp", "fp", "fn",
                  "precision", "recall", "f1"]]
      .sort_values(["RECORD_ID", "level"])
      .to_string(index=False, float_format=lambda x: f"{x:.3f}"))


# %% 4. Accuracy against a defined universe ----------------------------------
# Accuracy needs true negatives, so it needs a universe. Reported because it was
# asked for, but read it with care: the universe is large and mostly negative, so
# accuracy sits near 1.0 whatever the model does. Precision/recall/F1 above are the
# numbers that discriminate.

UNIVERSE = {
    "library risk":     set(rc.LIB_RISK_ID.dropna().unique()),
    "library control":  set(rc.L1_LIB_CTRL.dropna().unique()),
    "control instance": set(hinst.l1_control_id.dropna().unique()),
}

print("\n" + "=" * 72)
print("ACCURACY / SPECIFICITY (universe-dependent — see caveat)")
print("=" * 72)
arows = []
for name, col in LEVELS:
    U = UNIVERSE[name]
    accs, specs = [], []
    for rid in EVAL_RECORDS:
        p = {x for r, x in pairs(pred_e, col) if r == rid} & U
        g = {x for r, x in pairs(gold_e, col) if r == rid} & U
        tp, fp, fn = len(p & g), len(p - g), len(g - p)
        tn = len(U - (p | g))
        accs.append((tp + tn) / len(U))
        specs.append(tn / (tn + fp) if (tn + fp) else np.nan)
    arows.append({"level": name, "universe": len(U),
                  "accuracy": np.mean(accs), "specificity": np.mean(specs)})
print(pd.DataFrame(arows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# %% 5. Recall decomposition -------------------------------------------------
# Lane A can only ever find what the linked regulations reach. Split end-to-end
# recall into the ceiling imposed by stage 1, and how much of that ceiling Lane A
# actually converted. A low ceiling is a stage-1 problem, not a Lane A problem.

reach = (a_sum[["RECORD_ID", "REG_ID", "SUM_ID"]].drop_duplicates()
         .merge(rc, on=["REG_ID", "SUM_ID"], how="inner"))
reach["RECORD_ID"] = reach.RECORD_ID.astype(str).str.strip()

print("\n" + "=" * 72)
print("RECALL DECOMPOSITION")
print("=" * 72)
for name, col in [("library risk", "LIB_RISK_ID"), ("library control", "L1_LIB_CTRL")]:
    g   = pairs(gold_e, col)
    ceil = set(zip(reach.RECORD_ID, reach[col])) & g
    got  = pairs(pred_e, col) & g
    c_r = len(ceil) / len(g) if g else np.nan
    cond = len(got) / len(ceil) if ceil else np.nan
    print(f"\n{name}:")
    print(f"   golden pairs                       : {len(g):,}")
    print(f"   reachable via linked regulations   : {len(ceil):,}  ({c_r:.1%})  <- stage-1 ceiling")
    print(f"   of those, Lane A returned          : {len(got):,}  ({cond:.1%})  <- Lane A conversion")
    print(f"   end-to-end recall                  : {len(got)/len(g) if g else np.nan:.1%}")


# %% 6. Error listings -------------------------------------------------------

print("\n" + "=" * 72)
print("MISSES (in golden, not returned) — library control level")
print("=" * 72)
fn = pairs(gold_e, "L1_LIB_CTRL") - pairs(pred_e, "L1_LIB_CTRL")
fn_df = pd.DataFrame(sorted(fn), columns=["RECORD_ID", "L1_LIB_CTRL"])
if len(fn_df):
    fn_df = fn_df.merge(hcontrols, on="L1_LIB_CTRL", how="left")
    fn_df["reachable"] = list(zip(fn_df.RECORD_ID, fn_df.L1_LIB_CTRL))
    reachset = set(zip(reach.RECORD_ID, reach.L1_LIB_CTRL))
    fn_df["reachable"] = fn_df.reachable.isin(reachset)
    print(f"{len(fn_df):,} misses; {fn_df.reachable.sum():,} were reachable "
          "(Lane A's fault), the rest were never reachable (stage 1's)")
    print(fn_df.head(20)[["RECORD_ID", "L1_LIB_CTRL", "reachable", "control_title"]]
          .to_string(index=False))

print("\n" + "=" * 72)
print("EXTRAS (returned, not in golden) — library control level")
print("=" * 72)
fp = pairs(pred_e, "L1_LIB_CTRL") - pairs(gold_e, "L1_LIB_CTRL")
fp_df = pd.DataFrame(sorted(fp), columns=["RECORD_ID", "L1_LIB_CTRL"])
if len(fp_df):
    fp_df = (fp_df.merge(hcontrols, on="L1_LIB_CTRL", how="left")
                  .merge(pred_e[["RECORD_ID", "L1_LIB_CTRL", "branch", "confidence"]]
                         .drop_duplicates(), on=["RECORD_ID", "L1_LIB_CTRL"], how="left"))
    print(f"{len(fp_df):,} extras, by branch: {fp_df.branch.value_counts().to_dict()}")
    print(fp_df.head(20)[["RECORD_ID", "L1_LIB_CTRL", "branch", "confidence",
                          "control_title"]].to_string(index=False))
    print("\nNOTE: precision here is a LOWER BOUND. If the golden set records only the "
          "\ncontrols someone happened to map, a correct extra is scored as an error. "
          "\nHave a steward adjudicate a sample of these before quoting precision.")


# %% 7. Save -----------------------------------------------------------------
from pathlib import Path
OUTP = Path(OUT) if isinstance(OUT, str) else OUT
gold.to_parquet(OUTP / "golden_flattened.parquet", index=False)
per_record.to_parquet(OUTP / "laneA_eval_per_record.parquet", index=False)
micro.to_csv(OUTP / "laneA_eval_micro.csv", index=False)
print(f"\nsaved eval outputs -> {OUTP}")
