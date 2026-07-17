"""
Runs the full CAPTURE evaluation and writes results.json.
A single FL-session function is reused for every experiment so that all numbers
share one code path and one seeding scheme.
"""
import json, time, numpy as np
from sim_core import (make_corpus, dirichlet_partition, MLP, Tenant,
                      macro_f1, attack_success_rate, predict,
                      N_CLASSES, N_FEATURES, GLOBAL_SEED)
from sim_defense import (AGGREGATORS, CATO, Capture, selective_recovery)

TARGET, VICTIM = 3, 0            # backdoor: malware family 3 -> benign
ROUNDS = 28
LOCAL_EPOCHS = 2
LR = 0.15
BATCH = 128


# --------------------------------------------------------------------------- #
def build_env(n_tenants, alpha, mal_frac, seed, coalition=True):
    rng = np.random.default_rng(seed)
    X, y, _ = make_corpus(40000, rng)
    ntr = 32000
    Xtr, ytr = X[:ntr], y[:ntr]
    Xte, yte = X[ntr:ntr + 5000], y[ntr:ntr + 5000]
    Xval, yval = X[ntr + 5000:], y[ntr + 5000:]      # clean holdout probe
    parts = dirichlet_partition(ytr, n_tenants, alpha, rng)
    tenants = [Tenant(t, Xtr[ix], ytr[ix]) for t, ix in enumerate(parts)]
    n_mal = int(round(mal_frac * n_tenants))
    mal_ids = list(rng.choice(n_tenants, size=n_mal, replace=False)) if n_mal else []
    for k, i in enumerate(mal_ids):
        tenants[i].malicious = True
        tenants[i].coalition = 0 if coalition else k
    return tenants, mal_ids, (Xte, yte), (Xval, yval), rng


def run_session(tenants, mal_ids, test, val, rng, attack=None,
                defense="FedAvg", drift=False, rounds=ROUNDS,
                track_bound=False, keep_history=False):
    Xte, yte = test
    Xval, yval = val
    n = len(tenants)
    model = MLP().init(np.random.default_rng(GLOBAL_SEED + 1))
    cap = Capture(n, TARGET, VICTIM) if defense.startswith("CAPTURE") else None
    cato = None
    if attack and attack != "Naive":
        cato = CATO(TARGET, VICTIM, variant=attack, poison_scale=3.0)

    log = {"f1": [], "asr": [], "bound": [], "dev": []}
    history = []
    clean_ref = None
    if track_bound:
        clean_ref = MLP().init(np.random.default_rng(GLOBAL_SEED + 1))

    mal_set = set(mal_ids)
    for t in range(rounds):
        deltas, tids, feat_means = [], [], []
        # ---- honest first (attacker observes honest mean) ----
        honest_deltas = []
        for tn in tenants:
            if tn.malicious:
                continue
            lm = model.clone()
            Xd, yd = tn.X, tn.y
            if drift and t > rounds // 2:            # gradual covariate drift
                Xd = Xd + 0.015 * (t - rounds // 2) * np.sign(Xd)
            d = lm.local_train(Xd, yd, LOCAL_EPOCHS, LR, BATCH, rng)
            honest_deltas.append((tn.tid, d, Xd.mean(0)))
        if honest_deltas:
            hd = np.stack([d for (_, d, _) in honest_deltas])
            honest_mean = hd.mean(0)
            honest_norm = float(np.median(np.linalg.norm(hd, axis=1)))
        else:
            honest_mean = np.zeros(model.get_flat().size, np.float32)
            honest_norm = 1.0
        for (tid, d, fm) in honest_deltas:
            deltas.append(d); tids.append(tid); feat_means.append(fm)
        # ---- malicious ----
        for rank, i in enumerate(mal_ids):
            tn = tenants[i]
            if attack == "Naive":
                lm = model.clone()
                yflip = tn.y.copy()
                yflip[tn.y == TARGET] = VICTIM
                d = lm.local_train(tn.X, yflip, LOCAL_EPOCHS, LR, BATCH, rng) * 3.0
            elif cato is not None:
                # CATO-O: intermittent participation to thin the temporal trail
                if attack == "CATO-O" and (t + rank) % 2 == 1:
                    continue
                d = cato.craft(model, tn, honest_mean, honest_norm, rng,
                               n_colluders=len(mal_ids), colluder_rank=rank)
            else:
                lm = model.clone()
                d = lm.local_train(tn.X, tn.y, LOCAL_EPOCHS, LR, BATCH, rng)
            deltas.append(d); tids.append(i); feat_means.append(tn.X.mean(0))

        deltas = np.stack(deltas)
        order = np.argsort(tids)
        deltas, tids = deltas[order], list(np.array(tids)[order])
        feat_means = [feat_means[k] for k in order]
        weights = np.array([len(tenants[tid].y) for tid in tids], float)

        # ---- aggregate ----
        if defense in ("CAPTURE", "CAPTURE-Monitor"):
            q = weights / weights.sum()
            # Monitor mode runs the full evidence/attribution pipeline but leaves
            # aggregation untouched (FedAvg), modelling a deployment where
            # CAPTURE observes without intervening. Used to study recovery from
            # a genuinely compromised model.
            w_risk = q if defense == "CAPTURE-Monitor" else cap.agg_weights(q, tids)
            agg_delta = (deltas * w_risk[:, None]).sum(0)
            # feature extraction + evidence (uses provisional agg)
            for k, tid in enumerate(tids):
                f = cap.features(tid, deltas[k], honest_mean, honest_norm,
                                 model, w_risk[k], agg_delta, Xval, yval,
                                 t, feat_means[k])
                cap.update(tid, f, t)
            applied_w = w_risk
        else:
            agg = AGGREGATORS[defense]
            if defense == "FedAvg":
                agg_delta = agg(deltas, weights)
                applied_w = weights / weights.sum()
            else:
                agg_delta = agg(deltas)
                applied_w = np.ones(len(tids)) / len(tids)
        model.set_flat(model.get_flat() + agg_delta)

        if keep_history:
            history.append(dict(deltas=deltas.copy(),
                                weights=applied_w.copy(),
                                tenant_ids=list(tids)))
        if track_bound and clean_ref is not None:
            hd_only = [d for (tid, d, _) in honest_deltas]
            cw = np.array([len(tenants[tid].y) for (tid, _, _) in honest_deltas], float)
            clean_ref.set_flat(clean_ref.get_flat() +
                               (np.stack(hd_only) * (cw / cw.sum())[:, None]).sum(0))
            dev = float(np.linalg.norm(model.get_flat() - clean_ref.get_flat()))
            rho = float(sum(applied_w[k] for k, tid in enumerate(tids)
                            if tid in mal_set))
            log["dev"].append(dev)
            log["bound"].append(rho)                 # store rho_t; assemble later

        log["f1"].append(macro_f1(model, Xte, yte))
        log["asr"].append(attack_success_rate(model, Xte, yte, TARGET, VICTIM))

    out = dict(final_f1=log["f1"][-1], final_asr=log["asr"][-1],
               f1_curve=log["f1"], asr_curve=log["asr"])
    if defense.startswith("CAPTURE"):
        out["cap"] = cap
    if track_bound:
        out["dev"] = log["dev"]; out["rho"] = log["bound"]
    if keep_history:
        out["history"] = history; out["model"] = model
    return out


# --------------------------------------------------------------------------- #
def mean_ci(vals):
    v = np.array(vals, float)
    m = v.mean()
    ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return round(float(m), 4), round(float(ci), 4)


def detection_metrics(cap, mal_ids, n):
    y_true = np.zeros(n); y_true[list(mal_ids)] = 1
    pred = cap.flagged.astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    # AUROC from risk score
    score = cap.risk.copy()
    order = np.argsort(-score)
    yt = y_true[order]
    P, N = yt.sum(), (1 - yt).sum()
    tps = np.cumsum(yt); fps = np.cumsum(1 - yt)
    tpr = tps / (P + 1e-9); fpr = fps / (N + 1e-9)
    auroc = float(np.trapezoid(tpr, fpr)) if N > 0 and P > 0 else 1.0
    delays = [int(cap.flag_round[i]) for i in mal_ids if cap.flagged[i]]
    delay = float(np.mean(delays)) if delays else float(ROUNDS)
    return dict(precision=round(prec, 3), recall=round(rec, 3),
                f1=round(f1, 3), auroc=round(max(auroc, 0.0), 3),
                delay=round(delay, 1))


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / (len(a | b) + 1e-9)



# ========================================================================== #
#  Stage-based experiments (each writes results_<name>.json)                 #
# ========================================================================== #
import sys

def save(name, obj):
    with open(f"results_{name}.json", "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"  wrote results_{name}.json")


def stage_E1():
    print("E1 attack x defence matrix ...")
    defenses = ["FedAvg", "Median", "TrimmedMean", "Krum", "CAPTURE"]
    attacks = ["None", "Naive", "CATO-S", "CATO-C", "CATO-A", "CATO-O"]
    E1 = {d: {} for d in defenses}
    for d in defenses:
        for a in attacks:
            f1s, asrs = [], []
            for s in range(3):
                tenants, mal, test, val, rng = build_env(
                    20, 0.3, 0.0 if a == "None" else 0.3, 100 + s)
                r = run_session(tenants, mal, test, val, rng,
                                attack=None if a == "None" else a, defense=d)
                f1s.append(r["final_f1"]); asrs.append(r["final_asr"])
            E1[d][a] = dict(f1=mean_ci(f1s), asr=mean_ci(asrs))
        print("   ", d, "done")
    save("E1", E1)


def stage_E2():
    print("E2 detection ...")
    E2 = {}
    for a in ["CATO-S", "CATO-C", "CATO-A", "CATO-O"]:
        for mf in [0.1, 0.2, 0.3]:
            accs = {k: [] for k in ["precision", "recall", "f1", "auroc", "delay"]}
            jac = []
            for s in range(3):
                tenants, mal, test, val, rng = build_env(20, 0.3, mf, 200 + s)
                r = run_session(tenants, mal, test, val, rng, attack=a, defense="CAPTURE")
                dm = detection_metrics(r["cap"], mal, 20)
                for k in accs: accs[k].append(dm[k])
                groups = r["cap"].coalitions()
                jac.append(max([jaccard(g, mal) for g in groups], default=0.0))
            key = f"{a}@{int(mf*100)}"
            E2[key] = {k: round(float(np.mean(v)), 3) for k, v in accs.items()}
            E2[key]["coalition_jaccard"] = round(float(np.mean(jac)), 3)
        print("   ", a, "done")
    save("E2", E2)


def stage_E3():
    print("E3 drift safety ...")
    fp_d, fp_n = [], []
    for s in range(3):
        tenants, mal, test, val, rng = build_env(20, 0.3, 0.0, 300 + s)
        r = run_session(tenants, mal, test, val, rng, attack=None,
                        defense="CAPTURE", drift=True)
        fp_d.append(int(r["cap"].flagged.sum()))
        tenants, mal, test, val, rng = build_env(20, 0.3, 0.0, 350 + s)
        r = run_session(tenants, mal, test, val, rng, attack=None,
                        defense="CAPTURE", drift=False)
        fp_n.append(int(r["cap"].flagged.sum()))
    save("E3", dict(false_flags_drift=mean_ci(fp_d),
                    false_flags_nodrift=mean_ci(fp_n), honest_total=20))


def stage_E4():
    print("E4 recovery ...")
    rows = []
    for s in range(3):
        tenants, mal, test, val, rng = build_env(20, 0.3, 0.3, 400 + s)
        r = run_session(tenants, mal, test, val, rng, attack="CATO-C",
                        defense="CAPTURE-Monitor", keep_history=True)
        Xte, yte = test
        safe = 3
        msafe = MLP().init(np.random.default_rng(GLOBAL_SEED + 1))
        for rec in r["history"][:safe + 1]:
            agg = np.zeros_like(msafe.get_flat())
            for dlt, w, tid in zip(rec["deltas"], rec["weights"], rec["tenant_ids"]):
                agg += w * dlt
            msafe.set_flat(msafe.get_flat() + agg)
        t0 = time.time()
        m_rec, removed, res_asr, steps, n_exact = selective_recovery(
            r["history"], safe, msafe, r["cap"], Xte, yte, TARGET, VICTIM,
            asr_thresh=0.10, tenants=tenants, exact_every=4, seed=400 + s,
            cato=CATO(TARGET, VICTIM, variant="CATO-C", poison_scale=3.0),
            mal_ids=mal)
        rec_time = time.time() - t0
        t1 = time.time()
        t2, m2, te2, v2, rng2 = build_env(20, 0.3, 0.0, 400 + s)
        rclean = run_session(t2, m2, te2, v2, rng2, attack=None, defense="FedAvg")
        retrain_time = time.time() - t1
        rows.append(dict(pre_asr=r["final_asr"], pre_f1=r["final_f1"],
                         res_asr=res_asr, rec_f1=macro_f1(m_rec, Xte, yte),
                         clean_f1=rclean["final_f1"],
                         removed_frac=len(removed) / 20.0, removed_n=len(removed),
                         true_mal=len(mal), steps=steps, n_exact=n_exact,
                         rec_time=rec_time, retrain_time=retrain_time,
                         speedup=retrain_time / (rec_time + 1e-9)))
    keys = rows[0].keys()
    save("E4", {k: round(float(np.mean([r[k] for r in rows])), 4) for k in keys})


def stage_E5():
    """
    Empirical deviation vs. the conditional bound.

    Deviation is measured against the *counterfactual* model obtained by
    replaying the identical recorded per-tenant updates with the malicious
    contributions removed.  This isolates the poison-induced drift from ordinary
    stochastic training noise, which is what the theory actually bounds:
        ||M^T - M~^T|| <= eta G sum_t rho_t prod_{k>t} (1 + eta L_k).
    The server applies the aggregate directly, hence eta = 1.
    """
    print("E5 bound ...")
    tenants, mal, test, val, rng = build_env(20, 0.3, 0.3, 500)
    r = run_session(tenants, mal, test, val, rng, attack="CATO-C",
                    defense="CAPTURE", keep_history=True)
    hist = r["history"]
    mal_set = set(int(m) for m in mal)
    eta, L = 1.0, 0.05

    # rho_t = surviving malicious aggregation weight; G = max accepted norm
    rho, Gs = [], []
    for rec in hist:
        rho.append(float(sum(w for w, tid in zip(rec["weights"], rec["tenant_ids"])
                             if int(tid) in mal_set)))
        Gs.append(float(np.max(np.linalg.norm(rec["deltas"], axis=1))))
    G = float(np.max(Gs))

    # counterfactual replay: same updates, malicious removed
    m_poison = MLP().init(np.random.default_rng(GLOBAL_SEED + 1))
    m_clean = MLP().init(np.random.default_rng(GLOBAL_SEED + 1))
    dev, bound = [], []
    T = len(hist)
    for t, rec in enumerate(hist):
        agg_p = np.zeros_like(m_poison.get_flat())
        agg_c = np.zeros_like(m_clean.get_flat())
        for d, w, tid in zip(rec["deltas"], rec["weights"], rec["tenant_ids"]):
            agg_p += w * d
            if int(tid) not in mal_set:
                agg_c += w * d
        m_poison.set_flat(m_poison.get_flat() + agg_p)
        m_clean.set_flat(m_clean.get_flat() + agg_c)
        dev.append(float(np.linalg.norm(m_poison.get_flat() - m_clean.get_flat())))
        b = 0.0
        for k in range(t + 1):
            b += eta * G * rho[k] * ((1 + eta * L) ** (t - k))
        bound.append(float(b))
    tightness = [round(float(d / (b + 1e-9)), 4) for d, b in zip(dev, bound)]
    save("E5", dict(dev=[round(x, 4) for x in dev],
                    bound=[round(x, 4) for x in bound],
                    rho=[round(x, 4) for x in rho],
                    tightness=tightness, G=round(G, 4), L=L, eta=eta,
                    final_ratio=round(dev[-1] / (bound[-1] + 1e-9), 4)))


def stage_E6():
    print("E6 scalability ...")
    scal = {}
    for nt in [10, 20, 50, 100]:
        tenants, mal, test, val, rng = build_env(nt, 0.3, 0.3, 600)
        t0 = time.time()
        run_session(tenants, mal, test, val, rng, attack="CATO-C",
                    defense="CAPTURE", rounds=10)
        dt_cap = (time.time() - t0) / 10.0
        t0 = time.time()
        run_session(tenants, mal, test, val, rng, attack="CATO-C",
                    defense="FedAvg", rounds=10)
        dt_avg = (time.time() - t0) / 10.0
        scal[str(nt)] = dict(capture_s=round(dt_cap, 3), fedavg_s=round(dt_avg, 3),
                             overhead_pct=round(100 * (dt_cap - dt_avg) / (dt_avg + 1e-9), 1))
        print("   ", nt, "done")
    save("E6", scal)


def stage_E7():
    print("E7 ablation ...")
    abl = {}
    for name, dfn in [("Full CAPTURE", "CAPTURE"),
                      ("No risk-cap (FedAvg agg)", "FedAvg"),
                      ("Median only", "Median"),
                      ("Krum only", "Krum")]:
        vals_asr, vals_f1 = [], []
        for s in range(3):
            tenants, mal, test, val, rng = build_env(20, 0.3, 0.3, 700 + s)
            r = run_session(tenants, mal, test, val, rng, attack="CATO-C", defense=dfn)
            vals_asr.append(r["final_asr"]); vals_f1.append(r["final_f1"])
        abl[name] = dict(asr=mean_ci(vals_asr), f1=mean_ci(vals_f1))
    save("E7", abl)


def stage_DATA():
    rng = np.random.default_rng(GLOBAL_SEED)
    X, y, prior = make_corpus(40000, rng)
    save("DATA", dict(n=int(len(y)), n_features=N_FEATURES, n_classes=N_CLASSES,
                      class_counts=[int((y == c).sum()) for c in range(N_CLASSES)],
                      benign_frac=round(float((y == 0).mean()), 3),
                      target=TARGET, victim=VICTIM, rounds=ROUNDS))


def stage_curves():
    # representative F1/ASR curves for FedAvg vs CAPTURE under CATO-C (for a figure)
    out = {}
    for dfn in ["FedAvg", "Krum", "CAPTURE"]:
        tenants, mal, test, val, rng = build_env(20, 0.3, 0.3, 900)
        r = run_session(tenants, mal, test, val, rng, attack="CATO-C", defense=dfn)
        out[dfn] = dict(f1=r["f1_curve"], asr=r["asr_curve"])
    save("curves", out)


STAGES = {"E1": stage_E1, "E2": stage_E2, "E3": stage_E3, "E4": stage_E4,
          "E5": stage_E5, "E6": stage_E6, "E7": stage_E7,
          "DATA": stage_DATA, "curves": stage_curves}



def stage_E8():
    """Recovery fidelity vs. recomputation budget (exact_every sweep)."""
    print("E8 recovery sensitivity ...")
    out = {}
    for ee in [1, 2, 4, 8, 999]:      # 999 == stored updates only (no exact)
        rows = []
        for s in range(2):
            tenants, mal, test, val, rng = build_env(20, 0.3, 0.3, 400 + s)
            r = run_session(tenants, mal, test, val, rng, attack="CATO-C",
                            defense="CAPTURE-Monitor", keep_history=True)
            Xte, yte = test
            safe = 3
            msafe = MLP().init(np.random.default_rng(GLOBAL_SEED + 1))
            for rec in r["history"][:safe + 1]:
                agg = np.zeros_like(msafe.get_flat())
                for dlt, w, tid in zip(rec["deltas"], rec["weights"], rec["tenant_ids"]):
                    agg += w * dlt
                msafe.set_flat(msafe.get_flat() + agg)
            t0 = time.time()
            m_rec, removed, res_asr, steps, n_exact = selective_recovery(
                r["history"], safe, msafe, r["cap"], Xte, yte, TARGET, VICTIM,
                asr_thresh=0.10, tenants=tenants, exact_every=ee, seed=400 + s,
                cato=CATO(TARGET, VICTIM, variant="CATO-C", poison_scale=3.0),
                mal_ids=mal)
            dt = time.time() - t0
            rows.append(dict(res_asr=res_asr, rec_f1=macro_f1(m_rec, Xte, yte),
                             n_exact=n_exact, rec_time=dt,
                             removed_n=len(removed)))
        out[str(ee)] = {k: round(float(np.mean([r[k] for r in rows])), 4)
                        for k in rows[0]}
        print("   exact_every =", ee, "done")
    save("E8", out)


STAGES["E8"] = stage_E8


if __name__ == "__main__":
    args = sys.argv[1:] or list(STAGES)
    for a in args:
        STAGES[a]()
