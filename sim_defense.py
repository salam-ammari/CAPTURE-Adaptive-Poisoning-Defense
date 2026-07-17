"""
CATO attack, robust aggregation baselines, the CAPTURE defence pipeline,
and provenance-guided selective recovery.  All operate on flat update vectors
produced by MLP.local_train in sim_core.
"""
import numpy as np
from sim_core import MLP, macro_f1, attack_success_rate, N_CLASSES


# --------------------------------------------------------------------------- #
#  Robust aggregation baselines                                               #
# --------------------------------------------------------------------------- #
def agg_fedavg(deltas, weights):
    w = weights / weights.sum()
    return (deltas * w[:, None]).sum(0)


def agg_median(deltas, weights=None):
    return np.median(deltas, axis=0)


def agg_trimmed_mean(deltas, weights=None, beta=0.2):
    n = len(deltas)
    k = int(np.floor(beta * n))
    s = np.sort(deltas, axis=0)
    if k > 0:
        s = s[k:n - k]
    return s.mean(0)


def agg_krum(deltas, weights=None, f=None):
    n = len(deltas)
    if f is None:
        f = max(1, n // 5)
    m = n - f - 2
    if m < 1:
        return deltas.mean(0)
    d2 = np.zeros((n, n))
    for i in range(n):
        d2[i] = np.sum((deltas - deltas[i]) ** 2, axis=1)
    scores = np.array([np.sort(d2[i])[1:m + 1].sum() for i in range(n)])
    return deltas[np.argmin(scores)]


AGGREGATORS = {
    "FedAvg": agg_fedavg,
    "Median": agg_median,
    "TrimmedMean": agg_trimmed_mean,
    "Krum": agg_krum,
}


# --------------------------------------------------------------------------- #
#  CATO -- collusive adaptive temporal-orchestration attack                   #
# --------------------------------------------------------------------------- #
class CATO:
    """
    Targeted backdoor: force class `target` -> `victim`.
    Malicious tenants craft an update that (i) contains the poison direction,
    (ii) is scaled to sit inside a norm band and a cosine-similarity band to the
    honest mean, (iii) is spread across rounds (rate) and, for colluding tenants,
    complementary (each carries a disjoint slice of the poison budget).
    """

    def __init__(self, target=3, victim=0, variant="CATO-C",
                 poison_scale=3.0, norm_cap=None, cos_band=0.15):
        self.target, self.victim = target, victim
        self.variant = variant
        self.poison_scale = poison_scale
        self.norm_cap = norm_cap
        self.cos_band = cos_band

    def poison_direction(self, model, Xm, ym, rng):
        """Gradient that relabels target-class samples to the victim class."""
        mask = ym == self.target
        if mask.sum() < 2:
            # fall back to a synthetic target-class batch drawn from tenant data
            idx = rng.choice(len(ym), size=min(64, len(ym)), replace=False)
            Xp, yp = Xm[idx], np.full(len(idx), self.victim)
        else:
            Xp = Xm[mask]
            yp = np.full(mask.sum(), self.victim)
        g = model.grad(Xp, yp)              # descent dir for the poisoned labels
        return -g                            # move params toward that objective

    def craft(self, model, tenant, honest_mean, honest_norm, rng,
              n_colluders=1, colluder_rank=0):
        d = self.poison_direction(model, tenant.X, tenant.y, rng)
        d = d / (np.linalg.norm(d) + 1e-9)
        # ---- variant-specific stealth profile (norm multiplier, cosine band) --
        # CATO-S : single-source slow drip, modest magnitude
        # CATO-C : complementary split across colluders (no pairwise similarity)
        # CATO-A : white-box adaptive; tightest conformity and smallest
        #          magnitude, trading per-round strength for evasion
        prof = {"CATO-S": (1.25, 0.35), "CATO-C": (1.40, 0.15),
                "CATO-A": (1.05, 0.55), "CATO-D": (1.15, 0.45),
                "CATO-X": (1.35, 0.25), "CATO-M": (1.30, 0.20),
                "CATO-O": (1.50, 0.30)}
        nmult, cband = prof.get(self.variant, (1.40, self.cos_band))
        if self.variant == "CATO-C" and n_colluders > 1:
            slab = np.zeros_like(d)
            sl = slice(colluder_rank, len(d), n_colluders)
            slab[sl] = d[sl]
            d = slab * n_colluders * 0.6 + d * 0.4
            d = d / (np.linalg.norm(d) + 1e-9)
        # scale relative to the honest norm band; persistence across rounds makes
        # the accumulated poison effective while each update stays plausible.
        target_norm = honest_norm * (nmult + 0.2 * rng.random())
        upd = d * target_norm * self.poison_scale
        # blend toward the honest mean to satisfy the cosine acceptance band
        if honest_mean is not None:
            hm = honest_mean / (np.linalg.norm(honest_mean) + 1e-9)
            cos = np.dot(upd, hm) / (np.linalg.norm(upd) + 1e-9)
            if cos < cband:
                upd = upd + (cband - cos) * np.linalg.norm(upd) * hm
        if self.norm_cap is not None:
            nrm = np.linalg.norm(upd)
            if nrm > self.norm_cap:
                upd *= self.norm_cap / nrm
        return upd.astype(np.float32)


# --------------------------------------------------------------------------- #
#  CAPTURE feature extraction + detection                                     #
# --------------------------------------------------------------------------- #
class Capture:
    """
    Provenance-linked, drift-aware, causal-influence detector with temporal
    hypergraph coalition attribution and risk-bounded aggregation.
    History of accepted per-tenant deltas is retained for exact counterfactual
    replay and for selective recovery.
    """

    def __init__(self, n_tenants, target=3, victim=0, W=8, gamma=0.9,
                 alpha=0.02, beta=6.0, w_max=0.25, rho_max=0.4, margin=0.05):
        self.n = n_tenants
        self.target, self.victim = target, victim
        self.W, self.gamma = W, gamma
        self.alpha, self.beta = alpha, beta
        self.w_max, self.rho_max = w_max, rho_max
        self.margin = margin
        # per-tenant running state
        self.E = np.ones(n_tenants)                 # sequential e-process
        self.risk = np.zeros(n_tenants)
        self.infl_hist = [[] for _ in range(n_tenants)]     # (round, influence)
        self.z_hist = [[] for _ in range(n_tenants)]        # feature rows
        self.prev_feat_mean = [None] * n_tenants
        self.flagged = np.zeros(n_tenants, bool)
        self.flag_round = np.full(n_tenants, -1)

    # ---- counterfactual influence on the suspicious behaviour ----
    def _influence(self, model_before, delta_i, w_i, agg_delta, Xval, yval):
        """
        loss on the target->victim probe after removing tenant i's weighted
        contribution, minus loss with it present.  Positive => i raises the
        harmful behaviour (drives target toward victim).
        """
        base = model_before.clone().set_flat(model_before.get_flat() + agg_delta)
        cf = model_before.clone().set_flat(
            model_before.get_flat() + agg_delta - w_i * delta_i)
        # probe = target-class samples labelled with their TRUE class;
        # a poisoned model raises loss here (it wants them to be victim).
        mask = yval == self.target
        if mask.sum() == 0:
            return 0.0
        Xp, yp = Xval[mask], yval[mask]
        return float(cf.loss(Xp, yp) - base.loss(Xp, yp)) * -1.0
        # sign: influence>0 when removing i REDUCES harm => cf.loss<base? we invert
        # so that malicious i yields positive accumulated influence.

    def features(self, i, delta_i, honest_mean, honest_norm, model_before,
                 w_i, agg_delta, Xval, yval, round_t, tenant_feat_mean):
        nrm = np.linalg.norm(delta_i)
        hm = honest_mean / (np.linalg.norm(honest_mean) + 1e-9)
        cos = float(np.dot(delta_i, hm) / (nrm + 1e-9))
        # frequency-domain energy in high band (poison tends to be low-freq/smooth
        # OR spiky; we use spectral flatness as a conformity signal)
        sp = np.abs(np.fft.rfft(delta_i))
        flat = float(np.exp(np.log(sp + 1e-9).mean()) / (sp.mean() + 1e-9))
        infl = self._influence(model_before, delta_i, w_i, agg_delta, Xval, yval)
        # drift: shift in this tenant's input feature mean vs its previous round
        if self.prev_feat_mean[i] is None:
            drift = 0.0
        else:
            drift = float(np.linalg.norm(tenant_feat_mean - self.prev_feat_mean[i]))
        self.prev_feat_mean[i] = tenant_feat_mean
        return dict(norm=nrm / (honest_norm + 1e-9), cos=cos, flat=flat,
                    infl=infl, drift=drift)

    def update(self, i, feats, round_t):
        # accumulate decayed influence over window W
        self.infl_hist[i].append((round_t, feats["infl"]))
        acc = 0.0
        for (r, v) in self.infl_hist[i]:
            if round_t - r < self.W:
                acc += (self.gamma ** (round_t - r)) * v
        # behavioural concentration proxy: positive accumulated influence
        # sequential evidence: likelihood ratio of "poison" vs "drift"
        #   poison hypothesis rewards positive accumulated influence that is
        #   NOT explained by input drift.
        drift_adj = feats["drift"]
        # margin makes the evidence net-decaying for honest tenants (whose
        # accumulated causal influence hovers near zero) while still crossing
        # the threshold for tenants that persistently raise the harmful mapping.
        s = acc - 1.2 * drift_adj - self.margin
        lr = np.exp(np.clip(1.8 * s, -4, 4))           # calibrated LR
        self.E[i] *= lr
        self.E[i] = float(np.clip(self.E[i], 1e-6, 1e8))
        self.risk[i] = 0.9 * self.risk[i] + 0.1 * max(acc, 0.0)
        self.z_hist[i].append([feats["norm"], feats["cos"], feats["flat"],
                               acc, drift_adj, round_t])
        if not self.flagged[i] and self.E[i] >= 1.0 / self.alpha:
            self.flagged[i] = True
            self.flag_round[i] = round_t
        return acc

    # ---- risk-bounded aggregation weights ----
    def agg_weights(self, q, tids=None):
        risk = self.risk if tids is None else self.risk[np.asarray(tids)]
        w = q * np.exp(-self.beta * risk)
        w = np.maximum(w, 1e-6)
        w = w / w.sum()
        w = np.minimum(w, self.w_max)
        w = w / w.sum()
        return w

    # ---- temporal hypergraph coalition attribution ----
    def coalitions(self):
        """
        Group flagged tenants that share (a) positive accumulated influence and
        (b) overlapping active rounds.  Returns list of tenant-id sets.
        """
        flagged = [i for i in range(self.n) if self.flagged[i]]
        if not flagged:
            return []
        # active-round signature per tenant
        active = {}
        for i in flagged:
            rs = set(r for (r, v) in self.infl_hist[i] if v > 0)
            active[i] = rs
        # build hyperedges via temporal + influence overlap
        groups, used = [], set()
        for i in flagged:
            if i in used:
                continue
            grp = {i}
            for j in flagged:
                if j == i or j in used:
                    continue
                inter = len(active[i] & active[j])
                union = len(active[i] | active[j]) + 1e-9
                if inter / union > 0.25:
                    grp.add(j)
            used |= grp
            groups.append(grp)
        return groups


# --------------------------------------------------------------------------- #
#  Provenance-guided selective recovery                                       #
# --------------------------------------------------------------------------- #
def selective_recovery(history, safe_round, model_at_safe, cap,
                       Xval, yval, target, victim, asr_thresh=0.10,
                       tenants=None, exact_every=4, local_epochs=2,
                       lr=0.15, batch=128, seed=0, cato=None, mal_ids=()):
    """
    Provenance-guided minimum-removal recovery.

    Rebuild from the certified-safe checkpoint, ranking tenant-round
    contributions by CAPTURE's accumulated risk and removing the smallest
    prefix of that ranking which drives ASR below `asr_thresh`.

    Stored updates are *stale*: they were computed along the poisoned
    trajectory, so replaying them verbatim after removing contributions is only
    a first-order approximation and degrades utility. Following the
    FedRecover/Crab line of work, we recompute exact local updates from the
    surviving tenants every `exact_every` rounds and reuse the stored
    (compressed) updates in between.

    Crucially, a *surviving malicious* tenant does not become honest when asked
    to recompute: it re-crafts its CATO update against the replayed model. Only
    removal, not recomputation, eliminates a compromised tenant's influence.

    Returns (recovered_model, removed_set, residual_asr, n_removal_steps,
             n_exact_recomputations).
    """
    order = np.argsort(-cap.risk)
    counter = {"exact": 0}
    mal_set = set(int(m) for m in mal_ids)

    def replay(removed_set):
        m = model_at_safe.clone()
        rng = np.random.default_rng(seed)
        n_exact = 0
        for ridx, rec in enumerate(history[safe_round + 1:]):
            survivors = [(d, w, tid) for d, w, tid
                         in zip(rec["deltas"], rec["weights"], rec["tenant_ids"])
                         if tid not in removed_set]
            if not survivors:
                continue
            exact = (tenants is not None) and (ridx % exact_every == 0)
            # honest reference statistics for re-crafting poison this round
            if exact and cato is not None:
                hn = [np.linalg.norm(d) for d, w, tid in survivors
                      if int(tid) not in mal_set]
                honest_norm = float(np.median(hn)) if hn else 1.0
                hmean = np.mean([d for d, w, tid in survivors
                                 if int(tid) not in mal_set] or [np.zeros_like(m.get_flat())], axis=0)
            agg = np.zeros_like(m.get_flat())
            wsum = 0.0
            mal_rank = 0
            for d, w, tid in survivors:
                if exact:
                    tn = tenants[tid]
                    if int(tid) in mal_set and cato is not None:
                        # attacker keeps attacking during recovery
                        d = cato.craft(m, tn, hmean, honest_norm, rng,
                                       n_colluders=max(1, len(mal_set)),
                                       colluder_rank=mal_rank)
                        mal_rank += 1
                    else:
                        lm = m.clone()
                        d = lm.local_train(tn.X, tn.y, local_epochs, lr, batch, rng)
                    n_exact += 1
                agg += w * d
                wsum += w
            if wsum > 1e-9:
                m.set_flat(m.get_flat() + agg / wsum)
        counter["exact"] = n_exact
        return m

    removed = set()
    m = replay(removed)
    asr = attack_success_rate(m, Xval, yval, target, victim)
    steps = 0
    for tid in order:
        if asr <= asr_thresh:
            break
        removed.add(int(tid))
        m = replay(removed)
        asr = attack_success_rate(m, Xval, yval, target, victim)
        steps += 1
    return m, removed, asr, steps, counter["exact"]