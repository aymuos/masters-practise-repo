import numpy as np
from scipy.signal import welch

# ---------- Core Utilities ----------
def zero_cross(sp,pv):
    return np.sum(np.diff(np.sign(sp - pv)) != 0)

def oscillation_index(sp,pv):
    return np.mean(sp - pv) / (np.mean(sp) + 1e-9)

def overshoot_ratio(sp,pv):
    return abs(np.mean(abs(sp - pv)) / (np.mean(sp - pv) + 1e-9))

def PSD_features(pv, fs=1.0):
    f, Pxx = welch(pv - np.mean(pv), fs=1.0, nperseg=min(256,len(pv)))
    total = np.sum(Pxx) + 1e-9
    peak_ratio = np.max(Pxx) / total
    hf_ratio = np.sum(Pxx[int(0.2*len(f)) : ]) / total
    return peak_ratio, hf_ratio

def autocorr_peak(pv, lag=200):
    pv = pv - np.mean(pv)
    r = np.correlate(pv, pv, mode="full")
    mid = len(r)//2
    ac = r[mid+1 : mid+1+lag] / (r[mid] + 1e-9)
    return np.max(ac)

def sudden_spikes(x):
    q1,q3 = np.percentile(x,[25,75])
    iqr = q3-q1
    return np.sum((x > q3 + 2*iqr) | (x < q1 - 2*iqr))

def stiction_flatness(op):
    return np.mean(np.abs(np.diff(op)) < 1e-4)

def stiction_dead_slope(sp,pv,op,win):
    slopes=[]
    for i in range(0, len(op)-win, win):
        x = op[i:i+win]
        y = pv[i:i+win]
        if np.std(x) < 1e-9: slopes.append(0)
        else: slopes.append(np.polyfit(x,y,1)[0])
    return np.mean(np.abs(slopes) < 1e-3)


# ---------- MAIN FEATURE FUNCTION ----------
def extract_features(df, loop_id):

    pv = df["PV"].values
    sp = df["SP"].values
    op = df["OP"].values

    peak_ratio, hf_ratio = PSD_features(pv)

    feats = {
        "loop_id": loop_id,
        "zero_cross": zero_cross(sp,pv),
        "osc_index": oscillation_index(sp,pv),
        "overshoot_ratio": overshoot_ratio(sp,pv),
        "peak_ratio": peak_ratio,
        "hf_ratio": hf_ratio,
        "ac_peak": autocorr_peak(pv),
        "err_op_corr": np.corrcoef(sp - pv, op)[0,1],
        "pv_spikes": sudden_spikes(pv),
        "op_spikes": sudden_spikes(op),
        "pv_std_mean": np.std(pv)/(np.mean(pv)+1e-9),
        "op_std_mean": np.std(op)/(np.mean(op)+1e-9),
        "mean_abs_err": np.mean(np.abs(sp - pv)),
        "flat_fraction": stiction_flatness(op),
        "dead_slope": stiction_dead_slope(sp,pv,op, max(20,len(op)//20)),
        "variance_level": np.var(pv)
    }

    return feats
