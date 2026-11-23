import numpy as np
from sklearn.metrics import average_precision_score

# Bootstrap: we draw the test PATIENTS again with replacement (some twice, some not)
# and compute the AUPRC on that new test set. Repeating this 1000x shows how much the AUPRC
# would vary with a different test set -> 95% confidence interval.
# We draw patients (not nodules) because nodules of the same patient are not independent. So we avoid any data leakage.
# made with help of ClaudeAI Opus 5.5

def draw_patients(rng, patient_ids):
    patients = np.unique(patient_ids)
    drawn = rng.choice(patients, size=len(patients), replace=True) # This selects the same size of patients in the test but with replacement
    return np.concatenate([np.where(patient_ids == p)[0] for p in drawn]) # all nodules of the drawn patients


def bootstrap_auprc(y_true, y_prob, patient_ids, n_boot=1000, seed=12):
    rng = np.random.default_rng(seed) # Numpy random number generator with seed
    scores = []
    for _ in range(n_boot): # bootstrapping iterations (n_boot amount)
        idx = draw_patients(rng, patient_ids)
        scores.append(average_precision_score(y_true[idx], y_prob[idx]))
    lo, hi = np.percentile(scores, [2.5, 97.5]) # 95% CI
    return lo, hi, np.std(scores) # std = spread of the AUPRC over all bootstrap rounds


def compare_auprc(y_true, y_prob_a, y_prob_b, patient_ids, n_boot=1000, seed=12):
    # paired: both models get the SAME drawn patients each round, we keep the difference B - A
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = draw_patients(rng, patient_ids)
        diffs.append(average_precision_score(y_true[idx], y_prob_b[idx])
                     - average_precision_score(y_true[idx], y_prob_a[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5]) # if 0 is not in this interval -> significant difference
    return lo, hi, np.std(diffs)
