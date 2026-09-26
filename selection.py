"""Select one common hyperparameter configuration from validation scores only."""
import math

SHARED_FIELDS = ('period', 'cycles', 'regularization', 'revin', 'tau_cycles')
INTEGER_FIELDS = ('horizon', 'period', 'cycles', 'revin', 'parameters', 'lookback')
RULE = 'equal_mean_of_per_horizon_normalized_validation_mse'


def select_shared(candidate_rows, horizons, expected_groups=None):
    """Average MSE(h,config)/min_config MSE(h,config), equally across horizons.

    Every legal configuration must have a row for every requested horizon.
    A configuration with a nonfinite score at any horizon is ineligible.
    Boundary gains are train-fitted coefficients and remain horizon-specific.
    """
    horizons = tuple(horizons)
    if not horizons or len(set(horizons)) != len(horizons):
        raise ValueError('horizons must be nonempty and unique')
    fields = {'horizon', *SHARED_FIELDS, 'boundary_gain', 'parameters', 'lookback', 'validation_mse'}
    groups = {}
    minima = {h: float('inf') for h in horizons}
    count = 0
    for original in candidate_rows:
        if not fields.issubset(original) or set(original) - fields - {'status'}:
            raise ValueError('Selection accepts only validation candidate fields, never test metrics')
        row = {k: int(original[k]) if k in INTEGER_FIELDS else float(original[k]) for k in fields}
        h, p, n = row['horizon'], row['period'], row['cycles']
        if h not in horizons or not 1 <= n <= 29 or p < 1:
            raise ValueError('Unexpected horizon or invalid period/cycle count')
        if row['parameters'] != n + 1 or row['lookback'] != p * n or p * n > 720:
            raise ValueError('Candidate violates model constraints')
        if (not math.isfinite(row['regularization']) or row['regularization'] < 0
                or not math.isfinite(row['tau_cycles']) or row['tau_cycles'] <= 0
                or row['revin'] not in (0, 1) or not 0 <= row['boundary_gain'] <= 1
                or math.isnan(row['validation_mse']) or row['validation_mse'] < 0):
            raise ValueError('Invalid candidate values')
        key = tuple(row[k] for k in SHARED_FIELDS)
        group = groups.setdefault(key, {})
        if h in group:
            raise ValueError('Duplicate configuration/horizon row')
        group[h] = row
        minima[h] = min(minima[h], row['validation_mse'])
        count += 1
    if not groups or any(set(rows) != set(horizons) for rows in groups.values()):
        raise ValueError('Every shared candidate must cover every requested horizon')
    if expected_groups is not None and len(groups) != expected_groups:
        raise ValueError('Shared grid does not match the planned candidate count')
    if any(not math.isfinite(v) for v in minima.values()):
        raise ValueError('A horizon has no finite validation candidate')
    # Zero-error synthetic data needs a well-defined denominator. The benchmark
    # minima are strictly positive, so this fallback does not affect replication.
    normalizers = {h: v if v > 0 else 1e-12 for h, v in minima.items()}
    eligible = [(key, rows) for key, rows in groups.items()
                if all(math.isfinite(rows[h]['validation_mse']) for h in horizons)]
    if not eligible:
        raise ValueError('No configuration has finite validation MSE at every horizon')

    def score(rows):
        return sum(rows[h]['validation_mse'] / normalizers[h] for h in horizons) / len(horizons)

    def rank(item):
        key, rows = item
        p, n, strength, centered, tau = key
        return score(rows), n + 1, p * n, strength, centered, tau, p

    key, winners = min(eligible, key=rank)
    return {'selection_split': 'validation', 'selection_rule': RULE,
            'shared_fields': list(SHARED_FIELDS), 'shared_configuration': dict(zip(SHARED_FIELDS, key)),
            'shared_selection_score': score(winners),
            'validation_minima': {str(h): minima[h] for h in horizons},
            'validation_normalizers': {str(h): normalizers[h] for h in horizons},
            'candidate_rows': count, 'shared_candidates': len(groups),
            'eligible_shared_candidates': len(eligible),
            'winners': {str(h): winners[h] for h in horizons}}
