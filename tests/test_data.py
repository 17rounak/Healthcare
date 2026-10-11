"""Data and partitioning tests."""
import numpy as np
import pytest

from src import config, data_loader, partitioning, preprocessing


@pytest.fixture(scope="module")
def data():
    df = data_loader.load_framingham()
    return df


def test_dataset_shape(data):
    assert len(data) == 4240
    assert len(data.columns) == 16


def test_dataset_features(data):
    assert list(data.columns) == config.FEATURE_COLUMNS + [config.TARGET_COLUMN]


def test_dataset_target_counts(data):
    assert int(data[config.TARGET_COLUMN].sum()) == 644


def test_no_id_column(data):
    assert not any(c.lower() in {"id", "customerid", "row", "index"}
                   for c in data.columns)


def test_missing_values_present(data):
    assert data.isna().sum().sum() > 0


def test_train_test_split_stratified():
    df = data_loader.load_framingham()
    tr, te = data_loader.make_train_test_split(df, 0)
    assert len(tr) == 3392
    assert len(te) == 848
    # stratification
    p_tr = tr[config.TARGET_COLUMN].mean()
    p_te = te[config.TARGET_COLUMN].mean()
    assert abs(p_tr - p_te) < 0.02


def test_split_reproducible():
    df = data_loader.load_framingham()
    a_tr, a_te = data_loader.make_train_test_split(df, 3)
    b_tr, b_te = data_loader.make_train_test_split(df, 3)
    assert a_te.equals(b_te)


def test_partition_no_overlap_full_coverage():
    df = data_loader.load_framingham()
    tr, te, med = None, None, None
    # Use experiments.get_seed_data to get the same training split
    from src import experiments as E
    tr, te, med = E.get_seed_data(0, df)
    for regime in config.REGIMES:
        silos = partitioning.make_partition(tr, regime, 0, med)
        all_idx = np.concatenate(silos)
        assert len(np.unique(all_idx)) == len(tr), f"{regime} overlap/coverage"


def test_iid_sizes_balanced():
    df = data_loader.load_framingham()
    from src import experiments as E
    tr, te, med = E.get_seed_data(0, df)
    silos = partitioning.make_partition(tr, "iid", 0, med)
    sizes = sorted(len(s) for s in silos)
    assert sizes == [678, 678, 678, 679, 679]


def test_covariate_age_ordering():
    df = data_loader.load_framingham()
    from src import experiments as E
    tr, te, med = E.get_seed_data(0, df)
    silos = partitioning.make_partition(tr, "covariate_skew", 0, med)
    means = [tr.iloc[idx]["age"].mean() for idx in silos]
    assert means == sorted(means)


def test_label_skew_reproducible():
    df = data_loader.load_framingham()
    from src import experiments as E
    tr, te, med = E.get_seed_data(0, df)
    a = partitioning.make_partition(tr, "label_skew", 2, med)
    # reset cache and recompute
    from src.partitioning import _PARTITION_CACHE
    _PARTITION_CACHE.pop(("label_skew", 2))
    b = partitioning.make_partition(tr, "label_skew", 2, med)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_sufficient_statistics_aggregation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    half = X[:50], X[50:]
    stats = [preprocessing.compute_local_statistics(h) for h in half]
    agg = preprocessing.aggregate_statistics(stats)
    gs = preprocessing.derive_global_standardization(agg)
    whole = preprocessing.derive_global_standardization(
        preprocessing.aggregate_statistics(
            [preprocessing.compute_local_statistics(X)]))
    assert np.allclose(gs["mean"], whole["mean"])
    assert np.allclose(gs["std"], whole["std"])