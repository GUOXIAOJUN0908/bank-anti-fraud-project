"""Training/serving skew test: SQL batch features == real-time Python features."""

import duckdb
import numpy as np

from fraud.features.online import OnlineFeatureStore
from fraud.features.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def test_online_features_match_sql(pipeline):
    con = duckdb.connect(str(pipeline.db), read_only=True)
    raw = con.execute("SELECT * FROM transactions ORDER BY unix_time, trans_num").df()
    sql = con.execute("SELECT * FROM features ORDER BY unix_time, trans_num").df()
    con.close()
    assert (raw.trans_num.values == sql.trans_num.values).all()

    store = OnlineFeatureStore()
    online = [
        store.compute_and_update(
            cc_num=r.cc_num, ts=r.trans_date_trans_time.to_pydatetime(), category=r.category, amt=r.amt,
            lat=r.lat, lon=r.long, merch_lat=r.merch_lat, merch_long=r.merch_long, city_pop=r.city_pop, dob=r.dob,
        )
        for r in raw.itertuples()
    ]
    for feat in NUMERIC_FEATURES:
        got = np.array([row[feat] for row in online], dtype=float)
        np.testing.assert_allclose(got, sql[feat].to_numpy(dtype=float), rtol=1e-9, atol=1e-9, err_msg=feat)
    for feat in CATEGORICAL_FEATURES:
        assert [row[feat] for row in online] == sql[feat].tolist()


def test_no_lookahead_first_transaction_of_card_has_no_history(pipeline):
    con = duckdb.connect(str(pipeline.db), read_only=True)
    first = con.execute(
        "SELECT * FROM features QUALIFY ROW_NUMBER() OVER (PARTITION BY cc_num ORDER BY unix_time) = 1"
    ).df()
    con.close()
    assert (first.txn_count_1h == 1).all()
    assert (first.txn_count_24h == 1).all()
    assert (first.amt_ratio_to_avg_30d == 1.0).all()
    assert (first.secs_since_last_txn == 2_592_000).all()
