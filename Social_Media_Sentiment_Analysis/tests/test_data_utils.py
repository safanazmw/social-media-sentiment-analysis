from Analysis.data_utils import load_dataset, EXPECTED_COLUMNS

def test_headerless_dataset_has_explicit_columns():
    df = load_dataset()
    assert list(df.columns) == EXPECTED_COLUMNS
    assert {"Positive", "Negative", "Neutral", "Irrelevant"}.issubset(set(df["sentiment"].unique()))
    assert len(df) > 1000
    assert df["text"].astype(str).str.len().gt(0).all()
