import pandas as pd
from src.core.query import QuerySpec, FilterSpec, execute_query

def test_execute_query_eq():
    df = pd.DataFrame([
        {"Name":"A", "Department":"Artificial Intelligence"},
        {"Name":"B", "Department":"Data Science"},
    ])
    spec = QuerySpec(select=["Name"], filters=[FilterSpec(column="Department", op="eq", value="Artificial Intelligence")])
    out = execute_query(spec, df)
    assert out["Name"].tolist() == ["A"]
