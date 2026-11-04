from src.agent.schemas import ExtractedData

def test_schema_accepts_valid_employee():
    payload = {
        "employees": [
            {
                "user_id": 101,
                "name": "Michael Chen",
                "age": 29,
                "email": None,
                "salary": 120000,
                "join_date": "2024-01-01",
                "department": "Artificial Intelligence",
                "performance_score": 9.5,
                "location": "Chicago",
                "job_title": "Engineer",
            }
        ]
    }

    data = ExtractedData.model_validate(payload)
    assert data.employees[0].user_id == 101
