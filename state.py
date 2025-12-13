def initial_state():
    return {
        "age": None,
        "height": None,
        "weight": None,
        "has_diabetes": None,
        "pregnant": None,
        "medical_conditions": None
    }


def update_state(state: dict, field: str, value):
    if field in state:
        state[field] = value
