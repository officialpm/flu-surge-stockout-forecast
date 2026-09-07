from src.data_pipeline.region_mapping import get_hhs_region


def test_known_states_map_to_correct_hhs_region():
    assert get_hhs_region("CA") == 9
    assert get_hhs_region("TX") == 6
    assert get_hhs_region("WI") == 5
    assert get_hhs_region("NY") == 2


def test_unknown_state_raises_value_error():
    import pytest

    with pytest.raises(ValueError, match="Unknown state code"):
        get_hhs_region("ZZ")
