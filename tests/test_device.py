from vjepa_ac.device import map_visible


def test_no_cvd_passthrough():
    assert map_visible([0, 2, 3], None) == [0, 2, 3]


def test_cvd_remaps_physical_to_visible():
    assert map_visible([2], "2") == [0]
    assert map_visible([1, 3], "3,1") == [1, 0]


def test_cvd_hides_unlisted_gpus():
    assert map_visible([0, 1, 2], "1") == [0]
    assert map_visible([0], "2,3") == []


def test_unparsable_cvd_yields_empty():
    assert map_visible([0, 1], "GPU-uuid-abc") == []
