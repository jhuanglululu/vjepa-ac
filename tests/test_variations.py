from vjepa_ac.variations import MODELS, TRAININGS


def test_required_variations_exist():
    assert "tiny" in MODELS and "base" in MODELS
    assert "smoke" in TRAININGS and "full" in TRAININGS


def test_heads_divide_d_model():
    for name, cfg in MODELS.items():
        assert cfg.d_model % cfg.n_heads == 0, name
        assert (cfg.d_model // cfg.n_heads) % 2 == 0, name


def test_smoke_is_local_scale():
    smoke = TRAININGS["smoke"]
    assert smoke.data == "synthetic"
    assert smoke.total_steps <= 100
    assert not smoke.amp
    assert smoke.val_interval <= smoke.total_steps


def test_smoke_exercises_strided_path():
    assert TRAININGS["smoke"].stride > 1
