from vjepa_ac.variations import MODEL, SMOKE_MODEL, SMOKE_TRAINING, TRAINING


def test_flagship_is_compressed_with_rollout():
    assert MODEL.compressor
    assert TRAINING.rollout_loss
    assert TRAINING.data == "cache"
    assert TRAINING.stride > 1


def test_heads_divide_d_model():
    for cfg in (MODEL, SMOKE_MODEL):
        assert cfg.d_model % cfg.n_heads == 0
        assert (cfg.d_model // cfg.n_heads) % 2 == 0


def test_smoke_is_local_scale():
    assert SMOKE_MODEL.compressor
    assert SMOKE_TRAINING.data == "synthetic"
    assert SMOKE_TRAINING.total_steps <= 100
    assert not SMOKE_TRAINING.amp
    assert SMOKE_TRAINING.val_interval <= SMOKE_TRAINING.total_steps


def test_smoke_exercises_strided_path():
    assert SMOKE_TRAINING.stride > 1
