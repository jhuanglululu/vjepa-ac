import math

import torch

from vjepa_ac import data
from vjepa_ac.variations import MODELS


def test_window_starts_stride_one():
    episodes = [[0, 5], [5, 7], [7, 10]]
    assert data.window_starts(episodes, T=3, stride=1).tolist() == [0, 1, 2, 7]


def test_window_starts_strided():
    assert data.window_starts([[0, 7]], T=3, stride=2).tolist() == [0, 1, 2]
    assert data.window_starts([[0, 5]], T=3, stride=2).tolist() == [0]
    assert data.window_starts([[0, 4]], T=3, stride=2).tolist() == []


def test_split_episodes_disjoint_and_deterministic():
    episodes = [[i * 10, (i + 1) * 10] for i in range(10)]
    tr1, va1 = data.split_episodes(episodes, val_frac=0.2)
    tr2, va2 = data.split_episodes(episodes, val_frac=0.2)
    assert tr1 == tr2 and va1 == va2
    assert len(va1) == 2 and len(tr1) == 8
    assert sorted(tr1 + va1) == episodes


def test_split_episodes_holds_out_at_least_one():
    tr, va = data.split_episodes([[0, 5], [5, 9]], val_frac=0.01)
    assert len(va) == 1 and len(tr) == 1


def test_split_episodes_smaller_frac_is_subset():
    episodes = [[i, i + 1] for i in range(20)]
    _, va_small = data.split_episodes(episodes, val_frac=0.1)
    _, va_big = data.split_episodes(episodes, val_frac=0.2)
    assert all(e in va_big for e in va_small)


def test_synthetic_cache_shapes_and_dynamics():
    cfg = MODELS["tiny"]
    cache = data.synthetic_cache(cfg, seed=0)
    n = sum(b - a for a, b in cache.episodes)
    assert tuple(cache.latents.shape) == (n, cfg.n_patches, cfg.d_state)
    assert tuple(cache.actions.shape) == (n, cfg.d_action)
    assert tuple(cache.states.shape) == (n, cfg.d_action)
    assert cache.state_dim == cfg.d_action
    assert torch.isfinite(cache.latents).all()
    a, b = cache.episodes[0]
    deltas = (cache.latents[a + 1 : b] - cache.latents[a : b - 1]).flatten(1).norm(dim=1)
    assert (deltas > 0.01).all()


def test_synthetic_cache_states_track_actions():
    cache = data.synthetic_cache(MODELS["tiny"], seed=0)
    for a, b in cache.episodes:
        assert torch.equal(cache.states[a], torch.zeros(cache.state_dim))
        assert torch.allclose(
            cache.states[a + 1 : b] - cache.states[a : b - 1], cache.actions[a : b - 1]
        )


def test_conditioner_wrap_correction():
    states = torch.tensor([[3.0, 0.0], [-3.0, 1.0], [-2.5, 2.0]])
    cond = data.Conditioner(states, torch.zeros(2), torch.ones(2))
    f1 = cond.features(torch.tensor([0]), stride=1)
    assert abs(f1[0, 0].item() - (2 * math.pi - 6.0)) < 1e-6
    assert f1[0, 1].item() == 1.0
    f2 = cond.features(torch.tensor([0]), stride=2)
    assert abs(f2[0, 0].item() - (2 * math.pi - 6.0 + 0.5)) < 1e-6
    assert f2[0, 1].item() == 2.0


def test_conditioner_windows_alignment_and_padding():
    states = torch.tensor([[0.0, 0.0], [0.1, 1.0], [0.3, 2.0], [0.6, 3.0], [1.0, 4.0], [1.5, 5.0]])
    cond = data.Conditioner(states, torch.zeros(2), torch.ones(2))
    w = cond.windows(torch.tensor([0]), T=3, stride=2)
    assert w.shape == (1, 3, 2)
    assert torch.allclose(w[0, 0], torch.tensor([0.3, 2.0]))
    assert torch.allclose(w[0, 1], torch.tensor([0.7, 4.0]))
    assert torch.equal(w[0, 2], torch.zeros(2))


def test_conditioner_normalizes_with_stats():
    states = torch.tensor([[0.0, 0.0], [0.4, 1.0], [0.8, 2.0]])
    cond = data.Conditioner(states, torch.tensor([0.2, 0.5]), torch.tensor([0.1, 2.0]))
    w = cond.windows(torch.tensor([0]), T=2, stride=1)
    assert torch.allclose(w[0, 0], torch.tensor([2.0, 0.25]))


def test_fit_conditioner_uses_train_episodes_only():
    states = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [1.0, 9.0], [1.4, 9.0], [1.8, 9.0]])
    cond = data.fit_conditioner(states, [[3, 6]], stride=1)
    assert torch.allclose(cond.mean, torch.tensor([0.4, 9.0]), atol=1e-6)


def test_fit_conditioner_skips_episode_boundaries():
    states = torch.tensor([[0.0, 0.0], [0.5, 0.0], [50.0, 0.0], [50.5, 0.0]])
    cond = data.fit_conditioner(states, [[0, 2], [2, 4]], stride=1)
    assert abs(cond.mean[0].item() - 0.5) < 1e-6


def test_conditioner_stats_roundtrip():
    cache = data.synthetic_cache(MODELS["tiny"], seed=0)
    cond = data.fit_conditioner(cache.states, cache.episodes[:4], stride=2)
    cond2 = data.load_conditioner(cache.states, cond.stats())
    starts = torch.tensor([0, 5])
    assert torch.allclose(cond.windows(starts, T=3, stride=2), cond2.windows(starts, T=3, stride=2))


def test_gather_strided_frames_and_zero_last_action():
    cfg = MODELS["tiny"]
    cache = data.synthetic_cache(cfg, seed=0)
    cond = data.fit_conditioner(cache.states, cache.episodes, stride=2)
    starts = torch.tensor([0, 3])
    z, a = data.gather(cache, cond, starts, T=4, stride=2, device="cpu")
    assert z.shape == (2, 4, cfg.n_patches, cfg.d_state)
    assert a.shape == (2, 4, cfg.d_action)
    assert torch.equal(z[1, 0], cache.latents[3].float())
    assert torch.equal(z[0, 2], cache.latents[4].float())
    assert torch.equal(a[:, -1], torch.zeros(2, cfg.d_action))
