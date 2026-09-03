import pytest
import torch

from imagen_pytorch.elucidated_imagen import ElucidatedImagen


class _ScheduleOwner:
    device = torch.device("cpu")


def make_schedule(num_sample_steps):
    return ElucidatedImagen.sample_schedule(
        _ScheduleOwner(), num_sample_steps, rho=7, sigma_min=0.002, sigma_max=80.0
    )


def test_single_step_schedule_is_finite_and_reaches_zero():
    schedule = make_schedule(1)

    assert schedule.shape == (2,)
    assert torch.isfinite(schedule).all()
    assert schedule[0].item() == pytest.approx(80.0)
    assert schedule[1].item() == pytest.approx(0.0)


def test_multi_step_schedule_keeps_karras_endpoints():
    schedule = make_schedule(5)

    assert torch.isfinite(schedule).all()
    assert schedule[0].item() == pytest.approx(80.0)
    assert schedule[-2].item() == pytest.approx(0.002)
    assert schedule[-1].item() == pytest.approx(0.0)
    assert torch.all(schedule[:-1].diff() < 0)


def test_schedule_rejects_empty_denoising_plan():
    with pytest.raises(ValueError, match="at least 1"):
        make_schedule(0)
