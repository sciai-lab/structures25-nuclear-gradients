from omegaconf import OmegaConf

from mldft.ofdft.denop import denoprunner as run_module


def test_main_calls_print_extras_and_runner(monkeypatch):
    """Verify the Hydra entrypoint prints the config, applies extras, and runs denop."""
    calls = []
    cfg = OmegaConf.create({"x": 1})

    monkeypatch.setattr(
        run_module,
        "print_config_tree",
        lambda incoming_cfg: calls.append(("print", incoming_cfg)),
    )
    monkeypatch.setattr(
        run_module, "extras", lambda incoming_cfg: calls.append(("extras", incoming_cfg))
    )
    monkeypatch.setattr(
        run_module,
        "run_denop_single_device",
        lambda incoming_cfg: calls.append(("runner", incoming_cfg)),
    )

    run_module.main.__wrapped__(cfg)

    assert calls == [("print", cfg), ("extras", cfg), ("runner", cfg)]
