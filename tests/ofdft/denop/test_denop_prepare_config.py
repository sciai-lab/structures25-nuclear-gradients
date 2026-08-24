from pathlib import Path

from omegaconf import OmegaConf

from mldft.ofdft.denop import prepare_config as prepare_config_module


def test_prepare_chkpt_path_handles_optional_guess():
    """Check that primary and optional guess checkpoint paths are both resolved."""
    cfg = OmegaConf.create(
        {
            "ckpt_choice": "best",
            "guess_path": "/tmp/guess-run",
            "guess_ckpt_choice": "last",
        }
    )
    original = prepare_config_module.run_to_checkpoint_path
    prepare_config_module.run_to_checkpoint_path = (
        lambda base_path, choice: Path(base_path) / f"{choice}.ckpt"
    )
    try:
        ckpt_path, guess_ckpt_path = prepare_config_module.prepare_chkpt_path(
            Path("/tmp/run"), cfg
        )
    finally:
        prepare_config_module.run_to_checkpoint_path = original

    assert ckpt_path == Path("/tmp/run") / "best.ckpt"
    assert guess_ckpt_path == Path("/tmp/guess-run") / "last.ckpt"


def test_get_split_filepath_prefers_cfg_value():
    """Ensure an explicit runtime split path overrides model-config defaults."""
    cfg = OmegaConf.create({"split_file_path": "/tmp/manual.pkl"})
    model_config = OmegaConf.create(
        {"data": {"datamodule": {"split_file": {"output_file": "/tmp/model.pkl"}}}}
    )
    model_config_resolved = OmegaConf.create(
        {"data": {"datamodule": {"split_file": {"output_file": "/tmp/resolved.pkl"}}}}
    )

    split_file_path = prepare_config_module.get_split_filepath(
        cfg, model_config=model_config, model_config_resolved=model_config_resolved
    )

    assert split_file_path == "/tmp/manual.pkl"


def test_get_split_filepath_falls_back_to_resolved_then_model():
    """Confirm fallback order: resolved split file first, then raw model split field."""
    cfg = OmegaConf.create({})
    resolved_config = OmegaConf.create(
        {"data": {"datamodule": {"split_file": {"output_file": "/tmp/resolved.pkl"}}}}
    )
    raw_config = OmegaConf.create({"data": {"datamodule": {"split_file": "/tmp/raw.pkl"}}})

    resolved_path = prepare_config_module.get_split_filepath(
        cfg, model_config=resolved_config, model_config_resolved=resolved_config
    )
    raw_path = prepare_config_module.get_split_filepath(
        cfg, model_config=raw_config, model_config_resolved=OmegaConf.create(raw_config)
    )

    assert resolved_path == "/tmp/resolved.pkl"
    assert raw_path == "/tmp/raw.pkl"


def test_prepare_config_resolves_run_paths_and_merges_continue_config(monkeypatch, tmp_path: Path):
    """Validate config preparation resolves paths and applies continue-denop merge hook."""
    run_path = tmp_path / "run"
    run_path.mkdir()
    OmegaConf.save(
        OmegaConf.create({"data": {"datamodule": {"split_file": "/tmp/raw-split.pkl"}}}),
        run_path / "hparams.yaml",
        resolve=False,
    )
    OmegaConf.save(
        OmegaConf.create(
            {"data": {"datamodule": {"split_file": {"output_file": "/tmp/resolved-split.pkl"}}}}
        ),
        run_path / "hparams_resolved.yaml",
        resolve=False,
    )

    cfg = OmegaConf.create(
        {
            "run_path": str(run_path),
            "ckpt_choice": "last",
            "guess_path": None,
            "continue_denop": True,
        }
    )

    merged_cfg = OmegaConf.create({"merged": True})

    monkeypatch.setattr(prepare_config_module, "parse_run_path", lambda path: Path(path))
    monkeypatch.setattr(
        prepare_config_module,
        "run_to_checkpoint_path",
        lambda base_path, choice: Path(base_path) / f"{choice}.ckpt",
    )
    monkeypatch.setattr(
        prepare_config_module, "merge_denop_configs", lambda incoming_cfg: merged_cfg
    )

    resolved_cfg, model_config, ckpt_path, guess_ckpt_path = prepare_config_module.prepare_config(
        cfg
    )

    assert resolved_cfg is merged_cfg
    assert model_config.data.datamodule.split_file == "/tmp/raw-split.pkl"
    assert ckpt_path == run_path / "last.ckpt"
    assert guess_ckpt_path is None
    assert cfg.run_path == str(run_path)
    assert cfg.split_file_path == "/tmp/raw-split.pkl"
    assert cfg.ckpt_path == str(run_path / "last.ckpt")
    assert cfg.guess_ckpt_path == "None"
