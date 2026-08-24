from types import SimpleNamespace

from omegaconf import OmegaConf

from mldft.ofdft.denop import prepare_data as prepare_data_module


def test_prepare_transforms_updates_transform_pipeline(monkeypatch):
    """Assert denop preprocessing is prepended and transform flags are updated."""
    transforms = SimpleNamespace(
        pre_transforms=["existing"],
        add_transformation_matrix=False,
        use_cached_data=True,
    )
    basis_info = object()

    # TODO: I don´t understand enough of the meaning of all the transforms to
    # and the physics to fully review this test

    # mock grid requirement and prep transform for test purposes
    monkeypatch.setattr(prepare_data_module, "requires_grid", lambda target_key, penalty: True)
    monkeypatch.setattr(
        prepare_data_module,
        "PrepareForDensityOptimization",
        lambda *args: ("prepare", args),
    )

    updated = prepare_data_module.prepare_transforms(
        model_config=OmegaConf.create({"data": {"target_key": "density"}}),
        transforms=transforms,
        basis_info=basis_info,
        negative_integrated_density_penalty_weight=1.5,
        grid_level=2,
        grid_prune="nwchem_prune",
    )

    assert updated is transforms
    assert transforms.pre_transforms[0] == (
        "prepare",
        (
            basis_info,  # basis_info
            True,  # add_grid
            2,  # grid_level
            "nwchem_prune",  # grid_prune
        ),
    )
    assert transforms.pre_transforms[1] == "existing"
    assert transforms.add_transformation_matrix is True
    assert transforms.use_cached_data is False


def test_prepare_datasetkwargs_uses_sad_statistics_path(monkeypatch):
    """Check SAD initialization rewrites stats path and augments dataset kwargs."""
    monkeypatch.setattr(
        prepare_data_module,
        "to_no_basis_transforms_dataset_statistics",
        lambda path, transform_name: f"{path}:{transform_name}:sad",
    )

    dataset_statistics_path, dataset_kwargs = prepare_data_module.prepare_datasetkwargs(
        dataset_kwargs={"existing": True},
        initialization="sad",
        model_config=OmegaConf.create(
            {
                "data": {
                    "dataset_statistics": {"path": "/tmp/stats.pt"},
                    "transforms": {"name": "basis_transforms"},
                }
            }
        ),
    )

    assert dataset_statistics_path == "/tmp/stats.pt:basis_transforms:sad"
    assert dataset_kwargs["existing"] is True
    assert dataset_kwargs["limit_scf_iterations"] == -1
    assert "of_labels/energies/e_tot" in dataset_kwargs["additional_keys_at_ground_state"]
    for e in [
        "electron",
        "ext",
        "hartree",
        "kin",
        "kin_plus_xc",
        "kin_minus_apbe",
        "kinapbe",
        "xc",
    ]:
        assert (
            dataset_kwargs["additional_keys_at_ground_state"][f"of_labels/energies/e_{e}"]
            == prepare_data_module.Representation.SCALAR
        )


def test_prepare_datasetkwargs_non_sad_statistics_path(monkeypatch):
    """Check non-SAD initialization keeps original stats path and augments dataset kwargs."""
    monkeypatch.setattr(
        prepare_data_module,
        "to_no_basis_transforms_dataset_statistics",
        lambda path, transform_name: f"{path}:{transform_name}:sad",
    )
    dataset_statistics_path, dataset_kwargs = prepare_data_module.prepare_datasetkwargs(
        dataset_kwargs={"existing": True},
        initialization="minao",
        model_config=OmegaConf.create(
            {
                "data": {
                    "dataset_statistics": {"path": "/tmp/stats.pt"},
                    "transforms": {"name": "basis_transforms"},
                }
            }
        ),
    )

    assert dataset_statistics_path == "/tmp/stats.pt"
    assert dataset_kwargs["existing"] is True
    assert dataset_kwargs["limit_scf_iterations"] == -1
    assert "of_labels/energies/e_tot" in dataset_kwargs["additional_keys_at_ground_state"]
    for e in [
        "electron",
        "ext",
        "hartree",
        "kin",
        "kin_plus_xc",
        "kin_minus_apbe",
        "kinapbe",
        "xc",
    ]:
        assert (
            dataset_kwargs["additional_keys_at_ground_state"][f"of_labels/energies/e_{e}"]
            == prepare_data_module.Representation.SCALAR
        )


def test_prepare_dataset_and_loader_builds_dataset_and_loader(monkeypatch):
    """Validate dataset/loader builders receive expected arguments from config wiring."""
    created = {}

    def fake_prepare_datasetkwargs(dataset_kwargs, initialization, model_config):
        created["dataset_kwargs_input"] = dict(dataset_kwargs)
        created["initialization"] = initialization
        return "/tmp/stats.pt", {"foo": "bar", "limit_scf_iterations": -1}

    class FakeDataset:
        def __init__(self, **kwargs):
            created["dataset_kwargs"] = kwargs
            self.kwargs = kwargs

    class FakeLoader:
        def __init__(self, dataset, **kwargs):
            created["loader_dataset"] = dataset
            created["loader_kwargs"] = kwargs

    monkeypatch.setattr(prepare_data_module, "prepare_datasetkwargs", fake_prepare_datasetkwargs)
    monkeypatch.setattr(prepare_data_module, "instantiate", lambda cfg: {"instantiated": True})
    monkeypatch.setattr(prepare_data_module, "OFDataset", FakeDataset)
    monkeypatch.setattr(prepare_data_module, "OFLoader", FakeLoader)

    config = OmegaConf.create(
        {
            "data": {
                "datamodule": {
                    "dataloader_kwargs": {
                        "batch_size": 4,
                        "num_workers": 2,
                        "pin_memory": False,
                        "list_keys": ["a", "b"],
                    }
                }
            },
            "batch_size": 4,
        }
    )
    model_config = OmegaConf.create(
        {"data": {"datamodule": {"dataset_kwargs": {"_target_": "x"}}}}
    )
    denop_cfg = OmegaConf.create({"initialization": "minao"})

    dataset_statistics_path, dataset, dataloader = prepare_data_module.prepare_dataset_and_loader(
        config=config,
        model_config=model_config,
        denop_cfg=denop_cfg,
        paths=["path_a"],
        num_scf_iterations_per_path=[3],
        basis_info="basis-info",
        transforms="transforms",
    )

    assert dataset_statistics_path == "/tmp/stats.pt"
    assert isinstance(dataset, FakeDataset)
    assert isinstance(dataloader, FakeLoader)
    assert created["dataset_kwargs_input"] == {"instantiated": True}
    assert created["dataset_kwargs"]["paths"] == ["path_a"]
    assert created["dataset_kwargs"]["num_scf_iterations_per_path"] == [3]
    assert created["dataset_kwargs"]["basis_info"] == "basis-info"
    assert created["dataset_kwargs"]["transforms"] == "transforms"
    assert created["dataset_kwargs"]["foo"] == "bar"
    assert created["loader_dataset"] is dataset
    assert created["loader_kwargs"] == {
        "batch_size": 4,
        "shuffle": False,
        "num_workers": 2,
        "pin_memory": False,
        "list_keys": ["a", "b"],
    }
