import shutil

import numpy as np
import pytest
import torch
from hydra import compose, initialize
from pyscf import gto
from pyscf.lib import chkfile

from mldft.ofdft import run_geometry_optimization

TEST_XYZ = """4
XYZ from PySCF
He          0.08651       -0.00001        0.04971
He          0.03653        0.00098        0.75374
He          0.87797        0.00003        0.68655
He          0.87797        0.90003        0.68655
"""


@pytest.fixture(scope="session")
def generate_geomopt_test_data(tmp_path_factory):
    """Fixture to generate a test data directory for geometry optimization."""
    data_path = tmp_path_factory.mktemp("GeoOpt_test_data")
    xyz_path = data_path / "xyz"
    # if xyz_path.exists():
    #     shutil.rmtree(xyz_path)
    ks_path = data_path / "ks-opt"
    # if ks_path.exists():
    #     shutil.rmtree(ks_path)

    xyz_path.mkdir(parents=True, exist_ok=True)
    xyz_file = xyz_path / "test_01.xyz"
    ks_path.mkdir(parents=True, exist_ok=True)

    with open(xyz_file, "w") as f:
        f.write(TEST_XYZ)

    mol = gto.M(atom=str(xyz_file), unit="Angstrom", verbose=0)

    outfile = ks_path / "test_01.chk"
    chkfile.dump_chkfile_key(outfile, "scf/e_tot", np.random.rand())
    chkfile.dump_mol(mol, outfile)

    yield data_path

    shutil.rmtree(data_path)


def test_main(tmp_path_factory, generate_geomopt_test_data, generate_run_path):
    """Test the very basiscs of the main function of the geometry optimizaiton script.

    Does not test the actual optimization or even the plotting of the results. Just that it manages
    to setup and make a few steps without crashing.
    """

    geoopt_data = generate_geomopt_test_data
    run_path = generate_run_path
    geomopt_run_dir = tmp_path_factory.mktemp("geomopt_test_run_dir")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with initialize(version_base="1.3", config_path="../../configs/ofdft/"):
        cfg = compose(
            config_name="geometry_optimization",
            overrides=[
                "data_path=" + str(geoopt_data),
                "run_path=" + str(run_path),
                "device=" + device,
                "base_dir=" + str(geomopt_run_dir),
                "log_file" + "=" + str(geomopt_run_dir / "test_run.log"),
                "config_file" + "=" + str(geomopt_run_dir / "config.yaml"),
                "density_optimizer.max_cycle=2",
                "maxsteps=2",
                "denop_settings.initialization=dumb",
                "main_log_level=5",
                "molecule_log_level=5",
            ],
        )

    run_geometry_optimization.main(cfg)

    # Check if the output files are created
    assert (geomopt_run_dir / "geometry_opt_log/test_01.log").exists()
    assert (geomopt_run_dir / "geometry_opt_sample/test_01.pt").exists()

    sample = torch.load(
        geomopt_run_dir / "geometry_opt_sample/test_01.pt",
        weights_only=False,
        map_location=torch.device("cpu"),
    )
    assert hasattr(sample, "geometry_energy_offset_available")
    assert bool(sample.geometry_energy_offset_available) is False
    assert hasattr(sample, "ml_energy_at_ks_geometry_converged")
    assert bool(sample.ml_energy_at_ks_geometry_converged) is False
    assert hasattr(sample, "ml_energy_at_ks_geometry")
    assert np.isnan(float(sample.ml_energy_at_ks_geometry))
    assert hasattr(sample, "geometry_energy_offset")
    assert np.isnan(float(sample.geometry_energy_offset))
    assert hasattr(sample, "geometry_energy_offset_denop_time")
    assert np.isnan(float(sample.geometry_energy_offset_denop_time))

    # Finding a test case which works for both the model and rdkits bond determination
    # did not work, thus the plotting can not be tested (it probably will not work for
    # non nice structures with rdkit anyway)


def test_select_geometry_molecule_files_resolves_non_contiguous_file_stem_ids(tmp_path):
    """Explicit molecule IDs refer to geometry file stems, not sorted-list positions."""
    xyz_path = tmp_path / "xyz"
    xyz_path.mkdir()
    stems = ["0003452", "0047736", "omol-geom_03106275"]
    for stem in stems:
        (xyz_path / f"{stem}.xyz").write_text(TEST_XYZ)

    files = sorted(xyz_path.glob("*.xyz"))

    selected = run_geometry_optimization._select_geometry_molecule_files(
        files,
        molecule_ids=[3106275, 47736],
        start_idx=0,
        n_molecules=None,
    )

    assert selected == [
        (3106275, xyz_path / "omol-geom_03106275.xyz"),
        (47736, xyz_path / "0047736.xyz"),
    ]


def test_select_geometry_molecule_files_reports_missing_ids(tmp_path):
    """Missing explicit molecule IDs fail with the requested ID in the error."""
    xyz_path = tmp_path / "xyz"
    xyz_path.mkdir()
    (xyz_path / "0003452.xyz").write_text(TEST_XYZ)

    with pytest.raises(ValueError, match="999999"):
        run_geometry_optimization._select_geometry_molecule_files(
            sorted(xyz_path.glob("*.xyz")),
            molecule_ids=[999999],
            start_idx=0,
            n_molecules=None,
        )
