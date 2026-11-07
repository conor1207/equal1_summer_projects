"""Combined FEM + atomistic QTCAD calculation for Si/SiGe valley splitting.

Workflow
--------
1. Solve the device-scale Poisson equation.
2. Tune QP0 until the FEM ground-state energy is E_TARGET_EV.
3. Keep the converged FEM potential in the Device object ``d``.
4. Construct rough Si/SiGe interfaces around the 200--208 nm Si well.
5. Relax the atomic structure with the Keating solver.
6. Interpolate the FEM potential directly onto the atomistic QD.
7. Solve the atomistic tight-binding problem and calculate E1 - E0.

No HDF5 potential file, XYZ atomic file, or Si/Ge band-structure plots are
created.
"""

import copy
import pathlib
import csv
import traceback

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton

from qtcad.device import Device, SubDevice
from qtcad.device import constants as ct
from qtcad.device import io
from qtcad.device import materials as mt
from qtcad.device.mesh3d import Mesh, SubMesh
from qtcad.device.poisson_linear import Solver as PoissonSolver
from qtcad.device.schrodinger import Solver as FEMSchrodingerSolver

from qtcad.atoms import Atoms, SubAtoms
from qtcad.atoms.keating import Solver as KeatingSolver
from qtcad.atoms.keating import SolverParams as KeatingSolverParams
from qtcad.atoms.analysis import save_vtu
from qtcad.atoms.rough_surface import RoughSurface
from qtcad.atoms.schrodinger import Solver as TBSchrodingerSolver
from qtcad.atoms.schrodinger import SolverParams as TBSchrodingerSolverParams


# =============================================================================
# User-adjustable parameters
# =============================================================================

SCALING = 1e-9
TEMPERATURE_K = 0.5

# Fixed gate voltages
V_SC0 = 0.0
V_SC1 = 0.0
V_QT0 = 0.8
V_QT1 = 0.8
GATE_WORK_FUNCTION_EV = 4.46

# Tune QP0 until the lowest FEM eigenenergy reaches this value.
E_TARGET_EV = -0.05
QP0_INITIAL_GUESS_V = 0.0

# FEM mesh coordinates show that the Si well occupies z = 200--208 nm.
SI_WELL_BOTTOM_M = 200.0e-9
SI_WELL_TOP_M = 208.0e-9

# Rough-interface parameters taken from the QTCAD example.
ROUGH_HURST = 0.3
ROUGH_RMS_M = 0.5e-9
# Baseline study: hold the random SiGe alloy realization fixed and sample 100
# independent pairs of rough top/bottom interfaces. This isolates the effect of
# rough-interface disorder on the valley-splitting distribution.
ATOMIC_ALLOY_SEED = 104
ROUGH_SURFACE_MASTER_SEEDS = range(1, 101)

# Full atomic structure used for Keating relaxation.
# This covers the well and 10 nm of SiGe on each side.
ATOMS_X_M = np.array([-10.0, 10.0]) * 1e-9
ATOMS_Y_M = np.array([-10.0, 10.0]) * 1e-9
ATOMS_Z_M = np.array([190.0, 218.0]) * 1e-9

# Restricted atomistic domain used for the tight-binding solve.
# It contains the 8 nm Si well plus 2 nm of SiGe on each side.
TB_X_M = np.array([-8.0, 8.0]) * 1e-9
TB_Y_M = np.array([-8.0, 8.0]) * 1e-9
TB_Z_M = np.array([198.0, 210.0]) * 1e-9

TB_NUM_STATES = 10
TB_ENERGY_TARGET_EV = E_TARGET_EV

# Output controls.
SAVE_FEM_VTU = False
SAVE_TB_WAVEFUNCTIONS = False
WAVEFUNCTION_MASTER_SEED = 1
TB_WAVEFUNCTION_NUM_STATES = 3
SAVE_ROUGH_INTERFACE_PLOTS = False
HISTOGRAM_BINS = 15


# =============================================================================
# Paths
# =============================================================================

script_dir = pathlib.Path(__file__).parent.resolve()
mesh_path = script_dir / "meshes" / "1_dot.msh"
output_dir = script_dir / "output"
output_dir.mkdir(exist_ok=True)

if not mesh_path.exists():
    raise FileNotFoundError(
        f"Mesh not found: {mesh_path}\n"
        "Place 1_dot.msh in the meshes directory beside this script."
    )


# =============================================================================
# FEM electrostatics and effective-mass calculation
# =============================================================================

mesh = Mesh(SCALING, str(mesh_path))
d = Device(mesh, conf_carriers="e")
d.set_temperature(TEMPERATURE_K)

# Use the same FEM material definitions as the original calculation.
SiGe_material = mt.Material(mt.gen_SiGe(0.30))
SiGe_material.set_param("chi", 3.87 * ct.e)

Al2O3_material = copy.deepcopy(mt.Al2O3_ideal)
Al2O3_material.set_param("chi", 1.25 * ct.e)

d.new_region("Al2O3", Al2O3_material)
d.new_region("SiGe_1", SiGe_material)
d.new_region("SiGe_1_dot", SiGe_material)
d.new_region("Si", mt.Si)
d.new_region("SiGe_2_dot", SiGe_material)
d.new_region("SiGe_2", SiGe_material)

d.new_insulator("Al2O3")
d.new_insulator("SiGe_1")
d.new_insulator("SiGe_1_dot")
d.new_insulator("SiGe_2_dot")
d.new_insulator("SiGe_2")

Ew = GATE_WORK_FUNCTION_EV * ct.e
d.new_gate_bnd("QS0", phi=V_SC0, Ew=Ew)
d.new_gate_bnd("QS1", phi=V_SC1, Ew=Ew)
d.new_gate_bnd("QT0", phi=V_QT0, Ew=Ew)
d.new_gate_bnd("QT1", phi=V_QT1, Ew=Ew)
d.new_gate_bnd("QP0", phi=QP0_INITIAL_GUESS_V, Ew=Ew)
d.new_ohmic_bnd("Si_ohmic")

poisson_solver = PoissonSolver(d=d)

# The FEM Schrödinger calculation is restricted to the dot portion of the
# SiGe/Si/SiGe stack.
qd_regions = ["SiGe_1_dot", "Si", "SiGe_2_dot"]
submesh = SubMesh(d.mesh, qd_regions)


def ground_state_residual(V_QP0: float) -> float:
    """Return E0(V_QP0) - E_TARGET_EV in electronvolts."""
    print("\n" + "=" * 72)
    print(f"FEM solve for V_QP0 = {V_QP0:.8f} V")
    print("=" * 72)

    d.set_applied_potential("QP0", V_QP0)
    poisson_solver.solve()
    d.set_V_from_phi()

    fem_dot = SubDevice(d, submesh)
    fem_dot.set_V_from_phi()

    fem_solver = FEMSchrodingerSolver(d=fem_dot)
    fem_solver.solve()

    energies_ev = fem_dot.energies / ct.e
    print("FEM energies (eV):", energies_ev)

    return float(energies_ev[0] - E_TARGET_EV)


V_QP0_ROOT = newton(
    ground_state_residual,
    x0=QP0_INITIAL_GUESS_V,
    tol=1e-6,
    maxiter=20,
)

print("\n" + "#" * 72)
print(f"Converged QP0 voltage: {V_QP0_ROOT:.10f} V")
print("#" * 72)

# Re-solve at the converged voltage. The converged spatial potential remains
# directly available in d; no HDF5 save/load step is needed.
d.set_applied_potential("QP0", V_QP0_ROOT)
poisson_solver.solve()
d.set_V_from_phi()

fem_dot = SubDevice(d, submesh)
fem_dot.set_V_from_phi()
fem_solver = FEMSchrodingerSolver(d=fem_dot)
fem_solver.solve()

fem_energies_ev = fem_dot.energies / ct.e
print("Final FEM energies (eV):", fem_energies_ev)

if SAVE_FEM_VTU:
    cb_path = output_dir / "multiscalegates_CB.vtu"
    conduction_band_ev = d.cond_band_edge() / ct.e
    io.save(str(cb_path), {"EC (eV)": conduction_band_ev}, d.mesh)

    wf_path = output_dir / "multiscalegates_wf.vtu"
    fem_wf_dict = {
        "Ground state": fem_dot.eigenfunctions[:, 0],
        "1st excited": fem_dot.eigenfunctions[:, 1],
        "2nd excited": fem_dot.eigenfunctions[:, 2],
    }
    io.save(str(wf_path), fem_wf_dict, submesh)


# =============================================================================
# Atomistic tight-binding sweep and valley splitting
# =============================================================================

def rough_surface_seeds(master_seed: int) -> tuple[int, int]:
    """Return independent lower- and upper-interface seeds for one realization."""
    return tuple(
        int(value)
        for value in np.random.SeedSequence(master_seed).generate_state(2)
    )


def run_atomistic_realization(master_seed: int) -> tuple[float, float, float, int, int]:
    """Solve one rough-interface realization using the fixed alloy realization."""
    bottom_seed, top_seed = rough_surface_seeds(master_seed)
    print("\n" + "=" * 72)
    print(
        f"Rough-interface realization {master_seed}: alloy={ATOMIC_ALLOY_SEED}, "
        f"bottom={bottom_seed}, top={top_seed}"
    )
    print("=" * 72)

    atoms = Atoms(
        x=ATOMS_X_M,
        y=ATOMS_Y_M,
        z=ATOMS_Z_M,
        rng_seed=ATOMIC_ALLOY_SEED,
    )
    well_bottom = RoughSurface(
        hurst=ROUGH_HURST,
        mean=SI_WELL_BOTTOM_M,
        rms=ROUGH_RMS_M,
        rng_seed=bottom_seed,
    )
    well_top = RoughSurface(
        hurst=ROUGH_HURST,
        mean=SI_WELL_TOP_M,
        rms=ROUGH_RMS_M,
        rng_seed=top_seed,
    )

    if SAVE_ROUGH_INTERFACE_PLOTS:
        x_plot = np.linspace(ATOMS_X_M[0], ATOMS_X_M[1], 100)
        y_plot = np.linspace(ATOMS_Y_M[0], ATOMS_Y_M[1], 100)
        well_bottom.plot(
            x=x_plot,
            y=y_plot,
            type="2D",
            path=str(output_dir / f"rough_interface_bottom_seed_{master_seed:02d}_2d.png"),
            show_figure=False,
        )
        well_top.plot(
            x=x_plot,
            y=y_plot,
            type="2D",
            path=str(output_dir / f"rough_interface_top_seed_{master_seed:02d}_2d.png"),
            show_figure=False,
        )

    # Atoms are Si by default. Replace atoms below and above the rough Si well
    # with a random Si0.7Ge0.3 alloy and then relax the structure.
    SiGe_alloy = np.array([["Si", 0.7], ["Ge", 0.3]])
    atoms.new_layer(z_top=well_bottom, z_bot=ATOMS_Z_M[0], atom_species=SiGe_alloy)
    atoms.new_layer(z_top=ATOMS_Z_M[1], z_bot=well_top, atom_species=SiGe_alloy)

    keating_params = KeatingSolverParams()
    keating_params.verbose = True
    KeatingSolver(atoms=atoms, solver_params=keating_params).solve()

    atoms_QD = SubAtoms(parent=atoms, x=TB_X_M, y=TB_Y_M, z=TB_Z_M)
    atoms_QD.set_potential_from_device(d=d)

    tb_params = TBSchrodingerSolverParams()
    tb_params.verbose = True
    tb_params.num_states = TB_NUM_STATES
    tb_solver = TBSchrodingerSolver(atoms=atoms_QD, solver_params=tb_params)
    tb_solver.solve(energy_target=TB_ENERGY_TARGET_EV * ct.e)
    atoms_QD.print_energies()

    tb_energies_ev = np.asarray(atoms_QD.energies, dtype=float) / ct.e
    if tb_energies_ev.size < 2:
        raise RuntimeError("The tight-binding solver returned fewer than two states.")

    if SAVE_TB_WAVEFUNCTIONS and master_seed == WAVEFUNCTION_MASTER_SEED:
        num_wavefunction_states = min(
            TB_WAVEFUNCTION_NUM_STATES, len(atoms_QD.eigenfunctions)
        )
        if num_wavefunction_states == 0:
            raise RuntimeError("The tight-binding solver returned no eigenfunctions.")
        wavefunction_data = {
            f"Probability density (TB state {state_index})": (
                atoms_QD.eigenfunctions[state_index].get_prob_on_atoms()
            )
            for state_index in range(num_wavefunction_states)
        }
        wavefunction_path = output_dir / f"tb_wavefunctions_rough_seed_{master_seed:03d}.vtu"
        save_vtu(atoms=atoms_QD, out_dict=wavefunction_data, path=str(wavefunction_path))
        print(f"3D wavefunction data saved to: {wavefunction_path}")

    valley_splitting_ev = tb_energies_ev[1] - tb_energies_ev[0]
    print(f"Valley splitting: {valley_splitting_ev * 1e6:.12e} micro-eV")
    return (
        float(tb_energies_ev[0]),
        float(tb_energies_ev[1]),
        float(valley_splitting_ev),
        bottom_seed,
        top_seed,
    )


results = []

results_path = output_dir / "valley_splitting_rough_surface_sweep.csv"
failed_path = output_dir / "failed_realizations.log"

header=[
"rough_surface_master_seed","alloy_seed","bottom_interface_seed","top_interface_seed",
"V_QP0_V","FEM_E0_eV","TB_E0_eV","TB_E1_eV","valley_splitting_eV","valley_splitting_meV","valley_splitting_microeV"
]
with open(results_path,"w",newline="") as f:
    csv.writer(f).writerow(header)
open(failed_path,"w").write("Failed realizations")

for master_seed in ROUGH_SURFACE_MASTER_SEEDS:
    try:
        tb_e0_ev,tb_e1_ev,splitting_ev,bottom_seed,top_seed=run_atomistic_realization(master_seed)
        row=[master_seed,ATOMIC_ALLOY_SEED,bottom_seed,top_seed,V_QP0_ROOT,fem_energies_ev[0],
             tb_e0_ev,tb_e1_ev,splitting_ev,splitting_ev*1e3,splitting_ev*1e6]
        results.append(row)
        with open(results_path,"a",newline="") as f:
            csv.writer(f).writerow(row)
        print(f"Completed seed {master_seed}")
    except Exception:
        print(f"Seed {master_seed} FAILED")
        with open(failed_path,"a") as f:
            f.write(f"Seed {master_seed}")
            traceback.print_exc(file=f)
            f.write("")
        continue

if results:
    valley_splittings_uev=np.asarray([r[-1] for r in results])
    histogram_path=output_dir/"valley_splitting_rough_surface_pdf.png"
    fig,axis=plt.subplots(figsize=(7,4.5))
    axis.hist(valley_splittings_uev,bins=min(HISTOGRAM_BINS,len(valley_splittings_uev)),density=True,color="tab:blue",edgecolor="white")
    axis.set_xlabel("Valley splitting (micro-eV)")
    axis.set_ylabel("Probability density (1 / micro-eV)")
    axis.set_title(f"Valley splitting across {len(results)} successful realizations")
    axis.grid(axis="y",alpha=0.25)
    fig.tight_layout()
    fig.savefig(histogram_path,dpi=300)
    plt.close(fig)
    print(""+"#"*72)
    print(f"Successful realizations: {len(results)}")
    print(f"Mean valley splitting: {np.mean(valley_splittings_uev):.6f} micro-eV")
    if len(results)>1:
        print(f"Std. dev.: {np.std(valley_splittings_uev,ddof=1):.6f} micro-eV")
        print(f"Sweep results saved to: {results_path}")
        print(f"Failure log: {failed_path}")
        print("#"*72)
else:
    print("No successful realizations.")