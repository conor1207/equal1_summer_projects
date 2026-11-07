import copy
import csv
import pathlib

import numpy as np

from qtcad.device import constants as ct
from qtcad.device import materials as mt
from qtcad.device.mesh3d import Mesh, SubMesh
from qtcad.device import Device, SubDevice
from qtcad.device.poisson_linear import Solver as PoissonSolver
from qtcad.device.schrodinger import Solver as SchrodingerSolver
from qtcad.device import io
from qtcad.device.analysis import plot_slice, analyze_dot


# =============================================================================
# User settings
# =============================================================================

# Fixed-gate voltages
V_sc0 = 0.0
V_sc1 = 0.0
V_qt0 = 0.8
V_qt1 = 0.8

# Coarse Qp voltage sweep
V_Qp_values = np.round(np.arange(0.0, 1.0 + 0.05, 0.1), 1)

# Desired position of the quantum-dot ground state relative to the Fermi level:
#
#     E0 - EF = -5 meV
#
TARGET_E0_MINUS_EF_EV = -5.0e-3

# Stop tuning when the ground-state alignment is within 0.01 meV.
ENERGY_TOLERANCE_EV = 1.0e-5

# Maximum number of voltage-bisection iterations
MAX_TUNING_ITERATIONS = 30

# Voltage interval tolerance for the tuning procedure
VOLTAGE_TOLERANCE_V = 1.0e-6

# Whether to save detailed files at every coarse sweep point.
#
# Setting this to True produces many HDF5, VTU, and PNG files.
SAVE_COARSE_SWEEP_OUTPUTS = True


# =============================================================================
# Helper functions
# =============================================================================

def voltage_tag(voltage, prefix="Qp"):
    """
    Convert a voltage to a filename-safe tag.

    Examples
    --------
    0.3  -> Qp_0p300000
    -0.2 -> Qp_m0p200000
    """
    voltage_string = f"{float(voltage):.6f}"
    voltage_string = voltage_string.replace("-", "m")
    voltage_string = voltage_string.replace(".", "p")
    return f"{prefix}_{voltage_string}"


def get_scalar_fermi_level(device):
    """
    Return the device Fermi level as a scalar in joules.
    """
    EF_array = np.asarray(device.EF, dtype=float).reshape(-1)

    if EF_array.size == 0:
        raise RuntimeError("The device Fermi level d.EF is empty.")

    if EF_array.size > 1:
        print(
            "Warning: d.EF contains more than one value. "
            "The first value will be used."
        )

    return float(EF_array[0])


def get_schrodinger_energies(subdevice):
    """
    Retrieve the Schrödinger eigenenergies.

    The attribute name can differ between some QTCAD versions.
    """
    possible_names = (
        "energies",
        "eigenenergies",
        "eigenvalues",
    )

    for attribute_name in possible_names:
        if hasattr(subdevice, attribute_name):
            values = getattr(subdevice, attribute_name)

            if values is not None:
                energies = np.asarray(values, dtype=float).reshape(-1)

                if energies.size > 0:
                    return energies

    raise AttributeError(
        "Could not find the Schrödinger energies on the SubDevice. "
        "Checked attributes: energies, eigenenergies, and eigenvalues."
    )


def safely_delete_subdevice(subdevice):
    """
    Release the parent-device reference retained by a SubDevice, when the
    installed QTCAD version provides a delete() method.
    """
    if subdevice is not None and hasattr(subdevice, "delete"):
        subdevice.delete()


# =============================================================================
# Paths and mesh
# =============================================================================

script_dir = pathlib.Path(__file__).parent.resolve()

path_mesh = str(
    script_dir / "meshes" / "multiscalegates.msh"
)

output_dir = script_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)

# The mesh dimensions are assumed to have been defined in nanometres.
scaling = 1.0e-9
mesh = Mesh(scaling, path_mesh)


# =============================================================================
# Device and material definitions
# =============================================================================

# Electron device
d = Device(mesh, conf_carriers="e")
d.set_temperature(0.5)

# -----------------------------------------------------------------------------
# Material band alignment
# -----------------------------------------------------------------------------
#
# Use independent material objects rather than modifying QTCAD's global
# material definitions.
#
# The electron affinities below are the values supplied by the supervisor.
# -----------------------------------------------------------------------------

# Si0.70Ge0.30
SiGe_material = mt.Material(mt.gen_SiGe(0.30))
SiGe_material.set_param("chi", 3.87 * ct.e)

# Al2O3
Al2O3_material = copy.deepcopy(mt.Al2O3_ideal)
Al2O3_material.set_param("chi", 1.25 * ct.e)

# Define the heterostructure material stack
d.new_region("oxide", Al2O3_material)
d.new_region("sige_top", SiGe_material)
d.new_region("si_well", mt.Si)
d.new_region("sige_buffer", SiGe_material)


# =============================================================================
# Boundary conditions
# =============================================================================

# TiN (100) work function parameter
Ew = 2.9 * ct.e

# Define each gate once. The Qp voltage will be changed with
# d.set_applied_potential() during the sweep.
d.new_gate_bnd("gate_sc0", phi=V_sc0, Ew=Ew)
d.new_gate_bnd("gate_sc1", phi=V_sc1, Ew=Ew)
d.new_gate_bnd("gate_qt0", phi=V_qt0, Ew=Ew)
d.new_gate_bnd("gate_qt1", phi=V_qt1, Ew=Ew)
d.new_gate_bnd("gate_Qp", phi=float(V_Qp_values[0]), Ew=Ew)


# =============================================================================
# Solvers and quantum-dot submesh
# =============================================================================

poisson_solver = PoissonSolver(d=d)

# Slice plane normal to z
normal = (0.0, 0.0, 1.0)

# Middle of the silicon quantum well
origin = (0.0, 0.0, -40.0 * scaling)

# Regions included in the quantum calculation
quantum_dot_regions = [
    "sige_top",
    "si_well",
    "sige_buffer",
]

submesh = SubMesh(d.mesh, quantum_dot_regions)


# =============================================================================
# Output function
# =============================================================================

def save_solution_outputs(V_Qp, result, subdevice, tag):
    """
    Save the electrostatic, band-edge, confinement-potential, and wavefunction
    results for one voltage.
    """

    # -------------------------------------------------------------------------
    # Electrostatic potential
    # -------------------------------------------------------------------------

    path_phi_hdf5 = str(
        output_dir / f"multiscalegates_phi_{tag}.hdf5"
    )

    io.save(
        path_phi_hdf5,
        {"phi": np.asarray(d.phi)},
    )

    # -------------------------------------------------------------------------
    # Band edges and charge information
    # -------------------------------------------------------------------------

    EC_eV = np.asarray(d.cond_band_edge(), dtype=float) / ct.e
    EV_eV = np.asarray(d.vlnce_band_edge(), dtype=float) / ct.e

    EF_eV = result["EF_eV"]

    # VTU output requires a field defined at each mesh point.
    EF_field_eV = np.full(
        EC_eV.shape,
        EF_eV,
        dtype=float,
    )

    device_output = {
        "EC (eV)": EC_eV,
        "EV (eV)": EV_eV,
        "Fermi Level (eV)": EF_field_eV,
        "EC - EF (eV)": EC_eV - EF_eV,
        "EV - EF (eV)": EV_eV - EF_eV,
    }

    # Electron number density
    electron_density = getattr(d, "n", None)

    if electron_density is not None:
        electron_density = np.asarray(electron_density)

        if electron_density.shape == EC_eV.shape:
            device_output[
                "Electron density (1/m^3)"
            ] = electron_density

    # Hole number density
    hole_density = getattr(d, "p", None)

    if hole_density is not None:
        hole_density = np.asarray(hole_density)

        if hole_density.shape == EC_eV.shape:
            device_output[
                "Hole density (1/m^3)"
            ] = hole_density

    # Total charge density
    total_charge_density = getattr(d, "rho", None)

    if total_charge_density is not None:
        total_charge_density = np.asarray(total_charge_density)

        if total_charge_density.shape == EC_eV.shape:
            device_output[
                "Total charge density (C/m^3)"
            ] = total_charge_density

    # Background or fixed charge density
    background_charge_density = getattr(d, "rho_0", None)

    if background_charge_density is not None:
        background_charge_density = np.asarray(
            background_charge_density
        )

        if background_charge_density.shape == EC_eV.shape:
            device_output[
                "Background charge density (C/m^3)"
            ] = background_charge_density

    path_bands_vtu = str(
        output_dir / f"multiscalegates_bands_{tag}.vtu"
    )

    io.save(
        path_bands_vtu,
        device_output,
        d.mesh,
    )

    # -------------------------------------------------------------------------
    # Band-edge slice
    # -------------------------------------------------------------------------

    path_CB_slice = str(
        output_dir / f"multiscalegates_CB_relative_slice_{tag}.png"
    )

    plot_slice(
        d.mesh,
        EC_eV - EF_eV,
        normal=normal,
        origin=origin,
        cb_axis_label=r"$E_C-E_F$ (eV)",
        title=(
            "Conduction band relative to Fermi level, "
            f"V_Qp = {V_Qp:.6f} V"
        ),
        path=path_CB_slice,
        show_figure=False,
    )

    # An absolute-energy plot is also saved for comparison.
    path_CB_absolute_slice = str(
        output_dir / f"multiscalegates_CB_absolute_slice_{tag}.png"
    )

    plot_slice(
        d.mesh,
        EC_eV,
        normal=normal,
        origin=origin,
        cb_axis_label="Conduction band edge (eV)",
        title=(
            "Absolute conduction band edge, "
            f"V_Qp = {V_Qp:.6f} V"
        ),
        path=path_CB_absolute_slice,
        show_figure=False,
    )

    # -------------------------------------------------------------------------
    # Quantum confinement potential
    # -------------------------------------------------------------------------

    Vconf = np.asarray(subdevice.get_Vconf())

    path_Vconf_hdf5 = str(
        output_dir / f"multiscalegates_Vconf_{tag}.hdf5"
    )

    io.save(
        path_Vconf_hdf5,
        {"Vconf": Vconf},
    )

    # -------------------------------------------------------------------------
    # Eigenfunctions
    # -------------------------------------------------------------------------

    eigenfunctions = np.asarray(subdevice.eigenfunctions)

    if eigenfunctions.ndim != 2:
        raise RuntimeError(
            "Expected subdevice.eigenfunctions to be a two-dimensional array."
        )

    if eigenfunctions.shape[1] < 3:
        raise RuntimeError(
            "Fewer than three Schrödinger eigenfunctions were calculated."
        )

    wf0 = eigenfunctions[:, 0]
    wf1 = eigenfunctions[:, 1]
    wf2 = eigenfunctions[:, 2]

    path_wf_vtu = str(
        output_dir / f"multiscalegates_wf_{tag}.vtu"
    )

    io.save(
        path_wf_vtu,
        {
            "Ground state": wf0,
            "Ground-state probability density": np.abs(wf0) ** 2,
            "1st excited": wf1,
            "1st-excited probability density": np.abs(wf1) ** 2,
            "2nd excited": wf2,
            "2nd-excited probability density": np.abs(wf2) ** 2,
        },
        submesh,
    )

    # -------------------------------------------------------------------------
    # Ground-state slice
    # -------------------------------------------------------------------------

    path_wf0_slice = str(
        output_dir / f"multiscalegates_wf0_slice_{tag}.png"
    )

    plot_slice(
        submesh,
        np.abs(wf0) ** 2,
        normal=normal,
        origin=origin,
        cb_axis_label=r"Ground-state probability density $|\psi_0|^2$",
        title=(
            "Ground state, "
            f"V_Qp = {V_Qp:.6f} V, "
            f"E0 - EF = {result['E0_minus_EF_meV']:.4f} meV"
        ),
        path=path_wf0_slice,
        show_figure=False,
    )


# =============================================================================
# Single-voltage solver
# =============================================================================

def solve_at_Qp(
    V_Qp,
    save_outputs=False,
    output_tag=None,
    print_energies=False,
    run_dot_analysis=False,
):
    """
    Solve the Poisson and Schrödinger equations at one Qp voltage.

    Parameters
    ----------
    V_Qp : float
        Qp gate voltage in volts.

    save_outputs : bool
        Save HDF5, VTU, and PNG files when True.

    output_tag : str or None
        Filename tag. A voltage-based tag is generated when this is None.

    print_energies : bool
        Call the QTCAD print_energies() method when True.

    run_dot_analysis : bool
        Call analyze_dot() for the ground state when True.

    Returns
    -------
    result : dict
        Dictionary containing the gate voltage, eigenenergies, Fermi level,
        and alignment residual.
    """

    V_Qp = float(V_Qp)

    print("\n" + "=" * 78)
    print(f"Solving for V_Qp = {V_Qp:.8f} V")
    print("=" * 78)

    # Change the existing Qp gate voltage.
    d.set_applied_potential("gate_Qp", V_Qp)

    # -------------------------------------------------------------------------
    # Poisson equation
    # -------------------------------------------------------------------------

    poisson_solver.solve()

    # Construct the electron confinement potential from the electrostatic
    # potential and material band parameters.
    d.set_V_from_phi()

    # -------------------------------------------------------------------------
    # Schrödinger equation
    # -------------------------------------------------------------------------

    # Recreate the SubDevice at each voltage so that it receives the latest
    # potential from the parent Device.
    subdevice = SubDevice(d, submesh)
    subdevice.set_V_from_phi()

    schrodinger_solver = SchrodingerSolver(d=subdevice)
    schrodinger_solver.solve()

    energies_J = get_schrodinger_energies(subdevice)

    if energies_J.size < 3:
        safely_delete_subdevice(subdevice)

        raise RuntimeError(
            "The Schrödinger solver returned fewer than three energies. "
            "Increase the number of requested eigenstates in the solver "
            "parameters for your QTCAD version."
        )

    EF_J = get_scalar_fermi_level(d)

    E0_eV = energies_J[0] / ct.e
    E1_eV = energies_J[1] / ct.e
    E2_eV = energies_J[2] / ct.e
    EF_eV = EF_J / ct.e

    E0_minus_EF_eV = (energies_J[0] - EF_J) / ct.e
    E0_minus_EF_meV = E0_minus_EF_eV * 1.0e3

    # The residual is zero when E0 - EF equals -5 meV.
    target_residual_eV = (
        E0_minus_EF_eV - TARGET_E0_MINUS_EF_EV
    )

    target_residual_meV = target_residual_eV * 1.0e3

    result = {
        "V_Qp": V_Qp,
        "E0_eV": E0_eV,
        "E1_eV": E1_eV,
        "E2_eV": E2_eV,
        "EF_eV": EF_eV,
        "E0_minus_EF_eV": E0_minus_EF_eV,
        "E0_minus_EF_meV": E0_minus_EF_meV,
        "target_residual_eV": target_residual_eV,
        "target_residual_meV": target_residual_meV,
    }

    print(f"E0             = {E0_eV:.10f} eV")
    print(f"E1             = {E1_eV:.10f} eV")
    print(f"E2             = {E2_eV:.10f} eV")
    print(f"EF             = {EF_eV:.10f} eV")
    print(f"E0 - EF        = {E0_minus_EF_meV:.6f} meV")
    print(f"Desired value  = {-5.0:.6f} meV")
    print(f"Target residual = {target_residual_meV:.6f} meV")

    if print_energies:
        subdevice.print_energies()

    if run_dot_analysis:
        analyze_dot(
            subdevice,
            eigenstate=0,
            verbose=True,
        )

    if save_outputs:
        if output_tag is None:
            output_tag = voltage_tag(V_Qp)

        save_solution_outputs(
            V_Qp=V_Qp,
            result=result,
            subdevice=subdevice,
            tag=output_tag,
        )

    safely_delete_subdevice(subdevice)

    return result


# =============================================================================
# Coarse Qp sweep
# =============================================================================

coarse_results = []

for V_Qp in V_Qp_values:
    tag = voltage_tag(V_Qp)

    result = solve_at_Qp(
        V_Qp,
        save_outputs=SAVE_COARSE_SWEEP_OUTPUTS,
        output_tag=tag,
        print_energies=False,
        run_dot_analysis=False,
    )

    coarse_results.append(result)


# =============================================================================
# Save coarse-sweep summary
# =============================================================================

coarse_summary_path = (
    output_dir / "multiscalegates_Qp_sweep_energies.csv"
)

with open(
    coarse_summary_path,
    "w",
    newline="",
    encoding="utf-8",
) as summary_file:

    writer = csv.writer(summary_file)

    writer.writerow(
        [
            "V_Qp_V",
            "E0_eV",
            "E1_eV",
            "E2_eV",
            "EF_eV",
            "E0_minus_EF_meV",
            "target_residual_meV",
        ]
    )

    for result in coarse_results:
        writer.writerow(
            [
                f"{result['V_Qp']:.10g}",
                f"{result['E0_eV']:.12g}",
                f"{result['E1_eV']:.12g}",
                f"{result['E2_eV']:.12g}",
                f"{result['EF_eV']:.12g}",
                f"{result['E0_minus_EF_meV']:.12g}",
                f"{result['target_residual_meV']:.12g}",
            ]
        )


# =============================================================================
# Locate an interval containing the -5 meV target
# =============================================================================

V_Qp_tuned = None
voltage_bracket = None

# First check whether a coarse point already meets the tolerance.
for result in coarse_results:
    if abs(result["target_residual_eV"]) <= ENERGY_TOLERANCE_EV:
        V_Qp_tuned = result["V_Qp"]

        print(
            "\nA coarse sweep point already meets the requested "
            "energy tolerance."
        )
        break

# Otherwise, find adjacent points with residuals of opposite sign.
if V_Qp_tuned is None:
    for left_result, right_result in zip(
        coarse_results[:-1],
        coarse_results[1:],
    ):
        left_residual = left_result["target_residual_eV"]
        right_residual = right_result["target_residual_eV"]

        if left_residual * right_residual < 0.0:
            voltage_bracket = (
                left_result["V_Qp"],
                right_result["V_Qp"],
            )
            break

if V_Qp_tuned is None and voltage_bracket is None:
    residuals_meV = np.array(
        [
            result["target_residual_meV"]
            for result in coarse_results
        ],
        dtype=float,
    )

    raise RuntimeError(
        "\nThe condition E0 - EF = -5 meV was not bracketed by the "
        "specified Qp voltage sweep.\n"
        f"The scanned voltages were {V_Qp_values[0]:.3f} V to "
        f"{V_Qp_values[-1]:.3f} V.\n"
        "The target residual over this range was between "
        f"{residuals_meV.min():.6f} meV and "
        f"{residuals_meV.max():.6f} meV.\n"
        "Extend V_Qp_values so that the target residual changes sign."
    )


# =============================================================================
# Tune Qp using voltage bisection
# =============================================================================

if V_Qp_tuned is None:
    V_left, V_right = voltage_bracket

    print("\n" + "#" * 78)
    print("Beginning Qp voltage tuning")
    print(
        f"Initial bracket: {V_left:.8f} V to "
        f"{V_right:.8f} V"
    )
    print("#" * 78)

    left_result = solve_at_Qp(
        V_left,
        save_outputs=False,
    )

    right_result = solve_at_Qp(
        V_right,
        save_outputs=False,
    )

    f_left = left_result["target_residual_eV"]
    f_right = right_result["target_residual_eV"]

    if f_left * f_right > 0.0:
        raise RuntimeError(
            "The selected voltage interval no longer brackets the target."
        )

    best_result = (
        left_result
        if abs(f_left) < abs(f_right)
        else right_result
    )

    for iteration in range(1, MAX_TUNING_ITERATIONS + 1):
        V_middle = 0.5 * (V_left + V_right)

        print(
            f"\nTuning iteration {iteration}/"
            f"{MAX_TUNING_ITERATIONS}"
        )

        middle_result = solve_at_Qp(
            V_middle,
            save_outputs=False,
        )

        f_middle = middle_result["target_residual_eV"]

        if abs(f_middle) < abs(
            best_result["target_residual_eV"]
        ):
            best_result = middle_result

        energy_converged = (
            abs(f_middle) <= ENERGY_TOLERANCE_EV
        )

        voltage_converged = (
            abs(V_right - V_left)
            <= VOLTAGE_TOLERANCE_V
        )

        if energy_converged or voltage_converged:
            best_result = middle_result
            break

        if f_left * f_middle <= 0.0:
            V_right = V_middle
            f_right = f_middle
        else:
            V_left = V_middle
            f_left = f_middle

    V_Qp_tuned = best_result["V_Qp"]


# =============================================================================
# Final tuned calculation
# =============================================================================

print("\n" + "#" * 78)
print(f"Running final tuned solution at V_Qp = {V_Qp_tuned:.10f} V")
print("#" * 78)

final_tag = voltage_tag(
    V_Qp_tuned,
    prefix="Qp_tuned",
)

final_result = solve_at_Qp(
    V_Qp_tuned,
    save_outputs=True,
    output_tag=final_tag,
    print_energies=True,
    run_dot_analysis=True,
)

final_error_meV = abs(
    final_result["target_residual_meV"]
)

if final_error_meV > ENERGY_TOLERANCE_EV * 1.0e3:
    print(
        "\nWarning: the final alignment is outside the requested "
        "energy tolerance."
    )


# =============================================================================
# Save final tuned summary
# =============================================================================

final_summary_path = (
    output_dir / "multiscalegates_Qp_tuned.csv"
)

with open(
    final_summary_path,
    "w",
    newline="",
    encoding="utf-8",
) as final_file:

    writer = csv.writer(final_file)

    writer.writerow(
        [
            "V_Qp_V",
            "E0_eV",
            "E1_eV",
            "E2_eV",
            "EF_eV",
            "E0_minus_EF_meV",
            "target_E0_minus_EF_meV",
            "target_residual_meV",
        ]
    )

    writer.writerow(
        [
            f"{final_result['V_Qp']:.12g}",
            f"{final_result['E0_eV']:.12g}",
            f"{final_result['E1_eV']:.12g}",
            f"{final_result['E2_eV']:.12g}",
            f"{final_result['EF_eV']:.12g}",
            f"{final_result['E0_minus_EF_meV']:.12g}",
            "-5.0",
            f"{final_result['target_residual_meV']:.12g}",
        ]
    )


# =============================================================================
# Completion message
# =============================================================================

print("\n" + "=" * 78)
print("Qp voltage sweep and tuning complete")
print("=" * 78)
print(f"Tuned Qp voltage: {final_result['V_Qp']:.10f} V")
print(
    "Final ground-state position: "
    f"E0 - EF = {final_result['E0_minus_EF_meV']:.8f} meV"
)
print(
    "Distance from the -5 meV target: "
    f"{final_error_meV:.8f} meV"
)
print(f"Coarse sweep summary: {coarse_summary_path}")
print(f"Tuned result summary: {final_summary_path}")
print(f"Detailed output directory: {output_dir}")