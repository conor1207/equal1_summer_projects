Overview

This repository contains the simulation scripts, analysis and results from a summer project at [organisation / research group] ([month–month year]), supervised by [supervisor name, if they agree to be named].

The project used QTCAD, a commercial simulator for semiconductor quantum devices, to study valley splitting in silicon quantum dots. Valley splitting is a key property that limits how well silicon spin qubits can work.

Background: what is valley splitting, and why does it matter?

Silicon spin qubits store quantum information in the spin of a single electron trapped in a tiny region called a quantum dot. Silicon is attractive for this because it is compatible with existing chip manufacturing and can be purified to remove most nuclear spins, which cause noise.

Silicon has a complication, though. In bulk silicon, the lowest-energy states for a conduction electron sit in six equivalent "valleys" in momentum space. In a strained Si/SiGe quantum well, strain raises four of these valleys in energy, leaving two low-energy valleys. An electron in the dot can occupy either of these two valley states, so it has an extra quantum degree of freedom alongside its spin.

The sharp interface of the quantum well couples the two valleys and splits their energies. This energy gap is the valley splitting (E<sub>VS</sub>).

If E<sub>VS</sub> is large compared with the thermal energy and the qubit's Zeeman energy, the electron stays in the lower valley and the qubit behaves like a clean two-level spin system.
If E<sub>VS</sub> is small, the electron can leak into the excited valley state. This causes errors, faster relaxation near "spin–valley hotspots", and problems with spin readout methods such as Pauli spin blockade.

In Si/SiGe devices the valley splitting is often only tens to a few hundred µeV. It also varies from dot to dot, because it is very sensitive to atomic-scale details of the interface: steps, roughness and alloy disorder. For comparison, the thermal energy at 100 mK is about 8.6 µeV. Understanding and predicting valley splitting is therefore an active research problem.