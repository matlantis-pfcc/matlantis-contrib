# Style real
# mass = grams/mole
# distance = Angstroms
# time = femtoseconds
# energy = kcal/mol
# velocity = Angstroms/femtosecond
# force = (kcal/mol)/Angstrom
# torque = kcal/mol
# temperature = Kelvin
# pressure = atmospheres
# dynamic viscosity = Poise
# charge = multiple of electron charge (1.0 is a proton)
# dipole = charge*Angstroms
# electric field = volts/Angstrom
# density = g/cm^dim
#
# style metal
# mass = grams/mole
# distance = Angstroms
# time = picoseconds
# energy = eV
# velocity = Angstroms/picosecond
# force = eV/Angstrom
# torque = eV
# temperature = Kelvin
# pressure = bars
# dynamic viscosity = Poise
# charge = multiple of electron charge (1.0 is a proton)
# dipole = charge*Angstroms
# electric field = volts/Angstrom
# density = gram/cm^dim
from ase.units import kcal


def real_to_meal(masses, charges, positions, velocities, forces):
    masses = masses
    charges = charges
    positions = positions
    velocities = velocities * 1e3
    forces = forces / kcal
    return masses, charges, positions, velocities, forces
