import ase
import numpy as np
from ase import Atoms, units
from ase.data import atomic_masses
from typing import List, Optional
import torch
from torch import nn
from torch.optim.adam import Adam
from torch import Tensor


class LiquidGenerator(nn.Module):
    def __init__(
        self, atoms_list: List[Atoms], cell: Optional[np.ndarray] = None, density: Optional[float] = 1.0, 
        cubic_cell: bool = True, wall:bool = False,
    ) -> None:
        """
        args:
            atoms_list (List[Atoms]): The atoms to be filled into the simulation box.
            cell (np.ndarray or None, optional): The cell shape of simulation box. Default to None.
            density (float or None, optional): Create the simulation box from the density. If the cell is provided, this parameter will be 
                ignored. Defalt to 0.7.
            cubic_cell (bool, optional): Create the cubic simulation box. If false, the cell length is allow to change (orthogonal cell). 
                Default to False.
            wall (bool, optional): Add additional constraints to refrine the molecule to cross the periodical boundary along z axis. This 
                can be used to generate a liquid / soild interface. Default to False
        """
        super(LiquidGenerator, self).__init__()
        self.mol = None
        self.density = density
        self.cubic_cell = cubic_cell
        self.idx = np.array(
            [i + 1 for i, a in enumerate(atoms_list) for j in range(len(a))]
        )
        self.num_of_mols = len(atoms_list)

        positions = np.vstack([a.get_positions() for a in atoms_list])
        self.register_buffer("positions", torch.tensor(positions, dtype=torch.float32))

        self.numbers = np.concatenate([a.get_atomic_numbers() for a in atoms_list])
        self.mass = atomic_masses[self.numbers].sum()
        
        if cell is None:
            self.length = (self.mass / units.kg / self.density * 1e27) ** (1 / 3.0)
            if self.cubic_cell:
                cell = torch.eye(3, dtype=torch.float32) * self.length
                self.register_buffer("_cell", cell)
            else:
                self.length_a = nn.Parameter(torch.tensor(self.length, dtype=torch.float32))
                self.length_b = nn.Parameter(torch.tensor(self.length, dtype=torch.float32))
        else:
            self.length = np.linalg.norm(cell, axis=1).mean()
            cell = torch.tensor(cell, dtype=torch.float32)
            self.register_buffer("_cell", cell)

        self.translation = nn.Parameter(torch.rand([self.num_of_mols, 3]) * self.length)
        self.rotation = nn.Parameter(torch.rand([self.num_of_mols, 3]) * np.pi)
        self.wall = wall

    def fit(self, epochs=50, lr=0.1):
        optimizer = Adam(self.parameters(), lr=0.1)
        print("step  score  cell_x  cell_y  cell_z")
        for i in range(epochs):
            optimizer.zero_grad()
            score = self.score()
            score.backward()
            optimizer.step()
            print(
                f" {i:3d} {score.item():10.2f}"
                f" {self.cell[0,0]:5.2f} {self.cell[1,1]:5.2f} {self.cell[2,2]:5.2f}"
            )

    @property
    def cell(self) -> None:
        if hasattr(self, "_cell"):
            return self._cell
        else:
            length_c = self.length ** 3 / self.length_a / self.length_b
            return torch.eye(
                3, device=self.length_a.device, dtype=torch.float32
            ) * torch.stack([self.length_a, self.length_b, length_c])

    def _generate(self) -> List[np.ndarray]:
        mol_positions = [
            self.positions[self.idx == i + 1] for i in range(self.num_of_mols)
        ]
        translation = torch.mm(torch.remainder(self.translation, 1), self.cell)  # wrap

        gen_positions = [
            translation_and_rotation(pos, vec, rot)
            for pos, vec, rot in zip(mol_positions, translation, self.rotation)
        ]
        return gen_positions

    def to_atoms(self) -> Atoms:
        gen_positions = self._generate()
        positions = np.empty([len(self.positions), 3])
        for i in range(self.num_of_mols):
            positions[self.idx == i + 1] = gen_positions[i].detach().cpu().numpy()
        atoms = Atoms(
            positions=positions,
            cell=self.cell.detach().cpu().numpy(),
            numbers=self.numbers,
            pbc=[1, 1, 1],
        )
        return atoms

    def score(self) -> float:
        dist = distance_matrix(self._generate(), self.cell)
        s = torch.sum((3.0 - dist[dist < 3.0]) ** 2)
        if self.wall:
            wdist = wall_dist(self._generate(), self.cell)
            s += 10 * torch.sum(wdist)**2
        return s
    

def translation_and_rotation(positions: Tensor, vec: Tensor, rot: Tensor) -> Tensor:
    device = positions.device
    alpha, beta, gamma = rot

    rotation_x = torch.stack(
        [
            torch.tensor([1., 0., 0.], device=device),
            torch.stack([torch.tensor(0.0, device=device), torch.cos(alpha), -torch.sin(alpha)]),
            torch.stack([torch.tensor(0.0, device=device), torch.sin(alpha), torch.cos(alpha)]),
        ]
    )

    rotation_y = torch.stack(
        [
            torch.stack([torch.cos(beta), torch.tensor(0.0, device=device), -torch.sin(beta)]),
            torch.tensor([0.0, 1.0, 0.0], device=device),
            torch.stack([torch.sin(beta), torch.tensor(0.0, device=device), torch.cos(beta)]),
        ]
    )

    rotation_z = torch.stack(
        [
            torch.stack([torch.cos(gamma), -torch.sin(gamma), torch.tensor(0.0, device=device)]),
            torch.stack([torch.sin(gamma), torch.cos(gamma), torch.tensor(0.0, device=device)]),
            torch.tensor([0.0, 0.0, 1.0], device=device),
        ]
    )

    return torch.mm(
        torch.mm(torch.mm(positions, rotation_x), rotation_y), rotation_z
    ) + vec.view([1, 3])


def make_supercell(positions: Tensor, cell: Tensor) -> Tensor:
    device = positions.device
    vecs = torch.mm(
        torch.stack(torch.meshgrid([torch.tensor([-1.0, 0.0, 1.0], device=device)] * 3))
        .view([3, -1])
        .T,
        cell,
    )
    return (positions.unsqueeze(axis=0) + vecs.unsqueeze(axis=1)).view([-1, 3])


def distance_matrix(list_of_positions: List[Tensor], cell: Tensor) -> Tensor:
    positions = torch.cat(list_of_positions, axis=0)
    supercell = make_supercell(positions, cell)
    # TO FIX backward problem in torch.cdist in torch 1.5
    dist = torch.sqrt(
        torch.sum(
            torch.square(positions.unsqueeze(axis=1) - supercell.unsqueeze(axis=0)),
            axis=2,
        )
        + 1e-16
    )
    return dist

def wall_dist(list_of_positions: List[Tensor], cell: Tensor) -> Tensor:
    positions_z = torch.cat(list_of_positions, axis=0)[:,2]
    return torch.cat([-positions_z[positions_z < 0], positions_z[positions_z>cell[2,2]]-cell[2,2]])

