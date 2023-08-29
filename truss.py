"""
truss.py
August 2023
"""

import numpy as np
from numpy import linalg

from bar import Bar

class Truss:
    """A truss structure made up of 2-node bar elements.

    Args:
        - Youngs: Bar Young's moduli
        - densities: Bar densities
        - areas: Coss-sectional areas
        - node_positions: Node positions
        - el_nodes: A list of two nodes constituent bars
    """

    def __init__(self, Youngs, densities, areas, node_positions, element_nodes):

        # Assign node positions
        self.node_positions = node_positions
        self.num_nodes = len(node_positions)

        # Assign element nodes
        self.element_nodes = element_nodes
        self.num_elements = len(element_nodes)

        # Initialize node displacements & forces
        self.displacements = np.zeros(3*self.num_nodes)
        self.forces = np.zeros(3*self.num_nodes)

        # Initialize list of bar2 elements
        self.elements = []
        for e in range(self.num_elements):
            ePos = np.array([self.node_positions[n] for n in self.element_nodes[e]])
            new_bar = Bar(Youngs[e], densities[e], areas[e], ePos)
            self.elements.append(new_bar)

        # Compute the truss' weight
        self.weight = sum([el.weight for el in self.elements])

        # Call compute the truss' global stiffness matrix
        self.stiffness = np.zeros((3*self.num_nodes, 3*self.num_nodes))
        for e in range(0,self.num_elements):
            Ke = self.elements[e].stiffness
            for i in range(2):
                ii = self.element_nodes[e,i]
                for j in range(2):
                    jj = self.element_nodes[e,j]
                    Ke_block = Ke[3*i:3*(i+1), 3*j:3*(j+1)]
                    self.stiffness[3*ii:3*(ii+1), 3*jj:3*(jj+1)] += Ke_block
        
    # Apply nodal forces to the truss
    def applyForces(self, DOF, forces):

        # Modify the stiffness matrix given degrees of freedom
        Kmod = self.stiffness.copy()
        Kmod[:, DOF==0] = 0
        Kmod[DOF==0, :] = 0
        Kmod[DOF==0, DOF==0] = 1

        # Compute the modified applied force vector
        Fmod = forces.copy()
        Fmod[DOF!=0] -= self.stiffness[DOF!=0][:,DOF==0] @ forces[DOF==0]

        # Compute nodal displacements & reaction forces
        self.displacements = linalg.inv(Kmod) @ Fmod
        self.forces = self.stiffness @ self.displacements

        # Apply nodal displacements to elements
        for e in range(self.num_elements):
            inds = np.array([
                [3*i, 3*i+1, 3*i+2] for i in self.element_nodes[e]
            ])
            self.elements[e].apply_disp(self.displacements[inds.flatten()])

    # Re-initializes the truss with no nodal displacements
    def reset(self):
        self.displacements = np.zeros(3*self.num_nodes)
        self.forces = np.zeros(3*self.num_nodes)
        for el in self.elements: el.reset()
