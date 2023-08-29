"""
bar.py
August 2023
"""

import numpy as np

class Bar:
	"""A two-node bar element
	Args:
	Y: Young's modulus (Pa)
	density: Density (kg m^-3)
	area: Cross-sectional area (m^2)
	position: Positions of the two nodes (m)
	"""

	def __init__(self, Youngs, density, area, positions):

		# Assign bar parameters
		self.Youngs = Youngs
		self.density = density
		self.area = area
		self.positions = positions

		# Initialize internal forces & stresses
		self.force = 0
		self.stress = 0

		# Compute bar properties
		try:

			#
			self.span = self.positions[1] - self.positions[0]

			#
			self.length = (self.span @ self.span)**.5

			#
			self.weight = self.length * self.area * self.density

			#
			C = self.Youngs * self.area / self.length**3
			sub = np.outer(self.span, self.span)
			self.stiffness = C * np.block([[sub, -sub], [-sub, sub]])

		except TypeError:
			self.log.error()

	def apply_disp(self, disps):
		"""Applies displacements to computing internal forces & stresses

		Args:
		    disps: Nodal displacements
		"""

		# 
		dL = np.array([disps[i+3] - disps[i] for i in range(3)])
		C = self.Youngs * self.area / self.length**2

		#
		self.force = C * (self.span @ dL)
		self.stress = self.force / self.area

	# Resets internal forces & stresses
	def reset(self):
		self.force = 0
		self.stress = 0
