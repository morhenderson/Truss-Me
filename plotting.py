"""
Plotting utilites for the Truss-Me program

Modules:
    plot_struc: Plots a graphical representation of a truss structure
	plot_disps: Plots node displacements in a truss structure
	plot_stress: Plots element stresses in a truss structure
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as LSCM

def plot_struct(truss, label_nodes=True):
	"""Plots a graphical representation of a truss structure

	Args:
	    Args:
	    truss: A 3D Truss object
		label_nodes: (default True)
	"""

	# Copy useful quantities from the truss & its elements
	node_positions = truss.node_positions.copy()
	max_dim = 1.1 * abs(node_positions).max()

	# Initialize the figure
	fig = plt.figure(figsize=(7,5))
	fig.suptitle("Truss Structure", fontsize=18)
	ax3d = fig.add_subplot(111, projection='3d')
	ax3d.locator_params(nbins=4)
	for ax in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
		ax.set_major_formatter(lambda x, pos: "{:.0f} m".format(x))
	ax3d.grid(False)

	# Plot & label points for nodes if so indicated, plot lines for bar elements
	if label_nodes:
		ax3d.scatter(node_positions[:,0], node_positions[:,1], node_positions[:,2],
			color=[.95, .66, 0, 1.0], alpha=1, s=200)
		for i in range(truss.num_nodes):
			ax3d.text(node_positions[i,0], node_positions[i,1], node_positions[i,2],str(i+1),
				color="k",ha="center",va="center",fontsize=12)
	for el in truss.elements:
		ax3d.plot(el.positions[:,0],el.positions[:,1],el.positions[:,2],'k-',lw=4)
	
	# Configure remaining plot settings
	ax3d.set_xlim(-max_dim, max_dim)
	ax3d.set_ylim(-max_dim, max_dim)
	ax3d.set_zlim(0, max_dim)
	ax3d.tick_params(labelsize=14,pad=5)
	for ax in [ax3d.xaxis,ax3d.yaxis,ax3d.zaxis]:
		ax.set_pane_color((.8, .8, .8, 1.0))
		ax.pane.set_edgecolor((0, 0, 0, 1.0))
	fig.tight_layout()
	
	return fig, ax3d

def plot_disps(truss, magnify=10.):
	"""Plots node displacements in a truss structure

	Args:
	    truss: A 3D Truss object
		magnify: Magnification factor applied to displacements (default 10.)
	"""

	# Copy useful quantities from the truss & its elements
	node_positions = truss.node_positions.copy()
	max_dim = 1.1 * abs(node_positions).max()
	mag_disps = magnify * truss.displacements.reshape(truss.num_nodes,3)
	disp_positions = node_positions + mag_disps

	# Initialize the figure
	fig = plt.figure(figsize=(7,5))
	fig.suptitle("Nodal Diplacements", fontsize=18)
	ax3d = fig.add_subplot(111, projection='3d')
	ax3d.locator_params(nbins=4)
	for ax in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
		ax.set_major_formatter(lambda x, pos: "{:.0f} m".format(x))
	ax3d.grid(False)

	# Plot points & lines for nodes & bar elements (regular & displaced)
	ax3d.scatter(
		node_positions[:,0], node_positions[:,1], node_positions[:,2],
		color=[.9, .9, .9, 1.0], alpha=0.75, s=100
	)
	for e in range(truss.num_elements):
		ePos = truss.elements[e].positions.reshape(len(truss.element_nodes[e]),3)
		dePos = ePos + mag_disps[truss.element_nodes[e]]
		ax3d.plot(ePos[:,0], ePos[:,1], ePos[:,2], '-', color=[.9, .9, .9, 1.0], lw=4, alpha=0.75)
		ax3d.plot(dePos[:,0], dePos[:,1], dePos[:,2], '-', color='k', lw=4)
	ax3d.scatter(
		disp_positions[:,0], disp_positions[:,1], disp_positions[:,2], 
		color='k', alpha=1, s=100
	)

	# Configure remaining plot settings
	ax3d.set_xlim(-max_dim, max_dim)
	ax3d.set_ylim(-max_dim, max_dim)
	ax3d.set_zlim(0, max_dim)
	ax3d.tick_params(labelsize=14,pad=5)
	for ax in [ax3d.xaxis,ax3d.yaxis,ax3d.zaxis]:
		ax.set_pane_color((.7, .7, .7, 1.0))
		ax.pane.set_edgecolor((0, 0, 0, 1.0))
	fig.tight_layout()
	
	return fig, ax3d

def plot_stress(truss, stress_units="MPa"):
	"""Plots element stresses in a truss structure

	Args:
	    truss: A 3D Truss object
		stress_units: Units for scaling stress values (Pa, KPa, MPa, or GPa)
	"""

	# Apply choice of stress units
	units = {"Pa": 1., "KPa": 1.e3, "MPa": 1.e6, "GPa": 1.e9,}
	try:
		unit = units[stress_units]
	except KeyError:
		raise ValueError("'stress_units' must be one of: " + ", ".join(units))

	# Define a color mapping function and Scalar Mappable object
	def stress_color(stress,max_stress):
		R = 1-min(abs(stress / max_stress),1) * (stress<0)
		G = 1-min(abs(stress / max_stress),1)
		B = 1-min(abs(stress / max_stress),1) * (stress>0)
		return np.array([R,G,B])
	stress_cs = [stress_color(i-5, 5) for i in range(11)]
	stress_map = LSCM.from_list('stress_map', stress_cs, N=1e3)
	stress_vals = plt.cm.ScalarMappable(None, stress_map)

	# Copy useful quantities from the truss & its elements
	node_positions = truss.node_positions.copy()
	el_stresses = np.array([el.stress for el in truss.elements]) / unit
	max_dim = 1.1 * abs(node_positions).max()
	max_stress = abs(el_stresses).max()

	# Initialize the figure
	fig = plt.figure(figsize=(7,5))
	fig.suptitle("Element Stresses", fontsize=18)
	ax3d = fig.add_subplot(111, projection='3d')
	ax3d.locator_params(nbins=4)
	for ax in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
		ax.set_major_formatter(lambda x, pos: "{:.0f} m".format(x))
	ax3d.grid(False)

	# Plot lines for each bar element
	for el in truss.elements:
		ax3d.plot(
			el.positions[:,0], el.positions[:,1], el.positions[:,2], 
			'-', color=stress_color(el.stress / unit, max_stress), lw=4
		)

	# Create a color bar for indicating stress values
	cbar = fig.colorbar(stress_vals, ax=ax3d, shrink=.5, pad=.1)
	cbar.ax.set_title(r"Stress, $\sigma$", pad=20, fontsize=16)
	cbar.set_ticks([0, .5, 1])
	cbar.set_ticklabels([
		"{:.0f} {:s}".format(sval, stress_units) for sval in [-max_stress, 0., max_stress]
	])
	cbar.ax.tick_params(labelsize=14,pad=5)

	# Configure remaining plot settings
	ax3d.set_xlim(-max_dim, max_dim)
	ax3d.set_ylim(-max_dim, max_dim)
	ax3d.set_zlim(0, max_dim)
	ax3d.tick_params(labelsize=14,pad=5)
	for ax in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
		ax.set_pane_color((.1, .1, .1, 1.0))
		ax.pane.set_edgecolor((.8, .8, .8, 1.0))
	fig.tight_layout()

	return fig, ax3d
