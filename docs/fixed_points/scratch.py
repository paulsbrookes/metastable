import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from metastable.map.map import FixedPointMap, FixedPointType

# Load the map with stability analysis
map = FixedPointMap.load("/home/paul/Projects/misc/keldysh/metastable/docs/fixed_points/examples/map-with-stability.npz")

# Filter out imaginary parts where real parts are nan
real_parts = map.eigenvalues.real
imag_parts = map.eigenvalues.imag
imag_parts[np.isnan(real_parts)] = np.nan
map.eigenvalues = real_parts + 1j * imag_parts
map.save("map-with-stability-no-imag.npz")