"""Configuration constants for the Streamlit app."""

# Visualization settings
DEFAULT_DOWNSAMPLING_RATIO = 0.2
MAX_POINTS_FOR_VISUALIZATION = 100000
DEFAULT_POINT_SIZE = 2

# Accumulation settings
ACCUMULATION_STEP = 3

ACCUMULATION_STRATEGIES = {
    'default': 'Simple concatenation (no registration)',
    'greedy_grid': 'Greedy Grid registration-based alignment'
}

# Excluded segment classes (non-dynamic objects)
EXCLUDED_SEGMENT_CLASSES = {-1, 0, 1, 2, 4, 5, 9, 10, 12, 15, 20, 21, 17, 13}

# Waymo semantic classes
WAYMO_CLASSES = [
    "Car", "Truck", "Bus", "Other Vehicle", "Motorcyclist", "Bicyclist",
    "Pedestrian", "Sign", "Traffic Light", "Pole", "Construction Cone",
    "Bicycle", "Motorcycle", "Building", "Vegetation", "Tree Trunk",
    "Curb", "Road", "Lane Marker", "Other Ground", "Walkable", "Sidewalk"
]

# View modes
VIEW_MODES = ["Original", "Patched", "Side-by-Side", "Toggle"]
