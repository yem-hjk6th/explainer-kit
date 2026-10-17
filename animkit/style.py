"""Canvas defaults and palette. Every color in the animkit API is an RGB 0-255 tuple."""
W, H, FPS = 1080, 1440, 30

# palette carried over from epipolar_demo.py (there it was BGR)
GOLD = (255, 200, 80)       # rays, 3D points, the thing being asked about
CYAN = (90, 220, 230)       # results / projections / patch tokens
ORANGE = (255, 130, 60)     # geometry that ties things together (baseline, epipole, highlights)
WHITE = (235, 235, 235)
GREY = (150, 150, 150)
PANEL = (60, 80, 90)        # translucent image-plane / panel fill
VIOLET = (175, 140, 255)
GREEN = (120, 230, 150)
RED = (255, 95, 95)

# text colors (PIL RGB)
TEXT = (240, 240, 240)
TEXT_DIM = (200, 200, 200)
MATH = (255, 215, 140)      # formula default
LABEL_GOLD = (255, 200, 120)
LABEL_CYAN = (150, 240, 255)
LABEL_ORANGE = (255, 190, 140)

# token kinds (VGGT-style frame tokens); frame 1 has its own learnable camera/register tokens -> camera1/register1.
# pass your own dict when an episode needs more kinds
TOKEN = {"camera": GOLD, "register": VIOLET, "patch": CYAN, "scale": GREEN, "camera1": ORANGE,
         "register1": (235, 120, 200)}

AXIS = (RED, GREEN, (110, 160, 255))   # x, y, z

# layout
TITLE_XY, TITLE_SIZE = (70, 90), 86
SUB_XY, SUB_SIZE = (74, 196), 40
CAPTION_Y, CAPTION_SIZE, CAPTION_SPACING = 1265, 42, 14
CAPTION_MAX_LINES, CAPTION_MAX_WORDS = 2, 12  # reading-load budget per caption (text.caption_len)
TAG_XY, TAG_SIZE = (1040, 44), 26

# oblique direction (screen px per px of depth) used by tensor blocks: up and to the right
DEPTH_DIR = (0.62, -0.46)

# heatmap colormap stops (value, RGB): dark -> blue -> cyan -> white, monotone in brightness
CMAP_STOPS = [(0.0, (16, 20, 30)), (0.35, (30, 70, 140)), (0.7, (90, 220, 230)), (1.0, (250, 250, 250))]
# depth maps: near = bright yellow, far = dark purple (magma-like); value 0 = vmin = nearest
CMAP_DEPTH = [(0.0, (252, 236, 160)), (0.25, (252, 160, 100)), (0.5, (222, 80, 100)), (0.75, (140, 42, 128)),
              (1.0, (40, 20, 76))]
