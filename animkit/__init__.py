"""animkit: small numpy + OpenCV + PIL + ffmpeg kit for explainer shots. See animkit/README.md."""
from . import style
from .arch import along, arrow, arrow_path, branches, elbow, measure, module_box, path, pulse_dot
from .cam3d import (OrbitCam, PinholeCam, ViewCam, axes3d, camera_axes, draw_camera, floor_grid, fundamental, look_at, marker3d,
                    point_cloud, ray_fan, relative_pose, skew)
from .chart import attn_blocks, bar_chart, row_softmax
from .draw import Canvas, rrect_pts
from .layout import flicker, grid_pyramid, grid_square, input_slot, link, paper_card, split, table
from .ease import bump, fade, lerp, lin, ss
from .formula import Formula
from .i18n import T
from .panel import ImagePanel, colorbar, colorize, tile_starts
from .plot2d import Plot2D, nice_ticks
from .popup import Popup, popup_check
from .readout import counter, matrix, number, vector
from .richtext import Digits, Rich, wrap
from .style import CYAN, GOLD, GREEN, GREY, ORANGE, PANEL, RED, VIOLET, WHITE
from .tensor import (TokenStrip, attention, block_diag_mask, block_labels, box3, cmap, fly_token, fold, grid_xy,
                     heatmap, kind_colors, log_px, move_tokens, reshape_check, reshape_ids, reshape_rows,
                     softmax_rows, tensor_block, tensor_morph, token, tokens)
from .timeline import Checker, CheckFailed, Shot, default_out, load_shot
