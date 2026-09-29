"""Shared constants for the DRISHTI nowcasting pipeline.

The grid mirrors INSAT-3D/3DR infrared resolution (~4 km). History and lead
times follow the problem statement: the model sees the last 3 hours of
satellite frames at 30-minute steps and predicts 2 to 6 hours ahead.
"""

GRID = 32                 # cells per side (32 x 4 km = 128 km domain)
DX_KM = 4.0               # grid spacing, km
STEP_H = 0.5              # satellite frame spacing, hours
HIST_STEPS = 7            # frames at t = -3.0 ... 0.0 h
HIST_TIMES_H = tuple(-3.0 + STEP_H * i for i in range(HIST_STEPS))
LEADS_H = (2, 3, 4, 6)    # nowcast lead times, hours after issue

HEADS = ("thunderstorm", "cloudburst", "flash_flood")

# Event definitions used to build labels
CLOUDBURST_MM_PER_H = 100.0      # IMD cloudburst criterion
THUNDER_RAIN_MM_PER_H = 5.0      # convective rain ...
THUNDER_CTT_K = 235.0            # ... under a cold (deep convective) cloud top
FLOOD_MIN_FLOWACC = 8            # only channel cells (>= 8 upstream cells) can flash-flood
FLOOD_ROUTED_MM = 900.0          # routed runoff depth x upstream cells that marks a flash flood
