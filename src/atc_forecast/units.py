"""Physical constants and normalisation scales.

Every feature is normalised by a FIXED PHYSICAL CONSTANT rather than a dataset statistic.
A z-score on the sample mean moves when the data moves, so adding days would silently
invalidate previously trained models, and it does not transfer to another airspace whose
statistics differ. The divisors below are properties of civil aviation, so they are stable
across datasets, regions and years.
"""

M_TO_FT = 3.28084
MS_TO_KT = 1.94384
MS_TO_FPM = 196.850
KT_TO_MS = 0.514444
M_TO_NM = 1.0 / 1852.0
FT_TO_NM = 1.0 / 6076.12
DEG_TO_M = 111320.0

FL180_M = 5486.4

POS_SCALE_M = 10_000.0
ALT_SCALE_FT = 45_000.0
GS_SCALE_KT = 600.0
SPD_SCALE_KT = 600.0
VR_SCALE_FPM = 6_000.0

HEAD_BIN_DEG = 5.0
N_HEAD_BINS = 72
ALT_BIN_FT = 1_000.0
N_ALT_BINS = 46
SPD_BIN_KT = 10.0
N_SPD_BINS = 61

WAKE_VOCAB = ["L", "M", "H", "S", "unknown"]
ENGINE_VOCAB = ["piston", "turboprop", "jet", "electric", "unknown"]
