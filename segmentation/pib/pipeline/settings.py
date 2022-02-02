import os

BASE_DIR = "/homes/raphael/Projects/LowdosePET2/PiBVision/"
DATA_DIR = os.path.join(BASE_DIR, "data_anonymized_recon")
ETC_DIR = os.path.join(BASE_DIR, "etc")

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(BASE_DIR,
#                        "..",
#                        "data")
# DATA_DIR = os.path.join(BASE_DIR,
#                         "..",
#                         "test_data",
#                         "test_data_processing_mr_ctlike")

SIGMA = 1

REGION_NAMES = ("Prefrontal",
                "Orbito_frontal",
                "Parietal",
                "Temporal",
                "Cingulate",
                "Precuneus")

# !!! this does not exists
THRESHOLDED_REGIONS_DIR = os.path.join(ETC_DIR, "thresholded_regions")

# ETC_DIR = os.path.join(BASE_DIR, "..", "etc")
