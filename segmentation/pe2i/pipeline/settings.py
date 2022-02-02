import os
import subprocess
from collections import namedtuple
HOME_DIR = "/homes/raphael/Projects/LowdosePET2/PE2I"
DATA_DIR = os.path.join(HOME_DIR, "data_anonymized_recon")
# ETC_DIR = os.path.join(HOME_DIR, "etc")
ETC_DIR = "/homes/raphael/Projects/LowdosePET2/PE2I/etc"

# ATLAS2 = "atlas2.nii.gz"
# ATLAS3 = "atlas3.nii.gz"
ATLAS4 = "atlas4.nii.gz"
ATLAS_CAUDVENT = "atlas_caudvent.nii.gz"
TEMPLATE = "avg_template.nii.gz"
TEMPLATES = "templates.nii.gz"
TEMPLATES_CAUDVENT = "templates_caudvent.nii.gz"
MASK = "mask.nii.gz"

version = "123"

_ref_region_collection = namedtuple(
    "Reference_regions",
    ("NAMES", "DEFINITIONS"))

REFERENCE_REGIONS = _ref_region_collection(
    ("Cerebellum GM",),
    ("CerebellumGM",)
)

_target_region_collection = namedtuple(
    "Target_regions",
    ("NAMES", "DEFINITIONS"))


TARGET_REGIONS = _target_region_collection(
    ("Putamen",
     "Caudatus",
     "Cerebellum_GM"),
    (("Putamen",),
     ("Caudatus",),
     ("CerebellumGM",)
     ))
