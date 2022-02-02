"""
Job containing class used to define a processing job.
"""

import os
from pydicom import read_file as read_dicom
from pipeline import settings
import re
from pipeline.db.db import DBConn


class Job:

    """
    Class encapsulating basic information related to a given processing job. This includes
    data ID.
    Upon initialization, given folder containing DICOM files (in subfolders), integrity of data is checked.


    Args:

       datafolder (str) : Name of folder subfolders "PET" and "CT" containing PET resp. CT DICOM files located in data directory associated with module.

    Attributes:

       subject (str): Patient ID.

       CT_dicoms (list) : List of paths to CT DICOM files.

       PET_dicoms (list) : List of paths to CT DICOM files.

       base_dir (str) : Path to folder that will contain data produced during processing and given PET and CT DICOMs.

       data_id (str) : Data ID.

       PETAccessionNumber (str): Accession number of PET image.

       PETSeriesDescription (str) : Series description of PET image.

       CTAccessionNumber (str) : Accession number of CT image.

       CTSeriesDescription (str) : Series description of CT image.

    """

    def __init__(self, datafolder):
        print "Creating job"
        print datafolder
        print os.path.join(settings.DATA_DIR,
                           datafolder)
        assert os.path.isdir(os.path.join(settings.DATA_DIR,
                                          datafolder)), ("Given data" +
                                                         " folder does" +
                                                         " not exist")

        content = os.listdir(os.path.join(settings.DATA_DIR,
                                          datafolder))
        assert len(content) >= 2, "Too few folders in given data folder"

        assert any([d == "CT" for d in content]) and any(
            [d == "PET" for d in content]), "No CT or PET directory"

        self._base_dir = os.path.join(settings.DATA_DIR,
                                      datafolder)

        self._data_id = datafolder

        def get_dicoms(dir):
            return [os.path.join(settings.DATA_DIR,
                                 datafolder, dir, file)
                    for file in os.listdir(
                os.path.join(
                    settings.DATA_DIR,
                    datafolder,
                    dir))]

        self._CT_dicoms = get_dicoms("CT")

        self._PET_dicoms = get_dicoms("PET")

        assert self._PET_dicoms, "No PET files in PET folder"
        assert self._CT_dicoms, "No CT files in CT folder"

        loaded_CT_dicoms = [read_dicom(fn)
                            for fn in self._CT_dicoms]
        loaded_PET_dicoms = [read_dicom(fn)
                             for fn in self._PET_dicoms]

        assert len(set([(dcm.PatientID,
                         dcm.AccessionNumber,
                         dcm.SeriesDescription)
                        for dcm in loaded_PET_dicoms])
                   ) == 1, ("More than one patient ID," +
                            " accession number or series" +
                            " description for" +
                            " the given PET images")

        assert len(set([(dcm.PatientID,
                         dcm.AccessionNumber,
                         dcm.SeriesDescription)
                        for dcm in loaded_CT_dicoms])
                   ) == 1, ("More than one patient ID," +
                            " accession number or series" +
                            " description for the given CT images")

        assert len(set([dcm.SliceThickness
                        for dcm in loaded_PET_dicoms])
                   ) == 1, ("SliceThickness not identical " +
                            "across PET slices.")

        assert len(set([dcm.SliceThickness
                        for dcm in loaded_CT_dicoms])
                   ) == 1, ("SliceThickness not identical " +
                            "across CT slices.")

        assert read_dicom(self._PET_dicoms[0]).PatientID == read_dicom(
            self._CT_dicoms[0]).PatientID, ("Patient ID for PET" +
                                            " and CTs are different")

        self._subject = read_dicom(self._PET_dicoms[0]).PatientID

        self._PETAccessionNumber = read_dicom(
            self._PET_dicoms[0]).AccessionNumber
        self._PETSeriesDescription = read_dicom(
            self._PET_dicoms[0]).SeriesDescription
        self._PETSeriesNumber = read_dicom(
            self._PET_dicoms[0]).SeriesNumber

        self._CTAccessionNumber = read_dicom(
            self._CT_dicoms[0]).AccessionNumber
        self._CTSeriesDescription = read_dicom(
            self._CT_dicoms[0]).SeriesDescription

        print self.PETSeriesNumber

        # try:
        #     with DBConn() as conn:
        #         q_data = conn.data.find({"PatientID": self.subject,
        #                                  "PETSD": self.PETSeriesDescription,
        #                                  "PETACC": self.PETAccessionNumber,
        #                                  "CTSD": self.CTSeriesDescription,
        #                                  "CTACC": self.CTAccessionNumber,
        #                                  "pet_series_number": self.PETSeriesNumber
        #                                  }, {"data_id": 1})
        # except Exception as e:
        #     raise RuntimeError("Error in database: %s" % e)
        # else:
        #     assert q_data.count() < 2, ("Database inconsistency, more than " +
        #                                 "one document with given " +
        #                                 "identifiers.")
        #     if q_data.count() == 1:
        #         try:
        #             print q_data[0]["data_id"]
        #             with DBConn() as conn:
        #                 q_job = conn.jobs.find_one(
        #                     {"data_id": q_data[0]["data_id"]},
        #                     {"status": 1})["status"]
        #         except Exception as e:
        #             raise RuntimeError("Inconsistency in database %s" % e)
        #         else:
        #             if q_job == "DONE":
        #                 raise IOError("Data already processed.")
        #             elif q_job == "PROCESSING":
        #                 raise IOError(
        #                     "Data already being processed.")
        #             elif q_job == "ADDED":
        #                 raise RuntimeError(
        #                     "Inconsistency in database: Data still has status 'Added'.")
        #             with DBConn() as conn:
        #                 conn.data.update({"PatientID":
        #                                   self.subject,
        #                                   "PETSD":
        #                                   self.PETSeriesDescription,
        #                                   "PETACC":
        #                                   self.PETAccessionNumber,
        #                                   "CTSD":
        #                                   self.CTSeriesDescription,
        #                                   "CTACC":
        #                                   self.PETAccessionNumber,
        #                                   "pet_series_number":
        #                                   self.PETSeriesNumber
        #                                   },
        #                                  {"$set": {"data_id":
        #                                            self.data_id}})
        #     else:
        #         print "inserting"
        #         with DBConn() as conn:
        #             conn.data.insert_one({"PatientID":
        #                                   self.subject,
        #                                   "PETSD":
        #                                   self.PETSeriesDescription,
        #                                   "PETACC":
        #                                   self.PETAccessionNumber,
        #                                   "CTSD":
        #                                   self.CTSeriesDescription,
        #                                   "CTACC":
        #                                   self.CTAccessionNumber,
        #                                   "pet_series_number":
        #                                   self.PETSeriesNumber,
        #                                   "data_id":
        #                                   self.data_id})

        print self._subject

    @property
    def subject(self):
        return self._subject

    @property
    def CT_dicoms(self):
        return self._CT_dicoms

    @property
    def PET_dicoms(self):
        return self._PET_dicoms

    @property
    def base_dir(self):
        return self._base_dir

    @property
    def data_id(self):
        return self._data_id

    @property
    def PETAccessionNumber(self):
        return self._PETAccessionNumber

    @property
    def PETSeriesDescription(self):
        return self._PETSeriesDescription

    @property
    def PETSeriesNumber(self):
        return self._PETSeriesNumber

    @property
    def CTAccessionNumber(self):
        return self._CTAccessionNumber

    @property
    def CTSeriesDescription(self):
        return self._CTSeriesDescription
