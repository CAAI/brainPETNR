
from pydicom import dcmread

#from ...start.server import IP, PORT

IP = "goya.petnet.rh.dk"
PORT = 104
AET = "GOYA"


def upload(dcm_folder):
    return
    import SimpleITK as sitk
    from pydicom import dcmread
    from pynetdicom import AE
    from pynetdicom.sop_class import PositronEmissionTomographyImageStorage, SecondaryCaptureImageStorage
    from pipeline.process.pipelines.output.upload import IP, PORT, AET

    dcms = [dcmread(dcm_filename)

            for seriesID in (sitk
                             .ImageSeriesReader()
                             .GetGDCMSeriesIDs(dcm_folder))

            for dcm_filename in (sitk
                                 .ImageSeriesReader()
                                 .GetGDCMSeriesFileNames(dcm_folder,
                                                         seriesID=seriesID))
            ]

    try:
        # Initialise the Application Entity
        ae = AE(ae_title="VOLUMETRI")

        # Add a requested presentation context
        ae.add_requested_context(PositronEmissionTomographyImageStorage)
        ae.add_requested_context(SecondaryCaptureImageStorage)

        assoc = ae.associate(IP,
                             PORT,
                             ae_title=AET)

    except Exception as e:
        raise RuntimeError("Unable to associate with server: %s" % e)
    else:
        try:
            if assoc.is_established:
                # Use the C-STORE service to send the dataset
                # returns the response status as a pydicom Dataset

                for dcm in dcms:

                    status = assoc.send_c_store(dcm)

                    # Check the status of the storage request
                    if status and status.Status is 0x0000:
                        # If the storage request succeeded this will be 0x0000
                        print 'C-STORE request status: 0x{0:04x}'.format(status.Status)
                    else:
                        raise RuntimeError('Connection timed out, was '
                                           'aborted or received invalid'
                                           ' response')

                # Release the association
            else:
                raise RuntimeError('Association rejected, '
                                   'aborted or never connected')
        finally:
            assoc.release()
