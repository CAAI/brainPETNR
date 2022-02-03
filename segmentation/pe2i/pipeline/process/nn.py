import requests
import threading
import json
import os


def nn_predict(ct_file, pet_file):
    from subprocess import call
    from shutil import copy2
    import os

    # extract data_id from file names
    data_id = pet_file.split('/data/PETCTPE2I')[0].split('/')[-1]

    # instead of uploading files, copy to nn_webservice folder
    dst_dir = "/homes/raphael/Projects/LowdosePET/PE2I/nn_webservice/data/%s" % data_id
    dst_data_dir = os.path.join(dst_dir, "data")
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
        os.mkdir(dst_data_dir)

    if not os.path.exists(os.path.join(dst_data_dir, "ct.nii.gz")):
        print "copying files to nn_webservice dir"
        copy2(ct_file, os.path.join(dst_data_dir, "ct.nii.gz"))
        copy2(pet_file, os.path.join(dst_data_dir, "pet.nii.gz"))

    cmd = [
        '/homes/raphael/Projects/LowdosePET/PE2I/nn_webservice/.env/bin/python3',
        '/homes/raphael/Projects/LowdosePET/PE2I/nn_webservice/pe2i_put_caud_nn_webservice/processing/process.py',
        data_id
    ]

    pred_file = os.path.join(dst_dir, "prediction", "validation_case_0",
                             "prediction_masked.nii.gz")
    if not os.path.exists(pred_file):
        call(cmd)

    # download the files from nn_webserive back to data folder
    data_root_dir = pet_file.split('/pet_to_avg')[0]
    nn_dir = os.path.join(data_root_dir, 'nn_webservice')

    # prediction file path - to return
    filename = os.path.join(nn_dir, "prediction.nii.gz")
    if not os.path.exists(filename):
        print "copying prediction file back"
        copy2(pred_file, filename)

    return filename


if __name__ == "__main__":
    ct_file = "../PETCTSEGPE2I/data/19040216041554214489876739844937823026559/PETCTPE2I/ct_to_avg/struct_thresh_smooth_brain_flirt.nii.gz"
    pet_file = "../PETCTSEGPE2I/data/19040216041554214489876739844937823026559/PETCTPE2I/pet_to_avg/func_flirt.nii.gz"

    nn_predict(ct_file, pet_file)
