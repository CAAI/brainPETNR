#!/usr/bin/env python3
import shutil
import yaml
from pathlib import Path
from flask import Flask, request
from flask_restful import Api, Resource
from pipeline import JobProcess


def process_data(data_id, pet_folder, ct_folder):
    origin_folder = pet_folder.parent
    # data_dir = Path(
    #     "/var/lib/docker/volumes/inference_pipeline_temp_data/_data")
    data_dir = Path("data")
    config_file = Path("config_PiB_5pct.yaml")
    with open(config_file) as cf:
        configs = yaml.safe_load(cf)

    # step 1 - copy files to processing folder: input/data_id/
    #                                                   |__ pet_folder
    #                                                   |__ ct_folder
    print(f'Copying file to the processing folder /app/data/{data_id}')
    pet_folder_copy = data_dir.joinpath(data_id, pet_folder.name)
    ct_folder_copy = data_dir.joinpath(data_id, ct_folder.name)
    shutil.copytree(pet_folder, pet_folder_copy)
    shutil.copytree(ct_folder, ct_folder_copy)

    # step 2 - process data
    # try:
    print('Done copying. Setting up the process pipeline.')
    job = JobProcess(data_id, data_dir, configs, pet_folder.name,
                     ct_folder.name)
    job.wf.run()
    # except Exception as e:
    #     print("Error processing:", e)
    output_folder = data_dir.joinpath(data_id, pet_folder.name + '_DENOISED',
                                      'denoised_dicom')
    # step 3 - copy data back to origin folder from output/data_id/
    #                                                        |__ pet_folder
    #                                                        |__ ct_folder
    returned_pet_folder = origin_folder.joinpath(pet_folder.name + '_DENOISED')
    print(f'Done processing PET file. Copying back to {returned_pet_folder}')
    if output_folder.exists():
        shutil.move(output_folder, returned_pet_folder)

    # step 4 remove temporary files
    shutil.rmtree(pet_folder_copy.parent, ignore_errors=True)
    print('Cleaning up processing folder.')
    return returned_pet_folder


def main():
    """ Simple REST API to receive PET and CT data for denoising.
        The data is copied locally, processed and then copied 
        back to original location.
    """
    app = Flask(__name__)
    api = Api(app)

    class ProcessData(Resource):
        """ Single API resource which handles POST request.
            The POST method allows to send serialized data (dictionnary)
            instead of having to send info as part of url for GET method.
            
            Returns:
                path to de-noised data (same folder as original PET)
        """

        def post(self):
            """ POST request sent in json format. Example code 
                import requests
                BASE = 'http://0.0.0.0:8080/'
                data = {
                    'pet_folder': "/path/to/data/patient_id001/PET_lowdose",
                    'ct_folder': "/path/to/data/patient_id001/CT"
                }
                response = requests.post(BASE + 'process/', data)
                print(response.json())
            """
            pet_folder = Path(request.form['pet_folder'])
            ct_folder = Path(request.form['ct_folder'])
            data_id = pet_folder.parent.name
            print('Processing: ', data_id, pet_folder, ct_folder)
            output_data = process_data(data_id, pet_folder, ct_folder)

            # return path to denoised PET and status code
            return {'data': str(output_data), 'status': 200}

    api.add_resource(ProcessData, "/process/")

    app.run(host='0.0.0.0', port='8080', debug=True)


if __name__ == '__main__':
    main()
