# Inference pipeline - DICOM to DICOM

## Create a virtual environment
```
git clone https://github.com/CAAI/brainPETNR.git
cd brainPETNR/inference_pipeline
python3 -m venv .env
source .env/bin/activate
python3 -m pip install -r requirements.txt
```

## Run pipeline on one or several patients
Before running the pipeline, there needs to a config file and model from a trained model (trained with rh-torch) place into the folder. Then
```
python pipeline.py -i /path/to/patients/folders
```
where each patient folder contains a CT and PET. If several PET are present, the PET name can be specified with ```--tag```

## Run a microservice that can receive PET and CT
Run a server instance in one terminal:
```
python microservice.py
```
Then make a POST request containing the PET and CT image path. See example in ```client_call.py```

## Run as DOCKER container
```
cd inference_pipeline
docker build -t denoiser:latest .
```
In order to copy data between the container and the host system, use docker compose to instantiate the microservice container and a volume mount mirroring a directory from the host system. The configuration is found in ```docker-compose.yml```. Run in detached mode:
```
docker-compose up -d 
```
To stop the process
```
docker-compose down
```