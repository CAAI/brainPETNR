import requests

# BASE = 'http://172.16.78.169:8080/'
BASE = 'http://172.18.0.2:8080/'

data = {
    'pet_folder':
    "~/Projects/pipeline_test/data/PiB_001_000/PET_LD300sec600delay",
    'ct_folder': "~/Projects/pipeline_test/data/PiB_001_000/CT"
}
response = requests.post(BASE + 'process/', data)
print(response.json())