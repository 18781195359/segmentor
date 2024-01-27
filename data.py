import zipfile

with zipfile.ZipFile('zzseg/segmenter-master.zip', 'r') as zip_ref:
    zip_ref.extractall('.')
