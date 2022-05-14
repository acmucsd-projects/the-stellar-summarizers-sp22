import os
import shutil
import requests

'''Downloads the SumMe dataset and unzip if it's not in the directory'''

if not os.path.exists('SumMe'):
    if not os.path.exists('SumMe.zip'):
        print('dataset not found, downloading')
        url = 'https://data.vision.ee.ethz.ch/cvl/SumMe/SumMe.zip'
        r = requests.get(url)
        open('SumMe.zip', 'wb').write(r.content)
    print("Unzipping SumMe.zip")
    shutil.unpack_archive('SumMe.zip', 'SumMe')
else: 
    print('dataset already downloaded and unzipped')
