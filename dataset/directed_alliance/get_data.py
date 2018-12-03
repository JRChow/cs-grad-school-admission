#!python
"""
getdata.py

Fetch `Directed Alliance Data` from Google Drive.
"""
from __future__ import print_function
import os
import requests

URL = "https://docs.google.com/uc?export=download"
ID = "1qsc0LSBEPLnU0_-ZkeQc7ZoorN_8dJEY"
DATAFILE = "directed_alliance_data.csv"


if __name__ == '__main__':
	file_dir = os.path.dirname(os.path.realpath(__file__))
	if not os.path.exists(os.path.join(file_dir, DATAFILE)):
		print("dowloading file {} ...".format(DATAFILE))
		get_file(URL, ID, os.path.join(file_dir, DATAFILE))
		print("done, saved to {}".format(file_dir))
	else:
		print("file {} already existed".format(DATAFILE))

def get_file(url, id, destination):
	session = requests.Session()

	response = session.get(url, params = { 'id' : id }, stream = True)
	token = get_confirm_token(response)

	if token:
		params = { 'id' : id, 'confirm' : token }
		response = session.get(url, params = params, stream = True)

	save_response_content(response, destination)

def get_confirm_token(response):
	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			return value
	return None

def save_response_content(response, destination):
	CHUNK_SIZE = 32768

	if not os.path.exists(os.path.dirname(destination)):
                os.makedirs(os.path.dirname(destination))

	with open(destination, "wb") as f:
		for chunk in response.iter_content(CHUNK_SIZE):
			if chunk: # filter out keep-alive new chunks
				f.write(chunk)


