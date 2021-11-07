import os

path = '/pfs'

files = os.listdir(path)

for f in files:
	print(f)

path = '/pfs/pre-process-stage'

files = os.listdir(path)

for f in files:
	print(f)