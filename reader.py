import pandas as pd

class Reader:
	def readFile(self, filename, titleLine = True):
		ufo = pd.read_csv(filename)
		ufo.head(1)
		'''
		data = []
		with open(filename, newline = "") as f:
			reader = csv.reader(f)
			for row in reader:
				data.append(row)
		if titleLine:
			data = data[1::]
		self.data = data
		'''

'''
data = []
is_num = []

def readFile(filename, titleLine = True):
	with open(filename, newline = "") as f:
		reader = csv.reader(f)
		for row in reader:
			data.append(row)
	if titleLine:
		data = data[1::]
	is_num = [True] * (len(data[0]) - 1)
	for row in data:
		for i in range(len(row) - 1):
			if not is_num[i]: continue
			try:
				row[i] = int(row[i])
				break
			except ValueError:
				print("Feature %d is non-numeric", i)
				is_num[i] = False
'''
