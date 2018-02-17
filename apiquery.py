import requests
import pandas

data = pandas.read_csv('D:\\Masters\\Data Analysis and Data Mining\\Assignment 2\\Data\\busstops.txt',delim_whitespace = True)

for i in range(0,int(len(data.stopno))):
		url = 'http://data.smartdublin.ie/cgi-bin/rtpi/realtimebusinformation?stopid='+str(data.stopno[i])+'&format=xml'
		res = requests.get(url)
		filename = 'D:\\Masters\\Data Analysis and Data Mining\\Assignment 2\\Data\\'+str(data.stopno[i])+str(data.fileno[i])+'.xml'
		print filename
		f=open(filename,"w+")
		f.write(res.content)
		f.close()
		
		data.fileno[i] = data.fileno[i]+1

data.to_csv('D:\\Masters\\Data Analysis and Data Mining\\Assignment 2\\Data\\busstops.txt', sep='\t', index = False)
		