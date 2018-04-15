#This script gathers the 


import requests
import pandas
import json
import datetime

now_timestamp = datetime.datetime.utcnow()
#data = pandas.read_csv('D:\\Masters\\Data Analysis and Data Mining\\Assignment 2\\Data\\busstops.txt',delim_whitespace = True)

loc = ["Drumcondra_Library","Bull_Island","Ballyfermot_Civic_Centre","Ballymun_Library","Dublin_City_Council_Rowing_Club","Walkinstown_Library","Woodstock_Gardens","Navan_Road","Raheny_Library","Irishtown_Stadium","Chancery_Park","Blessington_St._Basin","Dolphins_Barn","Sean_Moore_Road","Mellows_Park"]

for i in range(0,int(len(loc))):
		url = 'http://dublincitynoise.sonitussystems.com/applications/api/dublinnoisedata.php?location='+str(i+1)
		res = requests.get(url)
		url2 = 'http://dublincitynoise.sonitussystems.com/applications/api/dublinnoisedata.php?returnLocationStrings=true&location='+str(i+1)
		res_loc = requests.get(url2)
		#print res_loc.content
		filename = 'C:\\temp\\original\\'+loc[i]+'_'+str(now_timestamp.date())+'.json'
#		print filename
		json_message = res.content
		f=open(filename,"w+")
		f.write(json_message)
		f.close()
		
		d = json.loads(json_message)
		df = pandas.DataFrame({"Location": res_loc.content, "Date": d["dates"], "Time": d["times"], "Sound_Level (DB[A])": d["aleq"]})

		filename1 = 'C:\\temp\\csv\\'+loc[i]+'_'+str(now_timestamp.date())+'.csv'
#		data.fileno[i] = data.fileno[i]+1
		df.to_csv(filename1, sep=',', index = False)
		