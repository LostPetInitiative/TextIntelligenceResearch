import json
import os

dataFiles = [x for x in os.listdir(".") if x.endswith(".json")]

result = []

counter =0

for dataFile in dataFiles:
    f = open(dataFile)
    ident = dataFile[5:-5]
    loaded = json.load(f)
    f.close()
    print(f"{ident} loaded")
    for i,(k,v) in enumerate(loaded.items()):
        result.append({"id":counter, "data": {"intraID": k,"group":ident,"text": v["text"]}})
        counter += 1

with open("label_studio_data.json", "w") as fp:
    json.dump(result , fp) 
    