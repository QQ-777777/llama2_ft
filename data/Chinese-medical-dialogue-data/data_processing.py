import json, os, random

def write_txt(file_path,datas):
    with open(file_path,"w",encoding="utf8") as f:
        for d in datas:
            f.write(json.dumps(d,ensure_ascii=False)+"\n")
        f.close()

data_dir_path = "data/Chinese-medical-dialogue-data"
save_path = os.path.join(data_dir_path,"merge_data.json")

changed_data=[]
for file in os.listdir(data_dir_path):
    
    basename = file.split("/")[-1]
    if(basename[-4:]==".csv"):
        with open(os.path.join(data_dir_path, file), encoding="GBK") as f:
            for i in range(0,5000):

                lin = f.readline()[0:-1].split(',')
                if i==0:
                    continue        
                if len(lin) == 4:
                    if len(lin[1]+','+lin[2])<200 and len(lin[3])<200:
                        changed_data.append({"text":"### Human: "+lin[2]+" ### Assistant: "+lin[3]})

# sample_num = 5000
# changed_data=random.sample(changed_data, sample_num)
                        
write_txt(save_path,changed_data)
