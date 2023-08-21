import os
import csv

cardpath = 'F:\idcard\card'
cardlist = os.listdir(cardpath)

maskcard_path = os.path.join(cardpath, cardlist[0])
front_path = os.path.join(cardpath, cardlist[1])
lhalfmask_parh = os.path.join(cardpath, cardlist[2])
normal_path = os.path.join(cardpath, cardlist[3])
shalfmask_parh = os.path.join(cardpath, cardlist[4])


mask_src = os.listdir(maskcard_path)
front_src = os.listdir(front_path)
lhalf_src = os.listdir(lhalfmask_parh)
normal_src = os.listdir(normal_path)
shalf_src = os.listdir(shalfmask_parh)


with open('D:\card.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(shalf_src)):
        writer.writerow([os.path.join (shalfmask_parh, shalf_src[i]), 1])