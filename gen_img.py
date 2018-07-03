from PIL import Image, ImageDraw
import math
import numpy as np
import random

def get_sample(count):
    img_size=224
    lane_w=20
    lane_num=4
    cur_lane=random.randint(0,lane_num-1)
    center=np.array([img_size/2, img_size/2])
    theta=random.uniform(-3.1415926/2, 3.1415926/2)
    bflip=random.randint(0,1)
    k=math.tan(theta)
    k_p=-1/k
    pp = np.array([1,k_p])
    norm_len=math.sqrt(1+k_p*k_p)
    offset_list=[]
    if bflip==0:
        for i in range(lane_num+1):
            offset_list.append(i*lane_w-lane_w/2-cur_lane*lane_w)
    else:
        for i in range(lane_num+1):
            offset_list.append(-i*lane_w+lane_w/2+cur_lane*lane_w)
    im = Image.new('RGB', (224, 224), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.point((center[0], center[0]), 'black')
    line_count=0
    for offset in offset_list:
        pp_offset=offset/norm_len*pp+center
        b=pp_offset[1]-k*pp_offset[0]
        x1=0
        y1=b
        x2=img_size
        y2=x2*k+b
        line_width=4
        if line_count==0:
            line_width=2
        line_count=line_count+1
        draw.line((x1, y1, x2, y2), fill=(0, 0, 0), width=line_width)
    #im.show()
    #im.save("re/chamo_"+str(cur_lane)+"_"+str(count)+".jpg")
    #print(cur_lane)
    im.save("re/chamo.jpg")
    return im, cur_lane
