#pip install opencv-python


# import the necessary packages
import pandas as pd
import time
import numpy as np
import argparse
import imutils
import cv2
import re
import os
from PIL import Image
import colorsys
import argparse


colorz_terms = {"blue":"ch00","green":"ch01","red":"ch02"}


def rgb22hsv(r,g,b):
    r,g,b= r/255.0, g/255.0, b/255.0
    h,s,v = colorsys.rgb_to_hsv(r,g,b)
    return round(h*180),round(s*255),round(v*255)
xiao = rgb22hsv(0,0,28)
da   = rgb22hsv(0,0,232)



def image_high_width(path,img_prefixname,file_extension):
    try:
        re.sub(r"\\",'/',path)
        #input_path=path+'/'+img_prefixname+".tif"
        input_path=path+'/'+img_prefixname+file_extension
        re.sub(r"\\",'/',input_path)
        os.chdir(os.path.dirname(input_path))
        img = Image.open(input_path)
        imgSize = img.size
        return (imgSize[0],imgSize[1])
    except Exception as  e:
        print('str(Exception):\t', str(Exception))
        print('str(e):\t\t', str(e))
        print('repr(e):\t', repr(e))
        print('e.message:\t', e.message)
        print('traceback.print_exc():', traceback.print_exc())
        print('traceback.format_exc():\n%s' % traceback.format_exc())
        print('########################################################' )


def import_image(path,img_prefixname,file_extension,flag):#读取图片
    try:
        re.sub(r"\\",'/',path)
        #input_path=path+'/'+img_prefixname+".tif"
        input_path=path+'/'+img_prefixname+file_extension
        #print(input_path)
        re.sub(r"\\",'/',input_path)
        os.chdir(os.path.dirname(input_path))
        return cv2.imread(input_path, flag)   
    except Exception as  e:
        print('str(Exception):\t', str(Exception))
        print('str(e):\t\t', str(e))
        print('repr(e):\t', repr(e))
        print('e.message:\t', e.message)
        print('traceback.print_exc():', traceback.print_exc())
        print('traceback.format_exc():\n%s' % traceback.format_exc())
        print('########################################################' )

        
        
def callback(shelf): 
    try:
        global img, loow,lower,cellcounted,image_contours,image_orig,series_area,dff_area,BB,GG,RR,KK,kkk,color
        # find the colors within the specified boundaries
        loow = cv2.getTrackbarPos('low', 'Track Bar')
        (BB, GG, RR) = cv2.split(image_orig.copy())
        gao= int(KK.max())
        upper = {"blue":np.array([gao,0,0]), "green":np.array([0,gao,0]), "red":np.array([0,0,gao])}[color]
        lower = {"blue":np.array([loow,0,0]),"green":np.array([0,loow,0]),"red":np.array([0,0,loow])}[color]
        image_mask = cv2.inRange(kkk.copy(), lower, upper)
        # apply the mask
        #image_to_process = image_orig.copy()
        image_res = cv2.bitwise_not(kkk.copy(), kkk.copy(), mask = image_mask)#cv2.bitwise_and(), cv2.bitwise_not(), cv2.bitwise_or(), cv2.bitwise_xor()分别执行按位与/或/非/异或运算。掩膜就是用来对图片进行全局或局部的遮挡

        # load the image, convert it to grayscale, and blur it slightly
        image_gray1 = cv2.cvtColor(image_res.copy(), cv2.COLOR_BGR2GRAY)
        image_gray = cv2.GaussianBlur(image_gray1.copy(), (5, 5), 0)

        # perform edge detection, then perform a dilation + erosion to close gaps in between object edges
        image_edged1 = cv2.Canny(image_gray.copy(), 50, 100)
        image_edged2 = cv2.dilate(image_edged1.copy(), None, iterations=1)
        image_edged = cv2.erode(image_edged2.copy(), None, iterations=1)

        # find contours in the edge map
        cnts0 = cv2.findContours(image_edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        cnts = cnts0[1] if imutils.is_cv2() else cnts0[0]
        #cnts = imutils.grab_contours(cnts)
        cellcounted = 0
        image_contours=kkk.copy()
        series_area = pd.Series()
        #dpi = 1282.61*1036.78/1440/1920#um^2/pix
        for c in cnts:
            # if the contour is not sufficiently large, ignore it
            if not   50 <= cv2.contourArea(c) <= 2000:
                continue
            hull = cv2.convexHull(c)
            #print( hull)
            cv2.drawContours(image_contours,[hull],0,(0,255,255),1)
            cellcounted += 1  
            series_area = series_area.append(pd.Series(cv2.contourArea(c),index = [cellcounted]))#计算面积
        print("{} Nuclei".format(cellcounted),loow)
        img = image_contours
        cv2.imshow('Track Bar', img)
    except Exception as  e:
        print('str(Exception):\t', str(Exception))
        print('str(e):\t\t', str(e))
        print('repr(e):\t', repr(e))
        print('e.message:\t', e.message)
        print('traceback.print_exc():', traceback.print_exc())
        print('traceback.format_exc():\n%s' % traceback.format_exc())
        print('########################################################' )

def cell_area_counter(path,img_prefixname,file_extension,colorz):
    try:
        global img, loow,lower,cellcounted,image_contours,image_orig,dff_area,BB,GG,RR,KK,kkk,color
        color = colorz
        output = img_prefixname+"_out.tif"
        counter = {}
        os.chdir(os.path.dirname(path))
        image_orig = import_image(path,img_prefixname,file_extension,-1)
        zeros = np.zeros(image_orig.shape[:2],dtype="uint8");#创建与image相同大小的零矩阵
        (BB, GG, RR) = cv2.split(image_orig) #分离出图片的B，R，G颜色通道
        KK ={"blue":BB,"green":GG,"red":RR}[color]  
        kkk = {"blue":cv2.merge([BB,zeros,zeros]),"green":cv2.merge([zeros,GG,zeros]),"red":cv2.merge([zeros,zeros,RR])}[color]
        height_orig, width_orig = image_orig.shape[:2]#获得图片的长宽
        # output image with contours
        image_contours = image_orig.copy()
        image_to_process1 = image_orig.copy()
        image_to_process = image_orig.copy()
        di = int(KK.mean())
        loow= di
        gao= int(KK.max())   
        upper = {"blue":np.array([gao,0,0]), "green":np.array([0,gao,0]), "red":np.array([0,0,gao])}[color]
        counter[color] = 0
        cellcounted = 0
        loow = cv2.getTrackbarPos('low', 'Track Bar')
        lower = {"blue":np.array([loow,0,0]),"green":np.array([0,loow,0]),"red":np.array([0,0,loow])}[color]
        cv2.namedWindow('Track Bar', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Track Bar", 800, 800)

        cv2.createTrackbar('low', 'Track Bar', 0, 255,callback)
        img = kkk.copy()
        while(1):
            cv2.imshow('Track Bar', img)
            k = cv2.waitKey(1)
            if k & 0xFF == ord('q'):
                break 
        cv2.destroyAllWindows()
        sum_area = pd.Series(series_area).sum()
        image_width,image_hight = image_high_width(path,img_prefixname,file_extension)
        area_per = sum_area/image_hight/image_width*10
        return pd.Series([img_prefixname,cellcounted, loow,gao,sum_area,area_per,image_width,image_hight],index=["imageName","counts","low","high","area_sum",'area_percentage','image_width','image_hight']) , pd.Series(series_area)
    
    except Exception as  e:
        print('str(Exception):\t', str(Exception))
        print('str(e):\t\t', str(e))
        print('repr(e):\t', repr(e))
        print('e.message:\t', e.message)
        print('traceback.print_exc():', traceback.print_exc())
        print('traceback.format_exc():\n%s' % traceback.format_exc())
        print('########################################################' )

def count_cell_area_inFolder(import_foldname,ouput_foldname,colorz):
    try:
        path=import_foldname

        os.chdir(os.path.dirname(path))
        file_list= os.listdir(path)
        print(file_list)
        filename_list = []
        df = pd.DataFrame({})
        dff_area = pd.DataFrame({})
        print(colorz_terms[colorz])
        for file in file_list:
            file_input_path=path+'/'+file
            img_prefixname, file_extension=os.path.splitext(file)
            if file_extension == '.tif':
                if img_prefixname.split("_")[-1] == colorz_terms[colorz]:
                    print(img_prefixname)
                    #print(img_prefixname,colorz)
                    filename_list.append(img_prefixname)
                    kk=cell_area_counter(path,img_prefixname,file_extension,colorz)
                    df = df.append(kk[0],ignore_index=True)
                    dff_area = dff_area.append(kk[1],ignore_index=True)
                else:
                    continue
                print(img_prefixname)

        print(filename_list)
        dff=df.set_index("imageName")                                    
        dff_area['imageName']=filename_list
        dff_area = dff_area.set_index('imageName')
        print("ok")
        save_path = import_foldname+"/"+ouput_foldname
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path1 = import_foldname+"/"+ouput_foldname+ "/"+colorz+"-counts.csv"
        dff.to_csv(os.path.basename(save_path1))
        
        save_path2 = import_foldname+"/"+ouput_foldname+ "/"+colorz+"-area.csv"
        dff_area.to_csv(os.path.basename(save_path2))
        
        print("saved")
        
    
        
    except Exception as  e:
        print('str(Exception):\t', str(Exception))
        print('str(e):\t\t', str(e))
        print('repr(e):\t', repr(e))
        print('e.message:\t', e.message)
        print('traceback.print_exc():', traceback.print_exc())
        print('traceback.format_exc():\n%s' % traceback.format_exc())
        print('########################################################')
        
parse = argparse.ArgumentParser()
parse.add_argument("import_foldname", type=str)
parse.add_argument("ouput_foldname", type=str)
parse.add_argument("color", type=str)
parse.add_argument("DPI", type=float,help = '1282.61*1036.78/1440/1920 #um^2/pix' )  
args = parse.parse_args()
dpi = args.DPI
import_foldname =args.import_foldname
ouput_foldname = args.ouput_foldname
colorz = args.color

count_cell_area_inFolder(import_foldname,ouput_foldname,colorz)

