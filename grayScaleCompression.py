'''Insert documentation later.
Author: Duc Phu'''
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg
import math
import os
import pandas as pd
import openpyxl
from PIL import Image, ImageOps

def mean_square_error(original, compressed):
    o_size = np.size(original)
    r_size = np.size(compressed)

    if o_size == r_size:
        # do sth
        mse = ((original - compressed)**2).mean(axis=None)
        #print(mse)
        return mse
    else:
        print('Error occured in the mean square error')
        return False
    
def count_zeros(sigma):
    index_list = []
    for i in range(len(sigma)):
        if sigma[i] == 0:
            index_list.append(i)
    
    return index_list

def gray_scale(file_name):
    img = Image.open(file_name) # name of image
    imggray = ImageOps.grayscale(img) # convert to gray scale
    
    imgmat = np.asarray(imggray) # convert image to array
    size = imgmat.shape
    
    rank = linalg.matrix_rank(imgmat)
    #U, sigma, V = np.linalg.svd(imgmat)
    #string = str(rank) + str(len(sigma)) + str(len(zeros))
    #with open("C:\Code8\rank.txt","w") as f:
        #f.write(string + "\n")
    return imgmat, rank, size
     
def reconstruct_SVD(file_name, PATH, file_format):
    YOUR_DPI = 96
    imgmat, rank, size = gray_scale(file_name)
    HEIGHT = size[0]
    WIDTH = size[1]
    plt.figure(figsize=(WIDTH/YOUR_DPI, HEIGHT/YOUR_DPI), dpi=YOUR_DPI)
    
    plt.imshow(imgmat, cmap='gray')
    plt.axis('off')
    original_path = PATH + "\\original" + str(file_format)
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    plt.savefig(original_path, dpi = YOUR_DPI)
    original_size = os.path.getsize(original_path)
    
    U, sigma, V = np.linalg.svd(imgmat)
    
    index_list = count_zeros(sigma)
    
    ratio_list = []
    num_sv_list = []
    MSE_list = []
    noise_list = []
    rank_list = []
    app_compress_list = []
    
    # first column of U, first column of Sigma, first row of V
    for i in range(1,len(sigma)):
        rank_list.append(rank)
        reconstimg = (U[:,:i])@np.diag(sigma[:i])@(V[:i,:])
        
        mse = mean_square_error(imgmat, reconstimg)
        MSE_list.append(mse)
        
        noise = 10*math.log((255**2/mse),10)###
        noise_list.append(noise) ###
        
        ## Approximation compression list
        #print(size[0], size[1])
        rank_approximation = i*(size[0] +size[1]+ 1)
        app_compress = int(size[0]*size[1])/rank_approximation
        app_compress_list.append(app_compress)
        
        # make and store the image
        dir_name = PATH + "\#sv_" + str(i)
        file_name = "\#sv" + str(file_format)
        file_path = dir_name + file_name
        os.mkdir(dir_name)
        plt.imshow(reconstimg, cmap='gray')
        plt.axis('off')
        #plt.title(title) # title function of pyplot 
        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        plt.savefig(file_path, dpi = YOUR_DPI)
        plt.clf()
        
        # get the size of new image and calculate ratio of orignal/new size
        file_size = os.path.getsize(file_path)
        ratio = (original_size / file_size)
        print('Done creating new image #sv_',str(i))
        ratio_list.append(ratio)
        num_sv_list.append(i)
    
    plt.close()
    return num_sv_list, ratio_list, MSE_list, noise_list, rank_list, app_compress_list

def plot():
    PATH = str(input("Type the folder you store this program:"))
    file_name = str(input("Type the name of the image you want to compress:"))
    file_format = input("Format you want to save: ")
    num_sv_list, ratio_list, MSE_list, noise_list, rank_list, app_compress_list = reconstruct_SVD(file_name, PATH, file_format)
    
    plt.plot(num_sv_list, ratio_list)
    plt.title('Compression ratio plot')
    plt.xlabel("Number of used singular values")
    plt.ylabel("Compression ratio")
    plt.grid()
    plt.savefig(PATH + '\\ratio_plot.png')
    plt.clf()
    
    #plt.subplot(3,1,2) # 2 rows, 1 columns, 1st plot
    plt.yscale('log')
    plt.plot(num_sv_list, MSE_list)
    plt.title('Mean squarred error plot')
    plt.xlabel("Number of used singular values")
    plt.ylabel("Log scale MSE")
    plt.grid()
    plt.savefig(PATH + '\\mse_plot.png')
    plt.clf()
    
    #plt.subplot(3,1,3) # 2 rows, 1 columns, 1st plot
    plt.yscale('log')
    plt.plot(num_sv_list, noise_list)
    plt.title('Peak to signal noise plot')
    plt.xlabel("Number of used singular values")
    plt.ylabel("Log scale peak to noise signal")
    plt.grid()
    plt.savefig(PATH + '\\noise_plot.png')
    plt.clf()
    
    ## Combine
    plt.yscale('log')
    plt.plot(num_sv_list, MSE_list, label = "Mean squarred error")
    plt.plot(num_sv_list, app_compress_list, label = "Required storage ratio")
    plt.title('Combination plot')
    plt.xlabel("Number of used singular values")
    plt.ylabel("Log scale MSE & Storage ratio ")
    plt.legend()
    plt.grid()
    plt.savefig(PATH + '\\combine_plot.png')
    plt.clf()
    
    
    print("Done plot")
    
    # Export to Excel file   
    data_frame = pd.DataFrame(list(zip(num_sv_list, ratio_list, MSE_list, noise_list, rank_list, app_compress_list)))
    data_frame.columns = ['# used singular values','Compression ratio',  'Mean squared error', 'Peak to signal', 'Rank', "Approximation compress ratio"]
    excel_path = PATH + "\\output.xlsx"
    data_frame.to_excel(excel_path)

plot()
    