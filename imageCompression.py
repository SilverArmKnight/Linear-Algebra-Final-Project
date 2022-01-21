'''Through Singular Value Decomposition, we create images with fewer singular
values while retaining most of the quality image, and reduce its size.
Author: Duy Anh, Duc Phu.'''
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import os
import pandas as pd

def imageCompression(imagePath, newImgName, yourDPI, SVBound):
    '''We use every function here. #TODO Add documentation later.'''
    colorArrayList = scaleImages(imagePath, yourDPI)
    PATH, SVNum_List, compRatio_List, MSE_List, noise_List, app_ratio_List = generateImages(
                                    colorArrayList, SVBound, newImgName, yourDPI)
    visualizeData(PATH, SVNum_List, compRatio_List, MSE_List, noise_List, app_ratio_List)

def scaleImages(imagePath, yourDPI):
    '''Given an image's path, we separate the image into red-scale, 
    green-scale and blue-scale equivalent. The function returns a list of RBG arrays.'''
    # Splitting the image into RGB arrays.
    img = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)
    blue, green, red = cv2.split(img)

    # Plotting the images. Check your computer's dpi before doing this.
    height = img.shape[0]
    width = img.shape[1]

    # Reshape the figure to match the original image's size.
    fig = plt.figure(figsize = (width / yourDPI, height/ yourDPI), dpi=yourDPI)

    # Scale values down between 0 and 1.
    df_blue = blue/255
    df_green = green/255
    df_red = red/255

    colorArrayList = [df_blue, df_green, df_red]

    return colorArrayList, height, width

def generateImages(colorArrayList, height, width, SVBound, newImgName, yourDPI):
    '''Given the original image's path, a list of RGB arrays of the image, the 
    maximum number of singular values to be used, your preferred image name and DPI, 
    we generate new images.
    The function returns Singular Value List, Compression Ratio List, 
    Mean-Squared Error List and Noise List.'''
    # Initiate RGB list to create new image with fewer singular values.
    bArr = 0
    rArr = 0
    gArr = 0

    newArrayList = [bArr, gArr, rArr]
    
    # Initiate lists of stuffs to return.
    SVNum_List = []
    compRatio_List = []
    MSE_List= [[],[],[],[]]     # MSE for blue, green, red-scale and full color.
    noise_List = [[],[],[],[]]  # Similar to the above.
    app_ratio_List = []

    PATH = "C:\\Users\\LA"  # So save new image processed through SVD.
    ORIGINAL_IMAGE = cv2.merge((colorArrayList[0], colorArrayList[1], colorArrayList[2]))

    for i in range(0, SVBound + 1, 5):
        app_ratio = (height*width)/(i*(height + width + 1))
        app_ratio_List.append(app_ratio)
        for j in range(3):
                U, sigma, V = np.linalg.svd(colorArrayList[j])
                newArrayList[j] = (U[:,:i])@np.diag(sigma[:i])@(V[:i,:])

        newImg = (cv2.merge((newArrayList[0], newArrayList[1], newArrayList[2])))
        plt.imshow(newImg) 

        file_name = newImgName + ' #sv_' + str(i) +'.jpg'
        plt.axis('off')

        # Remove white space around image.
        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        plt.savefig(PATH + "\\" + file_name, dpi = yourDPI)
        plt.clf()   # Clear MatPlotLib's console to avoid overlaying two images.

        SVNum_List.append(i)
        
        #MSE for 3 BGR channels
        for t in range(3):
            currMSE = MSE(colorArrayList[t], newArrayList[t])
            MSE_List[t].append(currMSE)
            noise = 10*math.log((255**2/currMSE),10)
            noise_List[t].append(noise)

        # A specific case for full color image.
        colorMSE = MSE(ORIGINAL_IMAGE, newImg)
        MSE_List[3].append(colorMSE)
        noise = 10*math.log((255**2/colorMSE),10)
        noise_List[3].append(noise)

        # Specify path to extract image's size later on.
        # Use raw string format.
        currFilePath = PATH + "\\" + file_name
        fileSize = os.path.getsize(currFilePath)
        originalSize = os.path.getsize(imagePath)
        ratio = (originalSize / fileSize)
        compRatio_List.append(ratio)
    
    plt.close()
    return PATH, SVNum_List, compRatio_List, MSE_List, noise_List, app_ratio_List

def MSE(original, newImg):
    '''Given the original image and the new one, we check how different they are.
    The function returns the difference, or MSE.'''
    # Check size of two input images.
    o_size = np.size(original)
    r_size = np.size(newImg)

    if o_size == r_size:
        # Calculate the Mean Square Error.
        mse = ((original - newImg)**2).mean(axis=None)
        return mse
    # If there are some errors.    
    else:
        print('Error occured in the mean square error')
        return False

def visualizeData(PATH, SVNum_List, compRatio_List, MSE_List, noise_List, app_ratio_List):
    '''Given the return values in generateImages(), we create an Excel file that
    contain the information for documentation purposes. We also plot each of these
    lists and save them as image files. The function does not return anything.'''

    # Export to Excel file   
    data_frame = pd.DataFrame(list(zip(SVNum_List, compRatio_List, MSE_List[0], 
                                        MSE_List[1], MSE_List[2], MSE_List[3], 
                                        noise_List[0], noise_List[1],noise_List[2], 
                                        noise_List[3])))
                                        
    data_frame.columns = ['Number of used singular values','Compression ratio', 
                        'MSE Blue', 'MSE Green','MSE Red', 'MSE Color', 'Noise Blue', 
                        'Noise Green','Noise Red', 'Noise Color']

    data_frame.to_excel('output.xlsx')
    plt.axis('on')

    # Plot the compression ratio.
    plt.plot(SVNum_List, compRatio_List)
    plt.title('Compression ratio plot')
    plt.xlabel("Number of used singular values")
    plt.ylabel("Compression ratio")
    plt.grid()
    plt.savefig(PATH + '\\ratio_plot.png')
    plt.clf()
    
    # Plot the APPROXIMATION compression ratio.
    plt.yscale('log')
    plt.plot(SVNum_List, app_ratio_List)
    plt.title("Required storage ratio plot")
    plt.xlabel("Number of used singular values")
    plt.ylabel("Storage ratio")
    plt.grid()
    plt.savefig(PATH + '\\app_plot.png')
    plt.clf()
    
    # Plot mean-squared error
    plt.yscale('log')
    plt.plot(SVNum_List, MSE_List[0], label ="Blue")
    plt.plot(SVNum_List, MSE_List[1], label ="Green")
    plt.plot(SVNum_List, MSE_List[2], label = "Red")
    plt.plot(SVNum_List, MSE_List[3], label = "Colored")
    plt.legend()
    plt.grid()
    plt.title('Mean squarred error plot')
    plt.xlabel("Number of used singular values")
    plt.ylabel("Log scale MSE")
    plt.savefig(PATH + '\\mse_plot.png')
    plt.clf()

    # Plot the APPROXIMATION compression ratio and MSE.
    plt.yscale('log')
    plt.plot(SVNum_List, app_ratio_List, label = "Storage ratio")
    plt.plot(SVNum_List, MSE_List[3], label = "Colored MSE")
    plt.title('Combination plot')
    plt.xlabel("Number of used singular values")
    plt.ylabel("Log scale MSE & Storage ratio ")
    plt.legend()
    plt.grid()
    plt.savefig(PATH + '\\combination_plot.png')
    plt.clf()
    
    # Plot list of singular values.
    plt.plot(SVNum_List, noise_List[0], label = "Blue")
    plt.plot(SVNum_List, noise_List[1], label = "Green")
    plt.plot(SVNum_List, noise_List[2], label = "Red")
    plt.plot(SVNum_List, noise_List[3], label = "Colored")
    plt.legend()
    plt.grid()
    plt.title('Peak to signal noise plot')
    plt.xlabel("Number of used singular values")
    plt.ylabel("Log scale peak to noise signal")
    plt.savefig(PATH + '\\noise_plot.png')
    plt.clf()
    print("Done plot!")

if __name__ == '__main__':
    '''If you are to run this file, these codes will be executed.'''
    imagePath = input("Insert file path: ")
    newImgName = str(input("What name to give to new images? "))
    yourDPI = int(input("What is your computer's DPI? "))
    SVBound = int(input("How many singular values do you want to use? "))
    imageCompression(imagePath, newImgName, yourDPI, SVBound)