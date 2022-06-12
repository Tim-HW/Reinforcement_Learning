from PIL import Image
import numpy as np
import cv2
import os
import glob
from os.path import isfile, join

def LoadImages():

    # Find the image to load
    mypath = os.path.dirname(__file__) + "/furnitures/"
    # load it
    
    onlyfiles = [f for f in os.listdir(mypath) if isfile(join(mypath, f))]

    for name in range(len(onlyfiles)):

        mypath = os.path.dirname(__file__) + "/furnitures" + "/"+ str(onlyfiles[name])
        #print(mypath + "\n")
        
        onlyfiles[name] = cv2.imread(mypath)

        if onlyfiles[name] is None:

            print("Can't load image")

        else:

            # if the image is loaded, resize it for odd number scale
            if onlyfiles[name].shape[0] % 2 == 1 :
                onlyfiles[name] = cv2.resize(onlyfiles[name],(onlyfiles[name].shape[1],onlyfiles[name].shape[0]-1))
            if onlyfiles[name].shape[1] % 2 == 1 :
                onlyfiles[name] = cv2.resize(onlyfiles[name],(onlyfiles[name].shape[1]-1,onlyfiles[name].shape[0]))
    

    return onlyfiles


def cv2pil(img):
    # You may need to convert the color.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    
    return im_pil

def pil2cv(img):
    # For reversing the operation:
    im_np = np.asarray(img)
    return im_np

image = LoadImages()




#cv2.imshow("Python Logo", image[3])
#cv2.waitKey(0)
#cv2.destroyAllWindows()

for i in range(len(image)):

    img = cv2pil(image[i])
    #img.show()
    img = img.resize((1000,1000))
    
    image[i] = pil2cv(img)
    print(image[i].shape)





"""
#image = Image.open('dog.png')
new_image = image.resize((1000,1500))
#new_image.save('image_400.jpg')

print(image.size) # Output: (1920, 1280)
print(new_image.size) # Output: (400, 400)

image.show(title='before')
new_image.show(title='after')
"""