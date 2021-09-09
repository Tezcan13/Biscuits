import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from cv2 import cv2

def check(singleIMG):

    #image = singleIMG #Image.open('tray.jpg')
    image=Image.fromarray(cv2.cvtColor(singleIMG, cv2.COLOR_BGR2RGB))

    #Resize the image to a 224x224.
    #Resizing the image to be at least 224x224 and then cropping from the center.
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #Turn the image into a numpy array
    image_array = np.asarray(image)

    #Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    #Load the image into the array
    data[0] = normalized_image_array

    #Result // prediction[0,0] represents our first model, prediction[0,1] represents second model and go on.
    prediction = model.predict(data)

    #Max Accuracy, Its for finding which model has max accuracy for this cropped biscuit.
    maxAccuracy=0
    itemName=""

    #Find the model that has max accuracy.
    for i in range(len(items)):
        itemAcc = prediction[0,i]
        if itemAcc>maxAccuracy:
            #Set itemName as a max accuracy model.
            itemName = items[i]    
            maxAccuracy = itemAcc
          
    #If the max accurancy has min 0.5 accuracy, then drive its rectangle.
    if maxAccuracy>=0.5:
        driveRectangle(itemName,maxAccuracy)

#Drive Rectangle function that drives a rectangle on biscuit and show it's name and accuracy on it. 
def driveRectangle(ID,Accuracy):

    global countOfBiscuits, countOfCocoa, countOfWhite, countOfRectangular, countOfCircular
    cv2.rectangle(result,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.putText(result, (ID+", "+ ("%.5f" % Accuracy)), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (209, 80, 0, 255), 1)

    #Set the variables for showing the biscuits numbers and types.
    countOfBiscuits +=1
    if "Cocoa" in ID:
        countOfCocoa += 1
    else:
        countOfWhite += 1
    if "Rectangular" in ID:
        countOfRectangular += 1
    else:
        countOfCircular += 1

#Show Text function that shows the biscuits numbers and types on top-left corner.  
def showText():
    text = "Number Of Biscuit: "+str(countOfBiscuits)+"\nCacao Biscuit: "+str(countOfCocoa)+" \nWhite Biscuit: "+str(countOfWhite)+" \nRectangular Biscuit: "+str(countOfRectangular)+" \nCircular Biscuit: "+str(countOfCircular)
    posx,posy=10,20
    for i, line in enumerate(text.split('\n')):
        pos = posx + (i+1)*posy
        cv2.putText(result, line, (10,pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

#My trained items name.
items = ["Circular", "Circular-Cocoa ","Rectangular","Rectangular-Cocoa "]

#My trays jpg list.
trays = ["trays/tray3.jpg","trays/tray2.jpg","trays/tray1.jpg"]

#Disable scientific notation for clarity
np.set_printoptions(suppress=True)

#Load the model
model = tf.keras.models.load_model('keras_model.h5',compile=False)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


#Return all trays.
for i in range(len(trays)):
    countOfBiscuits, countOfCocoa, countOfWhite, countOfRectangular, countOfCircular = 0,0,0,0,0

    #Take the tray image.
    imgReal=cv2.imread(trays[i])
    result = imgReal.copy()
    sizeX, sizeY, channels = imgReal.shape

    #Apply grayscale and treshold for finding all possible object on the tray.
    img = cv2.imread(trays[i],cv2.IMREAD_GRAYSCALE)
    ret,threshold = cv2.threshold(img,200,255,0)
    contours,hiearchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    #Then return this possible objects.
    for i in range(len(contours)):
        #Take this possible object areas position and scale values.
        x,y,w,h = cv2.boundingRect(contours[i])

        #Crop the possible object area from the tray.
        cropped = imgReal[y:y+h, x:x+w].copy()
    
        #If weight and height of area higher than 1 in 15 of total size of tray,
        #If this is not full tray,
        #If this area is not null(sometimes it comes empty), THEN go to check function for comparing trained models with this possible object area.
        if w>sizeY/15 and h>sizeY/15 and w<sizeX and h<sizeX and len(cropped)>0:
            check(cropped)

    #Show the result.
    showText()
    cv2.imshow("snc",result)
    cv2.waitKey(0)
 
cv2.destroyAllWindows()