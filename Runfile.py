import tkinter as tk
from tkinter import Message ,Text
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
#import datetime
import time
from datetime import datetime
from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.106-0.65.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]

##import font
##import Tkinter.ttk as ttk

#from tkfontchooser import askfont
##import Tkinter.font as font

window = tk.Tk()
#helv36 = tk.Font(family='Helvetica', size=36, weight='bold')
window.geometry("1500x1000")
window.title("Face_Recogniser")

dialog_title = 'QUIT'
dialog_text = 'Are you sure?'
#answer = messagebox.askquestion(dialog_title, dialog_text)
 
#window.geometry('1280x720')
window.configure(background='white')

#window.attributes('-fullscreen', True)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

message = tk.Label(window, text="Face Recognition Based Attendance System With Emotion Detection" ,bg="green"  ,fg="white"  ,width=50  ,height=3,font=('times', 30, 'bold')) 

message.place(x=200, y=20)

lbl = tk.Label(window, text="Enter ID",width=15  ,height=2  ,fg="green"  ,bg="snow" ,font=('times', 20, ' bold ') ) 
lbl.place(x=400, y=200)

txt = tk.Entry(window,width=15  ,bg="whitesmoke" ,fg="black",font=('times', 18, ' bold '))
txt.place(x=700, y=215)

lbl2 = tk.Label(window, text="Enter Name",width=15  ,fg="green"  ,bg="snow"    ,height=2 ,font=('times', 20, ' bold ')) 
lbl2.place(x=400, y=300)

txt2 = tk.Entry(window,width=20  ,bg="whitesmoke"  ,fg="black",font=('times', 18, ' bold ')  )
txt2.place(x=700, y=315)

lbl3 = tk.Label(window, text="Notification : ",width=15  ,fg="green"  ,bg="snow"  ,height=2 ,font=('times', 20, ' bold underline ')) 
lbl3.place(x=400, y=400)

message = tk.Label(window, text="" ,bg="whitesmoke"  ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
message.place(x=700, y=400)

lbl3 = tk.Label(window, text="Attendance : ",width=15  ,fg="red"  ,bg="snow"  ,height=2 ,font=('times', 20, ' bold  underline')) 
lbl3.place(x=400, y=650)


message2 = tk.Label(window, text="" ,fg="red"   ,bg="whitesmoke",activeforeground = "green",width=40  ,height=12  ,font=('times', 15, ' bold ')) 
message2.place(x=700, y=650)
 
def clear():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)    
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
 
def TakeImages():
    co=['Id']
    df=pd.read_csv("StudentDetails\StudentDetails.csv",names=co)
    
    namess = df['Id']
    ides=[]

    #print'Id:'
    #print namess
    
    Id=(txt.get())
    
    ides=Id
    #print 'Id='
    #print ides
    name=(txt2.get())
    estest=0
    if ides in namess:
        estest=1
    else:
        estest=0
    #print estest
    if (estest==0):
        if(is_number(Id) and name.isalpha()):
            cam = cv2.VideoCapture(0)
            harcascadePath = "haarcascade_frontalface_default.xml"
            detector=cv2.CascadeClassifier(harcascadePath)
            sampleNum=0
            while(True):
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                    #incrementing sample number 
                    sampleNum=sampleNum+1
                    #saving the captured face in the dataset folder TrainingImage
                    cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                    #display the frame
                    cv2.imshow('frame',img)
                #wait for 100 miliseconds 
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                # break if the sample number is morethan 100
                elif sampleNum>200:
                    break
            cam.release()
            cv2.destroyAllWindows() 
            res = "Images Saved for ID : " + Id +" Name : "+ name
            row = [Id , name]
            with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()
            message.configure(text= res)
        else:
            if(is_number(Id)):
                res = "Enter Alphabetical Name"
                message.configure(text= res)
            if(name.isalpha()):
                res = "Enter Numeric Id"
                message.configure(text= res)
        
    else:
        res = "Already Id Exist"
        message.configure(text= res)

def TakeImages1():
    co=['Id']
    df=pd.read_csv("StudentDetails\StudentDetails.csv",names=co)
    
    namess = df['Id']
    ides=[]

    #print'Id:'
    #print namess
    
    Id=(txt.get())
    
    ides=Id
   # print 'Id='
   # print ides
    name=(txt2.get())
    estest=0
    if ides in namess:
        estest=1
    else:
        estest=0
    #print estest
    if (estest==1):
        if(is_number(Id) and name.isalpha()):
            cam = cv2.VideoCapture(0)
            harcascadePath = "haarcascade_frontalface_default.xml"
            detector=cv2.CascadeClassifier(harcascadePath)
            sampleNum=0
            while(True):
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                    #incrementing sample number 
                    sampleNum=sampleNum+1
                    #saving the captured face in the dataset folder TrainingImage
                    cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                    #display the frame
                    cv2.imshow('frame',img)
                #wait for 100 miliseconds 
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                # break if the sample number is morethan 100
                elif sampleNum>200:
                    break
            cam.release()
            cv2.destroyAllWindows() 
            res = "Images Saved for ID : " + Id +" Name : "+ name
            message.configure(text= res)
        else:
            if(is_number(Id)):
                res = "Enter Alphabetical Name"
                message.configure(text= res)
            if(name.isalpha()):
                res = "Enter Numeric Id"
                message.configure(text= res)
        
    else:
        res = "Id is Not Trained Before"
        message.configure(text= res)
        
    
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()#recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained"#+",".join(str(f) for f in Id)
    message.configure(text= res)

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #print(imagePaths)
    
    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids

def calculate():
    import pandas as pd
    import glob
    dfstud = pd.read_csv('StudentDetails\StudentDetails.csv', index_col=None, header=0)
    
    studname=dfstud['Name'].values
    #print 'Student Name'
    #print studname

    path = r'Attendance' # use your path
    all_files = glob.glob(path + "/*.csv")
    ts = time.time() 
    date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')

    li = []

    for filename in all_files:
        if date in filename:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)
        else:
            print('inside else')

    frame = pd.concat(li, axis=0, ignore_index=True)
    ##print'DataFrame'
    #print frame
    
    esname=frame['Name'].values
    #print 'Esname:'
    present=[]
    coues=[]
    espresent={}
    #print esname
    for st in studname:
        cou=0;
        name=st
        #print'List name'
        #print name
        present.append(name)
        for es in esname:
            siva=es
            #print'List ==2 Name'
            siva=siva.replace("'","")
            siva2=siva.replace("[","")
            siva3=siva2.replace("]","")
            #print siva3
            #print name
            if name==siva3:
                cou=cou+1
                
        coues.append(cou)
                
        #print'present list and count list=='
        #print present
        #print coues
    lenList = len(present)
    for elements in range(0,lenList) :
        key = present[elements]
        value = coues[elements]
        espresent[key] = value
    #print'Espresent==='
    #print espresent.items()
    ts = time.time()      
    date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    prelist=[]
    abslist=[]
    for st in studname:
        stname=st
        precou=espresent.get(stname, 0)
        #print'precount & Student name'
        #print precou
        #print stname
        if precou>3:
            prelist.append(stname)
            fileName1="Attendance\Final_Present_"+date+"_"+Hour+".txt"
            with open(fileName1, 'w') as f:
                for item in prelist:
                    f.write("%s\n" % item)
                
        else:
            abslist.append(stname)
            fileName1="Attendance\Final_Absent_"+date+"_"+Hour+".txt"
            with open(fileName1, 'w') as f:
                for item in abslist:
                    f.write("%s\n" % item)
            
                
    
                
                
            
  
def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time','Emotion']
    co=['name']
    attendance = pd.DataFrame(columns = col_names)
    namess=""

    for index, row in df.iterrows():
        namess+= row['Name']+" "
    aa=""


    
    attendance1 = pd.DataFrame(columns = co)
    EmotionCounter = {}
    studentSet = set()

    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        frameClone = im.copy()
        if len(faces) > 0:
            faces = sorted(faces, reverse=True,
            key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
                        # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare
                # the ROI for classification via the CNN
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
            Id, conf = recognizer.predict(gray[fY:fY+fH,fX:fX+fW])
            print("id==",Id)
            print("conf==",conf)
            print("emotion==",label)

            name = Id
            emotion = label
            if name in studentSet:
                emotionCount = EmotionCounter[name]
                emotionCount[emotion] = emotionCount[emotion] + 1;
            else:
                studentSet.add(name)
                EmotionCounter[name] = {"angry":0 ,"disgust":0,"scared":0, "happy":0, "sad":0, "surprised":0, "neutral":0} 
                emotionCount = EmotionCounter[name]
                emotionCount[emotion] = emotionCount[emotion] + 1;

     
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                    # construct the label text
                    text = "{}: {:.2f}%".format(emotion, prob * 100)
                    w = int(prob * 300)
                    cv2.rectangle(canvas, (7, (i * 35) + 5),
                    (w, (i * 35) + 35), (0, 0, 255), -1)
                    cv2.putText(canvas, text, (10, (i * 35) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)
                    cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                                  (0, 0, 255), 2)

        
        #cv2.imshow("Probabilities", canvas)
                                          
        if(conf < 50):
            ts = time.time()      
            date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            timeStamp = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            aa=df.loc[df['Id'] == Id]['Name'].values
            #print str(aa)
            aaa=''.join(e for e in aa if e.isalnum())
            #print aaa
            tt=str(Id)+"-"+aa
            emotionCount = EmotionCounter[name]
            emotionMax = max(zip(emotionCount.values(),emotionCount.keys()))[1]
            # print(emotionMax)
            attendance.loc[len(attendance)] = [Id,aa,date,timeStamp,emotionMax]
            attendance1.loc[len(attendance)] = [aa]
            
            namess=namess.replace(aaa, " ")
            
            
            
            
        else:
            Id='Unknown'                
            tt=str(Id)  
        if(conf > 75):
            import os
            noOfFile=len(os.listdir("ImagesUnknown"))+1
            #cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
        cv2.putText(frameClone,str(tt),(fX,fY+fH), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')
        cv2.imshow('your_face', frameClone)
        #cv2.imshow('im',im)
        #from datetime import datetime
        local = datetime.now()
        aa= local.strftime("%M")
        #print aa
        status=0
        if int(aa)%2==0:
            status=1
            if status==1:
                ts = time.time()      
                date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                Hour,Minute,Second=timeStamp.split(":")
                fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+".csv"
                fileName1="Attendance\Absent_"+date+"_"+Hour+"-"+Minute+".txt"
                import os
                exists = os.path.isfile('/path/to/file')
                if exists:
                    print ("")
                    # Store configuration file values
                else:
                    attendance.to_csv(fileName,index=False)
                    #gradeBool = (df != attendance1).stack()  # Create Frame of comparison booleans
                    #gradediff = pd.concat([df['Name'].stack()[gradeBool],attendance1.stack()[gradeBool]], axis=1)

                    #df_1notin2 = df[~(df['Name'].isin(attendance1['Name']) )].reset_index(drop=True)
                    
                    file = open(fileName1,"w")  
                    file.write(namess)
                    file.close() 
                    #namess.to_csv(fileName,index=False)
                    #print 'Absent'
                    #print aa
                    #print namess

                            
##                cam.release()
##                cv2.destroyAllWindows()
                #print(attendance)
                res=attendance
                message2.configure(text= res)
            
            #break
        if (cv2.waitKey(1)==ord('q')):
            cam.release()
            cv2.destroyAllWindows()

            break
##    ts = time.time()      
##    date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
##    timeStamp = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
##    Hour,Minute,Second=timeStamp.split(":")
##    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
##    attendance.to_csv(fileName,index=False)
##    cam.release()
##    cv2.destroyAllWindows()
##    #print(attendance)
##    res=attendance
##    message2.configure(text= res)

clearButton = tk.Button(window, text="Clear", command=clear  ,fg="green"  ,bg="gainsboro"  ,width=15  ,height=2 ,activebackground = "Red" ,font=('times', 20, ' bold '))
clearButton.place(x=950, y=200)
clearButton2 = tk.Button(window, text="Clear", command=clear2  ,fg="green"  ,bg="gainsboro"  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 20, ' bold '))
clearButton2.place(x=950, y=300)    
retakeImg = tk.Button(window, text="Retrain", command=TakeImages1  ,fg="red"  ,bg="yellow"  ,width=10  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
retakeImg.place(x=200, y=500)
Attendance = tk.Button(window, text="Calculate", command=calculate  ,fg="red"  ,bg="yellow"  ,width=10  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
Attendance.place(x=350, y=500)
takeImg = tk.Button(window, text="Take Images", command=TakeImages  ,fg="red"  ,bg="yellow"  ,width=10  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
takeImg.place(x=500, y=500)
trainImg = tk.Button(window, text="Train Images", command=TrainImages  ,fg="red"  ,bg="yellow"  ,width=10  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
trainImg.place(x=650, y=500)
trackImg = tk.Button(window, text="Track Images", command=TrackImages  ,fg="red"  ,bg="yellow"  ,width=10  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
trackImg.place(x=800, y=500)
quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg="red"  ,bg="yellow"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=950, y=500)
 
window.mainloop()
