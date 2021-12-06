import torch

import flash

from flash.image import ImageClassificationData, ImageClassifier

import tkinter
from tkinter import *
from PIL import Image, ImageTk

root = Tk()

photos = [
        r"C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\with_mask_525.jpg",
        r"C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\with_mask_526.jpg",
        r'C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\without_mask_500.jpg',
        r'C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\without_mask_501.jpg',
        r'C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\with_mask_3710.jpg',
        r'C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\with_mask_3711.jpg',
        r'C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\without_mask_3200.jpg',
        r'C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\CYMERA_20210922_113550.jpg',
        r'C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\IMG_20211120_175110.jpg',
        r'C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\Screenshot_20211120-231252.jpg',
        r'C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\without_mask_2500.jpg',
        r'C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\WhatsApp Image 2021-11-22 at 9.25.13 PM.jpeg',
        r'C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\adi wom.jpeg',
        r'C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\sreek wom.jpg',
    ]

a=50
b=50
width = 50
height = 50
for photo in photos:
    image1 = Image.open(photo)
    resize_image = image1.resize((width, height))
    #image1 = ImageTk.PhotoImage(resize_image)
    #test = ImageTk.PhotoImage(image1)
    test = ImageTk.PhotoImage(resize_image)
    label1 = tkinter.Label(image=test)
    label1.image = test

    label1.place(x=a, y=b)
    b=b+60
    if b==350:
        a=a+300
        b=50

# 1. Create the DataModule

datamodule = ImageClassificationData.from_folders(
    train_folder="C:\\Users\\boorl\\OneDrive\\Desktop\\MASK DETECTION\\mask_detection_data\\train",
    val_folder="C:\\Users\\boorl\\OneDrive\\Desktop\\MASK DETECTION\\mask_detection_data\\val"
)

# 2. Build the task
model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Predict what's on a few images! With or without mask?
predictions = model.predict(
    [
        r"C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\with_mask_525.jpg",
        r"C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\with_mask_526.jpg",
        r'C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\without_mask_500.jpg',
        r'C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\without_mask_501.jpg',
        r'C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\with_mask_3710.jpg',
        r'C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\with_mask_3711.jpg',
        r'C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\without_mask_3200.jpg',
        r'C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\CYMERA_20210922_113550.jpg',
        r'C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\IMG_20211120_175110.jpg',
        r'C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\Screenshot_20211120-231252.jpg',
        r'C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\without_mask_2500.jpg',
        r'C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\WhatsApp Image 2021-11-22 at 9.25.13 PM.jpeg',
        r'C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\adi wom.jpeg',
        r'C:\Users\boorl\OneDrive\Desktop\MASK DETECTION\mask_detection_data\sreek wom.jpg',
    ]
)
print(predictions)

a=150
b=50
for prediction in predictions:
    user_name = Label(root, text = prediction).place(x = a,y = b)
    b=b+60
    if b==350:
        a=a+300
        b=50

# 5. Save the model!
trainer.save_checkpoint("image_classification_model.pt")

root.mainloop()