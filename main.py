from tkinter import *
from tkinter import messagebox
from tkinter import filedialog

from PIL import ImageTk, Image

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import load_model
from tensorflow.keras.preprocessing import image


from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

window = Tk()
model = load_model('ECG_CNN.h5')
img_analys = 0

def main():
    # model.summary()
    # print(model.get_weights())
    #Запуск GUI
    initGUI()


def initGUI():
    """Инициализация компонентов интерфейса"""
    menu_window = Menu(window)
    window.config(menu=menu_window)
    menu_file = Menu(menu_window, tearoff=0)
    menu_help = Menu(menu_window, tearoff=0)

    """Настройки компонентов интерфейса"""
    window.geometry("505x350+450+250")
    window.resizable(width=False, height=False)
    window.title("ECG analyzer")

    menu_file.add_command(label="Открыть фрагмент ЭКГ", command=openImageECG)
    menu_file.add_separator()
    menu_file.add_command(label="Выйти", command=window.destroy)

    menu_help.add_command(label="О программе", command=openWindowAboutProgram)
    menu_help.add_command(label="О нас", command=openWindowAboutUs)

    """Запуск компонентов интерфейса"""
    menu_window.add_cascade(label="Меню", menu=menu_file)
    menu_window.add_cascade(label="Справка", menu=menu_help)

    window.iconbitmap(default='icon.ico')
    window.mainloop()

def openWindowAboutUs():
    messagebox.showinfo('О нас', 'Разработчики (ПРО-418):\nГорбунов Владлен\nТишков Арсений\nУметбаев Айгиз')

def openWindowAboutProgram():
    messagebox.showinfo('О программе', 'Lorem ipsum dolor sit amit')

def openImageECG():
    """Открыввает файловый менеджер и импортирует изображение"""
    file = filedialog.askopenfilename(title="Выберите ЭКГ изображение", filetypes=[("Image files", "*.png *.jpg")])

    img_path = Image.open(file)
    img_analys = img_path
    img_path = img_path.resize((505, 300), Image.ADAPTIVE)
    print(img_path)
    img_path = ImageTk.PhotoImage(img_path)
    print(img_path)

    panel = Label(window, image=img_path)
    panel.image = img_path
    panel.grid(row=1)

    button = Button(window, text='Проанализировать изображение', command=analysisImageECG)
    button.grid(row=2)



def analysisImageECG():
    image_width = 300
    image_height = 505
    img_path = 'norm.jpg'


    img = image.load_img(img_path, target_size=(image_width, image_height), color_mode="rgb")

    plt.imshow(img.convert('RGBA'))
    plt.show()

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)

    print(result)

    if result[0][0] == 0:
        print('Есть признаки инфаркта')
    else:
        print('Нормальное ЭКГ')



if __name__ == '__main__':
    main()