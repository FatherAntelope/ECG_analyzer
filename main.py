from tkinter import *
from tkinter import messagebox
from tkinter import filedialog

from PIL import ImageTk, Image

import cv2 as cv

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

import matplotlib.pyplot as plt

window = Tk()
model = load_model('ECG_CNN_30.h5')

image_weight = 505
image_height = 300

color_tone = BooleanVar()
lbl_result = Label(text="")

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
    menu_color_tone = Menu(menu_window, tearoff=0)


    """Настройки компонентов интерфейса"""
    window.geometry("505x370+450+250")
    window.resizable(width=False, height=False)
    window.title("ECG analyzer")

    menu_file.add_command(label="Открыть фрагмент ЭКГ", command=openImageECG)
    """Разделитель между компонентами меню"""
    menu_file.add_separator()
    menu_file.add_command(label="Выйти", command=window.destroy)

    menu_help.add_command(label="О программе", command=openWindowAboutProgram)
    menu_help.add_command(label="О нас", command=openWindowAboutUs)

    menu_color_tone.add_radiobutton(label="Светлый", value=False, variable=color_tone)
    menu_color_tone.add_radiobutton(label="Темный", value=True, variable=color_tone)

    """Запуск компонентов интерфейса"""
    menu_window.add_cascade(label="Меню", menu=menu_file)
    menu_window.add_cascade(label="Оттенок изображения", menu=menu_color_tone)
    menu_window.add_cascade(label="Справка", menu=menu_help)



    window.iconbitmap(default='icon.ico')
    window.mainloop()

def openWindowAboutUs():
    messagebox.showinfo('О нас', 'Разработчики (ПРО-418):\nГорбунов Владлен\nТишков Арсений\nУметбаев Айгиз')

def openWindowAboutProgram():
    messagebox.showinfo('О программе', 'Программа представляет из себя сканер, распознающий признаки инфаркта на графике ЭКГ.\n'
                                       'Для того, чтобы корректно распознать изображение выберите преобладающий оттенок:\n'
                                       '1) Светлый - для изображений со светлым фоном;\n'
                                       '2) Темный - для изображений с темным фоном.\n'
                                       'После настройки оттенка выберите обрезанное изображение с элементом зубца графика ЭКГ.')

def openImageECG():
    """Открыввает файловый менеджер и импортирует изображение"""
    file = filedialog.askopenfilename(title="Выберите ЭКГ изображение", filetypes=[("Image files", "*.png *.jpg")])
    img_path = Image.open(file)

    img_path = img_path.resize((image_weight, image_height), Image.ADAPTIVE)
    img_path = ImageTk.PhotoImage(img_path)

    panel = Label(window, image=img_path)
    panel.image = img_path
    panel.grid(row=1)

    button = Button(window, text='Проанализировать изображение', command=lambda: analysisImageECG(file))
    button.grid(row=2)

    lbl_result.grid(row=3)



def analysisImageECG(file):
    # нужно сюда передать открытый файл с диспетчера
    img = cv.imdecode(np.fromfile(file, dtype=np.uint8), cv.IMREAD_UNCHANGED)
    chanels = np.array(img)
    if chanels.ndim == 3:
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # img = cv.equalizeHist(img)
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

    #20 138
    if color_tone.get() == False:
        threshold, img = cv.threshold(img, 138, 255, cv.THRESH_BINARY_INV)
    else:
        threshold, img = cv.threshold(img, 138, 255, cv.THRESH_BINARY)

    img = cv.resize(img, (image_weight, image_height), interpolation=cv.INTER_AREA)

    img = img.astype('float') / 255

    # plt.imshow(img)
    # plt.show()
    roi = image.img_to_array(img)
    roi = np.expand_dims(roi, axis=0)

    #print(roi)

    result = model.predict(roi)


    if result > 0.5:
        lbl_result.config(text="Нормальное ЭКГ")
    elif 0.000001 < result < 0.5:
        lbl_result.config(text="Обнаружены признаки инфаркта")
    else:
        lbl_result.config(text="Ничего не обнаружено")

    lbl_result["text"] += "\n" + str(result[0])




if __name__ == '__main__':
    main()