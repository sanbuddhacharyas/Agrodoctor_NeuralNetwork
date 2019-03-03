import os
import matplotlib.pyplot as plt
import cv2
from matplotlib.widgets import RectangleSelector
from generate_xml import write_xml
#global Constant
img= None
tl_list = []
br_list = []
object_list = []

#constant
image_folder ='Leaf'
savedir = 'annotation_tomato_leaf1'
obj = 'tomato_leaf'

def line_select_callback(clk ,rls):
    global tl_list
    global br_list
    tl_list.append((int(clk.xdata), int(clk.ydata)))
    br_list.append((int(rls.xdata), int(rls.ydata)))
    object_list.append(obj)

def toggle_selector(event):
    toggle_selector.RS.set_active(True)

def onekeypress(event):
    global br_list
    global tl_list
    global object_list
    global img
    if event.key == 'q':
        write_xml(image_folder , img ,object_list, tl_list, br_list,savedir)
        br_list = []
        tl_list = []
        object_list = []
        img = None
        plt.close()

if __name__ == '__main__':
    for n , image_file in enumerate(os.scandir(image_folder)): #Counts number of image in folder name image _folder
        img = image_file
        fig ,ax = plt.subplots(1)
        image = cv2.imread(image_file.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        toggle_selector.RS= RectangleSelector(

            ax, line_select_callback,
            drawtype='box',
            useblit=True,
            button=[1],
            minspanx=1,
            minspany=1,
            spancoords='pixels',
            interactive=True
        )
        bbox = plt.connect('key_press_event', toggle_selector)
        key = plt.connect('key_press_event',onekeypress)
        plt.show()