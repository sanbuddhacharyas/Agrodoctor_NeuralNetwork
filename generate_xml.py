import os
import cv2
from lxml import etree
import xml.etree.cElementTree as ET

def write_xml(folder, img ,objects, tl,br,savedir ):
    #Checing in path wheather there is directory or not
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    
    #Read image from path location
    image = cv2.imread(img.path, cv2.IMREAD_COLOR)
    height, width, depth = image.shape

    #Creating Tree for xml
    annotation = ET.Element('annotation')
    #Creating subtree folder from annotation
    ET.SubElement(annotation,'folder').text = folder
    ET.SubElement(annotation,'filename').text = img.name
    ET.SubElement(annotation,'segmented').text = '0'
    #Creating subtree width form size
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)

    #Single image can contatin differet object so determining different object in single image
    for obj,topl ,botr in zip(objects, tl, br):# zip ties all (objects[0],tl[0],br[0]) in single zip file
        ob = ET.SubElement(annotation,'object')
        ET.SubElement(ob, 'name').text = obj
        ET.SubElement(ob, 'pose').text = 'Unspecified'
        ET.SubElement(ob, 'truncated').text = '0'
        ET.SubElement(ob, 'difficult').text ='0'
        bbox = ET.SubElement(ob, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = str(topl[0])
        ET.SubElement(bbox, 'ymin').text = str(topl[1])
        ET.SubElement(bbox, 'xmax').text = str(botr[0])
        ET.SubElement(bbox, 'ymax').text = str(botr[1])

    #Creating tree
    xml_str = ET.tostring(annotation)
    root = etree.fromstring(xml_str)
    #Adding pretty_format to tree like tabs tree flow etc
    xml_str = etree.tostring(root, pretty_print = True)
    save_path  = os.path.join(savedir, img.name.replace('jpg','xml'))
    #Opening saving path in write mode and writing in same name and in that path 
    with open(save_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)


    return xml_str

if __name__ == '__main__':
    folder = 'image_folder'
    img = [im for im in os.scandir('image_folder') if 'dog' in im.name][0]
    objects = ['dog']
    tl = [(10, 10)]
    br = [(100, 100)]
    savedir = 'annoatation'
    write_xml(folder , img ,objects, tl, br, savedir)

