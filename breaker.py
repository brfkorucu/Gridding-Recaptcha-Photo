################################################################################################

import cv2
import numpy as np
import time
import sys
import os
import optparse

################################################################################################

parser = optparse.OptionParser("usage%prog " + "-p <name of jpg's>")
parser.add_option("-p", dest="path_name", type="str", help="specify jpg path")
options, args = parser.parse_args()
path_name = options.path_name
if path_name == None:
    print(parser.usage)
    exit(0)

################################################################################################

CONFIDENCE = 0.2
SCORE_THRESHOLD = 0.2
IOU_THRESHOLD = 0.2

config_path = "cfg/yolov3.cfg"
weights_path = "cfg/yolov3.weights"
labels = open("cfg/coco.names").read().strip().split("\n")

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

image = cv2.imread(path_name)
file_name = os.path.basename(path_name)
filename, ext = file_name.split(".")
h, w = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

#print("image.shape:", image.shape)
#print("blob.shape:", blob.shape)

net.setInput(blob)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

start = time.perf_counter()
layer_outputs = net.forward(ln)
time_took = time.perf_counter() - start
print(f"\nTime took: {time_took:.2f}s")

boxes, confidences, class_ids = [], [], []
for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > CONFIDENCE:
            box = detection[:4] * np.array([w, h, w, h])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

#print(detection.shape)
#print("\nObjects : ", boxes)

################################################################################################

if image.shape == (450, 450, 3):

    print("\nİmage shape ", image.shape)

    def box_path_x1(x1):

        if x1 <= 112.5:
            path_x1 = "G0"
            return path_x1
        if 112.5 < x1 <= 225:
            path_x1 = "G1"
            return path_x1 
        if 225 < x1 <= 337.5:
            path_x1 = "G2"
            return path_x1
        if 337.5 < x1 <= 450:
            path_x1 = "G3"
            return path_x1

    def box_path_x2(x2):

        if x2 <= 112.5:
            path_x2 = "G0"
            return path_x2
        if 112.5 < x2 <= 225:
            path_x2 = "G1"
            return path_x2
        if 225 < x2 <= 337.5:
            path_x2 = "G2"
            return path_x2
        if 337.5 < x2 <= 450:
            path_x2 = "G3"
            return path_x2

    def box_path_y1(y1):

        if y1 <= 112.5:
            path_y1 = "G0"
            return path_y1
        if 112.5 < y1 <= 225:
            path_y1 = "G4"
            return path_y1
        if 225 < y1 <= 337.5:
            path_y1 = "G8"
            return path_y1
        if 337.5 < y1 <= 450:
            path_y1 = "G12"
            return path_y1

    def box_path_y2(y2):

        if y2 <= 112.5:
            path_y2 = "G0"
            return path_y2
        if 112.5 < y2 <= 225:
            path_y2 = "G4"
            return path_y2
        if 225 < y2 <= 337.5:
            path_y2 = "G8"
            return path_y2
        if 337.5 < y2 <= 450:
            path_y2 = "G12"
            return path_y2

    def box_corner(x,y):

        if x == "G0" and y == "G0":
            path = "G0"
            return path
        if x == "G0" and y == "G4":
            path = "G4"
            return path    
        if x == "G0" and y == "G8":
            path = "G8"
            return path
        if x == "G0" and y == "G12":
            path = "G12"
            return path
        if x == "G1" and y == "G0":
            path = "G1"
            return path
        if x == "G1" and y == "G4":
            path = "G5"
            return path    
        if x == "G1" and y == "G8":
            path = "G9"
            return path
        if x == "G1" and y == "G12":
            path = "G13"
            return path
        if x == "G2" and y == "G0":
            path = "G2"
            return path
        if x == "G2" and y == "G4":
            path = "G6"
            return path
        if x == "G2" and y == "G8":
            path = "G10"
            return path
        if x == "G2" and y == "G12":
            path = "G14"
            return path
        if x == "G3" and y == "G0":
            path = "G3"
            return path
        if x == "G3" and y == "G4":
            path = "G7"
            return path
        if x == "G3" and y == "G8":
            path = "G11"
            return path
        if x == "G3" and y == "G12":
            path = "G15"
            return path

    box_bicycle = []
    box_car = []
    box_motorcycle = []
    box_bus = []
    box_boat = []
    box_traffic_light = []
    box_fire_hydrant = []
    box_parking_meter = []

    for a in range(len(boxes)):

        x1 = boxes[a][0]
        x1_box = box_path_x1(x1)
        x2 = boxes[a][2] + x1
        x2_box = box_path_x2(x2)
        y1 = boxes[a][1]
        y1_box = box_path_y1(y1)
        y2 = boxes[a][3] + y1
        y2_box = box_path_y2(y2)

        x1y1 = box_corner(x=x1_box, y=y1_box)
        x1y1_n = ""
        for n in x1y1:
            if n != "G":
                x1y1_n += n

        x2y1 = box_corner(x=x2_box, y=y1_box)
        x2y1_n = ""
        for n in x2y1:
            if n != "G":
                x2y1_n += n

        x1y2 = box_corner(x=x1_box, y=y2_box)
        x1y2_n = ""
        for n in x1y2:
            if n != "G":
                x1y2_n += n

        x2y2 = box_corner(x=x2_box, y=y2_box)
        x2y2_n = ""
        for n in x2y2:
            if n != "G":
                x2y2_n += n

        x1y1_n = int(x1y1_n)
        x2y1_n = int(x2y1_n)
        x1y2_n = int(x1y2_n)
        x2y2_n = int(x2y2_n)

        if class_ids[a] == 1:
            class_ids[a] = "Bicycle"
        
            if x1y1 == x2y1 and x1y1 == x1y2 and x1y1 == x2y2:
                click_box = [x1y1]
                if click_box[0] not in box_bicycle:
                    box_bicycle.append(click_box[0])
            else :
                row_1 = []
                row_2 = []
                for top_line in range((x2y1_n-x1y1_n)+1):
                    row_1.append(x1y1_n + top_line)
                for bottom_line in range((x2y2_n-x1y2_n)+1):
                    row_2.append(x1y2_n + bottom_line)

                click_box = []
                
                for i in range(len(row_1)):
                    click_box.append(row_1[i])
                    while row_1[i] < row_2[i]:
                        row_1[i] = row_1[i] + 4
                        click_box.append(row_1[i])

                    click_box.pop
                
                for i in range(len(click_box)):
                    click_box[i] = str("G") + str(click_box[i])
                    if click_box[i] not in box_bicycle:
                        box_bicycle.append(click_box[i])

        if class_ids[a] == 2:
            class_ids[a] = "Car"
            
            if x1y1 == x2y1 and x1y1 == x1y2 and x1y1 == x2y2:
                click_box = [x1y1]
                if click_box[0] not in box_car:
                    box_car.append(click_box[0])
            else :
                row_1 = []
                row_2 = []
                for top_line in range((x2y1_n-x1y1_n)+1):
                    row_1.append(x1y1_n + top_line)
                for bottom_line in range((x2y2_n-x1y2_n)+1):
                    row_2.append(x1y2_n + bottom_line)

                click_box = []
                
                for i in range(len(row_1)):
                    click_box.append(row_1[i])
                    while row_1[i] < row_2[i]:
                        row_1[i] = row_1[i] + 4
                        click_box.append(row_1[i])

                    click_box.pop
                
                for i in range(len(click_box)):
                    click_box[i] = str("G") + str(click_box[i])
                    if click_box[i] not in box_car:
                        box_car.append(click_box[i])

        if class_ids[a] == 3:
            class_ids[a] = "Motorcycle"

            if x1y1 == x2y1 and x1y1 == x1y2 and x1y1 == x2y2:
                click_box = [x1y1]
                if click_box[0] not in box_motorcycle:
                    box_motorcycle.append(click_box[0])
            else :
                row_1 = []
                row_2 = []
                for top_line in range((x2y1_n-x1y1_n)+1):
                    row_1.append(x1y1_n + top_line)
                for bottom_line in range((x2y2_n-x1y2_n)+1):
                    row_2.append(x1y2_n + bottom_line)

                click_box = []
                
                for i in range(len(row_1)):
                    click_box.append(row_1[i])
                    while row_1[i] < row_2[i]:
                        row_1[i] = row_1[i] + 4
                        click_box.append(row_1[i])

                    click_box.pop

                for i in range(len(click_box)):
                    click_box[i] = str("G") + str(click_box[i])
                    if click_box[i] not in box_motorcycle:
                        box_motorcycle.append(click_box[i])

        if class_ids[a] == 5:
            class_ids[a] = "Bus"

            if x1y1 == x2y1 and x1y1 == x1y2 and x1y1 == x2y2:
                click_box = [x1y1]
                if click_box[0] not in box_bus:
                    box_bus.append(click_box[0])
            else :
                row_1 = []
                row_2 = []
                for top_line in range((x2y1_n-x1y1_n)+1):
                    row_1.append(x1y1_n + top_line)
                for bottom_line in range((x2y2_n-x1y2_n)+1):
                    row_2.append(x1y2_n + bottom_line)

                click_box = []
                
                for i in range(len(row_1)):
                    click_box.append(row_1[i])
                    while row_1[i] < row_2[i]:
                        row_1[i] = row_1[i] + 4
                        click_box.append(row_1[i])

                    click_box.pop
                
                for i in range(len(click_box)):
                    click_box[i] = str("G") + str(click_box[i])
                    if click_box[i] not in box_bus:
                        box_bus.append(click_box[i])

        if class_ids[a] == 8:
            class_ids[a] = "Boat"

            if x1y1 == x2y1 and x1y1 == x1y2 and x1y1 == x2y2:
                click_box = [x1y1]
                if click_box[0] not in box_boat:
                    box_boat.append(click_box[0])
            else :
                row_1 = []
                row_2 = []
                for top_line in range((x2y1_n-x1y1_n)+1):
                    row_1.append(x1y1_n + top_line)
                for bottom_line in range((x2y2_n-x1y2_n)+1):
                    row_2.append(x1y2_n + bottom_line)

                click_box = []
                
                for i in range(len(row_1)):
                    click_box.append(row_1[i])
                    while row_1[i] < row_2[i]:
                        row_1[i] = row_1[i] + 4
                        click_box.append(row_1[i])

                    click_box.pop

                for i in range(len(click_box)):
                    click_box[i] = str("G") + str(click_box[i])
                    if click_box[i] not in box_boat:
                        box_boat.append(click_box[i])

        if class_ids[a] == 9:
            class_ids[a] = "Traffic Light"

            if x1y1 == x2y1 and x1y1 == x1y2 and x1y1 == x2y2:
                click_box = [x1y1]
                if click_box[0] not in box_traffic_light:
                    box_traffic_light.append(click_box[0])
            else :
                row_1 = []
                row_2 = []
                for top_line in range((x2y1_n-x1y1_n)+1):
                    row_1.append(x1y1_n + top_line)
                for bottom_line in range((x2y2_n-x1y2_n)+1):
                    row_2.append(x1y2_n + bottom_line)

                click_box = []
                
                for i in range(len(row_1)):
                    click_box.append(row_1[i])
                    while row_1[i] < row_2[i]:
                        row_1[i] = row_1[i] + 4
                        click_box.append(row_1[i])

                    click_box.pop

                for i in range(len(click_box)):
                    click_box[i] = str("G") + str(click_box[i])
                    if click_box[i] not in box_traffic_light:
                        box_traffic_light.append(click_box[i])

        if class_ids[a] == 12:
            class_ids[a] = "Parking meter"

            if x1y1 == x2y1 and x1y1 == x1y2 and x1y1 == x2y2:
                click_box = [x1y1]
                if click_box[0] not in box_parking_meter:
                    box_parking_meter.append(click_box[0])
            else :
                row_1 = []
                row_2 = []
                for top_line in range((x2y1_n-x1y1_n)+1):
                    row_1.append(x1y1_n + top_line)
                for bottom_line in range((x2y2_n-x1y2_n)+1):
                    row_2.append(x1y2_n + bottom_line)

                click_box = []
                
                for i in range(len(row_1)):
                    click_box.append(row_1[i])
                    while row_1[i] < row_2[i]:
                        row_1[i] = row_1[i] + 4
                        click_box.append(row_1[i])

                    click_box.pop
        
                for i in range(len(click_box)):
                    click_box[i] = str("G") + str(click_box[i])
                    if click_box[i] not in box_parking_meter:
                        box_parking_meter.append(click_box[i])

################################################################################################

if image.shape == (300, 300, 3):

    print("\nİmage shape ", image.shape)

    def box_path_x1(x1):

        if x1 <= 100:
            path_x1 = "G0"
            return path_x1
        if 100 < x1 <= 200:
            path_x1 = "G1"
            return path_x1 
        if 200 < x1 <= 300:
            path_x1 = "G2"
            return path_x1

    def box_path_x2(x2):

        if x2 <= 100:
            path_x2 = "G0"
            return path_x2
        if 100 < x2 <= 200:
            path_x2 = "G1"
            return path_x2
        if 200 < x2 <= 300:
            path_x2 = "G2"
            return path_x2    

    def box_path_y1(y1):

        if y1 <= 100:
            path_y1 = "G0"
            return path_y1
        if 100 < y1 <= 200:
            path_y1 = "G3"
            return path_y1
        if 200 < y1 <= 300:
            path_y1 = "G6"
            return path_y1    

    def box_path_y2(y2):

        if y2 <= 100:
            path_y2 = "G0"
            return path_y2
        if 100 < y2 <= 200:
            path_y2 = "G3"
            return path_y2
        if 200 < y2 <= 300:
            path_y2 = "G6"
            return path_y2    

    def box_corner(x,y):

        if x == "G0" and y == "G0":
            path = "G0"
            return path
        if x == "G0" and y == "G3":
            path = "G3"
            return path    
        if x == "G0" and y == "G6":
            path = "G6"
            return path
        if x == "G1" and y == "G0":
            path = "G1"
            return path
        if x == "G1" and y == "G3":
            path = "G4"
            return path
        if x == "G1" and y == "G6":
            path = "G7"
            return path    
        if x == "G2" and y == "G0":
            path = "G2"
            return path
        if x == "G2" and y == "G3":
            path = "G5"
            return path
        if x == "G2" and y == "G6":
            path = "G8"
            return path

    box_bicycle = []
    box_car = []
    box_motorcycle = []
    box_bus = []
    box_boat = []
    box_traffic_light = []
    box_fire_hydrant = []
    box_parking_meter = []

    for a in range(len(boxes)):

        x1 = boxes[a][0]
        x1_box = box_path_x1(x1)
        x2 = boxes[a][2] + x1
        x2_box = box_path_x2(x2)
        y1 = boxes[a][1]
        y1_box = box_path_y1(y1)
        y2 = boxes[a][3] + y1
        y2_box = box_path_y2(y2)

        x1y1 = box_corner(x=x1_box, y=y1_box)
        x1y1_n = ""
        for n in x1y1:
            if n != "G":
                x1y1_n += n

        x2y1 = box_corner(x=x2_box, y=y1_box)
        x2y1_n = ""
        for n in x2y1:
            if n != "G":
                x2y1_n += n

        x1y2 = box_corner(x=x1_box, y=y2_box)
        x1y2_n = ""
        for n in x1y2:
            if n != "G":
                x1y2_n += n

        x2y2 = box_corner(x=x2_box, y=y2_box)
        x2y2_n = ""
        for n in x2y2:
            if n != "G":
                x2y2_n += n

        x1y1_n = int(x1y1_n)
        x2y1_n = int(x2y1_n)
        x1y2_n = int(x1y2_n)
        x2y2_n = int(x2y2_n)

        if class_ids[a] == 1:
            class_ids[a] = "Bicycle"
        
            if x1y1 == x2y1 and x1y1 == x1y2 and x1y1 == x2y2:
                click_box = [x1y1]
                if click_box[0] not in box_bicycle:
                    box_bicycle.append(click_box[0])
            else :
                row_1 = []
                row_2 = []
                for top_line in range((x2y1_n-x1y1_n)+1):
                    row_1.append(x1y1_n + top_line)
                for bottom_line in range((x2y2_n-x1y2_n)+1):
                    row_2.append(x1y2_n + bottom_line)

                click_box = []
                for i in range(len(row_1)):
                    click_box.append(row_1[i])
                    while row_1[i] < row_2[i]:
                        row_1[i] = row_1[i] + 3
                        click_box.append(row_1[i])

                    click_box.pop
                
                for i in range(len(click_box)):
                    click_box[i] = str("G") + str(click_box[i])
                    if click_box[i] not in box_bicycle:
                        box_bicycle.append(click_box[i])

        if class_ids[a] == 2:
            class_ids[a] = "Car"
            
            if x1y1 == x2y1 and x1y1 == x1y2 and x1y1 == x2y2:
                click_box = [x1y1]
                if click_box[0] not in box_car:
                    box_car.append(click_box[0])
            else :
                row_1 = []
                row_2 = []
                for top_line in range((x2y1_n-x1y1_n)+1):
                    row_1.append(x1y1_n + top_line)
                for bottom_line in range((x2y2_n-x1y2_n)+1):
                    row_2.append(x1y2_n + bottom_line)

                click_box = []
                
                for i in range(len(row_1)):
                    click_box.append(row_1[i])
                    while row_1[i] < row_2[i]:
                        row_1[i] = row_1[i] + 3
                        click_box.append(row_1[i])

                    click_box.pop

                for i in range(len(click_box)):
                    click_box[i] = str("G") + str(click_box[i])
                    if click_box[i] not in box_car:
                        box_car.append(click_box[i])

        if class_ids[a] == 3:
            class_ids[a] = "Motorcycle"

            if x1y1 == x2y1 and x1y1 == x1y2 and x1y1 == x2y2:
                click_box = [x1y1]
                if click_box[0] not in box_motorcycle:
                    box_motorcycle.append(click_box[0])
            else :
                row_1 = []
                row_2 = []
                for top_line in range((x2y1_n-x1y1_n)+1):
                    row_1.append(x1y1_n + top_line)
                for bottom_line in range((x2y2_n-x1y2_n)+1):
                    row_2.append(x1y2_n + bottom_line)

                click_box = []
                
                for i in range(len(row_1)):
                    click_box.append(row_1[i])
                    while row_1[i] < row_2[i]:
                        row_1[i] = row_1[i] + 3
                        click_box.append(row_1[i])

                    click_box.pop

                for i in range(len(click_box)):
                    click_box[i] = str("G") + str(click_box[i])
                    if click_box[i] not in box_motorcycle:
                        box_motorcycle.append(click_box[i])

        if class_ids[a] == 5:
            class_ids[a] = "Bus"

            if x1y1 == x2y1 and x1y1 == x1y2 and x1y1 == x2y2:
                click_box = [x1y1]
                if click_box[0] not in box_bus:
                    box_bus.append(click_box[0])
            else :
                row_1 = []
                row_2 = []
                for top_line in range((x2y1_n-x1y1_n)+1):
                    row_1.append(x1y1_n + top_line)
                for bottom_line in range((x2y2_n-x1y2_n)+1):
                    row_2.append(x1y2_n + bottom_line)

                click_box = []
                
                for i in range(len(row_1)):
                    click_box.append(row_1[i])
                    while row_1[i] < row_2[i]:
                        row_1[i] = row_1[i] + 3
                        click_box.append(row_1[i])

                    click_box.pop

                for i in range(len(click_box)):
                    click_box[i] = str("G") + str(click_box[i])
                    if click_box[i] not in box_bus:
                        box_bus.append(click_box[i])

        if class_ids[a] == 8:
            class_ids[a] = "Boat"

            if x1y1 == x2y1 and x1y1 == x1y2 and x1y1 == x2y2:
                click_box = [x1y1]
                if click_box[0] not in box_boat:
                    box_boat.append(click_box[0])
            else :
                row_1 = []
                row_2 = []
                for top_line in range((x2y1_n-x1y1_n)+1):
                    row_1.append(x1y1_n + top_line)
                for bottom_line in range((x2y2_n-x1y2_n)+1):
                    row_2.append(x1y2_n + bottom_line)

                click_box = []
                
                for i in range(len(row_1)):
                    click_box.append(row_1[i])
                    while row_1[i] < row_2[i]:
                        row_1[i] = row_1[i] + 3
                        click_box.append(row_1[i])

                    click_box.pop
            
                for i in range(len(click_box)):
                    click_box[i] = str("G") + str(click_box[i])
                    if click_box[i] not in box_boat:
                        box_boat.append(click_box[i])

        if class_ids[a] == 9:
            class_ids[a] = "Traffic Light"

            if x1y1 == x2y1 and x1y1 == x1y2 and x1y1 == x2y2:
                click_box = [x1y1]
                if click_box[0] not in box_traffic_light:
                    box_traffic_light.append(click_box[0])
            else :
                row_1 = []
                row_2 = []
                for top_line in range((x2y1_n-x1y1_n)+1):
                    row_1.append(x1y1_n + top_line)
                for bottom_line in range((x2y2_n-x1y2_n)+1):
                    row_2.append(x1y2_n + bottom_line)

                click_box = []
                
                for i in range(len(row_1)):
                    click_box.append(row_1[i])
                    while row_1[i] < row_2[i]:
                        row_1[i] = row_1[i] + 3
                        click_box.append(row_1[i])

                    click_box.pop
            
                for i in range(len(click_box)):
                    click_box[i] = str("G") + str(click_box[i])
                    if click_box[i] not in box_traffic_light:
                        box_traffic_light.append(click_box[i])

        if class_ids[a] == 12:
            class_ids[a] = "Parking meter"

            if x1y1 == x2y1 and x1y1 == x1y2 and x1y1 == x2y2:
                click_box = [x1y1]
                if click_box[0] not in box_parking_meter:
                    box_parking_meter.append(click_box[0])
            else :
                row_1 = []
                row_2 = []
                for top_line in range((x2y1_n-x1y1_n)+1):
                    row_1.append(x1y1_n + top_line)
                for bottom_line in range((x2y2_n-x1y2_n)+1):
                    row_2.append(x1y2_n + bottom_line)

                click_box = []
                
                for i in range(len(row_1)):
                    click_box.append(row_1[i])
                    while row_1[i] < row_2[i]:
                        row_1[i] = row_1[i] + 3
                        click_box.append(row_1[i])

                    click_box.pop
                
                for i in range(len(click_box)):
                    click_box[i] = str("G") + str(click_box[i])
                    if click_box[i] not in box_parking_meter:
                        box_parking_meter.append(click_box[i])

################################################################################################

print("\n")
print("Bicycle       => ", box_bicycle)
print("Car           => ", box_car)
print("Motorcycle    => ", box_motorcycle)
print("Bus           => ", box_bus)
print("Boat          => ", box_boat)
print("Traffic light => ", box_traffic_light)
print("Fire hydrant  => ", box_fire_hydrant)
print("Parking meter => ", box_parking_meter)
