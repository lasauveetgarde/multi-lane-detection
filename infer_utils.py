import cv2
import numpy as np
import torch

from class_names import INSTANCE_CATEGORY_NAMES as coco_names

np.random.seed(2023)
# lane_array=[(0,0)]

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

def get_outputs(image, model, threshold):
    with torch.no_grad():
        # forward pass of the image through the model.
        outputs = model(image)
    
    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    # get the masks
    masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    # discard masks for objects which are below threshold
    masks = masks[:thresholded_preds_count]

    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]
    # discard bounding boxes below threshold value
    boxes = boxes[:thresholded_preds_count]
    # get the classes labels
    labels = [coco_names[i] for i in outputs[0]['labels']]
    return masks, boxes, labels    

def draw_segmentation_map(image, masks, boxes, labels, args, background=None):
    alpha = 1.0
    beta = 1.0 # transparency for the segmentation map
    gamma = 0.0 # scalar added to each sum
    #convert the original PIL image into NumPy format
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if background is not None:
        image = background.copy()
    for i in range(len(masks)):
        # apply a randon color mask to each object
        color = COLORS[coco_names.index(labels[i])]
        if  coco_names.index(labels[i]) == 6:
            red_map = np.zeros_like(masks[i]).astype(np.uint8)
            green_map = np.zeros_like(masks[i]).astype(np.uint8)
            blue_map = np.zeros_like(masks[i]).astype(np.uint8)
            red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1] = color
            # combine all the masks into a single image
            segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)

            # apply mask on the depth image
            cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image, dtype=cv2.CV_8UC1)

            # apply mask on the RGB image
            # cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)

            lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # Line width.
            tf = max(lw - 1, 1) # Font thickness.
            p1, p2 = boxes[i][0], boxes[i][1]

            if not args.no_boxes:
                # draw the bounding boxes around the objects
                cv2.rectangle(
                    image, 
                    p1, p2, 
                    color=color, 
                    thickness=lw,
                    lineType=cv2.LINE_AA
                )
                w, h = cv2.getTextSize(
                    labels[i], 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=lw / 3, 
                    thickness=tf
                )[0]  # text width, height

                img_height, img_width = image.shape[:2]

                if p1[0] > img_width/2:
                    cv2.line(
                        image, 
                        p1, 
                        p2, 
                        color=color, 
                        thickness=5, 
                        lineType=cv2.LINE_AA
                    )

                elif p1[0] < img_width/2:
                    cv2.line(
                        image, 
                        (p2[0],p1[1]), 
                        (p1[0],p2[1]), 
                        color=color, 
                        thickness=5, 
                        lineType=cv2.LINE_AA
                    )

                w = int(w - (0.20 * w))
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                # put the label text above the objects
                cv2.rectangle(
                    image, 
                    p1, 
                    p2, 
                    color=color, 
                    thickness=-1, 
                    lineType=cv2.LINE_AA
                )
                cv2.putText(
                    image, 
                    labels[i], 
                    (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=lw / 3.8, 
                    color=(255, 255, 255), 
                    thickness=tf, 
                    lineType=cv2.LINE_AA
                )

    return image

def array_segmentation_map(image, masks, boxes, labels):
    #convert the original PIL image into NumPy format
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    right_line_array_w = []
    right_line_array_h = []
    left_line_array_w = []
    left_line_array_h = []
    combined_points = []
    for i in range(len(masks)):
        if  coco_names.index(labels[i]) == 6:
            p1, p2 = boxes[i][0], boxes[i][1]
            # print('this', p1, p2)
            for right_i_h in range(p1[1], p2[1], int((p2[1]-p1[1])/20)):
                right_line_array_h.append(right_i_h)
            
            for right_i_w in range(p1[0], p2[0], int((p2[0]-p1[0])/len(right_line_array_h))):
                right_line_array_w.append(right_i_w)
            
            if len(right_line_array_h)!=len(right_line_array_w):
                difference_right=len(right_line_array_w)-len(right_line_array_h)
                if len(right_line_array_w) > len(right_line_array_h):
                    del right_line_array_w[-difference_right:]

            right_points = list(zip(right_line_array_w, right_line_array_h))

            for left_i_h in range(p1[1], p2[1], int((p2[1]-p1[1])/20)):
                left_line_array_h.append(left_i_h)
            
            for left_i_w in range(p1[0], p2[0], int((p2[0]-p1[0])/len(left_line_array_h))):
                left_line_array_w.append(left_i_w)
            
            if len(left_line_array_h)!=len(left_line_array_w):
                difference_right=len(left_line_array_w)-len(left_line_array_h)
                if len(left_line_array_w) > len(left_line_array_h):
                    del left_line_array_w[-difference_right:]

            left_points = list(zip(left_line_array_h, left_line_array_h))

            combined_points = right_points + left_points

        return combined_points