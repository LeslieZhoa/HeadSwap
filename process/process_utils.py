import numpy as np
import cv2

def crop_with_padding(image, lmks,bbox,scale=1.8,size=512,align=True,top_scale=0.5):

    l, t, r, b = bbox[:4]
    center = ((l + r) * 0.5, (t + b) * 0.5)
    mask = np.zeros_like(image) + 255
   
    if align:
        lm_eye_left      = lmks[36 : 42]  # left-clockwise
        lm_eye_right     = lmks[42 : 48]  # left-clockwise

        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        angle = np.arctan2((eye_right[1] - eye_left[1]), (eye_right[0] - eye_left[0])) / np.pi * 180

        RotateMatrix = cv2.getRotationMatrix2D(center, angle, scale=1)
       
        rotated_img = cv2.warpAffine(image, RotateMatrix, 
                    (image.shape[1], image.shape[0]),borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, RotateMatrix, (image.shape[1], image.shape[0]))
        rotated_lmks = apply_transform(RotateMatrix, lmks)
        
    else:
        rotated_img = image 
        rotated_lmks = lmks 
        RotateMatrix = np.array([[1,0,0],
                                [0,1,0]])


    cx_box = center[0]
    cy_box = center[1]
    
    bbox_size = int(max(b-t,r-l)*scale)
   
    x_min = int(cx_box-bbox_size / 2.)
    y_min = int(cy_box-bbox_size * top_scale)
    x_max = x_min + bbox_size
    y_max = y_min + bbox_size

    boundingBox = [max(x_min, 0), max(y_min, 0), min(x_max, rotated_img.shape[1]), min(y_max, rotated_img.shape[0])]
    imgCropped = rotated_img[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2]]
    imgCropped = cv2.copyMakeBorder(imgCropped, max(-y_min, 0), max(y_max - image.shape[0], 0), max(-x_min, 0),
                                    max(x_max - image.shape[1], 0),cv2.BORDER_CONSTANT,value=(0,0,0))

    mask = mask[boundingBox[1]:boundingBox[3], boundingBox[0]:boundingBox[2]]
    mask = cv2.copyMakeBorder(mask, max(-y_min, 0), max(y_max - image.shape[0], 0), max(-x_min, 0),
                                    max(x_max - image.shape[1], 0),cv2.BORDER_CONSTANT,value=(0,0,0))
    boundingBox = [x_min, y_min, x_max, y_max]

    scale_h = size / float(bbox_size)
    scale_w = size / float(bbox_size)
    rotated_lmks[:, 0] = (rotated_lmks[:, 0] - boundingBox[0]) * scale_w
    rotated_lmks[:, 1] = (rotated_lmks[:, 1] - boundingBox[1]) * scale_h
    # print(imgCropped.shape)
    imgResize = cv2.resize(imgCropped, (size, size))
    mask = cv2.resize(mask, (size, size))

        

    ### 计算变换(原图->crop box)
    m1 = np.concatenate((RotateMatrix,np.array([[0.0,0.0,1.0]])), axis=0) #rotate(+translation)
    m2 = np.eye(3) #translation
    m2[0][2] = -boundingBox[0]
    m2[1][2] = -boundingBox[1]
    m3 = np.eye(3) #scaling
    m3[0][0] = m3[1][1] = scale_h 
    m = np.matmul(np.matmul(m3,m2),m1)
    im = np.linalg.inv(m)
    info = {'rotated_lmk':rotated_lmks,
            'm':m,
            'im':im,
            'mask':mask}
    
    return imgResize,info


def apply_transform(transform_matrix, lmks):
    '''
    args
        transform_matrix: float (3,3)|(2,3)
        lmks: float (2)|(3)|(k,2)|(k,3)
    
    ret
        ret_lmks: float (2)|(3)|(k,2)|(k,3)
    '''
    if transform_matrix.shape[0] == 2:
        transform_matrix = np.concatenate((transform_matrix,np.array([[0.0,0.0,1.0]])), axis=0)
    only_one = False
    if len(lmks.shape) == 1:
        lmks = lmks[np.newaxis, :]
        only_one = True
    only_two_dim = False
    if lmks.shape[1] == 2:
        lmks = np.concatenate((lmks, np.ones((lmks.shape[0],1), dtype=np.float32)), axis=1)
        only_two_dim = True

    ret_lmks = np.matmul(transform_matrix, lmks.T).T

    if only_two_dim:
        ret_lmks = ret_lmks[:,:2]
    if only_one:
        ret_lmks = ret_lmks[0]
    
    return ret_lmks


def choose_one_detection(frame_faces,box):
    """
        frame_faces
            list of lists of length 5
            several face detections from one image

        return:
            list of 5 floats
            one of the input detections: `(l, t, r, b, confidence)`
    """
    frame_faces = list(filter(lambda x:x[-1]>0.9,frame_faces))
    if len(frame_faces) == 0:
        return None
    
    else:
        # sort by area, find the largest box
        largest_area, largest_idx = -1, -1
        for idx, face in enumerate(frame_faces):
            area = compute_iou(box,face)
            # area = abs(face[2]-face[0]) * abs(face[1]-face[3])
            if area > largest_area:
                largest_area = area
                largest_idx = idx
        
        if largest_area < 0.1:
            return None
        
        retval = frame_faces[largest_idx]
        
       
    return np.array(retval).tolist()


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (x0, y0, x1, y1), which reflects
            (top, left, bottom, right)
    :param rec2: (x0, y0, x1, y1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2], rec2[2])
    top_line = max(rec1[1], rec2[1])
    bottom_line = min(rec1[3], rec2[3])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect))*1.0
        # return intersect / S_rec2