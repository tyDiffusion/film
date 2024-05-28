import bisect
import os
import os.path
import sys
from tqdm import tqdm
import torch
import numpy as np
import cv2

sys.path.append(os.getenv("PYTHONPATH_FILM")) 

from util import load_image

def getModel(modelPath, gpu, half):
    model = torch.jit.load(modelPath, map_location='cpu')
    model.eval()
    
    if not half:    
        model.float()
            
    if gpu and torch.cuda.is_available():
        print("Using GPU for frame interpolation...\n")
        if half:
            model = model.half()
        else:
            model.float()
        model = model.cuda()
    else:
        print("Using CPU for frame interpolation...\n")
        
    return model
        

def padNum(num):
    neg = num < 0
    
    if (num < 0):
        num = -num
        
    numStr = str(num)
    if (len(numStr) == 4):
        numStr = "0" + numStr
    elif (len(numStr) == 3):
        numStr = "00" + numStr
    elif (len(numStr) == 2):
        numStr = "000" + numStr
    elif (len(numStr) == 1):
        numStr = "0000" + numStr
        
    if (neg):
        numStr = "-" + numStr
        
    return numStr
    
def interpTwoFramesFILM(model, gpu, half, img1, img2, inter_frames, save_num, save_name, save_path):
        
    newFrames = []
    img_batch_1, crop_region_1 = load_image(img1)
    img_batch_2, crop_region_2 = load_image(img2)

    img_batch_1 = torch.from_numpy(img_batch_1).permute(0, 3, 1, 2)
    img_batch_2 = torch.from_numpy(img_batch_2).permute(0, 3, 1, 2)

    results = [
        img_batch_1,
        img_batch_2
    ]

    idxes = [0, inter_frames + 1]
    remains = list(range(1, inter_frames + 1))

    splits = torch.linspace(0, 1, inter_frames + 2)
    
            
    for _ in range(len(remains)):
        starts = splits[idxes[:-1]]
        ends = splits[idxes[1:]]
        distances = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - .5).abs()
        matrix = torch.argmin(distances).item()
        start_i, step = np.unravel_index(matrix, distances.shape)
        end_i = start_i + 1

        x0 = results[start_i]
        x1 = results[end_i]

        if gpu and torch.cuda.is_available():
            if half:
                x0 = x0.half()
                x1 = x1.half()
            x0 = x0.cuda()
            x1 = x1.cuda()

        dt = x0.new_full((1, 1), (splits[remains[step]] - splits[idxes[start_i]])) / (splits[idxes[end_i]] - splits[idxes[start_i]])

        with torch.no_grad():
            prediction = model(x0, x1, dt)
        insert_position = bisect.bisect_left(idxes, remains[step])
        idxes.insert(insert_position, remains[step])
        results.insert(insert_position, prediction.clamp(0, 1).cpu().float())
        del remains[step]
    
    y1, x1, y2, x2 = crop_region_1
    frames = [(tensor[0] * 255).byte().flip(0).permute(1, 2, 0).numpy()[y1:y2, x1:x2].copy() for tensor in results]

    
    w, h = frames[0].shape[1::-1]
    
    for frame in frames:        
        framePath = save_path + save_name + padNum(save_num) + ".png"                        
        cv2.imwrite(framePath, frame)
        newFrames.append(framePath)
        save_num += 1
        
    return newFrames



if __name__ == '__main__':
    
    import argparse

