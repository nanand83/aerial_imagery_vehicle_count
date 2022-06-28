import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import localizerVgg
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plot
import visdom
import math
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from PIL import Image
from CARPK import CARPK
import cv2
from skimage import transform as sktransform
import sys

def detect_peaks(image):
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    background = (image == 0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks

def preprocess(img_file_name, min_size=720, max_size=1280):
    try:
        img = Image.open('CARPK/Images/' + img_file_name + '.png').convert('RGB')
    except:
        img = Image.open('CARPK/Images/' + img_file_name + '.PNG').convert('RGB')

    img = np.asarray(img, dtype=np.float32)

    H, W, C = img.shape

    print (H,W,C)

    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktransform.resize(img, (int(H * scale), int(W * scale), C), mode='reflect', anti_aliasing=True)
    img = np.asarray(img, dtype=np.float32)
    return img

if __name__ == '__main__':
    downsampling_ratio = 8 # Downsampling ratio
    cm_jet = mpl.cm.get_cmap('jet')
    model = localizerVgg.localizervgg16(pretrained=True, dsr=downsampling_ratio)
    model.load_state_dict(torch.load('trained_model_CARPK_x8_2_12.pt', map_location=torch.device('cpu')))

    model.eval()
    #model.cuda()
    model.to(torch.device('cpu'))

    gi = 0
    gRi = 0
    ind = 0
    with torch.no_grad():
        img_file_name = sys.argv[1]
        im = preprocess(img_file_name)
        if im.ndim == 2:
            im = im[np.newaxis]
        else:
            im = im.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        normalize = transforms.Normalize(mean=[0.39895892, 0.42411209, 0.40939609], std=[0.19080092, 0.18127358, 0.19950577])
        img = normalize(torch.from_numpy(im).unsqueeze(0))
        MAP = model(img)
        cMap = MAP[0,0,].data.cpu().numpy()
        cMap = (cMap - cMap.min()) / (cMap.max() - cMap.min())

        cMap[cMap < 0.05] = 0
        peakMAP = detect_peaks(cMap)
        
        print ("Image File: " + img_file_name + ", Num cars predicted : " + str(np.sum(peakMAP)))


        img = img.numpy()[0,]
        img = np.array(img)
        img = (img - img.min()) / (img.max() - img.min())
        #plot.imshow(img.transpose((1, 2, 0)))

        M1 = MAP.data.cpu().contiguous().numpy().copy()
        M1_norm = (M1[0, ] - M1[0, ].min()) / (M1[0, ].max() - M1[0, ].min())
        upsampler = transforms.Compose([transforms.ToPILImage(), transforms.Resize((img.shape[1] , img.shape[2]))])
        a = upsampler(torch.Tensor(M1_norm))
        a = np.uint8(cm_jet(np.array(a)) * 255)
        from PIL import Image
        ima = Image.fromarray(a)
        peakMAP = np.uint8(np.array(peakMAP) * 255)
        peakI = Image.fromarray(peakMAP).convert("RGB")
        peakI = peakI.resize((1280,720))
        ima.save("res2/heatmap-" + img_file_name + ".bmp")
        peakI.save("res2/peakmap-" + img_file_name + ".bmp")
        print(peakI.size)
        print(ima.size)
        # plot.imshow(a)
        # plot.show()

        # plot.imshow(peakMAP)
        # plot.show()


