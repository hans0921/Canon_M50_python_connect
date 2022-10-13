import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from torch import nn
import torch

def thread_Capture(CAMERA,Frame_share,stop_threads_share):  
    vid = cv2.VideoCapture(CAMERA,cv2.CAP_DSHOW)
    vid.set(cv2.CAP_PROP_FPS, 30)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while(True):
        ret, frame = vid.read()
        Frame_share.put(frame)
        cv2.imshow('Original', frame[:,:,:])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_threads_share.put(True)
            vid.release()
            cv2.destroyAllWindows() # Destroy all the windows
            break
        else:
            stop_threads_share.put(False)

def thread_Sobel(Frame_share,stop_threads_share):  
    while(True):
        cv2.waitKey(1)
        if stop_threads_share.get() == True:
            break
        else:
            frame = Frame_share.get()
            torch_sobel = Sobel()
            rgb_orig = frame[:,:,1]/np.max(frame[:,:,1])
            rgb_orig = cv2.resize(rgb_orig, (640, 480))
            rgb_edged = sobel_torch_version(rgb_orig, torch_sobel=torch_sobel)
            cv2.imshow('rgb_edged', rgb_edged)

def sobel_torch_version(img_np, torch_sobel):
    img_tensor = np_img_to_tensor(np.float32(img_np))
    img_edged = tensor_to_np_img(torch_sobel(img_tensor))
    img_edged = np.squeeze(img_edged)
    return img_edged

def tensor_to_np_img(img_tensor):
    img = img_tensor.cpu().permute(0, 2, 3, 1).numpy()
    return img  # get the first element since it's batch form

def np_img_to_tensor(img):
    if len(img.shape) == 2:
        img = img[..., np.newaxis]
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    return img_tensor

class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x