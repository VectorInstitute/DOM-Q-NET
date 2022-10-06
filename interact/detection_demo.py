import time
import os
import glob
import torch
import torch.nn as nn
import configs
from algorithms import qlearn
from dstructs import replay, prio_replay
from utils import track, trackX, schedule
import ipdb
from entries.template import create_build_f, create_env_f, create_action_space_f
from actors import dqn_actor

from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')
import numpy as np

from PIL import Image

NUM_CONSECUTIVE = 10

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def filter(predictions):
    predictions = predictions[0]
    return predictions['boxes'][(predictions['scores'] > 0.9) & (predictions['labels'] == 3)]

def detect(img_path, model):

    # for img_path in glob.glob('/h/jwilles/DOM-Q-NET/data/*.png'):
    #     img, pil_img = load_query_data(img_path)
    #     predictions = model(img.unsqueeze(0))
    #     print(predictions)
    #     visualize(pil_img, filter(predictions), img_path)

    # Inference
    img, pil_img = load_query_data(img_path)
    predictions = filter(model(img.unsqueeze(0)))
    visualize(pil_img, predictions, img_path)
    return predictions

def load_query_data(path):
    image = Image.open(path)
    pil_transform = transforms.Compose([
        transforms.PILToTensor()
    ])
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(image), pil_transform(image)


def visualize(img, bboxes, path):
    plt.rcParams["savefig.bbox"] = 'tight'
    colors = ['red' for _ in range(len(bboxes))]
    overlay = draw_bounding_boxes(img, bboxes, colors=colors)
    show(overlay, path)

def show(imgs, path):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.ion()
    plt.show()
    plt.pause(5.0) 
    # time.sleep(5.0)
    plt.close()
    #plt.savefig('car_bbox_' + path[26:-4])


num2words = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', \
             6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten', \
            11: 'Eleven', 12: 'Twelve', 13: 'Thirteen', 14: 'Fourteen', \
            15: 'Fifteen', 16: 'Sixteen', 17: 'Seventeen', 18: 'Eighteen', \
            19: 'Nineteen', 20: 'Twenty', 30: 'Thirty', 40: 'Forty', \
            50: 'Fifty', 60: 'Sixty', 70: 'Seventy', 80: 'Eighty', \
            90: 'Ninety', 0: 'Zero'}


def submit(res_dir, settings, hparams_list, paths_list, prints_dict, detections):

    paths = paths_list[0]
    buffer_device = torch.device(settings.get("buffer_device", "cuda:0"))
    device = torch.device(settings.get("batch_device", "cuda:0"))
    saved_path = paths["saved_path"]
    ckpt = torch.load(saved_path)
    nn_hs, qlearn_hs, replay_hs, other_hs = ckpt["hparams_list"]
    V_tag, V_text, V_class = ckpt["V_tag"], ckpt["V_text"], ckpt["V_class"]

    build_net_f, save_dict, common_track_f = create_build_f(
            nn_hs, qlearn_hs, other_hs, prints_dict, None, V_text, V_tag, V_class
            )
    q_net, net_track_f = build_net_f(buffer_device, device)
    q_net.load_state_dict(ckpt["net"])
    q_net.eval()

    # Configure env f 
    print("Detected: ", len(detections), "Cars")
    param = num2words[len(detections)]
    #param = str(len(detections))
    env_f = create_env_f(nn_hs, qlearn_hs, other_hs, settings, dynamic_params=param)
    i = 0
    #q = input("START")
    time.sleep(2.0)
    actor = dqn_actor.Actor(
        env_f, None, q_net,
        None, None, None,
        qlearn_hs["max_step_per_epi"], None, None, buffer_device, device
    )
    while True:
        reward, done = actor.just_act()
        time.sleep(2)
        if done:
            actor.reset()
            break
            # q = input("Press any (q quit) to continue...")
            # if q == "q":
            #     break
            # else:
            #     actor.reset()
            #     time.sleep(2.0)




def main(res_dir, settings, hparams_list, paths_list, prints_dict):
    #import pdb; pdb.set_trace()
    detector = fasterrcnn_resnet50_fpn(pretrained=True)
    detector.eval()
    for img_path in glob.glob('/h/jwilles/DOM-Q-NET/data/high_density/*.png'):
        detections = detect(img_path, detector)
        submit(res_dir, settings, hparams_list, paths_list, prints_dict, detections)

    # paths = paths_list[0]
    # buffer_device = torch.device(settings.get("buffer_device", "cuda:0"))
    # device = torch.device(settings.get("batch_device", "cuda:0"))
    # saved_path = paths["saved_path"]
    # ckpt = torch.load(saved_path)
    # nn_hs, qlearn_hs, replay_hs, other_hs = ckpt["hparams_list"]
    # V_tag, V_text, V_class = ckpt["V_tag"], ckpt["V_text"], ckpt["V_class"]

    # build_net_f, save_dict, common_track_f = create_build_f(
    #         nn_hs, qlearn_hs, other_hs, prints_dict, None, V_text, V_tag, V_class
    #         )
    # q_net, net_track_f = build_net_f(buffer_device, device)
    # q_net.load_state_dict(ckpt["net"])
    # q_net.eval()

    # # Configure env f 
    # env_f = create_env_f(nn_hs, qlearn_hs, other_hs, settings, dynamic_params="test")
    # i = 0
    # q = input("START")

    # actor = dqn_actor.Actor(
    #     env_f, None, q_net,
    #     None, None, None,
    #     qlearn_hs["max_step_per_epi"], None, None, buffer_device, device
    # )
    # while True:
    #     reward, done = actor.just_act()
    #     time.sleep(2)
    #     if done:
    #         q = input("Press any (q quit) to continue...")
    #         if q == "q":
    #             break
    #         else:
    #             actor.reset()
    #         time.sleep(2.0)
            
            # time.sleep(2.0)
            #q = input("Reward=%d, Press any (q quit) to continue..."%int(reward))
            #if q == "q":
            #    break
            #else:
            #    actor.reset()
            #    time.sleep(2.0)