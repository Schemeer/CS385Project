#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

Base_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(Base_DIR)

import torch
import torchvision
import finetune
import cv2
import numpy as np
import time
import gpustat
import threading
import queue

from util import constant as c


# for enhanced level data
DATA_DIR = c.QDATA_DIR
FV_DIR = c.FV_DIR


def get_gpu_usage(device=2):
    gpu_stats = gpustat.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    return item['memory.used'] / item['memory.total']


def extract_image_fv(q, model):

    def _extract_image(image):

        img = cv2.imread(image)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
        # img = cv2.resize(img, interpolation=cv2.INTER_CUBIC)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        # return img

        inputs = torch.from_numpy(img).type(torch.cuda.FloatTensor)
        pd = model(inputs)
        return pd

    while not q.empty():

        while get_gpu_usage() > 0.9:
            print("---gpu full---", get_gpu_usage())
            time.sleep(1)
            torch.cuda.empty_cache()

        gene = q.get()
        print("---extract -----", gene)
        gene_dir = os.path.join(DATA_DIR, gene)
        outpath = os.path.join(FV_DIR, "%s.npy" % gene)
        if os.path.exists(outpath):
            print("------already extracted---------", gene)
            continue

        pds = [_extract_image(os.path.join(gene_dir, p)).cpu()
               for p in os.listdir(gene_dir)
               if os.path.splitext(p)[-1] == ".jpg"]
        if pds:
            value = np.concatenate(pds, axis=0)
            print(value.shape)
            print("----save-----", outpath)
            np.save(outpath, value)


def extract():
    q = queue.Queue()
    for gene in os.listdir(DATA_DIR):  # gene=folder
        q.put(gene)

    # resnet18 = torchvision.models.resnet18(pretrained=True)
    resnet50 = torch.load('resnet10.pt')

    model = finetune.FineTuneModel(resnet50)
    for param in model.parameters():
        param.requires_grad = False
    model.share_memory()
    model.cuda()
    # print(model)

    extract_image_fv(q, model)
    # jobs = []
    # for i in range(8):
    #     p = threading.Thread(target=extract_image_fv, args=(q, model))
    #     jobs.append(p)
    #     p.daemon = True
    #     p.start()
    #
    # for j in jobs:
    #     j.join()


def test_mem():
    # 3000x3000 gpu mem is ok for res18 when batch=1
    resnet18 = torchvision.models.resnet18(pretrained=True)
    model = finetune.FineTuneModel(resnet18)
    for param in model.parameters():
        param.requires_grad = False
    model.share_memory()
    model.cuda()

    image = os.path.join(DATA_DIR, "ENSG00000278845", "51543_A_1_2.jpg")
    img = cv2.imread(image)
    img = np.transpose(img, (2, 0, 1))
    inputs = np.expand_dims(img, axis=0)
    tinputs = torch.from_numpy(inputs).type(torch.cuda.FloatTensor)
    value = model(tinputs)
    print(value)


if __name__ == "__main__":
    # test_mem()
    extract()
