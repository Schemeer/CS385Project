#!/usr/bin/env python
# -*- coding: utf-8 -*-

# multi instance train model
import tensorboardX
import time
import os
import torch
import torch.nn as nn
import numpy as np
from util import torch_util
from util import npmetrics
from util import evaluate
from transformer import Transformer
from model import fvloader
from model import matloader

use_gpu = torch.cuda.is_available()

device = torch.device("cuda:0" if use_gpu else "cpu")
num_class = 10

def run_origin_train(model, dloader, imbtrain_data, writer, step, criterion):
    print("------run origin imblance train data-----------", step)
    model.eval()
    with torch.no_grad():
        st = time.time()

        for item in dloader.batch_fv(imbtrain_data, len(imbtrain_data)):
            genes, nimgs, labels, timesteps = item

            inputs = torch.from_numpy(nimgs).type(torch.cuda.FloatTensor)
            gt = torch.from_numpy(labels).type(torch.cuda.FloatTensor)
            pd = model(inputs)

            # loss = criterion(pd, gt)
            # criterion = torch.nn.BCELoss(reduction='none')
            all_loss = criterion(pd, gt)
            label_loss = torch.mean(all_loss, dim=0)
            loss = torch.mean(label_loss)

            for i in range(num_class):
                writer.add_scalar("origin sl_%d_loss" % i,
                                  label_loss[i].item(), step)
            writer.add_scalar("origin loss", loss.item(), step)

            val_pd = torch_util.threshold_tensor_batch(pd)
            # val_pd = torch.ge(pd, 0.5)
            np_pd = val_pd.data.cpu().numpy()
            torch_util.torch_metrics(labels, np_pd, writer, step,
                                     mode="origin")

        et = time.time()
        writer.add_scalar("origin time", et - st, step)
        return loss.item()


def run_val(model, dloader, val_data, writer, val_step, criterion):
    print("------run val-----------", val_step)
    model.eval()
    with torch.no_grad():
        st = time.time()
        # loss = 0
        # lab_f1_macro = 0
        num = 0
        gt = torch.from_numpy(np.array([[0 for _ in range(10)]])).type(torch.cuda.FloatTensor)
        pd = torch.from_numpy(np.array([[0 for _ in range(10)]])).type(torch.cuda.FloatTensor)

        for item in dloader.batch_fv(val_data, batch=1):
            num += 1
            genes, nimgs, labels, timesteps = item

            inputs = torch.from_numpy(nimgs).type(torch.cuda.FloatTensor)
            gt = torch.cat((gt, torch.from_numpy(labels).type(torch.cuda.FloatTensor)))
            # print(gt)
            pd = torch.cat((pd, model(inputs)))
            # print(pd)

            # loss = criterion(pd, gt)
            # criterion = torch.nn.BCELoss(reduction='none')
        gt = gt[1:, :]
        pd = pd[1:, :]
        all_loss = criterion(pd, gt)
        label_loss = torch.mean(all_loss, dim=0)
        loss = torch.mean(label_loss)

        for i in range(num_class):
            writer.add_scalar("val sl_%d_loss" % i,
                              label_loss[i].item(), val_step)
        writer.add_scalar("val loss", loss.item(), val_step)

        val_pd = torch_util.threshold_tensor_batch(pd)
        # val_pd = torch.ge(pd, 0.5)
        np_pd = val_pd.data.cpu().numpy()  # hard predicts

        lab_f1_macro = torch_util.torch_metrics(gt.cpu().numpy(), np_pd, writer, val_step)
        # print(labels, pd.cpu().numpy())
        auc = evaluate.auc(gt.cpu().numpy(), pd.cpu().numpy())
        mif1 = evaluate.micro_f1(gt.cpu().numpy(), np_pd)
        maf1 = evaluate.macro_f1(gt.cpu().numpy(), np_pd)

        et = time.time()
        writer.add_scalar("val time", et - st, val_step)
        return loss.item(), lab_f1_macro, auc, mif1, maf1


def run_test(model, dloader, test_data, result):
    model.eval()
    with torch.no_grad():
        gt = np.array([[0 for _ in range(10)]])
        pd = np.array([[0 for _ in range(10)]])

        for item in dloader.batch_fv(test_data, batch=1):
            genes, nimgs, labels, timesteps = item

            inputs = torch.from_numpy(nimgs).type(torch.cuda.FloatTensor)

            # print(gt)
            apd = model(inputs)
            test_pd = torch_util.threshold_tensor_batch(apd)
            np_pd = test_pd.data.cpu().numpy()

            gt = np.concatenate((gt, labels))
            pd = np.concatenate((pd, np_pd))

        gt = gt[1:, :]
        pd = pd[1:, :]
        npmetrics.write_metrics(gt, pd, result)


def garbage_shuffle(train_data):
    genes, nimgs, labels, timesteps = zip(*train_data)
    np_labels = np.array(labels)
    np.random.shuffle(np_labels)
    s_labels = list(np_labels)
    garbage_data = list(zip(genes, nimgs, s_labels, timesteps))
    return garbage_data


def train(fv, model_name, criterion, balance=False,
          batchsize=64, size=0, fold=1):
    if fv == "matlab":
        dloader = matloader
    else:
        dloader = fvloader

    # train_data = dloader.load_train_data(size=size, balance=balance, fv=fv)
    # val_data = dloader.load_val_data(size=size, fv=fv)
    # test_data = dloader.load_test_data(size=size, fv=fv)
    train_data = dloader.load_kfold_train_data(fold=fold, fv=fv)
    val_data = dloader.load_kfold_val_data(fold=fold, fv=fv)
    test_data = dloader.load_kfold_test_data(fold=fold, fv=fv)
    # model_name = "transformer_%s_size%d_bce" % (fv, size)
    model_dir = os.path.join("./modeldir-revision/%s" % model_name)
    model_pth = os.path.join(model_dir, "model.pth")

    writer = tensorboardX.SummaryWriter(model_dir)

    if os.path.exists(model_pth):
        print("------load model--------")
        model = torch.load(model_pth)
    else:
        # model = Transformer(fv, NUM_HEADS=4, NUM_LAYERS=3).cuda()
        # model = Transformer(fv).cuda()
        model = Transformer(fv).to(device)
    model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.00005, weight_decay=0.001)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer, factor=0.5,
    #         patience=30, min_lr=1e-4)

    epochs = 800
    step = 1
    val_step = 1
    max_f1 = 0.0

    for e in range(epochs):
        model.train()
        print("------epoch--------", e)
        st = time.time()

        train_shuffle = fvloader.shuffle(train_data)
        num = -1
        gt = torch.from_numpy(np.array([[0 for _ in range(10)]])).type(torch.cuda.FloatTensor)
        pd = torch.from_numpy(np.array([[0 for _ in range(10)]])).type(torch.cuda.FloatTensor)

        for item in fvloader.batch_fv(train_shuffle, batch=batchsize):
            num += 1
            # for name, param in model.named_parameters():
            #     writer.add_histogram(
            #         name, param.clone().cpu().data.numpy(), step)

            # writer.add_histogram(
            #     "grad/"+name, param.grad.clone().cpu().data.numpy(), step)
            model.zero_grad()

            genes, nimgs, labels, timesteps = item
            inputs = torch.from_numpy(nimgs).type(torch.cuda.FloatTensor)
            # print("train", inputs.shape)

            # gt = torch.from_numpy(labels).type(torch.cuda.FloatTensor)
            # pd = model(inputs)
            gt = torch.cat((gt, torch.from_numpy(labels).type(torch.cuda.FloatTensor)))
            # print(gt)
            pd = torch.cat((pd, model(inputs)))

            if num % 32 == 31:  # 32次的结果
                # loss = criterion(pd, gt)
                gt = gt[1:, :]
                pd = pd[1:, :]
                # print(gt)
                # print(pd)
                all_loss = criterion(pd, gt)
                label_loss = torch.mean(all_loss, dim=0)
                loss = torch.mean(label_loss)
                # for i in range(6):
                #     writer.add_scalar("train sl_%d_loss" % i,
                #                       label_loss[i].item(), step)

                train_pd = torch_util.threshold_tensor_batch(pd)
                np_pd = train_pd.data.cpu().numpy()
                # print(np_pd)
                torch_util.torch_metrics(
                    gt.cpu().numpy(), np_pd, writer, step, mode="train")

                writer.add_scalar("train loss", loss, step)
                loss.backward()
                optimizer.step()
                step += 1
                gt = torch.from_numpy(np.array([[0 for _ in range(10)]])).type(torch.cuda.FloatTensor)
                pd = torch.from_numpy(np.array([[0 for _ in range(10)]])).type(torch.cuda.FloatTensor)

        et = time.time()
        writer.add_scalar("train time", et - st, e)
        for param_group in optimizer.param_groups:
            writer.add_scalar("lr", param_group['lr'], e)

        # run_origin_train(model, imbtrain_data, writer, e, criterion)

        if e % 1 == 0:  # val per two epochs
            val_loss, val_f1, auc, mif1, maf1 = run_val(
                model, dloader, val_data, writer, val_step, criterion)
            # scheduler.step(val_loss)
            val_step += 1
            print("----------val result--------\n\tauc===>:{}, micro_f1===>:{}, macro_f1===>:{}".format(auc, mif1, maf1))
            if e == 0:
                start_loss = val_loss
                min_loss = start_loss

            # if val_loss > 2 * min_loss:
            #     print("early stopping at %d" % e)
            #     break
            # if e % 50 == 0:
            #     pt = os.path.join(model_dir, "%d.pt" % e)
            #     torch.save(model.state_dict(), pt)
            #     result = os.path.join(model_dir, "result_epoch%d.txt" % e)
            #     run_test(model, test_data, result)

            if min_loss > val_loss or max_f1 < val_f1:  # loss 降低了或者f1提高了
                if min_loss > val_loss:
                    print("---------save best----------\n\t", "loss", val_loss)
                    min_loss = val_loss
                if max_f1 < val_f1:
                    print("---------save best----------\n\t", "f1", val_f1)
                    max_f1 = val_f1
                torch.save(model, model_pth)
                result = os.path.join(model_dir, "result_epoch%d.txt" % e)
                run_test(model, dloader, test_data, result)


def final_test(fv="res18-128", size=2):
    test_data = fvloader.load_test_data(size=size)

    model_name = "transformer_%s_size%d" % (fv, size)
    model_dir = os.path.join("./modeldir/%s" % model_name)
    model_pth = os.path.join(model_dir, "model.pth")

    if os.path.exists(model_pth):
        print("------load model for test--------")
        model = torch.load(model_pth)
    else:
        raise Exception("train model first")

    model.eval()
    result = os.path.join(model_dir, "result.txt")
    run_test(model, test_data, result)


if __name__ == "__main__":
    # train("res18-128", 0)
    # train("matlab", 0)
    pass
