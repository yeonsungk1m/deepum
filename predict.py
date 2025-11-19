import sys
sys.path.insert(0, "/iobio/yt/gss/DeepUMQA/")
import deepUMQA as umqa
import torch
import numpy as np
from os.path import join, isfile, isdir, basename, normpath
from os import listdir
import os
import argparse

def main():

    parser = argparse.ArgumentParser(description="predictor network error")
    parser.add_argument("name",
                        action="store",
                        help="name")
    
    args = parser.parse_args()

    name = args.name
    distination = "/iobio/yt/gss/DeepUMQA/"
    multi_dir = True
    lengthmax = 300
    
    for model in ["best", "second", "third", "fourth", "fifth"]:
        
        net = umqa.myDeepUMQA(num_chunks = 3,
                                num_channel = 128,
                                twobody_size = 33)

        checkpoint = torch.load(join(name, "%s.pkl"%(model)))
        net.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint["epoch"]+1
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        print("loading")

        dirpath = join(distination, basename(normpath(name))+"_"+model)
        if not isdir(dirpath):
            os.mkdir(dirpath)

        proteins = np.load("/iobio/yt/gss/DeepUMQA/data/test_decoys.npy")
        for pname in proteins:
            print(pname)
            dirpath = join(distination, basename(normpath(name))+"_"+model, pname)
            if not isdir(dirpath):
                os.mkdir(dirpath)
            decoys = umqa.DecoyDataset(targets = [pname],
                                       lengthmax = lengthmax,
                                       multi_dir = multi_dir)

            if pname in decoys.proteins:
                with torch.no_grad():
                    for i in range(len(decoys.samples_dict[pname])):
                        data = decoys.__getitem__(0, pindex=i)
                        idx, val, f1d, f2d, dev_1hot, dev, mask = data["idx"], data["val"], data["1d"], data["2d"],\
                                                         data["devgram_1hot"], data["devgram"], data["mask"]

                        idx = torch.Tensor(idx).long().to(device)
                        val = torch.Tensor(val).to(device)
                        f1d = torch.Tensor(f1d).to(device)
                        f2d = torch.Tensor(f2d).to(device)
                        dev = torch.Tensor(dev).to(device)
                        dev_1hot = torch.Tensor(dev_1hot).to(device)
                        mask = torch.Tensor(mask).to(device)
                        lddt = umqa.calculate_LDDT(dev_1hot[0], mask[0])

                        dev_pred, mask_pred, lddt_pred, dmy = net(idx, val, f1d, f2d)

                        samplename = basename(decoys.samples_dict[pname][i])
                        filepath = join(distination, basename(normpath(name))+"_"+model, pname, samplename+".npz")
                        np.savez_compressed(filepath,
                                            lddt = lddt_pred.cpu().detach().numpy(),
                                            devgram = dev_pred.cpu().detach().numpy(),
                                            mask = mask_pred.cpu().detach().numpy(),
                                            lddt_true = lddt.cpu().detach().numpy(),
                                            devgram_true = dev_1hot[0].cpu().detach().numpy(),
                                            mask_true = mask[0].cpu().detach().numpy())
    
if __name__== "__main__":
    main()
