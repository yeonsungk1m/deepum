import sys
import argparse
import os
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
import pandas as pd
import multiprocessing

import sys
sys.path.insert(0, "./")
import deepUMQA
    
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader
import torch




def main():
    parser = argparse.ArgumentParser(description="predictor network train")
    
    parser.add_argument("folder",
                        action="store",
                        help="Location of folder to save checkpoints to.")
    
    parser.add_argument("--epoch",
                        "-e", action="store",
                        type=int,
                        default=100,
                        )
    

    parser.add_argument("--multi_dir",
                        "-multi_dir",
                        action="store_true",
                        default=False,
                        help="Run with multiple direcotory sources")
    
    parser.add_argument("--num_blocks",
                        "-numb", action="store",
                        type=int,
                        default=3,
                        help="# reidual blocks")
    
    parser.add_argument("--num_filters",
                        "-numf", action="store",
                        type=int,
                        default=128,
                        help="# of base filter size in residual blocks")
    
    parser.add_argument("--size_limit",
                        "-size_limit", action="store",
                        type=int,
                        default=300,
                        help="protein size limit")
    
    parser.add_argument("--decay",
                        "-d", action="store",
                        type=float,
                        default=0.99,
                        help="Decay rate for learning rate (Default: 0.99)")
    
    parser.add_argument("--base",
                        "-b", action="store",
                        type=float,
                        default=0.0005,
                        help="Base learning rate (Default: 0.0005)")
    
    parser.add_argument("--debug",
                        "-debug",
                        action="store_true",
                        default=False,
                        help="Debug mode (Default: False)")

    parser.add_argument("--runtime_pdb",
                        action="store_true",
                        default=False,
                        help="Featurize PDBs on the fly instead of loading precomputed .npz files")

    parser.add_argument("--model_type",
                        choices=["resnet", "graph"],
                        default="resnet",
                        help="Choose between the original ResNet model or the new graph-transformer")

    parser.add_argument("--graph_neighbor_cutoff",
                        type=float,
                        default=10.0,
                        help="Distance (Å) threshold on CB–CB map for graph edges when using the graph model")
    
    parser.add_argument("--silent",
                        "-s",
                        action="store_true",
                        default=False,
                        help="Run in silent mode (Default: False)")
   
    args = parser.parse_args()
    
    script_dir = os.path.dirname(__file__)
    base = join(script_dir, "data/")

    epochs = args.epoch
    base_learning_rate = args.base
    decay = args.decay
    loss_weight = [1, 1, 10]
    validation = True
    name = args.folder
    lengthmax = args.size_limit
    
    if not args.silent: print("Loading samples")
        
    proteins = np.load(join(base, "train_decoys.npy"), allow_pickle=True)
    if args.debug: proteins = proteins[:100]
    train_decoys = deepUMQA.DecoyDataset(targets = proteins,
                                           lengthmax = lengthmax,
                                           multi_dir = args.multi_dir,
                                           use_pdb_runtime = args.runtime_pdb)
    train_dataloader = DataLoader(train_decoys, batch_size=1, shuffle=True, num_workers=4)

    proteins = np.load(join(base, "valid_decoys.npy"), allow_pickle=True)
    if args.debug: proteins = proteins[:100]
    valid_decoys = deepUMQA.DecoyDataset(targets = proteins,
                                           lengthmax = lengthmax,
                                           multi_dir = args.multi_dir,
                                           use_pdb_runtime = args.runtime_pdb)
    valid_dataloader = DataLoader(valid_decoys, batch_size=1, shuffle=True, num_workers=4)

    if not args.silent: print("instantitate a model")

    if args.model_type == "resnet":
        net = deepUMQA.myDeepUMQA(num_chunks   = args.num_blocks,
                                  num_channel  = args.num_filters,
                                  twobody_size = 33)
    else:
        # Peek one sample to infer feature dimensions for the graph model
        sample = train_decoys[0]
        node_dim = sample["1d"].shape[-1]
        pair_dim = sample["2d"].shape[1]
        num_bins = len(train_decoys.digits) + 1
        net = deepUMQA.GraphFeatureNet(
            node_in_dim=node_dim,
            pair_in_dim=pair_dim,
            num_bins=num_bins,
            neighbor_cutoff=args.graph_neighbor_cutoff,
        )
    rdevreModel = False
    
    if isfile(join(name, "model.pkl")):
        if not args.silent: print("checkpoint")
        checkpoint = torch.load(join(name, "model.pkl"))
        net.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint["epoch"] + 1
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]
        best_models = checkpoint["best_models"]
        if not args.silent: print("Restarting at epoch", epoch)
        assert len(train_loss["total"]) == epoch
        assert len(valid_loss["total"]) == epoch
        rdevreModel = True
    else:
        if not args.silent: print("Training")
    epoch = 0
    train_loss = {"total": [], "dev": [], "mask": [], "lddt": []}
    valid_loss = {"total": [], "dev": [], "mask": [], "lddt": []}
    best_models = []
    if not isdir(name):
        if not args.silent: print("Creating new dir", name)
        os.makedirs(name, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    net.to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    if rdevreModel:
        checkpoint = torch.load(join(name, "model.pkl"))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = epoch
    for epoch in range(start_epoch, epochs):  

        lr = base_learning_rate*np.power(decay, epoch)#The learning rate decreased gradually with the epoch of training algebra
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        net.train(True)
        temp_loss = {"total":[], "dev":[], "mask":[], "lddt":[]}
        for i, data in enumerate(train_dataloader):

            idx, val, f1d, f2d, dev, dev_1hot, mask = data["idx"], data["val"], data["1d"], data["2d"],\
                                                        data["deviation"], data["deviation_1hot"], data["mask"]
            idx = idx[0].long().to(device)
            val = val[0].to(device)
            f1d = f1d[0].to(device)
            f2d = f2d[0].to(device)
            dev_true = dev[0].to(device)

            dev_1hot_true = dev_1hot[0].to(device)
            mask_true = mask[0].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if args.model_type == "resnet":
                dev_pred, mask_pred, lddt_pred, (dev_logits, mask_logits) = net(idx, val, f1d, f2d)
                lddt_true = deepUMQA.calculate_LDDT(dev_1hot_true[0], mask_true[0])
            else:
                dev_pred, mask_pred, lddt_pred, (dev_logits, mask_logits) = net(f1d.unsqueeze(0), f2d.unsqueeze(0))
                lddt_true = deepUMQA.calculate_LDDT(dev_1hot_true[0], mask_true[0])
            Esto_Loss = torch.nn.CrossEntropyLoss()
            Mask_Loss = torch.nn.BCEWithLogitsLoss()
            Lddt_Loss = torch.nn.MSELoss()

            dev_loss = Esto_Loss(dev_logits, dev_true.long())
            mask_loss = Mask_Loss(mask_logits, mask_true)
            lddt_loss = Lddt_Loss(lddt_pred, lddt_true.float())

            loss = loss_weight[0]*dev_loss + loss_weight[1]*mask_loss + loss_weight[2]*lddt_loss
            loss.backward()
            optimizer.step()

            # Get training loss
            temp_loss["total"].append(loss.cpu().detach().numpy())
            temp_loss["dev"].append(dev_loss.cpu().detach().numpy())
            temp_loss["mask"].append(mask_loss.cpu().detach().numpy())
            temp_loss["lddt"].append(lddt_loss.cpu().detach().numpy())

            # Display training results
            sys.stdout.write("\rEpoch: [%2d/%2d], Batch: [%2d/%2d], loss: %.2f, dev-loss: %.2f, lddt-loss: %.2f, mask: %.2f"
                             %(epoch+1, epochs, i+1, len(train_decoys),
                               temp_loss["total"][-1], temp_loss["dev"][-1], temp_loss["lddt"][-1], temp_loss["mask"][-1]))
            file1 = open("/data7/yeonsungkim/DeepUMQA-main/train.txt",'a')
            file1.write("\rEpoch: [%2d/%2d], Batch: [%2d/%2d], loss: %.2f, dev-loss: %.2f, lddt-loss: %.2f, mask: %.2f"
                             %(epoch+1, epochs, i+1, len(train_decoys),
                               temp_loss["total"][-1], temp_loss["dev"][-1], temp_loss["lddt"][-1], temp_loss["mask"][-1]))
            file1.close()

        train_loss["total"].append(np.array(temp_loss["total"]))
        train_loss["dev"].append(np.array(temp_loss["dev"]))
        train_loss["mask"].append(np.array(temp_loss["mask"]))
        train_loss["lddt"].append(np.array(temp_loss["lddt"]))

        if validation:
            net.eval() # turn off training mode
            temp_loss = {"total":[], "dev":[], "mask":[], "lddt":[]}
            with torch.no_grad(): # wihout tracking gradients
                for i, data in enumerate(valid_dataloader):

                    # Get the data, Hardcoded transformation for whatever reasons.
                    idx, val, f1d, f2d, dev, dev_1hot, mask = data["idx"], data["val"], data["1d"], data["2d"],\
                                                                data["deviation"], data["deviation_1hot"], data["mask"]
                    idx = idx[0].long().to(device)
                    val = val[0].to(device)
                    f1d = f1d[0].to(device)
                    f2d = f2d[0].to(device)
                    dev_true = dev[0].to(device)
                    dev_1hot_true = dev_1hot[0].to(device)
                    mask_true = mask[0].to(device)

                    # forward + backward + optimize
                    if args.model_type == "resnet":
                        dev_pred, mask_pred, lddt_pred, (dev_logits, mask_logits) = net(idx, val, f1d, f2d)
                        lddt_true = deepUMQA.calculate_LDDT(dev_1hot_true[0], mask_true[0])
                    else:
                        dev_pred, mask_pred, lddt_pred, (dev_logits, mask_logits) = net(f1d.unsqueeze(0), f2d.unsqueeze(0))
                        lddt_true = deepUMQA.calculate_LDDT(dev_1hot_true[0], mask_true[0])

                    Esto_Loss = torch.nn.CrossEntropyLoss()
                    Mask_Loss = torch.nn.BCEWithLogitsLoss()
                    Lddt_Loss = torch.nn.MSELoss()

                    dev_loss = Esto_Loss(dev_logits, dev_true.long())
                    mask_loss = Mask_Loss(mask_logits, mask_true)
                    lddt_loss = Lddt_Loss(lddt_pred, lddt_true.float())

                    loss = loss_weight[0]*dev_loss + loss_weight[1]*mask_loss + loss_weight[2]*lddt_loss

                    # Get training loss
                    temp_loss["total"].append(loss.cpu().detach().numpy())
                    temp_loss["dev"].append(dev_loss.cpu().detach().numpy())
                    temp_loss["mask"].append(mask_loss.cpu().detach().numpy())
                    temp_loss["lddt"].append(lddt_loss.cpu().detach().numpy())

                    sys.stdout.write("\rEpoch: [%2d/%2d], Batch: [%2d/%2d], loss: %.2f, dev-loss: %.2f, lddt-loss: %.2f, mask: %.2f"
                             %(epoch+1, epochs, i+1, len(valid_decoys),
                               temp_loss["total"][-1], temp_loss["dev"][-1], temp_loss["lddt"][-1], temp_loss["mask"][-1]))

                    file2 = open("/data7/yeonsungkim/DeepUMQA-main/valid.txt",'a')
                    file2.write("\rEpoch: [%2d/%2d], Batch: [%2d/%2d], loss: %.2f, dev-loss: %.2f, lddt-loss: %.2f, mask: %.2f"
                             %(epoch+1, epochs, i+1, len(valid_decoys),
                               temp_loss["total"][-1], temp_loss["dev"][-1], temp_loss["lddt"][-1], temp_loss["mask"][-1]))

            valid_loss["total"].append(np.array(temp_loss["total"]))
            valid_loss["dev"].append(np.array(temp_loss["dev"]))
            valid_loss["mask"].append(np.array(temp_loss["mask"]))
            valid_loss["lddt"].append(np.array(temp_loss["lddt"]))

            # Saving the model if needed.
            if name != "" and validation:
                
                folder = name
                # Name of ranked models. I know it is not optimal way to do it but the easiest fix is this.
                name_map = ["best.pkl", "second.pkl", "third.pkl", "fourth.pkl", "fifth.pkl"]
                
                new_model = (epoch, np.mean(valid_loss["total"][-1]))
                new_best_models = best_models[:]
                new_best_models.append(new_model)
                new_best_models.sort(key=lambda x:x[1])
                temp = new_best_models[:len(name_map)]
                new_best_models = [(temp[i][0], temp[i][1], name_map[i]) for i in range(len(temp))]

                # Saving and moving
                for i in range(len(new_best_models)):
                    m, performance, filename = new_best_models[i]
                    if m in [j[0] for j in best_models]:
                        index = [j[0] for j in best_models].index(m)
                        command = "mv %s %s"%(join(folder, best_models[index][2]), join(folder, "temp_"+new_best_models[i][2]))
                        os.system(command)
                    else:
                         torch.save({
                            'epoch': epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': train_loss,
                            'valid_loss': valid_loss,
                        }, join(folder, "temp_"+new_best_models[i][2]))

                for i in range(len(new_best_models)):
                    command = "mv %s %s"%(join(folder, "temp_"+name_map[i]), join(folder, name_map[i]))
                    os.system(command)

                # Update best list                              
                best_models = new_best_models
            
            # Save all models
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    'best_models' : best_models
                    }, join(name, "model.pkl"))



if __name__== "__main__":
    main()
