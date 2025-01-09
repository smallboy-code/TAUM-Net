import torch,time
import torch.nn.functional as F
import setproctitle
from utils.tools import all_reduce_tensor
import logging
from sklearn.metrics import roc_auc_score,accuracy_score
import numpy as np
def train_initial(args,train_loader,DDP_model,optimizer,epoch,train_sampler,itr,num_gpu):
    DDP_model['en'].train()
    DDP_model['seg'].train()
    DDP_model['idh'].train()
    DDP_model['mtl'].train()
    train_sampler.set_epoch(epoch)  # shuffle
    setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch + 1, args.end_epoch))
    start_epoch = time.time()

    epoch_train_loss = 0.0
    epoch_train_seg_loss = 0.0
    epoch_train_idh_loss = 0.0

    for i, data in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)
        # optimizer.adjust_learning_rate(epoch, args.end_epoch, args.lr)

        optimizer.zero_grad()
        x, target, grade, idh, seg_weight = data
        x = x.cuda(args.local_rank, non_blocking=True)
        target = target.cuda(args.local_rank, non_blocking=True)
        idh = idh.cuda(args.local_rank, non_blocking=True)
        weight = torch.tensor([57, 91]).float().cuda(args.local_rank, non_blocking=True)  # [57, 91]

        x1_1, x2_1, x3_1, x4_1, encoder_output = DDP_model['en'](x)
        y = DDP_model['seg'](x1_1, x2_1, x3_1, encoder_output)
        idh_out = DDP_model['idh'](x4_1, encoder_output)

        loss, seg_loss, idh_loss, loss1, loss2, loss3, seg_std, idh_std, log_var_1, log_var_2 = DDP_model['mtl'](
            [y, idh_out], [target, idh], [None, weight])

        reduce_idh_loss = all_reduce_tensor(idh_loss, world_size=num_gpu).data.cpu().numpy()
        seg_std = all_reduce_tensor(seg_std, world_size=num_gpu).data.cpu().numpy()
        idh_std = all_reduce_tensor(idh_std, world_size=num_gpu).data.cpu().numpy()
        seg_vars = all_reduce_tensor(log_var_1, world_size=num_gpu).data.cpu().numpy()
        idh_vars = all_reduce_tensor(log_var_2, world_size=num_gpu).data.cpu().numpy()

        reduce_loss = all_reduce_tensor(loss, world_size=num_gpu).data.cpu().numpy()
        reduce_seg_loss = all_reduce_tensor(seg_loss, world_size=num_gpu).data.cpu().numpy()
        reduce_loss1 = all_reduce_tensor(loss1, world_size=num_gpu).data.cpu().numpy()
        reduce_loss2 = all_reduce_tensor(loss2, world_size=num_gpu).data.cpu().numpy()
        reduce_loss3 = all_reduce_tensor(loss3, world_size=num_gpu).data.cpu().numpy()

        epoch_train_loss += reduce_loss / len(train_loader)
        epoch_train_seg_loss += reduce_seg_loss / len(train_loader)
        epoch_train_idh_loss += reduce_idh_loss / len(train_loader)

        if args.local_rank == 0:
            logging.info(
                'Epoch: {}_Iter:{}  loss: {:.5f} seg_loss: {:.5f} idh_loss: {:.5f} || 1:{:.4f} | 2:{:.4f} | 3:{:.4f} ||  seg_std:{:.4f} idh_std:{:.4f} seg_vars:{:.4f} idh_vars:{:.4f}'
                .format(epoch, i, reduce_loss, reduce_seg_loss, reduce_idh_loss, reduce_loss1, reduce_loss2,
                        reduce_loss3, seg_std, idh_std, seg_vars, idh_vars))
        loss.backward()
        optimizer.step()

    return epoch_train_loss,epoch_train_seg_loss,epoch_train_idh_loss


def train_regular(args,train_loader,DDP_model,optimizer,epoch,train_sampler,itr,num_gpu):
    DDP_model['en'].train()
    DDP_model['seg'].train()
    DDP_model['idh'].train()
    DDP_model['mtl'].train()
    train_sampler.set_epoch(epoch)  # shuffle
    setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch + 1, args.end_epoch))
    start_epoch = time.time()

    epoch_train_loss = 0.0
    epoch_train_seg_loss = 0.0
    epoch_train_idh_loss = 0.0

    for i, data in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, args.end_epoch, args.lr)
        # optimizer.adjust_learning_rate(epoch, args.end_epoch, args.lr)

        optimizer.zero_grad()
        x, target, grade, idh, seg_weight = data
        x = x.cuda(args.local_rank, non_blocking=True)
        target = target.cuda(args.local_rank, non_blocking=True)
        idh = idh.cuda(args.local_rank, non_blocking=True)
        weight = torch.tensor([57, 91]).float().cuda(args.local_rank, non_blocking=True)  # [57, 91]  # 46 72

        x1_1, x2_1, x3_1, x4_1, encoder_output = DDP_model['en'](x)
        y = DDP_model['seg'](x1_1, x2_1, x3_1, encoder_output)
        idh_out = DDP_model['idh'](x4_1, encoder_output)

        loss, seg_loss, idh_loss, loss1, loss2, loss3, seg_std, idh_std, log_var_1, log_var_2 = DDP_model['mtl'](
            [y, idh_out], [target, idh], [None, weight])

        reduce_idh_loss = all_reduce_tensor(idh_loss, world_size=num_gpu).data.cpu().numpy()
        seg_std = all_reduce_tensor(seg_std, world_size=num_gpu).data.cpu().numpy()
        idh_std = all_reduce_tensor(idh_std, world_size=num_gpu).data.cpu().numpy()
        seg_vars = all_reduce_tensor(log_var_1, world_size=num_gpu).data.cpu().numpy()
        idh_vars = all_reduce_tensor(log_var_2, world_size=num_gpu).data.cpu().numpy()

        reduce_loss = all_reduce_tensor(loss, world_size=num_gpu).data.cpu().numpy()
        reduce_seg_loss = all_reduce_tensor(seg_loss, world_size=num_gpu).data.cpu().numpy()
        reduce_loss1 = all_reduce_tensor(loss1, world_size=num_gpu).data.cpu().numpy()
        reduce_loss2 = all_reduce_tensor(loss2, world_size=num_gpu).data.cpu().numpy()
        reduce_loss3 = all_reduce_tensor(loss3, world_size=num_gpu).data.cpu().numpy()

        epoch_train_loss += reduce_loss / len(train_loader)
        epoch_train_seg_loss += reduce_seg_loss / len(train_loader)
        epoch_train_idh_loss += reduce_idh_loss / len(train_loader)

        if args.local_rank == 0:
            logging.info(
                'Epoch: {}_Iter:{}  loss: {:.5f} seg_loss: {:.5f} idh_loss: {:.5f} || 1:{:.4f} | 2:{:.4f} | 3:{:.4f} ||  seg_std:{:.4f} idh_std:{:.4f} seg_vars:{:.4f} idh_vars:{:.4f}'
                .format(epoch, i, reduce_loss, reduce_seg_loss, reduce_idh_loss, reduce_loss1, reduce_loss2,
                        reduce_loss3, seg_std, idh_std, seg_vars, idh_vars))
        loss.backward()
        optimizer.step()

    return epoch_train_loss,epoch_train_seg_loss,epoch_train_idh_loss


def valid_regular(args,valid_loader,DDP_model):
    idh_probs = []
    idh_class = []
    idh_target = []

    with torch.no_grad():
        DDP_model['en'].eval()
        DDP_model['seg'].eval()
        DDP_model['idh'].eval()
        DDP_model['mtl'].eval()
        epoch_valid_loss = 0.0
        epoch_seg_loss = 0.0
        epoch_idh_loss = 0.0
        epoch_dice_1 = 0.0
        epoch_dice_2 = 0.0
        epoch_dice_3 = 0.0
        for i, data in enumerate(valid_loader):
            # [t.cuda(args.local_rank, non_blocking=True) for t in data]
            x, target, grade, idh = data

            x = x.cuda(args.local_rank, non_blocking=True)
            target = target.cuda(args.local_rank, non_blocking=True)
            idh = idh.cuda(args.local_rank, non_blocking=True)

            encoder_outs = DDP_model['en'](x)
            seg_out = DDP_model['seg'](encoder_outs[0], encoder_outs[1], encoder_outs[2], encoder_outs[4])
            idh_out = DDP_model['idh'](encoder_outs[3], encoder_outs[4])

            valid_loss, seg_loss, idh_loss, loss1, loss2, loss3, std_1, std_2, var_1, var_2 = DDP_model['mtl'](
                [seg_out, idh_out], [target, idh],
                [None, None])


            epoch_valid_loss += valid_loss / len(valid_loader)
            epoch_seg_loss += seg_loss / len(valid_loader)
            epoch_idh_loss += idh_loss / len(valid_loader)

            epoch_dice_1 += loss1 / len(valid_loader)
            epoch_dice_2 += loss2 / len(valid_loader)
            epoch_dice_3 += loss3 / len(valid_loader)

            idh_pred = F.softmax(idh_out, 1)
            # idh_pred = idh_out.sigmoid()
            idh_pred_class = torch.argmax(idh_pred, dim=1)
            # idh_pred_class = (idh_pred > 0.5).float()
            idh_probs.append(idh_pred[0][1])
            # idh_probs.append(idh_pred[0])
            idh_class.append(idh_pred_class.item())  # .item()
            idh_target.append(idh.item())  # .item()

        accuracy = accuracy_score(idh_target, idh_class)
        auc = roc_auc_score(idh_target, idh_probs)
    return epoch_valid_loss,epoch_seg_loss,epoch_idh_loss,epoch_dice_1,epoch_dice_2,epoch_dice_3,accuracy,auc

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def pseudo_labeling(args,data_loader,DDP_model,names,itr):
    import numpy as np
    from sklearn.metrics import accuracy_score,confusion_matrix
    DDP_model['en'].eval()
    DDP_model['seg'].eval()
    DDP_model['idh'].eval()
    select_mask =[]
    IDH_p_label =[]
    IDH_n_label = []
    patients =[]
    idh_class = []
    confidence = []
    uncertainty = []
    idh_truth = []
    y_pred = []
    y_truth = []
    idh_pred_class= []
    if args.uncertainty:
        f_pass = 10
        enable_dropout(DDP_model['en'])
        enable_dropout(DDP_model['seg'])
        enable_dropout(DDP_model['idh'])
    else:
        f_pass =1

    with torch.no_grad():
        for batch_idx,(x,grade,idh) in enumerate(data_loader):
            x = x.cuda(non_blocking=True)
            # target = target.cuda(args.local_rank,non_blocking=True)
            idh = idh.cuda(non_blocking=True)

            out_prob = []
            for _ in range(f_pass):
                encoder_outs = DDP_model['en'](x)
                # seg_out = DDP_model['seg'](encoder_outs[0], encoder_outs[1], encoder_outs[2], encoder_outs[4])
                idh_out = DDP_model['idh'](encoder_outs[3], encoder_outs[4])
                idh_pred = F.softmax(idh_out, 1)  ## for selecting positive pseudo-labels
                out_prob.append(idh_pred)
            out_prob = torch.stack(out_prob)
            out_std = torch.std(out_prob,dim=0)
            out_prob = torch.mean(out_prob,dim=0)
            class_pres = torch.argmax(out_prob)

            max_value,max_idx = torch.max(out_prob,dim=1)
            max_std = out_std.gather(1,max_idx.view(-1,1))

            #selecting positive pseduo-labels

            if args.uncertainty:
                selected_idx = (max_value>=args.tau_p)*(max_std.squeeze(1)<args.kappa_p)

            # print('selected_idx:',selected_idx,"class_pres:",class_pres.item())
            if selected_idx and class_pres.item()==1:
                label = 1
                IDH_p_label.append(class_pres.item())
            elif selected_idx and class_pres.item()==0:
                label = 0
                IDH_n_label.append(class_pres.item())
            else:
                label = -1

            if selected_idx:
                y_pred.append(out_prob[0].cpu().numpy())
                y_truth.append(idh.item())
                idh_pred_class.append(class_pres.item())


            idh_class.append(label)
            patients.append(names[batch_idx])
            confidence.append(max_value.item())
            uncertainty.append(max_std.item())
            print("name:", names[batch_idx], "selected_idx:", selected_idx, "max_value:", max_value.item(), "uncertainty:",
                  max_std.item(), "pred_idh:", class_pres.item(), "idh:", idh.item())
            idh_truth.append(idh.item())
    print("1:", len(IDH_p_label), '0:', len(IDH_n_label))
    ece_scpre = ECE_score(y_pred,y_truth)
    acc = accuracy_score(y_truth,idh_pred_class)
    print(confusion_matrix(y_truth,idh_pred_class))
    print("kappa_p:",args.kappa_p,"ece_scpre:",ece_scpre,"acc:",acc)
    ## class blance for IDH

    negative_index = np.array(idh_class)==0

    negative_idh = np.array(idh_class)[negative_index].tolist()
    postivate_idh = np.array(idh_class)[~negative_index].tolist()

    negative_patient = np.array(patients)[negative_index].tolist()
    postivate_patient = np.array(patients)[~negative_index].tolist()

    postivate_u = np.array(uncertainty)[~negative_index].tolist()
    negative_u = np.array(uncertainty)[negative_index].tolist()

    postivate_c = np.array(confidence)[~negative_index].tolist()
    negative_c = np.array(confidence)[negative_index].tolist()

    # n_selectd_idx = np.argsort(np.array(negative_u))[int(len(IDH_p_label)*1.6):] # select the

    # for inx in n_selectd_idx:
    #     negative_idh[inx] = -1

    postivate_patient.extend(negative_patient)
    postivate_c.extend(negative_c)
    postivate_u.extend(negative_u)
    postivate_idh.extend(negative_idh)

    import pandas as pd
    data_frame = pd.DataFrame({'id':postivate_patient,'confidence':postivate_c,'uncertainty':postivate_u,'idh_class':postivate_idh})
    data_frame.to_csv('data/UPS_test_data_{}_{}.csv'.format(args.kappa_p,args.tau_p))
    # data_frame.to_csv('data/addition_data_{}.csv'.format(str(itr+1)))

    import numpy as np

def pseudo_labeling_1p19q(args,data_loader,DDP_model,names,itr):
    import numpy as np
    from sklearn.metrics import accuracy_score,confusion_matrix
    DDP_model['en'].eval()
    DDP_model['seg'].eval()
    DDP_model['idh'].eval()
    select_mask =[]
    IDH_0_label =[]
    IDH_1_label = []
    IDH_2_label = []
    patients =[]
    idh_class = []
    confidence = []
    uncertainty = []
    idh_truth = []
    y_pred = []
    y_truth = []
    idh_pred_class= []
    if args.uncertainty:
        f_pass = 10
        enable_dropout(DDP_model['en'])
        enable_dropout(DDP_model['seg'])
        enable_dropout(DDP_model['idh'])
    else:
        f_pass =1

    with torch.no_grad():
        for batch_idx,(x,grade,idh) in enumerate(data_loader):
            x = x.cuda(non_blocking=True)
            # target = target.cuda(args.local_rank,non_blocking=True)
            idh = idh.cuda(non_blocking=True)

            out_prob = []
            for _ in range(f_pass):
                encoder_outs = DDP_model['en'](x)
                # seg_out = DDP_model['seg'](encoder_outs[0], encoder_outs[1], encoder_outs[2], encoder_outs[4])
                idh_out = DDP_model['idh'](encoder_outs[3], encoder_outs[4])
                idh_pred = F.softmax(idh_out, 1)  ## for selecting positive pseudo-labels
                out_prob.append(idh_pred)
            out_prob = torch.stack(out_prob)
            out_std = torch.std(out_prob,dim=0)
            out_prob = torch.mean(out_prob,dim=0)
            class_pres = torch.argmax(out_prob)

            max_value,max_idx = torch.max(out_prob,dim=1)
            max_std = out_std.gather(1,max_idx.view(-1,1))

            #selecting positive pseduo-labels

            if args.uncertainty:
                selected_idx = (max_value>=args.tau_p)*(max_std.squeeze(1)<args.kappa_p)

            # print('selected_idx:',selected_idx,"class_pres:",class_pres.item())
            if selected_idx and class_pres.item()==1:
                label = 1
                IDH_1_label.append(class_pres.item())
            elif selected_idx and class_pres.item()==0:
                label = 0
                IDH_0_label.append(class_pres.item())
            elif selected_idx and class_pres.item() == 2:
                label = 2
                IDH_2_label.append(class_pres.item())
            else:
                label = -1

            if selected_idx:
                y_pred.append(out_prob[0].cpu().numpy())
                y_truth.append(idh.item())
                idh_pred_class.append(class_pres.item())

            idh_class.append(label)
            patients.append(names[batch_idx])
            confidence.append(max_value.item())
            uncertainty.append(max_std.item())
            print("name:", names[batch_idx], "selected_idx:", selected_idx, "max_value:", max_value.item(), "uncertainty:",
                  max_std.item(), "pred_idh:", class_pres.item(), "idh:", idh.item())
            idh_truth.append(idh.item())
    print("1:", len(IDH_1_label), '0:', len(IDH_0_label),'2:', len(IDH_2_label))
    # ece_scpre = ECE_score(y_pred,y_truth)
    # acc = accuracy_score(y_truth,idh_pred_class)
    # print(confusion_matrix(y_truth,idh_pred_class))
    # print("kappa_p:",args.kappa_p,"ece_scpre:",ece_scpre,"acc:",acc)
    ## class blance for IDH

    negative_index = np.array(idh_class)==0

    negative_idh = np.array(idh_class)[negative_index].tolist()
    postivate_idh = np.array(idh_class)[~negative_index].tolist()

    negative_patient = np.array(patients)[negative_index].tolist()
    postivate_patient = np.array(patients)[~negative_index].tolist()

    postivate_u = np.array(uncertainty)[~negative_index].tolist()
    negative_u = np.array(uncertainty)[negative_index].tolist()

    postivate_c = np.array(confidence)[~negative_index].tolist()
    negative_c = np.array(confidence)[negative_index].tolist()

    # n_selectd_idx = np.argsort(np.array(negative_u))[int(len(IDH_p_label)*1.6):] # select the

    # for inx in n_selectd_idx:
    #     negative_idh[inx] = -1

    postivate_patient.extend(negative_patient)
    postivate_c.extend(negative_c)
    postivate_u.extend(negative_u)
    postivate_idh.extend(negative_idh)

    import pandas as pd
    data_frame = pd.DataFrame({'id':postivate_patient,'confidence':postivate_c,'uncertainty':postivate_u,'idh_class':postivate_idh})
    data_frame.to_csv('data/UPS_1p19q_test_data_{}_{}.csv'.format(args.kappa_p,args.tau_p))
    # data_frame.to_csv('data/addition_data_{}.csv'.format(str(itr+1)))


def ECE_score(y_pred, y_truth, n_bins=10):
    y_pred = np.array(y_pred)
    y_truth = np.array(y_truth)

    if y_truth.ndim > 1:
        y_truth = np.argmax(y_truth, axis=1)
    py_index = np.argmax(y_pred, axis=1)

    py_value = []
    for i in range(y_pred.shape[0]):
        py_value.append(y_pred[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(y_pred.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_truth[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)

def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)

