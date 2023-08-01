from torch.autograd import Variable
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import argparse
from datetime import datetime

from dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr
import torch.nn.functional as F
import numpy as np
import logging


from lib.CGMA.model import CGMA
import matplotlib.pyplot as plt

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


epsilon = 1e-8
def recall_np(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + epsilon)
    return recall


def precision_np(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return precision


def dice_np(y_true, y_pred):
    precision = precision_np(y_true, y_pred)
    recall = recall_np(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + epsilon))


def iou_np(y_true, y_pred):
    intersection = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + epsilon)

def test(model, path, dataset):
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC = 0.0
    mean_precision = 0
    mean_recall = 0
    mean_iou = 0
    mean_dice = 0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        out = model(image)
        # eval Dice
        res = F.upsample(out[0], size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        pr = np.reshape(input, (-1))
        gt = np.reshape(target, (-1))
        mean_precision += precision_np(gt, pr)
        mean_recall += recall_np(gt, pr)
        mean_iou += iou_np(gt, pr)
        dice = dice_np(gt, pr)
        mean_dice += dice

    return mean_precision/num1, mean_recall/num1, mean_iou/num1, mean_dice/num1


def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    global best
    size_rates = [1]
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            out = model(images)
            # ---- loss function ----
            total_loss = structure_loss(out[0], gts)


            if i % 20 == 0 or i == total_step:
                print(f'{datetime.now()} Epoch[{epoch}/{opt.epoch}], step[{i}/{total_step}] loss0:{total_loss:.4f}',
                      end="")
                # 将张量转换为 numpy 数组
                input_np = images[0].detach().cpu().numpy()
                target_np = gts[0][0].detach().cpu().numpy()
                output_np = out[0][0][0].detach().cpu().numpy()

                # 创建一个包含 3 个子图的画布
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))

                # 在第一个子图中绘制输入图像
                img = input_np.transpose(1, 2, 0)
                axs[0].set_title('Input')
                axs[0].imshow(img)  # 将 CxHxW 转换为 HxWxC

                # 在第二个子图中绘制真实值
                axs[1].imshow(target_np, cmap='gray')
                axs[1].set_title('Target')

                # 在第三个子图中绘制模型输出
                axs[2].imshow(output_np, cmap='gray')
                axs[2].set_title('Output')

                # 显示图像
                plt.show()


            for _, _pred in enumerate(out):
                if _ == 0:
                    continue
                loss = structure_loss(_pred, gts)
                total_loss += loss

                if i % 20 == 0 or i == total_step:
                    print(f' | loss{_}:{loss:.4f}', end="")

            if i % 20 == 0 or i == total_step:
                print()

            # ---- backward ----
            total_loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

    # save model 
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # choose the best model

    global dict_plot

    if (epoch + 1) % 1 == 0 and epoch >= 0:
        meandice = test(model, test_path, 'val')
        meandice = meandice[3]
        print(f'\n\nmean dice: {meandice}\nbest val dice: {best}\n\n')
        logging.info(
            f'epoch{epoch}:{meandice}')
        # torch.save(model.state_dict(), save_path + f'{epoch}.pth')
        if meandice > best:
            best = meandice
            torch.save(model.state_dict(), save_path + 'CGMA.pth')
            print('##############################################################################best_on_val',
                  best)
        test_res = test(model, test_path, 'test')
        logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, 'test', test_res[3]))
        print(f'pre: {test_res[0]}, recall: {test_res[1]}, iou: {test_res[2]}, dice: {test_res[3]}', )


if __name__ == '__main__':
    dataset_name = 'kvasir-seg'
    model_name = f'CGMA_{dataset_name}'
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=50, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default="False", help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='dataset/'+dataset_name+'/train',
                        help='path to train dataset')


    parser.add_argument('--test_path', type=str,
                        default='dataset/'+dataset_name,
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='./model/' + model_name + '/')

    opt = parser.parse_args()
    logging.basicConfig(filename='../train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = CGMA().cuda()


    total = sum([param.nelement() for param in model.parameters()])
    print('Number of parameter: %.2fM' % (total / 1e6))

    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)
    edge_root = '{}/edges/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, model, optimizer, epoch, opt.test_path)

    # plot the eval.png in the training stage
    # plot_train(dict_plot, name)
