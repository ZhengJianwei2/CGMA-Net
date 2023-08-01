import argparse
import numpy as np
from glob import glob
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
import cv2

from lib.CGMA.model import CGMA

class Dataset(torch.utils.data.Dataset):

    def __init__(self, img_paths, mask_paths, aug=True, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug
        self.img_transform = transforms.Compose([
            transforms.Resize((352, 352)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

        self.gt_transform = transforms.Compose([
            # transforms.Resize((352, 352)),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        image = self.img_transform(image)
        mask = self.gt_transform(mask)
        return np.asarray(image), np.asarray(mask), img_path


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


def get_scores(gts, prs, path):
    mean_precision = 0
    mean_recall = 0
    mean_iou = 0
    mean_dice = 0
    for gt, pr, pt in zip(gts, prs, path):
        mean_precision += precision_np(gt, pr)
        mean_recall += recall_np(gt, pr)
        mean_iou += iou_np(gt, pr)
        dice = dice_np(gt, pr)
        mean_dice += dice
        # print(dice)
        # if dice < 0.6:
        #     print(pt)

    mean_precision /= len(gts)
    mean_recall /= len(gts)
    mean_iou /= len(gts)
    mean_dice /= len(gts)

    print("scores: dice={}, miou={}, precision={}, recall={}".format(mean_dice, mean_iou, mean_precision, mean_recall))

    return (mean_iou, mean_dice, mean_precision, mean_recall)


def inference(model, args, save_path=None):
    print("#" * 20)

    X_test = glob('{}/images/*'.format(args.test_path))
    X_test.sort()
    y_test = glob('{}/masks/*'.format(args.test_path))
    y_test.sort()

    test_dataset = Dataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    gts = []
    prs = []
    names = []
    results = []
    for i, pack in enumerate(test_loader, start=1):
        image, gt, _ = pack
        gt = gt[0][0]
        gt = np.asarray(gt, np.float32)
        image = image.cuda()

        out = model(image)
        res = F.upsample(out[0], size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()

        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        results.append(res)
        pr = res.round()
        gts.append(gt)
        prs.append(pr)
        names.append(_)

    get_scores(gts, prs, names)

    # if save_path:
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     for img, name in zip(prs, names):
    #         save_name = save_path + '/' + name[0].split('\\')[-1]
    #         img = img*255
    #         cv2.imwrite(save_name, img)



if __name__ == '__main__':
    test_path = "../dataset/kvasir-seg/test"
    # test_path = "../dataset/CVC-ClinicDB/test"
    # test_path = "../dataset/CVC-ClinicDB"
    # test_path = "../dataset/kvasir-seg"
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str,
                        default=test_path, help='path to dataset')
    args = parser.parse_args()

    model = PolypPVT()
    model.load_state_dict(torch.load('./model/sg/sg_fff_kvasir_32.pth'))

    # model = UNet(3,1)
    # model.load_state_dict(torch.load('./model/unet/kvasir.pth'))
    # model = UNet()
    # model.load_state_dict(torch.load('./model/unet_plus/clinic.pth'))
    #
    # model = EncoderBlock()
    # model.load_state_dict(torch.load('./model/msrf/clinic.pth'))

    #
    # model = models.FCBFormer()
    # model.load_state_dict(torch.load('./model/FCB/FCB_clinic_no_aug.pth'))
    # model = build(model_name="mit_PLD_b4", class_num=1)  # ssformer
    # model.load_state_dict(torch.load('./model/ssfomer/ssformer_l_clinic.pth'))
    # model.load_state_dict(torch.load('./model/ssfomer/ssformer_clinic_no_aug.pth'))
    # model = ESFPNetStructure()
    # model.load_state_dict(torch.load('./model/ESFP/ESFP_L_kvasir.pth'))
    # model = PraNet().cuda()
    # model.load_state_dict(torch.load('./model/pranet/kvasir.pth'))

    # model = HarDMSEG().cuda()
    # model.load_state_dict(torch.load('./model/hard/kvasir.pth'))

    model.cuda()
    model.eval()

    inference(model, args, 'result_map/hard/kvasir')
