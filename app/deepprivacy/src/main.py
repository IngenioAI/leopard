# IMPORT PACKAGES
import argparse
import fractions
import glob
import json
import os
import pickle
import shutil

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm, trange

from emotion_classification import test_emotion_class
from insightface_func.face_detect_crop_multi import Face_detect_crop
from models.models import create_model
from options.test_options import TestOptions
from parsing_model.model import BiSeNet
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from util.reverse2original import reverse2wholeimage

model_path = '/model'


def config():
    opt = TestOptions().parse()
    opt.no_simswaplogo = True
    opt.checkpoints_dir = f'{model_path}/checkpoints'
    opt.Arc_path = f'{model_path}/arcface_model/arcface_checkpoint.tar'
    crop_size = opt.crop_size
    torch.nn.Module.dump_patches = True
    if crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'

    logoclass = None
    model = create_model(opt)
    model.eval()
    mse = torch.nn.MSELoss().to(device)
    spNorm = SpecificNorm()

    app = Face_detect_crop(name='antelope', root=f'{model_path}/insightface_func/models')
    app.prepare(ctx_id=0, det_thresh=0.3, det_size=(640, 640), mode=mode)
    return app, opt, crop_size, model, mse, spNorm, logoclass


device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

male_emb_dir = [f'{model_path}/embeddings/m/']
female_emb_dir = [f'{model_path}/embeddings/w/']
target_emb_list = ['onlyGen']  # onlyGen, 'aihub', 'cel']  # ,'oldaihub']
num = 2
simswap, opt, crop_size, model, mse, spNorm, logoclass = config()
model = model.to(device)


# HELPER FUNCTION
def lcm(a, b):
    return abs(a * b) / fractions.gcd(a, b) if a and b else 0


def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)


def _toarctensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)


def compute_embedding_distance(asain_face_emb, model, mse, specific_person_id_nonorm, index, target_id_list):
    align_crop_tensor_arcnorm = asain_face_emb[target_id_list[index]].to(device)
    align_crop_tensor_arcnorm_downsample = F.interpolate(align_crop_tensor_arcnorm, size=(112, 112))
    align_crop_id_nonorm = model.netArc(align_crop_tensor_arcnorm_downsample)
    return mse(align_crop_id_nonorm, specific_person_id_nonorm).detach().cpu().numpy()


def process_batch(crops):
    results = [process_crop(crop) for crop in crops]
    return results


def process_crop(spNorm, model, mse, b_align_crop, specific_person_id_nonorm):
    # print(b_align_crop.shape)
    b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop, cv2.COLOR_BGR2RGB))[None, ...].to(device)
    b_align_crop_tenor_arcnorm = spNorm(b_align_crop_tenor)
    b_align_crop_tenor_arcnorm_downsample = F.interpolate(b_align_crop_tenor_arcnorm, size=(112, 112))
    b_align_crop_id_nonorm = model.netArc(b_align_crop_tenor_arcnorm_downsample.to(device))
    mse_value = mse(b_align_crop_id_nonorm, specific_person_id_nonorm).detach().cpu().numpy()
    return mse_value, b_align_crop_tenor


def get_embedding_specific_person(image, simswap, crop_size, model):
    transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    specific_person_whole = image
    try:
        print(specific_person_whole.shape)
    except AttributeError:
        return
    specific_person_align_crop, _ = simswap.get(specific_person_whole, crop_size)
    specific_person_align_crop_pil = Image.fromarray(cv2.cvtColor(specific_person_align_crop[0], cv2.COLOR_BGR2RGB))
    specific_person = transformer_Arcface(specific_person_align_crop_pil)
    specific_person = specific_person.view(-1, specific_person.shape[0], specific_person.shape[1],
                                           specific_person.shape[2])
    specific_person = specific_person.to(device)
    specific_person_downsample = F.interpolate(specific_person, size=(112, 112))
    specific_person_id_nonorm = model.netArc(specific_person_downsample)
    specific_person_id_norm = F.normalize(specific_person_id_nonorm, p=2, dim=1)
    return specific_person_align_crop, specific_person_id_nonorm


def target_hu_inwhole(img_pic_whole, simswap, crop_size, spNorm, model, opt, mse, nonorm):
    # img_pic_whole = cv2.imread(img_pic_whole_path)
    img_pic_align_crop_list, mat_list = simswap.get(img_pic_whole, crop_size)
    swap_result_list = []
    self_id_compare_values = []
    b_align_crop_tenor_list = []
    for b_align_crop in img_pic_align_crop_list:
        b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop, cv2.COLOR_BGR2RGB))[None, ...].to(device)
        b_align_crop_tenor_arcnorm = spNorm(b_align_crop_tenor)
        b_align_crop_tenor_arcnorm_downsample = F.interpolate(b_align_crop_tenor_arcnorm, size=(112, 112))
        b_align_crop_id_nonorm = model.netArc(b_align_crop_tenor_arcnorm_downsample.to(device))

        self_id_compare_values.append(mse(b_align_crop_id_nonorm, nonorm).detach().cpu().numpy())
        b_align_crop_tenor_list.append(b_align_crop_tenor)
    self_id_compare_values_array = np.array(self_id_compare_values)  # 비슷한지 확인해서???
    self_min_index = np.argmin(self_id_compare_values_array)  # 제일 작은게 그 사람이다???
    self_min_value = self_id_compare_values_array[self_min_index]

    if self_min_value < opt.id_thres:
        return b_align_crop_tenor_list[self_min_index], mat_list, self_min_index
    else:
        return None


def swap_deepfake(iscf, target_id_list, target_index, img_pic_whole, image_name, target_emb_name,
                  specific_person_align_crop, asain_face_emb, target_hu_align_crop_tensor, mat_list, self_min_index,
                  model, opt, crop_size, spNorm, logoclass, nonorm, net):
    if iscf:  # closest인 경우
        n = 'closest'
    else:
        n = 'furthest'

    output_file_name = f"deepfake_result/{image_name}_{target_emb_name}_{n}_{target_index}.jpg"
    print("Output File Name:", output_file_name)

    swap_result_list = []
    b_align_crop_tenor_list = []
    for b_align_crop in specific_person_align_crop:
        swap_result, b_align_crop_tenor = process_crop(spNorm, model, mse, b_align_crop, nonorm)
        swap_result_list.append(swap_result)
        b_align_crop_tenor_list.append(b_align_crop_tenor)
    print("@#@", len(asain_face_emb), len(target_id_list), target_id_list, target_index)
    target_img_id = asain_face_emb[target_id_list[target_index]].to(device)
    target_img_id_downsample = F.interpolate(target_img_id, size=(112, 112))
    latend_id = model.netArc(target_img_id_downsample)
    latend_id = F.normalize(latend_id, p=2, dim=1)

    swap_result = model(None, target_hu_align_crop_tensor, latend_id, None, True)[0]
    return reverse2wholeimage(target_hu_align_crop_tensor, [swap_result], [mat_list[self_min_index]], crop_size,
                              img_pic_whole, logoclass,
                              os.path.join(opt.output_path, output_file_name), opt.no_simswaplogo, pasring_model=net,
                              use_mask=opt.use_mask, norm=spNorm)


def get_emb(asain_face_emb_dir, target='onlyGen'):
    asain_face_emb = {}
    if target == 'onlyGen':
        for emb_path in tqdm(glob.glob(asain_face_emb_dir[0] + '*')):
            print("EMB load:", emb_path)
            try:
                emb_ = np.load(emb_path, allow_pickle=True).item()
            except AttributeError:
                with open(file=emb_path, mode='rb') as f:
                    emb_ = pickle.load(f)
            if len(asain_face_emb.keys()) == 0:
                asain_face_emb = emb_
            else:
                asain_face_emb.update(emb_)
    return asain_face_emb


# CONFIGURE SIMSWAP


# ROUTING

def get_emb_idx(target_id_list, asain_face_emb, id_nonorm, if_demo=False):
    if if_demo:
        return [1060, 1854, 2444, 2555, 5748]

    # FIND INDICES
    print("start emb")
    id_compare_values_list = []
    for i in trange(0, len(target_id_list)):
        id_compare_values_list.append(
            compute_embedding_distance(asain_face_emb, model, mse, id_nonorm, i, target_id_list))
    id_compare_values_array = np.array(id_compare_values_list)
    print(id_compare_values_array[:3])

    closest_value = np.sort(id_compare_values_array)[:5]

    closest_idx = [np.where(id_compare_values_array == closest_value[ii])[0][0] for ii in range(len(closest_value))]

    return closest_idx


def generate(args):
    with open(f"/data/input/{args.input}", "rt", encoding="UTF-8") as fp:
        input_json = json.load(fp)

    img_path = input_json['image_path']
    specific_gender = input_json.get('sex', 'M')
    is_demo = input_json.get('is_demo', False)
    print(is_demo)

    if img_path == "demo_sample?":
        shutil.copyfile("sample/one2.jpg", "/data/input/demo_sample.jpg")
        img_path = "/data/input/demo_sample.jpg"
    else:
        img_path = os.path.join("/data/input", img_path)
    image = Image.open(img_path)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #cv2.imwrite('test.png', image)

    original_emotion = ''

    results = dict()
    results['outputs'] = []
    align_crop, id_nonorm = get_embedding_specific_person(image, simswap, crop_size, model)

    for target_emb in target_emb_list:

        # FIND INDICES
        print("Target Embbeding:", target_emb)
        if specific_gender == 'W':
            asain_face_emb = get_emb(female_emb_dir, target_emb)
        else:
            asain_face_emb = get_emb(male_emb_dir, target_emb)

        target_id_list = list(asain_face_emb.keys())
        print("Number of Target Embedding Image: ", len(target_id_list))

        closest_idx = get_emb_idx(target_id_list, asain_face_emb, id_nonorm, is_demo)

        print(f"Closet person ID:", closest_idx)

        # SWAP
        if opt.use_mask:
            n_classes = 19
            net = BiSeNet(n_classes=n_classes)
            net.to(device)
            save_pth = os.path.join(f'{model_path}/parsing_model/checkpoint', '79999_iter.pth')
            net.load_state_dict(torch.load(save_pth))
            net.eval()
        else:
            net = None

        target_hu_align_crop_tensor, mat_list, self_min_index = target_hu_inwhole(image, simswap, crop_size, spNorm,
                                                                                  model, opt, mse, id_nonorm)
        if target_hu_align_crop_tensor is None:
            print('The person you specified is not found on the picture: {}'.format(image))
            continue

        for target_index in closest_idx:
            result = swap_deepfake(True, target_id_list, target_index, image, 'result', target_emb, align_crop,
                                   asain_face_emb, target_hu_align_crop_tensor, mat_list, self_min_index, model,
                                   opt, crop_size, spNorm, logoclass, id_nonorm, net)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            file = dict()

            output_file_name = f"result_{target_emb}_{target_index}.jpg"
            result_img = Image.fromarray(result)
            result_img.save(os.path.join("/data/output", output_file_name), format='PNG')
            original, faked = test_emotion_class(image, result_img)
            original_emotion = original
            file['filename'] = output_file_name
            file['emotion'] = faked
            results['outputs'].append(file)

    results['original_image_path'] = img_path
    results['original_image_emotion'] = original_emotion

    with open(f"/data/output/{args.output}", "w") as f:
        json.dump(results, f)

    print(results)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="params.json")
    parser.add_argument("--output", type=str, default="result.json")

    return parser.parse_args()


if __name__ == '__main__':
    generate(parse_arguments())
