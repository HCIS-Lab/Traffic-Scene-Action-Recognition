import os
import json
from PIL import Image

import numpy as np
import torch 
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
import json 
import random
from tool import get_rot
import torchvision.transforms as transforms
# from torchvideotransforms import video_transforms, volume_transforms


class Video_Data(Dataset):

    def __init__(self, 
                seq_len=8, 
                training=True,
                viz=False,
                seg=False,
                ped='all',
                num_cam=1,
                num_class=12,
                scale=2.0,
                model_name='i3d',
                num_slots=21,
                root='/data/carla_dataset/data_collection'):
        root = '/media/hankung/ssd/carla_13/CARLA_0.9.13/PythonAPI/examples/data_collection'
        self.training = training
        self.scale=scale
        self.seg = seg
        self.ped = ped
        self.viz = viz
        self.model_name = model_name
        self.id = []
        self.variants = []

        self.front_list = []
        self.seg_front_list = []

        self.gt_ego = []
        self.gt_actor = []
        self.slot_eval_gt = []
        self.num_cam = num_cam
        self.step = []
        self.start_idx = []
        self.num_class = num_class
        self.seq_len = seq_len


        type_list = ['interactive', 'non-interactive', 'ap_Town01', 
        'ap_Town02','ap_Town03', 'ap_Town04', 'ap_Town05', 'ap_Town06', 'ap_Town07', 'ap_Town10HD']
        n=0
        save_actor = []
        actor_stat_table = {'z1-z2': 0, 'z1-z3':0, 'z1-z4':0,
                    'z2-z1': 0, 'z2-z3': 0, 'z2-z4': 0,
                    'z3-z1': 0, 'z3-z2': 0, 'z3-z4': 0,
                    'z4-z1': 0, 'z4-z2': 0, 'z4-z3': 0}

        ped_stat = {'c1-c2': 0, 'c1-c4': 0, 
                        'c2-c1': 0, 'c2-c3': 0, 
                        'c3-c2': 0, 'c3-c4': 0, 
                        'c4-c1': 0, 'c4-c3': 0 
                        }

        ego_stat = {'None': 0,'e:z1-z2': 0, 'e:z1-z3':0, 'e:z1-z4': 0}

        for t, type in enumerate(type_list):
            basic_scenarios = [os.path.join(root, type, s) for s in os.listdir(os.path.join(root, type))]

            # iterate scenarios
            print('searching data')
            for s in tqdm(basic_scenarios, file=sys.stdout):
                # a basic scenario
                scenario_id = s.split('/')[-1]
                # if 1:
                # if training and scenario_id.split('_')[0] == '3' \
                #     or training and scenario_id.split('_')[0] == '5' \
                #     or not training and scenario_id.split('_')[0] == '10':
                if training:
                    if type == 'interactive' or type == 'non-interactive':
                        if scenario_id.split('_')[0] == '10':
                            continue
                    else:

                        if type == 'ap_Town10HD':
                            continue
                else:
                    if type == 'interactive' or type == 'non-interactive':
                        if scenario_id.split('_')[0] != '10':
                            continue

                    else:
                        if type != 'ap_Town10HD':
                            continue

                # if ((training and scenario_id.split('_')[0] != '10') or
                #     (training and type != 'ap_Town10HD')) \
                # or ((not training and scenario_id.split('_')[0] == '10') or 
                #     (not training and type =='ap_Town10HD'))\
                # :
                # if training or not training and scenario_id.split('_')[0] == '10':
                variants_path = os.path.join(s, 'variant_scenario')
                if os.path.isdir(variants_path):
                    variants = [os.path.join(variants_path, v) for v in os.listdir(variants_path)]
                    
                    for v in variants:
                        # print(v)
                        v_id = v.split('/')[-1]
                        if 'DS' in v_id:
                            continue
                        # if training:
                        #     if int(v_id)%6 == 0:
                        #         continue
                        # else:
                        #     if int(v_id)%6 !=0:
                        #         continue
                        if os.path.isfile(v+'/retrieve_gt.txt'):
                            with open(v+'/retrieve_gt.txt') as f:
                                gt = []

                                for line in f:
                                    line = line.replace('\n', '')
                                    if line != '\n':
                                        gt.append(line)
                                gt = list(set(gt))
                        else:
                            continue
                        if 'slot' in model_name:
                            gt_ego, gt_actor, eval_gt, actor_stat_table, ped_stat, ego_stat = get_multi_class(gt, scenario_id, v_id, self.num_class, actor_stat_table, ped_stat, ego_stat, ped=ped, model_name=model_name, num_slots=num_slots)
                        else:
                            gt_ego, gt_actor, actor_stat_table, ped_stat, ego_stat = get_multi_class(gt, scenario_id, v_id, self.num_class, actor_stat_table, ped_stat, ego_stat, ped=ped, model_name=model_name, num_slots=num_slots)
                        if gt_ego == None:
                            continue
                        if not torch.count_nonzero(gt_actor):
                            continue
                        # if torch.count_nonzero(gt_actor) < 3:
                        #     continue

                        if os.path.isdir(v+"/rgb/front/"):
                            check_data = [v+"/rgb/front/"+ img for img in os.listdir(v+"/rgb/front/") if os.path.isfile(v+"/rgb/front/"+ img)]
                            check_data.sort()
                        else:
                            continue
                        if len(check_data) < 50:
                            continue

                        fronts = []
                        segs_f = []
                        
                        start_frame = int(check_data[0].split('/')[-1].split('.')[0])
                        end_frame = int(check_data[-1].split('/')[-1].split('.')[0])
                        num_frame = end_frame - start_frame + 1

                        step = num_frame // seq_len

                        max_num = 50
                        # step = ((end_frame-start + 1) // (seq_len+1)) -1
                        for m in range(max_num):
                            start = start_frame + m
                            # step = ((end_frame-start + 1) // (seq_len+1)) -1
                            if start_frame + (seq_len-1)*step > end_frame:
                                break
                            front_temp = []
                            seg_f_temp = []
                            for i in range(start, end_frame+1, step):
                                imgname = f"{str(i).zfill(8)}.jpg"
                                if os.path.isfile(v+"/rgb/front/"+imgname):
                                    front_temp.append(v+"/rgb/front/"+imgname)
                                if self.seg:
                                    if len(front_temp) == seq_len and len(seg_f_temp) == seq_len:
                                        break
                                else:
                                    if len(front_temp) == seq_len:
                                        break

                            if len(front_temp) == seq_len:
                                fronts.append(front_temp)
                                if len(seg_f_temp) == seq_len:
                                    segs_f.append(seg_f_temp)

                        if len(fronts) == 0:
                            continue
                        if self.seg and len(segs_f) == 0:
                            continue


                        self.id.append(s.split('/')[-1])
                        self.variants.append(v.split('/')[-1])
                        self.front_list.append(fronts)
                        if self.seg:
                            self.seg_front_list.append(segs_f)
                        self.gt_ego.append(gt_ego)
                        self.gt_actor.append(gt_actor)
                        self.slot_eval_gt.append(eval_gt)
                        save_actor.append(gt_actor)
        print('num_variant: ' + str(len(self.variants)))
        print('actor_stat:')
        print(actor_stat_table)
        print('ped_stat')
        print(ped_stat)
        print('ego_stat')
        print(ego_stat)
        out = [0]*self.num_class
        out = torch.FloatTensor(out)       
        # for a in save_actor:
        #     out = out + a
        # np.save(str(training)+'.npy', out)
        

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.front_list)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['fronts'] = []
        data['seg_front'] = []
        
        data['ego'] = self.gt_ego[index]
        data['actor'] = self.gt_actor[index]
        data['id'] = self.id[index]
        data['variants'] = self.variants[index]

        if 'slot' in self.model_name:
            data['slot_eval_gt'] = self.slot_eval_gt[index]

        if self.training:
            sample_idx = random.randint(0, len(self.front_list[index])-1)
        else:
            sample_idx = len(self.front_list[index])//2

        if self.viz:
            data['img_front'] = []

        seq_fronts = self.front_list[index][sample_idx]
        if self.seg:
            seq_seg_front = self.seg_front_list[index][sample_idx]


        ######################################
        for i in range(self.seq_len):
            x = Image.open(seq_fronts[i]).convert('RGB')
            x = scale(x, self.scale, self.model_name)
            # height, width = x.shape[:2]
            # print(height)
            # x = {"image": x, "height": height, "width": width}
            data['fronts'].append(x)
            # data['fronts'].append(np.float32(np.array(scale(Image.open(seq_fronts[i]).convert('RGB'))))/255)
            if self.seg:
                data['seg_front'].append(torch.from_numpy(np.load(seq_seg_front[i])))
            # if self.viz:
            #     data['img_front'].append(np.float32(np.array(scale(Image.open(seq_fronts[i]).convert('RGB'))))/255)
            #     data['img_front'].append(np.float32(np.array(Image.open(seq_fronts[i]).convert('RGB')))/255)
        # data['fronts'] = aug(data['fronts'])
        data['fronts'] = to_np(data['fronts'], self.model_name)
        return data

    
def scale_and_crop_image(image, scale=2.0, crop=256):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    # image = Image.open(filename)
    (width, height) = (int(image.width // scale), int(image.height // scale))

    im_resized = image.resize((width, height))
    image = np.asarray(im_resized)
    # start_x = height//2 - crop//2
    # start_y = width//2 - crop//2
    # cropped_image = image[start_x:start_x+crop, start_y:start_y+crop]
    # cropped_image = np.transpose(cropped_image, (2,0,1))
    cropped_image = np.transpose(image, (2,0,1))

    return cropped_image


def scale(image, scale=2.0, model_name=None):

    if model_name == 'mvit':
        (width, height) = (224, 224)
    else:
        (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height), Image.ANTIALIAS)

    

    # image = np.asarray(im_resized)
    # image = np.transpose(image, (2,0,1))
    return im_resized

def aug(x):
    sometimes = lambda aug: va.Sometimes(0.5, aug)
    seq = va.Sequential([
    sometimes(va.RandomRotate(degrees=(10,10))), # randomly rotates the video with a degree randomly choosen from [-10, 10]  
    # sometimes(va.GaussianBlur(2)),
    sometimes(va.TemporalElasticTransformation())
    # # sometimes(va.RandomShear(10,10))
    ])
    x = seq(x)

    # video_transform_list = [video_transforms.RandomRotation(20),
    #                         video_transforms.ColorJitter(5,5,5,0.2),
    #                         volume_transforms.ClipToTensor()]
    # transforms = video_transforms.Compose(video_transform_list)
    # x = transforms(x)
    return x

def to_np(v, model_name):
    for i, _ in enumerate(v):

        # transform = transforms.Compose([
        #                         transforms.ToTensor()])
        if model_name == 'x3d' or model_name == 'slowfast' \
        or model_name == 'mvit' or model_name == 'i3d_kinetics' \
        or model_name =='csn' or 'slot' in model_name:
            transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])])
            transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        v[i] = transform(v[i])
    return v

def get_multi_class(gt_list, s_id, v_id, num_class, actor_stat_table, ped_stat, ego_stat, ped='all', model_name='', num_slots=21):   
    road_type = {'i-': 0, 't1': 1, "t2": 2, "t3": 3, 's-': 4, 'r-': 5, 'i': 0, 't': 0}

    # ego_table = {'e:z1-z1': 0, 'e:z1-z2': 1, 'e:z1-z3':2, 'e:z1-z4': 3,
    #                 'e:s-s': 4, 'e:s-sl': 5, 'e:s-sr': 6,
    #                 'e:ri-r1': 7, 'e:r1-r2': 8, 'e:r1-ro':9}

    ego_table = {'e:z1-z2': 1, 'e:z1-z3':2, 'e:z1-z4': 3}

    # actor_table = {'c:z1-z1': 0, 'c:z1-z2': 1, 'c:z1-z3':2, 'c:z1-z4':3,
    #                 'c:z2-z1': 4, 'c:z2-z2':5, 'c:z2-z3': 6, 'c:z2-z4': 7,
    #                 'c:z3-z1': 8, 'c:z3-z2': 9, 'c:z3-z3': 10, 'c:z3-z4': 11,
    #                 'c:z4-z1': 12, 'c:z4-z2': 13, 'c:z4-z3': 14, 'c:z4-z4': 15,

    #                 'b:z1-z1': 16, 'b:z1-z2': 17, 'b:z1-z3': 18, 'b:z1-z4': 19,
    #                 'b:z2-z1': 20, 'b:z2-z2':21, 'b:z2-z3': 22, 'b:z2-z4': 23,
    #                 'b:z3-z1': 24, 'b:z3-z2': 25, 'b:z3-z3': 26, 'b:z3-z4': 27,
    #                 'b:z4-z1': 28, 'b:z4-z2': 29, 'b:z4-z3': 30, 'b:z4-z4': 31,

    #                 'c:s-s': 32, 'c:s-sl': 33, 'c:s-sr': 34,
    #                 'c:sl-s': 35, 'c:sl-sl': 36,
    #                 'c:sr-s': 37, 'c:sr-sr': 38,
    #                 'c:jl-jr': 39, 'c:jr-jl': 40,

    #                 'b:s-s': 41, 'b:s-sl': 42, 'b:s-sr': 43,
    #                 'b:sl-s': 44, 'b:sl-sl': 45,
    #                 'b:sr-s': 46, 'b:sr-sr': 47,
    #                 'b:jl-jr': 48, 'b:jr-jl': 49,

    #                 'p:c1-c2': 50, 'p:c1-c4': 51, 'p:c1-cl': 52,  'p:c1-cf': 53, 
    #                 'p:c2-c1': 54, 'p:c2-c3': 55, 'p:c2-cl': 56,
    #                 'p:c3-c2': 57, 'p:c3-c4': 58, 'p:c3-cr': 59,
    #                 'p:c4-c1': 60, 'p:c4-c3': 61, 'p:c4-cr': 62, 'p:c4-cf': 63,
    #                 'p:cf-c1': 64, 'p:cf-c4': 65,
    #                 'p:cl-c1': 66, 'p:cl-c2': 67,
    #                 'p:cr-c3': 68, 'p:cr-c4': 69,
    #                 'p:jl-jr': 70, 'p:jr-jl': 71,

    #                 'c:ri-r1': 72, 'c:rl-r1': 73, 'c:r1-r2': 74, 'c:r1-ro': 75, 'c:ri-r2': 76, 'c:r1-r2': 77,
    #                 'b:ri-r1': 78, 'b:rl-r1': 79, 'b:r1-r2': 80, 'b:r1-ro': 81, 'b:ri-r2': 82, 'b:r1-r2': 83}


    # actor_table = {'z1-z1': 0, 'z1-z2': 1, 'z1-z3':2, 'z1-z4':3,
    #                 'z2-z1': 4, 'z2-z2':5, 'z2-z3': 6, 'z2-z4': 7,
    #                 'z3-z1': 8, 'z3-z2': 9, 'z3-z3': 10, 'z3-z4': 11,
    #                 'z4-z1': 12, 'z4-z2': 13, 'z4-z3': 14, 'z4-z4': 15,

    #                 'c1-c2': 16, 'c1-c4': 17, 'c1-cl': 18,  'c1-cf': 19, 
    #                 'c2-c1': 20, 'c2-c3': 21, 'c2-cl': 22,
    #                 'c3-c2': 23, 'c3-c4': 24, 'c3-cr': 25,
    #                 'c4-c1': 26, 'c4-c3': 27, 'c4-cr': 28, 'c4-cf': 29,
    #                 'cf-c1': 30, 'cf-c4': 31,
    #                 'cl-c1': 32, 'cl-c2': 33,
    #                 'cr-c3': 34, 'cr-c4': 35}

    # intersection only
    if ped == 'no_ped':
         # actor_table = {'z1-z1': 0, 'z1-z2': 1, 'z1-z3':2, 'z1-z4':3,
         #            'z2-z1': 4, 'z2-z2':5, 'z2-z3': 6, 'z2-z4': 7,
         #            'z3-z1': 8, 'z3-z2': 9, 'z3-z3': 10, 'z3-z4': 11,
         #            'z4-z1': 12, 'z4-z2': 13, 'z4-z3': 14, 'z4-z4': 15}
        actor_table = {'z1-z2': 0, 'z1-z3':1, 'z1-z4':2,
                    'z2-z1': 3, 'z2-z3': 4, 'z2-z4': 5,
                    'z3-z1': 6, 'z3-z2': 7, 'z3-z4': 8,
                    'z4-z1': 9, 'z4-z2': 10, 'z4-z3': 11}
    elif ped == 'ped_only':
        actor_table = {'c1-c2': 0, 'c1-c4': 1, 
                        'c2-c1': 2, 'c2-c3': 3, 
                        'c3-c2': 4, 'c3-c4': 5, 
                        'c4-c1': 6, 'c4-c3': 7 
                        }
    else:
        actor_table = { 'z1-z2': 0, 'z1-z3':1, 'z1-z4':2,
                        'z2-z1': 3, 'z2-z3': 4, 'z2-z4': 5,
                        'z3-z1': 6, 'z3-z2': 7, 'z3-z4': 8,
                        'z4-z1': 9, 'z4-z2': 10, 'z4-z3': 11,

                        'c1-c2': 12, 'c1-c4': 13, 
                        'c2-c1': 14, 'c2-c3': 15, 
                        'c3-c2': 16, 'c3-c4': 17, 
                        'c4-c1': 18, 'c4-c3': 19 
                        }


    if '_' in s_id:
        road_class = s_id.split('_')[1][:2]
    else:
        road_class = s_id[0]
    road_class = road_type[road_class]

    # z1, z2, z3, z4, 0~3
    # c1, c2, c3, c4, cf, cl, cr, 4~10
    # s, sl, sr, jl, jr, 11~15
    # r1, r2, ri, ro 16~19


    if road_class == 0:
        road_para = [1,1,1,1,
                    1,1,1,1,0,0,0,
                    ]
        # return

    elif road_class == 1:
        road_para = [1,1,0,1,
                    1,0,0,1,1,0,0,
                    ]
        # return

    elif road_class == 2:
        road_para = [1,1,1,0,
                    1,1,0,0,0,1,0,
                    ]
        # return

    elif road_class == 3:
        road_para = [1,0,1,1,
                    0,0,1,1,0,0,1,
                    ]
        # return

    elif road_class == 4:
        road_para = [0,0,0,0,
                    0,0,0,0,0,0,0,
                    1,1,1,1,1,
                    0,0,0,0]
        return

    elif road_class == 5:
        road_para = [0,0,0,0,
                    0,0,0,0,0,0,0,
                    0,0,0,0,0,
                    1,1,1,1]
        return
    road_para = torch.FloatTensor(road_para)

    ego_class = None
    actor_class = [0]*num_class

    slot_gt = []
    for gt in gt_list:
        gt = gt.lower()
        if gt[0] == 'p':
            gt = gt[2:]
            ped_stat[gt]+=1
            if ped == 'no_ped':
                continue
            else:
                if 'slot' in model_name:
                    if not actor_table[gt] in slot_gt:
                        slot_gt.append(actor_table[gt])
                # else:
                actor_class[actor_table[gt]] = 1
        elif gt[0] == 'e':
            ego_class = ego_table[gt]
            ego_stat[gt] +=1
        else:
            gt = gt[2:]
            if gt != 'ne':
                if not gt in actor_table.keys():
                    print(gt)
                    return
                else:
                    if 'slot' in model_name:
                        if not actor_table[gt] in slot_gt:
                            slot_gt.append(actor_table[gt])
                    # else:
                    actor_class[actor_table[gt]] = 1
                    if not ped == 'ped_only':
                        actor_stat_table[gt]+=1
    if ego_class == None:
        ego_class = 0
        ego_stat['None'] +=1
    ego_label = torch.tensor(ego_class)
    if 'slot' in model_name:
        while (len(slot_gt)!= num_slots):
            slot_gt.append(num_class-1)
        slot_gt = torch.LongTensor(slot_gt)
        actor_class = actor_class[:-1]
        actor_class = np.array(actor_class)
        return ego_label, slot_gt, actor_class, actor_stat_table, ped_stat, ego_stat
    else:
        actor_label = torch.FloatTensor(actor_class)
        return ego_label, actor_label, actor_stat_table, ped_stat, ego_stat



