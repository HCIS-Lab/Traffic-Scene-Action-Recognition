import torch
from torch import nn
import numpy as np

import sys
sys.path.append('/data/hanku/Interaction-benchmark/models')

import cnnfc
import vivit
import cnn_gru
import cnn_convgru
import res3d
import i3d
import i3d_kinetics
import x3d
import csn
import mvit
import slowfast
import slot_video
from retrieval_head import Head
# def generate_deeplab(model_name):
# 	if model_name == 'deeplab':
# 		model = deeplab.ResNet()
# 	backbone_params = (
#     list(model.conv1.parameters()) +
#     list(model.bn1.parameters()) +
#     list(model.layer1.parameters()) +
#     list(model.layer2.parameters()) +
#     list(model.layer3.parameters()) +
#     list(model.layer4.parameters()))
# 	last_params = list(model.aspp.parameters())

# 	for t in backbone_params:
# 	  t.requires_grad=False
# 	for t in last_params:
# 		t.requires_grad=True

# 	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# 	params = sum([np.prod(p.size()) for p in model_parameters])
# 	print ('Total trainable parameters: ', params)

# 	return backbone_params, last_params, model

# def generate_model(model_name, num_cam, num_ego_class, num_actor_class, seq_len, road):
# 	if model_name == 'cnnlstm_maskformer':
# 		model = cnnlstm_backbone.CNNLSTM_maskformer(num_cam, num_ego_class, num_actor_class, road)
# 	elif model_name == 'cat_cnnlstm':
# 		model = cat_cnnlstm.Cat_CNNLSTM_maskformer(num_cam, num_ego_class, num_actor_class, road)
# 	elif model_name == 'convlstm':
# 		model = convlstm_seg.CONVLSTM_Seg(num_cam, num_ego_class, num_actor_class, road)
# 	elif model_name == 'cnnlstm':
# 		model = cnnlstm_seg.CNNLSTM_Seg(num_cam, num_ego_class, num_actor_class, road)
# 	for param in model.parameters():
# 	    param.requires_grad = True


# 	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# 	params = sum([np.prod(p.size()) for p in model_parameters])
# 	print ('Total trainable parameters: ', params)

# 	return model

def load_pretrained_model(model, num_ego_class, num_actor_class, \
	pretrain_path='/media/hankung/ssd/retrieval/models/r3d50_K_200ep.pth'):
    if pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, map_location='cpu')

        model.load_state_dict(pretrain['state_dict'], strict=False)
        tmp_model = model

        tmp_model.head = Head(tmp_model.head_in_c,
                                     num_ego_class, num_actor_class)

    return model

def generate_model(args, model_name, num_ego_class, num_actor_class, seq_len, road, use_backbone):
	# if model_name == 'cnnlstm':
	# 	model = full_cnnlstm_seg.FULL_CNNLSTM_Seg(num_ego_class, num_actor_class, road)
	# 	for t in model.backbone.parameters():
	#   		t.requires_grad=False
	if model_name == 'cnngru':
		model = cnn_gru.CNNGRU(num_ego_class, num_actor_class, road, use_backbone)
		for t in model.parameters():
			t.requires_grad=True

	if model_name == 'cnn_convgru':
		model = cnn_convgru.CNN_CONVGRU(num_ego_class, num_actor_class, road)
		for t in model.parameters():
			t.requires_grad=True

	elif model_name == 'cnnfc':
		model = cnnfc.CNNFC(num_ego_class, num_actor_class, road, seq_len, use_backbone)
		if use_backbone:
			for t in model.backbone.parameters():
		  		t.requires_grad=False
			# for t in model.backbone.backbone.res5.parameters():
			# 	t.requires_grad=True
			# for t in model.backbone.backbone.res4.parameters():
			# 	t.requires_grad=True
	elif model_name == 'vivit':
		model = vivit.ViViT((256, 768), 16, seq_len, num_ego_class, num_actor_class)
	elif model_name == '3dres':
		model = res3d.ResNet3D(num_ego_class=num_ego_class, num_actor_class=num_actor_class)
		model = load_pretrained_model(model, num_ego_class, num_actor_class)
		for t in model.parameters():
	  		t.requires_grad=False
		for t in model.head.parameters():
			t.requires_grad=True
		for t in model.layer4.parameters():
			t.requires_grad=True
	elif model_name =='i3d':
		model = i3d.InceptionI3d(num_ego_class, num_actor_class, in_channels=3)
		model.load_state_dict(torch.load('/media/hankung/ssd/retrieval/models/rgb_charades.pt'), strict=False)
		model.replace_logits()
		for t in model.parameters():
	  		t.requires_grad=False
		for t in model.end_points['Mixed_3b'].parameters():
			t.requires_grad=True
		for t in model.end_points['Mixed_3c'].parameters():
			t.requires_grad=True
		for t in model.end_points['Mixed_4b'].parameters():
			t.requires_grad=True
		for t in model.end_points['Mixed_4c'].parameters():
			t.requires_grad=True
		for t in model.end_points['Mixed_4d'].parameters():
			t.requires_grad=True
		for t in model.end_points['Mixed_4e'].parameters():
			t.requires_grad=True
		for t in model.end_points['Mixed_4f'].parameters():
			t.requires_grad=True
		for t in model.end_points['Mixed_5b'].parameters():
			t.requires_grad=True
		for t in model.end_points['Mixed_5b'].parameters():
			t.requires_grad=True
		for t in model.end_points['Mixed_5c'].parameters():
			t.requires_grad=True
		for t in model.logits.parameters():
			t.requires_grad=True
	elif model_name == 'i3d_kinetics':
		model = i3d_kinetics.I3D_KINETICS(num_ego_class, num_actor_class)
		for t in model.model.parameters():
	  		t.requires_grad=False
		for t in model.model.blocks[-1].parameters():
			t.requires_grad=True
		for t in model.model.blocks[-2].parameters():
			t.requires_grad=True
		for t in model.model.blocks[-3].parameters():
			t.requires_grad=True

	elif model_name == 'x3d':
		model = x3d.X3D(num_ego_class, num_actor_class)
		for t in model.model.parameters():
	  		t.requires_grad=False
		for t in model.model.blocks[-1].parameters():
			t.requires_grad=True
		for t in model.model.blocks[-2].parameters():
			t.requires_grad=True
		for t in model.model.blocks[-3].parameters():
			t.requires_grad=True

	elif model_name == 'csn':
		model = csn.CSN(num_ego_class, num_actor_class)
		for t in model.model.parameters():
	  		t.requires_grad=False
		for t in model.model.blocks[-1].parameters():
			t.requires_grad=True
		for t in model.model.blocks[-2].parameters():
			t.requires_grad=True
		for t in model.model.blocks[-3].parameters():
			t.requires_grad=True

	elif model_name == 'slowfast':
		model = slowfast.SlowFast(num_ego_class, num_actor_class)
		for t in model.parameters():
	  		t.requires_grad=False
		for t in model.model.blocks[-1].parameters():
			t.requires_grad=True
		for t in model.model.blocks[-2].parameters():
			t.requires_grad=True
		for t in model.model.blocks[-3].parameters():
			t.requires_grad=True

	elif model_name == 'mvit':
		model = mvit.MViT(num_ego_class, num_actor_class)
		for t in model.parameters():
	  		t.requires_grad=False
		for t in model.head.parameters():
			t.requires_grad=True
		for t in model.model.cls_positional_encoding.parameters():
			t.requires_grad=True
		for t in model.model.patch_embed.parameters():
			t.requires_grad=True
		for t in model.model.blocks[1].parameters():
			t.requires_grad=True
		for t in model.model.blocks[2].parameters():
			t.requires_grad=True
		for t in model.model.blocks[-1].parameters():
			t.requires_grad=True
		for t in model.model.blocks[-2].parameters():
			t.requires_grad=True
		for t in model.model.blocks[-3].parameters():
			t.requires_grad=True
		# for t in model.model.blocks[-4].parameters():
		# 	t.requires_grad=True
		# for t in model.model.blocks[-5].parameters():
		# 	t.requires_grad=True

	elif model_name == 'slot':
		model = slot_video.SLOT_VIDEO(args, num_ego_class, num_actor_class, args.num_slots)
		for t in model.parameters():
			t.requires_grad=True
		for t in model.resnet.parameters():
			t.requires_grad=False
		for t in model.resnet[-1].parameters():
			t.requires_grad=True
		# for t in model.resnet[-2].parameters():
		# 	t.requires_grad=True

	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print ('Total trainable parameters: ', params)
	return model 