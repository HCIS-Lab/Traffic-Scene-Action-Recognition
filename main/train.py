import argparse
import json
import os
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn

torch.backends.cudnn.benchmark = True

import sys
# sys.path.append('/data/hanku/Interaction_benchmark/datasets')
sys.path.append('/media/hankung/ssd/retrieval/datasets')
sys.path.append('/media/hankung/ssd/retrieval/config')
sys.path.append('/media/hankung/ssd/retrieval/models')

# from .configs.config import GlobalConfig
import video_data
import feature_data

from sklearn.metrics import average_precision_score, precision_score, f1_score, recall_score, accuracy_score, hamming_loss
# from torchmetrics import F1Score

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image
from PIL import Image

# import cnnlstm_backbone
from model import generate_model
from utils import *
from torchvision import models
import matplotlib.image
from scipy.optimize import linear_sum_assignment
from torchviz import make_dot


torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='cnnlstm_imagenet', help='Unique experiment identifier.')

parser.add_argument('--inter_only', help="", action="store_true")
parser.add_argument('--seg', help="", action="store_true")
parser.add_argument('--use_backbone', help="", action="store_true")
parser.add_argument('--ped', type=str, help="")
parser.add_argument('--backbone', type=str, help="i3d-2")

parser.add_argument('--num_slots', type=int, default=21, help='')
parser.add_argument('--seq_len', type=int, default=16, help='')
parser.add_argument('--scale', type=float, default=4, help='')
parser.add_argument('--bce', type=float, default=1, help='')
parser.add_argument('--ego_weight', type=float, default=1, help='')
parser.add_argument('--weight', type=float, default=15, help='')
parser.add_argument('--ce_weight', type=float, default=0.05, help='')


parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--epochs', type=int, default=100, help='Number of train epochs.')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')
parser.add_argument('--val_every', type=int, default=5, help='Validation frequency (epochs).')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
parser.add_argument('--test', help="", action="store_true")
parser.add_argument('--viz', help="", action="store_true")

args = parser.parse_args()
args.logdir = os.path.join(args.logdir, args.id)
print(args)
writer = SummaryWriter(log_dir=args.logdir)

class Engine(object):
	"""Engine that runs training and inference.
	Args
		- cur_epoch (int): Current epoch.
		- print_every (int): How frequently (# batches) to print loss.
		- validate_every (int): How frequently (# epochs) to run validation.
		
	"""

	def __init__(self, args, cur_epoch=0, cur_iter=0, bce_weight=1, ego_weight=1):
		self.cur_epoch = cur_epoch
		self.cur_iter = cur_iter
		self.bestval_epoch = cur_epoch
		self.train_loss = []
		self.val_loss = []
		self.bestval = 1e10
		self.best_f1 = 1e-5
		self.bce_weight = bce_weight
		self.ego_weight = ego_weight
		self.args = args

	def train(self, model, optimizer, epoch, model_name='', ce_weight=5):
		loss_epoch = 0.
		ego_loss_epoch = 0.
		seg_loss_epoch = 0.
		actor_loss_epoch = 0.
		num_batches = 0
		correct_ego = 0
		total_ego = 0
		inter_meter = AverageMeter()
		union_meter = AverageMeter()

		label_actor_list = []
		pred_actor_list = []

		model.train()
		empty_weight = torch.ones(num_actor_class)*ce_weight
		empty_weight[-1] = 0.2
		slot_ce = nn.CrossEntropyLoss(reduction='mean', weight=empty_weight).cuda()
		# Train loop
		for data in tqdm(dataloader_train):
			# model =model.cuda()
			# efficiently zero gradients
			# for p in model.parameters():
			# 	p.grad = None

			# create batch and move to GPU
			fronts_in = data['fronts']
			if args.seg:
				seg_front_in = data['seg_front']

			inputs = []
			seg_front = []
			for i in range(seq_len):
				inputs.append(fronts_in[i].to(args.device, dtype=torch.float32))
				if args.seg:
					seg_front.append(seg_front_in[i].to(args.device, dtype=torch.long))
					
			# labels
			batch_size = inputs[0].shape[0]
			ego = data['ego'].cuda()
			if 'slot' in model_name:
				actor = data['actor'].to(args.device)
			else:
				actor = torch.FloatTensor(data['actor']).to(args.device)
			
			optimizer.zero_grad()
			if args.seg:
				h, w = seg_front[0].shape[-2], seg_front[0].shape[-1]
				seg_front = torch.stack(seg_front, 0)
				seg_front = torch.permute(seg_front, (1, 0, 2, 3)) #[batch, len, 1, h, w]
				seg_front = seg_front.contiguous().view(batch_size*seq_len, h, w)

				pred_ego, pred_actor, pred_seg = model(inputs)
			else:
				# pred_ego, pred_actor = model(inputs)
				pred_actor = model(inputs)


			pos_weight = torch.ones([num_actor_class])*args.weight
			# ego_ce = nn.CrossEntropyLoss(reduction='mean').cuda()
			seg_ce = nn.CrossEntropyLoss(reduction='mean').cuda()
			# ego_loss = ego_ce(pred_ego, ego)

			if 'slot' in model_name:
				bs, num_queries = pred_actor.shape[:2]
				out_prob = pred_actor.clone().detach().flatten(0, 1).softmax(-1)
				actor_gt_np = actor.clone().detach()
				tgt_ids = torch.cat([v for v in actor_gt_np.detach()])
				C = -out_prob[:, tgt_ids].clone().detach()
				C = C.view(bs, num_queries, -1).cpu()
				sizes = [len(v) for v in actor_gt_np]
				indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
				indx = [(torch.as_tensor(i, dtype=torch.int64).detach(), torch.as_tensor(j, dtype=torch.int64).detach()) for i, j in indices]
				batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indx)]).detach()
				src_idx = torch.cat([src for (src, _) in indx]).detach()
				idx = (batch_idx, src_idx)
				target_classes_o = torch.cat([t[J] for t, (_, J) in zip(actor, indx)]).cuda()
				target_classes = torch.full(pred_actor.shape[:2], num_actor_class-1,
				                    dtype=torch.int64, device=out_prob.device)
				target_classes[idx] = target_classes_o

				actor_loss = slot_ce(pred_actor.transpose(1, 2), target_classes)

			else:
				bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight).cuda()
				bce_loss = bce(pred_actor, actor)
			if args.seg:
				seg_loss = seg_ce(pred_seg, seg_front)
				loss = self.bce_weight * bce_loss + ce_loss
				# loss = ego_loss + self.bce_weight * bce_loss + seg_loss
			else:
				# loss = self.ego_weight*ego_loss + self.bce_weight*bce_loss
				loss = actor_loss 
			loss.backward(retain_graph = True)
			loss_epoch += float(loss.item())
			if args.seg:
				seg_loss_epoch +=float(seg_loss.item())
				_, pred_seg = torch.max(pred_seg, 1)
				pred_seg = pred_seg.data.cpu().numpy().squeeze().astype(np.uint8)
				seg_front = seg_front.cpu()
				mask = seg_front.numpy().astype(np.uint8)
				inter, union = inter_and_union(pred_seg, mask, 5)
				inter_meter.update(inter)
				union_meter.update(union)

			# _, pred_ego = torch.max(pred_ego.data, 1)
			# total_ego += ego.size(0)
			# correct_ego += (pred_ego == ego).sum().item()


			# if 'slot' in model_name:
			# 	pred_actor = torch.nn.functional.softmax(pred_actor, dim=-1)
			# 	_, pred_actor = torch.max(pred_actor.data, -1)
			# 	pred_actor = pred_actor.detach().cpu().numpy().astype(int)
			# 	batch_new_pred_actor = []
			# 	for i, b in enumerate(pred_actor):
			# 		new_pred = np.zeros(num_actor_class-1, dtype=int)
			# 		for pred in b:
			# 			if pred != num_actor_class-1:
			# 				new_pred[pred] = 1
			# 		batch_new_pred_actor.append(new_pred)
			# 	batch_new_pred_actor = np.array(batch_new_pred_actor)
			# 	pred_actor_list.append(batch_new_pred_actor)
			# 	label_actor_list.append(data['slot_eval_gt'])
			# else:
			# 	pred_actor = torch.sigmoid(pred_actor)
			# 	pred_actor = pred_actor > 0.5
			# 	pred_actor = pred_actor.float()
			# 	pred_actor_list.append(pred_actor.detach().cpu().numpy())
			# 	label_actor_list.append(actor.detach().cpu().numpy())

			actor_loss_epoch += float(actor_loss.item())
			# ego_loss_epoch += float(ego_loss.item())

			num_batches += 1
			optimizer.step()
			# writer.add_scalar('train_loss', loss.item(), self.cur_iter)

			self.cur_iter += 1
		# scheduler.step()

		# pred_actor_list = np.stack(pred_actor_list, axis=0)
		# label_actor_list = np.stack(label_actor_list, axis=0)

		# if 'slot' in model_name:
		# 	# pred_actor_list = pred_actor_list.reshape((pred_actor_list.shape[0]*args.batch_size*args.num_slots))
		# 	# label_actor_list = label_actor_list.reshape((label_actor_list.shape[0]*args.batch_size*args.num_slots))
		# 	pred_actor_list = pred_actor_list.reshape((pred_actor_list.shape[0]*args.batch_size, num_actor_class-1))
		# 	label_actor_list = label_actor_list.reshape((label_actor_list.shape[0]*args.batch_size, num_actor_class-1))
		# else:
		# 	pred_actor_list = pred_actor_list.reshape((pred_actor_list.shape[0]*args.batch_size, num_actor_class))
		# 	label_actor_list = label_actor_list.reshape((label_actor_list.shape[0]*args.batch_size, num_actor_class))



		# if 'slot' in model_name:
		# 	mean_f1 = f1_score(
		# 			pred_actor_list.astype('int64'),
		# 			label_actor_list.astype('int64'), 
		# 			average='samples',
		# 			labels=[i for i in range(num_actor_class-1)],
		# 			zero_division=0)
		# else:
		# 	mean_f1 = f1_score(
		# 			pred_actor_list.astype('int64'),
		# 			label_actor_list.astype('int64'), 
		# 			average='samples',
		# 			zero_division=0)

		loss_epoch = loss_epoch / num_batches
		if args.seg:
			seg_loss_epoch = seg_loss_epoch / (num_batches)
		actor_loss_epoch = actor_loss_epoch / num_batches
		# ego_loss_epoch = ego_loss_epoch / num_batches

		# print(f'acc of the ego: {correct_ego/total_ego}')

		print('total loss')
		print(loss_epoch)
		if args.seg:
			print('seg loss:')
			print(seg_loss_epoch)
		print('actor loss:')
		print(actor_loss_epoch)
		# print('ego loss:')
		# print(ego_loss_epoch)

		if args.seg:
			print('----------------------Road--------------------------------')
			iou = inter_meter.sum / (union_meter.sum + 1e-10)
			for i, val in enumerate(iou):
					print('IoU {0}: {1:.2f}'.format(i, val * 100))
			print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))

		# print(f'(train) f1 of the actor: {mean_f1}')
		# writer.add_scalar('train_f1', mean_f1, epoch)
		self.train_loss.append(loss_epoch)
		self.cur_epoch += 1

	def validate(self, model, dataloader, epoch, cam=False, model_name='', ce_weight=5):
		model.eval()
		if 'slot' in model_name:
			empty_weight = torch.ones(num_actor_class)*ce_weight
			empty_weight[-1] = 0.2
			slot_ce = nn.CrossEntropyLoss(reduction='mean', weight=empty_weight).cuda()
		with torch.no_grad():	
			num_batches = 0
			total_loss = 0.
			loss = 0.

			total_ego = 0
			total_actor = 0

			correct_ego = 0
			correct_actor = 0
			mean_f1 = 0
			label_actor_list = []
			pred_actor_list = []

			inter_meter = AverageMeter()
			union_meter = AverageMeter()

			for batch_num, data in enumerate(tqdm(dataloader)):
				id = data['id'][0]
				v = data['variants'][0]
				fronts_in = data['fronts']

				if args.seg:
					seg_front_in = data['seg_front']
				inputs = []
				seg_front = []

				for i in range(seq_len):

					inputs.append(fronts_in[i].to(args.device, dtype=torch.float32))
					if args.seg:
						seg_front.append(seg_front_in[i].to(args.device, dtype=torch.long))

				batch_size = inputs[0].shape[0]
				ego = data['ego'].to(args.device)
				if 'slot' in model_name:
					actor = data['actor'].to(args.device)
				else:
					actor = torch.FloatTensor(data['actor']).to(args.device)

				if args.seg:
					h, w = seg_front[0].shape[-2], seg_front[0].shape[-1]
					seg_front = torch.stack(seg_front, 0)
					seg_front = torch.permute(seg_front, (1, 0, 2, 3)) #[batch, len, 1, h, w]
					seg_front = seg_front.contiguous().view(batch_size*seq_len, h, w)
					pred_ego, pred_actor, pred_seg = model(inputs)
				else:
					# pred_ego, pred_actor = model(inputs)
					pred_actor = model(inputs)

				ego_ce = nn.CrossEntropyLoss(reduction='mean').cuda()
				seg_ce = nn.CrossEntropyLoss(reduction='mean').cuda()
				pos_weight = torch.ones([num_actor_class])*args.weight
				bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight).cuda()
				if 'slot' in model_name:
					bs, num_queries = pred_actor.shape[:2]
					out_prob = pred_actor.clone().detach().flatten(0, 1).softmax(-1)
					# out_prob = out_prob.cpu().numpy()

					actor_gt_np = actor.clone().detach()
					tgt_ids = torch.cat([v for v in actor_gt_np.detach()])
					C = -out_prob[:, tgt_ids].clone().detach()
					C = C.view(bs, num_queries, -1).cpu()
					sizes = [len(v) for v in actor_gt_np]
					indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
					indx = [(torch.as_tensor(i, dtype=torch.int64).detach(), torch.as_tensor(j, dtype=torch.int64).detach()) for i, j in indices]
					batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indx)]).detach()
					src_idx = torch.cat([src for (src, _) in indx]).detach()
					idx = (batch_idx, src_idx)
					target_classes_o = torch.cat([t[J] for t, (_, J) in zip(actor, indx)]).cuda()
					target_classes = torch.full(pred_actor.shape[:2], num_actor_class-1,
					                    dtype=torch.int64, device=out_prob.device)
					target_classes[idx] = target_classes_o
					actor_loss = slot_ce(pred_actor.transpose(1, 2), target_classes)


				else:
					bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight).cuda()
					bce_loss = bce(pred_actor, actor)
				if args.seg:
					seg_loss = seg_ce(pred_seg, seg_front)
					loss = self.bce_weight * bce_loss + ce_loss
					# loss = ego_loss + self.bce_weight * bce_loss + seg_loss
				else:
					# loss = self.ego_weight*ego_loss + self.bce_weight*bce_loss
					loss = actor_loss 
					# loss = ego_loss

				num_batches += 1


				total_loss += float(loss.item())
				# pred_ego = torch.nn.functional.softmax(pred_ego, dim=1)
				# _, pred_ego = torch.max(pred_ego.data, 1)

				if 'slot' in model_name:
					pred_actor = torch.nn.functional.softmax(pred_actor, dim=-1)
					_, pred_actor = torch.max(pred_actor.data, -1)
					pred_actor = pred_actor.detach().cpu().numpy().astype(int)
					batch_new_pred_actor = []
					for i, b in enumerate(pred_actor):
						new_pred = np.zeros(num_actor_class-1, dtype=int)
						for pred in b:
							if pred != num_actor_class-1:
								new_pred[pred] = 1
						batch_new_pred_actor.append(new_pred)
					batch_new_pred_actor = np.array(batch_new_pred_actor)
					pred_actor_list.append(batch_new_pred_actor)
					label_actor_list.append(data['slot_eval_gt'])
				else:
					pred_actor = torch.sigmoid(pred_actor)
					pred_actor = pred_actor > 0.5
					pred_actor = pred_actor.float()
					pred_actor_list.append(pred_actor.detach().cpu().numpy())
					label_actor_list.append(actor.detach().cpu().numpy())

				if args.seg:
					_, pred_seg = torch.max(pred_seg, 1)
					pred_seg = pred_seg.data.cpu().numpy()
					seg_front = seg_front.cpu().numpy().astype(np.uint8)
					for i in range(seq_len):
						pred_seg_temp = pred_seg[i]
						# print(pred_seg_temp.shape)
						pred_seg_temp = pred_seg_temp.squeeze().astype(np.uint8)
						seg_front_temp = seg_front[i]
						inter, union = inter_and_union(pred_seg_temp, seg_front_temp, 5)
						inter_meter.update(inter)
						union_meter.update(union)
						# if epoch == args.epochs -1:
						scene_name = os.path.join(args.logdir, id+'_'+v)
						if not os.path.isdir(scene_name):
							os.makedirs(scene_name)
						if not os.path.isdir(scene_name):
							os.makedirs(scene_name)

						matplotlib.image.imsave(os.path.join(scene_name, str(i))+'_pred'+'.png', pred_seg_temp)
						matplotlib.image.imsave(os.path.join(scene_name, str(i))+'_gt'+'.png', seg_front_temp)
			
			pred_actor_list = np.stack(pred_actor_list, axis=0)
			label_actor_list = np.stack(label_actor_list, axis=0)
			if 'slot' in model_name:
				# pred_actor_list = pred_actor_list.reshape((pred_actor_list.shape[0]*args.num_slots))
				# label_actor_list = label_actor_list.reshape((label_actor_list.shape[0]*args.num_slots))
				pred_actor_list = pred_actor_list.reshape((pred_actor_list.shape[0], num_actor_class-1))
				label_actor_list = label_actor_list.reshape((label_actor_list.shape[0], num_actor_class-1))
			else:
				pred_actor_list = pred_actor_list.reshape((pred_actor_list.shape[0]*args.batch_size, num_actor_class))
				label_actor_list = label_actor_list.reshape((label_actor_list.shape[0]*args.batch_size, num_actor_class))

				# label_actor_list.extend(actor.detach().cpu().numpy())
				# pred_actor_list.extend(pred_actor.detach().cpu().numpy())
				# total_ego += ego.size(0)
				# correct_ego += (pred_ego == ego).sum().item()

			pred_actor_list = np.array(pred_actor_list)
			label_actor_list = np.array(label_actor_list)



			if 'slot' in model_name:
				# mean_f1 = f1_score(
				# 	pred_actor_list.astype('int64'),
				# 	label_actor_list.astype('int64'), 
				# 	labels=[i for i in range(num_actor_class-1)],
				# 	average='micro',
				# 	zero_division=0)
				mean_f1 = f1_score(
						pred_actor_list.astype('int64'),
						label_actor_list.astype('int64'), 
						labels=[i for i in range(num_actor_class-1)],
						average='samples',
						zero_division=0)
			else:
				mean_f1 = f1_score(
						pred_actor_list.astype('int64'),
						label_actor_list.astype('int64'), 
						average='samples',
						zero_division=0)
			f1_class = f1_score(
					pred_actor_list.astype('int64'),
					label_actor_list.astype('int64'), 
					average=None,
					zero_division=0)

			print(f'(val) f1 of the actor: {mean_f1}')
			# writer.add_scalar('val_f1', mean_f1, epoch)
			# print(f'acc of the ego: {correct_ego/total_ego}')
			# writer.add_scalar('ego', correct_ego/total_ego, epoch)
			if mean_f1 > self.best_f1:
				self.best_f1 = mean_f1
			print(f'best f1 : {self.best_f1}')
			print('f1 vehicle:')
			print(f1_class[:12])
			print('f1 ped:')
			print(f1_class[12:])

			# np.save(os.path.join(args.logdir, str(args.road)+'actor_f1.npy'), mean_f1)


			if args.seg:
				print('----------------------Road--------------------------------')
				iou = inter_meter.sum / (union_meter.sum + 1e-10)
				for i, val in enumerate(iou):
 					print('IoU {0}: {1:.2f}'.format(i, val * 100))
				print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))

			total_loss = total_loss / float(num_batches)
			tqdm.write(f'Epoch {self.cur_epoch:03d}, Batch {batch_num:03d}:' + f' Loss: {total_loss:3.3f}')

			# writer.add_scalar('val_loss', total_loss, self.cur_epoch)
			
			self.val_loss.append(total_loss)



	def save(self):

		save_best = False
		if self.val_loss[-1] <= self.bestval:
			self.bestval = self.val_loss[-1]
			self.bestval_epoch = self.cur_epoch
			save_best = True
		
		# Create a dictionary of all data to save

		# log_table = {
		# 	'epoch': self.cur_epoch,
		# 	'iter': self.cur_iter,
		# 	'bestval': float(self.bestval.data),
		# 	'bestval_epoch': self.bestval_epoch,
		# 	'train_loss': self.train_loss,
		# 	'val_loss': self.val_loss,
		# }

		# Save ckpt for every epoch
		torch.save(model.state_dict(), os.path.join(args.logdir, 'model_%d.pth'%self.cur_epoch))

		# Save the recent model/optimizer states
		torch.save(model.state_dict(), os.path.join(args.logdir, 'model.pth'))
		torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'recent_optim.pth'))

		# Log other data corresponding to the recent model
		# with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
		# 	f.write(json.dumps(log_table))

		tqdm.write('====== Saved recent model ======>')
		
		if save_best:
			torch.save(model.state_dict(), os.path.join(args.logdir, 'best_model.pth'))
			torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'best_optim.pth'))
			tqdm.write('====== Overwrote best model ======>')

# Config
# config = GlobalConfig()
torch.cuda.empty_cache() 
seq_len = args.seq_len

num_ego_class = 4
if args.ped == 'no_ped':
	num_actor_class = 12
elif args.ped == 'ped_only':
	num_actor_class = 8
else:
	num_actor_class = 20

if 'slot' in args.id:
	num_actor_class +=1


# Data
if args.use_backbone:
	train_set = video_data.Video_Data(seq_len=seq_len, seg=args.seg, ped=args.ped, num_class=num_actor_class, scale=args.scale, model_name=args.id, num_slots=args.num_slots)
	val_set = video_data.Video_Data(seq_len=seq_len, training=False, viz=args.viz, seg=args.seg, ped=args.ped, num_class=num_actor_class, scale=args.scale, model_name=args.id, num_slots=args.num_slots)
else:
	train_set = feature_data.Feature_Data(seq_len=seq_len, seg=args.seg, ped=args.ped, num_class=num_actor_class, scale=args.scale)
	val_set = feature_data.Feature_Data(seq_len=seq_len, training=False, viz=args.viz, seg=args.seg, ped=args.ped, num_class=num_actor_class, scale=args.scale)
dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
dataloader_val = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
# dataloader_val_train = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
# Model
model = generate_model(args, args.id, num_ego_class, num_actor_class, args.seq_len, road=args.seg, use_backbone=args.use_backbone).cuda()

params = [{'params': model.parameters()}]
optimizer = optim.AdamW(params, lr=args.lr)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
trainer = Engine(args, bce_weight=args.bce, ego_weight=args.ego_weight)

# Create logdir
# if not os.path.isdir(args.logdir):
# 	os.makedirs(args.logdir)
# 	print('Created dir:', args.logdir)
# elif os.path.isfile(os.path.join(args.logdir, 'recent.log')):
# 	print('Loading checkpoint from ' + args.logdir)
# 	with open(os.path.join(args.logdir, 'recent.log'), 'r') as f:
# 		log_table = json.load(f)

# 	# Load variables
# 	trainer.cur_epoch = log_table['epoch']
# 	if 'iter' in log_table: trainer.cur_iter = log_table['iter']
# 	trainer.bestval = log_table['bestval']
# 	trainer.train_loss = log_table['train_loss']
# 	trainer.val_loss = log_table['val_loss']

# 	# Load checkpoint
# 	model.load_state_dict(torch.load(os.path.join(args.logdir, 'model.pth')))
# 	optimizer.load_state_dict(torch.load(os.path.join(args.logdir, 'recent_optim.pth')))

# Log args
# with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
# 	json.dump(args.__dict__, f, indent=2)
if not args.test:
	for epoch in range(trainer.cur_epoch, args.epochs): 
		trainer.train(model, optimizer, epoch, model_name=args.id, ce_weight=args.ce_weight)
		if epoch % args.val_every == 0 or epoch == args.epochs-1: 
				trainer.validate(model, dataloader_val, None, model_name=args.id, ce_weight=args.ce_weight)
				# trainer.validate(dataloader_val_train, None)
				trainer.save()
		if args.viz and epoch % 20 == 0:
				trainer.vizualize(cam)
else:
	trainer.validate(cam=cam)
