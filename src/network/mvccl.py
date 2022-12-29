import sys
sys.path.append('/kaggle/input/timm-pytorch-image-models/pytorch-image-models-master')

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .chianets import eff_selection
except ImportError:
    from chianets import eff_selection

import numpy as np

'''
https://arxiv.org/pdf/2209.10478.pdf
[1] Multi-view Local Co-occurrence and Global Consistency Learning Improve Mammogram
Classification Generalisation
'''

class Backbone(nn.Module):
	def __init__(self, ):
		super(Backbone, self).__init__()
		self.register_buffer('mean', torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1,3,1,1))
		self.register_buffer('std',  torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1,3,1,1))
		self.encoder = eff_selection(model_type='v2_s')

	def forward(self, x):
		x = (x - self.mean) / self.std
		x = self.encoder(x)
		return x

class GlobalConsistency(nn.Module):
	def __init__(self,dim):
		super(GlobalConsistency, self).__init__()

		self.project = nn.Linear(dim,dim) #<todo> try mlp?

	def forward(self, u_m, u_a):
		B, C, H, W = u_m.shape

		g_m = F.adaptive_max_pool2d(u_m,1)
		g_a = F.adaptive_max_pool2d(u_a,1)
		g_m = torch.flatten(g_m, 1)
		g_a = torch.flatten(g_a, 1)

		p_a = self.project(g_m)
		p_m = self.project(g_a)

		return g_m, p_m, g_a, p_a

#----
#<todo>
#do we need norm?

class CrossAttention(nn.Module):
	def __init__(self, dim, num_head=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
		super(CrossAttention, self).__init__()

		assert dim % num_head == 0, 'dim should be divisible by num_heads'
		self.num_head = num_head
		head_dim = dim // num_head
		self.scale = head_dim ** (-0.5)

		self.qkv_a = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.qkv_m = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.proj_a = nn.Linear(dim, dim)
		self.proj_m = nn.Linear(dim, dim)

		self.attn_drop = nn.Dropout(attn_drop)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, u_m, u_a):
		B,L,dim = u_m.shape

		qkv_m = self.qkv_m(u_m)
		qkv_m = qkv_m.reshape(B, L, 3, self.num_head, dim // self.num_head).permute(2, 0, 3, 1, 4)
		q_m, k_m, v_m = qkv_m.unbind(0)

		qkv_a = self.qkv_m(u_a)
		qkv_a = qkv_a.reshape(B, L, 3, self.num_head, dim // self.num_head).permute(2, 0, 3, 1, 4)
		q_a, k_a, v_a = qkv_a.unbind(0)

		attn_m = (q_m @ k_a.transpose(-2, -1)) * self.scale
		attn_m = attn_m.softmax(dim=-1)
		attn_m = self.attn_drop(attn_m)

		attn_a = (q_a @ k_m.transpose(-2, -1)) * self.scale
		attn_a = attn_a.softmax(dim=-1)
		attn_a = self.attn_drop(attn_a)


		x_m = (attn_m @ v_m).transpose(1, 2).reshape(B, L, dim)
		x_m = self.proj_m(x_m)
		x_m = self.proj_drop(x_m)

		x_a = (attn_a @ v_a).transpose(1, 2).reshape(B, L, dim)
		x_a = self.proj_a(x_a)
		x_a = self.proj_drop(x_a)

		return  x_m, x_a

class LocalCoccurrence(nn.Module):
	def __init__(self,dim, num_head=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
		super(LocalCoccurrence, self).__init__()

		self.norm1 = nn.LayerNorm(dim)
		self.attn  = CrossAttention(dim, num_head, qkv_bias, attn_drop, proj_drop)

	def forward(self, u_m, u_a):
		B,C,H,W = u_m.shape
		L = H*W
		dim = C

		u_m = u_m.reshape(B,dim,L).permute(0,2,1)
		u_a = u_a.reshape(B,dim,L).permute(0,2,1)

		x_m = self.norm1(u_m)
		x_a = self.norm1(u_a)
		x_m, x_a = self.attn(x_m, x_a)

		gap_m = x_m.mean(1)
		gap_a = x_a.mean(1)
		return gap_m, gap_a

class Net(nn.Module):
	def load_pretrain(self, ):
		return

	def __init__(self,):
		super(Net, self).__init__()
		self.output_type = ['inference', 'loss']

		self.backbone = Backbone()
		dim = 1280

		self.lc  = LocalCoccurrence(dim)
		self.gl  = GlobalConsistency(dim)
		self.mlp = nn.Sequential(
			nn.LayerNorm(dim*3),
			nn.Linear(dim*3, dim),
			nn.GELU(),
			nn.Linear(dim, dim),
		)#<todo> mlp needs to be deep if backbone is strong?
		self.cancer = nn.Linear(dim,1)

	def forward(self, batch):
		x = batch['image']
		batch_size,num_view,C,H,W = x.shape
		x = x.reshape(-1, C, H, W)

		u = self.backbone(x)
		_,c,h,w = u.shape

		u = u.reshape(batch_size,num_view,c,h,w)
		u_m = u[:,0]
		u_a = u[:,1]
		gap_m, gap_a = self.lc(u_m, u_a)

		g_m, p_m, g_a, p_a = self.gl(u_m, u_a)
		gp_m = g_m + p_m

		last = torch.cat([gp_m, gap_m, gap_a ],-1)
		last = self.mlp(last)
		cancer = self.cancer(last).reshape(-1)


		output = {}
		if  'loss' in self.output_type:
			output['cancer_loss'] = F.binary_cross_entropy_with_logits(cancer, batch['cancer'])
			output['global_loss'] = criterion_global_consistency(g_m, p_m, g_a, p_a)


		if 'inference' in self.output_type:
			output['cancer'] = torch.sigmoid(cancer)

		return output

def similiarity(x1,x2):
	p12 = (x1*x2).sum(-1)
	p1 = torch.sqrt((x1*x1).sum(-1))
	p2 = torch.sqrt((x2*x2).sum(-1))
	s = p12/(p1*p2+1e-6)
	return s

def criterion_global_consistency(g_m, p_m, g_a, p_a):
	loss =  -0.5*(similiarity(g_m, p_m)+similiarity(g_a, p_a))
	loss = loss.mean()
	return loss


def run_check_net():

	h,w = 1536, 768 #about 0.50
	batch_size = 7

	# ---
	batch = {
		'image' : torch.from_numpy(np.random.uniform(0,1,(batch_size,2,1,h,w))).float(),#.cuda(),
		'cancer': torch.from_numpy(np.random.choice(2,(batch_size))).float(),#.cuda(),
	}
	#batch = {k: v.cuda() for k, v in batch.items()}

	net = Net()#.cuda()
	net.load_pretrain()

	with torch.no_grad():
		with torch.cuda.amp.autocast(enabled=True):
			output = net(batch)

	print('batch')
	for k, v in batch.items():
		if 'index' in k: continue
		print('%32s :' % k, v.shape)

	print('output')
	for k, v in output.items():
		if 'loss' not in k:
			print('%32s :' % k, v.shape)
	for k, v in output.items():
		if 'loss' in k:
			print('%32s :' % k, v.item())


# main #################################################################
if __name__ == '__main__':
	run_check_net()
