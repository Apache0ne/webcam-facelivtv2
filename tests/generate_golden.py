#!/usr/bin/env python3
"""Golden-reference generator. NOT executed when this package was created."""
import argparse
import copy
import os
import sys
import numpy as np
import torch

ap=argparse.ArgumentParser()
ap.add_argument("--repo",required=True,help="path to official FaceLiVT repo root")
ap.add_argument("--checkpoint",required=True)
ap.add_argument("--raw-bgr",required=True,help="exact 112*112*3 BGR bytes")
ap.add_argument("--out",required=True)
a=ap.parse_args()
sys.path.insert(0,os.path.abspath(a.repo))
from backbones import get_model,reparameterize

raw=np.fromfile(a.raw_bgr,dtype=np.uint8)
if raw.size != 112*112*3: raise SystemExit("raw input must be 112x112x3")
bgr=raw.reshape(112,112,3)
rgb=bgr[:,:,::-1].copy()
x=torch.from_numpy(rgb.transpose(2,0,1)).unsqueeze(0).float()
x.div_(255).sub_(0.5).div_(0.5)

sd=torch.load(a.checkpoint,map_location="cpu")
if isinstance(sd,dict) and "state_dict" in sd: sd=sd["state_dict"]
model=get_model("facelivtv2_l",dropout=0.0,fp16=False,num_features=512)
model.load_state_dict(sd); model.eval()
with torch.no_grad(): y=model(x).float().cpu().numpy()
model2=copy.deepcopy(model)
model2=reparameterize(model2).eval()
with torch.no_grad(): yr=model2(x).float().cpu().numpy()
yn=y/np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-20)
yrn=yr/np.maximum(np.linalg.norm(yr,axis=1,keepdims=True),1e-20)
np.savez(a.out,input_bgr=bgr,fp32=y,reparam_fp32=yr,fp32_norm=yn,reparam_fp32_norm=yrn)
print(a.out)
