# Original DeepSCAN2020 code written and graciously provided by 
# Richard McKinley (Inselgrupe AG, Support Centre for Advanced Neuroimaging)

# Dockerhub for the original model code is scan/brats2020

# Lightly refactored for FeTS inference by Micah Sheller (Intel Corporation) 
# Refactoring changes were primarily 1) removing training code and 2) changing from script to module (e.g. global variables to member variables). 

import argparse
import os
import random
import copy
from collections import OrderedDict

import cv2
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.nn import ModuleList
from batchgenerators.augmentations.utils import pad_nd_image
from scipy.ndimage import label

from fets.data import get_appropriate_file_paths_from_subject_dir


# MSHELLER FIXME: Ensure parameters are being passed all the way through. Defaults could be being used erroneously.


class DeepSCANInferenceDataObject():
    def __init__(self, data_path='data', inference_patient=None, output_tag='deepscan'):
        self.data_path = data_path
        self.inference_patient = inference_patient
        self.output_tag = output_tag
    
    def iter_patients(self):
        patient_dirs = [p for p in os.scandir(self.data_path) if p.is_dir()]
        if self.inference_patient is not None:
            patient_dirs = [p for p in patient_dirs if p.name == self.inference_patient]

        for p in patient_dirs:
            allFiles = get_appropriate_file_paths_from_subject_dir(p.path)

            nifti_orig = nib.load(allFiles['FLAIR'])
            flair = np.copy(nifti_orig.get_fdata())
            t1 = np.copy(nib.load(allFiles['T1']).get_fdata())
            t2 = np.copy(nib.load(allFiles['T2']).get_fdata())
            t1ce = np.copy(nib.load(allFiles['T1CE']).get_fdata())
            
            gt = np.zeros_like(flair)

            val_subject_volume = np.stack([flair,t1,t2,t1ce,gt]).astype(np.int16)

            cropped, bbox = crop_to_nonzero(val_subject_volume)

            yield nifti_orig, cropped, bbox, val_subject_volume, flair, t1, p.name
    
    def save_nifti(self, data, pname):
        filename = f'{pname}_{self.output_tag}_seg.nii.gz'
        nib.save(data, os.path.join(self.data_path, pname, filename))


# FIXME: test that these intialization parameters are honored throughout
class DeepSCANBraTS2020Inference():
    def __init__(self, 
                 data,
                 *args,
                 use_attention=True,
                 model_device='cuda',
                 patch_size = (5,194,194),
                 target_label_sets=[[4], [1,4], [1,2,4]],
                 target_label_names=['enhancing', 'tumor_core', 'whole_tumor'],
                 save_intemediate=False,
                 axes=['axial','sagittal','coronal'],
                 pad_size = (192,192),
                 use_gaussian=True,
                 network_type='3D-to-2D',
                 native_model_weights_filepath=None,
                 **kwargs):
        self.data = data
        self.use_attention = use_attention
        self.device = model_device
        self.models = []
        self.patch_size = patch_size
        self.target_label_sets = target_label_sets
        self.target_label_names = target_label_names
        self.save_intemediate = save_intemediate
        self.axes = axes
        self.pad_size = pad_size
        self.use_gaussian = use_gaussian
        self.network_type = network_type

        if native_model_weights_filepath is not None:
            self.load_native(native_model_weights_filepath)

    def load_native(self, weights_dir):
        for fold in [0,1,2,3,4]:
            model = UNET_3D_to_2D(0,channels_in=4,channels=64, growth_rate =8, dilated_layers=[4,4,4,4], output_channels=len(self.target_label_names))
            weights_path = os.path.join(weights_dir, f'fold{fold}_retrained_nn.pth.tar')
            model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            model.to(self.device)
            self.models.append(model)
    
    def run_inference_and_store_results(self, output_file_tag=''):
        for nifti_orig, cropped, bbox, val_subject_volume, flair, t1, pname in self.data.iter_patients():
            self.predict_with_uncertainty(nifti_orig, cropped, bbox, val_subject_volume, flair, t1, pname)

    def predict_with_uncertainty(self, nifti_orig, cropped, bbox, val_subject_volume, flair, t1, pname):
        im_size = np.max(cropped.shape)

        print(f'cropped input image max dimension = {im_size}')

        if im_size > self.patch_size[1]:
            self.patch_size = (5, 2*((im_size+1)//2), 2*((im_size+1)//2))
            print(f'cropped image exceeds patch size: new patch size = {self.patch_size}')

        angles = [-45,0,45]

        logits, gradient_mask, target, target_logits, target_probs = self.apply_to_case_uncert_and_get_target_mask_probs_logits(cropped,
                                                                                                                                rot_angles=angles,
                                                                                                                                rot_axes = [0,1,2],
                                                                                                                                do_mirroring=True)

        target_probs_uncropped = np.zeros_like(np.stack([val_subject_volume[0]]*len(self.target_label_sets))).astype(np.float32)

        target_probs_uncropped[:, bbox[0][0]:bbox[0][1],
            bbox[1][0]:bbox[1][1],
            bbox[2][0]:bbox[2][1]] = target_probs

        nifti_affine = nifti_orig.affine
        all_nifti_100 = nib.Nifti1Image((target_probs_uncropped*100).transpose((1,2,3,0)).astype(np.uint8), nifti_affine)

        flip = self.mask_flip_probabilities(logits, gradient_mask)

        flip_uncropped = np.zeros_like(np.stack([val_subject_volume[0]]*len(self.target_label_sets))).astype(np.float32)

        flip_uncropped[:, bbox[0][0]:bbox[0][1],
            bbox[1][0]:bbox[1][1],
            bbox[2][0]:bbox[2][1]] = flip


        flip_nifti_100 = nib.Nifti1Image((flip_uncropped*100).transpose((1,2,3,0)).astype(np.uint8), nifti_affine)

        flip_prob_fusion = ((target_probs_uncropped>0.5)*(1-flip_uncropped) + (target_probs_uncropped<0.5)*(flip_uncropped))

        flip_prob_fusion_100 = nib.Nifti1Image((flip_prob_fusion*100).transpose((1,2,3,0)).astype(np.uint8), nifti_affine)

        seg = make_brats_segmentation(flip_prob_fusion*100)

        seg_nifti = nib.Nifti1Image((seg).astype(np.uint8), nifti_affine)

        seg_postprocessed, uncertainty = postprocess_brats_segmentation(seg, (flip_prob_fusion*100).transpose((1,2,3,0)), flair,t1)        

        postprocessed_nifti = nib.Nifti1Image((seg_postprocessed).astype(np.uint8), nifti_affine)


        # nib.save(postprocessed_nifti, f'{out_dir}/tumor_SCAN2020_class.nii.gz')
        self.data.save_nifti(postprocessed_nifti, pname)

        # for idx, name in zip([0,1,2],['enhance','core','whole']):
        #     unc_map = nib.Nifti1Image((uncertainty[:,:,:,idx]).astype(np.uint8), nifti_affine)
            # nib.save(unc_map, f'{out_dir}/tumor_SCAN2020_unc_{name}.nii.gz')

        return seg_postprocessed

    def apply_to_case_uncert_and_get_target_mask_probs_logits(self, subject, do_mirroring=False, rot_angles = [0],
                                                              rot_axes = [0]):
        
        subject_data, subject_metadata = load_subject(subject, axis='sagittal')       
        
        logits = self.apply_to_case_uncert(subject=subject, do_mirroring=do_mirroring, rot_angles=rot_angles, rot_axes = rot_axes)
    
        target = self.make_target(subject_data[-1][None,None], train=False)[0]  

        target_logits = logits[:len(self.target_label_names)]
        
        target_probs = torch.from_numpy(target_logits).sigmoid().numpy()
        
        gradient_mask = get_gradient_mask(subject_data[:-1]).astype(np.float)
        
        target_logits = target_logits*gradient_mask[None]
        
        target_probs = target_probs*gradient_mask[None]
        
        return logits, gradient_mask, target, target_logits, target_probs
    
    def apply_to_case_uncert(self, subject, do_mirroring=[False], rot_angles = [-15,0,15], rot_axes = [0], bg_zero = True):
        print(f'applying {len(self.models)} model(s) over {len(self.axes)} axes rotating through {len(rot_angles)} angle(s)')
        with torch.no_grad():
            ensemble_logits = []
            ensemble_flips = []
            slice_masks = []
            case_data = load_subject_and_preprocess(subject, axis='sagittal', bg_zero=bg_zero)
            
            if do_mirroring == False:
                do_mirroring = [False]
            if do_mirroring == True:
                do_mirroring = [True,False]
                
            for model in self.models:  
                model.eval()

                for axis in self.axes:
                        
                    for mirror in do_mirroring:
                        
                        for angle in rot_angles:
                            
                            for rot_axis in rot_axes:
                            
                                if angle != 0:
                                    image_data = rotate_stack(case_data.copy(), angle, rot_axis)

                                else:
                                    image_data = case_data.copy()

                                if mirror:
                                    image_data = image_data[:, ::-1].copy()

                                if axis == 'coronal':
                                    image_data = np.swapaxes(image_data, 1, 2)

                                if axis == 'axial':
                                    image_data = np.swapaxes(image_data, 1, 3)

                                if self.network_type == '3D-to-2D':

                                    input, slicer =  pad_nd_image(image_data, (0, self.patch_size[1], self.patch_size[2]), return_slicer=True)

                                if self.network_type == '3D':

                                    input, slicer =  pad_nd_image(image_data, self.patch_size, return_slicer=True)


                                slicer[0] = slice(0, len(self.target_label_sets)*2, None)

                                # MSHELLER: input likely needs to be to(device) and not cuda()
                                output = model.predict_3D(torch.from_numpy(input).float().cuda(),do_mirroring=do_mirroring, patch_size=self.patch_size,
                                                        use_sliding_window=True, use_gaussian = self.use_gaussian)

                                output = output[1][tuple(slicer)]


                                slice_sums = np.sum(np.any(image_data>0, 0), (1,2))

                                slice_mask = np.stack([np.stack([slice_sums>2500]*image_data.shape[2],-1)]*image_data.shape[3],-1)   

                                slice_mask = np.stack([slice_mask]*len(self.target_label_sets)).astype(np.uint8)


                                if axis == 'coronal':
                                    output = np.swapaxes(output, 1,2)
                                    image_data = np.swapaxes(image_data, 1, 2)
                                    slice_mask = np.swapaxes(slice_mask, 1, 2)


                                if axis == 'axial':
                                    output = np.swapaxes(output, 1,3)
                                    image_data = np.swapaxes(image_data, 1, 3)
                                    slice_mask = np.swapaxes(slice_mask, 1, 3)

                                if mirror:
                                    output = output[:, ::-1].copy()
                                    image_data = image_data[:, ::-1].copy()
                                    slice_mask = slice_mask[:, ::-1].copy()

                                if angle != 0:    
                                    output = rotate_stack(output.copy(), -angle, rot_axis)
                                    slice_mask = (rotate_stack(slice_mask, -angle, rot_axis)>0).astype(np.uint8)


                                output[:len(self.target_label_names)][np.logical_not(slice_mask)] = np.nan
                                ensemble_logits.append(output[:len(self.target_label_names)])

                                flip = ((-torch.from_numpy(output[len(self.target_label_names):]).exp()).sigmoid()).numpy()

                                flip_logit = ((-torch.from_numpy(output[len(self.target_label_names):]).exp())).numpy()

                                flip[np.logical_not(slice_mask)] = np.nan
                                ensemble_flips.append(flip)

                                slice_masks.append(slice_mask)

        ensemble_counts = np.sum(slice_masks, axis=0)
        
        
        uncertainty_weighted_logits = -np.sign(np.array(ensemble_logits))*np.array(flip_logit)
        
        full_logit = np.sum(np.divide(np.nan_to_num(np.array(ensemble_logits),0), ensemble_counts, 
                            out = np.zeros_like(np.array(ensemble_logits)), where = ensemble_counts!=0),axis=0)
        
        
        
        ensemble_predictions = np.greater(np.nan_to_num(ensemble_logits,-10),0)
        
        full_predictions = np.stack([np.greater(full_logit,0)]*(len(self.axes)*len(do_mirroring)*len(rot_angles)*len(rot_axes)*len(self.models)),0)
        
        preds_agree = np.equal(full_predictions, ensemble_predictions).astype(np.uint8)
        
        with np.errstate(divide = 'ignore', invalid = 'ignore'):    
            full_flips = np.sum(np.nan_to_num(np.array(ensemble_flips),0)*(preds_agree) +
                            np.nan_to_num((1 - np.array(ensemble_flips)),0)*(1-preds_agree),axis=0)/ensemble_counts
        
        full_flips = np.nan_to_num(full_flips, 0)
        

        return np.concatenate([full_logit,full_flips])

    
    def make_target(self, gt, train=True):
        target = np.concatenate([np.isin(gt,np.array(labelset)).astype(np.float) for labelset in self.target_label_sets], axis =1)
        if train and self.network_type=='3D-to-2D':
            target = target[:,:,self.patch_size[0]//2]
        return target
        
    def mask_flip_probabilities(self, output, gradient_mask):        
        flip_probs = output[len(self.target_label_names):] * gradient_mask[None]
        return flip_probs

def concatenate(link, layer):
    concat = torch.cat([link, layer], 1)
    return concat

def dense_atrous_bottleneck(in_channel, growth_rate = 12, depth = [4,4,4,4], use_attention=True):
    layer_dict = OrderedDict()
    for idx, growth_steps in enumerate(depth):
        dilation_rate = 2**idx
        for y in range(growth_steps):
            layer_dict["dilated_{}_{}".format(dilation_rate,y)] = DilatedDenseUnit(in_channel, 
                                                                        growth_rate, 
                                                                        kernel_size=3, 
                                                                        dilation = dilation_rate)
            in_channel = in_channel + growth_rate
        
        if use_attention:
            layer_dict["attention_{}".format(dilation_rate)] = AttentionModule(in_channel, in_channel//4, in_channel)
    return nn.Sequential(layer_dict), in_channel

def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]

def crop_to_nonzero(data, seg=None, nonzero_label=0):

    bbox = get_bbox_from_mask(np.all(data[:-1]>0, axis=0), 0)

    cropped_data = []
    for c in range(data.shape[0]):
        cropped = crop_to_bbox(data[c], bbox)
        cropped_data.append(cropped[None])
    data = np.vstack(cropped_data)

    return data, bbox

def down_layer(in_channel, out_channel, kernel_size, padding):
    layer = nn.Sequential(OrderedDict([
            ("pad1", nn.ReplicationPad2d(1)),
            ("conv1", nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn1", nn.InstanceNorm2d(out_channel, affine = False)),
            ("relu1", nn.ReLU()),
            ("dropout1", nn.Dropout(p=0.0)),
            ("pad2", nn.ReplicationPad2d(1)),
            ("conv2", nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn2", nn.InstanceNorm2d(out_channel, affine = False)),
            ("relu2", nn.ReLU()),
            ("dropout2", nn.Dropout(p=0.0))]))
    return layer

def up_layer(in_channel, out_channel, kernel_size, padding):
    layer = nn.Sequential(OrderedDict([
            ("pad1", nn.ReplicationPad2d(1)),
            ("conv1", nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn1", nn.InstanceNorm2d(out_channel, affine = False)),
            ("relu1", nn.ReLU()),
            ("dropout1", nn.Dropout(p=0.0)),
            ("pad2", nn.ReplicationPad2d(1)),
            ("conv2", nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn2", nn.InstanceNorm2d(out_channel, affine = False)),
            ("relu2", nn.ReLU()),
            ("dropout2", nn.Dropout(p=0.0))]))
    return layer

class DilatedDenseUnit(nn.Module):
    def __init__(self, in_channel, growth_rate , kernel_size, dilation):
        super(DilatedDenseUnit,self).__init__()
        self.layer = nn.Sequential(OrderedDict([
            ("bn1", nn.InstanceNorm2d(in_channel, affine = False)),
            ("relu1", nn.ReLU()),
            ("pad1", nn.ReplicationPad2d(dilation)),
            ("conv1", nn.Conv2d(in_channel, growth_rate, kernel_size=kernel_size, dilation = dilation,padding=0)),
            ("dropout", nn.Dropout(p=0.0))]))
    def forward(self, x):
        out = x
        out = self.layer(out)
        out = concatenate(x, out)
        return out
    
class AttentionModule(nn.Module):
    def __init__(self, in_channel , intermediate_channel, out_channel, kernel_size=3):
        super(AttentionModule,self).__init__()
        self.layer = nn.Sequential(OrderedDict([
            ("bn1", nn.InstanceNorm2d(in_channel, affine = False)),
            ("relu1", nn.ReLU()),
            ("pad1", nn.ReplicationPad2d(1)),
            ("conv1", nn.Conv2d(in_channel, intermediate_channel, kernel_size=kernel_size,padding=0)),
            ("bn2", nn.InstanceNorm2d(intermediate_channel, affine = False)),
            ("relu2", nn.ReLU()),
            ("pad2", nn.ReplicationPad2d(1)),
            ("conv2", nn.Conv2d(intermediate_channel, out_channel, kernel_size=kernel_size,padding=0)),
            ("sigmoid", nn.Sigmoid())]))
    def forward(self, x):
        out = x
        out = self.layer(out)
        out = x * out
        return out
    

def reduce_3d_depth (in_channel, out_channel, kernel_size, padding):
    layer = nn.Sequential(OrderedDict([
            ("pad1", nn.ReplicationPad3d((1,1,1,1,0,0))),
            ("conv1", nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)),
            ("bn1", nn.InstanceNorm3d(out_channel, affine = False)),
            ("relu1", nn.ReLU()),
            #("dropout", nn.Dropout(p=0.2))
    ]))
    return layer

class UNET_3D_to_2D(nn.Module):
    def __init__(self, depth, channels_in = 1, 
                 channels_2d_to_3d=32, channels=32, output_channels = 1, slices=5, 
                 dilated_layers = [4,4,4,4],
                growth_rate = 12):
        super(UNET_3D_to_2D, self).__init__()
        
        self.output_channels = output_channels
        self.main_modules = []
        
        self.depth = depth
        self.slices = slices

        
        self.depth_reducing_layers = ModuleList([reduce_3d_depth(in_channel, channels_2d_to_3d, kernel_size=3, padding=0)
                                                 for in_channel in [channels_in]+[channels_2d_to_3d]*(slices//2 - 1)])
        
        
        self.down1 = down_layer(in_channel=channels_2d_to_3d, out_channel=channels, kernel_size=3, padding=0)
        self.main_modules.append(self.down1)
        self.max1 = nn.MaxPool2d(2)
        self.down_layers = ModuleList([down_layer(in_channel = channels*(2**i), 
                                  out_channel = channels * (2**(i+1)),
                                  kernel_size = 3,
                                  padding=0
                                 ) for i in range(self.depth)])
        self.main_modules.append(self.down_layers)
        self.max_layers = ModuleList([nn.MaxPool2d(2) for i in range(self.depth)])
        
        self.bottleneck, bottleneck_features  = dense_atrous_bottleneck(channels*2**self.depth, growth_rate = growth_rate, 
                                                                       depth = dilated_layers)
        self.main_modules.append(self.bottleneck)
        
        self.upsampling_layers = ModuleList([nn.Sequential(OrderedDict([
                ("upsampling",nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)), # since v0.4.0 align_corners= False is default, before was True
                ("pad", nn.ReplicationPad2d(1)),
                ("conv", nn.Conv2d(in_channels= bottleneck_features, 
                                   out_channels=bottleneck_features, 
                                   kernel_size=3, 
                                   padding=0))]))  for i in range(self.depth, -1, -1)])
        self.main_modules.append(self.upsampling_layers)
        self.up_layers = ModuleList([up_layer(in_channel= bottleneck_features+ channels*(2**(i)), 
                                   out_channel=bottleneck_features, 
                                   kernel_size=3, 
                                   padding=0) for i in range(self.depth, -1, -1)])
        
        self.main_modules.append(self.up_layers)
        self.last = nn.Conv2d(in_channels=bottleneck_features, out_channels=output_channels, kernel_size=1)
        self.main_modules.append(self.last)
        
        self.logvar = nn.Conv2d(in_channels=bottleneck_features, out_channels=output_channels, kernel_size=1)

    def forward(self, x):

        # down
        
        out = x
        
        for i in range(self.slices//2):
            out = self.depth_reducing_layers[i](out)
        
        out.transpose_(1, 2).contiguous()
        size = out.size()
        out = out.view((-1, size[2], size[3], size[4]))
        
        links = []
        out = self.down1(out)
        links.append(out)
        out = self.max1(out)
        
        for i in range(self.depth):
            out = self.down_layers[i](out)
            links.append(out)
            out = self.max_layers[i](out)
        
        out = self.bottleneck(out)
        
        links.reverse()

        # up
        
        for i in range(self.depth+1):

            out = self.upsampling_layers[i](out)

            out = concatenate(links[i], out)
            out = self.up_layers[i](out)

        pred = self.last(out)
        logvar = self.logvar(out)

        return torch.cat([pred, logvar], axis=1)
    
    def predict_3D(self, x, do_mirroring=False, mirror_axes=None, 
                    use_sliding_window=True, use_gaussian = True,
                    step_size = 1, patch_size=(5,194,194), batch_size = 2):
        self.eval()
        with torch.no_grad():
            
            logit_total = []
             
          
            num_batches = x.shape[1]

            stack_depth = patch_size[0]

            padding = stack_depth//2

            input = torch.nn.ConstantPad3d((0,0,0,0,padding,padding),0)(x)
          
            slice = 0

            for idx in range(x.shape[1]//batch_size+1):

                batch_list = []

                for y in range(batch_size):
                    if slice == x.shape[1]:
                        break
                    batch_list.append(input[:,slice:slice+stack_depth])  
                    slice +=1
                if len(batch_list) ==0:
                    break
                batch_tensor = torch.stack(batch_list, 0)  

                logit = self.forward(batch_tensor).transpose(0,1)

                logit_total.append(logit)

              
            full_logit = torch.cat(logit_total, 1)
        return None, full_logit.detach().cpu().numpy()

def load_subject(subject, axis = 'axial'):
        assert axis in ['axial', 'sagittal', 'coronal']

        if isinstance(subject, np.ndarray):
            data = subject
        else:
            data = subject_volumes[subject]

        if axis == 'coronal':
          data = np.swapaxes(data, 1,2)

        if axis == 'axial':
          data = np.swapaxes(data, 1,3)
        
        metadata = {}
        
        metadata['means'] = []
        
        metadata['sds'] = []
        
        for modality in range(data.shape[0]-1):
        
            metadata['means'].append(np.mean(data[modality][data[modality]>0]))
            metadata['sds'].append(np.std(data[modality][data[modality]>0]))
            
        metadata['means'] = np.array(metadata['means'])
        metadata['sds'] = np.array(metadata['sds'])
        

        #metadata = load_pickle(subject + ".pkl")

        return data, metadata
    
def load_subject_and_preprocess(subject, axis, bg_zero=True):
    all_data, subject_metadata = load_subject(subject, axis=axis)
    image_data =  all_data[:-1]
    zero_mask = (image_data != 0).astype(np.uint8)
    image_data = image_data - np.array(subject_metadata['means'])[:,np.newaxis,np.newaxis,np.newaxis]
    sds_expanded = np.array(subject_metadata['sds'])[:,np.newaxis,np.newaxis,np.newaxis]
    image_data = np.divide(image_data, sds_expanded,                            out=np.zeros_like(image_data), where=sds_expanded!=0)

    if bg_zero:
        image_data = (image_data * zero_mask).astype(np.float)
        
    return image_data

def get_gradient_mask(volumes, GRADIENT_MASK_ZERO_VOXELS=True):
    if GRADIENT_MASK_ZERO_VOXELS:
        nonzero_mask_all = np.any(volumes>0, axis=0)
        return (nonzero_mask_all).astype(np.float)
    else:
        return (np.ones_like(volumes[0]).astype(np.float))

def rotate_image_on_axis(image, angle, rot_axis):
    return np.swapaxes(rotateImage(np.swapaxes(image,2,rot_axis),angle,cv2.INTER_LINEAR)
                                 ,2,rot_axis)

def rotate_stack(stack, angle, rot_axis):
    images = []
    for idx in range(stack.shape[0]):
        images.append(rotate_image_on_axis(stack[idx], angle, rot_axis))
    return np.stack(images, axis=0)

def rotateImage(image, angle, interp = cv2.INTER_NEAREST):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=interp)
    return result

def make_brats_segmentation(target_probs, threshold = 50):
    seg = (target_probs[2]>threshold).astype(np.uint8)*2
    seg[target_probs[1]>threshold] = 1
    seg[target_probs[0]>threshold] = 4
    
    return seg

def postprocess_brats_segmentation(seg, ensemble, flair, t1, low_core_1 = 75, low_core_2 = 55, low_enhancing = 80, low_whole = 90, size = 100, bleed_size = None):
    new_seg = np.copy(seg)
    
    bleed_threshold = np.percentile(flair[flair>0], 10)
    tissue_threshold = np.percentile(t1[t1>0], 5)
    potential_bleed = np.logical_and(flair<bleed_threshold, t1>tissue_threshold)
    potential_bleed_seg =  np.logical_and(np.logical_and(ensemble[:,:,:,2]<50,ensemble[:,:,:,2]>5), potential_bleed)    
    if bleed_size is not None:
        if np.sum(potential_bleed_seg)>bleed_size:
            print('potential missed bleed')
            new_seg[np.logical_and(np.logical_and(potential_bleed, ensemble[:,:,:,2]>25), new_seg==0)] = 2
            new_seg[np.logical_and(np.logical_and(potential_bleed, ensemble[:,:,:,1]>25), new_seg!=4)] = 1 

    tumor_in_this_case = np.logical_or(ensemble[:,:,:,2]>50, ensemble[:,:,:,2]>50)
    mean_tumor_certainty = np.nan_to_num(np.mean(ensemble[:,:,:,2][tumor_in_this_case]),0)

    adjusted_tumor_threshold = 50
    adjusted_core_threshold = 50
    adjusted_enhancing_threshold = 50
    
    if mean_tumor_certainty <low_whole:
                print('low tumor certainty')
                adjusted_tumor_threshold = 15
                new_seg[np.logical_and(ensemble[:,:,:,2]>adjusted_tumor_threshold, new_seg ==0)] = 2
                print(f'ill-segmented tumor, adjusting threshold {adjusted_tumor_threshold}')
                
    labelmap, num_labels = label(new_seg>0)
    
    core_in_this_case = np.logical_or(ensemble[:,:,:,1]>50, ensemble[:,:,:,0]>50)
    mean_core_certainty = np.nan_to_num(np.mean(ensemble[:,:,:,1][core_in_this_case]),0)
    
    enhancing_in_this_case = np.logical_or(ensemble[:,:,:,0]>50, ensemble[:,:,:,0]>50)
    mean_enhancing_uncertainty = np.nan_to_num(np.mean(ensemble[:,:,:,1][core_in_this_case]),1)
    #print(mean_core_certainty)
    
    if np.sum(new_seg==4)<size:
        new_seg[new_seg==4] = 1
        print(f'enhancing less than {size} mm3: deleting compartment')
        enhancing_present=False
    else:
        enhancing_present = True

    if mean_tumor_certainty < low_whole: 
        tumor_thresh = 50
    else:
        tumor_thresh = 65
    for y in range(num_labels):
        if (np.median(ensemble[:,:,:,2][labelmap == y+1])<tumor_thresh) and num_labels >1:
            new_seg[labelmap == y+1] = 0  
        else:
            if mean_enhancing_uncertainty <low_enhancing and enhancing_present:
                print('low enhancing certainty')
                adjusted_enhancing_threshold = 5
                new_seg[np.logical_and(ensemble[:,:,:,0]>adjusted_enhancing_threshold, labelmap == y+1)] = 4
                print(f'ill-segmented enhancement, adjusting threshold {adjusted_enhancing_threshold}')
                
            labelmap_enhancing, num_labels_enhancing = label((new_seg * (labelmap == y+1).astype(np.uint8)) ==4)
            for w in range(num_labels_enhancing):
                    if (np.sum(labelmap_enhancing == w+1)<10):
                        new_seg[labelmap_enhancing == w+1] = 1
                        #print(f'enhancing tumor, deleted {np.sum(labelmap_enhancing == w+1)} voxels (too small)')
            
            
            labelmap_core, num_labels_core = label(np.logical_or(new_seg * (labelmap == y+1).astype(np.uint8) == 1,
                                                                 new_seg * (labelmap == y+1).astype(np.uint8) == 4))
            #print(np.sum(ensemble[:,:,:,1][labelmap == y+1]>80), np.sum(ensemble[:,:,:,1][labelmap == y+1]>10) - np.sum(ensemble[:,:,:,1][labelmap == y+1]>80))
            if mean_core_certainty >=low_core_1:
                for z in range(num_labels_core):
                    if np.median(ensemble[:,:,:,1][labelmap_core == z+1])<60 and np.sum(new_seg[labelmap_core == z+1] == 4) == 0:
                        new_seg[labelmap_core == z+1] = 2
                        #print(f'core tumor, deleted {np.sum(labelmap_core == z+1)} voxels (too uncertain, no enhancing)')
            
            
            if mean_core_certainty <low_core_1:
                print('low core certainty')
                adjusted_core_threshold = 5
                if mean_core_certainty <low_core_2:
                    adjusted_core_threshold = 1
                #if 2*np.sum(ensemble[:,:,:,1][labelmap == y+1]>50)< np.sum(ensemble[:,:,:,1][labelmap == y+1]>25):
                new_seg[np.logical_and(np.logical_and(ensemble[:,:,:,1]>adjusted_core_threshold, labelmap == y+1), new_seg !=4)] = 1
                print(f'ill-segmented core, adjusting threshold {adjusted_core_threshold}')
    

    #did we delete the whole tumor?  something went wrong with the postprocessing, reset!
    if np.sum(new_seg >0)<100:
        print('tumor missing, resetting segmentation')
        new_seg = np.copy(seg)

    #is there still no segmentation?  the we missing the tumor!  Look for it
    
    if np.sum(new_seg >0)<100:        
        for emergencythreshold in [45,40,35,30,25,20,15,10,5]:
            if np.sum(ensemble[:,:,:,2]>emergencythreshold)>1000:
                print(f'tumor found at threshold {emergencythreshold}')
                adjusted_tumor_threshold = emergencythreshold
                new_seg[(ensemble[:,:,:,2]>emergencythreshold)] = 1
                break

    #is there still no core? core moust be the whole tumor
        
    if np.sum(np.logical_or(new_seg == 1, new_seg == 4))<10:
        print('core missing, setting to whole tumor')
        new_seg[(new_seg == 2)] = 1
    
    labelmap, num_labels = label(new_seg>0)
    
    labelmap_core, num_labels_core = label(np.logical_or(new_seg == 1,
                                                                 new_seg == 4)) 
    
    print(f'{num_labels} tumor compartments')                

    print(f'{num_labels_core} core compartments')
   
    in_tumor = (new_seg >0) 
    out_of_tumor = np.logical_not(in_tumor)
    if adjusted_tumor_threshold != 50:
        ensemble[:,:,:,2][in_tumor] = np.clip(ensemble[:,:,:,2][in_tumor] +(50 - adjusted_tumor_threshold) , 0, 100)
        ensemble[:,:,:,2][out_of_tumor] = np.clip(ensemble[:,:,:,2][out_of_tumor]*50/adjusted_tumor_threshold , 0, 100)

    in_core = np.logical_or(new_seg == 1,  new_seg == 4)
    out_of_core = np.logical_not(in_core)
    if adjusted_core_threshold != 50:
            ensemble[:,:,:,1][in_core] = np.clip(ensemble[:,:,:,1][in_core] +(50 - adjusted_core_threshold) , 0, 100)
            ensemble[:,:,:,1][out_of_core] = np.clip(ensemble[:,:,:,1][out_of_core]*50/adjusted_core_threshold , 0, 100)

    in_enhancing = (new_seg == 4)
    out_of_enhancing = np.logical_not(in_enhancing)
    if adjusted_enhancing_threshold != 50:
        ensemble[:,:,:,0][in_enhancing] = np.clip(ensemble[:,:,:,0][in_enhancing] +(50 - adjusted_enhancing_threshold) , 0, 100)
        ensemble[:,:,:,0][out_of_enhancing] = np.clip(ensemble[:,:,:,0][out_of_enhancing]*50/adjusted_enhancing_threshold , 0, 100)

    uncertainty = np.zeros_like(ensemble)

    uncertainty[:,:,:,0][out_of_enhancing] = ensemble[:,:,:,0][out_of_enhancing]*2
    uncertainty[:,:,:,0][in_enhancing] = 0

    uncertainty[:,:,:,1][out_of_core] = ensemble[:,:,:,1][out_of_core]*2
    uncertainty[:,:,:,1][in_core] = 0

    uncertainty[:,:,:,2][out_of_tumor] = ensemble[:,:,:,2][out_of_tumor]*2
    uncertainty[:,:,:,2][in_tumor] = 0

    return new_seg, np.clip(uncertainty, 0, 100)


def main(data_dir):
    data = DeepSCANInferenceDataObject(data_path=data_dir)
    inference_wrapper = DeepSCANBraTS2020Inference(data)
    inference_wrapper.run_inference_and_store_results()


# I have to move all top level code into a 'main' block to avoid running it
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', default='data', type=str)
    # TODO: do we want to support lite-mode?
    # parser.add_argument('-l', default = False, action = 'store_true', dest='lite_mode')
    args = parser.parse_args()
    main(**vars(args))

