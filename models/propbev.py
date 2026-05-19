import queue
import torch
import numpy as np
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner import get_dist_info
from mmcv.runner.fp16_utils import cast_tensor_type
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from .utils import GridMask, pad_multiple, GpuPhotoMetricDistortion

NUM_CAMS = 6  # nuScenes: 6 cameras per frame

@DETECTORS.register_module()
class PropBEV(MVXTwoStageDetector):
    def __init__(self,
                 data_aug=None,
                 stop_prev_grad=0,
                 num_propgated=256,  # Number of queries propagated sequentially: t-2 → t-1 → t
                 t1_slot=1,   # Slot index of t-1 in the frame sequence (8f past-only: 1, 15f interleave: 1)
                 t2_slot=2,   # Slot index of t-2 in the frame sequence (8f past-only: 2, 15f interleave: 3)
                 use_t2=True, # Whether to use t-2 keyframe (False saves GPU memory for large backbones)
                 use_mask_camera=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 **kwargs):
        super(PropBEV, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.data_aug = data_aug
        self.stop_prev_grad = stop_prev_grad
        self.num_propgated = num_propgated
        self.t1_slot = t1_slot
        self.t2_slot = t2_slot
        self.use_t2 = use_t2    # whether to use t-2 keyframe
        self.use_mask_camera = use_mask_camera
        self.color_aug = GpuPhotoMetricDistortion()
        self.grid_mask = GridMask(ratio=0.5, prob=0.7)
        self.use_grid_mask = True
        self.fp16_enabled = False

        self.memory = {}
        self.queue = queue.Queue()

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img):
        if self.use_grid_mask:
            img = self.grid_mask(img)

        img_feats = self.img_backbone(img)

        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        return img_feats

    def extract_feat(self, img, img_metas):
        if len(img.shape) == 6:
            img = img.flatten(1, 2)  # [B, TN, C, H, W]

        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        img = img.float()

        # move some augmentations to GPU
        if self.data_aug is not None:
            if 'img_color_aug' in self.data_aug and self.data_aug['img_color_aug'] and self.training:
                img = self.color_aug(img)

            if 'img_norm_cfg' in self.data_aug:
                img_norm_cfg = self.data_aug['img_norm_cfg']

                norm_mean = torch.tensor(img_norm_cfg['mean'], device=img.device)
                norm_std = torch.tensor(img_norm_cfg['std'], device=img.device)

                if img_norm_cfg['to_rgb']:
                    img = img[:, [2, 1, 0], :, :]  # BGR to RGB

                img = img - norm_mean.reshape(1, 3, 1, 1)
                img = img / norm_std.reshape(1, 3, 1, 1)

            for b in range(B):
                img_shape = (img.shape[2], img.shape[3], img.shape[1])
                img_metas[b]['img_shape'] = [img_shape for _ in range(N)]
                img_metas[b]['ori_shape'] = [img_shape for _ in range(N)]

            if 'img_pad_cfg' in self.data_aug:
                img_pad_cfg = self.data_aug['img_pad_cfg']
                img = pad_multiple(img, img_metas, size_divisor=img_pad_cfg['size_divisor'])

        input_shape = img.shape[-2:]
        # update real input shape of each single img
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        if self.training and self.stop_prev_grad > 0:
            H, W = input_shape
            img = img.reshape(B, -1, NUM_CAMS, C, H, W)

            img_grad = img[:, :self.stop_prev_grad]
            img_nograd = img[:, self.stop_prev_grad:]

            all_img_feats = [self.extract_img_feat(img_grad.reshape(-1, C, H, W))]

            with torch.no_grad():
                self.eval()
                for k in range(img_nograd.shape[1]):
                    all_img_feats.append(self.extract_img_feat(img_nograd[:, k].reshape(-1, C, H, W)))
                self.train()

            img_feats = []
            for lvl in range(len(all_img_feats[0])):
                C, H, W = all_img_feats[0][lvl].shape[1:]
                img_feat = torch.cat([feat[lvl].reshape(B, -1, NUM_CAMS, C, H, W) for feat in all_img_feats], dim=1)
                img_feat = img_feat.reshape(-1, C, H, W)
                img_feats.append(img_feat)
        else:
            img_feats = self.extract_img_feat(img)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped

    def _reorder_hist_feats(self, feats_past, feats_future):
        return [torch.cat([p, f], dim=1) for p, f in zip(feats_past, feats_future)]
    
    def _transform_pos_to_current_frame(self, pos_metric, lidar_to_target):
        B, N, _ = pos_metric.shape
        pos_homo = torch.cat([pos_metric, torch.ones(B, N, 1, device=pos_metric.device)], dim=-1)
        pos_t = torch.bmm(pos_homo, lidar_to_target.transpose(-1, -2).to(pos_metric.dtype))[..., :3]
        return pos_t
    
    def _transform_occ_to_current_frame(self, occ_loc, lidar_to_target, pc_range, voxel_size=0.4):
        device = occ_loc.device
        pc_range_tensor = torch.tensor(pc_range[:3], device=device)
        
        pts = occ_loc.float() * voxel_size + pc_range_tensor
        pts_homo = torch.cat([pts, torch.ones(pts.shape[0], pts.shape[1], 1, device=device)], dim=-1)
        pts_t = torch.bmm(pts_homo, lidar_to_target.transpose(-1, -2).to(pts.dtype))[..., :3]
        
        new_occ_loc = ((pts_t - pc_range_tensor) / voxel_size).round().long()
        # Clamp to avoid out of bounds (Grid 200x200x16)
        new_occ_loc[..., 0] = new_occ_loc[..., 0].clamp(0, 199)
        new_occ_loc[..., 1] = new_occ_loc[..., 1].clamp(0, 199)
        new_occ_loc[..., 2] = new_occ_loc[..., 2].clamp(0, 15)
        return new_occ_loc
    
    def _select_topk_occ_and_feat(self, occ_loc, seg_pred, query_feat, k):
        # Calculate non-empty probability (1 - background probability)
        # Assuming background class is the last index in seg_pred
        non_free_prob = 1 - torch.softmax(seg_pred, dim=-1)[..., -1]
        
        k = min(k, non_free_prob.shape[1])
        _, idx = torch.topk(non_free_prob, k, dim=1)

        occ_topk = torch.gather(occ_loc, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        feat_topk = torch.gather(query_feat, 1, idx.unsqueeze(-1).expand(-1, -1, query_feat.shape[-1]))
        return occ_topk, feat_topk
    
    def _prepare_metas(self, img_metas, keyframe='t'):
        total_imgs = len(img_metas[0]['filename'])  # num_frames * NUM_CAMS
        rot = {'t': 0, 't1': self.t1_slot * NUM_CAMS, 't2': self.t2_slot * NUM_CAMS}[keyframe]

        metas = []
        for m in img_metas:
            new_m = {k: v for k, v in m.items()}
            lidar_key = {'t': 'lidar2img', 't1': 'lidar2img_t1', 't2': 'lidar2img_t2'}[keyframe]
            if rot > 0:
                new_m['lidar2img'] = m[lidar_key][rot:total_imgs] + m[lidar_key][0:rot]
                new_m['img_timestamp'] = m['img_timestamp'][rot:total_imgs] + m['img_timestamp'][0:rot]
                new_m['filename'] = m['filename'][rot:total_imgs] + m['filename'][0:rot]
            else:
                new_m['lidar2img'] = m[lidar_key][:total_imgs]
                new_m['img_timestamp'] = m['img_timestamp'][:total_imgs]
                new_m['filename'] = m['filename'][:total_imgs]
            metas.append(new_m)
        return metas

    def _get_lidar2lidar_transform(self, img_metas, from_frame='t1', to_frame='t'):
        def get_ego_pose(meta, frame):
            if frame == 't':
                return meta['ego_pose']
            elif frame == 't1':
                return meta['ego_pose_t1']
            else:  # 't2'
                return meta['ego_pose_t2']
        
        transforms = []
        for meta in img_metas:
            ego_pose_from = get_ego_pose(meta, from_frame)
            ego_pose_to = get_ego_pose(meta, to_frame)
            # lidar_{from_frame} → lidar_{to_frame}
            transforms.append(np.linalg.inv(ego_pose_to) @ ego_pose_from)

        device = img_metas[0]['lidar2img'][0].device if torch.is_tensor(img_metas[0]['lidar2img'][0]) else 'cuda'
        return torch.tensor(np.stack(transforms), dtype=torch.float32, device=device)

    def _process_keyframe(self, img_feats, img_metas, keyframe='t1', prev_pos=None, prev_feat=None, to_frame='t'):
        metas = self._prepare_metas(img_metas, keyframe=keyframe)

        with torch.no_grad():
            outs = self.pts_bbox_head(img_feats, metas, prev_pos=prev_pos, prev_feat=prev_feat)
            last_occ = outs['occ_preds'][-1]
            occ_loc = last_occ[0]      # [B, K, 3] 
            seg_pred = last_occ[2]     # [B, K, CLS] 
            query_feat = last_occ[3]   # [B, K, C]
            pc_range = self.pts_bbox_head.pc_range

        lidar_to_target = self._get_lidar2lidar_transform(img_metas, from_frame=keyframe, to_frame=to_frame)
        pos_aligned = self._transform_occ_to_current_frame(occ_loc, lidar_to_target, pc_range)

        occ_topk, feat_topk = self._select_topk_occ_and_feat(pos_aligned, seg_pred, query_feat, self.num_propgated)
        # # 基于占据语义概率提取 Top-K
        # non_free_prob = 1 - torch.softmax(seg_pred, dim=-1)[..., -1]
        # k = min(self.num_propgated, non_free_prob.shape[1])
        # _, idx = torch.topk(non_free_prob, k, dim=1)
        
        # pos_topk = torch.gather(pos_aligned, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        # feat_topk = torch.gather(query_feat, 1, idx.unsqueeze(-1).expand(-1, -1, query_feat.shape[-1]))

        return occ_topk, feat_topk

    def _process_dual_keyframes(self, img_feats_t1, img_feats_t2, img_metas):
        if self.use_t2:
            pos_t2, feat_t2 = self._process_keyframe(img_feats_t2, img_metas, 't2', prev_pos=None, prev_feat=None, to_frame='t1')
        else:
            pos_t2, feat_t2 = None, None
        
        pos_t1, feat_t1 = self._process_keyframe(img_feats_t1, img_metas, 't1', prev_pos=pos_t2, prev_feat=feat_t2, to_frame='t')
        return pos_t1, feat_t1

    def forward_pts_train(self, 
                          mlvl_feats, 
                          voxel_semantics, 
                          voxel_instances, 
                          instance_class_ids, 
                          mask_camera, 
                          img_metas, 
                          prev_pos=None, 
                          prev_feat=None):
        outs = self.pts_bbox_head(mlvl_feats, img_metas, prev_pos=prev_pos, prev_feat=prev_feat)
        loss_inputs = [voxel_semantics, voxel_instances, instance_class_ids, outs]
        if mask_camera is not None:
            loss_inputs.append(mask_camera)
        return self.pts_bbox_head.loss(*loss_inputs)

    @force_fp32(apply_to=('img'))
    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self, 
                      img_metas=None, 
                      img=None, 
                      voxel_semantics=None, 
                      voxel_instances=None, 
                      instance_class_ids=None, 
                      mask_camera=None, 
                      **kwargs):
        B, N, C, H, W = img.shape

        if self.num_propgated > 0:
            img_feats_all = self.extract_feat(img, img_metas)
            img_feats_t = img_feats_all

            r1 = self.t1_slot * NUM_CAMS
            img_feats_t1 = self._reorder_hist_feats(
                [feat[:, r1:N] for feat in img_feats_all],
                [feat[:, 0:r1] for feat in img_feats_all])

            if self.use_t2:
                r2 = self.t2_slot * NUM_CAMS
                img_feats_t2 = self._reorder_hist_feats(
                    [feat[:, r2:N] for feat in img_feats_all],
                    [feat[:, 0:r2] for feat in img_feats_all])
            else:
                img_feats_t2 = None
            
            use_temporal = torch.rand(1).item() > 0.1
            if use_temporal:
                prev_pos, prev_feat = self._process_dual_keyframes(img_feats_t1, img_feats_t2, img_metas)
            else:
                prev_pos, prev_feat = None, None
            
            metas_t = self._prepare_metas(img_metas, keyframe='t')
            losses = self.forward_pts_train(img_feats_t, voxel_semantics, voxel_instances, instance_class_ids, mask_camera, metas_t, prev_pos=prev_pos, prev_feat=prev_feat)
        else:
            img_feats = self.extract_feat(img, img_metas)
            losses = self.forward_pts_train(img_feats, voxel_semantics, voxel_instances, instance_class_ids, mask_camera, img_metas)

        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        output = self.simple_test(img_metas, img)

        sem_pred = output['sem_pred'].cpu().numpy().astype(np.uint8)
        occ_loc = output['occ_loc'].cpu().numpy().astype(np.uint8)
        batch_size = sem_pred.shape[0]

        if 'pano_inst' in output and 'pano_sem' in output:
            pano_inst = output['pano_inst'].cpu().numpy().astype(np.int16)
            pano_sem = output['pano_sem'].cpu().numpy().astype(np.uint8)
            return [{
                'sem_pred': sem_pred[b:b+1],
                'pano_inst': pano_inst[b:b+1],
                'pano_sem': pano_sem[b:b+1],
                'occ_loc': occ_loc[b:b+1]
            } for b in range(batch_size)]
        else:
            return [{
                'sem_pred': sem_pred[b:b+1],
                'occ_loc': occ_loc[b:b+1]
            } for b in range(batch_size)]

    def simple_test_pts(self, x, img_metas, rescale=False, prev_pos=None, prev_feat=None):
        outs = self.pts_bbox_head(x, img_metas, prev_pos=prev_pos, prev_feat=prev_feat)
        outs = self.pts_bbox_head.merge_occ_pred(outs)
        return outs
    
    def simple_test(self, img_metas, img=None, rescale=False):
        world_size = get_dist_info()[1]
        if world_size == 1:
            return self.simple_test_online(img_metas, img, rescale)
        else:
            return self.simple_test_offline(img_metas, img, rescale)

    def simple_test_offline(self, img_metas, img=None, rescale=False):
        if isinstance(img, list):
            img = img[0]
        if isinstance(img_metas[0], list):
            img_metas = img_metas[0]
        B, N, C, H, W = img.shape

        if self.num_propgated > 0:
            img_feats_all = self.extract_feat(img, img_metas)
            img_feats_t = img_feats_all

            r1 = self.t1_slot * NUM_CAMS
            img_feats_t1 = self._reorder_hist_feats(
                [feat[:, r1:N] for feat in img_feats_all],
                [feat[:, 0:r1] for feat in img_feats_all])

            if self.use_t2:
                r2 = self.t2_slot * NUM_CAMS
                img_feats_t2 = self._reorder_hist_feats(
                    [feat[:, r2:N] for feat in img_feats_all],
                    [feat[:, 0:r2] for feat in img_feats_all])
            else:
                img_feats_t2 = None

            prev_pos, prev_feat = self._process_dual_keyframes(img_feats_t1, img_feats_t2, img_metas)
            metas_t = self._prepare_metas(img_metas, keyframe='t')
            return self.simple_test_pts(img_feats_t, metas_t, rescale=rescale, prev_pos=prev_pos, prev_feat=prev_feat)
            
        else:
            img_feats = self.extract_feat(img=img, img_metas=img_metas)
            return self.simple_test_pts(img_feats, img_metas, rescale=rescale)

    def simple_test_online(self, img_metas, img=None, rescale=False):
        self.fp16_enabled = False
        assert len(img_metas) == 1

        B, N, C, H, W = img.shape
        img = img.reshape(B, N//NUM_CAMS, NUM_CAMS, C, H, W)

        img_filenames = img_metas[0]['filename']
        num_frames = len(img_filenames) // NUM_CAMS

        img_shape = (H, W, C)
        img_metas[0]['img_shape'] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]['ori_shape'] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]['pad_shape'] = [img_shape for _ in range(len(img_filenames))]

        img_feats_list, img_metas_list = [], []

        for i in range(num_frames):
            img_indices = list(np.arange(i * NUM_CAMS, (i + 1) * NUM_CAMS))

            img_metas_curr = [{}]
            for k in img_metas[0].keys():
                if isinstance(img_metas[0][k], list):
                    img_metas_curr[0][k] = [img_metas[0][k][i] for i in img_indices]

            if img_filenames[img_indices[0]] in self.memory:
                img_feats_curr = self.memory[img_filenames[img_indices[0]]]
            else:
                img_feats_curr = self.extract_feat(img[:, i], img_metas_curr)
                self.memory[img_filenames[img_indices[0]]] = img_feats_curr
                self.queue.put(img_filenames[img_indices[0]])
                while self.queue.qsize() >= 16:
                    pop_key = self.queue.get()
                    self.memory.pop(pop_key)

            img_feats_list.append(img_feats_curr)
            img_metas_list.append(img_metas_curr)

        feat_levels = len(img_feats_list[0])
        img_feats_reorganized = []
        for j in range(feat_levels):
            feat_l = torch.cat([img_feats_list[i][j] for i in range(len(img_feats_list))], dim=0)
            feat_l = feat_l.flatten(0, 1)[None, ...]
            img_feats_reorganized.append(feat_l)

        img_metas_reorganized = img_metas_list[0]
        for i in range(1, len(img_metas_list)):
            for k, v in img_metas_list[i][0].items():
                if isinstance(v, list):
                    img_metas_reorganized[0][k].extend(v)

        img_feats = img_feats_reorganized
        img_metas = img_metas_reorganized
        img_feats = cast_tensor_type(img_feats, torch.half, torch.float32)

        return self.simple_test_pts(img_feats, img_metas, rescale=rescale)