import os
import mmcv
import glob
import torch
import numpy as np
from tqdm import tqdm
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from nuscenes.eval.common.utils import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from torch.utils.data import DataLoader
from models.utils import sparse2dense
from .ray_metrics import main_rayiou, main_raypq
from .ego_pose_dataset import EgoPoseDataset
from .pipelines.loading import compose_lidar2img  # Import from loading.py

occ3d_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

openocc_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]
@DATASETS.register_module()
class NuSceneOcc(NuScenesDataset):    
    def __init__(self, occ_gt_root, *args, **kwargs):
        super().__init__(filter_empty_gt=False, *args, **kwargs)
        self.occ_gt_root = occ_gt_root
        self.data_infos = self.load_annotations(self.ann_file)

        self.token2scene = {}
        for gt_path in glob.glob(os.path.join(self.occ_gt_root, '*/*/*.npz')):
            token = gt_path.split('/')[-2]
            scene_name = gt_path.split('/')[-3]
            self.token2scene[token] = scene_name

        for i in range(len(self.data_infos)):
            scene_name = self.token2scene[self.data_infos[i]['token']]
            self.data_infos[i]['scene_name'] = scene_name

    def collect_sweeps(self, index, into_past=150, into_future=0):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            all_sweeps_prev.append(self.data_infos[curr_index - 1]['cams'])
            curr_index = curr_index - 1
        
        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            all_sweeps_next.extend(curr_sweeps[::-1])
            all_sweeps_next.append(self.data_infos[curr_index]['cams'])
            curr_index = curr_index + 1

        return all_sweeps_prev, all_sweeps_next

    def get_data_info(self, index):
        info = self.data_infos[index]
        sweeps_prev, sweeps_next = self.collect_sweeps(index)

        # ==================== Keyframe t (current) ego pose ====================
        ego2global_translation = np.array(info['ego2global_translation'])
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = np.array(info['lidar2ego_translation'])
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation_mat = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation_mat = Quaternion(lidar2ego_rotation).rotation_matrix

        input_dict = dict(
            sample_idx=info['token'],
            sweeps={'prev': sweeps_prev, 'next': sweeps_next},
            timestamp=info['timestamp'] / 1e6,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation_mat,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation_mat,
            occ_path=os.path.join(self.occ_gt_root, info['scene_name'], info['token'], 'labels.npz') # Occ path
        )

        ego2lidar = transform_matrix(lidar2ego_translation, Quaternion(lidar2ego_rotation), inverse=True)
        input_dict['ego2lidar'] = [ego2lidar for _ in range(6)]

        current_scene_token = info['scene_token'] if 'scene_token' in info else self.token2scene[info['token']]
        
        # t-1
        if index > 0:
            t1_info = self.data_infos[index - 1]
            t1_ego2global_translation = np.array(t1_info['ego2global_translation'])
            t1_ego2global_rotation = Quaternion(t1_info['ego2global_rotation']).rotation_matrix
            t1_lidar2ego_translation = np.array(t1_info['lidar2ego_translation'])
            t1_lidar2ego_rotation = Quaternion(t1_info['lidar2ego_rotation']).rotation_matrix
        else:
            t1_ego2global_translation = ego2global_translation
            t1_ego2global_rotation = ego2global_rotation_mat
            t1_lidar2ego_translation = lidar2ego_translation
            t1_lidar2ego_rotation = lidar2ego_rotation_mat
        
        # t-2
        if index > 1:
            t2_info = self.data_infos[index - 2]
            t2_ego2global_translation = np.array(t2_info['ego2global_translation'])
            t2_ego2global_rotation = Quaternion(t2_info['ego2global_rotation']).rotation_matrix
            t2_lidar2ego_translation = np.array(t2_info['lidar2ego_translation'])
            t2_lidar2ego_rotation = Quaternion(t2_info['lidar2ego_rotation']).rotation_matrix
        else:
            t2_ego2global_translation = t1_ego2global_translation
            t2_ego2global_rotation = t1_ego2global_rotation
            t2_lidar2ego_translation = t1_lidar2ego_translation
            t2_lidar2ego_rotation = t1_lidar2ego_rotation
            
        # 
        def get_ego_pose(e2g_t, e2g_r, l2e_t, l2e_r):
            e2g_mat, l2e_mat = np.eye(4), np.eye(4)
            e2g_mat[:3, :3], e2g_mat[:3, 3] = e2g_r, e2g_t
            l2e_mat[:3, :3], l2e_mat[:3, 3] = l2e_r, l2e_t
            return e2g_mat @ l2e_mat

        input_dict['ego_pose'] = get_ego_pose(ego2global_translation, ego2global_rotation_mat, lidar2ego_translation, lidar2ego_rotation_mat)
        input_dict['ego_pose_t1'] = get_ego_pose(t1_ego2global_translation, t1_ego2global_rotation, t1_lidar2ego_translation, t1_lidar2ego_rotation)
        input_dict['ego_pose_t2'] = get_ego_pose(t2_ego2global_translation, t2_ego2global_rotation, t2_lidar2ego_translation, t2_lidar2ego_rotation)

        input_dict['t1_ego2global_translation'] = t1_ego2global_translation
        input_dict['t1_ego2global_rotation'] = t1_ego2global_rotation
        input_dict['t1_lidar2ego_translation'] = t1_lidar2ego_translation
        input_dict['t1_lidar2ego_rotation'] = t1_lidar2ego_rotation
        
        input_dict['t2_ego2global_translation'] = t2_ego2global_translation
        input_dict['t2_ego2global_rotation'] = t2_ego2global_rotation
        input_dict['t2_lidar2ego_translation'] = t2_lidar2ego_translation
        input_dict['t2_lidar2ego_rotation'] = t2_lidar2ego_rotation

        if self.modality['use_camera']:
            img_paths, img_timestamps = [], []
            lidar2img_rts, lidar2img_rts_t1, lidar2img_rts_t2 = [], [], []

            for _, cam_info in info['cams'].items():
                img_paths.append(os.path.relpath(cam_info['data_path']))
                img_timestamps.append(cam_info['timestamp'] / 1e6)

                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3], lidar2cam_rt[3, :3] = lidar2cam_r.T, -lidar2cam_t
                
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rts.append((viewpad @ lidar2cam_rt.T))

                # Compute sensor2global for keyframe t
                sensor2global_rotation = (ego2global_rotation_mat @ lidar2ego_rotation_mat @ cam_info['sensor2lidar_rotation']).T
                sensor2global_translation = (
                    ego2global_rotation_mat @ lidar2ego_rotation_mat @ cam_info['sensor2lidar_translation']
                    + ego2global_rotation_mat @ lidar2ego_translation
                    + ego2global_translation
                )

                # ==================== lidar2img_t1 & t2 ====================
                lidar2img_rts_t1.append(compose_lidar2img(
                    t1_ego2global_translation, t1_ego2global_rotation, t1_lidar2ego_translation, t1_lidar2ego_rotation,
                    sensor2global_translation, sensor2global_rotation, intrinsic))
                lidar2img_rts_t2.append(compose_lidar2img(
                    t2_ego2global_translation, t2_ego2global_rotation, t2_lidar2ego_translation, t2_lidar2ego_rotation,
                    sensor2global_translation, sensor2global_rotation, intrinsic))

            input_dict.update(dict(
                img_filename=img_paths, img_timestamp=img_timestamps,
                lidar2img=lidar2img_rts, lidar2img_t1=lidar2img_rts_t1, lidar2img_t2=lidar2img_rts_t2,
            ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict
    
    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        occ_gts, occ_preds, inst_gts, inst_preds, lidar_origins = [], [], [], [], []
        print('\nStarting Evaluation...')

        sample_tokens = [info['token'] for info in self.data_infos]

        for batch in DataLoader(EgoPoseDataset(self.data_infos), num_workers=8):
            token = batch[0][0]
            output_origin = batch[1]
            
            data_id = sample_tokens.index(token)
            info = self.data_infos[data_id]

            occ_path = os.path.join(self.occ_gt_root, info['scene_name'], info['token'], 'labels.npz')
            occ_gt = np.load(occ_path, allow_pickle=True)
            gt_semantics = occ_gt['semantics']

            occ_pred = occ_results[data_id]
            sem_pred = torch.from_numpy(occ_pred['sem_pred'])  # [B, N]
            occ_loc = torch.from_numpy(occ_pred['occ_loc'].astype(np.int64))  # [B, N, 3]
            
            data_type = self.occ_gt_root.split('/')[-1]
            if data_type == 'occ3d' or data_type == 'occ3d_panoptic':
                occ_class_names = occ3d_class_names
            elif data_type == 'openocc_v2':
                occ_class_names = openocc_class_names
            else:
                raise ValueError
            free_id = len(occ_class_names) - 1
            
            occ_size = list(gt_semantics.shape)
            sem_pred, _ = sparse2dense(occ_loc, sem_pred, dense_shape=occ_size, empty_value=free_id)
            sem_pred = sem_pred.squeeze(0).numpy()

            if 'pano_inst' in occ_pred.keys():
                pano_inst = torch.from_numpy(occ_pred['pano_inst'])
                pano_sem = torch.from_numpy(occ_pred['pano_sem'])

                pano_inst, _ = sparse2dense(occ_loc, pano_inst, dense_shape=occ_size, empty_value=0)
                pano_sem, _ = sparse2dense(occ_loc, pano_sem, dense_shape=occ_size, empty_value=free_id)
                pano_inst = pano_inst.squeeze(0).numpy()
                pano_sem = pano_sem.squeeze(0).numpy()
                sem_pred = pano_sem

                gt_instances = occ_gt['instances']
                inst_gts.append(gt_instances)
                inst_preds.append(pano_inst)

            lidar_origins.append(output_origin)
            occ_gts.append(gt_semantics)
            occ_preds.append(sem_pred)
        
        if len(inst_preds) > 0:
            results = main_raypq(occ_preds, occ_gts, inst_preds, inst_gts, lidar_origins, occ_class_names=occ_class_names)
            results.update(main_rayiou(occ_preds, occ_gts, lidar_origins, occ_class_names=occ_class_names))
            return results
        else:
            return main_rayiou(occ_preds, occ_gts, lidar_origins, occ_class_names=occ_class_names)

    def format_results(self, occ_results, submission_prefix, **kwargs):
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            sample_token = info['token']
            save_path = os.path.join(submission_prefix, '{}.npz'.format(sample_token))
            np.savez_compressed(save_path, occ_pred.astype(np.uint8))
        
        print('\nFinished.')
