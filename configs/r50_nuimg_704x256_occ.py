dataset_type = 'NuSceneOcc'
dataset_root = 'data/nuscenes/'
occ_gt_root = 'data/nuscenes/occ3d' 

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True
)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
voxel_size = [0.2, 0.2, 8]
occ_size = [200, 200, 16]

# arch config
embed_dims = 256
num_layers = 6
num_query = 900
num_frames = 9  
num_levels = 4
num_points = 4

_num_groups_ = 4
_num_layers_ = 2
_num_queries_ = 100
_topk_training_ = [4000, 16000, 64000]
_topk_testing_ = [2000, 8000, 32000]

img_backbone = dict(
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN2d', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    with_cp=True)
img_neck = dict(
    type='FPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=embed_dims,
    num_outs=num_levels)
img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True)

model = dict(
    type='PropBEV',
    data_aug=dict(
        img_color_aug=True,  # Move some augmentations to GPU
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32)),
    num_propgated=256,  # Queries propagated sequentially: t-2 → t-1 → t (only t-1 predictions sent to t)
    t1_slot=1,   # Past-only: [curr, prev1, prev2, prev3, ...] → t-1 at slot 1
    t2_slot=2,   # Past-only: [curr, prev1, prev2, prev3, ...] → t-2 at slot 2
    use_t2=True, # Use t-2 keyframe (set False for large backbones to save memory)
    img_backbone=img_backbone,
    img_neck=img_neck,
    pts_bbox_head=dict(
        type='SparseOccHead',
        class_names=occ_class_names,
        embed_dims=embed_dims,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        transformer=dict(
            type='SparseOccTransformer',
            embed_dims=embed_dims,
            num_layers=_num_layers_,
            num_frames=num_frames,
            num_points=num_points,
            num_groups=_num_groups_,
            num_queries=_num_queries_,
            num_levels=4,
            num_classes=len(occ_class_names),
            pc_range=point_cloud_range,
            occ_size=occ_size,
            topk_training=_topk_training_,
            topk_testing=_topk_testing_),
        loss_cfgs=dict(
            loss_mask2former=dict(
                type='Mask2FormerLoss',
                num_classes=len(occ_class_names),
                no_class_weight=0.1,
                loss_cls_weight=2.0,
                loss_mask_weight=5.0,
                loss_dice_weight=5.0,
            ),
            loss_geo_scal=dict(
                type='GeoScalLoss',
                num_classes=len(occ_class_names),
                loss_weight=1.0
            ),
            loss_sem_scal=dict(
                type='SemScalLoss',
                num_classes=len(occ_class_names),
                loss_weight=1.0
            )
        ),
    ),
)

ida_aug_conf = {
    'resize_lim': (0.38, 0.55),
    'final_dim': (256, 704),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900, 'W': 1600,
    'rand_flip': True,
}

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.2,
    flip_dy_ratio=0.2
)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames-1),  # past sweeps + 1 current = num_frames total
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, classes=class_names, is_train=True),
    dict(type='LoadOccGTFromFile', num_classes=len(occ_class_names)),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', 
         keys=['img', 'voxel_semantics', 'voxel_instances', 'instance_class_ids'], 
         meta_keys=(
            'filename', 'ori_shape', 'img_shape', 'pad_shape', 
            'lidar2img', 'lidar2img_t1', 'lidar2img_t2', 'img_timestamp',
            'ego_pose', 'ego_pose_t1', 'ego_pose_t2', 'ego2lidar'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=num_frames-1, test_mode=True),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, classes=class_names, is_train=False),
    dict(type='LoadOccGTFromFile', num_classes=len(occ_class_names)),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['img', 'voxel_semantics', 'voxel_instances', 'instance_class_ids'], meta_keys=(
                'filename', 'ori_shape', 'img_shape', 'pad_shape', 
                'lidar2img', 'lidar2img_t1', 'lidar2img_t2', 'img_timestamp',
                'ego_pose', 'ego_pose_t1', 'ego_pose_t2', 'ego2lidar'))
        ])
]

data = dict(
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=dataset_root,
        occ_gt_root=occ_gt_root,
        ann_file=dataset_root + 'nuscenes_infos_train_sweep.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=dataset_root,
        occ_gt_root=occ_gt_root,
        ann_file=dataset_root + 'nuscenes_infos_val_sweep.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=dataset_root,
        occ_gt_root=occ_gt_root,
        ann_file=dataset_root + 'nuscenes_infos_test_sweep.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.2),
        'sampling_offset': dict(lr_mult=0.1),
    }),
    weight_decay=0.01
)

optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale=512.0,
    grad_clip=dict(max_norm=35, norm_type=2)
)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 20, 22],
    gamma=0.4)
total_epochs = 24
batch_size = 8

# load pretrained weights
load_from = 'pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
revise_keys = [('backbone', 'img_backbone')]

# resume the last training
resume_from = None

# checkpointing
checkpoint_config = dict(interval=1, max_keep_ckpts=1)

# logging
log_config = dict(
    interval=1,
    hooks=[
        dict(type='MyTextLoggerHook', interval=1, reset_flag=True),
        dict(type='MyTensorboardLoggerHook', interval=500, reset_flag=True)
    ]
)

# evaluation
eval_config = dict(interval=total_epochs)

# other flags
debug = False