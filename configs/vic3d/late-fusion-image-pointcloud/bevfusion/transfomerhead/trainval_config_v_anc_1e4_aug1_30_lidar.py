dataset_type = "KittiDataset"
data_root = "./data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side/"
class_names = ["Pedestrian", "Cyclist", "Car"]
# point_cloud_range= [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
# voxel_size= [0.075, 0.075, 0.2]
point_cloud_range = [0, -39.68, -3, 92.16, 39.68, 1]
# voxel_size = [0.16, 0.16, 4]
# voxel_size = [0.1, 0.1, 0.1]
voxel_size = [0.064, 0.05511, 0.1]
# voxel_size = [0.05, 0.05, 0.1]
length = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])  # 576
height = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])  # 496
z_num = int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2])
image_size = [256, 704]
# image_size = [height, length]
output_shape = [height, length]
z_center_pedestrian = -0.6
z_center_cyclist = -0.6
z_center_car = -1.78

work_dir = "/workspace/mnt/storage/guangcongzheng/zju_zgc/DAIR-V2X/checkpoints/vehicle-side/anc_1e4_aug1_30_fix_data"

model = dict(
    type="BEVFusionNew",
    encoders=dict(
  
        lidar=dict(
            voxelize=dict(
                max_num_points=5,
                point_cloud_range=point_cloud_range,
                voxel_size=voxel_size,
                max_voxels=(16000, 40000)
            ),
            voxel_encoder=dict(type='HardSimpleVFE'),
            middle_encoder=dict(
                type='SparseEncoder',
                in_channels=4,
                sparse_shape=[z_num+1, height, length],
                order=('conv', 'norm', 'act')),
            #走HardSimpleVFE SparseEncoder 不走backbone
            # backbone=dict(
            #     type='SparseEncoderBEVFusion',
            #     in_channels=4,
            #     sparse_shape=[length, height,z_num+1],
            #     output_channels=128,
            #     order=['conv', 'norm', 'act'],

            # ),
        ),

    ),
    fuser=None,
    heads=dict(
        map=None,
        object=dict(
            type='TransFusionHead',
            num_proposals=200,
            auxiliary=True,
            in_channels=512,
            hidden_channel=128,
            num_classes=3,
            num_decoder_layers=1,
            num_heads=8,
            nms_kernel_size=3,
            ffn_channel=256,
            dropout=0.1,
            bn_momentum=0.1,
            activation='relu',
            train_cfg=dict(
                dataset='Waymo',
                point_cloud_range=point_cloud_range,
                grid_size=[length, height, z_num+1],
                # voxel_size= [0.075, 0.075, 0.2],
                voxel_size=voxel_size,
                out_size_factor=8,
                gaussian_overlap=0.1,
                min_radius=2,
                pos_weight=-1,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                assigner=dict(
                    type='HungarianAssigner3D',
                    iou_calculator=dict(
                        type='BboxOverlaps3D',
                        coordinate='lidar'
                    ),
                    cls_cost=dict(
                        type='FocalLossCost',
                        gamma=2.0,
                        alpha=0.25,
                        weight=0.15
                    ),
                    reg_cost=dict(
                        type='BBoxBEVL1Cost',
                        weight=0.25
                    ),
                    iou_cost=dict(
                        type='IoU3DCost',
                        weight=0.25
                    )
                ),
            ),
            test_cfg=dict(
                dataset='Waymo',
                grid_size=[length, height, z_num+1],
                out_size_factor=8,
                # voxel_size= [0.075, 0.075],
                voxel_size=voxel_size[:2],
                #这里对吗
                pc_range=point_cloud_range[:2],
                nms_type=None
            ),
            common_heads=dict(
                center=[2, 2],
                height=[1, 2],
                dim=[3, 2],
                rot=[2, 2],
                # vel= [2, 2],
            ),
            bbox_coder=dict(
                type='TransFusionBBoxCoder',
                pc_range=point_cloud_range[:2],
                post_center_range=[-10, -49.68, -13, 102.16, 49.68, 11.0],
                score_threshold=0.0,
                out_size_factor=8,
                voxel_size=voxel_size[:2],
                code_size=8
            ),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                reduction='mean',
                loss_weight=1.0
            ),
            loss_heatmap=dict(
                type='GaussianFocalLoss',
                reduction='mean',
                loss_weight=1.0
            ),
            loss_bbox=dict(
                type='L1Loss',
                reduction='mean',
                loss_weight=0.25
            ),
        )
    ),

    train_cfg=dict(
        assigner=[
            dict(
                type="MaxIoUAssigner",
                iou_calculator=dict(type="BboxOverlapsNearest3D"),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1,
            ),
            dict(
                type="MaxIoUAssigner",
                iou_calculator=dict(type="BboxOverlapsNearest3D"),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1,
            ),
            dict(
                type="MaxIoUAssigner",
                iou_calculator=dict(type="BboxOverlapsNearest3D"),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1,
            ),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50,
    ),

    decoder=dict(
        backbone=dict (
            type='SECOND',
            in_channels= 256,
            out_channels= [128, 256],
            layer_nums= [5, 5],
            layer_strides= [1, 2],
            norm_cfg= dict(
                type= 'BN',
                eps= 0.001,
                momentum= 0.01
            ),
            conv_cfg=dict (
                type= 'Conv2d',
                bias= False
            ),
        ),
        neck=dict(
            type= 'SECONDFPN',
            in_channels= [128, 256],
            out_channels= [256, 256],
            upsample_strides= [1, 2],
            norm_cfg=dict(
                type= 'BN',
                eps= 0.001,
                momentum= 0.01
            ),
            upsample_cfg=dict(
                type= 'deconv',
                bias=False
            ),
            use_conv_for_no_stride= True
        ),
    ),

)

file_client_args = dict(backend="disk")
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        type="RepeatDataset",
        times=2,
        dataset=dict(
            type="KittiDataset",
            data_root=data_root,
            ann_file=data_root + "/kitti_infos_train.pkl",
            split="training",
            pts_prefix="velodyne_reduced",
            pipeline=[
                dict(type="LoadImageFromFileBEVFusion",to_float32=True),
                dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4),
                dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True,with_attr_label=False),
                # dict(
                #     type="ObjectSample",
                #     db_sampler=dict(
                #         data_root=data_root,
                #         info_path=data_root + "/kitti_dbinfos_train.pkl",
                #         rate=1.0,
                #         prepare=dict(
                #             filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)
                #         ),
                #         classes=class_names,
                #         sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
                #     ),
                # ),
                dict(type="loadcalibBEVFusion"),
                dict(type="ImageAug3D", final_dim=[256, 704], resize_lim=[0.38, 0.55], bot_pct_lim=[0.0, 0.0],rot_lim=[-5.4, 5.4],rand_flip=True,is_train=True),
                dict(type="GlobalRotScaleTransBEVFusion", resize_lim=[0.9, 1.1], rot_lim= [-0.78539816, 0.78539816], trans_lim=0.5,is_train=False),
                dict(type="RandomFlip3DBEVFusion"),
                dict(type="PointsRangeFilter",point_cloud_range=point_cloud_range,  ),
                dict(type="ObjectRangeFilter",point_cloud_range =point_cloud_range ),
                dict(type="ImageNormalize",mean= [0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225] ),
                dict(type="GridMask",use_h=True, use_w=True,max_epoch=6,rotate=1,offset=False,ratio=0.5,mode=1,prob=0,fixed_prob=True),
                dict(type="PointShuffle") ,
                dict(type="DefaultFormatBundle3DBEVFusion", classes=class_names),
                dict(type="Collect3DBEVFusion", keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],\
                     meta_keys= ['camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera', 'camera2lidar', 'lidar2image', 'img_aug_matrix', 'lidar_aug_matrix']),
            ],
            modality=dict(use_lidar=True, use_camera=True),
            classes=class_names,
            test_mode=False,
            pcd_limit_range=point_cloud_range,
            box_type_3d="LiDAR",
        ),
    ),
    val= dict(
        type="KittiDataset",
        data_root=data_root,
        ann_file=data_root + "/kitti_infos_val.pkl",
        split="training",
        pts_prefix="velodyne_reduced",
        pipeline=[
            dict(type="LoadImageFromFileBEVFusion",to_float32=True),
            dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4),
            dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True,with_attr_label=False),
            # dict(
            #     type="ObjectSample",
            #     db_sampler=dict(
            #         data_root=data_root,
            #         info_path=data_root + "/kitti_dbinfos_train.pkl",
            #         rate=1.0,
            #         prepare=dict(
            #             filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)
            #         ),
            #         classes=class_names,
            #         sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
            #     ),
            # ),
            dict(type="loadcalibBEVFusion"),
            dict(type="ImageAug3D", final_dim=[256, 704], resize_lim=[0.48, 0.48], bot_pct_lim=[0.0, 0.0],rot_lim=[0, 0],rand_flip=False,is_train=False),
            dict(type="GlobalRotScaleTransBEVFusion", resize_lim=[1.0, 1.0], rot_lim= [0.0, 0.0], trans_lim=0,is_train=False),
            # dict(type="RandomFlip3DBEVFusion"),
            dict(type="PointsRangeFilter",point_cloud_range=point_cloud_range,  ),
            dict(type="ObjectRangeFilter",point_cloud_range =point_cloud_range ),
            dict(type="ImageNormalize",mean= [0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225] ),
            # dict(type="GridMask",use_h=True, use_w=True,max_epoch=6,rotate=1,offset=False,ratio=0.5,mode=1,prob=0,fixed_prob=True),
            # dict(type="PointShuffle") ,
            dict(type="DefaultFormatBundle3DBEVFusion", classes=class_names),
            dict(type="Collect3DBEVFusion", keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],\
                    meta_keys= ['camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera', 'camera2lidar', 'lidar2image', 'img_aug_matrix', 'lidar_aug_matrix']),
        ],
        modality=dict(use_lidar=True, use_camera=True),
        classes=class_names,
        test_mode=False,
        pcd_limit_range=point_cloud_range,
        box_type_3d="LiDAR",
        
    ),
    test= dict(
        type="KittiDataset",
        data_root=data_root,
        ann_file=data_root + "/kitti_infos_val.pkl",
        split="training",
        pts_prefix="velodyne_reduced",
        pipeline=[
            dict(type="LoadImageFromFileBEVFusion",to_float32=True),
            dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4),
            dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True,with_attr_label=False),
            # dict(
            #     type="ObjectSample",
            #     db_sampler=dict(
            #         data_root=data_root,
            #         info_path=data_root + "/kitti_dbinfos_train.pkl",
            #         rate=1.0,
            #         prepare=dict(
            #             filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)
            #         ),
            #         classes=class_names,
            #         sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
            #     ),
            # ),
            dict(type="loadcalibBEVFusion"),
            dict(type="ImageAug3D", final_dim=[256, 704], resize_lim=[0.48, 0.48], bot_pct_lim=[0.0, 0.0],rot_lim=[0, 0],rand_flip=False,is_train=False),
            dict(type="GlobalRotScaleTransBEVFusion", resize_lim=[1.0, 1.0], rot_lim= [0.0, 0.0], trans_lim=0,is_train=False),
            # dict(type="RandomFlip3DBEVFusion"),
            dict(type="PointsRangeFilter",point_cloud_range=point_cloud_range,  ),
            dict(type="ObjectRangeFilter",point_cloud_range =point_cloud_range ),
            dict(type="ImageNormalize",mean= [0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225] ),
            # dict(type="GridMask",use_h=True, use_w=True,max_epoch=6,rotate=1,offset=False,ratio=0.5,mode=1,prob=0,fixed_prob=True),
            # dict(type="PointShuffle") ,
            dict(type="DefaultFormatBundle3DBEVFusion", classes=class_names),
            dict(type="Collect3DBEVFusion", keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],\
                    meta_keys= ['camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera', 'camera2lidar', 'lidar2image', 'img_aug_matrix', 'lidar_aug_matrix']),
        ],
        modality=dict(use_lidar=True, use_camera=True),
        classes=class_names,
        test_mode=True,
        pcd_limit_range=point_cloud_range,
        box_type_3d="LiDAR",
        
    ),
)
evaluation = dict(
    interval=5,
    pipeline=[
        dict(type="LoadImageFromFileBEVFusion",to_float32=True),
        dict(
            type="LoadPointsFromFile", coord_type="LIDAR", load_dim=4, use_dim=4, file_client_args=dict(backend="disk")
        ),
        dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True,with_attr_label=False),
        dict(type="loadcalibBEVFusion"),
        dict(type="ImageAug3D", final_dim=[256, 704], resize_lim=[0.38, 0.55], bot_pct_lim=[0.0, 0.0],rot_lim=[-5.4, 5.4],rand_flip=False,is_train=False),
        dict(type="GlobalRotScaleTransBEVFusion", resize_lim=[1.0, 1.0], rot_lim= [0.0, 0.0], trans_lim=0,is_train=False),
        dict(type="PointsRangeFilter",point_cloud_range=point_cloud_range,  ),
        dict(type="ImageNormalize",mean= [0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225] ),

        # dict(type="ObjectRangeFilter",point_cloud_range =point_cloud_range ),
        dict(type="DefaultFormatBundle3DBEVFusion", class_names=class_names),
        dict(type="Collect3DBEVFusion", keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],\
                    meta_keys= ['camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera', 'camera2lidar', 'lidar2image', 'img_aug_matrix', 'lidar_aug_matrix']),
    ],
)
lr = 0.0002
optimizer = dict(type="AdamW", lr=0.0002, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy="CosineAnnealing", warmup="linear", warmup_iters=500, warmup_ratio=0.33333333,min_lr_ratio=0.001)
momentum_config = dict(policy="cyclic")
runner = dict(type="EpochBasedRunner", max_epochs=30)
checkpoint_config = dict(interval=5)
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")])
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]
gpu_ids = range(0, 8)
