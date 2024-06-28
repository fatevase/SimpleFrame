
Net=dict(
    type='Classifier',
    in_channels=3,
    classify=100,
    in_shape=(32, 32),
    attn_shape=(16, 16),
    attention_in=64,
    attention_times=1,
    multi_head=2,
    embad_type='conv',
)


TDataset=dict(
    type='Cifar10Dataset',
    data_root='D:/datasets/cifar100',
    test_mode=False,
    data_prefix=dict(),
    ann_list=['train'],
    pipeline = [
        dict(type='ReInDict', img_label='target'),
        # dict(type='Resize', size=(224, 224)),
        dict(type='RandomHorizontalFlip', p=0.5, keys=['img']),
        dict(type='RandRotation', p=0.5, angle=(-20, 20), keys=['img']),
        dict(type='RanddjustSharpness', p=0.5, alpha=(0.2, 0.5), keys=['img']),
        dict(type='RandErasing', p=0.5, max_holes=2, min_holes=0, max_width=8, min_width=0, keys=['img']),
        dict(type='Normalize',img_norm=dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])),
        dict(type='ToTensor'),
        dict(type='ReOutDict', img='input'),
    ],
)
VDataset=dict(
    type='Cifar10Dataset',
    data_root='D:/datasets/cifar100',
    test_mode=True,
    data_prefix=dict(),
    ann_list=['test'],
    pipeline = [
        dict(type='ReInDict', img_label='target'),
        dict(type='Normalize',img_norm=dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])),
        dict(type='ToTensor'),
        dict(type='ReOutDict', img='input'),
    ],
)
Optimizer=dict(
    # type='SGD',
    # lr=1e-2,
    # momentum=0.9,
    type='AdamW',
    lr=1e-2,
    weight_decay=0.01,
)
Scheduler=[
    dict(type='LinearLR',
         start_factor=0.001,
         by_epoch=True,
         convert_to_iter_based=True,
         begin=0,
         end=5
    ),
    # 在 [100, 900) 迭代时使用余弦学习率
    dict(type='CosineAnnealingLR',
         T_max=198,
         by_epoch=True,
         convert_to_iter_based=True,
         begin=5,
         end=200
    ),
]

Metric=dict(
    type='ClassifyAccuracy',
)

Visbackend=dict(
    type='Visualizer',
    vis_backends=[
        dict(type='WandbVisBackend',
        save_dir='./logs/train_cifar10',
        init_kwargs=dict(
            project="macbook",
        )
    )]
)
train_cfg=dict(by_epoch=True, max_epochs=200, val_interval=4)
custom_imports=dict(imports=['models', 'datasets', 'utils', 'app', 'hooks'], allow_failed_imports=False)

batch_size=100
work_dir='./logs/train_cifar10'
