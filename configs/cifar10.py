
Net=dict(
    type='Classifier',
    in_channels=3,
    classify=10,
    activate='relu',
)


TDataset=dict(
    type='Cifar10Dataset',
    data_root='D:/datasets/cifar10',
    test_mode=False,
    data_prefix=dict(),
    ann_list=['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'],
    pipeline = [
        dict(type='ReInDict', img_label='target'),
        # dict(type='Resize', size=(224, 224)),
        dict(type='Normalize',img_norm=dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])),
        dict(type='ToTensor'),
        dict(type='ReOutDict', img='input'),
    ],
)
VDataset=dict(
    type='Cifar10Dataset',
    data_root='D:/datasets/cifar10',
    test_mode=True,
    data_prefix=dict(),
    ann_list=['test_batch'],
    pipeline = [
        dict(type='ReInDict', img_label='target'),
        # dict(type='Resize', size=(224, 224)),
        dict(type='Normalize',img_norm=dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])),
        dict(type='ToTensor'),
        dict(type='ReOutDict', img='input'),
    ],
)
Optimizer=dict(
    # type='SGD',
    # lr=0.03,
    # momentum=0.9,
    type='AdamW',
    lr=3e-4,
    weight_decay=0.01,
)
Scheduler=[
    dict(type='LinearLR',
         start_factor=0.001,
         by_epoch=False,
         begin=0,
         end=2000
    ),
    # 在 [100, 900) 迭代时使用余弦学习率
    dict(type='CosineAnnealingLR',
         T_max=3000*10,
         by_epoch=False,
         begin=2000,
         end=3000*10
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
train_cfg=dict(by_epoch=True, max_epochs=1, val_interval=1)
custom_imports=dict(imports=['models', 'datasets', 'utils', 'app', 'hooks'], allow_failed_imports=False)

batch_size=20
work_dir='./logs/train_cifar10'
