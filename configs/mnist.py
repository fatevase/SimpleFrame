
Net=dict(
    type='Classifier',
    classify=10,
)
TDataset=dict(
    type='MMNISTDataset',
    root='MNIST',
    download=True,
    test_mode=False,
    pipeline = [
        dict(type='ReInDict', img_label='target'),
        dict(type='ToTensor'),
        dict(type='ReOutDict', img='input'),
    ],
)
VDataset=dict(
    type='MMNISTDataset',
    root='MNIST',
    download=True,
    test_mode=True,
    pipeline = [
        dict(type='ReInDict', img_label='target'),
        dict(type='ToTensor'),
        dict(type='ReOutDict', img='input'),
    ],
)
Optimizer=dict(
    type='SGD',
    lr=0.03,
    momentum=0.9,
#     type='AdamW',
#     lr=0.001,
#     weight_decay=0.01,
)
Scheduler=[
    dict(type='LinearLR',
         start_factor=0.001,
         by_epoch=False,
         begin=0,
         end=2500
    ),
    # 在 [100, 900) 迭代时使用余弦学习率
    dict(type='CosineAnnealingLR',
         T_max=3000*10,
         by_epoch=False,
         begin=2000,
         end=2500*20
    ),
]

Metric=dict(
    type='MnistAccuracy',
)

Visbackend=dict(
    type='Visualizer',
    vis_backends=[
        dict(type='WandbVisBackend',
        save_dir='./train_mnist',
        init_kwargs=dict(
            project="macbook",
        )
    )]
)
train_cfg=dict(by_epoch=True, max_epochs=20, val_interval=4)
custom_imports=dict(imports=['models', 'datasets', 'utils', 'app', 'hooks'], allow_failed_imports=False)
