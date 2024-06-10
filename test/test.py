from utils.config import Config
import pickle
import numpy as np
cfg = Config(filename='/Users/vase/Documents/Coding/Python/SimpleFrame/configs/classify.py')
print(cfg)
print(type(cfg))
print(cfg.hasKey('cdict.t'))


# def _serialize(data):
#     buffer = pickle.dumps(data, protocol=4)
#     return np.frombuffer(buffer, dtype=np.uint8)
# n = 6000
# t_data = np.random.randint(0, 255, (n, 32, 32))

# data_list = []
# for i in range(n):
#     data_list.append(dict(img=t_data[i], img_label=i))

# data_list = [_serialize(x) for x in data_list]

t = np.random.randint(0, 2550, (32, 32))
d = dict()

print(pickle.dumps(d, protocol=4), pickle.dumps(t.tobytes(), protocol=4))