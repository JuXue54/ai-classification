import warnings
import sys


class DefaultConfig(object):
    env = 'default'
    model = 'AlexNet'
    # model = 'resnet34'
    if sys.platform.startswith('linux'): 
        print('current os is linux')
        train_data_root = '/mnt/e/data/dogs-vs-cats-redux-kernels-edition/train'
        test_data_root = '/mnt/e/data/dogs-vs-cats-redux-kernels-edition/train'
        load_model_path = None
        use_gpu = True
    elif sys.platform.startswith('darwin'):
        print('current os is macos')
        train_data_root = '/Users/jon/data/dogs-vs-cats-redux-kernels-edition/train'
        test_data_root = '/Users/jon/data/dogs-vs-cats-redux-kernels-edition/test'
        load_model_path = 'checkpoints/AlexNet_0621_23:42:18.pth'
        use_gpu = True
    else:
        raise Exception('Unsupported OS!!!')
    # load_model_path = None
    need_save = True
    batch_size = 32
    num_workers = 4
    print_freq = 20
    debug_file = '/tmp/debug'
    result_file = 'result.csv'
    max_epoch = 10
    lr = 0.001
    lr_decay = 0.95
    weight_decay = 1e-4

    def parse(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has no attribute %s" % k)
            setattr(self, k, v)
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))
