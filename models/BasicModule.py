import time

import torch.nn as nn
import torch as t


class BasicModule(nn.Module):

    def __init__(self, name):
        super(BasicModule, self).__init__()
        if name is None:
            self.model_name = str(type(self))
        else:
            self.model_name = name

    def load(self, path):
        self.load_state_dict(t.load(path))
        # for name, para in self.named_parameters():
        #     print('load state successfully, parameters: %s->%s' % (name, para))

    def save(self, name=None):
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
            # for para_name, para in self.named_parameters():
            #     print('saving state in file %s, parameters: %s -> %s' % (name, para_name, para))
            t.save(self.state_dict(), name)
            return name
