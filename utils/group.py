import numpy as np
import torch

def match_format(dic):
    loc = dic['loc_k'][0,:,0,:]
    val = dic['val_k'][0,:,:]
    ans = np.hstack((loc, val))
    ans = np.expand_dims(ans, axis = 0) 
    ret = []
    ret.append(ans)
    return ret

class HeatmapParser:
    def __init__(self):
        from torch import nn
        self.pool = nn.MaxPool2d(3, 1, 1)

    def nms(self, det):
        maxm = self.pool(det)
        maxm = torch.eq(maxm, det).float()
        det = det * maxm
        return det

    def calc(self, det):
        det = torch.autograd.Variable(torch.Tensor(det), volatile=True)

        det = self.nms(det)
        h = det.size()[2]
        w = det.size()[3]
        det = det.view(det.size()[0], det.size()[1], -1)
        val_k, ind = det.topk(1, dim=2)

        x = ind % w
        y = (ind / w).long()
        ind_k = torch.stack((x, y), dim=3)
        ans = {'loc_k': ind_k, 'val_k': val_k}
        return {key:ans[key].cpu().data.numpy() for key in ans}

    def adjust(self, ans, det):
        for batch_id, people in enumerate(ans): 
            for people_id, i in enumerate(people):
                for joint_id, joint in enumerate(i):
                    if joint[2]>0:
                        y, x = joint[0:2]
                        xx, yy = int(x), int(y)
                        tmp = det[0][joint_id]
                        if tmp[xx, min(yy+1, tmp.shape[1]-1)]>tmp[xx, max(yy-1, 0)]:
                            y+=0.25
                        else:
                            y-=0.25

                        if tmp[min(xx+1, tmp.shape[0]-1), yy]>tmp[max(0, xx-1), yy]:
                            x+=0.25
                        else:
                            x-=0.25
                        ans[0][0, joint_id, 0:2] = (y+0.5, x+0.5)
        return ans

    def parse(self, det, adjust=True):
        ans = match_format(self.calc(det))
        if adjust:
            ans = self.adjust(ans, det)
        return ans
