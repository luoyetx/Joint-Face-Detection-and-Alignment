# pylint: disable=bad-indentation, no-member, invalid-name, line-too-long
from .detector import JfdaDetector
import minicaffe as caffe


class MiniCaffeDetector(JfdaDetector):

    def __init__(self, nets):
        assert len(nets) in [2, 4, 6, 8], 'wrong number of nets'
        self.pnet, self.rnet, self.onet, self.lnet = None, None, None, None
        if len(nets) >= 2:
            self.pnet = caffe.Net(nets[0], nets[1])
        if len(nets) >= 4:
            self.rnet = caffe.Net(nets[2], nets[3])
        if len(nets) >= 6:
            self.onet = caffe.Net(nets[4], nets[5])
        if len(nets) >= 8:
            self.lnet = caffe.Net(nets[6], nets[7])
        self.pnet_single_forward = False
