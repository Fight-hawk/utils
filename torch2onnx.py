import torch.onnx
from model.segnet_old import SegMattingNet
from collections import OrderedDict

net = SegMattingNet()

pthfile = r'ckpt_lastest.pth'
model_dict = torch.load(pthfile, map_location='cpu')['state_dict']
new_model_dict = OrderedDict()
for k, v in model_dict.items():
    if 'edge' not in k:
        new_model_dict[k] = v

net.load_state_dict(new_model_dict)
net.eval()
dummy_input1 = torch.randn(1, 3, 256, 256)
input_names = ["input"]
output_names = ["seg", 'alpha']
torch.onnx.export(net, dummy_input1, "tiny.onnx", input_names=input_names, output_names=output_names, verbose=True)

