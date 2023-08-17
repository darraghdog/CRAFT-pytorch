import os
'''
os.chdir('/Users/dhanley/Documents/dread/CRAFT-pytorch')
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from basenet.vgg16_bn import vgg16_bn, init_weights

from craft import CRAFT
import numpy as np
from PIL import Image
import imgproc
import onnxruntime

wts = torch.load('weights/craft_mlt_25k.pth', map_location=torch.device('cpu'))
wts = {k.replace('module.', ''):v for k,v in wts.items()}
model = CRAFT(pretrained=False)#.cuda()
model.load_state_dict(wts)
model.eval()

img = Image.open('sample/B1154001572.jpeg').convert('RGB')
# img.thumbnail(tuple((i//2 for i in img.size)), Image.Resampling.LANCZOS)


img_in = imgproc.normalizeMeanVariance(np.array(img))

torch_input = torch.from_numpy(np.array(img_in)).permute(2,0,1)[None,:,:,:]

%timeit with torch.no_grad(): output, _ = model(torch_input)
# (1496, 505) - 2.11 s ± 22.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
with torch.no_grad(): 
    torch_output, _ = model(torch_input)#.cuda())
print(torch_output.shape)


# pip3 install onnxoptimizer
# Onnx convert

if False:
    # Export the model
    x = torch.randn(1, 3, 1024, 1024, requires_grad=False)
    torch.onnx.export(model,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      "weights/craft_mlt_25k.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=13,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size', 2 : 'height', 3 : 'width'}, 
                                    'output' : {0 : 'batch_size', 2 : 'height', 3 : 'width'},})
    '''
    # Run the below for another slight increase
    pip3 install onnxoptimizer
    python -m onnxoptimizer weights/craft_mlt_25k.onnx weights/craft_mlt_25k_opt.onnx
    '''


ort_session = onnxruntime.InferenceSession("weights/craft_mlt_25k.onnx")

# compute ONNX Runtime output prediction
onnx_input = torch_input.numpy()
ort_outs = ort_session.run(None, {'input': onnx_input})

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(torch_output.numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)

%timeit ort_outs = ort_session.run(None, {'input': onnx_input})
# (1496, 505) - 1.26 s ± 20.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# 


# Visualise results
viz_torch_output = torch.cat([torch_output[0]]*2, -1)[:,:,:3]
viz_onnx_output = np.concatenate([ort_outs[0][0]]*2, -1)[:,:,:3]

img
Image.fromarray(imgproc.denormalizeMeanVariance(viz_torch_output.numpy()))
Image.fromarray(imgproc.denormalizeMeanVariance(viz_onnx_output))

