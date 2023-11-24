---
title: "dynamic|static X fp16|fp32 Tensrort Engine Inference Using Yolov5 Model"
date: "2023-11-23"
author: "🤽奔波鲅"
tags: ['tensorrt', 'deploy', 'model-convert', 'deeplearning', 'yolov5', '2d-detection']
---
    experiment condition:
        python==3.8.0
        torch==2.1.0
        ultralytics:77fc5ccf02ac0fdf9e7bb1eeb003e9bf3e719702
        tensorrt==8.6.1.post1

#

|        |**dynamic**|**static**| 
|:-------|:--------:|---------:|
|**fp16**| [dynamic_fp16](#dynamic_fp16)| [static_fp16](#static_fp16)    |
|**fp32**|                              | [static_fp32](#static_fp32)   |

另外训练时候的padding模式决定了模型inference时候的前处理形式，进而导致不同的运行时间,在letterbox函数进行
前处理的时候，以下代码都保证了模型的正确输出，但并不能保证此种调用方式是效率最佳的方式，还需要继续优化。

## Dynamic_Fp16

    inshape from get_binding_shape(0): (1, 3, -1, 640)
    outshape from get_binding_shape(1): (0)
    dynamic inshape: (1, 3, 640, 640)
    dynamicoutshape: (1, 7, 8400)
    trt_run.py:71: DeprecationWarning: Use set_input_shape instead.
      context.set_binding_shape(0, image.shape)
    trt_run.py:72: DeprecationWarning: Use set_input_shape instead.
      context.set_binding_shape(1, output.shape)
    [11/23/2023-23:19:05] [TRT] [E] 3: [executionContext.cpp::setBindingDimensions::1511] Error Code 3: API Usage Error (Parameter check failed at: runtime/api/executionContext.cpp::setBindingDimensions::1511, condition: mEngine.bindingIsInput(bindingIndex)
    )
    set binding time: 0.010830402374267578
    preprocess time: 0.03390836715698242
    infer time: 0.08650636672973633
    pred shape (7, 8400)
    postprocess time: 0.439532995223999
    1.jpg
    total time: 0.6134130954742432


```py
import tensorrt as trt
import os
import time
import tensorrt as trt
import pycuda.driver as cuda
#import pycuda.driver as cuda2
import pycuda.autoinit
from ultralytics.utils import ops 
import numpy as np
import cv2
import tqdm
from utils.augmentations import letterbox
def plot_with_xyxys(xyxys, img_path):
    objs = xyxys
    img = cv2.imread(img_path)
    for obj in objs:
        img = cv2.rectangle(img, \
            ( int(obj[0]), int(obj[1])), \
            ( int(obj[2]), int(obj[3])),\
            color=(0,0,255), thickness=2)
    print(img_path.split('/')[-1])
    if not os.path.exists(img_path):
        cv2.imwrite(img_path.split('/')[-1], img)
    else:
        cv2.imwrite('new_'+img_path.split('/')[-1], img)

def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

path = "/media/lvsolo/CA89-5817/datasets/helmet/hard-hat-detection/codes/runs/detect/train15/weights/best.engine.fp16_static_input"#
#path = "/media/lvsolo/CA89-5817/datasets/helmet/hard-hat-detection/codes/runs/detect/train15/weights/best.engine.dynamicinput"#
#path = "/media/lvsolo/CA89-5817/datasets/helmet/hard-hat-detection/codes/runs/detect/train15/weights/best.engine.fp32"#
engine = load_engine(path)
imgpath = '1.jpg'
#imgpath = 'R-C.jpg'

context = engine.create_execution_context()
image1 = cv2.imread(imgpath)

total_st = time.time()
st = total_st
"""auto=False表示会padding到640*640,耗时变大,但是如果模型训练时候是False即padding模式那就只能是False"""
image = letterbox(image1, (640,640), stride=32, auto=False)[0]
image = image / 255.
image = image[:,:,::-1].transpose((2, 0, 1))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
image = np.ascontiguousarray(image)
image = np.expand_dims(image, axis=0)

# tensorrt8.6.1即使是fp16的engine也不能直接使用fp16输入，8.5版本貌似可以
image = image.astype(np.float32)
print('input image type:', image.dtype)
print('input image shape:', image.shape)

"""动态输入通过get_binding_shape(0|1)时获得的值可能是0或者包含-1,此时应指定输入输出的大小为实际大小,
并通过set_binding_shape方式进行指定
"""
#outshape = context.get_tensor_shape('output0')
outshape= (1,7,8400)#context.get_binding_shape(1) 
print('outshape:',outshape)
org_inshape= context.get_binding_shape(0)
org_outshape= context.get_binding_shape(1) 
print('inshape from get_binding_shape(0):', org_inshape)
print('outshape from get_binding_shape(1):', org_outshape)
output = np.empty((outshape), dtype=np.float32)

"""动态输入通过指定输入输出的大小为实际大小
"""
st_set_binding = time.time()
context.set_binding_shape(0, image.shape)
context.set_binding_shape(1, output.shape)
print('set binding time:', time.time() - st_set_binding)
#print(outshape, output.size, image.size, image.dtype.itemsize, output.dtype.itemsize)

d_input = cuda.mem_alloc(1 * image.size * image.dtype.itemsize)
d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)

bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()
st = time.time()
cuda.memcpy_htod(d_input,image)
print('preprocess time:',time.time()- st, flush=True)
st = time.time()
context.execute_v2(bindings)
print('infer time:',time.time()- st, flush=True)
st = time.time()
cuda.memcpy_dtoh(output, d_output)
import torch
pred = output
print('pred shape', pred[0].shape, flush=True)
pred = torch.from_numpy(pred).cuda()
#TODO 如何通过内存地址、数据类型和数据size来init一个torch.cuda.tensor
#pred = torch.cuda.FloatTensor(d_output, output.size)
pred = ops.non_max_suppression(pred,
                               conf_thres=0.2,#self.args.conf,
                               iou_thres=0.5,#self.args.iou,
                               classes=None,
                               agnostic=True,#agnostic=False,#self.args.agnostic_nms,
                               max_det=300,#self.args.max_det,
                               nc=3
                               )#classes=None)#self.args.classes)[0]
print('postprocess time:',time.time()- st)
#visualization
xyxys = []
confs = []
cls = []
for i, det in enumerate(pred):
    #print('before scalebox:', det)
    det[:,:4] = ops.scale_boxes(image.shape[2:], det[:,:4], image1.shape).round()
    #print('after scalebox:', det)
    xyxys.append(det[:,:4].cpu().numpy().tolist())
    confs.append(det[:,4].cpu().numpy().tolist())
    cls.append(det[:,5].cpu().numpy().astype(np.int32).tolist())
xyxys = xyxys[0]
confs = confs[0]
cls = cls[0]
plot_with_xyxys(xyxys, imgpath)

print(time.time()- st)
```

## Static_Fp32
    inshape from get_binding_shape(0): (1, 3, 640, 640)
    outshape from get_binding_shape(1): (1, 7, 8400)
    dynamic inshape: (1, 3, 640, 640)
    dynamicoutshape: (1, 7, 8400)
    preprocess time: 0.029306411743164062
    infer time: 0.1014707088470459
    pred shape (7, 8400)
    postprocess time: 0.41349053382873535
    1.jpg
    0.5975463390350342
 

```python
import tensorrt as trt
import os
import time
import tensorrt as trt
import pycuda.driver as cuda
#import pycuda.driver as cuda2
import pycuda.autoinit
from ultralytics.utils import ops 
import numpy as np
import cv2
import tqdm
from utils.augmentations import letterbox
def plot_with_xyxys(xyxys, img_path):
    objs = xyxys
    img = cv2.imread(img_path)
    for obj in objs:
        img = cv2.rectangle(img, \
            ( int(obj[0]), int(obj[1])), \
            ( int(obj[2]), int(obj[3])),\
            color=(0,0,255), thickness=2)
    print(img_path.split('/')[-1])
    if not os.path.exists(img_path):
        cv2.imwrite(img_path.split('/')[-1], img)
    else:
        cv2.imwrite('new_'+img_path.split('/')[-1], img)

def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

#path = "/media/lvsolo/CA89-5817/datasets/helmet/hard-hat-detection/codes/runs/detect/train15/weights/best.engine.fp16_static_input"#
#path = "/media/lvsolo/CA89-5817/datasets/helmet/hard-hat-detection/codes/runs/detect/train15/weights/best.engine.dynamicinput"#
path = "/media/lvsolo/CA89-5817/datasets/helmet/hard-hat-detection/codes/runs/detect/train15/weights/best.engine.fp32"#
engine = load_engine(path)
imgpath = '1.jpg'
#imgpath = 'R-C.jpg'

context = engine.create_execution_context()
image1 = cv2.imread(imgpath)
total_st = time.time()
st = total_st
"""auto=False表示会padding到640*640,耗时变大,但是如果模型训练时候是False即padding模式那就只能是False"""
image = letterbox(image1, (640,640), stride=32, auto=False)[0]
"""对于static input的模型，在不进行特殊内存处理的情况下，输入图片尺寸必须等于static input shape, 因此auto=False"""
image = letterbox(image1, (640,640), stride=32, auto=False)[0]
image = image / 255.
image = image[:,:,::-1].transpose((2, 0, 1))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
image = np.ascontiguousarray(image)
image = np.expand_dims(image, axis=0)

# tensorrt8.6.1即使是fp16的engine也不能直接使用fp16输入，8.5版本貌似可以
image = image.astype(np.float32)
print('input image type:', image.dtype)
print('input image shape:', image.shape)

"""动态输入通过get_binding_shape(0|1)时获得的值可能是0或者包含-1,此时应指定输入输出的大小为实际大小,
并通过set_binding_shape方式进行指定
"""
#outshape = context.get_tensor_shape('output0')
outshape= (1,7,8400)#context.get_binding_shape(1) 
print('outshape:',outshape)
org_inshape= context.get_binding_shape(0)
org_outshape= context.get_binding_shape(1) 
output = np.empty((outshape), dtype=np.float32)
print('inshape from get_binding_shape(0):', org_inshape)
print('outshape from get_binding_shape(1):', org_outshape)
print('dynamic inshape:', image.shape)
print('dynamicoutshape:', output.shape)

"""static输入不需要指定输入输出的大小
"""
#st_set_binding = time.time()
#context.set_binding_shape(0, image.shape)
#context.set_binding_shape(1, output.shape)
#print('set binding time:', time.time() - st_set_binding)
#print(outshape, output.size, image.size, image.dtype.itemsize, output.dtype.itemsize)

d_input = cuda.mem_alloc(1 * image.size * image.dtype.itemsize)
d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)

bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()
cuda.memcpy_htod(d_input,image)
print('preprocess time:',time.time()- st)
st = time.time()
context.execute_v2(bindings)
print('infer time:',time.time()- st)
st = time.time()
cuda.memcpy_dtoh(output, d_output)
import torch
pred = output
print('pred shape', pred[0].shape, flush=True)
pred = torch.from_numpy(pred).cuda()
#TODO 如何通过内存地址、数据类型和数据size来init一个torch.cuda.tensor
#pred = torch.cuda.FloatTensor(d_output, output.size)
pred = ops.non_max_suppression(pred,
                               conf_thres=0.2,#self.args.conf,
                               iou_thres=0.5,#self.args.iou,
                               classes=None,
                               agnostic=True,#agnostic=False,#self.args.agnostic_nms,
                               max_det=300,#self.args.max_det,
                               nc=3
                               )#classes=None)#self.args.classes)[0]
print('postprocess time:',time.time()- st)
#visualization
xyxys = []
confs = []
cls = []
for i, det in enumerate(pred):
    #print('before scalebox:', det)
    det[:,:4] = ops.scale_boxes(image.shape[2:], det[:,:4], image1.shape).round()
    #print('after scalebox:', det)
    xyxys.append(det[:,:4].cpu().numpy().tolist())
    confs.append(det[:,4].cpu().numpy().tolist())
    cls.append(det[:,5].cpu().numpy().astype(np.int32).tolist())
xyxys = xyxys[0]
confs = confs[0]
cls = cls[0]
plot_with_xyxys(xyxys, imgpath)

print(time.time()- total_st)

```

## Static_Fp16
    inshape from get_binding_shape(0): (1, 3, 640, 640)
    outshape from get_binding_shape(1): (1, 7, 8400)
    dynamic inshape: (1, 3, 640, 640)
    dynamicoutshape: (1, 7, 8400)
    preprocess time: 0.009712457656860352
    infer time: 0.07206177711486816
    pred shape (7, 8400)
    postprocess time: 0.41475963592529297
    1.jpg
    0.548168420791626

由于在tensorrt8.6.1post1中，fp16模型的输入也是float32类型，所以static fp16与static fp32代码完全一致。在tensorrt8.5中可能不同。

```python
import tensorrt as trt
import os
import time
import tensorrt as trt
import pycuda.driver as cuda
#import pycuda.driver as cuda2
import pycuda.autoinit
from ultralytics.utils import ops 
import numpy as np
import cv2
import tqdm
from utils.augmentations import letterbox
def plot_with_xyxys(xyxys, img_path):
    objs = xyxys
    img = cv2.imread(img_path)
    for obj in objs:
        img = cv2.rectangle(img, \
            ( int(obj[0]), int(obj[1])), \
            ( int(obj[2]), int(obj[3])),\
            color=(0,0,255), thickness=2)
    print(img_path.split('/')[-1])
    if not os.path.exists(img_path):
        cv2.imwrite(img_path.split('/')[-1], img)
    else:
        cv2.imwrite('new_'+img_path.split('/')[-1], img)

def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

path = "/media/lvsolo/CA89-5817/datasets/helmet/hard-hat-detection/codes/runs/detect/train15/weights/best.engine.fp16_static_input"#
#path = "/media/lvsolo/CA89-5817/datasets/helmet/hard-hat-detection/codes/runs/detect/train15/weights/best.engine.dynamicinput"#
# path = "/media/lvsolo/CA89-5817/datasets/helmet/hard-hat-detection/codes/runs/detect/train15/weights/best.engine.fp32"#
engine = load_engine(path)
imgpath = '1.jpg'
#imgpath = 'R-C.jpg'

context = engine.create_execution_context()
image1 = cv2.imread(imgpath)
total_st = time.time()
st = total_st
"""对于static input的模型，在不进行特殊内存处理的情况下，输入图片尺寸必须等于static input shape, 因此auto=False"""
image = letterbox(image1, (640,640), stride=32, auto=False)[0]
image = image / 255.
image = image[:,:,::-1].transpose((2, 0, 1))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
image = np.ascontiguousarray(image)
image = np.expand_dims(image, axis=0)

# tensorrt8.6.1即使是fp16的engine也不能直接使用fp16输入，8.5版本貌似可以
image = image.astype(np.float32)
print('input image type:', image.dtype)
print('input image shape:', image.shape)

"""动态输入通过get_binding_shape(0|1)时获得的值可能是0或者包含-1,此时应指定输入输出的大小为实际大小,
并通过set_binding_shape方式进行指定
"""
#outshape = context.get_tensor_shape('output0')
outshape= (1,7,8400)#context.get_binding_shape(1) 
print('outshape:',outshape)
org_inshape= context.get_binding_shape(0)
org_outshape= context.get_binding_shape(1) 
output = np.empty((outshape), dtype=np.float32)
print('inshape from get_binding_shape(0):', org_inshape)
print('outshape from get_binding_shape(1):', org_outshape)
print('dynamic inshape:', image.shape)
print('dynamicoutshape:', output.shape)

"""static输入不需要指定输入输出的大小
"""
#st_set_binding = time.time()
#context.set_binding_shape(0, image.shape)
#context.set_binding_shape(1, output.shape)
#print('set binding time:', time.time() - st_set_binding)
#print(outshape, output.size, image.size, image.dtype.itemsize, output.dtype.itemsize)

d_input = cuda.mem_alloc(1 * image.size * image.dtype.itemsize)
d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)

bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()
cuda.memcpy_htod(d_input,image)
print('preprocess time:',time.time()- st)
st = time.time()
context.execute_v2(bindings)
print('infer time:',time.time()- st)
st = time.time()
cuda.memcpy_dtoh(output, d_output)
import torch
pred = output
print('pred shape', pred[0].shape, flush=True)
pred = torch.from_numpy(pred).cuda()
#TODO 如何通过内存地址、数据类型和数据size来init一个torch.cuda.tensor
#pred = torch.cuda.FloatTensor(d_output, output.size)
pred = ops.non_max_suppression(pred,
                               conf_thres=0.2,#self.args.conf,
                               iou_thres=0.5,#self.args.iou,
                               classes=None,
                               agnostic=True,#agnostic=False,#self.args.agnostic_nms,
                               max_det=300,#self.args.max_det,
                               nc=3
                               )#classes=None)#self.args.classes)[0]
start_logic = time.time()
#visualization
xyxys = []
confs = []
cls = []
for i, det in enumerate(pred):
    #print('before scalebox:', det)
    det[:,:4] = ops.scale_boxes(image.shape[2:], det[:,:4], image1.shape).round()
    #print('after scalebox:', det)
    xyxys.append(det[:,:4].cpu().numpy().tolist())
    confs.append(det[:,4].cpu().numpy().tolist())
    cls.append(det[:,5].cpu().numpy().astype(np.int32).tolist())
print('postprocess time:',time.time()- st)
xyxys = xyxys[0]
confs = confs[0]
cls = cls[0]
plot_with_xyxys(xyxys, imgpath)

print(time.time()- total_st)

```

