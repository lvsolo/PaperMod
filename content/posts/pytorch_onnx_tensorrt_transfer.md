---
title: "Pytorch Onnx Tensorrt Transfer Methods, Advantages and Disadvantages Using Yolov5 Model"
date: "2023-11-23"
author: "lvsolo"
math: true
tags: ['pytorch', 'onnx', 'tensorrt', 'deploy', 'model-convert', 'deeplearning', 'yolov5', '2d-detection']
---
    experiment condition:
        python==3.8.0
        torch==2.1.0
        ultralytics:77fc5ccf02ac0fdf9e7bb1eeb003e9bf3e719702
        tensorrt==8.6.1.post1

- [一. pt转onnx](#一-pt转onnx)
  - [1.ultralytics工程自带的model.export功能](#1ultralytics工程自带的modelexport功能)
    - [1.1 dynamic input and output](#11-dynamic-input-and-output)
    - [1.2 static input and output](#12-static-input-and-output)
  - [2.torch自带的export功能](#2torch自带的export功能)
    - [2.1 dynamic input and output](#21-dynamic-input-and-output)
    - [2.2 static input and output](#22-static-input-and-output)
- [二. onnx转tensorrt-engine](#二-onnx转tensorrt-engine)
  - [1.torch自带的export功能](#1torch自带的export功能)
- [三.使用ultralytic中的model.export直接pt转为engine](#三使用ultralytic中的modelexport直接pt转为engine)

主要是以yolov5模型为例，记录该模型在不同转换工具下的转换方法、转换后模型调用方式、模型调用效率测试。
想要从pt文件转换为tensorrt的engine类型，有两种大路径，其中又可以分化：
- 1. pt-->onnx-->engine
  - 1) using torch.export to realize pt-->onnx convertion
    ```
        torch.onnx.export(model,
            input_tensor,
            model_name,
            opset_version=11,
            input_names=['input'],
            output_names=['output0','output1','output2','output3'],
            dynamic_axes=None)
    ```
  - 2) using model.export in ultralytics to realize pt-->onnx convertion
    ```
        path = model.export(format="onnx", dynamic=True,  simplify=True, half=True)
    ```
- 2. pt-->engine
  -  using model.export in ultralytics to directly realize pt-->engine convertion `model.export(format="engine", dynamic=True,  simplify=True, half=True)`
    - there are bugs in this way, because the mediate layers contains 'slow_conv2d_cpu' which cannot be converted into cuda tensor mode
    - half=True and dynamic=True cannot be set at the same time
## 一. pt转onnx
### 1.ultralytics工程自带的model.export功能
#### 1.1 dynamic input and output
* bug1:ultralytics工程自带的`model.export` 功能不能转换yolov5模型到half模式onnx,因此无法输出fp16的onnx|engine。
    RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'
```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
model = YOLO("../runs/detect/train15/weights/best.pt")
model.cuda().half()
path = model.export(format="engine", dynamic=True,  simplify=True)#, half=True)#half and dynamic cannot be True at the same time 
```
#### 1.2 static input and output
```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
model = YOLO("../runs/detect/train15/weights/best.pt")
model.cuda().half()
path = model.export(format="engine", dynamic=False,  simplify=True, half=True)#half and dynamic cannot be True at the same time 
```
### 2.torch自带的export功能
#### 2.1 dynamic input and output
#### 2.2 static input and output
下面的代码包含了`static input`,`input/output * batch/hw`的dynamic动态尺寸转换和onnx模型的测试.
几个问题：
- [ ] 多输出模型，可以通过export中的dynamic参数实现输出层的动态尺寸么？必须设置output的尺寸还是会自动根据input去变化？
- [ ] 多输出模型，可以通过export中的output_names参数实现输出层的增减么？
- [ ] 动态输入输出的模型在转换trt engine的过程中有问题么？

```python
import torch
import torch.nn
import cv2
import time 
import onnx
import onnxruntime
import numpy as np
from ultralytics.nn.tasks import attempt_load_weights, attempt_load_one_weight
model_path='/media/lvsolo/CA89-5817/datasets/helmet/hard-hat-detection/codes/runs/detect/train15/weights/best.pt'
model = attempt_load_weights(model_path,
                               device=torch.device('cuda'),
                               inplace=True,
                               fuse=True)
#model.load_state_dict(torch.load(model_path))
input_tensor = torch.ones((1,3,640,640)).cuda()

model_names = ['u2net.onnx','u2net_dynamic_batch.onnx','u2net_dynamic_hw.onnx']
dynamic_batch = {'input':{0:'batch'},
        'output0':{0:'batch'},
        #'output1':{0:'batch'},
        #'output2':{0:'batch'},
        #'output3':{0:'batch'},
}
dynamic_hw ={'input':{0:'batch',2:'H',3:'W'},
        'output0':{2:'H',3:'W'},
        #'output1':{2:'H',3:'W'},
        #'output2':{2:'H',3:'W'},
        #'output3':{2:'H',3:'W'},
}
dynamic_=[None,dynamic_batch,dynamic_hw]
with torch.no_grad():
    for i,model_name in enumerate(model_names):
        print(f'process model:{model_name}...')
        torch.onnx.export(model,
                input_tensor,
                model_name,
                opset_version=11,
                input_names=['input'],
                output_names=['output0'],
                #output_names=['output0','output1','output2','output3'],
                dynamic_axes=dynamic_[i])
        print(f'onnx model:{model_name} saved successfully...')
        print(f'begin check onnx model:{model_name}...')
        onnx_model = onnx.load(model_name)
        try:
            onnx.checker.check_model(onnx_model)
        except Exception as e:
            print('model incorrect')
            print(e)
        else:
            print('model correct')
print('*'*50)
print('Begin to test...')
case_1 = np.random.rand(1,3,640,640).astype(np.float32)
case_2 = np.random.rand(2,3,640,640).astype(np.float32)
case_3 = np.random.rand(1,3,480,640).astype(np.float32)
cases = [case_1,case_2,case_3]
model_names = ["../runs/detect/train4/weights/best.onnx"]

providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
for model_name in model_names:
    print('-'*50)
    onnx_session = onnxruntime.InferenceSession(model_name,providers=providers)
    for i,case in enumerate(cases):
        onnx_input = {'images':case}
        try:
            onnx_output = onnx_session.run(['output0'],onnx_input)[0]
            #onnx_output = onnx_session.run(['output0','output1','output2','output3'],onnx_input)[0]
        except Exception as e:
            print(f'Input:{i} on model:{model_name} failed')
            print(e)
        else:
            print(f'Input:{i} on model:{model_name} succeed')
```
## 二. onnx转tensorrt-engine
### 1.torch自带的export功能
四种不同状态：`dynamic|static`X`fp16|fp32`
```python
import tensorrt as trt
import os

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger()

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.max_workspace_size = 1 << 32  # 4GB
            
            # fp16
            config.flags = 1<<int(trt.BuilderFlag.FP16)
            
            builder.max_batch_size = 1
            assert (builder.platform_has_fast_fp16 == True), "not support fp16"
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(onnx_file_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            # network.get_input(0).shape = [1, 3, 608, 608]
            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_file_path))

            # Dynamic input setting 动态输入在builder里面设置
            profile = builder.create_optimization_profile()
            profile.set_shape('images',(1,3,1,640),(1,3,480,640),(1,3,640,640))#最小的尺寸,常用的尺寸,最大的尺寸,推理时候输入需要在这个范围内
            profile.set_shape('output0',(1,7,8400))
            config.add_optimization_profile(profile)

            plan = builder.build_serialized_network(network, config)
            print('plan:', plan, network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

def main():
    # Try to load a previously generated YOLOv5-608 network graph in ONNX format:
    onnx_file_path = "/media/lvsolo/CA89-5817/datasets/helmet/hard-hat-detection/codes/runs/detect/train15/weights/best.onnx"#
    engine_file_path = onnx_file_path.replace('.onnx', '.engine')
    get_engine(onnx_file_path, engine_file_path)
if __name__ == "__main__":
    main()
```
## 三.使用ultralytic中的model.export直接pt转为engine
model.export支持的转换格式
https://docs.ultralytics.com/modes/export/#arguments
| Format            | format_argument | Model                       | Metadata | Arguments                            |
|-------------------|-----------------|-----------------------------|----------|--------------------------------------|
| PyTorch           | -               | yolov8n.pt                  | ✅        | -                                    |
| TorchScript       | torchscript     | yolov8n.torchscript         | ✅        | imgsz, optimize                      |
| ONNX              | onnx            | yolov8n.onnx                | ✅        | imgsz, half, dynamic, simplify, opset|
| OpenVINO          | openvino         | yolov8n_openvino_model/     | ✅        | imgsz, half, int8                    |
| TensorRT          | engine          | yolov8n.engine              | ✅        | imgsz, half, dynamic, simplify, workspace|
| CoreML            | coreml          | yolov8n.mlpackage           | ✅        | imgsz, half, int8, nms               |
| TF SavedModel     | saved_model      | yolov8n_saved_model/        | ✅        | imgsz, keras, int8                   |
| TF GraphDef       | pb              | yolov8n.pb                  | ❌        | imgsz                                |
| TF Lite           | tflite          | yolov8n.tflite              | ✅        | imgsz, half, int8                    |
| TF Edge TPU       | edgetpu         | yolov8n_edgetpu.tflite      | ✅        | imgsz                                |
| TF.js             | tfjs            | yolov8n_web_model/          | ✅        | imgsz                                |
| PaddlePaddle      | paddle          | yolov8n_paddle_model/       | ✅        | imgsz                                |
| ncnn              | ncnn            | yolov8n_ncnn_model/         | ✅        | imgsz, half                          |


```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
model = YOLO("../runs/detect/train15/weights/best.pt")
model.cuda().half()
path = model.export(format="engine", dynamic=True,  simplify=True)#, half=True)#half and dynamic cannot be True at the same time 
```
* bug 1: half和dynamic不能同时为True，所以只能转换出`dynamic fp32|static fp32|static fp16`
* bug 2: ultralytics/engine/export.py中的meta保存部分会造成生成的engine文件用trt api调用时无法deserialization，需要注释掉。
    ```
    diff --git a/ultralytics/engine/exporter.py b/ultralytics/engine/exporter.py
    index 522c049b..7fb5c814 100644
    --- a/ultralytics/engine/exporter.py
    +++ b/ultralytics/engine/exporter.py
    @@ -632,10 +632,10 @@ class Exporter:
             # Write file
             with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
    -            # Metadata
    -            meta = json.dumps(self.metadata)
    -            t.write(len(meta).to_bytes(4, byteorder='little', signed=True))
    -            t.write(meta.encode())
    +#            # Metadata
    +#            meta = json.dumps(self.metadata)
    +#            t.write(len(meta).to_bytes(4, byteorder='little', signed=True))
    +#            t.write(meta.encode())
                 # Model
                 t.write(engine.serialize())
      
    ```

当使用`profile.set_shape('images',(1,3,1,model_input_shape[1]),(1,3,model_input_shape[0]*3//4,model_input_shape[1]),(1,3,*model_input_shape))`时，转换的结果如下：
|onnx\trt|ul st fp32|ul dy fp32|tc st fp32|tc dy fp32|
|---|---|---|---|---|
|st fp32|O|O|O|X|
|dy fp32|X|O|O|X|
|st fp16|X|O|O|X|
|dy fp16|X|O|O|X|

如果将命令中的shape维度从4变为3，省去dim0，还可以有新的两个模型