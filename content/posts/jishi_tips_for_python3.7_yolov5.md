---
title: "jishi tips for yolov5 project and python3.7"
date: "2023-11-29"
author: "lvsolo"
math: true
tags: ['jishi', 'competation', 'yolov5']
---

- [requirements](#requirements)
  - [python env](#python-env)
  - [yolov5 commit id](#yolov5-commit-id)
  - [yolov5 requirements.txt](#yolov5-requirementstxt)
  - [pip](#pip)
- [train code](#train-code)
- [pytorch 转换为 tensorrt engine](#pytorch-转换为-tensorrt-engine)
- [run](#run)
  - [pt](#pt)
  - [run engine](#run-engine)
- [TODO](#todo)

# requirements

## python env
    conda 22.9.0
    python==3.7.4

## yolov5 commit id
    yolov5 commit id: 3f02fdee1d8f1a6cf18a24be3438096466367d9f
 
## yolov5 requirements.txt
    # Usage: pip install -r requirements.txt
    # Base ------------------------------------------------------------------------
    gitpython #>=3.1.30
    matplotlib==3.2.2 #>=3.3
    numpy==1.16.5 #>=1.22.2
    opencv-python==4.6.0.66 #>=4.1.1
    Pillow # >=10.0.1
    psutil  # system resources
    PyYAML>=5.3.1
    requests>=2.23.0
    scipy>=1.4.1
    thop>=0.1.1  # FLOPs computation
    #torch>=1.8.0  # see https://pytorch.org/get-started/locally (recommended)
    #torchvision>=0.9.0
    tqdm>=4.64.0
    ultralytics #>=8.0.147
    # protobuf<=3.20.1  # https://github.com/ultralytics/yolov5/issues/8012
    
    # Logging ---------------------------------------------------------------------
    # tensorboard>=2.4.1
    # clearml>=1.2.0
    # comet
    
    # Plotting --------------------------------------------------------------------
    pandas>=1.1.4
    seaborn>=0.11.0
    
    # Export ----------------------------------------------------------------------
    # coremltools>=6.0  # CoreML export
    # onnx>=1.10.0  # ONNX export
    # onnx-simplifier>=0.4.1  # ONNX simplifier
    # nvidia-pyindex  # TensorRT export
    # nvidia-tensorrt  # TensorRT export
    # scikit-learn<=1.1.2  # CoreML quantization
    # tensorflow>=2.4.0  # TF exports (-cpu, -aarch64, -macos)
    # tensorflowjs>=3.9.0  # TF.js export
    # openvino-dev>=2023.0  # OpenVINO export
    
    # Deploy ----------------------------------------------------------------------
    setuptools>=65.5.1 # Snyk vulnerability fix
    # tritonclient[all]~=2.24.0
    
    # Extras ----------------------------------------------------------------------
    # ipython  # interactive notebook
    # mss  # screenshots
    # albumentations>=1.0.3
    # pycocotools>=2.0.6  # COCO mAP
## pip
    absl-py==2.0.0
    cachetools==5.3.2
    certifi @ file:///croot/certifi_1671487769961/work/certifi
    charset-normalizer==3.3.2
    cycler==0.11.0
    Cython==3.0.5
    fonttools==4.38.0
    gitdb==4.0.11
    GitPython==3.1.40
    google-auth==2.23.4
    google-auth-oauthlib==0.4.6
    grpcio==1.59.3
    idna==3.4
    importlib-metadata==6.7.0
    kiwisolver==1.4.5
    Markdown==3.4.4
    MarkupSafe==2.1.3
    matplotlib==3.2.2
    numpy==1.16.5
    nvidia-cublas-cu11==11.10.3.66
    nvidia-cuda-nvrtc-cu11==11.7.99
    nvidia-cuda-runtime-cu11==11.7.99
    nvidia-cudnn-cu11==8.5.0.96
    nvidia-tensorrt==8.4.1.5
    oauthlib==3.2.2
    opencv-python==4.6.0.66
    packaging==23.2
    pandas==1.2.5
    Pillow==9.5.0
    protobuf==3.20.3
    psutil==5.9.6
    py-cpuinfo==9.0.0
    pyasn1==0.5.0
    pyasn1-modules==0.3.0
    pycocotools @ git+https://github.com/cocodataset/cocoapi.git@8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9#subdirectory=PythonAPI
    pyparsing==3.1.1
    python-dateutil==2.8.2
    pytz==2023.3.post1
    PyYAML==6.0.1
    requests==2.31.0
    requests-oauthlib==1.3.1
    rsa==4.9
    scipy==1.7.3
    seaborn==0.11.2
    sentry-sdk==1.37.1
    six==1.16.0
    smmap==5.0.1
    tensorboard==2.11.2
    tensorboard-data-server==0.6.1
    tensorboard-plugin-wit==1.8.1
    thop==0.1.1.post2209072238
    torch @ file:///media/lvsolo/CA89-5817/datasets/helmet/hard-hat-detection/codes/yolov5/torch-1.10.0%2Bcu113-cp37-cp37m-linux_x86_64.whl
    torchaudio @ file:///media/lvsolo/CA89-5817/datasets/helmet/hard-hat-detection/codes/yolov5/torchaudio-0.10.0%2Bcu113-cp37-cp37m-linux_x86_64.whl
    torchvision @ file:///media/lvsolo/CA89-5817/datasets/helmet/hard-hat-detection/codes/yolov5/torchvision-0.11.0%2Bcu113-cp37-cp37m-linux_x86_64.whl
    tqdm==4.66.1
    typing_extensions==4.7.1
    ultralytics==8.0.145
    urllib3==2.0.7
    Werkzeug==2.2.3
    zipp==3.15.0

# train code

```
python train.py --data data.yaml --weights yolov5s.pt --epochs 3 --img 640
```

# pytorch 转换为 tensorrt engine
来自pt2trt_trt8415.py

```python
import os
model_input_shape = (640,640)
pt_model_path="test_models/best.pt"
onnx_model_path=pt_model_path.split('.')[0]+".onnx"
trt_model_path=pt_model_path.split('.')[0]+".engine"
model_dir = '/'.join(list(pt_model_path.split('/')[:-1])) +'/'

output_shape_for_dynamic = (1,7,8400)

print("*"*50)
print("onnx path:", onnx_model_path)
print("trt path:", trt_model_path)
print("model dir:", model_dir)
print("-"*50)

"""using ultralytics model.export"""
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
model = YOLO(pt_model_path)  # load a pretrained model (recommended for training)
model.model.cuda().half()

path = model.export(format="onnx", dynamic=True,  simplify=True)#, half=True)  # export the model to ONNX format
os.system("mv "+onnx_model_path + " " + onnx_model_path.split('.')[0] + "_ultrlytics_export_dynamic_fp32.onnx")

path = model.export(format="onnx", dynamic=False,  simplify=True)#, half=True)  # export the model to ONNX format
os.system("mv "+onnx_model_path + " " + onnx_model_path.split('.')[0] + "_ultrlytics_export_static_fp32.onnx")

model.model.cuda()
path = model.export(format="engine", dynamic=True,  simplify=True, device=0)#, half=True)  # export the model to ONNX format
os.system("mv "+trt_model_path + " " + trt_model_path.split('.')[0] + "_ultrlytics_export_pt2trt_dynamic_fp32.engine")

path = model.export(format="engine", dynamic=False,  simplify=True, device=0)#, half=True)  # export the model to ONNX format
os.system("mv "+trt_model_path + " " + trt_model_path.split('.')[0] + "_ultrlytics_export_pt2trt_static_fp32.engine")


"""using torch export for pt2onnx convertion"""
import torch
import torch.nn
import cv2
import time 
import onnx
import onnxruntime
import numpy as np

from ultralytics.nn.tasks import attempt_load_weights, attempt_load_one_weight


model = attempt_load_weights(pt_model_path,
                             device=torch.device('cuda'),
                             inplace=True,
                             fuse=True)
#input_tensor = torch.ones((1,3,640,640)).cuda()
input_tensor = torch.ones((1,3,*model_input_shape)).cuda()
# static fp32
with torch.no_grad():
    print(f'process model:{pt_model_path}...')
    torch.onnx.export(model,
            input_tensor,
            onnx_model_path,
            opset_version=11,
            input_names=['images'],
            output_names=['output0'],
            dynamic_axes=None)
    onnx_model = onnx.load(onnx_model_path)
    try:
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        print('model incorrect')
        print(e)
    else:
        os.system("mv "+onnx_model_path + " " + onnx_model_path.split('.')[0] + "_torch_export_static_fp32.onnx")
        print('model correct')

# dynamic fp32
#dynamic_axes ={'input':{0:'batch',2:'H',3:'W'},
#dynamic_axes ={'input':{2:'H',3:'W'},
dynamic_axes ={'images':{2:'H', 3:'W'},
        #'output0':{2:'H',3:'W'},
        #'output1':{2:'H',3:'W'},
}
with torch.no_grad():
    print(f'process model:{pt_model_path}...')
    torch.onnx.export(model,
            input_tensor,
            onnx_model_path,
            opset_version=11,
            input_names=['images'],
            output_names=['output0'],
            dynamic_axes=dynamic_axes)
    onnx_model = onnx.load(onnx_model_path)
    try:
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        print('model incorrect')
        print(e)
    else:
        os.system("mv "+onnx_model_path + " " + onnx_model_path.split('.')[0] + "_torch_export_dynamic_fp32.onnx")
        print('model correct')

"""using trt api for onnx2trt convertion"""
import tensorrt as trt
import os
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger()
def get_engine(onnx_file_path, engine_file_path="", fp16=False, dynamic_in=False, dynamic_out=False):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(onnx_file_path, engine_file_path, fp16=False, dynamic_in=False, dynamic_out=False):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.max_workspace_size = 1 << 32  # 4GB
            if fp16:
                assert (builder.platform_has_fast_fp16 == True), "not support fp16"
                config.flags = 1<<int(trt.BuilderFlag.FP16)
            builder.max_batch_size = 1
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

            if dynamic_in or dynamic_out:
                print("dynamic_in or out:", dynamic_in, dynamic_out)
                # Dynamic input setting ��̬������builder��������
                profile = builder.create_optimization_profile()
                #��С�ĳߴ�,���õĳߴ�,���ĳߴ�,����ʱ��������Ҫ�������Χ��
                profile.set_shape('images',(1,3,1,model_input_shape[1]),\
                        (1,3,model_input_shape[0]*3//4,model_input_shape[1]),(1,3,*model_input_shape))

#                profile.set_shape('images',(3,1,model_input_shape[1]),\
#                        (3,model_input_shape[0]*3//4,model_input_shape[1]),(3,*model_input_shape))
#                profile.set_shape('output0', output_shape_for_dynamic)
                config.add_optimization_profile(profile)

            plan = builder.build_serialized_network(network, config)
            print('plan:', plan, network, config, flush=True)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine
    return build_engine(onnx_file_path, engine_file_path,\
            fp16=fp16, dynamic_in=dynamic_in, dynamic_out=dynamic_out)

for modelname in [os.path.join(model_dir, item) for item in os.listdir(model_dir)]:
#for modelname in [model_dir+"best_ultrlytics_export_static_fp32.onnx"]:
#for modelname in [model_dir+"best_ultrlytics_export_dynamic_fp32.onnx"]:
    if not modelname.endswith('.onnx'):
        continue
    bare_name = modelname.split('.')[0]
    engine_name = bare_name + '.engine'
    print('-'*50)
    print("src modelname:", modelname)
    print('dst engine name:', engine_name)
    dynamic_in = False
    if 'dynamic' in bare_name.split('/')[-1]:
        dynamic_in = True
    dynamic_out = False
    try:
        # static fp32
        print('static fp32:')
        if os.path.exists(engine_name.split('.')[0] + "_onnx_trtapi_static_fp32.engine"):
            print(engine_name.split('.')[0] + "_onnx_trtapi_static_fp32.engine exists.")
            assert 0
        get_engine(modelname, engine_name, fp16=False, dynamic_in=dynamic_in, dynamic_out=dynamic_out)
        os.system("mv "+ engine_name + " " + engine_name.split('.')[0] + "_onnx_trtapi_static_fp32.engine")
        print(modelname + " static fp32 convert success")
    except:
        print(modelname + " static fp32 convert failed")

    print('-'*50)
    try:
        # dynamic fp32
        print('dynamic fp32:')
        if os.path.exists(engine_name.split('.')[0] + "_onnx_trtapi_dynamic_fp32.engine"):
            print(engine_name.split('.')[0] + "_onnx_trtapi_dynamic_fp32.engine exists")
            assert 0
        dynamic_out = True
        get_engine(modelname, engine_name, fp16=False, dynamic_in=dynamic_in, dynamic_out=dynamic_out)
        os.system("mv "+ engine_name + " " + engine_name.split('.')[0] + "_onnx_trtapi_dynamic_fp32.engine")
        print(modelname + " dynamic fp32 convert success")
    except:
        print(modelname + " dynamic fp32 convert failed")

    print('-'*50)
    try:
        # static fp16
        print('static fp16:')
        if os.path.exists(engine_name.split('.')[0] + "_onnx_trtapi_static_fp16.engine"):
            print(engine_name.split('.')[0] + "_onnx_trtapi_static_fp16.engine exists")
            assert 0
        get_engine(modelname, engine_name, fp16=True, dynamic_in=dynamic_in, dynamic_out=dynamic_out)
        os.system("mv "+ engine_name + " " + engine_name.split('.')[0] + "_onnx_trtapi_static_fp16.engine")
        print(modelname + " static fp16 convert success")
    except:
        print(modelname + " static fp16 convert failed")

    print('-'*50)
    try:
        # dynamic fp16
        print('dynamic fp16:')
        dynamic_out = True
        if os.path.exists(engine_name.split('.')[0] + "_onnx_trtapi_dynamic_fp16.engine"):
            print(engine_name.split('.')[0] + "_onnx_trtapi_dynamic_fp16.engine exists")
            assert 0
        get_engine(modelname, engine_name, fp16=True, dynamic_in=dynamic_in, dynamic_out=dynamic_out)
        os.system("mv "+ engine_name + " " + engine_name.split('.')[0] + "_onnx_trtapi_dynamic_fp16.engine")
        print(modelname + " dynamic fp16 convert success")
    except:
        print(modelname + " dynamic fp16 convert failed")
    print('-'*50)
```
# run

## pt

```python
import json
import numpy as np
from collections import defaultdict
import sys
sys.path.append('/project/train/src_repo')
import os
from ultralytics.nn.tasks import attempt_load_weights
import torch
from utils.augmentations import letterbox

def init():
    """Initialize model
        Returns: model
    """   
    torch.backends.cudnn.benchmark = True
    #w = '/project/train/models/detect/train5/weights/best.pt'
    # model = torch.jit.load() if 'torchscript' in w else attempt_load(weights, map_location='cpu')

    # model = attempt_load_weights('/project/train/models/detect/train5/weights/epoch60.pt',
    model = attempt_load_weights('/project/train/models/detect/train5/weights/best.pt',
                                    device=torch.device('cuda'),
                                    inplace=True,
                                    fuse=True)
    #stride = max(int(model.stride.max()), 32)  # model stride
    #names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    model.half()# if fp16 else model.float()
    RGB2BGR = True
    #if model trained in rgb, first conv RGB to BGR
    if RGB2BGR:
        for name, param in model.named_parameters():
            if name in ('model.0.conv.weight'):
                tmp = param[:,0,:,:].clone()
                param[:,0,:,:] = param[:,2,:,:]/255.
                param[:,2,:,:] = tmp/255.
                param[:,1,:,:] = param[:,1,:,:]/255.
                # print(model)
    # model_int8 = torch.quantization.quantize_dynamic(
    #                     model,  # the original model
    #                     {torch.nn.Conv2d},  # a set of layers to dynamically quantize
    #                     dtype=torch.qint8)  #

    
    return model#model_int8#.eval()

def calc_box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def in_box(person_box, head_box, ratio_threshold=0.25):
    min_x = max(person_box[0], head_box[0])
    min_y = max(person_box[1], head_box[1])
    max_x = min(person_box[2], head_box[2])
    max_y = min(person_box[3], head_box[3])
    inter = 0
    if (max_x-min_x)> 0 and  (max_y - min_y) > 0:
        inter = (max_x-min_x) * (max_y - min_y)
    head_box_area = calc_box_area(head_box)
    if head_box_area <= 0:
        return False
    if (inter / head_box_area) > ratio_threshold:
        return True
    else:
        return False
                                                      
from ultralytics.engine.results import Results
from ultralytics.utils import ops 
import cv2
import time
def process_image(handle=None, input_image=None, args=None, ** kwargs):
    """Do inference to analysis input_image and get output
        Attributes:
            handle: algorithm handle returned by init()
            input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
        Returns: process result
    """
    # Process image here
    # start_pre = time.time()

    # input_shape = (640, 640)
    input_shape = (1024, 1024)
    org_shape = (0,0)
    if isinstance(input_image, str):
        cv_image = cv2.imread(input_image)
        org_shape = cv_image.shape
        torch_image = cv_image
        torch_image = letterbox(torch_image, input_shape, stride=32, auto=True)[0]
        input_shape = torch_image.shape[:2]
        torch_image = torch_image /255.
        torch_image = torch_image.transpose((2, 0, 1))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        torch_image = np.expand_dims(torch_image, axis=0)
        torch_image = torch.from_numpy(torch_image).cuda().half()
    else:
        cv_image = input_image
        org_shape = cv_image.shape
        torch_image = cv_image#.astype(np.int8)
        torch_image = letterbox(torch_image, input_shape, stride=32, auto=False)[0]#True)[0]
        input_shape = torch_image.shape[:2]
        torch_image = torch_image.transpose((2, 0, 1))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        torch_image = np.expand_dims(torch_image, axis=0)
        torch_image = torch.from_numpy(torch_image).cuda().half()
        # torch_image = torch_image /255.
                
    # print('torchimage shape:', torch_image.shape)
    # print('org_shape:', org_shape)

    # print('pre time:', time.time()-start_pre)

    # start_infer = time.time()
    pred = handle(torch_image, augment=False, visualize=False)[0]
    # print('infer time:', time.time()-start_infer)
    # start_post = time.time()
    
    pred = ops.non_max_suppression(pred,
                                    0.25,#self.args.conf,
                                    0.5,#self.args.iou,
                                    agnostic=False,#self.args.agnostic_nms,
                                    max_det=300,#self.args.max_det,
                                    classes=None)#self.args.classes)[0]
    # print('nms time:', time.time()-start_post)
    # start_logic = time.time()
    xyxys = []
    confs = []
    cls = []
    for i, det in enumerate(pred):
        det[:,:4] = ops.scale_boxes(input_shape, det[:,:4], org_shape)#.round()
        # det[:,:4] = ops.scale_boxes(torch_image.shape[2:], det[:,:4], cv_image.shape).round()
        xyxys.append(det[:,:4].cpu().numpy().tolist())
        confs.append(det[:,4].cpu().numpy().tolist())
        cls.append(det[:,5].cpu().numpy().astype(np.int32).tolist())
    xyxys = xyxys[0]
    confs = confs[0]
    cls = cls[0]
    # print('xxxx:', xyxys, confs, cls, flush=True)
    
    dict_res = defaultdict(list)
    fake_result = {"algorithm_data":{},
                    "model_data":{'objects':[]}
                    }
    #classes=['motorbike_person','electric_scooter_person','head','helmet','hat','bicycle_helmet']
    #              0                  1                        2   3        4     5
    map_cls={'0':'motorbike_person','1':'electric_scooter_person','2':'head','3':'helmet','4':'hat','5':'bicycle_helmet'}
    
    # return xyxys

    for ind in range(len(xyxys)):
        dict_res[str(cls[ind])].append({'xyxy':xyxys[ind], 'conf': confs[ind], 'cl': str(cls[ind])})
        fake_result['model_data']['objects'].append({
            'x':xyxys[ind][0],
            'y':xyxys[ind][1],
            'height':xyxys[ind][3]-xyxys[ind][1],
            'width':xyxys[ind][2]-xyxys[ind][0],
            'confidence':confs[ind],
            'name': map_cls[str(cls[ind])]
            })
    person_has_head = {'flags':[], 'heads':[]} #has head hat except helmat
    person_has_helmat = {'flags':[], 'helmats':[]}
    for person in dict_res['0'] + dict_res['1']:
        flag_has_head = False
        for head in dict_res['2'] + dict_res['4'] + dict_res['5']:
            if (not flag_has_head) and in_box(person['xyxy'], head['xyxy']):
                # print('0000',head)
                person_has_head['flags'] += [True]
                person_has_head['heads'] += [head]
                flag_has_head = True
        if not flag_has_head:
            person_has_head['flags'] += [False]
            person_has_head['heads'] += [None]
        
        flag_has_helmat = False
        for helmat in dict_res['3']:
            if (not flag_has_helmat) and in_box(person['xyxy'], helmat['xyxy']):
                person_has_helmat['flags'] += [True]
                person_has_helmat['helmats'] += [helmat]
                flag_has_helmat = True
        if not flag_has_helmat:
            person_has_helmat['flags'] += [False]
            person_has_helmat['helmats'] += [None]

    target_count = 0
    target_info = []
    for ind in range(len(person_has_head['flags'])):
        if person_has_head['flags'][ind] and (not person_has_helmat['flags'][ind]):
            target_count += 1
            tmp_head = person_has_head['heads'][ind]
            target_info.append({
                'x': tmp_head['xyxy'][0],
                'y': tmp_head['xyxy'][1],
                'width': tmp_head['xyxy'][2] - tmp_head['xyxy'][0],
                'height': tmp_head['xyxy'][3] - tmp_head['xyxy'][1],
                'confidence': tmp_head['conf'],
                'name': map_cls[tmp_head['cl']]
                })
    if target_count:
        fake_result['algorithm_data'] = {
            "is_alert": True,
            "target_count": target_count,
            "target_info": target_info
        }
    else:
        fake_result["algorithm_data"] = {
            "is_alert": False,
            "target_count": 0,
            "target_info": []
            }
    print(fake_result, flush=True)

    # print('logic time:', time.time()-start_logic)
    # print('total time:', time.time()-start_pre)
    return json.dumps(fake_result, indent=4)
    # return fake_result
```

## run engine

```python
import json
import numpy as np
from collections import defaultdict
import sys
sys.path.append('/project/train/src_repo')
import os
# os.system('source activate && conda activate yolov8')

# from ultralytics.models.yolo import classify, detect, segment
from ultralytics.nn.tasks import attempt_load_weights
import torch
from utils.augmentations import letterbox

import tensorrt as trt
import os
import time
import pycuda.driver as cuda
#import pycuda.driver as cuda2
import pycuda.autoinit
import numpy as np
import cv2
import tqdm
from ultralytics.utils import ops 


def load_engine(engine_path):
        #TRT_LOGGER = trt.Logger(trt.Logger.WARNING)  # INFO
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


pt_model_path="/project/train/models/detect/train5/weights/last.pt"
output_shape_for_dynamic = (1, 10, 21504)
model_input_shape = (1024,1024)

def convert_pt2trt(pt_model_path):
    import os
    onnx_model_path=pt_model_path.split('.')[0]+".onnx"
    trt_model_path=pt_model_path.split('.')[0]+".engine"
    model_dir = '/'.join(list(pt_model_path.split('/')[:-1])) +'/'

    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator
    model = YOLO(pt_model_path)
    path = model.export(format="onnx", dynamic=False,  simplify=True)#, half=True)  # export the model to ONNX format
    os.system("mv "+onnx_model_path + " " + onnx_model_path.split('.')[0] + "_ultrlytics_export_static_fp32.onnx")

    import tensorrt as trt
    import os
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    TRT_LOGGER = trt.Logger()
    def get_engine(onnx_file_path, engine_file_path="", fp16=False, dynamic_in=False, dynamic_out=False):
        """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

        def build_engine(onnx_file_path, engine_file_path, fp16=False, dynamic_in=False, dynamic_out=False):
            """Takes an ONNX file and creates a TensorRT engine to run inference with"""
            with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
                EXPLICIT_BATCH
            ) as network, builder.create_builder_config() as config, trt.OnnxParser(
                network, TRT_LOGGER
            ) as parser, trt.Runtime(
                TRT_LOGGER
            ) as runtime:
                config.max_workspace_size = 1 << 32  # 4GB
                if fp16:
                    assert (builder.platform_has_fast_fp16 == True), "not support fp16"
                    config.flags = 1<<int(trt.BuilderFlag.FP16)
                builder.max_batch_size = 1
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

                if dynamic_in or dynamic_out:
                    # Dynamic input setting ��̬������builder��������
                    profile = builder.create_optimization_profile()
                    #��С�ĳߴ�,���õĳߴ�,���ĳߴ�,����ʱ��������Ҫ�������Χ��
                    profile.set_shape('images',(1,3,1,model_input_shape[1]),\
                            (1,3,model_input_shape[0]*3//4,model_input_shape[1]),(1,3,*model_input_shape))

    #                profile.set_shape('images',(3,1,model_input_shape[1]),\
    #                        (3,model_input_shape[0]*3//4,model_input_shape[1]),(3,*model_input_shape))
    #                profile.set_shape('output0', output_shape_for_dynamic)
                    config.add_optimization_profile(profile)

                plan = builder.build_serialized_network(network, config)
                print('plan:', plan, network, config, flush=True)
                engine = runtime.deserialize_cuda_engine(plan)
                print("Completed creating Engine")
                with open(engine_file_path, "wb") as f:
                    f.write(plan)
                return engine
        return build_engine(onnx_file_path, engine_file_path,\
                fp16=fp16, dynamic_in=dynamic_in, dynamic_out=dynamic_out)

    for modelname in [onnx_model_path.split('.')[0] + "_ultrlytics_export_static_fp32.onnx"]: 
    #for modelname in [os.path.join(model_dir, item) for item in os.listdir(model_dir)]:
                
        if not modelname.endswith('.onnx'):
            continue
        bare_name = modelname.split('.')[0]
        engine_name = bare_name + '.engine'
        print('-'*50)
        print("src modelname:", modelname)
        print('dst engine name:', engine_name)
        dynamic_in = False
        if 'dynamic' in bare_name.split('/')[-1]:
            dynamic_in = True
        dynamic_out = False
        print('-'*50)
        try:
            # static fp16
            print('static fp16:')
            if os.path.exists(engine_name.split('.')[0] + "_onnx_trtapi_static_fp16.engine"):
                print(engine_name.split('.')[0] + "_onnx_trtapi_static_fp16.engine exists")
                # assert 0
            get_engine(modelname, engine_name, fp16=True, dynamic_in=dynamic_in, dynamic_out=dynamic_out)
            os.system("mv "+ engine_name + " " + engine_name.split('.')[0] + "_onnx_trtapi_static_fp16.engine")
            print(modelname + " static fp16 convert success")
        except:
            print(modelname + " static fp16 convert failed")

class engine_detector:
    def __init__(self, engine_path):                   
        
        # pt_model_path="/project/train/models/detect/train5/weights/epoch60.pt"

        
        self.engine = load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        inshape= self.context.get_binding_shape(0)
        outshape= self.context.get_binding_shape(1)
        if len(outshape) < 3:
            outshape = output_shape_for_dynamic

        self.output = np.empty((outshape), dtype=np.float32)
        imgpath = os.listdir('/home/data/1233')[0]
        imgpath = '/home/data/1233/' + imgpath 
        image1 = cv2.imread(imgpath)
        image1 = cv2.resize(image1,(1024,1024))
        image1 = image1.transpose(2,0,1) / 255.
        image = np.expand_dims(image1, axis=0)
        image = image.astype(np.float32)
        image = np.ascontiguousarray(image)

        # print(inshape, outshape, self.output.size, image.size, image.dtype.itemsize, self.output.dtype.itemsize)
        self.d_input = cuda.mem_alloc(1 * image.size * image.dtype.itemsize)
        self.d_output = cuda.mem_alloc(1*self.output.size * self.output.dtype.itemsize)
        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream = cuda.Stream()
        # warm up
        for _ in range(10):
            cuda.memcpy_htod(self.d_input, image)
            self.context.execute_v2(self.bindings)
            cuda.memcpy_dtoh(self.output, self.d_output)
        # return self
    
    def __call__(self, image):
        
        # print(type(self.d_input), type(image), flush=True)
        # print(type(self.d_input), image.dtype, flush=True)
        cuda.memcpy_htod(self.d_input, image)
        self.context.execute_v2(self.bindings)
        cuda.memcpy_dtoh(self.output, self.d_output)
        return self.output

def init():
    """Initialize model
        Returns: model
    """   
    # pt_model_path = '/project/train/models/detect/train5/weights/best.pt'
    # trt_model_path = conver_pt2trt(pt_model_path)
    # path = trt_model_path
    
    convert_pt2trt(pt_model_path)
    path = pt_model_path.split('.')[0] + '_ultrlytics_export_static_fp32_onnx_trtapi_static_fp16.engine'
    # path = '/project/train/models/detect/train5/weights/best_ultrlytics_export_static_fp32_onnx_trtapi_static_fp16.engine'
    # '/project/train/models/detect/train5/weights/best.engine'
    detector = engine_detector(path)
    return detector

def calc_box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def in_box(person_box, head_box, ratio_threshold=0.25):
    min_x = max(person_box[0], head_box[0])
    min_y = max(person_box[1], head_box[1])
    max_x = min(person_box[2], head_box[2])
    max_y = min(person_box[3], head_box[3])
    inter = 0
    if (max_x-min_x)> 0 and  (max_y - min_y) > 0:
        inter = (max_x-min_x) * (max_y - min_y)
    head_box_area = calc_box_area(head_box)
    if head_box_area <= 0:
        return False
    if (inter / head_box_area) > ratio_threshold:
        return True
    else:
        return False
                                                                                                                                                                                                        
                                                                
def process_image(handle=None, input_image=None, args=None, ** kwargs):
    """Do inference to analysis input_image and get output
        Attributes:
            handle: algorithm handle returned by init()
            input_image (numpy.ndarray): image to be process, format: (h, w, c), BGR
        Returns: process result
    """
    # Process image here
    # start_pre = time.time()

    # input_shape = (640, 640)
    input_shape = (1024, 1024)
    org_shape = (0,0)
    if isinstance(input_image, str):
        cv_image = cv2.imread(input_image)
        org_shape = cv_image.shape
        torch_image = cv_image
        torch_image = letterbox(torch_image, input_shape, stride=32, auto=True)[0]
        input_shape = torch_image.shape[:2]
        torch_image = torch_image /255.
        torch_image = torch_image.transpose((2, 0, 1))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        torch_image = np.expand_dims(torch_image, axis=0)
        torch_image = torch.from_numpy(torch_image).cuda().half()
    else:
        cv_image = input_image
        org_shape = cv_image.shape
        torch_image = cv_image
        torch_image = letterbox(torch_image, input_shape, stride=32, auto=False)[0]#True)[0]#.astype(np.float16)
        input_shape = torch_image.shape[:2]
        torch_image = torch_image.astype(np.float32)[:,:,::-1].transpose((2, 0, 1))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        torch_image = np.ascontiguousarray(torch_image)
        torch_image = np.expand_dims(torch_image, axis=0)
        #torch_image = torch.from_numpy(torch_image).cuda().half()
        torch_image = torch_image /255.
                
    # print('torchimage shape:', torch_image.shape)
    # print('torchimage type:', torch_image.dtype)
    # print('org_shape:', org_shape)

    # print('pre time:', time.time()-start_pre)

    # start_infer = time.time()
    pred = handle(torch_image)#[0]#, augment=False, visualize=False)#[0]
    # print('infer time:', time.time()-start_infer)
    # start_post = time.time()
    
    # print('pred shape:', pred.shape)

    pred = torch.from_numpy(pred).cuda()
    #TODO ���ͨ���ڴ��ַ���������ͺ�����size��initһ��torch.cuda.tensor
    #pred = torch.cuda.FloatTensor(d_output, output.size)
    pred = ops.non_max_suppression(pred,
                                   conf_thres=0.2,#self.args.conf,
                                   iou_thres=0.5,#self.args.iou,
                                   classes=None,
                                   agnostic=True,#agnostic=False,#self.args.agnostic_nms,
                                   max_det=300,#self.args.max_det,
                                   )#classes=None)#self.args.classes)[0]

    xyxys = []
    confs = []
    cls = []
    for i, det in enumerate(pred):
        det[:,:4] = ops.scale_boxes(input_shape, det[:,:4], org_shape).round()
        # det[:,:4] = ops.scale_boxes(torch_image.shape[2:], det[:,:4], cv_image.shape).round()
        xyxys.append(det[:,:4].cpu().numpy().tolist())
        confs.append(det[:,4].cpu().numpy().tolist())
        cls.append(det[:,5].cpu().numpy().astype(np.int32).tolist())
    xyxys = xyxys[0]
    confs = confs[0]
    cls = cls[0]
    # print('xxxx:', xyxys, confs, cls, flush=True)
    
    dict_res = defaultdict(list)
    fake_result = {"algorithm_data":{},
                    "model_data":{'objects':[]}
                    }
    #classes=['motorbike_person','electric_scooter_person','head','helmet','hat','bicycle_helmet']
    #              0                  1                        2   3        4     5
    map_cls={'0':'motorbike_person','1':'electric_scooter_person','2':'head','3':'helmet','4':'hat','5':'bicycle_helmet'}
    
    # return xyxys

    for ind in range(len(xyxys)):
        dict_res[str(cls[ind])].append({'xyxy':xyxys[ind], 'conf': confs[ind], 'cl': str(cls[ind])})
        fake_result['model_data']['objects'].append({
            'x':xyxys[ind][0],
            'y':xyxys[ind][1],
            'height':xyxys[ind][3]-xyxys[ind][1],
            'width':xyxys[ind][2]-xyxys[ind][0],
            'confidence':confs[ind],
            'name': map_cls[str(cls[ind])]
            })
    person_has_head = {'flags':[], 'heads':[]} #has head hat except helmat
    person_has_helmat = {'flags':[], 'helmats':[]}
    for person in dict_res['0'] + dict_res['1']:
        flag_has_head = False
        for head in dict_res['2'] + dict_res['4'] + dict_res['5']:
            if (not flag_has_head) and in_box(person['xyxy'], head['xyxy']):
                # print('0000',head)
                person_has_head['flags'] += [True]
                person_has_head['heads'] += [head]
                flag_has_head = True
        if not flag_has_head:
            person_has_head['flags'] += [False]
            person_has_head['heads'] += [None]
        
        flag_has_helmat = False
        for helmat in dict_res['3']:
            if (not flag_has_helmat) and in_box(person['xyxy'], helmat['xyxy']):
                person_has_helmat['flags'] += [True]
                person_has_helmat['helmats'] += [helmat]
                flag_has_helmat = True
        if not flag_has_helmat:
            person_has_helmat['flags'] += [False]
            person_has_helmat['helmats'] += [None]

    target_count = 0
    target_info = []
    for ind in range(len(person_has_head['flags'])):
        if person_has_head['flags'][ind] and (not person_has_helmat['flags'][ind]):
            target_count += 1
            tmp_head = person_has_head['heads'][ind]
            target_info.append({
                'x': tmp_head['xyxy'][0],
                'y': tmp_head['xyxy'][1],
                'width': tmp_head['xyxy'][2] - tmp_head['xyxy'][0],
                'height': tmp_head['xyxy'][3] - tmp_head['xyxy'][1],
                'confidence': tmp_head['conf'],
                'name': map_cls[tmp_head['cl']]
                })
    if target_count:
        fake_result['algorithm_data'] = {
            "is_alert": True,
            "target_count": target_count,
            "target_info": target_info
        }
    else:
        fake_result["algorithm_data"] = {
            "is_alert": False,
            "target_count": 0,
            "target_info": []
            }
    # print(fake_result, flush=True)

    # print('logic time:', time.time()-start_logic)
    # print('total time:', time.time()-start_pre)
    return json.dumps(fake_result, indent=4)
    # return fake_result


```
more details in [jishi_20231204.tar](http://localhost:1313/codes/jishi_20231204/jishi_20231204.tar)

# TODO
训练一个能够接受输入长宽比非1：1的模型
用torch重构加速前处理的letterbox模块和后处理的nms模块