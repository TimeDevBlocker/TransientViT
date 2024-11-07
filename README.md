# TransientViT: A novel CNN - Vision Transformer hybrid real/bogus transient classifier for the Kilodegree Automatic Transient Survey
TransientViT is a CNN - Vision Transformer (ViT) hybird model to differentiate between transients (real detections) and image artifacts (bogus detections) for the Kilodegree Automatic Transient Survey (KATS). Some image samples are provided in ./images for evaluation.

## Quickstart Guide
### 0. 环境
```
pip install -r requirements.txt
```
---
### 1. 数据准备
将待分类的图片放置于`images`下

images的目录结构如下所示
```
images
├── 1230719191550003741.jpg
├── 1230719194852001012.jpg
├── 1230719195125006555.jpg
├── ...
└── 6230719202803006276.jpg
```
---
### 2. 推理

#### 2.1 多文件推理

运行下述代码，完成对images文件夹内所有图片的推理，并输出json文件
```
python inference.py --img-src images --device 0 --out-path output/result.json
```
运行结束后，检查输出文件：`output/results.json`，即可查看类别，置信度以及cross inference的具体推理结果

注：上述代码同`rb_classify.sh`脚本的内容相同

---
#### 2.2 单文件推理

运行下述代码，完成单个图像的推理，并输出json文件
```
python inference.py --img-src images/6230719202803006276.jpg --device 0 --out-path output/result_single.json
```
运行结束后，检查输出文件：`output/results_single.json`，即可查看类别，置信度以及cross inference的具体推理结果

注：上述代码同`rb_single_classify.sh`脚本的内容相同

---
#### 2.3 推理结果

推理结果的格式为列表嵌套多个字典，每个字典即为一张3x2或3x3的instance分类结果

（1）file_name：该instance的图片名

（2）class：该instance的类别

（3）conf：该instance为上述类别的置信度

（4）cross_inf_result：键（如'0,1', '0,2'）即为cross inference所取的NRD图像的索引，值的conf即为该索引下推理的置信度，vote为该索引下推理得到的类别
```
[
    {
        "file_name": "1230719194852001012.jpg",
        "class": "real",
        "conf": 0.958899974822998,
        "cross_inf_result": {
            "0,1": {
                "conf": 0.9520036578178406,
                "vote": "real"
            },
            "0,2": {
                "conf": 0.9649118781089783,
                "vote": "real"
            },
            "1,2": {
                "conf": 0.9597845077514648,
                "vote": "real"
            }
        }
    },
]
```

