安装：

由于代码版本较老，依赖对应版本的coremltools， 请使用requirements.txt 进行安装
```
pip install -r requirements.txt
```


将maskrcnn tensorflow2训练的模型转换为coreml。
```
./venv/bin/python convert2coreml.py --weights_path /Volumes/exFAT/cad/maskrcnn_tf2.14.0/train2/mask_rcnn_floorplan_0020.h5
```

以上命令会在h5文件的同目录下存放创建同名文件夹放置生成的模型文件。