##darknet model to caffemodel

python ./python/ConvertYolo2Caffe/convert.py train_val.prototxt  yolo.weights caffemodel_save_path

##Test

./build/tools/yolo deploy.prototxt caffmodel_path image_path

##My trained caffemodel

[yolo-face](https://pan.baidu.com/s/1o8rmBKe)


##Reference
1, http://blog.csdn.net/u012235274/article/details/52120152.
2, https://github.com/xingwangsfu/caffe-yolo.
3, https://github.com/loswensiana/BWN-XNOR-caffe.
4, https://github.com/BVLC/caffe.
5, http://pjreddie.com/darknet/yolo/.
