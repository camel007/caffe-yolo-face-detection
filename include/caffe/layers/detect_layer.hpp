#ifndef CAFFE_DETECT_LAYER_HPP_
#define CAFFE_DETECT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/detect_layer.hpp"

namespace caffe {

template<typename Dtype>
class DetectLayer : public Layer<Dtype>{
public:
    explicit DetectLayer(const LayerParameter& param);
    virtual ~DetectLayer(){}

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "Detect";}
    virtual inline int ExactNumBottomBlobs() const {return 2;}
    virtual inline int ExactNumTopBlobs() const { return 1;}

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& top);

    int classes;
    int coords;
    int rescore;
    int side;
    int num;
    bool softmax;
    bool sqrt;
    float jiter;
    float object_scale;
    float noobject_scale;
    float class_scale;
    float coord_scale;
};
}
#endif //namespace
