#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/detect_layer.hpp"
#include "caffe/util/math_functions.hpp"

using namespace std;

namespace caffe {
template<typename Dtype>
Dtype lap(Dtype x1_min,Dtype x1_max,Dtype x2_min,Dtype x2_max){
    if(x1_min < x2_min){
        if(x1_max < x2_min){
            return 0;
        }else{
            if(x1_max > x2_min){
                if(x1_max < x2_max){
                    return x1_max - x2_min;
                }else{
                    return x2_max - x2_min;
                }
            }else{
                return 0;
            }
        }
    }else{
        if(x1_min < x2_max){
            if(x1_max < x2_max)
                return x1_max-x1_min;
            else
                return x2_max-x1_min;
        }else{
            return 0;
        }
    }
}

template<typename Dtype>
Dtype box_iou(const vector<Dtype> box1, const vector<Dtype> box2){
    Dtype lap_x = lap(box1[0]-box1[2]/2,box1[0]+box1[2]/2,box2[0]-box2[2]/2,box2[0]+box2[2]/2);
    Dtype lap_y = lap(box1[1]-box1[3]/2,box1[1]+box1[3]/2,box2[1]-box2[3]/2,box2[1]+box2[3]/2);

    Dtype are = box1[2]*box1[3]+box2[2]*box2[3]-lap_x*lap_y;
    if(are < 0.00001)
        return 0.0;
    else
        return (lap_x*lap_y)/are;
}

template <typename Dtype>
DetectLayer<Dtype>::DetectLayer(const LayerParameter& param) : Layer<Dtype>(param){
    this->layer_param_.add_propagate_down(true);
    this->layer_param_.add_propagate_down(false);
    const DetectParameter& detect_param = this->layer_param_.detect_param();
    classes = detect_param.classes();
    coords = detect_param.coords();
    rescore = detect_param.rescore();
    side = detect_param.side();
    num = detect_param.num();
    softmax = detect_param.softmax();
    sqrt = detect_param.sqrt();
    jiter = detect_param.jitter();
    object_scale = detect_param.object_scale();
    noobject_scale = detect_param.noobject_scale();
    class_scale = detect_param.class_scale();
    coord_scale = detect_param.coord_scale();
}

template <typename Dtype>
void DetectLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
    Layer<Dtype>::LayerSetUp(bottom, top);
    this->layer_param_.add_loss_weight(Dtype(1));
    int inputs = (side*side*(((1+coords)*num)+classes));
    CHECK_EQ(inputs, bottom[0]->count(1)) << "input dimensions error";
    CHECK_EQ(top.size(), 1) << "top size must be 1";
}
template <typename Dtype>
void DetectLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
    vector<int> shape(0);
    top[0]->Reshape(shape);
}
template <typename Dtype>
void DetectLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
    int input_num_each = side*side*(((1+coords)*num)+classes);
    int batch = bottom[0]->num();
    int locations = side*side;
    const Dtype* truth = bottom[1]->cpu_data();
    const Dtype* input = bottom[0]->cpu_data();

    Dtype* delta = bottom[0]->mutable_cpu_diff();
    Dtype& cost = top[0]->mutable_cpu_data()[0];
    cost = Dtype(0.0);

    for(int i = 0; i < bottom[0]->count(); ++i){
        delta[i] = Dtype(0.0);
    }
    float avg_iou = 0;
    float avg_cat = 0;
    float avg_allcat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    for(int b = 0; b < batch; ++b){
        int input_index = b*input_num_each;
        for(int l = 0; l < locations; ++l){
            int truth_index = (b*locations+l)*(1+coords+classes);
            Dtype is_obj = truth[truth_index];
            for(int n = 0; n < num;++n){
                int delta_index = input_index + locations*classes + l*num + n;
                delta[delta_index] = noobject_scale*(0 - input[delta_index]);
                cost += noobject_scale*pow(input[delta_index],2);
                avg_anyobj += input[delta_index];
            }

            int best_index = 0;
            float best_iou = 0;
            float best_rmse = 400;

            if(is_obj < 0.0001) continue;

            int class_index = input_index + l*classes;
            for(int j = 0; j < classes; ++j){
                delta[class_index+j]= class_scale * (truth[truth_index+1+j] - input[class_index+j]);
                if(truth[truth_index+1+j]) avg_cat += input[class_index+j];
                avg_allcat += input[class_index+j];
            }//classes

            vector<float> truth_box;
            truth_box.push_back(float(truth[truth_index+1+classes]/side));
            truth_box.push_back(float(truth[truth_index+1+classes+1]/side));
            truth_box.push_back(float(truth[truth_index+1+classes+2]));
            truth_box.push_back(float(truth[truth_index+1+classes+3]));
            for(int n = 0; n < num; ++n){
                int box_index = input_index + locations*(classes+num)+(l*num+n)*coords;
                vector<float> out_box;
                out_box.push_back(float(input[box_index]/side));
                out_box.push_back(float(input[box_index+1]/side));
                if(sqrt){
                    out_box.push_back(float(input[box_index+2]*input[box_index+2]));
                    out_box.push_back(float(input[box_index+3]*input[box_index+3]));
                }else{
                    out_box.push_back(float(input[box_index+2]));
                    out_box.push_back(float(input[box_index+3]));
                }
                float iou = box_iou(truth_box, out_box);
                float rmse = (pow(truth_box[0]-out_box[0],2)+pow(truth_box[1]-out_box[1],2)+pow(truth_box[2]-out_box[2],2)+pow(truth_box[3]-out_box[3],2));
                if(best_iou > 0 || iou > 0){
                    if(iou > best_iou){
                        best_iou = iou;
                        best_index = n;
                    }
                }else{
                    if(rmse < best_rmse){
                        best_rmse = rmse;
                        best_index = n;
                    }
                }
            }//for num
            int box_index = input_index + locations*(classes+num)+(l*num+best_index)*coords;
            int tbox_index = truth_index+1+classes;

            vector<float> best_box;
            best_box.push_back(float(input[box_index]/side));
            best_box.push_back(float(input[box_index+1]/side));
            if(sqrt){
                best_box.push_back(float(input[box_index+2]*input[box_index+2]));
                best_box.push_back(float(input[box_index+3]*input[box_index+3]));
            }else{
                best_box.push_back(float(input[box_index+2]));
                best_box.push_back(float(input[box_index+3]));
            }
            int p_index = input_index + locations*classes + l*num + best_index;
            cost -= noobject_scale*pow(input[p_index],2);
            cost += object_scale*pow(1-input[p_index],2);
            avg_obj += input[p_index];
            delta[p_index] = object_scale*(1. - input[p_index]);
            if(rescore){
                //delta[p_index] = object_scale*(best_iou - input[p_index]);
            }

            delta[box_index] = coord_scale*(truth[tbox_index]-input[box_index]);
            delta[box_index+1] = coord_scale*(truth[tbox_index+1]-input[box_index+1]);
            delta[box_index+2] = coord_scale*(truth[tbox_index+2]-input[box_index+2]);
            delta[box_index+3] = coord_scale*(truth[tbox_index+3]-input[box_index+3]);
            if(sqrt) {
                delta[box_index+2] = coord_scale*(std::sqrt(truth[tbox_index+2])-input[box_index+2]);
                delta[box_index+3] = coord_scale*(std::sqrt(truth[tbox_index+3])-input[box_index+3]);
            }
            cost += pow(1-best_iou, 2);
            avg_iou += best_iou;
            ++count;
        }//locations
    }//batch
    for(int i = 0; i < bottom[0]->count(); ++i){
        delta[i] = -delta[i];
    }
    printf("Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d\n", avg_iou/count, avg_cat/count, avg_allcat/(classes * count), avg_obj/count, avg_anyobj/(locations * batch * num), count);
}
template <typename Dtype>
void DetectLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& top){
}

template float lap(float x1_min,float x1_max,float x2_min,float x2_max);
template double lap(double x1_min,double x1_max,double x2_min,double x2_max);
template float box_iou(const vector<float> box1, const vector<float> box2);
template double box_iou(const vector<double> box1, const vector<double> box2);

#ifdef CPU_ONLY
STUB_GPU(DetectLayer);
#endif
INSTANTIATE_CLASS(DetectLayer);
REGISTER_LAYER_CLASS(Detect);
}//namespace caffe
