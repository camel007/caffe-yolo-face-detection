#ifndef TTRUN_H
#define TTRUN_H
#include <opencv2/opencv.hpp>
#include "list.h"
#include <iostream>
#include <fstream>
#include <iterator>

void list_insert(list *l, void *val)
{
    node *nw = (node *)malloc(sizeof(node));
    nw->val = (void *)val;
    nw->next = 0;

    if(!l->back){
        l->front = nw;
        nw->prev = 0;
    }else{
        l->back->next = nw;
        nw->prev = l->back;
    }
    l->back = nw;
    ++l->size;
}

list *make_list()
{
    list *l = (list *)malloc(sizeof(list));
    l->size = 0;
    l->front = 0;
    l->back = 0;
    return l;
}

float rand_uniform(float min, float max)
{
    return ((float)rand()/RAND_MAX * (max - min)) + min;
}

char *find_replace(char *str, char *orig, char *rep)
{
    static char buffer[4096];
    static char buffer2[4096];
    static char buffer3[4096];
    char *p;

    if(!(p = strstr(str, orig)))  // Is 'orig' even in 'str'?
        return str;

    strncpy(buffer2, str, p-str); // Copy characters from 'str' start to 'orig' st$
    buffer2[p-str] = '\0';

    sprintf(buffer3, "%s%s%s", buffer2, rep, p+strlen(orig));
    sprintf(buffer, "%s", buffer3);

    return buffer;
}

void file_error(char *s)
{
    fprintf(stderr, "Couldn't open file: %s\n", s);
    exit(0);
}

void malloc_error()
{
    fprintf(stderr, "Malloc error\n");
    exit(-1);
}

char *fgetl(FILE *fp)
{
    if(feof(fp)) return 0;
    size_t size = 512;
    char *line = (char *)malloc(size*sizeof(char));
    if(!fgets(line, size, fp)){
        free(line);
        return 0;
    }

    size_t curr = strlen(line);

    while((line[curr-1] != '\n') && !feof(fp)){
        if(curr == size-1){
            size *= 2;
            line = (char *)realloc(line, size*sizeof(char));
            if(!line) {
                printf("%ld\n", size);
                malloc_error();
            }
        }
        size_t readsize = size-curr;
        if(readsize > INT_MAX) readsize = INT_MAX-1;
        fgets(&line[curr], readsize, fp);
        curr = strlen(line);
    }
    if(line[curr-1] == '\n') line[curr-1] = '\0';

    return line;
}

list *get_paths(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    list *lines = make_list();
    while((path=fgetl(file))){
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}

void **list_to_array(list *l)
{
    void **a = (void **)calloc(l->size, sizeof(void*));
    int count = 0;
    node *n = l->front;
    while(n){
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}

typedef struct{
    int id;
    float x,y,w,h;
    float left, right, top, bottom;
} box_label;

template<typename T>
T constrain(T min, T max, T a)
{
    if( a < min)
        return min;
    else if(a > max)
        return max;
    else
        return a;
}

box_label *read_boxes(char *filename, int *n)
{
    box_label *boxes = (box_label *)calloc(1, sizeof(box_label));
    FILE *file = fopen(filename, "r");
    if(!file)
    {
        std::cout<<"open file failed\n"<<std::endl;
        return 0;
    }
    float x, y, h, w;
    int id;
    int count = 0;
    int k =0;
    while(fscanf(file, "%d %f %f %f %f", &id, &x, &y, &w, &h) == 5){
        boxes = (box_label *)realloc(boxes, (count+1)*sizeof(box_label));
        boxes[count].id = id;
        boxes[count].x = x;
        boxes[count].y = y;
        boxes[count].h = h;
        boxes[count].w = w;
        boxes[count].left   = x - w/2;
        boxes[count].right  = x + w/2;
        boxes[count].top    = y - h/2;
        boxes[count].bottom = y + h/2;

        ++count;
    }
    fclose(file);
    *n = count;
    return boxes;
}

void correct_boxes(box_label *boxes, int n, float dx, float dy, float sx, float sy)
{
    int i;
    for(i = 0; i < n; ++i){
        boxes[i].left   = boxes[i].left  * sx - dx;
        boxes[i].right  = boxes[i].right * sx - dx;
        boxes[i].top    = boxes[i].top   * sy - dy;
        boxes[i].bottom = boxes[i].bottom* sy - dy;

        boxes[i].left =  constrain(0.0f, 1.0f, boxes[i].left);
        boxes[i].right = constrain(0.0f, 1.0f, boxes[i].right);
        boxes[i].top =   constrain(0.0f, 1.0f,  boxes[i].top);
        boxes[i].bottom =   constrain(0.0f, 1.0f, boxes[i].bottom);

        boxes[i].x = (boxes[i].left+boxes[i].right)/2;
        boxes[i].y = (boxes[i].top+boxes[i].bottom)/2;
        boxes[i].w = (boxes[i].right - boxes[i].left);
        boxes[i].h = (boxes[i].bottom - boxes[i].top);

        boxes[i].w = constrain(0.0f, 1.0f,  boxes[i].w);
        boxes[i].h = constrain(0.0f, 1.0f,  boxes[i].h);
    }
}

void crop_image(cv::Mat &cropped, cv::Mat &im, int dx, int dy, int w, int h)
{
    unsigned char *pC = (unsigned char *)cropped.data, *pc;
    unsigned char *pI = (unsigned char *)im.data;
    int cStride = *cropped.step.p, iStride = *im.step.p;
    int i,j, k;
    int ch = im.channels();
    for(j = 0; j < h; ++j, pC +=cStride)
    {
        pc = pC;
        for(i = 0; i < w; ++i, pc+=ch)
        {
            int r, c;
            r = j + dy;
            c = i + dx;
            r = constrain(0, im.rows-1,r);
            c = constrain(0, im.cols-1,c);
            int index = r * iStride + c * ch;
            for(k = 0;  k < ch; ++k)
            {
                pc[k] = pI[index + k];
            }
        }
    }
}

void fill_truth_region(char *path, float *truth, int classes, int num_boxes, float dx, float dy, float sx, float sy)
{
    char *labelpath = find_replace(path, "images", "labels");
    labelpath = find_replace(labelpath, "JPEGImages", "labels");

    labelpath = find_replace(labelpath, ".jpg", ".txt");
    labelpath = find_replace(labelpath, ".JPG", ".txt");
    labelpath = find_replace(labelpath, ".JPEG", ".txt");
    int count = 0;
    box_label *boxes = read_boxes(labelpath, &count);

    correct_boxes(boxes, count, dx, dy, sx, sy);
    float x,y,w,h;
    int id;
    int i;
    for (i = 0; i < count; ++i) {
        x =  boxes[i].x;
        y =  boxes[i].y;
        w =  boxes[i].w;
        h =  boxes[i].h;
        id = boxes[i].id;

        if (w < .01 || h < .01) continue;

        int col = (int)(x*num_boxes);
        int row = (int)(y*num_boxes);

        x = x*num_boxes - col;
        y = y*num_boxes - row;

        int index = (col+row*num_boxes)*(5+classes);
        if (truth[index]) continue;
        truth[index++] = 1;

        if (id < classes) truth[index+id] = 1;
        index += classes;

        truth[index++] = x;
        truth[index++] = y;
        truth[index++] = w;
        truth[index++] = h;

    free(boxes);
}
}

cv::Mat load_data_region(std::vector<float> &y, char *paths, int w, int h, int size, int classes, float jitter = 0.2f)
{
    char *random_paths = paths;
    int i;

    int k = size*size*(5+classes);
    y.resize(k, 0.0f);

    cv::Mat orig = cv::imread(std::string(random_paths));

    int oh = orig.rows;
    int ow = orig.cols;

    int dw = (ow*jitter);
    int dh = (oh*jitter);

    int pleft  = rand_uniform(-dw, dw);
    int pright = rand_uniform(-dw, dw);
    int ptop   = rand_uniform(-dh, dh);
    int pbot   = rand_uniform(-dh, dh);

    int swidth =  ow - pleft - pright;
    int sheight = oh - ptop - pbot;

    float sx = (float)swidth  / ow;
    float sy = (float)sheight / oh;

    cv::Mat cropped =cv::Mat(sheight, swidth, orig.type());
    crop_image(cropped, orig, pleft, ptop, swidth, sheight);


    float dx = ((float)pleft/ow)/sx;
    float dy = ((float)ptop/oh)/sy;

    cv::Mat sized;
    cv::resize(cropped, sized, cv::Size(w, h));

    fill_truth_region(random_paths, &y[0], classes, size, dx, dy, 1./sx, 1./sy);

    return sized;
}

bool write_data(cv::Mat &im, const char *path, std::vector<float> &truth, std::ofstream &fout)
{
     if(!fout)
         file_error("write to ground truth!");

     cv::imwrite(std::string(path), im);
     fout<<std::string(path)<<" ";
     std::copy(truth.begin(), truth.end(), std::ostream_iterator<float>(fout, " "));
     fout<<std::endl;
}

void process()
{
    char* train_images = "/data/celeba_dababase/train.txt";
    std::string ground_truth_train_path = "/data/celeba_dababase/yolo/crop_train.txt";
    std::string ground_truth_val_path = "/data/celeba_dababase/yolo/crop_val.txt";

    list *plist = get_paths(train_images);
    char **paths = (char **)list_to_array(plist);

    int samples_num = plist->size;

    std::ofstream fout_train, fout_val, fout;
    fout_train.open(std::string(ground_truth_train_path));
    fout_val.open(std::string(ground_truth_val_path));
    if(!fout_train || !fout_val)
        file_error("open file failed!");

    std::vector<float> truth;
    size_t i = 0;
    for(i = 0; i < samples_num; ++i)
    {
        int rn = int(10 * rand_uniform(0.0f, 1.0f));

        char *path = paths[i];
        truth.clear();
        cv::Mat ret = load_data_region(truth, path, 448, 448, 11, 1);

        char * final_path = find_replace(path, "JPEGImages", "crop_celeba");
        if(rn % 3 == 0)
            write_data(ret, final_path, truth, fout_val);
        else
            write_data(ret, final_path, truth, fout_train);

        if(i % 1000 == 0)
            printf("Finished %d samples\n", i);
    }
    printf("Finished %d samples\n", i);
    fout_train.close();
    fout_val.close();
}

#endif // TTRUN_H
