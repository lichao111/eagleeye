#include "eagleeye/engine/nano/op/keepratio2_op.h"
#include "eagleeye/common/EagleeyeLog.h"
#include <fstream>
#if defined (__ARM_NEON) || defined (__ARM_NEON__)  
#include "eagleeye/engine/math/arm/interpolate.h"
#else
#include "eagleeye/engine/math/x86/interpolate.h"
#endif


namespace eagleeye{
namespace dataflow{
KeepRatio2Op::KeepRatio2Op(){
    this->m_ratio = 1.0f;
}

KeepRatio2Op::KeepRatio2Op(const KeepRatio2Op& op){
    this->m_ratio = op.m_ratio;
    this->m_out_size = op.m_out_size;
}

KeepRatio2Op::~KeepRatio2Op(){

}

int KeepRatio2Op::init(std::map<std::string, std::vector<float>> params){
    if(params.find("ratio") != params.end()){
        this->m_ratio = params["ratio"][0];
    }
    if(params.find("aspect_ratio") != params.end()){
        this->m_ratio = params["aspect_ratio"][0];
    }
    if(params.find("out_size") != params.end()){
        this->m_out_size.resize(2);
        this->m_out_size[0] = int64_t(params["out_size"][0]);       // width
        this->m_out_size[1] = int64_t(params["out_size"][1]);       // height
    }
    return 0;
}

int KeepRatio2Op::runOnCpu(const std::vector<Tensor>& input){
    Tensor image = input[0];
    unsigned char* image_ptr = image.cpu<unsigned char>();

    Dim image_dim = image.dims();
    int image_h = image_dim[0];
    int image_w = image_dim[1];

    if(image_dim.size() != 3 && image_dim.size() != 2){
        EAGLEEYE_LOGE("KeepRatio2Op only support image rgb/bgr/gray");
        return -1;
    }

    // width/height == ratio
    int after_image_h = image_h;
    int after_image_w = image_w;
    float image_ratio = float(image_w)/float(image_h);
    if(image_ratio < this->m_ratio){
        after_image_w = this->m_ratio * image_h;
    }
    else{
        after_image_h = image_w / this->m_ratio;
    }

    float w_scale = (float)this->m_out_size[0] / (float)after_image_w;
    float h_scale = (float)this->m_out_size[1] / (float)after_image_h;

    int after_image_content_w = image_w * w_scale;
    int after_image_content_h = image_h * h_scale;

    int after_image_canvas_w = this->m_out_size[0];
    int after_image_canvas_h = this->m_out_size[1];
    
    int offset_x = (after_image_canvas_w - after_image_content_w)/2;
    int offset_y = (after_image_canvas_h - after_image_content_h)/2;

    if(image_dim.size() == 3){
        // rgb/bgr
        if(this->m_outputs[0].numel() != after_image_canvas_h*after_image_canvas_w*3){
            Dim out_dim(std::vector<int64_t>{after_image_canvas_h, after_image_canvas_w, 3});
            this->m_outputs[0] = Tensor(out_dim.data(),image.type(),image.format(),CPU_BUFFER);
        }

        if(this->m_temp.numel() != after_image_content_w*after_image_content_h*3){
            Dim out_dim(std::vector<int64_t>{after_image_content_h, after_image_content_w, 3});
            this->m_temp = Tensor(out_dim.data(),image.type(),image.format(),CPU_BUFFER);
        }

        unsigned char* output_ptr = this->m_outputs[0].cpu<unsigned char>();
        memset(output_ptr, 0, after_image_canvas_h*after_image_canvas_w*3);
        unsigned char* temp_ptr = this->m_temp.cpu<unsigned char>();
#if defined (__ARM_NEON) || defined (__ARM_NEON__)    
        math::arm::bilinear_rgb_8u_3d_interp(
            image_ptr,
            temp_ptr,
            image_w,
            image_h,
            0, 0, image_w,
            after_image_content_w,
            after_image_content_h
        );
#else
        math::x86::bilinear_rgb_8u_3d_interp(
            image_ptr,
            temp_ptr,
            image_w,
            image_h,
            0, 0, image_w,
            after_image_content_w,
            after_image_content_h
        );
#endif

        for(int i=0; i<after_image_content_h; ++i){
            unsigned char* row_output_ptr = output_ptr + (i+offset_y)*after_image_canvas_w*3 + offset_x*3;
            unsigned char* row_image_ptr = temp_ptr + i*after_image_content_w*3;
            memcpy(row_output_ptr, row_image_ptr, after_image_content_w*3);
        }    
    }
    else{
        // gray
        if(this->m_outputs[0].numel() != after_image_canvas_h*after_image_canvas_w){
            Dim out_dim(std::vector<int64_t>{after_image_canvas_h, after_image_canvas_w});
            this->m_outputs[0] = Tensor(out_dim.data(),image.type(),image.format(),CPU_BUFFER);
        }

        if(this->m_temp.numel() != after_image_content_w*after_image_content_h){
            Dim out_dim(std::vector<int64_t>{after_image_content_h, after_image_content_w});
            this->m_temp = Tensor(out_dim.data(),image.type(),image.format(),CPU_BUFFER);
        }

        unsigned char* output_ptr = this->m_outputs[0].cpu<unsigned char>();
        memset(output_ptr, 0, after_image_canvas_h*after_image_canvas_w);
        unsigned char* temp_ptr = this->m_temp.cpu<unsigned char>();

#if defined (__ARM_NEON) || defined (__ARM_NEON__)    
        math::arm::bilinear_gray_8u_1d_interp(
            image_ptr,
            temp_ptr,
            image_w,
            image_h,
            0, 0, image_w,
            after_image_content_w,
            after_image_content_h
        );
#else
        math::x86::bilinear_gray_8u_1d_interp(
            image_ptr,
            temp_ptr,
            image_w,
            image_h,
            0, 0, image_w,
            after_image_content_w,
            after_image_content_h
        );
#endif

        for(int i=0; i<after_image_content_h; ++i){
            unsigned char* row_output_ptr = output_ptr + (i+offset_y)*after_image_canvas_w + offset_x;
            unsigned char* row_image_ptr = temp_ptr + i*after_image_content_w;
            memcpy(row_output_ptr, row_image_ptr, after_image_content_w);
        }  
    }

    // 保存布局信息
    this->m_outputs[1] = Tensor(std::vector<int64_t>{6}, EAGLEEYE_FLOAT, DataFormat::AUTO, CPU_BUFFER);
    float* layout_ptr = this->m_outputs[1].cpu<float>();
    layout_ptr[0] = 1.0f/w_scale;
    layout_ptr[1] = 1.0f/h_scale;
    layout_ptr[2] = offset_x;
    layout_ptr[3] = offset_y;
    layout_ptr[4] = image_w;
    layout_ptr[5] = image_h;
    return 0;
}

int KeepRatio2Op::runOnGpu(const std::vector<Tensor>& input){
    return -1;
}
}
}