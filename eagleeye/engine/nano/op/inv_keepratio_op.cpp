#include "eagleeye/engine/nano/op/inv_keepratio_op.h"
#include "eagleeye/common/EagleeyeLog.h"
#include <fstream>

namespace eagleeye{
namespace dataflow{
InvKeepRatioOp::InvKeepRatioOp(){}
InvKeepRatioOp::InvKeepRatioOp(const InvKeepRatioOp& op){}
InvKeepRatioOp::~InvKeepRatioOp(){}

int InvKeepRatioOp::init(std::map<std::string, std::vector<float>> params){

    return 0;
}

int InvKeepRatioOp::runOnCpu(const std::vector<Tensor>& input){
    // 0: layout 
    // 1: location
    const float* layout_info = input[0].cpu<float>();
    float x_scale = layout_info[0];
    float y_scale = layout_info[1];
    int layout_offset_x = layout_info[2];
    int layout_offset_y = layout_info[3];
    int layout_w = layout_info[4];
    int layout_h = layout_info[5];

    const Tensor position = input[1];
    Dim position_dim = position.dims();
    if(this->m_outputs[0].numel() != position.numel()){
        this->m_outputs[0] = Tensor(position_dim.data(), position.type(), position.format(), CPU_BUFFER);
    }

    int num = position_dim[0];
    const float* position_ptr = position.cpu<float>();
    float* inv_position_ptr = this->m_outputs[0].cpu<float>();
    for(int i=0; i<num; ++i){
        const float* position_row_ptr = position_ptr + i * position_dim[1];
        float* inv_position_row_ptr = inv_position_ptr + i * position_dim[1];
        memcpy(inv_position_row_ptr, position_row_ptr, sizeof(float)*position_dim[1]);

        inv_position_row_ptr[0] = std::min(std::max((position_row_ptr[0]-layout_offset_x) * x_scale, 0.0f), float(layout_w));
        inv_position_row_ptr[1] = std::min(std::max((position_row_ptr[1]-layout_offset_y) * y_scale, 0.0f), float(layout_h));
        inv_position_row_ptr[2] = std::min(std::max((position_row_ptr[2]-layout_offset_x) * x_scale, 0.0f), float(layout_w));
        inv_position_row_ptr[3] = std::min(std::max((position_row_ptr[3]-layout_offset_y) * y_scale, 0.0f), float(layout_h));
    }
    return 0;
}

int InvKeepRatioOp::runOnGpu(const std::vector<Tensor>& input){
    return -1;
}
}
}