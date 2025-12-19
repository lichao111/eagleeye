#ifndef _EAGLEEYE_KEEPRATIO_BY_SCALE_OP_
#define _EAGLEEYE_KEEPRATIO_BY_SCALE_OP_

#include "eagleeye/engine/nano/dataflow/base.h"
#include "eagleeye/basic/Tensor.h"
#include "eagleeye/engine/nano/op/dynamiccreater.h"
#include <string>
#include <vector>

namespace eagleeye{
namespace dataflow{
class KeepRatio2Op:public BaseOp<1, 2>, DynamicCreator<KeepRatio2Op>{
public:
    using BaseOp<1, 2>::init;
    KeepRatio2Op();
    KeepRatio2Op(const KeepRatio2Op& op);
    virtual ~KeepRatio2Op();

    virtual int init(std::map<std::string, std::vector<float>> params);
    virtual int init(std::map<std::string, std::vector<std::vector<float>>> params){return 0;};
    virtual int init(std::map<std::string, std::vector<std::string>> params){return 0;}

    virtual int runOnCpu(const std::vector<Tensor>& input);
    virtual int runOnGpu(const std::vector<Tensor>& input);

protected:
    float m_ratio;
    std::vector<int> m_out_size;    // width, height
    Tensor m_temp;
};
}
}

#endif