#ifndef _EAGLEEYE_GATHER_ND_OP_
#define _EAGLEEYE_GATHER_ND_OP_
#include "eagleeye/engine/nano/dataflow/base.h"
#include "eagleeye/basic/Tensor.h"
#include <string>
#include <vector>

namespace eagleeye{
namespace dataflow{
class GatherNdOp:public BaseOp<Tensor, 2, 1>{
public:
    GatherNdOp(const GatherNdOp& op);

    virtual ~GatherNdOp(){};

    GatherNdOp() = default;

    virtual int init(std::map<std::string, std::vector<float>> params);
    virtual int runOnCpu(const std::vector<Tensor>& input);
    virtual int runOnGpu(const std::vector<Tensor>& input);

protected:    
};
}    
}

#endif