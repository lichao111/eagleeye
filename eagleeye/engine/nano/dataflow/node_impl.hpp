#ifndef _EAGLEEYE_NODE_IMPL_H_
#define _EAGLEEYE_NODE_IMPL_H_
#include "eagleeye/common/EagleeyeMacro.h"
#include <eagleeye/engine/nano/dataflow/node.hpp>
#include <eagleeye/engine/nano/dataflow/meta.hpp>
#include "eagleeye/common/EagleeyeRuntime.h"
#include "eagleeye/common/EagleeyeTime.h"
#include "eagleeye/basic/type.h"
#include <memory>
#include <tuple>
#include <vector>
#include <functional>
#include <numeric>

namespace eagleeye{
namespace dataflow{
template <class F, class ... Args>
class NodeImpl : public Node{
public:
  /**
   * @brief Construct a new Node Impl object
   * 
   * @param handler 
   * @param id 
   * @param fixed 
   */
  NodeImpl(F handler, int id, EagleeyeRuntime fixed=EagleeyeRuntime(EAGLEEYE_UNKNOWN_RUNTIME))
  : Node(id, fixed){
      m_handler = handler;
  }

  /**
   * @brief Destroy the Node Impl object
   * 
   */
  virtual ~NodeImpl(){
  }

  /**
   * @brief run op
   * 
   * @param runtime 
   * @return float 
   */
  float fire(EagleeyeRuntime runtime=EAGLEEYE_CPU, void* data=NULL) noexcept override {
    float time = fireImpl(typename F::INS(), runtime);
    return time;
  }

  /**
   * @brief transfer which output to device
   * 
   * @param which 
   * @param runtime 
   * @param asyn 
   */
  virtual void transfer(int which, EagleeyeRuntime runtime, bool asyn) noexcept override{
    (&m_handler)->getOutput(which).transfer(runtime, asyn);
  }
  
  /**
   * @brief get data size
   * 
   * @param index 
   * @return size_t 
   */
  size_t size(int index) noexcept override {
    return (&m_handler)->getOutputSize(index);
  }

  /**
   * @brief initialize node
   * 
   * @param runtime 
   * @param data 
   */
  int init(EagleeyeRuntime runtime, std::map<std::string, std::vector<float>> data) noexcept override{
    if(!this->init_){
      // 1.step initialize hanlder
      int result = (&m_handler)->init(data);

      // 3.step reset flag
      this->init_ = true;
      return result;
    }
  }

  bool update(typename F::Type data, int index=0){
    (&m_handler)->update(data, index);
    return true;
  }

  virtual bool update(void* data, int index=0){
    // block
    (&m_handler)->update(data, index);
    return true;
  }

  virtual bool fetch(void*& data, int index=0, bool block=false){
    // block / no block
    return (&m_handler)->fetch(data, index, block);
  }

private:
  float fireImpl(index_sequence<>, EagleeyeRuntime runtime) {
    long start_time = EagleeyeTime::getCurrentTime();
    switch(runtime.type()){
      case EAGLEEYE_CPU:
        (&m_handler)->runOnCpu();
        break;
      case EAGLEEYE_GPU:
        (&m_handler)->runOnGpu();
        break;
      default:
        (&m_handler)->runOnCpu();
        break;
    }
    long end_time = EagleeyeTime::getCurrentTime();
    EAGLEEYE_LOGD("Run %s with %d us on %s.", name.c_str(), (end_time-start_time),runtime.device().c_str());
    return float(end_time-start_time);
  }

  template <std::size_t ... Is>
  float fireImpl(index_sequence<Is ...>, EagleeyeRuntime runtime) {
    std::vector<typename F::Type> unordered_input = {((NodeImpl<F, Args...>*)(data_[Is]))->m_handler.getOutput(index_[Is]) ...};
    std::vector<typename F::Type> ordered_input = {unordered_input[inv_order_[Is]] ...};

    long start_time = EagleeyeTime::getCurrentTime();
    switch(runtime.type()){
      case EAGLEEYE_CPU:
        (&m_handler)->runOnCpu(ordered_input);
        break;
      case EAGLEEYE_GPU:
        (&m_handler)->runOnGpu(ordered_input);
        break;
      default:
        (&m_handler)->runOnCpu(ordered_input);
        break;
    }
    long end_time = EagleeyeTime::getCurrentTime();
    EAGLEEYE_LOGD("Run %s with %d us on %s.", name.c_str(), (end_time-start_time),runtime.device().c_str());
    return float(end_time-start_time);
  }

private:
  F m_handler;
};

namespace impl {
  template <class F, class ... Args>
  struct Deducenode_type {
    using type = NodeImpl<F, Args...>;
  };
};

template <class F, class ... Args>
using deduce_node_type = typename impl::Deducenode_type<F, Args...>::type;

template <class F, class ... Args>
deduce_node_type<F, Args...> * makeNode(F f, int id, EagleeyeRuntime fixed) {
  using node_type = deduce_node_type<F, Args...>;
  return new node_type(f, id, fixed);
}

}
}
#endif

