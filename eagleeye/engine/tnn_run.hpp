namespace eagleeye
{
template<typename Enabled>
ModelRun<TNNRun, Enabled>::ModelRun(std::string model_name, 
			 std::string device,
			 std::vector<std::string> input_names,
		     std::vector<std::vector<int64_t>> input_shapes,
			 std::vector<std::string> output_names,
		     std::vector<std::vector<int64_t>> output_shapes,
		     int num_threads, 
		     RunPower model_power, 
		     std::string writable_path, std::vector<bool> inner_preprocess)
    	:ModelEngine(model_name,
				 device,
				 input_names,
				 input_shapes,
				 output_names,
				 output_shapes,
				 num_threads,
				 model_power,
				 writable_path){
    this->m_tnnproto = model_name + ".tnnproto";
    this->m_tnnmodel = model_name + ".tnnmodel";
    this->m_is_init = false;
    this->m_inner_preprocess = inner_preprocess;
}

template<typename Enabled>
ModelRun<TNNRun, Enabled>::~ModelRun(){
	// do nothing
}


template<typename Enabled>
bool ModelRun<TNNRun, Enabled>::run(std::map<std::string, const unsigned char*> inputs, 
				   std::map<std::string, unsigned char*>& outputs){	
    // ignore outside inputs
    if(!this->m_predictor.get()){
        EAGLEEYE_LOGE("TNN predictor null.");
        return false;
    }

    // set input
    for(int index=0; index < this->m_input_names.size(); ++index){
        // 输入节点名字
        std::string node_name = this->m_input_names[index];
		if(inputs.find(node_name) == inputs.end()){
			continue;
		}

        // 输入节点数据
        unsigned char* node_data = const_cast<unsigned char*>(inputs[node_name]);
        if(this->m_inner_preprocess[index] && this->m_input_convert_params.find(node_name) != this->m_input_convert_params.end()){
            // 需要内部预处理，输入的数据为uchar
            // 输入节点形状
            std::vector<int64_t> node_shape = this->m_input_shapes[index];
            TNN_NS::DimsVector dims = {1,1,1,1};
            dims[0] = node_shape[0];
            dims[1] = node_shape[1];
            dims[2] = node_shape[2];
            dims[3] = node_shape[3];

            ConvertParam convert_param = this->m_input_convert_params[node_name];
            TNN_NS::MatConvertParam tnn_cvt_param;
            tnn_cvt_param.scale = convert_param.scale;
            tnn_cvt_param.bias = convert_param.bias;
            tnn_cvt_param.reverse_channel = convert_param.reverse_channel;

            // 设置模型输入
            auto frame_mat = 
                std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_ARM, TNN_NS::N8UC3, dims, (uint8_t *) node_data);
            this->m_predictor->SetInputMat(frame_mat, tnn_cvt_param, node_name);
        }
        else{
            // 需要外部预处理，输入的数据为float
            TNN_NS::MatConvertParam input_cvt_param;
            input_cvt_param.scale = {1.0,1.0,1.0,1.0};//std
            input_cvt_param.bias  = {0.0,0.0,0.0,0.0};//mean/std

            std::vector<int64_t> shape = m_input_shapes[this->m_input_map_index[node_name]];
            TNN_NS::DimsVector target_dims;
            target_dims.resize(shape.size());
            std::transform(shape.begin(),shape.end(),target_dims.begin(), [](int64_t d){return (int)d;});
            
            std::shared_ptr<TNN_NS::Mat> target_mat = 
                    std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_ARM, TNN_NS::NCHW_FLOAT, target_dims, node_data);  
                    
            TNN_NS::Status stats = 
                this->m_predictor->SetInputMat(target_mat, input_cvt_param, node_name);                
        }
    }

    // run model
    this->m_predictor->Forward();
    
    // get output
    if(outputs.size() > 0){
		std::map<std::string, unsigned char*>::iterator iter, iend(outputs.end());
		for(iter = outputs.begin(); iter != iend; ++iter){
            std::shared_ptr<TNN_NS::Mat> tensor_mat = nullptr;
            this->m_predictor->GetOutputMat(tensor_mat,  TNN_NS::MatConvertParam(), iter->first);
			iter->second = (unsigned char*)(tensor_mat->GetData());
		}
    }
    return true;
}

template<typename Enabled>
bool ModelRun<TNNRun, Enabled>::initialize(){
    if(this->m_is_init){
        EAGLEEYE_LOGD("TNN Has been finished model initialize.");
        return true;
    }
    this->m_is_init = true;

    std::string model_folder = this->getModelFolder();
    if(model_folder == ""){
        EAGLEEYE_LOGD("TNN Dont set model folder.");
        return false;
    }

    // 加载模型
    TNN_NS::ModelConfig model_config;
    model_config.model_type = TNN_NS::MODEL_TYPE_TNN;

    // 加载proto
    std::string network_path;
    if(endswith(model_folder, "/")){
        network_path=model_folder + this->m_tnnproto;
    }
    else{
        network_path=model_folder + "/" + this->m_tnnproto;
    }

    // 1.检查文件是否存在，否则更换查找位置
    if(!isfileexist(network_path.c_str())){
        std::string so_folder = this->getModelRoot();
        if(endswith(so_folder, "/")){
            network_path = so_folder + this->m_tnnproto;
        }
        else{
            network_path = so_folder + std::string("/") + this->m_tnnproto;
        }
    }

    // 2. 检查文件是否存在，否则更换查找位置
    if(!isfileexist(network_path.c_str())){
        // android platform: /sdcard/models/
        // x86 platform: /${HOME}/models/
        #ifdef _ANDROID_
            network_path = std::string( "/sdcard/models/") + this->m_tnnproto;
        #else
            const char* home_folder = std::getenv("HOME");
            if(home_folder != NULL){
                network_path = std::string(home_folder) + std::string("/models/") + this->m_tnnproto;
            }
        #endif
    }

    // 读取proto文件
    EAGLEEYE_LOGD("Load TNN proto from %s", network_path.c_str());
    FILE* proto_fp = fopen(network_path.c_str(), "r");
    char buf[1024];
    std::string proto_str;
    while(fgets(buf, 1024, proto_fp)){
        std::string str = std::string(buf);
        proto_str = proto_str + str;
    }
    fclose(proto_fp);
    model_config.params.push_back(proto_str);

    // 读取model文件
    std::string model_path;
    if(endswith(model_folder, "/")){
        model_path = model_folder + this->m_tnnmodel;
    }
    else{
        model_path = model_folder + "/" + this->m_tnnmodel;
    }
    // 1.检查文件是否存在，否则更换查找位置
    if(!isfileexist(model_path.c_str())){
        std::string so_folder = this->getModelRoot();
        if(endswith(so_folder, "/")){
            model_path = so_folder + this->m_tnnmodel;
        }
        else{
            model_path = so_folder + std::string("/") + this->m_tnnmodel;
        }
    }
    // 2. 检查文件是否存在，否则更换查找位置
    if(!isfileexist(model_path.c_str())){
        // android platform: /sdcard/models/
        // x86 platform: /${HOME}/models/
        #ifdef _ANDROID_
            model_path = std::string( "/sdcard/models/") + this->m_tnnmodel;
        #else
            const char* home_folder = std::getenv("HOME");
            if(home_folder != NULL){
                model_path = std::string(home_folder) + std::string("/models/") + this->m_tnnmodel;
            }
        #endif
    }

    EAGLEEYE_LOGD("Load TNN model from %s", model_path.c_str());

    // 读取model文件
    FILE* model_fp = fopen(model_path.c_str(), "rb");
    fseek(model_fp, 0L, SEEK_END);
    int model_numbytes = ftell(model_fp);
    fseek(model_fp, 0L, SEEK_SET);
    char* model_buffer = (char*)calloc(model_numbytes, sizeof(char));
    fread(model_buffer, sizeof(char), model_numbytes, model_fp);
    fclose(model_fp);
    std::string model_buffer_str;
    model_buffer_str.assign(model_buffer,model_buffer+model_numbytes);
    model_config.params.push_back(model_buffer_str);
    free(model_buffer);

    TNN_NS::TNN net;
    TNN_NS::Status ret = net.Init(model_config);
    if(ret != TNN_NS::TNN_OK){
        EAGLEEYE_LOGD("TNN net init error.");
        return false;
    }

    model_config.params.clear();

    TNN_NS::NetworkConfig network_config;
    if(this->m_device == "GPU"){
        network_config.device_type = TNN_NS::DEVICE_OPENCL;
    }
    else{
        network_config.device_type = TNN_NS::DEVICE_ARM;
    }

    network_config.network_type = TNN_NS::NETWORK_TYPE_AUTO;
    if(this->getWritablePath() != ""){
        if(!isdirexist(this->getWritablePath().c_str())){
            createdirectory(this->getWritablePath().c_str());
        }
        network_config.cache_path = this->getWritablePath();
    }

    this->m_predictor = net.CreateInst(network_config, ret);
    if(ret != TNN_NS::TNN_OK){
        EAGLEEYE_LOGD("TNN create instance error.");
        return false;
    }
    EAGLEEYE_LOGD("TNN Finished model initialize.");
    return true;
}


template<typename Enabled>
void* ModelRun<TNNRun, Enabled>::getInputPtr(std::string input_name){
    EAGLEEYE_LOGE("dont support getInputPtr");
    return NULL;
}

template<typename Enabled>
const void* ModelRun<TNNRun, Enabled>::getOutputPtr(std::string output_name){
    std::shared_ptr<TNN_NS::Mat> tensor_mat = nullptr;
    this->m_predictor->GetOutputMat(tensor_mat,  TNN_NS::MatConvertParam(), output_name);
    return (void*)(tensor_mat->GetData());
}


template<typename Enabled>
void ModelRun<TNNRun, Enabled>::resize(std::string input_name, std::vector<int64_t> shape){
    EAGLEEYE_LOGE("dont support resize");
}

template<typename Enabled>
void ModelRun<TNNRun, Enabled>::getInput(std::string input_name, void*& input_ptr, std::vector<int64_t>& shape){
    EAGLEEYE_LOGE("dont support getInput");
}

template<typename Enabled>
void ModelRun<TNNRun, Enabled>::getOutput(std::string output_name, void*& output_ptr, std::vector<int64_t>& shape){
    std::shared_ptr<TNN_NS::Mat> tensor_mat = nullptr;
    this->m_predictor->GetOutputMat(tensor_mat,  TNN_NS::MatConvertParam(), output_name);
    // 数据指针
    output_ptr = (void*)(tensor_mat->GetData());

    // 形状
    TNN_NS::DimsVector dims_vec = tensor_mat->GetDims();
    shape.resize(dims_vec.size());
    std::transform(dims_vec.begin(),dims_vec.end(),shape.begin(), [](int d){return (int64_t)d;});
}


template<typename Enabled>
void ModelRun<TNNRun, Enabled>::refresh(){
    EAGLEEYE_LOGE("dont support refresh");
}
} // namespace eagleeye
