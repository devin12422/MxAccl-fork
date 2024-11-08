#include <fstream>
#include <memx/accl/MxModel.h>
#include <memx/accl/prepost.h>


using namespace MX::Runtime;
using namespace MX::Types;
using namespace MX::Utils;

// template class MxModel<uint8_t>;
template class MxModel<float>;

#define VECTOR_INIT_BUFFER_LEN 500
std::chrono::milliseconds INPUT_TASK_TIMEOUT = 500ms;

template <typename T>
MxModel<T>::MxModel(int model_id, Dfp::DfpObject *dfp, const std::vector<int>* popen_contexts) : model_id_{model_id},
                                                                        dfp_{dfp},
                                                                        open_contexts{popen_contexts}{

    //Initate the model
    model_run.store(false);
    model_recv_run.store(false);
    model_manual_run.store(false);
    model_manual_in_done = false;
    num_streams_=0;
    parallel_fmap_convert_threads = 1;
    input_num_workers_ = 0;
    output_num_workers_ = 0;
    meta_ = dfp_->get_dfp_meta();
    in_ports_ = meta_.model_inports[model_id];
    out_ports_ = meta_.model_outports[model_id];
    mxa_gen = meta_.mxa_gen;
    int num_in_ports = in_ports_.size();
    int num_op_ports = out_ports_.size();
    model_info.model_index = model_id;
    model_info.num_in_featuremaps = num_in_ports;
    model_info.num_out_featuremaps = num_op_ports;
    model_info.input_layer_names.reserve(num_in_ports);
    model_info.in_featuremap_shapes.reserve(num_in_ports);
    model_info.in_featuremap_sizes.reserve(num_in_ports);
    model_info.output_layer_names.reserve(num_op_ports);
    model_info.out_featuremap_shapes.reserve(num_op_ports);
    model_info.out_featuremap_sizes.reserve(num_op_ports);
    context_send_current_index = 0;
    number_of_contexts = open_contexts->size();
    
    
    for(int ip = 0; ip<num_in_ports ; ++ip){
        int port_idx = in_ports_[ip];
        int64_t h = dfp_->input_port(port_idx)->dim_h;
        int64_t w = dfp_->input_port(port_idx)->dim_w;
        int64_t z = dfp_->input_port(port_idx)->dim_z;
        int64_t c = dfp_->input_port(port_idx)->dim_c;

        MX::Types::ShapeVector featureMap_shape{h,w,z,c};
        
        model_info.input_layer_names.push_back(std::string(dfp_->input_port(port_idx)->layer_name));
        model_info.in_featuremap_shapes.push_back(featureMap_shape);
        model_info.in_featuremap_sizes.push_back(dfp_->input_port(port_idx)->total_size);
        // std::cout<<"In Layer Name = "<<dfp_->input_port(port_idx)->layer_name<<"\n";
    }

    for(int op = 0; op<num_op_ports ; ++op){
        int port_idx = out_ports_[op];
        int64_t h = dfp_->output_port(port_idx)->dim_h;
        int64_t w = dfp_->output_port(port_idx)->dim_w;
        int64_t z = dfp_->output_port(port_idx)->dim_z;
        int64_t c = dfp_->output_port(port_idx)->dim_c;

        MX::Types::ShapeVector featureMap_shape{h,w,z,c};
        
        model_info.output_layer_names.push_back(std::string(dfp_->output_port(port_idx)->layer_name));
        model_info.out_featuremap_shapes.push_back(featureMap_shape);
        model_info.out_featuremap_sizes.push_back(dfp_->output_port(port_idx)->total_size);
        // std::cout<<"Out Layer Name = "<<dfp_->output_port(port_idx)->layer_name << "\n";
    }
    input_task_flag = true;
}

template<typename T>
void MxModel<T>::create_and_append_in_fm(){
    if(!pre_model_path.empty()){
        PrePost* temp_model = mx_create_prepost(pre_model_path);
        if(temp_model == nullptr){
            throw(std::runtime_error("The given post-processing model has dynamic output size. Please provide the largest \
                                            possible size of output in the second argument of connect_post_model()"));
        }
        pre_model.push_back(temp_model);
    }

    if(!pre_model_path.empty()){
        vector<FeatureMap<T> *> temp_piv;
        for(int l = 0; l < (int)pre_info_model->get_input_names().size() ;++l){
            FeatureMap<T>* t = new FeatureMap<T>(pre_model_info.in_featuremap_sizes[l],MX_FMT_FP32);
            t->fm_type = FM_PRE;
            temp_piv.push_back(t);
        }
        pre_in_featuremaps_.push_back(temp_piv);
    }

    vector<FeatureMap<T> *> temp_v;
    vector<FeatureMap<T> *> temp_iv;
    for (int k = 0; k < static_cast<int>(in_ports_.size()); ++k)
    {
        int port_idx = in_ports_[k];
        FeatureMap<T> *t = new FeatureMap<T>(dfp_->input_port(port_idx)->total_size,
                        (MX_data_format) dfp_->input_port(port_idx)->format, 
                        dfp_->input_port(port_idx)->dim_h, 
                        dfp_->input_port(port_idx)->dim_w, 
                        dfp_->input_port(port_idx)->dim_z,
                        dfp_->input_port(port_idx)->dim_c,
                        parallel_fmap_convert_threads);
        temp_v.push_back(t);
        FeatureMap<T>* t_in = new FeatureMap<T>(*t);
        temp_iv.push_back(t_in);
    }
    in_featuremaps_.push_back(temp_v);
    transposed_in_featuremaps_.push_back(temp_iv);
}

template<typename T>
void MxModel<T>::create_and_append_out_fm(){
    vector<FeatureMap<float> *> temp_ov;
    vector<FeatureMap<float> *> temp_to;
    for (int k = 0; k < static_cast<int>(out_ports_.size()); ++k)
    {
        int port_idx = out_ports_[k];
        FeatureMap<float> *t = new FeatureMap<float>(dfp_->output_port(port_idx)->total_size,
                            (MX_data_format) dfp_->output_port(port_idx)->format, 
                            dfp_->output_port(port_idx)->dim_h,
                            dfp_->output_port(port_idx)->dim_w,
                            dfp_->output_port(port_idx)->dim_z,
                            dfp_->output_port(port_idx)->dim_c,
                            parallel_fmap_convert_threads);
        temp_ov.push_back(t);
        FeatureMap<float> *t_out = new FeatureMap<float>(*t);
        temp_to.push_back(t_out);
    }
    out_featuremaps_.push_back(temp_ov);
    transposed_out_featuremaps_.push_back(temp_to);
    if(!post_model_path_.empty()){
        PrePost* temp_model = mx_create_prepost(post_model_path_);
        if(temp_model == nullptr){
            throw(std::runtime_error("Failed to create post procesing model"));
        }
        post_model.push_back(temp_model);
    }
    if(!post_model_path_.empty()){
        vector<FeatureMap<float> *> temp_pov;
        for(int l = 0; l < (int)post_info_model->get_output_names().size() ;++l){
            FeatureMap<float>* t = new FeatureMap<float>(post_model_info.out_featuremap_sizes[l]);
            t->fm_type = FM_POST;
            temp_pov.push_back(t);
        }
        post_out_featuremaps_.push_back(temp_pov);
    }
}

template <typename T>
void MxModel<T>::model_set_post(std::filesystem::path post_path, const std::vector<size_t>& post_out_sizelist){
    post_model_path_ = post_path;
    post_out_size = post_out_sizelist;
    post_info_model = mx_create_prepost(post_model_path_,post_out_size);
    if(post_info_model->dynamic_output){
        if(post_out_size.size()==0){
            throw std::runtime_error("The post-processing model has dynamic outputs but output sizes vector not passed");
        }
    }

    post_info_model->match_names(model_info.output_layer_names,Process_Post);

    post_model_info.num_in_featuremaps = model_info.num_out_featuremaps;
    post_model_info.in_featuremap_shapes = model_info.out_featuremap_shapes;
    post_model_info.in_featuremap_sizes = model_info.out_featuremap_sizes;
    post_model_info.input_layer_names = model_info.output_layer_names;

    post_model_info.num_out_featuremaps = post_info_model->get_output_sizes().size();
    for(int op = 0; op<post_model_info.num_out_featuremaps ; ++op){
        if(post_out_size.size()==0){
            std::vector<int64_t>output_shape = post_info_model->get_output_shapes()[op];
            MX::Types::ShapeVector featureMap_shape(static_cast<int>(output_shape.size()));
            for(int i=0 ;i < static_cast<int>(output_shape.size()); ++i){
                featureMap_shape[i] = output_shape[i];
            }
            post_model_info.out_featuremap_sizes.push_back(post_info_model->get_output_sizes()[op]);
            post_model_info.output_layer_names.push_back(post_info_model->get_output_names()[op]);
            post_model_info.out_featuremap_shapes.push_back(featureMap_shape);
        }

        else{
            post_model_info.out_featuremap_sizes.push_back(post_out_size[op]);
        }
    }

    for(int i=0; i<(int)post_info_model->real_featuremaps.size(); ++i){
        post_model_info.num_out_featuremaps++;
        if(post_info_model->type == Plugin_Onnx)
        model_info.out_featuremap_shapes[post_info_model->real_featuremaps[i]].set_ch_first();

        post_model_info.out_featuremap_shapes.push_back(model_info.out_featuremap_shapes[post_info_model->real_featuremaps[i]]);
        post_model_info.out_featuremap_sizes.push_back(model_info.out_featuremap_sizes[post_info_model->real_featuremaps[i]]);
        post_model_info.output_layer_names.push_back(model_info.output_layer_names[post_info_model->real_featuremaps[i]]);
    }
}

template <typename T>
void MxModel<T>::model_set_pre(std::filesystem::path pre_path){
    pre_model_path = pre_path;
    pre_info_model = mx_create_prepost(pre_model_path);
    pre_info_model->match_names(model_info.input_layer_names,Process_Pre);

    pre_model_info.num_in_featuremaps = pre_info_model->get_input_sizes().size();
    for(int ip = 0; ip<pre_model_info.num_in_featuremaps ; ++ip){
        std::vector<int64_t>input_shape = pre_info_model->get_input_shapes()[ip];
        MX::Types::ShapeVector featureMap_shape(static_cast<int>(input_shape.size()));
        for(int i=0 ;i < static_cast<int>(input_shape.size()); ++i){
            featureMap_shape[i] = input_shape[i];
        }
        pre_model_info.in_featuremap_shapes.push_back(featureMap_shape);
        pre_model_info.in_featuremap_sizes.push_back(pre_info_model->get_input_sizes()[ip]);
        pre_model_info.input_layer_names.push_back(pre_info_model->get_input_names()[ip]);
    }
    pre_model_info.num_out_featuremaps = model_info.num_in_featuremaps;
    pre_model_info.out_featuremap_shapes = model_info.in_featuremap_shapes;
    pre_model_info.out_featuremap_sizes = model_info.in_featuremap_sizes;
    pre_model_info.output_layer_names = model_info.input_layer_names;

    for(int i=0; i<(int)pre_info_model->real_featuremaps.size(); ++i){
        pre_model_info.num_in_featuremaps++;
        if(pre_info_model->type == Plugin_Onnx)
        model_info.in_featuremap_shapes[pre_info_model->real_featuremaps[i]].set_ch_first();
        
        pre_model_info.in_featuremap_shapes.push_back(model_info.in_featuremap_shapes[pre_info_model->real_featuremaps[i]]);
        pre_model_info.in_featuremap_sizes.push_back(model_info.in_featuremap_sizes[pre_info_model->real_featuremaps[i]]);
        pre_model_info.input_layer_names.push_back(model_info.input_layer_names[pre_info_model->real_featuremaps[i]]);
    }
}

template <typename T>
void MxModel<T>::set_num_workers(int input_workers, int output_workers){
    if(input_workers<0 || output_workers <0){
        throw logic_error("number of workers must be 0 (auto) or a number >= 1");
    }
    input_num_workers_ = input_workers;
    output_num_workers_ = output_workers;
}        

template <typename T>
void MxModel<T>::set_parallel_fmap_convert(int num_threads){
    if(num_threads < 2){
        parallel_fmap_convert_threads = 1;
    } else {
        parallel_fmap_convert_threads = num_threads;
    }
}        

template <typename T>
void MxModel<T>::model_start(){
    //Create input vector of featureMaps for all streams
    if(!pre_model_path.empty()){
        for (int n = 0; n < num_streams_; ++n){
            pre_model.push_back(mx_create_prepost(pre_model_path));
            if(pre_model[n] == nullptr){
                throw(std::runtime_error("The given post-processing model has dynamic output size. Please provide the largest \
                                                possible size of output in the second argument of connect_post_model()"));
            }
        }
    }

    for(int i=0; i<num_streams_; ++i){
            create_and_append_in_fm();
            create_and_append_out_fm();
        }


    input_thread_counter.store(num_streams_);

    for(int i = 0; i< num_streams_;++i){
        out_task_mutex.push_back(new std::mutex);
        out_task_cv.push_back(new std::condition_variable);
    }

    //starting the model by setting the corresponding flags to true
    model_run.store(true);
    model_recv_run.store(true);

    //Create and start model threads
    model_send_thread = new std::thread(&MxModel<T>::model_send_fun, this);
    model_recv_thread = new std::thread(&MxModel<T>::model_recv_fun, this);

    int num_cpu_cores = std::thread::hardware_concurrency();
    int num_models = meta_.num_models;

    int default_num_workers = min(num_streams_,num_cpu_cores/(2*num_models));
    if(input_num_workers_==0 || output_num_workers_==0){
        input_num_workers_ = default_num_workers;
        output_num_workers_ = default_num_workers;
    }
    if(input_num_workers_>num_streams_){
        input_num_workers_ = num_streams_;
        std::cout<<"Warning!! Input number of workers are set to be more than number of streams. \
                                \n Default mode is activated and num workers is set to num streams"<<std::endl;
    }
    if(output_num_workers_>num_streams_){
        output_num_workers_ = num_streams_;
        std::cout<<"Warning!! Output number of workers are set to be more than number of streams. \
                                \n Default mode is activated and num workers is set to num streams"<<std::endl;
    }
    input_pool = new thread_pool("input_pool", input_num_workers_,true,num_streams_);
    output_pool = new thread_pool("output_pool",output_num_workers_,false,num_streams_);

    for(int i = 0; i<num_streams_; ++i){
        vector<const FeatureMap<T>*> temp(in_featuremaps_[i].begin(),in_featuremaps_[i].end());
        input_pool->submitTask(&MxModel<T>::inputTask,this,comb_in_call[i],std::move(temp),std::move(i),stream_id_list[i]);
    }
}

template <typename T>
void MxModel<T>::_pre_copy(int stream){
    if(pre_model[stream]->type==Plugin_Onnx){
        for(int i =0; i<(int)pre_info_model->real_featuremaps.size();++i){
            pre_in_featuremaps_[stream].push_back(transposed_in_featuremaps_[stream][pre_info_model->real_featuremaps[i]]);
        }
    }
    else{
        for(int i =0; i<(int)pre_info_model->real_featuremaps.size();++i){
            pre_in_featuremaps_[stream].push_back(in_featuremaps_[stream][pre_info_model->real_featuremaps[i]]);
        }                
    }
}

template <typename T>
void MxModel<T>::_pre_inference(int stream){
    vector<FeatureMap<T>*> premuted_output;
    if(pre_model[stream]->type==Plugin_Onnx){
        for(int i =0;i<(int)pre_info_model->dfp_pattern.size();++i){
            premuted_output.push_back(transposed_in_featuremaps_[stream][pre_info_model->dfp_pattern[i]]);
        }
        pre_model[stream]->runinference(pre_in_featuremaps_[stream],premuted_output);
        for(int i=0; i<model_info.num_in_featuremaps;++i)
        in_featuremaps_[stream][i]->set_data(transposed_in_featuremaps_[stream][i]->get_data_ptr(),true);
    }
    else{
        for(int i =0;i<(int)pre_info_model->dfp_pattern.size();++i){
            premuted_output.push_back(in_featuremaps_[stream][pre_info_model->dfp_pattern[i]]);
        }
        pre_model[stream]->runinference(pre_in_featuremaps_[stream],premuted_output);
    }
}


template <typename T>
void MxModel<T>::_post_inference(int stream){
    std::vector<FeatureMap<float>* > premuted_output;
    if(post_model[stream]->type == Plugin_Onnx){
        for(int i=0; i< model_info.num_out_featuremaps; ++i){
            out_featuremaps_[stream][i]->get_data(transposed_out_featuremaps_[stream][i]->get_data_ptr(),true);
        }
        if(post_model[stream]->dynamic_output){
            for(int m =0 ;m < static_cast<int>(post_out_size.size());++m)
            memset(post_out_featuremaps_[stream][m]->get_data_ptr(),0,post_out_size[m]*sizeof(float));
        }
        for(int i =0; i< (int)post_info_model->dfp_pattern.size();++i){
            premuted_output.push_back(transposed_out_featuremaps_[stream][post_info_model->dfp_pattern[i]]);
        }
        post_model[stream]->runinference(premuted_output,post_out_featuremaps_[stream]);
        for(int i =0; i< (int)post_info_model->real_featuremaps.size();++i){
            transposed_out_featuremaps_[stream][post_info_model->real_featuremaps[i]]->fm_type = FM_POST;
            post_out_featuremaps_[stream].push_back(transposed_out_featuremaps_[stream][post_info_model->real_featuremaps[i]]);
        }
    }
    else{
        if(post_model[stream]->dynamic_output){
            for(int m =0 ;m < static_cast<int>(post_out_size.size());++m)
            memset(post_out_featuremaps_[stream][m]->get_data_ptr(),0,post_out_size[m]*sizeof(float));
        }
        for(int i =0; i< (int)post_info_model->dfp_pattern.size();++i){
            premuted_output.push_back(out_featuremaps_[stream][post_info_model->dfp_pattern[i]]);
        }
        post_model[stream]->runinference(premuted_output,post_out_featuremaps_[stream]);
        for(int i =0; i< (int)post_info_model->real_featuremaps.size();++i){
            post_out_featuremaps_[stream].push_back(out_featuremaps_[stream][post_info_model->real_featuremaps[i]]);
        }
    }
}

template <typename T>
bool MxModel<T>::inputTask(combined_input_callback_t in_cb, vector<const FeatureMap<T>* >inputs,int stream, int stream_idx){
    if(in_featuremaps_[stream][0]->get_in_ready()){
        bool send_flag = false;
        if(!pre_model_path.empty()){
            _pre_copy(stream);
            vector<const FeatureMap<T>*> temp(pre_in_featuremaps_[stream].begin(),pre_in_featuremaps_[stream].end());
            send_flag = in_cb(temp,stream_idx);
            _pre_inference(stream);
        }
        else{
            send_flag = in_cb(inputs,stream_idx);
        }

        in_featuremaps_[stream][0]->set_in_ready(false);
        if(!send_flag){
            return false;
        }
        stream_queue.push(stream);
        input_thread_counter--;
        {
            std::unique_lock lock(input_task_mutex);
            input_task_flag = true;
            input_task_cv.notify_one();
        }
    }
    else{
        std::unique_lock lock(input_thread_mutex);
        auto now = std::chrono::steady_clock::now();
        input_thread_cv.wait_until(lock,now+INPUT_TASK_TIMEOUT,[this]() { return (this->input_thread_counter.load()>0); });
    }
    return true;
}

template <typename T>
bool MxModel<T>::outputTask(combined_output_callback_t out_cb, vector<const FeatureMap<float>* >outputs,int stream, int stream_idx){
    if(!post_model_path_.empty()){
        _post_inference(stream);
        vector<const FeatureMap<float>*> temp(post_out_featuremaps_[stream].begin(),post_out_featuremaps_[stream].end());
        out_cb(temp,stream_idx);
        for(int i =0; i< (int)post_info_model->real_featuremaps.size();++i){
            transposed_out_featuremaps_[stream][post_info_model->real_featuremaps[i]]->fm_type = FM_DFP;
        }
    }
    else{
        out_cb(outputs,stream_idx);
    }    
    {
        std::unique_lock<std::mutex> lock(*out_task_mutex[stream]);
        out_featuremaps_[stream][0]->set_out_ready(true);
    }
    out_task_cv[stream]->notify_one();
    return true;
}

template <typename T>
void MxModel<T>::create_append_manual_mem(){
    manual_recv_mutex.push_back(new std::mutex);
    manual_recv_cv.push_back(new std::condition_variable);
    manual_recv_task_mutex.push_back(new std::mutex);
    manual_recv_task_cv.push_back(new std::condition_variable);
}

// Manual threading model start to init model features and featureMap
template <typename T>
void MxModel<T>::model_manual_start(){
    // setting the model run flags to false for manual threading // repeating this for sanity
    model_manual_recv_thread = new std::thread(&MxModel<T>::model_manual_recv_fun, this);
    model_run.store(false);
    model_recv_run.store(false);
    model_manual_run.store(true);
    out_featuremaps_.reserve(VECTOR_INIT_BUFFER_LEN);
    in_featuremaps_.reserve(VECTOR_INIT_BUFFER_LEN);
    post_model.reserve(VECTOR_INIT_BUFFER_LEN);
    pre_model.reserve(VECTOR_INIT_BUFFER_LEN);
    post_out_featuremaps_.reserve(VECTOR_INIT_BUFFER_LEN);
    pre_in_featuremaps_.reserve(VECTOR_INIT_BUFFER_LEN);
    transposed_in_featuremaps_.reserve(VECTOR_INIT_BUFFER_LEN);
    transposed_out_featuremaps_.reserve(VECTOR_INIT_BUFFER_LEN);
    manual_recv_cv.reserve(VECTOR_INIT_BUFFER_LEN);
    manual_recv_mutex.reserve(VECTOR_INIT_BUFFER_LEN);
    manual_recv_task_cv.reserve(VECTOR_INIT_BUFFER_LEN);
    manual_recv_task_mutex.reserve(VECTOR_INIT_BUFFER_LEN);
}

template <typename T>
int MxModel<T>::get_num_streams(){
    return num_streams_;
}

template< typename T>
void MxModel<T>::model_wait(){
    if(!model_run.load()){
        throw logic_error("Model wait called when model is not running");
    }
    //Waiting of send stream threads to be done
    input_pool->wait();
}

template <typename T>
void MxModel<T>::model_stop()
{
    if(!model_run.load()){
        throw logic_error("Model stop called when model is not running");
    }
    //Stop signal for model
    input_pool->stop();
    model_run.store(false);
    input_task_cv.notify_one();

    //waiting for model threads to be done
    model_send_thread->join();
    //stopping the model recv thread
    model_recv_run.store(false);
    model_recv_cv.notify_one();
    model_recv_thread->join();
    //Giving the recv stream threads to finish iteration before last iteration
    for(int i =0;i<num_streams_;++i){
        while(!out_featuremaps_[i][0]->get_out_ready())
        {
            this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    output_pool->stop();

    for(int i =0;i<num_streams_;++i){
        delete out_task_cv[i];
        delete out_task_mutex[i];
    }
    out_task_cv.clear();
    out_task_mutex.clear();

    //Flushing the MPU (Sake of sanity and shouldn't be required if everything goes as intended)
    if(num_streams_>0){
        for(int ctx = 0; ctx < number_of_contexts; ctx++){

            int context_id = open_contexts->at(ctx);
            // memx_set_stream_enable(context_id, 0);
            memx_status status = MEMX_STATUS_OK;
            while(status==MEMX_STATUS_OK){
                for (int i = 0; i < static_cast<int>(out_ports_.size()); ++i)
                {
                    uint8_t* temp_blob = new uint8_t[out_featuremaps_[0][i]->get_formatted_size()];
                    status = (memx_status) ((int)status | memx_stream_ofmap( context_id, out_ports_[i], temp_blob, 100));
                    
                    if(memx_status_no_error(status)){
                        std::cerr<<"Entered flushing"<<std::endl;
                    }
                    delete [] temp_blob;
                    temp_blob = NULL;
                }
            }
        }
    }
    
    delete input_pool;
    input_pool = NULL;
    delete output_pool;
    output_pool = NULL;
    
    delete model_send_thread;
    model_send_thread = NULL;
    delete model_recv_thread;
    model_recv_thread = NULL;

    //Deleting the created featureMaps
    if(!pre_model_path.empty()){
        for(int j = 0; j<num_streams_;++j){
            for(int k = 0; k< static_cast<int>(pre_model[j]->get_input_sizes().size());++k)
            delete pre_in_featuremaps_[j][k];
            delete pre_model[j];
        }
        delete pre_info_model;
        pre_in_featuremaps_.clear();
    }
    for (int j = 0; j < num_streams_; ++j)
    {
        for (int k = 0; k < static_cast<int>(in_ports_.size()); ++k)
        {
            delete in_featuremaps_[j][k];
            in_featuremaps_[j][k] = NULL;
            delete transposed_in_featuremaps_[j][k];
            transposed_in_featuremaps_[j][k] = NULL;
        }
        in_featuremaps_[j].clear();
    }
    in_featuremaps_.clear();
    transposed_in_featuremaps_.clear();
    for (int j = 0; j < num_streams_; ++j)
    {
        for (int k = 0; k < static_cast<int>(out_ports_.size()); ++k)
        {
            delete out_featuremaps_[j][k];
            out_featuremaps_[j][k] = NULL;
            delete transposed_out_featuremaps_[j][k];
            transposed_out_featuremaps_[j][k] = NULL;
        }
        out_featuremaps_[j].clear();
    }
    out_featuremaps_.clear();
    transposed_out_featuremaps_.clear();
    if(!post_model_path_.empty()){
        for(int j = 0; j<num_streams_;++j){
            for(int k = 0; k<static_cast<int>(post_model[j]->get_output_sizes().size());++k)
            delete post_out_featuremaps_[j][k];
            delete post_model[j];
        }
        delete post_info_model;
        post_out_featuremaps_.clear();
    }
}

template<typename T>
void MxModel<T>::model_manual_stop(){
    if(!model_manual_run.load()){
         throw logic_error("Model stop called when model is not running");
    }
    model_manual_run.store(false);
    model_manual_cv.notify_one();
    for(int i =0; i<static_cast<int>(in_featuremaps_.size());++i){
        manual_recv_cv[i]->notify_one();
    }
    model_manual_recv_thread->join();
    delete model_manual_recv_thread;
    model_manual_recv_thread = NULL;

    // std::cout<<"MODEL STOPPPPP CALLED \n\n";
    //Flushing the MPU (Sake of sanity and shouldn't be required if everything goes as intended)
    if(out_featuremaps_.size()>0){
        for(int ctx = 0; ctx < number_of_contexts; ctx++){
            int context_id = open_contexts->at(ctx);
            memx_status status = MEMX_STATUS_OK;
            while(status==MEMX_STATUS_OK){
                for (int i = 0; i < static_cast<int>(out_ports_.size()); ++i)
                {
                    uint8_t* temp_blob = new uint8_t[out_featuremaps_[0][i]->get_formatted_size()];
                    status = (memx_status) ((int)status | memx_stream_ofmap(context_id, out_ports_[i], temp_blob, 100));
                    if(memx_status_no_error(status)){
                        std::cerr<<"Entered flushing\r";
                    }
                    delete [] temp_blob;
                }
            }
        }
    }

    //Deleting the created featureMaps
    for(int i =0; i<static_cast<int>(in_featuremaps_.size());++i){
        for (int k = 0; k < static_cast<int>(in_ports_.size()); ++k)
        {
            delete in_featuremaps_[i][k];
            delete transposed_in_featuremaps_[i][k];
            in_featuremaps_[i][k] = NULL;
            transposed_in_featuremaps_[i][k] = NULL;
        }
        if(!pre_model_path.empty()){
            for(int k = 0; k< static_cast<int>(pre_model[i]->get_input_sizes().size());++k){
                delete pre_in_featuremaps_[i][k];
            }
            delete pre_model[i];
        }
    }
    
    if(!pre_model_path.empty()){
        delete pre_info_model;
        pre_in_featuremaps_.clear();
    }

    transposed_in_featuremaps_.clear();
    in_featuremaps_.clear();

    for(int i =0; i<static_cast<int>(out_featuremaps_.size());++i){
        for (int k = 0; k < static_cast<int>(out_ports_.size()); ++k)
        {
            delete out_featuremaps_[i][k];
            delete transposed_out_featuremaps_[i][k];
            out_featuremaps_[i][k] = NULL;
            transposed_out_featuremaps_[i][k] = NULL;
        }
            delete manual_recv_cv[i];
            delete manual_recv_mutex[i];
            delete manual_recv_task_cv[i];
            delete manual_recv_task_mutex[i];
    }

    if(!post_model_path_.empty()){
        for(int i =0; i<static_cast<int>(post_model.size());++i){
            for(int k = 0; k< static_cast<int>(post_model[i]->get_output_sizes().size());++k){
                delete post_out_featuremaps_[i][k];
            }
            delete post_model[i];
        }
    }
    
    transposed_out_featuremaps_.clear();
    out_featuremaps_.clear();

    if(!post_model_path_.empty()){
        delete post_info_model;
        post_out_featuremaps_.clear();
    }
}

// FYI This is a test function and will be removed
// template <typename T>
// void MxModel<T>::log_model_info(){
//     std::cout<<"\n******** Model Index : "<<model_info.model_index<<" ********\n";
//     std::cout<<"\nNum of in featureMaps : "<<model_info.num_in_featuremaps<<"\n";

//     std::cout<<"\nIn featureMap Shapes \n";
//     for(int i =0; i<model_info.num_in_featuremaps ; ++i){
//         std::cout<<"Shape of featureMap : "<<i+1<<"\n";
//         std::cout<<"Layer Name : "<<model_info.input_layer_names[i]<<"\n";
//         std::cout<<"H = "<<model_info.in_featuremap_shapes[i][0]<<"\n";
//         std::cout<<"W = "<<model_info.in_featuremap_shapes[i][1]<<"\n";
//         std::cout<<"Z = "<<model_info.in_featuremap_shapes[i][2]<<"\n";
//         std::cout<<"C = "<<model_info.in_featuremap_shapes[i][3]<<"\n";
//     }

//     std::cout<<"\n\nNum of out featureMaps : "<<model_info.num_out_featuremaps<<"\n";
//     std::cout<<"\nOut featureMap Shapes \n";
//     for(int i =0; i<model_info.num_out_featuremaps ; ++i){
//         std::cout<<"Shape of featureMap : "<<i+1<<"\n";
//         std::cout<<"Layer Name : "<<model_info.output_layer_names[i]<<"\n";
//         std::cout<<"H = "<<model_info.out_featuremap_shapes[i][0]<<"\n";
//         std::cout<<"W = "<<model_info.out_featuremap_shapes[i][1]<<"\n";
//         std::cout<<"Z = "<<model_info.out_featuremap_shapes[i][2]<<"\n";
//         std::cout<<"C = "<<model_info.out_featuremap_shapes[i][3]<<"\n";
//     }
// }

template <typename T>
MX::Types::MxModelInfo MxModel<T>::return_model_info(){
    return this->model_info;
}

template <typename T>
MX::Types::MxModelInfo MxModel<T>::return_pre_model_info(){
    return this->pre_model_info;
}

template <typename T>
MX::Types::MxModelInfo MxModel<T>::return_post_model_info(){
    return this->post_model_info;
}

template <typename T>
void MxModel<T>::model_send_fun()
{

    //Run till model is running or there are streams left to send to ifmap
    while (model_run.load() || stream_queue.size()>0)
    {
        if (stream_queue.size() > 0)
        {
            memx_status send_status =  MEMX_STATUS_OTHERS;
            int stream = stream_queue.pop();

            while(memx_status_error(send_status)){
                int context_to_send = open_contexts->at(context_send_current_index);

                for (int i = 0; i < static_cast<int>(in_ports_.size()); ++i)
                {                         
                    // int timeout = (i == 0)? 1 : 0;
                    send_status = memx_stream_ifmap(context_to_send , in_ports_[i], in_featuremaps_[stream][i]->get_formatted_data(), 0);
                }
                
                //update context_id every iteration
                context_send_current_index = (context_send_current_index + 1) % number_of_contexts;
                
                if(memx_status_no_error(send_status)){

                    in_featuremaps_[stream][0]->set_in_ready(true);
                    {
                        std::unique_lock lock(input_thread_mutex);
                        input_thread_counter++;
                        input_thread_cv.notify_all();
                    }

                    //Push the stream id and context to recv to out_queue right after sending it to ifmap
                    pair_stream_context_queue.push(std::make_pair(stream, context_to_send));
                    std::unique_lock lock(model_recv_mutex);
                    model_recv_flag = true;
                    model_recv_cv.notify_one();
                    break;
                }
            }    
        }
        else{
            std::unique_lock lock(input_task_mutex);
            input_task_cv.wait(lock,[this]() { return ((this->input_task_flag||!this->model_run.load())); });
            input_task_flag = false;
        }
    }

    model_recv_cv.notify_one();
}

template <typename T>
void MxModel<T>::model_recv_fun()
{
    //Run till model is running or there are streams left to send to ofmap
    while (model_recv_run.load() || pair_stream_context_queue.size()>0)
    {
        if (pair_stream_context_queue.size() > 0)
        {
            std::pair<int, int> pop_data = pair_stream_context_queue.pop(); 
            int stream = pop_data.first;
            int context_to_recv = pop_data.second;
            //wait till particular stream thread is done with previous
            // ofmap results
            {
                std::unique_lock<std::mutex> lock(*out_task_mutex[stream]);
                while(!out_featuremaps_[stream][0]->get_out_ready())
                {
                    out_task_cv[stream]->wait(lock);
                    // continue;
                }
            }
            for (int i = 0; i < static_cast<int>(out_ports_.size()); ++i)
            {
                memx_status status;
                {
                    status = memx_stream_ofmap(context_to_recv , out_ports_[i], out_featuremaps_[stream][i]->get_formatted_data(), 0);
                }
                if(memx_status_error(status)){
                    throw runtime_error("stream_ofmap failed, try resetting the MXA");              
                }
            }
            //Specifing a specific recv stream thread that the ofmap is done
            out_featuremaps_[stream][0]->set_out_ready(false);
            vector<const FeatureMap<float>*> temp(out_featuremaps_[stream].begin(),out_featuremaps_[stream].end());
            output_pool->submitTask(&MxModel<T>::outputTask,this,comb_out_call[stream],std::move(temp),std::move(stream),stream_id_list[stream]);       
        }
        else{
            {
                std::unique_lock lock(model_recv_mutex);
                model_recv_cv.wait(lock,[this]() { return ((this->model_recv_flag||!this->model_run.load())); });
                model_recv_flag = false;
            }
        }
    }
}

template <typename T>
MxModel<T>::~MxModel()
{
    //stop the model if destructor is called before calling model_stop
    if(model_run.load()){
        this->model_stop();
    }
    else if(model_manual_run.load()){
        this->model_manual_stop();
    }
}

template <typename T>
void MxModel<T>::connect_stream(MxModel<T>::combined_input_callback_t in_cb, MxModel<T>::combined_output_callback_t out_cb, int stream_id)
{
    //Don't connect streams after starting the Model
    if(model_run.load()){
        throw logic_error("connect_stream called after starting MxAccl");
    }
    //Throw an error if either of the callback funtions are NULL
    if(in_cb ==NULL || out_cb == NULL){
        throw invalid_argument("input callback or output callback got a NULL ptr!");
    }
    //connect stream only accepts unique stream ids over the Accl
    auto search = stream_set_.find(stream_id);
    if(search != stream_set_.end()){
        throw invalid_argument("duplicate stream id passed in connect_stream");
    }
    stream_set_.insert(stream_id);
    stream_id_list.push_back(stream_id);
    comb_in_call.push_back(in_cb);
    comb_out_call.push_back(out_cb);

    num_streams_ += 1;
}


template<typename T>
bool MxModel<T>::model_manual_send(std::vector<T *> in_data, int pstream_id, bool channel_first, int32_t timeout){

    if(stream_id_map_.find(pstream_id) == stream_id_map_.end()){
        unique_lock lock(fm_create_mutex);
        if(stream_id_map_.find(pstream_id) == stream_id_map_.end()){
            int map_size = stream_id_map_.size();
            create_and_append_in_fm();
            create_and_append_out_fm();
            create_append_manual_mem();
            stream_id_map_[pstream_id] = map_size;
            manual_init_cv.notify_all();
        }
    }

    // this->out_queue.push(pstream_id);
    int stream_idx = stream_id_map_[pstream_id];

    if(!pre_model_path.empty()){
        for(int i=0; i<this->pre_model_info.num_in_featuremaps;i++){
            // copy data from user to inernal feature map
            this->pre_in_featuremaps_[stream_idx][i]->set_data(in_data[i], channel_first);
        }
        _pre_inference(stream_idx);
    }
    else{
        for(int i=0; i<this->model_info.num_in_featuremaps;i++){
            // copy data from user to inernal feature map
            this->in_featuremaps_[stream_idx][i]->set_data(in_data[i], channel_first);
        }
    }

    {
        unique_lock lock(manual_mutex_in);
        
        int context_to_send = open_contexts->at(context_send_current_index);
        for(int i=0; i<this->model_info.num_in_featuremaps;i++){    
            memx_status status;
            status = memx_stream_ifmap(context_to_send, in_ports_[i], this->in_featuremaps_[stream_idx][i]->get_formatted_data(), timeout);

            // if ifmap is success set in ready to true until next set_data is called to copy data from user
            if(memx_status_error(status)){
                throw runtime_error("stream_ifmap failed, try resetting the MXA");
            }
        }
        context_send_current_index = (context_send_current_index + 1) % number_of_contexts;
        pair_stream_context_queue.push(std::make_pair(pstream_id, context_to_send));
    }
    
    {
        std::lock_guard model_manual_send_lock(manual_mutex);
        model_manual_in_done = true;
    }
    model_manual_cv.notify_one();
    return true;

}

template <typename T>
void MxModel<T>::model_manual_recv_fun(){
    //Run till model is running or there are streams left to send to ofmap
    while (model_manual_run.load() || pair_stream_context_queue.size()>0)
    {
        if (pair_stream_context_queue.size() > 0)
        {
            std::pair<int, int> pop_data = pair_stream_context_queue.pop(); 
            int pstream_id = pop_data.first;
            int context_to_recv = pop_data.second;
            int stream_idx = stream_id_map_[pstream_id];
            if(!out_featuremaps_[stream_idx][0]->get_out_ready())
            {
                std::unique_lock lock(*manual_recv_mutex[stream_idx]);
                manual_recv_cv[stream_idx]->wait(lock);
            }
            if(!model_manual_run.load()){
                return;
            }
            for (int i = 0; i < static_cast<int>(out_ports_.size()); ++i){

                memx_status status;
                status = memx_stream_ofmap(context_to_recv, out_ports_[i], this->out_featuremaps_[stream_idx][i]->get_formatted_data(), 0);

                if(memx_status_error(status)){
                    throw runtime_error("stream_ofmap failed, try resetting the MXA");              
                }
            }
            //Specifing a specific recv stream thread that the ofmap is done
            out_featuremaps_[stream_idx][0]->set_out_ready(false);     
            {
                std::unique_lock lock(*manual_recv_task_mutex[stream_idx]);
                manual_recv_task_cv[stream_idx]->notify_one();
            }
        }
        else{
            {
                std::unique_lock lock(manual_mutex);
                model_manual_cv.wait(lock,[this]() { return  this->model_manual_in_done || !model_manual_run.load(); });
                model_manual_in_done = false;
            }
        }
    }
}

template<typename T>
bool MxModel<T>::model_manual_receive(std::vector<float*> &out_data, int pstream_id, bool channel_first, int32_t timeout){
    int stream_idx = 0;
    while(stream_id_map_.find(pstream_id) == stream_id_map_.end()){
        std::unique_lock lock(manual_init_mutex);
        if(timeout>0){
            auto cv_timeout = std::chrono::system_clock::now() + std::chrono::seconds(timeout);
            if(!manual_init_cv.wait_until(lock,cv_timeout,[this, pstream_id]{return stream_id_map_.find(pstream_id) != stream_id_map_.end();})){
                return false;
            }
        }
        else{
            manual_init_cv.wait(lock,[this, pstream_id]{return stream_id_map_.find(pstream_id) != stream_id_map_.end();});
        }
        std::this_thread::sleep_for(10us);
    }
    stream_idx = stream_id_map_[pstream_id];
    std::chrono::milliseconds timeout_ms(timeout);
    if(out_featuremaps_[stream_idx][0]->get_out_ready()){
        std::unique_lock lock(*(manual_recv_task_mutex[stream_idx]));
        if(timeout>0){
            if(!manual_recv_task_cv[stream_idx]->wait_for(lock,timeout_ms,[this,stream_idx] { return !this->out_featuremaps_[stream_idx][0]->get_out_ready(); }))
            return false;
        }
        else{
            manual_recv_task_cv[stream_idx]->wait(lock);
        }
    }
    if(!post_model_path_.empty()){
        _post_inference(stream_idx);
        for (int i = 0; i < post_model_info.num_out_featuremaps; ++i){
            // copy data into user's memory
            this->post_out_featuremaps_[stream_idx][i]->get_data(out_data[i], channel_first);
        }
    }
    else{
        for (int i = 0; i < static_cast<int>(out_ports_.size()); ++i){
            // copy data into user's memory
            this->out_featuremaps_[stream_idx][i]->get_data(out_data[i], channel_first);
        }
    }
    out_featuremaps_[stream_idx][0]->set_out_ready(true);
    manual_recv_cv[stream_idx]->notify_one();
    return true; 
}

template<typename T>
bool MxModel<T>::manual_run(std::vector<T *> in_data, std::vector<float*> &out_data, int pstream_id, bool in_channel_first, bool out_channel_first, int32_t timeout){
    if(!this->model_manual_send(in_data,pstream_id,in_channel_first,timeout)){
        return false;
    }
    if(!this->model_manual_receive(out_data,pstream_id,out_channel_first,timeout)){
        return false;
    }
    return true;
}
