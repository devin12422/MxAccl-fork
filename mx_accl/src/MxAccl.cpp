#include <memx/accl/MxAccl.h>
#include <sstream>

using namespace MX::Runtime;
using namespace MX::Types;
using namespace MX::Utils;

MxAccl::MxAccl(){
    dfp_valid = false;
    setup_status = false;
    device_manager = new MX::Runtime::DeviceManager();
    // device_manager->print_available_devices();

}

int MxAccl::connect_dfp(const std::filesystem::path pdfp_path,int group_id){
    std::vector<int> devices_to_use = {group_id};
    return connect_dfp(pdfp_path,devices_to_use);
}

int MxAccl::connect_dfp(const uint8_t *dfp_bytes,int group_id){
    std::vector<int> devices_to_use = {group_id};
    return connect_dfp(dfp_bytes,devices_to_use);
}

int MxAccl::connect_dfp(const std::filesystem::path pdfp_path,std::vector<int>& device_ids_to_use){
    if(dfp_valid){
        throw std::runtime_error("Only one dfp allowed per Accl object");
    }

    if(device_ids_to_use.empty()){
        throw runtime_error("device_ids_to_use parameter cannot be empty");
    }

    dfp_path = pdfp_path;
    dfp_tag = 0;
    run.store(false);

    device_manager->opendfp(dfp_path, dfp_tag);
    dfp_valid = device_manager->get_dfp_validity(dfp_tag);
    if(!dfp_valid){
        throw runtime_error("Cannot parse dfp file - Please check given dfp");
    }

    setup_status = device_manager->setup_mxa(dfp_tag, device_ids_to_use);
    if(setup_status){

        device_manager->attach_dfp_to_device(dfp_tag);
        device_manager->download_dfp_to_device(dfp_tag);
        device_manager->init_mx_models(dfp_tag, &models);

    }
    else{
         throw runtime_error("MxAccl setup failed: Cannot configure the MXA devices given");
    }
    return dfp_tag;
}

int MxAccl::connect_dfp(const uint8_t *dfp_bytes, std::vector<int>& device_ids_to_use){
    if(dfp_valid){
        throw std::runtime_error("Only one dfp allowed per Accl object");
    }

    if(device_ids_to_use.empty()){
        throw runtime_error("device_ids_to_use parameter cannot be empty");
    }

    dfp_path = std::filesystem::path("<BYTES>");
    dfp_tag = 0;
    run.store(false);

    device_manager->opendfp_bytes(dfp_bytes, dfp_tag);
    dfp_valid = device_manager->get_dfp_validity(dfp_tag);
    if(!dfp_valid){
        throw runtime_error("Cannot parse dfp file - Please check given dfp");
    }

    setup_status = device_manager->setup_mxa(dfp_tag, device_ids_to_use);
    if(setup_status){

        device_manager->attach_dfp_to_device(dfp_tag);
        device_manager->download_dfp_to_device(dfp_tag);
        device_manager->init_mx_models(dfp_tag, &models);

    }
    else{
         throw runtime_error("MxAccl setup failed: Cannot configure the MXA devices given");
    }
    return dfp_tag;
}


void MxAccl::start(){
    if (dfp_valid)
    {
        int num_models =  device_manager->get_dfp_num_models(dfp_tag);

        if(get_num_streams() == 0){
            throw logic_error("accl start called before connect_stream for auto threading");
        }
        else{
            // set run status to true
            run.store(true);
            for (int i = 0; i < num_models; ++i)
            {
                //start models that have a stream connected to it
                if(models[i]->get_num_streams()>0)
                    models[i]->model_start();
            }
        }
    }
    else
    {
        printf("MxAccl::start(): ERROR, valid==false. Did initialization fail?");
    }
}

void MxAccl::wait(){
    if(dfp_valid){
        int num_models =  device_manager->get_dfp_num_models(dfp_tag);
        for (int i = 0; i < num_models; ++i)
        {
            //wait for the all the models to finish streaming
            if(models[i]->get_num_streams()>0)
                models[i]->model_wait();
        }
    }
}

void MxAccl::stop()
{

    if (dfp_valid && run.load())
    {
        int num_models =  device_manager->get_dfp_num_models(dfp_tag);
        for (int i = 0; i < num_models; ++i)
        {   
            //Stop all models
            if(models[i]->get_num_streams()>0)
                models[i]->model_stop();
        }
        run.store(false);
    }
}

void MxAccl::connect_stream(float_callback_t in_cb, float_callback_t out_cb, int stream_id, int model_id, int dfp_id){
    //!!!!TODO: Need to use dfp_id for future
    if(dfp_id!=0){
        throw std::runtime_error("only one dfp per MxAccl allowed");
    }
    models[model_id]->connect_stream(in_cb,out_cb,stream_id);
}

// void MxAccl::connect_stream(int_callback_t in_cb, float_callback_t out_cb, int stream_id, int model_id){
//     models[model_id]->connect_stream(in_cb,out_cb,stream_id);
// }

void MxAccl::connect_post_model(std::filesystem::path post_model_path, int model_idx, const std::vector<size_t>& post_size_list){
    models[model_idx]->model_set_post(post_model_path,post_size_list);
}

void MxAccl::connect_pre_model(std::filesystem::path pre_model_path, int model_idx){
    models[model_idx]->model_set_pre(pre_model_path);
}

MxAccl::~MxAccl()
{
    //if destructor is called without calling stop, need to exit gracefully
    if(run.load()){
        this->stop();
    }
    //Close the MXA
    // close_mxa();
    if(dfp_valid && setup_status ){
        int num_models =  device_manager->get_dfp_num_models(dfp_tag);
        for (int i = 0; i < num_models; ++i)
        {
            //delete all the models created
            delete models[i];
        }
        models.clear();
        device_manager->cleanup__all_dfps();
        device_manager->close_all_devices();
    }

    
    // device_manager->cleanup_all_setup_maps();
    //delete the DFP object
    if(device_manager!=NULL){
        // if(we_opened_dfp){
            delete device_manager;
            device_manager = NULL;
        }
        else{
            device_manager = NULL;
        }
    // }
}

int MxAccl::get_num_models(){
    if(!dfp_valid){
        return 0;
    }
    return  device_manager->get_dfp_num_models(dfp_tag);
}

int MxAccl::get_num_streams(){

    int ans = 0;
    if(!dfp_valid){
        return ans;
    }
    //return sum of num streams in each model
    int num_models =  device_manager->get_dfp_num_models(dfp_tag);
    for(int i =0; i<num_models; ++i){
        ans+=models[i]->get_num_streams();
    }
    return ans;
}

int MxAccl::get_dfp_num_chips(){
    if(!dfp_valid){
        throw std::runtime_error("dfp is not connected.");
    }
    return  device_manager->get_dfp_num_chips(dfp_tag);
}

MX::Types::MxModelInfo MxAccl::get_model_info(int model_id) const{
    if(model_id>= static_cast<int>(models.size())){
        std::ostringstream oss;
        int num_models = models.size();
        oss << "Invalid model ID passed : Number of models available = "<<num_models<<"\n model_id range is 0 to "<<num_models-1;
        throw runtime_error(oss.str());
    }
    else{
        return models[model_id]->return_model_info();
    }
}

MX::Types::MxModelInfo MxAccl::get_pre_model_info(int model_id) const{
    if(model_id>= static_cast<int>(models.size())){
        std::ostringstream oss;
        int num_models = models.size();
        oss << "Invalid model ID passed : Number of models available = "<<num_models<<"\n model_id range is 0 to "<<num_models-1;
        throw runtime_error(oss.str());
    }
    else{
        return models[model_id]->return_pre_model_info();
    }
}

MX::Types::MxModelInfo MxAccl::get_post_model_info(int model_id) const{
    if(model_id>= static_cast<int>(models.size())){
        std::ostringstream oss;
        int num_models = models.size();
        oss << "Invalid model ID passed : Number of models available = "<<num_models<<"\n model_id range is 0 to "<<num_models-1;
        throw runtime_error(oss.str());
    }
    else{
        return models[model_id]->return_post_model_info();
    }
}


void MxAccl::set_num_workers(int input_num_workers, int output_num_workers, int model_idx){
    if(model_idx>= static_cast<int>(models.size())){
        std::ostringstream oss;
        int num_models = models.size();
        oss << "Invalid model ID passed : Number of models available = "<<num_models<<"\n model_id range is 0 to "<<num_models-1;
        throw runtime_error(oss.str());
    }
    models[model_idx]->set_num_workers(input_num_workers,output_num_workers);
}

void MxAccl::set_parallel_fmap_convert(int num_threads, int model_idx){
    if(model_idx>= static_cast<int>(models.size())){
        std::ostringstream oss;
        int num_models = models.size();
        oss << "Invalid model ID passed : Number of models available = "<<num_models<<"\n model_id range is 0 to "<<num_models-1;
        throw runtime_error(oss.str());
    }
    models[model_idx]->set_parallel_fmap_convert(num_threads);
}


