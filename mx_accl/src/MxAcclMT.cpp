#include <memx/accl/MxAcclMT.h>
#include <sstream>

using namespace MX::Runtime;
using namespace MX::Types;
using namespace MX::Utils;

MxAcclMT::MxAcclMT(){
    dfp_valid = false;
    setup_status = false;
    device_manager = new MX::Runtime::DeviceManager();
    // device_manager->print_available_devices();

}

int MxAcclMT::connect_dfp(const std::filesystem::path pdfp_path, int group_id)
{
    std::vector<int> devices_to_use = {group_id};
    return connect_dfp(pdfp_path,devices_to_use);
}

int MxAcclMT::connect_dfp(const uint8_t *dfp_bytes,int group_id){
    std::vector<int> devices_to_use = {group_id};
    return connect_dfp(dfp_bytes,devices_to_use);
}

int MxAcclMT::connect_dfp(const std::filesystem::path pdfp_path,  std::vector<int>& device_ids_to_use)
{
    if(dfp_valid){
        throw std::runtime_error("Only one dfp allowed per Accl object");
    }
    if(device_ids_to_use.empty()){
        throw(std::runtime_error("device_ids_to_use parameter cannot be empty"));
    }

    dfp_path = pdfp_path;
    manual_run.store(false);
    dfp_tag = 0;
    
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

        int num_models =  device_manager->get_dfp_num_models(dfp_tag);
        manual_run.store(true);
        for(int i=0; i<num_models; i++){
            models[i]->model_manual_start();
        }
    }
    else{
         throw runtime_error("MxAcclMT setup failed: Cannot configure the MXA device");
    }
    return dfp_tag;
}

int MxAcclMT::connect_dfp(const uint8_t *dfp_bytes, std::vector<int>& device_ids_to_use){
    if(dfp_valid){
        throw std::runtime_error("Only one dfp allowed per Accl object");
    }

    if(device_ids_to_use.empty()){
        throw runtime_error("device_ids_to_use parameter cannot be empty");
    }

    dfp_path = std::filesystem::path("<BYTES>");
    dfp_tag = 0;

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
         throw runtime_error("MxAcclMT setup failed: Cannot configure the MXA devices given");
    }
    return dfp_tag;
}

void MxAcclMT::connect_post_model(std::filesystem::path post_model_path, int model_idx, const std::vector<size_t>& post_size_list){
    models[model_idx]->model_set_post(post_model_path,post_size_list);
}

void MxAcclMT::connect_pre_model(std::filesystem::path pre_model_path, int model_idx){
    models[model_idx]->model_set_pre(pre_model_path);
}

MxAcclMT::~MxAcclMT()
{
    if(dfp_valid && setup_status ){
        int num_models =  device_manager->get_dfp_num_models(dfp_tag);

        for (int i = 0; i < num_models; ++i)
        {
            //delete all the models created
            delete models[i];
        }
        models.clear();
    }
    device_manager->cleanup__all_dfps();
    device_manager->close_all_devices();

    //delete the DFP object
    if(device_manager!=NULL){
            delete device_manager;
            device_manager = NULL;
        }
        else{
            device_manager = NULL;
        }
}

int MxAcclMT::get_num_models(){
    if(!dfp_valid){
        return 0;
    }
    return  device_manager->get_dfp_num_models(dfp_tag);
}

int MxAcclMT::get_dfp_num_chips(){
    if(!dfp_valid){
        throw std::runtime_error("dfp is not connected.");
    }
    return  device_manager->get_dfp_num_chips(dfp_tag);
}

MX::Types::MxModelInfo MxAcclMT::get_model_info(int model_id) const{
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

MX::Types::MxModelInfo MxAcclMT::get_pre_model_info(int model_id) const{
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

MX::Types::MxModelInfo MxAcclMT::get_post_model_info(int model_id) const{
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

bool MxAcclMT::send_input(std::vector<float*> in_data, int model_id, int pstream_id, int dfp_id, bool channel_first, int32_t timeout ){
    //!!!!TODO: Need to use dfp_id for future
    if(dfp_id!=0){
        throw std::runtime_error("only one dfp per MxAccl allowed");
    }
    if(model_id>= static_cast<int>(models.size())){
        std::ostringstream oss;
        int num_models = models.size();
        oss << "Invalid model ID passed : Number of models available = "<<num_models<<"\n model_id range is 0 to "<<num_models-1;
        throw runtime_error(oss.str());
    }
    return models[model_id]->model_manual_send(in_data, pstream_id,channel_first,timeout);
}

// bool MxAcclMT::send_input(std::vector<uint8_t*> in_data, int model_id, int pstream_id, bool channel_first, int32_t timeout ){
//     return models[model_id]->model_manual_send(in_data, pstream_id,channel_first,timeout);
// }

bool MxAcclMT::receive_output(std::vector<float*> &out_data, int pmodel_id, int pstream_id, int dfp_id, bool channel_first, int32_t timeout){
    //!!!!TODO: Need to use dfp_id for future
    if(dfp_id!=0){
        throw std::runtime_error("only one dfp per MxAccl allowed");
    }
    if(pmodel_id>= static_cast<int>(models.size())){
        std::ostringstream oss;
        int num_models = models.size();
        oss << "Invalid model ID passed : Number of models available = "<<num_models<<"\n model_id range is 0 to "<<num_models-1;
        throw runtime_error(oss.str());
    }
    return models[pmodel_id]->model_manual_receive(out_data, pstream_id, channel_first,timeout);
}

void MxAcclMT::set_parallel_fmap_convert(int num_threads, int model_idx){
    if(model_idx>= static_cast<int>(models.size())){
        std::ostringstream oss;
        int num_models = models.size();
        oss << "Invalid model ID passed : Number of models available = "<<num_models<<"\n model_id range is 0 to "<<num_models-1;
        throw runtime_error(oss.str());
    }
    models[model_idx]->set_parallel_fmap_convert(num_threads);
}

bool MxAcclMT::run(std::vector<float *> in_data, std::vector<float*> &out_data, int pmodel_id, int pstream_id, int dfp_id, bool in_channel_first, bool out_channel_first, int32_t timeout){
    //!!!!TODO: Need to use dfp_id for future
    if(dfp_id!=0){
        throw std::runtime_error("only one dfp per MxAccl allowed");
    }
    if(pmodel_id>= static_cast<int>(models.size())){
        std::ostringstream oss;
        int num_models = models.size();
        oss << "Invalid model ID passed : Number of models available = "<<num_models<<"\n model_id range is 0 to "<<num_models-1;
        throw runtime_error(oss.str());
    }
    return models[pmodel_id]->manual_run(in_data,out_data,pstream_id,in_channel_first,out_channel_first,timeout);
}
