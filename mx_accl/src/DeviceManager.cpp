#include <unordered_map>
#include <memx/accl/DeviceManager.h>
#include <sstream>
#include <fstream>

using namespace MX::Runtime;
using namespace MX::Types;
using namespace MX::Utils;
using namespace std;

DeviceManager::DeviceManager(){
    
    all_devices_count = 0;
    available_devices = 0;
    required_devices = 0;

    // let' get all devices and manage them
    this->get_available_devices();
}

bool DeviceManager::opendfp_bytes(const uint8_t *b, int dfp_tag){

    dfp_rt_info ddi;
    ddi.dfp = new Dfp::DfpObject(b);
    Dfp::DfpMeta temp_meta = ddi.dfp->get_dfp_meta();

    ddi.is_bytes = true;
    ddi.dfp_filename_path = std::filesystem::path("<BYTES>");
    ddi.dfp_num_chips = temp_meta.num_chips;
    ddi.num_models = temp_meta.num_models;
    ddi.mxa_gen = temp_meta.mxa_gen;
    ddi.dfp_meta = temp_meta;
    ddi.use_multigroup_lb = temp_meta.use_multigroup_lb;
    ddi.context_ids_vector  = {};
    ddi.valid = ddi.dfp->valid;

    auto it = this->dfp_mxa_map.find(dfp_tag);
    if(it == this->dfp_mxa_map.end()){
        this->dfp_mxa_map.emplace(dfp_tag, std::move(ddi));
    }
    else{  
        it->second = ddi;
    }

    return ddi.valid;
}

bool DeviceManager::opendfp(const std::filesystem::path dfp_filename, int dfp_tag){

    dfp_rt_info ddi;
    ddi.dfp = new Dfp::DfpObject(dfp_filename.string().c_str());
    Dfp::DfpMeta temp_meta = ddi.dfp->get_dfp_meta();

    ddi.is_bytes = false;
    ddi.dfp_filename_path = dfp_filename;
    ddi.dfp_num_chips = temp_meta.num_chips;
    ddi.num_models = temp_meta.num_models;
    ddi.mxa_gen = temp_meta.mxa_gen;
    ddi.dfp_meta = temp_meta;
    ddi.use_multigroup_lb = temp_meta.use_multigroup_lb;
    ddi.context_ids_vector  = {};
    ddi.valid = ddi.dfp->valid;

    auto it = this->dfp_mxa_map.find(dfp_tag);
    if(it == this->dfp_mxa_map.end()){
        this->dfp_mxa_map.emplace(dfp_tag, std::move(ddi));
    }
    else{  
        it->second = ddi;
    }

    return ddi.valid;
}


void DeviceManager::get_available_devices(){
   
    memx_status status = memx_operation_get_device_count(&all_devices_count);
    if (memx_status_error(status))
    {
        throw runtime_error("Couldn't get device count");
    }

    // MX::Runtime::device_info di;
    available_devices = 0;
    for(int d = 0; d < all_devices_count ; d++){
        int device_id = d;
        memx_status status = memx_trylock(device_id);
        if(memx_status_error(status)){
            std::cout<<"device locked - Trying next device \n";
        }
        else{
            available_devices++;
            available_devices_id.push_back(device_id);

            MX::Runtime::device_info di;    
            uint8_t device_chip_count = 0;
            status = memx_get_total_chip_count(device_id, &device_chip_count);

            di.chip_count = device_chip_count;
            di.is_device_open = false;
            di.number_of_contexts_attached = 0;
            di.contexts_ids_attached = {};
            di.current_config = MEMX_MPU_GROUP_CONFIG_ONE_GROUP_FOUR_MPUS;
            
            auto device_it = this->available_mxa_device_map.find(device_id);
            
            if(device_it == this->available_mxa_device_map.end()){
                this->available_mxa_device_map.emplace(device_id , di);
            }
            else{
                device_it->second = di;
            }
            memx_unlock(d);
        }
    }
}

void DeviceManager::throw_chip_exception(int pdfp_chips, int pdevice_chips, int device_id){
    std::ostringstream oss;
    oss << "this dfp is made for " << pdfp_chips << " but only " << pdevice_chips << " are available on device "<<device_id;
    throw runtime_error(oss.str());
}

void DeviceManager::throw_mxa_gen_exception(int pdfp_num_chips){
    std::ostringstream oss;
    oss << "this dfp is made for CASCADE gen for " << pdfp_num_chips << "Cannot be configured at runtime \n";
    throw runtime_error(oss.str());
}

void DeviceManager::print_available_devices(){

    std::cout << "\n\tAvailable Devices: \n\n";
    if (available_devices_id.empty()) {
        std::cout << "None";
    } else {
            std::cout << "-----------------------------\n";
            std::cout << "| " << std::setw(10) << "Device ID" << " | " << std::setw(12) << "Chip Count" << " |\n";
            std::cout << "-----------------------------\n";

            for (size_t i = 0; i < available_devices_id.size(); i++) {
                int device_id = available_devices_id[i];
                std::cout << "| " << std::setw(10) << device_id
                          << " | " << std::setw(12) << available_mxa_device_map.at(device_id).chip_count
                          << " |\n";
            }

            std::cout << "-----------------------------\n\n";
    }
}

void DeviceManager::throw_device_not_available_exception(int pdevice_id){

    std::cout<<"Device " << pdevice_id <<" is not available to use \n";
    print_available_devices();
    throw runtime_error("Try using the available devices given above");

}


void DeviceManager::set_power_mode(int device_id, int num_chips){

  #ifdef __GNUC__
    // ignore the fact this variable is unused, to satisfy -Werror
    __attribute__((unused)) memx_status status;
  #else
    // else we're kind of stuck, lol
    memx_status status;
  #endif

    uint16_t c4_freq = 600;
    uint16_t c4_volt = 700;
    uint16_t c2_freq = 600;
    uint16_t c2_volt = 700;
   
  #ifdef __linux__
    // LINUX READ FILE

    if(std::filesystem::exists("/etc/memryx/power.conf")){

        // read each line
        std::ifstream fd("/etc/memryx/power.conf");
        for( std::string line; getline( fd, line ); ){
            if(line[0] == '#')
                continue;
            std::string varname = line.substr(0,6);
            if(varname == "FREQ4C"){
                std::string val = line.substr(7,3);
                c4_freq = (uint16_t) std::stoi(val);
            } else if(varname == "VOLT4C"){
                std::string val = line.substr(7,3);
                c4_volt = (uint16_t) std::stoi(val);
            } else if(varname == "FREQ2C"){
                std::string val = line.substr(7,3);
                c2_freq = (uint16_t) std::stoi(val);
            } else if(varname == "VOLT2C"){
                std::string val = line.substr(7,3);
                c2_volt = (uint16_t) std::stoi(val);
            }
        }

    }

    // else we use the defaults

  #else
    // Windows: just use defaults for now
  #endif

#ifdef __linux__
    // SET THE STUFF
    if(num_chips == 4){
        status = memx_set_feature(device_id, 0, OPCODE_SET_FREQUENCY, c4_freq);
        status = memx_set_feature(device_id, 1, OPCODE_SET_FREQUENCY, c4_freq);
        status = memx_set_feature(device_id, 2, OPCODE_SET_FREQUENCY, c4_freq);
        status = memx_set_feature(device_id, 3, OPCODE_SET_FREQUENCY, c4_freq);
        status = memx_set_feature(device_id, 0, OPCODE_SET_VOLTAGE, c4_volt);
    } else if(num_chips == 2){
        status = memx_set_feature(device_id, 0, OPCODE_SET_FREQUENCY, c2_freq);
        status = memx_set_feature(device_id, 1, OPCODE_SET_FREQUENCY, c2_freq);
        status = memx_set_feature(device_id, 0, OPCODE_SET_VOLTAGE, c2_volt);
    }
#else
    //Not supported for windows currently
#endif


}



bool DeviceManager::configure_device(int device_id, int device_chip_count, int pdfp_num_chips, float pmxa_gen){

    memx_status status = MEMX_STATUS_OK;

    if(pmxa_gen == MEMX_DEVICE_CASCADE){
        memx_unlock(device_id);
        throw_mxa_gen_exception(device_chip_count);
    }
    else{
        if(pdfp_num_chips>device_chip_count){
            memx_unlock(device_id);
            throw_chip_exception(pdfp_num_chips, device_chip_count, device_id);
        }
        //Change the MPU config based on DFP if needed
        else if(pdfp_num_chips==8 && device_chip_count==8){
            status = memx_config_mpu_group(device_id, MEMX_MPU_GROUP_CONFIG_ONE_GROUP_EIGHT_MPUS);
            available_mxa_device_map.at(device_id).current_config = MEMX_MPU_GROUP_CONFIG_ONE_GROUP_EIGHT_MPUS;
        }
        else if(pdfp_num_chips==4){
            status = memx_config_mpu_group(device_id, MEMX_MPU_GROUP_CONFIG_ONE_GROUP_FOUR_MPUS);
            available_mxa_device_map.at(device_id).current_config = MEMX_MPU_GROUP_CONFIG_ONE_GROUP_FOUR_MPUS;
        }
        else if(pdfp_num_chips==2){
            status = memx_config_mpu_group(device_id, MEMX_MPU_GROUP_CONFIG_TWO_GROUP_TWO_MPUS);
            available_mxa_device_map.at(device_id).current_config = MEMX_MPU_GROUP_CONFIG_TWO_GROUP_TWO_MPUS;
        }
        else{
            memx_unlock(device_id);
            throw_chip_exception(pdfp_num_chips, device_chip_count, device_id);
        }
    }

    if(!memx_status_error(status))
        return true;
    else    
        return false; 
}

bool DeviceManager::connect_device(int dfp_tag, int device_id){

    memx_status lock_status;
    bool configure_status;
    // Lock MXA device
    lock_status = memx_trylock(device_id);
    if(memx_status_error(lock_status)){
        std::ostringstream oss;
        oss << "Cannot acquire lock on an available MXA device, device ID = "<<device_id;
        throw runtime_error(oss.str());
    }
    
    // check chip count and see if dfp chip requires that much
    uint8_t device_chip_count = this->available_mxa_device_map.at(device_id).chip_count;
    int l_dfp_num_chips = this->dfp_mxa_map.at(dfp_tag).dfp_num_chips;
    float l_mxa_gen = this->dfp_mxa_map.at(dfp_tag).mxa_gen;
    bool l_use_mg_lb = this->dfp_mxa_map.at(dfp_tag).use_multigroup_lb;
    
    // if the dfp does not require a 8 chip MXA skip this device
    if(device_chip_count > 4 && l_dfp_num_chips <=4 ){
        memx_unlock(device_id);
        configure_status = false;
        // continue;
    }
    else{
        //if device has the required number of chips configure the device and open necessary contexts    
        configure_status = configure_device(device_id, device_chip_count, l_dfp_num_chips, l_mxa_gen);
        if(l_dfp_num_chips == 4){
            set_power_mode(device_id, 4);
        } else if(l_dfp_num_chips == 2){
            if(l_use_mg_lb){
                set_power_mode(device_id, 4);
            } else {
                set_power_mode(device_id, 2);
            }
        }
        open_devices.push_back(device_id);            
        available_mxa_device_map.at(device_id).is_device_open = true;
    }
    return configure_status;
}

bool DeviceManager::setup_mxa(int dfp_tag, std::vector<int>& pgroup_ids){

    
    required_devices = pgroup_ids.size();

    open_devices.reserve(required_devices);

    bool all_device_connected = true;

    for(int d = 0 ; d < required_devices ; d++){
        bool connect_status;
        int device_id = pgroup_ids[d];

        auto device_it = this->available_mxa_device_map.find(device_id);
        if (device_it != available_mxa_device_map.end() && !device_it->second.is_device_open) {

            connect_status = connect_device(dfp_tag, device_id);
            if(!connect_status){
                all_device_connected = false;
                throw runtime_error("Error while configuring a device, Please check the device.  Device ID = "+to_string(device_id));
            }
        }
        else{
            all_device_connected = false;
            throw_device_not_available_exception(device_id);
        }
       
    }

    return all_device_connected;
}

void DeviceManager::attach_dfp_to_device(int dfp_tag){


    // While attaching dfp to an already configured device - check configuration and decide number of contexts required for tat dfp
    // this means that the dfps added later should have the same configuration as the initial dfp
    // if a new config dfp is added then setup and config has to be called again

    for(int d = 0 ; d < required_devices ; d++){
        int device_id = open_devices[d];
        int number_of_contexts = 0;
        if(available_mxa_device_map.at(device_id).current_config == MEMX_MPU_GROUP_CONFIG_ONE_GROUP_FOUR_MPUS){
            number_of_contexts = 1;            
        }
        else{
            if(dfp_mxa_map.at(dfp_tag).use_multigroup_lb)
                number_of_contexts = 2;
            else
                number_of_contexts = 1;
        }

        //reserving two contexts per device as maximum two contexts are possible. (ignoring model swaping)
        int context_init = device_id*2;
        for(int i = 0; i < number_of_contexts; i++){
            // Context IDs are limited based on driver limitation (0 to 31) across all the devices
            // hence the context_id_tracker that keeps running count on all IDs
            // get the recent context_id to assign for a dfp
            
            if(context_init >= 32){
                throw runtime_error("cannot open more than 32 contexts\n");
            }

            memx_status status = memx_open(context_init, device_id, MEMX_DEVICE_CASCADE_PLUS);
            if (memx_status_error(status)){
                throw runtime_error("Couldn't open a context with a device, please verify the MXA connection");
            }
            else{
                dfp_mxa_map.at(dfp_tag).context_ids_vector.push_back(context_init);
                available_mxa_device_map.at(device_id).contexts_ids_attached.push_back(context_init);
                available_mxa_device_map.at(device_id).number_of_contexts_attached++;
                context_init++;
            }
        }

        int mpu_group_count = 0;
        memx_status status = memx_operation_get_mpu_group_count(device_id, &mpu_group_count);
        if (memx_status_error(status))
        {
            throw runtime_error("Couldn't get the mpu group count");
        }
    }
}

void DeviceManager::download_dfp_to_device(int dfp_tag){

        // Since download of dfp has to happen a lot of times this function has been separated and can be called.
        // will download to all the contexts that has been assigned to that dfp and will enable the stream for that context
        int dfp_num_contexts = dfp_mxa_map.at(dfp_tag).context_ids_vector.size();
        for(int i = 0; i < dfp_num_contexts; i++){
            int ctx = dfp_mxa_map.at(dfp_tag).context_ids_vector[i];

            memx_status status;
            if(dfp_mxa_map.at(dfp_tag).is_bytes){
                status = memx_download_model(ctx,  (const char*) dfp_mxa_map.at(dfp_tag).dfp->src_dfp_bytes, 0 /*model_idx? */, MEMX_DOWNLOAD_TYPE_WTMEM_AND_MODEL_BUFFER);
            } else {
                status = memx_download_model(ctx,  dfp_mxa_map.at(dfp_tag).dfp->path().c_str(), 0 /*model_idx? */, MEMX_DOWNLOAD_TYPE_WTMEM_AND_MODEL);
            }
            if (memx_status_error(status))
            {
                std::ostringstream oss;
                oss<< "Download of DFP "<<  dfp_mxa_map.at(dfp_tag).dfp->path() <<" failed";
                throw runtime_error(oss.str());
            }

            // start stream
            status = memx_set_stream_enable(ctx, 0 /*wait time?*/);
            if (memx_status_error(status))
            {
                throw runtime_error("Enable stream failed");
            }
    }
}

void DeviceManager::cleanup__all_dfps(){
    for(auto& it  : this->dfp_mxa_map ){
        it.second.context_ids_vector.clear();

        if(it.second.dfp!=NULL){
            delete it.second.dfp;
            it.second.dfp = NULL;
        }
        else{
            it.second.dfp = NULL;
            continue;
        }
        it.second.valid = false;
    }
    this->dfp_mxa_map.clear();
}

void DeviceManager::close_all_devices(){

    if(!open_devices.empty()){
        for (int i =0; i<static_cast<int>(open_devices.size()); i++){
            int device_id = open_devices[i];

            int num_contexts = available_mxa_device_map.at(device_id).number_of_contexts_attached;
            for(int ctx = 0; ctx < num_contexts ; ctx++){
                memx_status status;
                status = memx_close(available_mxa_device_map.at(device_id).contexts_ids_attached[ctx]);
                if (memx_status_error(status))
                {
                    throw runtime_error("MXA context close failed");
                }
            }
            available_mxa_device_map.at(device_id).number_of_contexts_attached = 0;
            available_mxa_device_map.at(device_id).contexts_ids_attached.clear();

            memx_status ulock_status;
            ulock_status = memx_unlock(device_id);
            if (memx_status_error(ulock_status)){
                throw runtime_error("MXA device unlock failed");
            }
            else{
                
            }
        }
        open_devices.clear();
        available_devices_id.clear();
        available_devices = 0;
        available_mxa_device_map.clear();
    }
}

void DeviceManager::init_mx_models(int dfp_tag, std::vector<ModelBase *>* mxmodel_vector ){

    int num_models = dfp_mxa_map.at(dfp_tag).num_models;
    for (int i = 0; i < num_models; ++i)
    {

        vector<uint8_t> in_ports = dfp_mxa_map.at(dfp_tag).dfp_meta.model_inports[i];
        uint8_t format = dfp_mxa_map.at(dfp_tag).dfp->input_port(in_ports[0])->format;
        
        if(format == MX_FMT_RGB888){
            // MxModel<uint8_t> *im = new MxModel<uint8_t>(i, dfp_mxa_map.at(dfp_tag).dfp, &dfp_mxa_map.at(dfp_tag).context_ids_vector);
            // mxmodel_vector->push_back(im); 
            throw(std::runtime_error("int inputs are currently not supported"));               
        }
        else{
            MxModel<float> *fm = new MxModel<float>(i, dfp_mxa_map.at(dfp_tag).dfp, &dfp_mxa_map.at(dfp_tag).context_ids_vector);
            mxmodel_vector->push_back(fm);
        }
    }
}

// bool DeviceManager::dfp_tag_duplicate_check(int dfp_tag){
//     bool is_tag_duplicate;
//     auto it = this->dfp_mxa_map.find(dfp_tag);
//     if(it == this->dfp_mxa_map.end()){
//         is_tag_duplicate = false;
//     }
//     else{  
//         is_tag_duplicate =  true;
//     }

//     return is_tag_duplicate;

// }

int DeviceManager::get_dfp_num_chips(int dfp_tag){
    return dfp_mxa_map.at(dfp_tag).dfp_num_chips;
}

int DeviceManager::get_dfp_num_models(int dfp_tag){
    return dfp_mxa_map.at(dfp_tag).num_models;
}

bool DeviceManager::get_dfp_validity(int dfp_tag){
    return dfp_mxa_map.at(dfp_tag).valid;
}

// void DeviceManager::cleanup_all_setup_maps(){
    
//     DeviceManager::dfp_mxa_map.clear();
//     DeviceManager::available_mxa_device_map.clear();
// }

/*
// Additional get function disabled for now but might need later

float DeviceManager::get_dfp_mxa_gen(){
    return mxa_gen;
}

int DeviceManager::get_connected_devices_count(){
    return all_devices_count;

}

int DeviceManager::get_available_device_count(){
    return available_devices;
}


Dfp::DfpMeta DeviceManager::get_dfp_meta(){
    return this->dfp_meta;
}
*/
