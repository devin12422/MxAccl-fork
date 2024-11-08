#include <iostream>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <getopt.h> /* getopt_long() */
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <errno.h>
#include <thread>
#include <iomanip>
#include <numeric>

#include "memx/accl/MxAccl.h"
#include "memx/accl/MxAcclMT.h"

#define MAX_FPS_OPT 1000
#define IW_OPT 1001
#define OW_OPT 1002
#define ND_REQ 1003
#define MT_MODE 1004
#define MD_IDS 1005
#define DS_AL 1006

const char  *default_dfp_path = "model/single_ssd_mobilenet_300_MX3.dfp";
int frame_count = 1000;

static char *dfp_path = NULL;
static int grp_id = 0;
static int max_fps = 0;
std::chrono::milliseconds custom_duration_ms = 0ms;
float custom_duration =0;

static bool multi_stream_bench = false;
static int num_fmap_convert_threads = 1;
static bool verbose = false;
static bool manual_threading = false;
static bool bench_tool = false;
int ms_done_flag = 0;
std::atomic_bool runflag;
std::atomic_bool hello_flag;
static int num_streams = 1;
int num_models = 0;
char fps_text[64] = "FPS = ";
float fps_number = 0.0;
std::chrono::milliseconds start_ms;
std::chrono::milliseconds temp_start_ms;
// accl object
MX::Runtime::MxAccl* accl = NULL;
MX::Runtime::MxAcclMT* accl_mt = NULL;


int num_input_workers = 0;
int num_output_workers = 0;
int num_devices = 1;

//mutit device support
std::vector<int> device_ids;
bool multi_device_bench = false;
// multistream variables
std::vector<int> sent_frame_count_vector;
std::vector<int> recv_frame_count_vector;
std::vector<MX::Types::MxModelInfo> model_info_vector;
std::vector<std::vector<float*>> ifmap_vector;
std::vector<std::vector<float*>> ofmap_vector;
std::vector<std::chrono::milliseconds> temp_start_ms_vector;
std::vector<float> fps_values;
std::vector<int> fps_avg_counters;;

int dfp_num_chips = 0;
int recv_all_count = 0;
//Manual threads
std::thread **stream_send_threads;
std::thread **stream_recv_threads;

//signal handler
void signal_handler(int p_signal){
    runflag.store(false);
}


static void _error_exit(const char *s){
        fprintf(stderr, "%s error\n", s);
        exit(EXIT_FAILURE);
}

static void print_usage(int argc, char **argv){
        std::cout << "Usage: " << argv[0] << " [options] \n\n" <<
                    "Options:\n" <<
                      "-H | --hello           Check connection to MXA devices and get device info\n" <<
                      "-d | --dfp filename    DFP model file to test, such as '" << default_dfp_path << "'\n" <<
                      "-m | --multistream     Run accl bench for multistream\n" <<
                      "-n | --numstreams      Number of streams to run multistream accl bench, default= " << num_streams << " for singlestream 2 if multistream is chosen\n"
                      "-c | --convert_threads Number of feature map format conversion threads, default= " << num_fmap_convert_threads << "\n" << 
                      "-h | --help            Print this message\n" <<
                      "-g | --group           Accerator group ID, default=" << grp_id << "\n" <<
                      "-f | --frames          Number of frame for testing inference performance, default=" << frame_count << " secs\n" <<
                      "-v | --verbose         print all the required logs\n"
                      "--max_fps              maximum allowed FPS per stream\n"<<
                      "--iw                   number of input pre-processing workers per model\n"<<
                      "--ow                   number of output post-processing workers per model\n"<<
                      "--device_ids           MXA device IDs to be used to run benchmark, used in cases of multi device use cases. Takes in a comma separated list of device IDss\n"<<
                      "--ls                   Allows lenient setup in multi device use cases, uses available devices in case if some of the passed IDs are not available.\n"<<
                      "--mt                   Runs benchmark tool with Manual Threading model of c++ API\n"
                      " ";
}

void parse_device_ids(const std::string& input) {
    std::stringstream ss(input);
    std::string token;
   while (std::getline(ss, token, ',')) {
        try {
            device_ids.push_back(std::stoi(token));
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid device ID: " << token << ". Skipping..." << std::endl;
        }
    }
}


static const char short_options[] = "d:Hhmvbn:c:g:f:";

static const struct option
    long_options[] = {
        {"dfp", required_argument, NULL, 'd'},
        {"hello", no_argument, NULL, 'H'},
        {"help", no_argument, NULL, 'h'},
        {"multistream", no_argument, NULL, 'm'},
        {"verbose", no_argument, NULL,'v'},
        {"numstreams", required_argument, NULL, 'n'},
        {"convert_threads", required_argument, NULL, 'c'},
        {"groupID", required_argument, NULL, 'g'},
        {"frames", required_argument, NULL, 'f'},
        {"max_fps", required_argument, 0,MAX_FPS_OPT},
        {"iw", required_argument, 0,IW_OPT},
        {"ow", required_argument, 0,OW_OPT},
        {"mt", no_argument, NULL, MT_MODE},
        {"device_ids", required_argument, 0, MD_IDS},
        {"ls", no_argument, NULL, DS_AL},
        {"bench", no_argument, 0,'b'},

        {0, 0, 0, 0 }};


void print_model_info(MX::Types::MxModelInfo pmodel_info){
    std::cout << "\033[3;33m*************************************************\n";
    std::cout << "*               Model Information               *\n";
    std::cout << "*************************************************\033[m\n";
    std::cout << "\n         Model Index : " << pmodel_info.model_index << "        \n";
    std::cout << "\nNum of in featureMaps : " << pmodel_info.num_in_featuremaps << "\n";
    
    std::cout << "\nIn featureMap Shapes \n";
    for(int i = 0; i<pmodel_info.num_in_featuremaps ; ++i){
        std::cout << "Shape of featureMap : " << i+1 << "\n";
        std::cout << "Layer Name : " << pmodel_info.input_layer_names[i] << "\n";
        std::cout << "H = " << pmodel_info.in_featuremap_shapes[i][0] << "\n";
        std::cout << "W = " << pmodel_info.in_featuremap_shapes[i][1] << "\n";
        std::cout << "Z = " << pmodel_info.in_featuremap_shapes[i][2] << "\n";
        std::cout << "C = " << pmodel_info.in_featuremap_shapes[i][3] << "\n";
    }

    std::cout << "\n\nNum of out featureMaps : " << pmodel_info.num_out_featuremaps << "\n";
    std::cout << "\nOut featureMap Shapes \n";
    for(int i = 0; i<pmodel_info.num_out_featuremaps ; ++i){
        std::cout << "Shape of featureMap : " << i+1 << "\n";
        std::cout << "Layer Name : " << pmodel_info.output_layer_names[i] << "\n";
        std::cout << "H = " << pmodel_info.out_featuremap_shapes[i][0] << "\n";
        std::cout << "W = " << pmodel_info.out_featuremap_shapes[i][1] << "\n";
        std::cout << "Z = " << pmodel_info.out_featuremap_shapes[i][2] << "\n";
        std::cout << "C = " << pmodel_info.out_featuremap_shapes[i][3] << "\n";
    }

    std::cout << "\033[3;33m*************************************************\033[m\n";
}



void generate_input_data(MX::Types::MxModelInfo pmodel_info, std::vector<float*>& pinput_data){

    pinput_data.reserve(pmodel_info.num_in_featuremaps);
    // std::cout<<"generating data\n";
    srand((unsigned)time(0));
    for(int i = 0; i<pmodel_info.num_in_featuremaps; i++){
        float* ifmap = new float[pmodel_info.in_featuremap_sizes[i]];
        for(int j = 0; j< pmodel_info.in_featuremap_sizes[i]; j++){
            ifmap[j] = rand()%256;
        //     ifmap[j] = 0;
        }
        pinput_data.push_back(ifmap);
    }
 
}


void cleanup(){

        for(auto ind : ifmap_vector){
                for (auto& ifmap : ind) {
                        if(ifmap!= NULL){
                                delete[] ifmap;
                                ifmap = NULL;
                        }
                }
        }
        ifmap_vector.clear();
        
        for(auto ofd : ofmap_vector){
                for (auto& ofmap : ofd) {
                        if(ofmap!=NULL){
                                delete[] ofmap;
                                ofmap = NULL;
                        }
                }
        }
        ofmap_vector.clear();

        if(manual_threading){
                for(int i=0; i<(num_models*num_streams);++i){
                        if(stream_send_threads[i]->joinable()){
                                  stream_send_threads[i]->join();      
                        }
                                
                        delete stream_send_threads[i];
                        stream_send_threads[i]=NULL;

                        stream_recv_threads[i]->join();
                        delete stream_recv_threads[i];
                        stream_recv_threads[i]=NULL;
                }
                delete[] stream_send_threads;
                stream_send_threads = NULL;
                delete[] stream_recv_threads;
                stream_recv_threads = NULL;
        }
}



bool incallback_ms(vector<const MX::Types::FeatureMap<float>*> dst, int streamLabel){

        if((sent_frame_count_vector[streamLabel]  < frame_count) && runflag.load()){
                // std::cout<< "incallback called \n";
                for(int i = 0; i<model_info_vector[streamLabel].num_in_featuremaps; i++){
                        dst[i]->set_data(ifmap_vector[streamLabel][i], false);
                }
                sent_frame_count_vector[streamLabel]++;
                return true;
        }
        else{
                ms_done_flag++;
                
                return false;
        }    
}

bool outcallback_ms(vector<const MX::Types::FeatureMap<float>*> src, int streamLabel){

    if(recv_frame_count_vector[streamLabel] < frame_count){
        // std::cout<<"outcallback called \n";
        for(int i = 0; i<model_info_vector[streamLabel].num_out_featuremaps; ++i){
                src[i]->get_data(ofmap_vector[streamLabel][i], false);
        }


        if( recv_frame_count_vector[streamLabel]!=0 && recv_frame_count_vector[streamLabel] % 50 == 0){
                std::chrono::milliseconds duration =
                        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()) - temp_start_ms_vector[streamLabel];

                if(max_fps>0 && duration.count()<custom_duration){
                        duration = custom_duration_ms;
                }
                float fps = (float) 50 * 1000 / (float)(duration.count());
                fps_avg_counters[streamLabel]++;
                fps_values[streamLabel] = ((fps_values[streamLabel]*(fps_avg_counters[streamLabel]-1))+fps) / fps_avg_counters[streamLabel];
                temp_start_ms_vector[streamLabel] = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
            
        }
        recv_frame_count_vector[streamLabel]++;
        return true;
    }
    else{

        return false;
    }

}

void print_bench_setting_info(){

        std::cout << "\nNumber of models in DFP               = " << num_models << "\n";
        std::cout << "Number of streams                     = " << num_streams << "\n";
        std::cout << "Number of streams to be connected     = " << num_streams * num_models << "\n";
        std::cout << "Number of frame per stream            = " << frame_count << "\n";
        std::cout << "Number of input workers set to        = " << ((num_input_workers == 0 || num_input_workers > num_streams) ? num_streams : num_input_workers) <<"\n";
        std::cout << "Number of output workers set to       = " << ((num_output_workers == 0 || num_output_workers > num_streams) ? num_streams : num_output_workers) <<"\n";
        std::cout << "number of devices used                = " << num_devices << "\n";
        std::cout << "Number of FMap conversion threads     = " << num_fmap_convert_threads << "\n";

}

void display_fps() {
        
        // Print table header
        std::cout << std::setw(10) << "Model" << std::setw(10) << "Stream" << std::setw(15) << "FPS" << std::endl;
        std::cout << std::setw(35) << std::setfill('-') << "-" << std::endl;
        std::cout << std::setfill(' ');

        // Print values
        for (size_t i = 0; i < fps_values.size(); ++i) {
                std::cout << std::setw(10) << model_info_vector[i].model_index << std::setw(10) << i << std::setw(15) << std::fixed << std::setprecision(2) << fps_values[i] << std::endl;
        }
}


void model_bench(int num_models){
        
        int model_unique_stream_start = 0;
        start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        fps_values.reserve(num_models*num_streams);
        fps_avg_counters.reserve(num_models*num_streams);
        ifmap_vector.reserve(num_models*num_streams);
        ofmap_vector.reserve(num_models*num_streams);
        sent_frame_count_vector.reserve(num_models*num_streams);
        recv_frame_count_vector.reserve(num_models*num_streams);
        temp_start_ms_vector.reserve(num_models*num_streams);
        model_info_vector.reserve(num_models*num_streams);

        for(int model_index = 0; model_index < num_models ; model_index++){
                MX::Types::MxModelInfo minfo = accl->get_model_info(model_index);
                if(verbose)
                print_model_info(minfo);
                
                for(int stream_id = 0; stream_id < num_streams ; stream_id++){

                        temp_start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
                        
                        model_info_vector.push_back(minfo);
                        

                        // model_info = minfo;
                        //input config
                        std::vector<float*> in_data;
                        generate_input_data(minfo, in_data);
                        ifmap_vector.push_back(in_data);

                        //output config 
                        std::vector<float*> out_data;
                        out_data.reserve(minfo.num_out_featuremaps);
                        for(int i=0; i<minfo.num_out_featuremaps ; i++){
                                float* ofmap = new float[minfo.out_featuremap_sizes[i]];
                                // std::cout<<"size of featuremap " << i << "  " << minfo.out_featuremap_sizes[i] << "\n";
                                out_data.push_back(ofmap);
                        }

                        ofmap_vector.push_back(out_data);

                        sent_frame_count_vector.push_back(0);
                        recv_frame_count_vector.push_back(0);
                        temp_start_ms_vector.push_back(temp_start_ms);
                        fps_values.push_back(0.0);
                        fps_avg_counters.push_back(0);
                        // running_fps_values.push_back(0.0);
                        accl->connect_stream(&incallback_ms, &outcallback_ms, model_unique_stream_start+stream_id, model_index );
                        if(verbose){
                                std::cout<<"Connected stream "<< model_unique_stream_start+stream_id <<" for model "<< model_index << "\n\n"; 
                                std::cout << "\033[3;33m*************************************************\033[m\n";   
                        }   
                }
                model_unique_stream_start+=num_streams;
        }
        int connected_streams = accl->get_num_streams();
        accl->start();
        std::cout << "Number of chips the dfp is compiled for = " << dfp_num_chips << "\n\n";
        if(verbose){
                std::cout<<"\n\n\n"<<std::flush;
                
                std::cout << "*************************************************\033[m\n\n";
        }
        if(!bench_tool){
                while(ms_done_flag < connected_streams){
                        display_fps();
                        std::cout<<std::flush;
                        std::cout << "\033["<<connected_streams+2<<"A";
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                }
        }
        
        accl->wait();

        if(!bench_tool)
        display_fps();

        accl->stop();
        if(!bench_tool)
        std::cout<<"\n";
        std::chrono::milliseconds duration =
                        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()) - start_ms;
        // fps_number = (float)frame_count * 1000 / (float)(duration.count());
        float fps_total = std::accumulate(fps_values.begin(), fps_values.end(), 0.0);
        float fps_per_stream = fps_total / connected_streams;
        std::cout << "\rAverage FPS per stream : "<< fps_per_stream << "\033[m\n";
        std::cout << "\rAverage FPS for DFP    : "<< fps_total << "\033[m\n";
        if(verbose){
                std::cout << "\n\n*************************************************\033[m\n";
                std::cout<<"\n\n";
        }
        cleanup();
}

void send_data(int stream_label){
        int sent_frame = 0; 

        while(sent_frame_count_vector[stream_label]  < frame_count && runflag.load()){
                accl_mt->send_input(ifmap_vector[stream_label], model_info_vector[stream_label].model_index, stream_label, false);
                sent_frame_count_vector[stream_label]++;
        }
}

void receive_data(int model_index,int stream_id_recv){
        MX::Types::MxModelInfo minfo = accl_mt->get_model_info(model_index);
        std::vector<float*> out_data;
        out_data.reserve(minfo.num_out_featuremaps);
        for(int i=0; i<minfo.num_out_featuremaps ; i++){
                float* ofmap = new float[minfo.out_featuremap_sizes[i]];
                out_data.push_back(ofmap);
        }
        float fps = 0.0;
        while(recv_frame_count_vector[stream_id_recv]  < frame_count && runflag.load()){
                if(sent_frame_count_vector[stream_id_recv]<=recv_frame_count_vector[stream_id_recv]){
                        continue;
                }
                accl_mt->receive_output(out_data, model_index, stream_id_recv, false);

                if( recv_frame_count_vector[stream_id_recv]!=0 && recv_frame_count_vector[stream_id_recv] % 50 == 0){
                        std::chrono::milliseconds duration =
                                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()) - temp_start_ms_vector[stream_id_recv];

                        if(max_fps>0 && duration.count()<custom_duration){
                                duration = custom_duration_ms;
                        }
                        fps = (float) 50 * 1000 / (float)(duration.count());
                        fps_avg_counters[stream_id_recv]++;
                        fps_values[stream_id_recv] = ((fps_values[stream_id_recv]*(fps_avg_counters[stream_id_recv]-1))+fps) / fps_avg_counters[stream_id_recv];
                        temp_start_ms_vector[stream_id_recv] = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
            
                }
                recv_frame_count_vector[stream_id_recv]++;
        } 
        for(auto& fmap : out_data){
                delete [] fmap;
                fmap = NULL;
        }   
}

void manual_model_bench(int num_models){

        int model_unique_stream_start = 0;
        start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        fps_values.reserve(num_models*num_streams);
        fps_avg_counters.reserve(num_models*num_streams);
        ifmap_vector.reserve(num_models*num_streams);
        ofmap_vector.reserve(num_models*num_streams);
        sent_frame_count_vector.reserve(num_models*num_streams);
        recv_frame_count_vector.reserve(num_models*num_streams);
        temp_start_ms_vector.reserve(num_models*num_streams);
        model_info_vector.reserve(num_models*num_streams);
        // accl->start(true);
        stream_recv_threads = new std::thread *[num_models*num_streams];
        stream_send_threads = new std::thread *[num_models*num_streams];
        for(int model_index = 0; model_index < num_models ; model_index++){
                MX::Types::MxModelInfo minfo = accl_mt->get_model_info(model_index);
                if(verbose)
                print_model_info(minfo);
                
                for(int stream_id = 0; stream_id < num_streams ; stream_id++){

                        temp_start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
                        
                        model_info_vector.push_back(minfo);
                        
                        std::vector<float*> in_data;
                        generate_input_data(minfo, in_data);
                        ifmap_vector.push_back(in_data);

                        sent_frame_count_vector.push_back(0);
                        recv_frame_count_vector.push_back(0);
                        temp_start_ms_vector.push_back(temp_start_ms);
                        fps_values.push_back(0.0);
                        fps_avg_counters.push_back(0);

                       

                        stream_send_threads[model_unique_stream_start+stream_id] = new std::thread(send_data, model_unique_stream_start+stream_id);
                        stream_recv_threads[model_unique_stream_start+stream_id] = new std::thread(receive_data, model_index,model_unique_stream_start+stream_id );

                        if(verbose){
                                std::cout<<"Created Thread "<< model_unique_stream_start+stream_id <<" for model "<< model_index << "\n\n"; 
                                std::cout << "\033[3;33m*************************************************\033[m\n";   
                        }   
                }
                model_unique_stream_start+=num_streams;
        }
        std::cout << "Number of chips used per device   = " << dfp_num_chips << "\n\n";
        if(verbose){
                std::cout<<"\n\n\n"<<std::flush;
                
                std::cout << "*************************************************\033[m\n\n";
        }

        while(runflag.load()){
                display_fps();
                std::cout<<std::flush;
                std::cout << "\033["<<(num_models*num_streams)+2<<"A";
                std::this_thread::sleep_for(std::chrono::seconds(1));
                for(int i = 0; i < (num_models*num_streams); i++){
                        if(recv_frame_count_vector[i] >= frame_count){
                                recv_all_count++;
                        }
                        else{
                                break;
                        }
                }
                if(recv_all_count == (num_models*num_streams)){
                        runflag.store(false);
                }
        }

        display_fps();
        std::cout<<"\n";
        std::chrono::milliseconds duration =
                        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()) - start_ms;
        float fps_total = std::accumulate(fps_values.begin(), fps_values.end(), 0.0);
        float fps_per_stream = fps_total / (num_models*num_streams);
        std::cout << "\rAverage FPS per stream : "<< fps_per_stream << "\033[m\n";
        std::cout << "\rAverage FPS for DFP    : "<< fps_total << "\033[m\n";
        if(verbose){
                std::cout << "\n\n*************************************************\033[m\n";
                std::cout<<"\n\n";
        }
        cleanup();
}

int main(int argc, char **argv)
{

        if (argc == 1){
                print_usage(argc, argv);
                exit(EXIT_FAILURE);
        }
        for (;;){
                int idx;
                int c;

                c = getopt_long(argc, argv,
                                short_options, long_options, &idx);

                if (-1 == c || hello_flag.load())
                        break;

                switch (c){
                        case 0: /* getopt_long() flag */
                                break;
                        case 'H':
                                hello_flag.store(true);
                                if(argc > 2){
                                        std::cout<<"Given arguments along with hello option \n";
                                        std::cout<<"Parameters along with -H or --hello are ignored \n\n";       
                                }
                                break;

                        case 'd':
                                dfp_path = optarg;
                                break;

                        case 'm':
                                multi_stream_bench = true;
                                num_streams = 2;
                                break;
                        
                        case 'v':
                                verbose = true;
                                break;

                        case 'b':
                                bench_tool = true;
                                break;

                        case 'h':
                                print_usage(argc, argv);
                                exit(EXIT_SUCCESS);
                                break;

                        case 'n':
                                errno = 0;
                                num_streams = strtol(optarg, NULL, 0);
                                if (errno)
                                        _error_exit(optarg);
                                break;
                        
                        case 'c':
                                errno = 0;
                                num_fmap_convert_threads = strtol(optarg, NULL, 0);
                                if (errno)
                                        _error_exit(optarg);
                                break;

                        case 'g':
                                errno = 0;
                                grp_id = strtol(optarg, NULL, 0);
                                if (errno)
                                        _error_exit(optarg);
                                break;

                        case 'f':
                                errno = 0;
                                frame_count = strtol(optarg, NULL, 0);
                                if (errno)
                                        _error_exit(optarg);
                                break;

                        case MAX_FPS_OPT:
                                errno =0;
                                max_fps = strtol(optarg, NULL, 0);
                                if (errno)
                                        _error_exit(optarg);
                                break;

                        case IW_OPT:
                                errno =0;
                                num_input_workers = strtol(optarg, NULL, 0);
                                if (errno)
                                        _error_exit(optarg);
                                break;

                        case OW_OPT:
                                errno =0;
                                num_output_workers = strtol(optarg, NULL, 0);
                                if (errno)
                                        _error_exit(optarg);
                                break;
                        case MD_IDS:
                                errno = 0;
                                if(optarg){
                                        parse_device_ids(optarg);
                                }
                                if(device_ids.size() > 1)
                                        multi_device_bench = true;
                                else
                                        grp_id = device_ids[0];
                                if (errno)
                                        _error_exit(optarg);
                                break;

                        case MT_MODE:
                                frame_count++;
                                manual_threading = true;
                                break;              

                        default:
                                print_usage(argc, argv);
                                exit(EXIT_FAILURE);
                }
         }

        signal(SIGINT, signal_handler);


        if(hello_flag){
                runflag.store(false);
                std::cout<<"Hello from MXA! \n";
                MX::Runtime::MxAccl hello_accl;
        }
        else{
                std::cout << "\033[3;34m*************************************************\n";
                std::cout << "*      Evaluate dfp performance using MX3       *\n";
                std::cout << "*************************************************\033[m\n";

                runflag.store(true);
                if(max_fps!=0){
                        custom_duration = 1000*50/max_fps;
                        custom_duration_ms = std::chrono::milliseconds(int(custom_duration));
                }
                if (dfp_path == NULL){
                        std::cout<< "please specify the dfp file\n";
                        print_usage(argc, argv);
                        exit(EXIT_FAILURE);
                }
                if(manual_threading){
                        if(multi_device_bench){
                                accl_mt = new MX::Runtime::MxAcclMT;
                                accl_mt->connect_dfp(dfp_path, device_ids);
                        }
                        else {
                                accl_mt = new MX::Runtime::MxAcclMT;
                                accl_mt->connect_dfp(dfp_path, grp_id);
                        }
                        num_models = accl_mt->get_num_models();
                        for(int i=0; i < num_models; i++){
                                accl_mt->set_parallel_fmap_convert(num_fmap_convert_threads, i);
                        }
                        
                        dfp_num_chips = accl_mt->get_dfp_num_chips();
                        if(verbose)
                                print_bench_setting_info();
                        manual_model_bench(num_models);

                }
                else{
                        if(multi_device_bench){
                                accl = new MX::Runtime::MxAccl;
                                accl->connect_dfp(dfp_path, device_ids);

                        }
                        else{
                                accl = new MX::Runtime::MxAccl;
                                accl->connect_dfp(dfp_path, grp_id);
                        }

                        accl->set_num_workers(num_input_workers,num_output_workers);
                        num_models = accl->get_num_models();
                        for(int i=0; i < num_models; i++){
                                accl->set_parallel_fmap_convert(num_fmap_convert_threads, i);
                        }
                        dfp_num_chips = accl->get_dfp_num_chips();
                        if(verbose)
                                print_bench_setting_info();
                        model_bench(num_models);
                }

                if(accl!=NULL){
                        delete accl;
                        accl = NULL;
                }
                else{
                        accl = NULL;
                }
                if(accl_mt!=NULL){
                        delete accl_mt;
                        accl_mt = NULL;
                }
                else{
                        accl_mt = NULL;
                }
                
                if(!bench_tool){
                        std::cout<<"\n\n\033[3;33m Bench for "<< num_models << " Model(s) Done \n";
                }
        }        
        //         std::cout << "\033[3;33m*************************************************\n\n\033[m";
        //         std::cout<<"Wait, we are not done yet \n ";
        //         std::cout<<"Calculating latency...\n";
        // }
        return 0;
}
