#ifndef PREPOST_H
#define PREPOST_H

#include <vector>
#include <string>
#include <stdint.h>
#include <memx/accl/utils/general.h>
#include <memx/accl/utils/featureMap.h>
#include <memx/accl/utils/path.h>

enum PluginType{
    Plugin_Onnx,
    Plugin_Tflite,
    Plugin_Tf,
    Plugin_None
};

enum ProcessType{
    Process_Pre,
    Process_Post  
};
class PrePost{
    public:
        virtual void runinference(std::vector<MX::Types::FeatureMap<float>*> input, std::vector<MX::Types::FeatureMap<float>*> output) = 0;
        virtual void runinference(std::vector<MX::Types::FeatureMap<uint8_t>*>, std::vector<MX::Types::FeatureMap<uint8_t>*>){
            throw std::runtime_error("uint_8 format is not supported for model inputs");
        };
        virtual std::vector<std::vector<int64_t>> get_output_shapes() = 0;
        virtual std::vector<std::vector<int64_t>> get_input_shapes() = 0;
        virtual std::vector<size_t> get_output_sizes() = 0;
        virtual std::vector<size_t> get_input_sizes() = 0;
        virtual std::vector<std::string> get_input_names() = 0;
        virtual std::vector<std::string> get_output_names() = 0;
        virtual ~PrePost(){};
        PluginType type = Plugin_None;
        bool dynamic_output = false;
        std::vector<int> dfp_pattern;
        std::vector<int> real_featuremaps;
        void match_names(std::vector<std::string> dfp_names, ProcessType prc_type);
        PrePost();
    private:
};

typedef PrePost* (*CreatePlugin)(const char* model_path, const std::vector<size_t>& out_sizes);

PrePost* mx_create_prepost(std::filesystem::path model_path, const std::vector<size_t>& out_sizes = {});

std::vector<std::string> prepost_split(const std::string& str);
PrePost* createObject(std::string soFileName, std::string functionName, std::string model_path, const std::vector<size_t>& out_sizes, PluginType type);

#endif
