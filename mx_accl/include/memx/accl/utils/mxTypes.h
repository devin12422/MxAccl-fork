#ifndef MXTYPES_H
#define MXTYPES_H
#include <vector>
#include <stdint.h>
#include <stdexcept>
#include <iostream>
namespace MX
{
    namespace Types
    {

        class ShapeVector {
            private:
               std::vector<int64_t> shape;
               int64_t h=0, w=0, z=0, c=0;
               int size_ = 4;
            public:   
                /**
                * @brief Construct a new ShapeVector type object
                */
                ShapeVector();// Initialize shape with 4 elements, all initialized to 0
                /**
                * @brief Construct a new ShapeVector type object
                * @param h Height 
                * @param w Width 
                * @param z Batch
                * @param c Channel
                */
                ShapeVector(int64_t h, int64_t w, int64_t z, int64_t c) ;
                /**
                * @brief Construct a new ShapeVector type object
                */
                ShapeVector(int size);// Initialize shape with size #elements, all initialized to 1
                // // Overload [] operator const
                // const int64_t& operator[](int64_t index) const ;
                // Overload [] operator 
                int64_t& operator[](int64_t index) ;
                
                /**
                * @brief returns a vector of shape with channel first format
                */
                std::vector<int64_t> chfirst_shape();
                
                /**
                * @brief returns a vector of shape with channel last format
                */
                std::vector<int64_t> chlast_shape();

                /**
                * @brief returns a data pointer of the sape vector
                */
                int64_t* data();

                /**
                * @brief returns size of the shape vector
                */
                int64_t size() const ;           

                /**
                 * @brief sets the shape to channel first format
                 */
                void set_ch_first();    
        };

        //struct with necessary model information collated for internal and external purposes
        
        /** @struct MxModelInfo 
            @brief struct with necessary information of a model
            @var MxModelInfo::model_index
            index of a model to identify
            @var MxModelInfo::num_in_featuremaps
            Number of input featuremaps
            @var MxModelInfo::num_out_featuremaps
            Number of output featuremaps
            @var MxModelInfo::input_layer_names
            Vector of strings containing input layer names
            @var MxModelInfo::output_layer_names
            Vector of strings containing output layer names
            @var MxModelInfo::in_featuremap_shapes
            Vector of Shapevector containing input featuremap shapes
            @var MxModelInfo::out_featuremap_shapes
            Vector of Shapevector containing output featuremap shapes            
            @var MxModelInfo::in_featuremap_sizes
            Vector of size_t containing sizes fo the input featuremaps      
            @var MxModelInfo::out_featuremap_sizes
            Vector of size_t containing sizes fo the output featuremaps                          
        */
        struct MxModelInfo{
            int model_index;
            int num_in_featuremaps;
            int num_out_featuremaps;
            std::vector<std::string> input_layer_names;
            std::vector<std::string> output_layer_names; 
            std::vector<MX::Types::ShapeVector> in_featuremap_shapes;
            std::vector<MX::Types::ShapeVector> out_featuremap_shapes;
            std::vector<size_t> in_featuremap_sizes;
            std::vector<size_t> out_featuremap_sizes;
        };

    } // Namespace Types
} // Namespace MX


#endif