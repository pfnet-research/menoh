#include <iostream>

#include <menoh/array.hpp>
#include <menoh/graph.hpp>
#include <menoh/utility.hpp>

#include <NvInfer.h>
using namespace nvinfer1;

#include <menoh/tensorrt/Exception.hpp>
#include <menoh/tensorrt/Parser.hpp>

namespace menoh_impl {
    namespace tensorrt_backend {

        std::string NodeName( const menoh_impl::node& node ) {
            std::string name;
            for(auto it = node.input_name_list.begin(); it != node.input_name_list.end(); ++it) {
                name += *it;
            }          
            for(auto it = node.output_name_list.begin(); it != node.output_name_list.end(); ++it) {
                name += *it;
            }          
            return name;
        }

        std::string PrefixNodeName( const menoh_impl::node& node ) {
            return std::string(node.op_type + ":" + NodeName(node));
        }

        std::string TensorName( const std::string name ) {
            return std::string("tensor:" + name);
        }

        class SingleLayerOperation : public Operation{
        public:
            SingleLayerOperation(Parser* parser, const menoh_impl::node& node, ILayer* layer)
            : Operation(parser, node)
            , m_Layer(layer)
            {
            }

            ITensor* Output(unsigned int index) override
            {
                assert(m_Layer);

                if ((int)index >= m_Layer->getNbOutputs())
                {
                    std::string msg("The requested output slot ");
                    msg += index;
                    msg += "for ";
                    msg += m_Layer->getName();
                    msg += " does not exist Indext";
                    throw ParseException(msg);
                }

                return m_Layer->getOutput(index);
            }

        protected:
            ILayer* m_Layer;
        };

        ITensor* Parser::GetTensor( OutputOfOperation& input ){
            return input.m_Value->Output(input.m_Index);
        }

        void Parser::SetLayer(ILayer* layer, const menoh_impl::node& node) {
            layer->setName(PrefixNodeName(node).c_str());
            layer->getOutput(0)->setName(TensorName(NodeName(node)).c_str());
            m_Layer = layer;
        }

        std::vector<OutputOfConstNodeDef>
        Parser::InputNodes(const menoh_impl::node& node) const {
	    std::vector<OutputOfConstNodeDef> ret;

            if (node.op_type == "Const" || node.op_type == "Placeholder")
            {
                return ret;
            }

            ret.reserve(static_cast<size_t>(node.input_name_list.size()));
            
            for( unsigned int j=0; j<node.input_name_list.size(); ++j )
            {
                bool found = false; 
                auto input = node.input_name_list.at(j);
                for( auto const& n : m_Nodes )
                {
                    auto my_node = n.second;
                    for( unsigned int i=0 ; i<my_node->output_name_list.size() ; i++ )
                    {
                        if (input == my_node->output_name_list.at(i))
                        {
                            ret.push_back(OutputOfConstNodeDef(my_node,i));
                            found = true;
                            break;
                        } 
                    }
                }

                auto it = m_Params.find(input);
                if (it != m_Params.end())
                {
                    found = true;
                }

                if( !found )
                {
                    throw ParseException("Can't find node '" + node.input_name_list.at(j) +
                                         "', which is listed as an input of '" + node.op_type + "'");
                }

            }

            return ret;
        }

        std::vector<OutputOfOperation>
        Parser::InputCheck(const menoh_impl::node& node, std::size_t expectedNumInputs){
            std::string name = NodeName(node);

            const std::size_t numInputs = node.input_name_list.size();
            if (numInputs != expectedNumInputs)
            {
                std::string msg("Unexpected number of inputs for node ");
                msg += name;
                msg += ". Expected ";
                msg += std::to_string(expectedNumInputs);
                msg += ", found ";
                msg += std::to_string(numInputs);
                throw ParseException(msg);
				
            }

            std::vector<OutputOfOperation> result;
            std::vector<OutputOfConstNodeDef> nodes = InputNodes(node);
            for (auto&& node : nodes)
            {
                auto it = m_Operations.find(NodeName(*(node.m_Value)));
                if (it == m_Operations.end())
                {
                    throw ParseException("Node with name '" + NodeName(*(node.m_Value)) + "' has not been parsed");
                }
                Operation* parsedOp = it->second.get();
                parsedOp = parsedOp->IdentityOperations();
                result.push_back(OutputOfOperation(parsedOp,node.m_Index));
            }

#ifdef TENSORRT_DEBUG
            std::cout << std::endl << " [node] : " << node.op_type << " , " << name << std::endl;
            for( unsigned int j=0; j<node.input_name_list.size(); ++j )
            std::cout << "    input : " << node.input_name_list.at(j) << std::endl;

            for( unsigned int j=0; j<node.output_name_list.size(); ++j )
            std::cout << "   output : " << node.output_name_list.at(j) << std::endl;
#endif
            return result;
        }  

        template <typename T>
        class ConstOperation : public SingleLayerOperation {
        public:
            ConstOperation(Parser* parser, const menoh_impl::node& node,
                           const T* data, const DataType dataType, const Dims& dims, int64_t numElements)
              : SingleLayerOperation(parser, node, nullptr)
              , dimentions(dims)
              , weights{dataType, data, numElements}
            {
                std::string name = NodeName(node);

                assert(m_Layer == nullptr);
                IConstantLayer* layer;
                {
                    layer = m_Parser->Network()->addConstant(dimentions, weights);
                    assert(layer);
                    layer->setName(name.c_str());
                    layer->getOutput(0)->setName(TensorName(name).c_str());
                }
                m_Layer = layer;
            }

            Dims& getDims(){
                return dimentions;
            }            

            Weights& getWeights() { 
                return weights;
            }

        private:
            Dims dimentions;
            Weights weights;
        };      

        template<typename Type>
        bool Parser::HasParsedConst(const std::string & nodeName) const {
            auto it = m_Operations.find(nodeName);
            return (it != m_Operations.end() &&
                    dynamic_cast<ConstOperation<Type>*>(it->second.get()) != nullptr);
        }

        bool Parser::HasParsedConst(OutputOfOperation& input) {
            return HasParsedConst<float>(NodeName(input.m_Value->GetNode()));
        }

        OperationPtr Parser::ParseConst(const menoh_impl::node& node) {
            std::string name = NodeName(node);
            
            auto it = m_Params.find(name);
            if (it == m_Params.end() )
            {
                throw ParseException("ParseConst : not found " + name);
            }

#ifdef TENSORRT_DEBUG
            std::cerr << "[node] : Const, " << name << std::endl;
#endif
            auto arr = m_Params[name];
            std::vector<unsigned int> sizes(arr.dims().data(), arr.dims().data()+arr.dims().size());

            Dims dims;
            int64_t numElements = 1U;
            {
                dims.nbDims = arr.dims().size();
                for(int i=0 ; i<dims.nbDims ; i++ )
                {
                    dims.d[i]    = arr.dims().data()[i];
                    dims.type[i] = DimensionType::kSEQUENCE;
                    numElements *= dims.d[i];
                }
            }  

            const DataType dataType = DataType::kFLOAT; // dtype_t::float_

            return std::make_unique<ConstOperation<float>>(this, node,
                                                           reinterpret_cast<const float*>((const int8_t*)arr.data()),
                                                           dataType, dims, numElements);
        }

        class ParsedIdentityOperation : public Operation {
        public:
            ParsedIdentityOperation(Parser* parser, const menoh_impl::node& node, Operation* op)
                : Operation(parser, node)
                , m_Op(op)
            {
            }

            virtual ITensor* Output(unsigned int index) override
            {
                assert(m_Op);
                return m_Op->Output(index);
            }

            virtual Operation* IdentityOperations() override
            {
                return m_Op->IdentityOperations();
            }

        private:
            Operation* m_Op;
        };

        OperationPtr Parser::ParseIdentity(const menoh_impl::node& node) {
	    std::vector<OutputOfOperation> inputs = InputCheck(node, 1);
            return std::make_unique<ParsedIdentityOperation>(this, node, inputs[0].m_Value);
        }

        OperationPtr Parser::ParseBatchNormalization(const menoh_impl::node& node) {
            std::string name = NodeName(node);

            std::vector<OutputOfConstNodeDef> nodes = InputNodes(node);
            unsigned int numInputs = static_cast<unsigned int>(nodes.size());
            std::vector<OutputOfOperation> inputs = InputCheck(node, numInputs);

            if (!HasParsedConst(inputs[1]) || !HasParsedConst(inputs[2])
             || !HasParsedConst(inputs[3]) || !HasParsedConst(inputs[4]))
            {
                throw ParseException("only supports BatchNormalization layers with constant weights");
            }

            ITensor* input0 = GetTensor(inputs[0]);

            ConstOperation<float>* scaleNode = static_cast<ConstOperation<float>*>(inputs[1].m_Value);
            Weights& w_scale = scaleNode->getWeights(); 

            ConstOperation<float>* biasNode = static_cast<ConstOperation<float>*>(inputs[2].m_Value);
            Weights& w_bias = biasNode->getWeights(); 

            ConstOperation<float>* meanNode = static_cast<ConstOperation<float>*>(inputs[3].m_Value);
            Weights& w_mean = meanNode->getWeights();

            ConstOperation<float>* varianceNode = static_cast<ConstOperation<float>*>(inputs[4].m_Value);
            Weights& w_variance = varianceNode->getWeights();

            auto epsilon = optional_attribute_float(node, "epsilon", 1e-5f);

            size_t nweight = input0->getDimensions().d[0];
            for( size_t i=0; i<nweight; ++i ) {
                float scale    = (static_cast<float const*>(w_scale.values)[i]);
                float bias     = (static_cast<float const*>(w_bias.values)[i]);
                float mean     = (static_cast<float const*>(w_mean.values)[i]);
                float variance = (static_cast<float const*>(w_variance.values)[i]);

                float& combined_scale_ref = const_cast<float*>(
                    static_cast<float const*>(w_scale.values))[i];
                float& combined_bias_ref  = const_cast<float*>(
                    static_cast<float const*>(w_bias.values))[i];

                combined_scale_ref = scale / sqrtf(variance + epsilon);
                combined_bias_ref  = bias - mean * combined_scale_ref;
            }

            IScaleLayer* scale;
            {
                scale = Network()->addScale(*input0, ScaleMode::kCHANNEL, w_bias, w_scale, {});
                assert(scale);
                SetLayer(scale, node);
            }

            return std::make_unique<SingleLayerOperation>(this, node, scale);
        }

        OperationPtr Parser::ParseFC(const menoh_impl::node& node) {
            std::string name = NodeName(node);

            std::vector<OutputOfOperation> inputs = InputCheck(node, 3);

            ConstOperation<float>* weightNode = static_cast<ConstOperation<float>*>(inputs[1].m_Value);
            Weights& weight = weightNode->getWeights(); 

            ConstOperation<float>* biasNode = static_cast<ConstOperation<float>*>(inputs[2].m_Value);
            Weights& bias = biasNode->getWeights(); 

            if (weightNode->getDims().d[0] != biasNode->getDims().d[0])
            {
                throw ParseException("shape of weight and bias do not match");
            }

            IFullyConnectedLayer* full;
            {
                full = Network()->addFullyConnected(*GetTensor(inputs[0]), bias.count, weight, bias);
                assert(full);
                SetLayer(full, node);
            }

            return std::make_unique<SingleLayerOperation>(this, node, full);
        }

        OperationPtr Parser::ParseConv2D(const menoh_impl::node& node) {
            std::string name = NodeName(node);

            std::vector<OutputOfConstNodeDef> nodes = InputNodes(node);
            unsigned int numInputs = static_cast<unsigned int>(nodes.size());
            std::vector<OutputOfOperation> inputs = InputCheck(node, numInputs);

            if (!HasParsedConst(inputs[1]) || (numInputs == 3 && !HasParsedConst(inputs[2])))
            {
                throw ParseException("only supports Convolution layers with constant weights and biases");
            }

            ITensor* input0 = GetTensor(inputs[0]);

            std::vector<int> strides, kernel_shape, pads;
            std::tie(strides, kernel_shape, pads) = attributes_for_2d_data_processing(node);

            DimsHW begin_pad{pads[0], pads[1]}, end_pad{(pads.size()<=2)? pads[0] : pads[2],
                                                        (pads.size()<=2)? pads[1] : pads[3]};
            if( (begin_pad.h() != end_pad.h()) || (begin_pad.w() != end_pad.w()) )
            {
                auto layer = Network()->addPadding(*input0, begin_pad, end_pad );
                input0 = layer->getOutput(0);
            }

            ConstOperation<float>* weightNode = static_cast<ConstOperation<float>*>(inputs[1].m_Value);
            Weights& weight = weightNode->getWeights();
            
            Weights bias{weight.type, nullptr, 0};
            if(numInputs == 3){
                ConstOperation<float>* biasNode = static_cast<ConstOperation<float>*>(inputs[2].m_Value);
                Weights& w_bias = biasNode->getWeights();
                bias.type   = w_bias.type;
                bias.values = w_bias.values;
                bias.count  = w_bias.count;
            }

#ifdef TENSORRT_DEBUG
            std::cout << "           strides      = " << strides[0]      << ", " << strides[1]      << std::endl;
            std::cout << "           kernel_shape = " << kernel_shape[0] << ", " << kernel_shape[1] << std::endl;
            std::cout << "           pads         = " << pads[0]         << ", " << pads[1];
            if( pads.size() >= 4 )
                std::cout << ",  = " << pads[2]         << ", " << pads[3] << std::endl ;
            else
                std::cout << std::endl;
#endif
            int output = weightNode->getDims().d[0];

            IConvolutionLayer* conv;
            {
                conv = Network()->addConvolution(*input0, 
                                                  output, DimsHW{kernel_shape[0], kernel_shape[1]}, weight, bias);
                assert(conv);
                conv->setStride(DimsHW{strides[0], strides[1]});
                if( (begin_pad.h() == end_pad.h()) || (begin_pad.w() == end_pad.w()) )
                {
                    conv->setPadding(begin_pad);
                }    
                SetLayer(conv, node);   
            }
            return std::make_unique<SingleLayerOperation>(this, node, conv);
        }  
 
        OperationPtr Parser::ParseConcat(const menoh_impl::node& node) {
            std::string name = NodeName(node);
	    
            std::vector<OutputOfConstNodeDef> nodes = InputNodes(node);
            unsigned int numInputs = static_cast<unsigned int>(nodes.size());
            std::vector<OutputOfOperation> inputs = InputCheck(node, numInputs);

            ITensor* input0 = GetTensor(inputs[0]);

            auto axis = get<int>(node.attribute_table.at("axis"));
            axis += (axis<0) ? input0->getDimensions().nbDims : (-1);

            if( axis == 0 )
            {
                std::vector<ITensor*> itensors;
                for(unsigned int i=0 ; i<numInputs ; i++ )
                {
                    itensors.push_back(inputs[i].m_Value->Output(inputs[i].m_Index));
                }

                IConcatenationLayer* concat;
                {
                    concat = Network()->addConcatenation(itensors.data(), itensors.size());
                    assert(concat);
                    SetLayer(concat, node);
                }
                return std::make_unique<SingleLayerOperation>(this, node, concat);
            }
            else
            {
                throw ParseException("only supports Concat layers with legal axis");
            }
        }

        OperationPtr Parser::ParseLrn(const menoh_impl::node& node) {
            std::string name = NodeName(node);

            std::vector<OutputOfOperation> inputs = InputCheck(node, 1);

            float alpha = attribute_float(node, "alpha");
            float beta  = attribute_float(node, "beta");
            float k     = attribute_float(node, "bias");
            int window  = attribute_int(node, "depth_radius");

            window = window * 2 + 1;

            ILRNLayer* lrn;
            {
                lrn = Network()->addLRN(*GetTensor(inputs[0]), window, alpha, beta, k);
                assert(lrn);
                SetLayer(lrn, node);
            }

            return std::make_unique<SingleLayerOperation>(this, node, lrn);
        }

        OperationPtr Parser::ParseSum(const menoh_impl::node& node) {
            return ParseElementWise(node, ElementWiseOperation::kSUM);
        }  

        OperationPtr Parser::ParseMul(const menoh_impl::node& node) {
            return ParseElementWise(node, ElementWiseOperation::kPROD);
        }  

        OperationPtr Parser::ParseAdd(const menoh_impl::node& node) {
            return ParseElementWise(node, ElementWiseOperation::kSUM);
        }  

        OperationPtr Parser::ParseElementWise(const menoh_impl::node& node, ElementWiseOperation op) {
            std::string name = NodeName(node);
	    
            std::vector<OutputOfOperation> inputs = InputCheck(node, 2);

            IElementWiseLayer* layer;
            {
                layer = Network()->addElementWise(*GetTensor(inputs[0]), *GetTensor(inputs[1]), op);
                assert(layer);
                SetLayer(layer, node);
            }
            return std::make_unique<SingleLayerOperation>(this, node, layer);
        }            

        OperationPtr Parser::ParsePlaceholder(const menoh_impl::node& node) {
            std::string name = NodeName(node);
	    
            std::vector<OutputOfOperation> inputs = InputCheck(node, 0);

            auto dims = (std::vector<uint32_t> const&)attribute_ints(node, "dims");

            ITensor* placeholder;
            {
                Dims inputDims;

                unsigned int size = dims.size();
                if( size > 3 ) {
                    int offset = size - 3;
                    inputDims.nbDims = size - offset ;
                    for( unsigned int i=offset ; i<size ; i++ )
                        inputDims.d[i-offset] = dims[i];  
                } else {
#if 0
                    inputDims.nbDims = 3;
                    for( unsigned int i=0 ; i<3-size ; i++ )
                        inputDims.d[i] = 1;
                    for( unsigned int i=3-size ; i<3 ; i++ )
                        inputDims.d[i] = dims[i-(3-size)];
#else
                    inputDims.nbDims = size;
                    for( unsigned int i=0 ; i<size ; i++ )
                        inputDims.d[i] = dims[i];
#endif
                }

#ifdef TENSORRT_DEBUG
                std::cout << "           dims.size() = " << dims.size();
                for( unsigned int i=0 ; i<dims.size() ; i++ )
                    std::cout << std::endl << "           dims[" << i << "] = " << dims[i];
                std::cout << std::endl; 
                std::cout << "           inputDims.nbDims = " << inputDims.nbDims;
                for( int i=0 ; i<inputDims.nbDims ; i++ )
                    std::cout << std::endl << "           inputDims.d[" << i << "] = " << inputDims.d[i];
                std::cout << std::endl; 
#endif
                placeholder = Network()->addInput(PrefixNodeName(node).c_str(), nvinfer1::DataType::kFLOAT, inputDims);
                assert(placeholder);
            }

#if 1
            IScaleLayer* layer;
            {
                layer = Network()->addScale(*placeholder, ScaleMode::kUNIFORM, {}, {}, {});
                assert(layer);
                SetLayer(layer, node);
            }
#else
            IIdentityLayer* layer;
            {
                layer = Network()->addIdentity(*placeholder);
                assert(layer);
                SetLayer(layer, node);
            }
#endif
            return std::make_unique<SingleLayerOperation>(this, node, layer);
        }

        OperationPtr Parser::ParseRelu(const menoh_impl::node& node) {
            return AddActivationLayer(node, ActivationType::kRELU);
        }

        OperationPtr Parser::ParseSigmoid(const menoh_impl::node& node) {
            return AddActivationLayer(node, ActivationType::kSIGMOID);
        }

        OperationPtr Parser::ParseTanh(const menoh_impl::node& node) {
            return AddActivationLayer(node, ActivationType::kTANH);
        }

        OperationPtr Parser::AddActivationLayer(const menoh_impl::node& node, ActivationType activationType) {
            std::string name = NodeName(node);

            std::vector<OutputOfOperation> inputs = InputCheck(node, 1);

            IActivationLayer* activate;
            {
                activate = Network()->addActivation(*GetTensor(inputs[0]), activationType);
                assert(activate);
                SetLayer(activate, node);
            }
            return std::make_unique<SingleLayerOperation>(this, node, activate);
        }

        OperationPtr Parser::ParseSoftmax(const menoh_impl::node& node) {
            std::string name = NodeName(node);

            std::vector<OutputOfOperation> inputs = InputCheck(node, 1);

            bool found = true;
            int  axis = 0;
            try {
                axis = get<int>(node.attribute_table.at("axis"));
            }
            catch (const std::out_of_range &e){
                std::cerr << "Exception at " << e.what() << std::endl; 
                found = false;
            }  

            ISoftMaxLayer* softmax;
            {
                softmax = Network()->addSoftMax(*GetTensor(inputs[0]));
                assert(softmax);
#ifdef TENSORRT_DEBUG
            std::cout << "           softmax.getAxes() = " << softmax->getAxes() << std::endl;
#endif            
                if(found && (axis!=(int)softmax->getAxes())) {
                    softmax->setAxes(axis);                
                } 
                SetLayer(softmax, node);
            }
                
#ifdef TENSORRT_DEBUG
            std::cout << "           axis = " << axis << std::endl;
            std::cout << "           softmax.getAxes() = " << softmax->getAxes() << std::endl;
            std::cout << "           output.name = " << softmax->getOutput(0)->getName() << std::endl;
#endif            
            return std::make_unique<SingleLayerOperation>(this, node, softmax);
        }

        OperationPtr Parser::ParseMaxPool(const menoh_impl::node& node) {
            std::string name = NodeName(node);

            std::vector<OutputOfOperation> inputs = InputCheck(node, 1);
            if (inputs.size() != 1)
            {
                throw ParseException("MaxPooling expects one input!");
            }
            ITensor* input0 = GetTensor(inputs[0]);

            std::vector<int> strides, kernel_shape, pads;
            std::tie(strides, kernel_shape, pads) = attributes_for_2d_data_processing(node);
            DimsHW begin_pad{pads[0], pads[1]}, end_pad{(pads.size()<=2)? pads[0] : pads[2],
                                                        (pads.size()<=2)? pads[1] : pads[3]};
            
#ifdef TENSORRT_DEBUG
            std::cout << "           strides      = " << strides[0]      << ", " << strides[1]      << std::endl;
            std::cout << "           kernel_shape = " << kernel_shape[0] << ", " << kernel_shape[1] << std::endl;
            std::cout << "           pads         = " << pads[0]         << ", " << pads[1];
            if( pads.size() >= 4 )
                std::cout << ",  = " << pads[2]         << ", " << pads[3] << std::endl ;
            else
                std::cout << std::endl;
#endif
            if( (begin_pad.h() != end_pad.h()) || (begin_pad.w() != end_pad.w()) )
            {
                auto layer = Network()->addPadding(*input0, begin_pad, end_pad );
                input0 = layer->getOutput(0);
            }
            
            IPoolingLayer* pool;
            {
                pool = Network()->addPooling(*input0, PoolingType::kMAX, DimsHW{kernel_shape[0], kernel_shape[1]});
                assert(pool);
                pool->setStride(DimsHW{strides[0], strides[1]});
                if( (begin_pad.h() == end_pad.h()) && (begin_pad.w() == end_pad.w()) )
                {
                    pool->setPadding(begin_pad);
                }    
                SetLayer(pool, node);
            } 
            return std::make_unique<SingleLayerOperation>(this, node, pool);
        }          

        OperationPtr Parser::ParseAvgPool(const menoh_impl::node& node) {
            std::string name = NodeName(node);

            std::vector<OutputOfOperation> inputs = InputCheck(node, 1);
            ITensor* input0 = GetTensor(inputs[0]);

            std::vector<int> strides, kernel_shape, pads;
            std::tie(strides, kernel_shape, pads) = attributes_for_2d_data_processing(node);
            DimsHW begin_pad{pads[0], pads[1]}, end_pad{(pads.size()<=2)? pads[0] : pads[2],
                                                        (pads.size()<=2)? pads[1] : pads[3]};
            
#ifdef TENSORRT_DEBUG
            std::cout << "           strides      = " << strides[0]      << ", " << strides[1]      << std::endl;
            std::cout << "           kernel_shape = " << kernel_shape[0] << ", " << kernel_shape[1] << std::endl;
            std::cout << "           pads         = " << pads[0]         << ", " << pads[1];
            if( pads.size() >= 4 )
                std::cout << ", " << pads[2]         << ", " << pads[3] << std::endl ;
            else
                std::cout << std::endl;
#endif
            IPoolingLayer* pool;
            {
                pool = Network()->addPooling(*input0, PoolingType::kAVERAGE, DimsHW{kernel_shape[0], kernel_shape[1]});
                assert(pool);
                pool->setStride(DimsHW{strides[0],strides[1]});
                SetLayer(pool, node);
                input0 = pool->getOutput(0);
            } 

            DimsHW pre_crop(0,0), post_crop(0,0);
            for( int i=0 ; i<2 ; i++ )
            {
                if( end_pad.d[i] == begin_pad.d[i] )
                {
                    // No action
                }
                else if( end_pad.d[i] == (begin_pad.d[i] + 1) )
                {
                    begin_pad.d[i] += strides[i];
                    pre_crop.d[i] = 1;
                }
                else
                {
                    throw ParseException("only supports AvgPool layers with legal pads");
                }
            }

            pool->setPadding(begin_pad);

            if( !(!pre_crop.d[0] && !pre_crop.d[1]) || !(!post_crop.d[0] && !post_crop.d[1]) )
            {
                auto layer = Network()->addPadding(*input0, DimsHW{ -pre_crop.d[0], -pre_crop.d[1]}, 
                                                            DimsHW{-post_crop.d[0],-post_crop.d[1]});
                assert(layer);
                SetLayer(layer, node);

                return std::make_unique<SingleLayerOperation>(this, node, layer);
            }
            return std::make_unique<SingleLayerOperation>(this, node, pool);
        }

        OperationPtr Parser::ParseGlobalPool(const menoh_impl::node& node, PoolingType type ) {
            std::string name = NodeName(node);
	    
            std::vector<OutputOfOperation> inputs = InputCheck(node, 1);
            ITensor* input0 = GetTensor(inputs[0]);

            IPoolingLayer* pool;
            {
                Dims dims = input0->getDimensions();
                if( dims.nbDims != 3 ){
                    std::string msg("GlobalAvg");
                    msg += std::string(type == PoolingType::kMAX ? "Max" : "Avg");
                    msg += std::string("Pool layser's input dimensions must be 3 (three).");
                    throw ParseException(msg);
                }
                DimsHW kernel_shape({dims.d[1], dims.d[2]});
                pool = Network()->addPooling(*input0, type, kernel_shape);
                assert(pool);
                SetLayer(pool, node);
            } 
            return std::make_unique<SingleLayerOperation>(this, node, pool);
        }
        
        OperationPtr Parser::ParseGlobalMaxPool(const menoh_impl::node& node) {
            return ParseGlobalPool(node, PoolingType::kMAX);
        }
        
        OperationPtr Parser::ParseGlobalAvgPool(const menoh_impl::node& node) {
            return ParseGlobalPool(node, PoolingType::kAVERAGE);
        }

        OperationPtr Parser::ParseGemm(const menoh_impl::node& node) {
            std::string name = NodeName(node);
            
            std::vector<OutputOfOperation> inputs = InputCheck(node, 3);

            ConstOperation<float>* weightNode = static_cast<ConstOperation<float>*>(inputs[1].m_Value);
            Weights& weight = weightNode->getWeights();

            ConstOperation<float>* biasNode = static_cast<ConstOperation<float>*>(inputs[2].m_Value);
            Weights& bias = biasNode->getWeights();

            if (weightNode->getDims().d[0] != biasNode->getDims().d[0])
            {
                throw ParseException("shape of weight and bias do not match");
            }

            auto alpha = optional_attribute_float(node, "alpha", 1.f);
            if(alpha != 1) {
                throw failed_to_configure_operator(
                  node.op_type, node.output_name_list.at(0),
                  "alpha of Gemm must be 1 but given: " +
                    std::to_string(alpha));
            }

            auto beta = optional_attribute_float(node, "beta", 1.f);
            if(beta != 1) {
                throw failed_to_configure_operator(
                  node.op_type, node.output_name_list.at(0),
                  "beta of Gemm must be 1 but given: " + std::to_string(alpha));
            }

            auto trans_a = optional_attribute_int(node, "transA", 0);
            if(trans_a) {
                throw failed_to_configure_operator(
                  node.op_type, node.output_name_list.at(0),
                  "transA of Gemm must be 0 but given: " +
                    std::to_string(alpha));
            }
            auto trans_b = optional_attribute_int(node, "transB", 0);
            if(!trans_b) {
                throw failed_to_configure_operator(
                  node.op_type, node.output_name_list.at(0),
                  "transB of Gemm must be 0 but given: " +
                    std::to_string(alpha));
            }
      
            IFullyConnectedLayer* full;
            {
                full = Network()->addFullyConnected(*GetTensor(inputs[0]), bias.count, weight, bias);
                assert(full);
                SetLayer(full, node);
            }

            return std::make_unique<SingleLayerOperation>(this, node, full);
        }

        OperationPtr Parser::ParseUnsqueeze(const menoh_impl::node& node) {
            std::cerr << "ParseUnsqueeze" << std::endl;
            std::string name = NodeName(node);
            std::cerr << "ParseUnsqueeze" << std::endl;            
            std::vector<OutputOfOperation> inputs = InputCheck(node, 1);
            std::cerr << "ParseUnsqueeze" << std::endl;
            ITensor* input0 = GetTensor(inputs[0]);
            std::cerr << "ParseUnsqueeze" << std::endl;
            auto axes = get<std::vector<int>>(node.attribute_table.at("axes"));
            std::set<int> axes_set(axes.begin(), axes.end());
            Dims old_shape = input0->getDimensions();
            int ndim_out = old_shape.nbDims + axes_set.size();
            if( !(ndim_out <= Dims::MAX_DIMS) )
            {
                throw ParseException("Illegal axes");
            }

            Dims new_shape;
            new_shape.nbDims = ndim_out;
            for( int i=0,j=0; i<new_shape.nbDims; ++i ) {
                if( axes_set.count(i) == 0 ) {
                    new_shape.d[i] = old_shape.d[j++];
                } else {
                    new_shape.d[i] = 1;
                }
            }         

            IShuffleLayer* shuffle;
            {
                shuffle = Network()->addShuffle(*input0);
                assert(shuffle);
                shuffle->setReshapeDimensions(new_shape);
                SetLayer(shuffle, node);
            }
            std::cerr << "ParseUnsqueeze" << std::endl;
            return std::make_unique<SingleLayerOperation>(this, node, shuffle);
        }

        void Parser::LoadNode(const menoh_impl::node& node) {
            std::string name = NodeName(node);
	    
            const std::string& operation = node.op_type;
            auto it = m_Functions.find(operation);
            if (it != m_Functions.end())
            {
                auto func = it->second;
                OperationPtr operation = (this->*func)(node);
                auto it = m_Operations.find(name);
                if (it != m_Operations.end())
                {
                    std::string msg("Name ");
                    msg += name;
                    msg += " used by more than one node";
                    throw ParseException(msg);
                }

                m_Operations[name] = std::move(operation);
            }
            else
            {
                throw ParseException("Unsupported operation " + operation + " in Menoh::graph");
            }
        }

        void Parser::CheckOutput(const menoh_impl::graph& graph, const std::vector<std::string>& outputs) {

            for (const std::string& name : outputs)
            {
                bool found = false;

#ifdef TENSORRT_DEBUG
                std::cout << "OutputName = " << name << std::endl;
#endif
                for( unsigned int i=0; i<graph.node_list().size(); ++i)
                {
                    const menoh_impl::node& node = graph.node_list().at(i);
                     
                    auto nodeIt = std::find(node.output_name_list.begin(), node.output_name_list.end(), name);
                    if (nodeIt != node.output_name_list.end())
                    {
                        found = true;
                        break;
                    }
                }

                if( !found )
                    throw ParseException("Couldn't find requested output node '" + name + "' in graph");
            }
        }

        void Parser::LoadParameter(std::unordered_map<std::string, array> const& parameter_table) {
            for( auto param : parameter_table )
            {
                auto arr = param.second;
                array param_arr(arr.dtype(), std::move(arr.dims()), std::move(arr.data()));
                m_Params[param.first] = param_arr;
            }
        }
          
        void Parser::LoadGraph(const menoh_impl::graph& graph) {

            for( unsigned int i=0; i<graph.node_list().size() ; ++i)
            {
                const menoh_impl::node& my_node = graph.node_list().at(i);
                m_Nodes[NodeName(my_node)] = &my_node;
            }

            for( unsigned int i=0; i<graph.node_list().size(); ++i)
            {
                 LoadNode(graph.node_list().at(i));
            }

#ifdef TENSORRT_DEBUG
            std::cout << "markOutput.node   = " << m_Layer->getName() << std::endl;
            std::cout << "markOutput.output = " << m_Layer->getOutput(0)->getName() << std::endl;
#endif
            Network()->markOutput(*m_Layer->getOutput(0));
        }

        INetworkDefinition* Parser::CreateNetwork(
                                         IBuilder* builder,
                                         const menoh_impl::graph& graph,
                                         std::unordered_map<std::string, array> const& parameter_table,
                                         const std::vector<std::string>& outputs){
          std::cerr << "CreateNetwork" << std::endl;
            Cleanup();

            m_Network = builder->createNetwork();
            assert(m_Network);

            if (outputs.size() == 0)
            {
                throw ParseException("outputs must have at least one entry");
            }

            try
            {
                CheckOutput(graph, outputs);
                LoadParameter(parameter_table);
                LoadGraph(graph);
            }
            catch (const ParseException& e)
            {
                Cleanup();
                throw e;
            }

            return Network();
        }

        INetworkDefinition* Parser::Network()
        {
            return m_Network;
        }

        void Parser::Cleanup(){
            m_Nodes.clear();
            m_Params.clear();
            m_Operations.clear();
        }  

        const std::map<std::string, Parser::ParseFunction> Parser::m_Functions = {
          { "Const",                 &Parser::ParseConst },
          { "FC",                    &Parser::ParseFC },
          { "Gemm",                  &Parser::ParseGemm },
          { "Unsqueeze",             &Parser::ParseUnsqueeze },
          { "Identity",              &Parser::ParseIdentity },
          { "Sum",                   &Parser::ParseSum },
          { "BatchNormalization",    &Parser::ParseBatchNormalization },
          { "Conv",                  &Parser::ParseConv2D },
          { "Concat",                &Parser::ParseConcat },
          { "LRN",                   &Parser::ParseLrn },
          { "Mul",                   &Parser::ParseMul },
          { "Add",                   &Parser::ParseAdd },
          { "Placeholder",           &Parser::ParsePlaceholder },
          { "Relu",                  &Parser::ParseRelu },
          { "Sigmoid",               &Parser::ParseSigmoid },
          { "Softmax",               &Parser::ParseSoftmax },
          { "Tanh",                  &Parser::ParseTanh },
          { "MaxPool",               &Parser::ParseMaxPool },
          { "AveragePool",           &Parser::ParseAvgPool },
          { "GlobalMaxPool",         &Parser::ParseGlobalMaxPool },
          { "GlobalAveragePool",     &Parser::ParseGlobalAvgPool },
        };

    } // namespace tensorrt_backend
} // namespace menoh_impl
