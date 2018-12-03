
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>

#include <menoh/array.hpp>
#include <menoh/graph.hpp>
#include <menoh/utility.hpp>

#include <boost/assert.hpp>
#include <boost/format.hpp>
#include <boost/core/ignore_unused.hpp>
#include <boost/log/trivial.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/polymorphic_cast.hpp>

#include <NvInfer.h>

using namespace nvinfer1;

#include <menoh/tensorrt/Exception.hpp>
#include <menoh/tensorrt/Tensor.hpp>
#include <menoh/tensorrt/TypesUtils.hpp>
#include <menoh/tensorrt/Util.hpp>
#include <menoh/tensorrt/Parser.hpp>

namespace menoh_impl {
    namespace tensorrt_backend {

        constexpr unsigned int GetDataTypeSize(DataType dataType)
        {
            switch (dataType)
            {
                case DataType::kINT32:
                case DataType::kFLOAT: return 4U;
                default:               return 0U;
            }
        }

        class TensorProto {

        public:

          TensorProto() : data_f(nullptr), data_i(nullptr), size_(0) {}

          void set_float( float* data, unsigned int size ) {
              data_f = data;
              size_  = size;
          }

          void set_int( int32_t* data, unsigned int size ) {
              data_i = data;
              size_  = size;
          }

          float*  float_val()   const { return data_f; }
          int32_t*  int_val()   const { return data_i; }
          unsigned int size()   const { return size_ ; }
          int8_t*  int8_val()   const { return data_i ? (int8_t*)data_i : data_f ? (int8_t*)data_f : (int8_t*)nullptr; }
          unsigned int int8_size() const {
              if( data_i )      return size_*(sizeof(unsigned int));
              else if( data_f ) return size_*(sizeof(float));
              else              return 0;
          }
    
        private:
          float        *data_f;
          int32_t      *data_i;
          unsigned int size_;
        };

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
                BOOST_ASSERT(m_Layer);
                if ((int)index >= m_Layer->getNbOutputs())
                {
                    throw ParseException(
                        boost::str(boost::format("The requested output slot #%1% "
                            "for %2% does not exist Indext") % index % m_Layer->getName()));
                }
                return m_Layer->getOutput(index);
            }

        protected:
            ILayer* m_Layer;
        };

        class DeferredSingleLayerOperation : public SingleLayerOperation {
        public:
            DeferredSingleLayerOperation(Parser* parser, const menoh_impl::node& node)
            : SingleLayerOperation(parser, node, nullptr)
            {
            }

            ITensor* Output(unsigned int index) override
            {
                if (!m_Layer)
                {
                    CreateLayerDeferred();
                }
                return SingleLayerOperation::Output(index);
            }

        private:
            virtual void CreateLayerDeferred() = 0;
        };
         

        Parser::Parser()
            : m_Network(){
        }

        void Parser::SetLayer(ILayer* layer, const menoh_impl::node& node) {

            layer->setName(PrefixNodeName(node).c_str());
            layer->getOutput(0)->setName(TensorName(NodeName(node)).c_str());

            m_LayerMap[layer->getName()] = layer;
            m_Layer = layer;
        }

        std::vector<OutputOfConstNodeDef>
        Parser::InputNodes(const menoh_impl::node& node) const {
	    std::vector<OutputOfConstNodeDef> ret;

            if (node.op_type == "Const" || node.op_type == "Placeholder")
            {
                return ret;
            }

            ret.reserve(boost::numeric_cast<size_t>(node.input_name_list.size()));

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

                for( auto const& n : m_Params )
                {
                    if (input == n.first)
                    {
                        found = true;
                        break;
                    }
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
                throw ParseException(boost::str(boost::format("Unexpected number of inputs for node %1%. "
							      "Expected %2%, found %3%") % name % expectedNumInputs % numInputs));
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

        ITensor* Parser::GetTensor( std::vector<OutputOfOperation>& inputs, int index ){
            return inputs[index].m_Value->Output(inputs[index].m_Index);
        }

        class ParsedIdentityOperation : public Operation {
        public:
            ParsedIdentityOperation(Parser* parser, const menoh_impl::node& node, Operation* representative)
                : Operation(parser, node)
                , m_Representative(representative)
            {
            }

            virtual ITensor* Output(unsigned int index) override
            {
                BOOST_ASSERT(m_Representative);
                return m_Representative->Output(index);
            }

            virtual Operation* IdentityOperations() override
            {
                return m_Representative->IdentityOperations();
            }

        private:
            Operation* m_Representative;
        };

        OperationPtr Parser::ParseIdentity(const menoh_impl::node& node) {
	    std::vector<OutputOfOperation> inputs = InputCheck(node, 1);
            return std::make_unique<ParsedIdentityOperation>(this, node, inputs[0].m_Value);
        }

        template <typename T>
        class Vector {

        public :

          Vector( const T* data_ = nullptr, const unsigned int size_ = 0 )
	    : my_data(data_)
	    , my_size(size_) {}
	    
          const T* data() const noexcept { return my_data; }
          unsigned int size() { return my_size; }
          void set_data(const T* data_)  { my_data = data_; }
          void set_size(unsigned int size_) { my_size = size_; }
	  
        private:

          const T*  my_data;
          unsigned int my_size;
        };

        template <typename T>
        class ConstOperation : public DeferredSingleLayerOperation {
        public:
            ConstOperation(Parser* parser, const menoh_impl::node& node,
                           const T* tensorData, const TensorInfo& tensorInfo)
                : DeferredSingleLayerOperation(parser, node)
                , m_Storage(tensorData, tensorInfo.GetNumElements())
                , m_TensorInfo(tensorInfo)
                , name(NodeName(node))
            {
                dimentions.nbDims = tensorInfo.GetNumDimensions(); 
                TensorShape shape = tensorInfo.GetShape();
                for( int i=0 ; i<dimentions.nbDims ; i++ ) {
                    dimentions.d[i]    = shape[i];
                    dimentions.type[i] = DimensionType::kSEQUENCE;
                }

                weights.type   = nvinfer1::DataType::kFLOAT;
                weights.values = tensorData;
                weights.count  = tensorInfo.GetNumElements();
            }

            void CreateLayerDeferred() override
            {
                BOOST_ASSERT(m_Layer == nullptr);
                IConstantLayer* const1;
                const1 = m_Parser->Network()->addConstant(dimentions, weights);
                const1->setName(name.c_str());
                const1->getOutput(0)->setName(TensorName(name).c_str());
                m_Layer = const1;
            }

            ConstTensor GetConstTensor(Vector<T>& outputTensorData)
            {
                const TensorInfo outInfo = m_TensorInfo;

                outputTensorData.set_data(m_Storage.data());
                outputTensorData.set_size(m_Storage.size());

                ConstTensor constTensor(outInfo, outputTensorData);
                return constTensor;
            }

            ConstTensor GetConstTensor(bool swizzleForConvolutionWeights, std::vector<T>& outputTensorData)
            {
                const TensorInfo outInfo = m_TensorInfo;

                outputTensorData.resize(m_TensorInfo.GetNumElements());
                memcpy(outputTensorData.data(), m_Storage.data(), m_TensorInfo.GetNumBytes());

                ConstTensor constTensor(outInfo, outputTensorData);
                return constTensor;
            }

        private:
            Vector<T> m_Storage;
            TensorInfo m_TensorInfo;
            Weights weights;
            Dims dimentions;
            std::string name;
        };      

        template <template<typename> class OperatorType, typename T = int8_t>
        struct MakeOperation {
            template<typename DataType, class... Args>
            inline static std::unique_ptr<OperatorType<DataType>> Parse(Parser* parser, const menoh_impl::node& node,
                Args&&... args)
            {
                return std::make_unique<OperatorType<DataType>>(parser, node, std::forward<Args>(args)...);
            }
        };
      
        template <>
        struct MakeOperation<ConstOperation> {
            template<typename DataType, class... Args>
            inline static std::unique_ptr<ConstOperation<DataType>> Parse(Parser* parser,
						      const menoh_impl::node& node,
						      const Vector<int8_t>& tensorData,
						      const TensorInfo& tensorInfo)
            {
                return std::make_unique<ConstOperation<DataType>>(parser, node,
                            reinterpret_cast<const DataType*>(tensorData.data()), tensorInfo);
            }
        };

        template <class FuncType>
        struct InvokeParseFunction {
            template<class ResType, class... Args>
            inline static ResType Result(DataType dataType, Args&&... args)
            {
                if (dataType == DataType::kFLOAT)
                {
                    return FuncType::template Parse<float>(std::forward<Args>(args)...);
                }
                else if (dataType == DataType::kINT32)
                {
                    return FuncType::template Parse<int32_t>(std::forward<Args>(args)...);
                }

                return ResType();
            }

            template<class... Args>
            inline static void Result(DataType dataType, Args&&... args)
            {
                if (dataType == DataType::kFLOAT)
                {
                    FuncType::template Parse<float>(std::forward<Args>(args)...);
                }
                else if (dataType == DataType::kINT32)
                {
                    FuncType::template Parse<int32_t>(std::forward<Args>(args)...);
                }
            }
        };  

        OperationPtr Parser::ParseConst(const menoh_impl::node& node) {
            std::string name = NodeName(node);

            BOOST_ASSERT(node.op_type == "Const");

            auto it = m_Params.find(name);
            if (it == m_Params.end() )
            {
                throw ParseException(boost::str(boost::format("ParseConst : not found %1%") % name));
            }

            const DataType dataType = DataType::kFLOAT; // dtype_t::float_
            unsigned int numElements = 0U;
            auto arr  = m_Params[name];
            std::vector<unsigned int> dimensionSizes(arr.dims().data(), arr.dims().data()+arr.dims().size());
            if (!dimensionSizes.empty())
            {
                numElements = std::accumulate(dimensionSizes.begin(), dimensionSizes.end(),
                                              1U, std::multiplies<unsigned int>());
            }

            const TensorInfo tensorInfo(static_cast<unsigned int>(dimensionSizes.size()), dimensionSizes.data(), dataType);
#ifdef TENSORRT_DEBUG
	    std::cout << "   output = " << tensorInfo.GetNumDimensions() << ", " << tensorInfo.GetNumElements() << std::endl;
	    std::cout << "   outputShape = ";
	    for( unsigned int i=0 ; i<tensorInfo.GetNumDimensions() ; i++ )
	      std::cout << tensorInfo.GetShape()[i] << " ";
	    std::cout << std::endl;
#endif

            Vector<int8_t> tensorData((const int8_t *)arr.data(), numElements*GetDataTypeSize(dataType));
            if ((unsigned int)tensorData.size() > tensorInfo.GetNumBytes())
            {
                throw ParseException(boost::str(
                    boost::format("Number of elements (%1%) should be less than or equal \
                    to the number of elements implied by the shape argument (%2%) for Const node - %3%")
                    % (tensorData.size() / GetDataTypeSize(dataType))
                    % tensorInfo.GetNumElements()
                    % name));
            }

            return InvokeParseFunction<MakeOperation<ConstOperation>>::Result<OperationPtr>(
                                       dataType, this, node, tensorData, tensorInfo);
        }

        template<typename Type>
        bool Parser::HasParsedConstTensor(const std::string & nodeName) const {
            auto it = m_Operations.find(nodeName);
            if (it == m_Operations.end() ||
                dynamic_cast<ConstOperation<Type>*>(it->second.get()) == nullptr)
            {
                return false;
            }
            else
            {
                return true;
            }
        }

        OperationPtr Parser::ParseBatchNormalization(const menoh_impl::node& node) {
            std::string name = NodeName(node);

            std::vector<OutputOfConstNodeDef> nodes = InputNodes(node);
            unsigned int numInputs = static_cast<unsigned int>(nodes.size());
            std::vector<OutputOfOperation> inputs = InputCheck(node, numInputs);

            if (!HasParsedConstTensor<float>(NodeName(inputs[1].m_Value->GetNode()))
             || !HasParsedConstTensor<float>(NodeName(inputs[2].m_Value->GetNode()))
             || !HasParsedConstTensor<float>(NodeName(inputs[3].m_Value->GetNode()))
             || !HasParsedConstTensor<float>(NodeName(inputs[4].m_Value->GetNode())))
            {
                throw ParseException("only supports BatchNormalization layers with constant weights");
            }

            ITensor* input0 = GetTensor(inputs,0);
          
            ConstOperation<float>* scaleNode =
                boost::polymorphic_downcast<ConstOperation<float>*>(inputs[1].m_Value);
            Vector<float> scaleTensorData;
            ConstTensor scaleTensor = scaleNode->GetConstTensor(scaleTensorData);

            ConstOperation<float>* biasNode =
                boost::polymorphic_downcast<ConstOperation<float>*>(inputs[2].m_Value);
            Vector<float> biasTensorData;
            ConstTensor biasTensor = biasNode->GetConstTensor(biasTensorData);

            ConstOperation<float>* meanNode =
                boost::polymorphic_downcast<ConstOperation<float>*>(inputs[3].m_Value);
            Vector<float> meanTensorData;
            ConstTensor meanTensor = meanNode->GetConstTensor(meanTensorData);

            ConstOperation<float>* varianceNode =
                boost::polymorphic_downcast<ConstOperation<float>*>(inputs[4].m_Value);
            Vector<float> varianceTensorData;
            ConstTensor varianceTensor = varianceNode->GetConstTensor(varianceTensorData);

            auto epsilon = optional_attribute_float(node, "epsilon", 1e-5f);

            Weights combined_scale_weights{ scaleTensor.GetDataType(), scaleTensorData.data(), scaleTensorData.size()};
            Weights combined_bias_weights{ biasTensor.GetDataType(), biasTensorData.data(), biasTensorData.size()};

            size_t nweight = input0->getDimensions().d[0];
            for( size_t i=0; i<nweight; ++i ) {
                float scale    = (static_cast<float const*>(scaleTensorData.data())[i]);
                float bias     = (static_cast<float const*>(biasTensorData.data())[i]);
                float mean     = (static_cast<float const*>(meanTensorData.data())[i]);
                float variance = (static_cast<float const*>(varianceTensorData.data())[i]);
                float& combined_scale_ref = const_cast<float*>(
                    static_cast<float const*>(combined_scale_weights.values))[i];
                float& combined_bias_ref  = const_cast<float*>(
                    static_cast<float const*>(combined_bias_weights.values))[i];
                combined_scale_ref = scale / sqrtf(variance + epsilon);
                combined_bias_ref  = bias - mean * combined_scale_ref;
            }
  
            IScaleLayer* scale;
            {
                scale = Network()->addScale(*input0, ScaleMode::kCHANNEL, combined_bias_weights, combined_scale_weights, {});
                assert(scale);
                SetLayer(scale, node);
            }
            return std::make_unique<SingleLayerOperation>(this, node, scale);
        }

        OperationPtr Parser::ParseFC(const menoh_impl::node& node) {
            std::string name = NodeName(node);

            std::vector<OutputOfOperation> inputs = InputCheck(node, 3);
            ITensor* input0 = GetTensor(inputs,0);
            
            ConstOperation<float>* weightNode = nullptr;
            ConstOperation<float>* biasNode   = nullptr;

            weightNode = boost::polymorphic_downcast<ConstOperation<float>*>(inputs[1].m_Value);
            biasNode   = boost::polymorphic_downcast<ConstOperation<float>*>(inputs[2].m_Value);

            Vector<float> weightTensorData;
            ConstTensor weightTensor = weightNode->GetConstTensor(weightTensorData);
            Weights weight{weightTensor.GetDataType(), weightTensorData.data(), weightTensorData.size()};
            Vector<float> biasTensorData;
            ConstTensor biasTensor  = biasNode->GetConstTensor(biasTensorData);
            Weights bias{biasTensor.GetDataType(), biasTensorData.data(), biasTensorData.size()};
            if (weightTensor.GetShape()[0] != biasTensor.GetShape()[0])
            {
                throw ParseException("shape of weight and bias do not match");
            }

            IFullyConnectedLayer* full;
            {
                full = Network()->addFullyConnected(*input0, bias.count, weight, bias);
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

            if (!HasParsedConstTensor<float>(NodeName(inputs[1].m_Value->GetNode()))
                || (numInputs == 3 && !HasParsedConstTensor<float>(NodeName(inputs[2].m_Value->GetNode()))))
            {
                throw ParseException("only supports Convolution layers with constant weights and biases");
            }

            ConstOperation<float>* weightNode =
                boost::polymorphic_downcast<ConstOperation<float>*>(inputs[1].m_Value);
            Vector<float> weightTensorData;
            ConstTensor weightTensor = weightNode->GetConstTensor(weightTensorData);

            ConstOperation<float>* biasNode   = 
                (numInputs == 3) ? boost::polymorphic_downcast<ConstOperation<float>*>(inputs[2].m_Value)
                                 : nullptr;
            Vector<float> biasTensorData;
            ConstTensor biasTensor = (numInputs == 3) ? biasNode->GetConstTensor(biasTensorData) : ConstTensor();

            DataType biasType = (numInputs == 3) ? biasNode->GetConstTensor(biasTensorData).GetDataType() : DataType::kFLOAT;

            std::vector<int> strides, kernel_shape, pads;
            std::tie(strides, kernel_shape, pads) = attributes_for_2d_data_processing(node);

            ITensor* input0 = GetTensor(inputs,0);

#ifdef TENSORRT_DEBUG
	    std::cout << "     weight(" << weightTensor.GetNumDimensions() << ") = "
		      << weightTensor.GetShape()[0] << ", " << weightTensor.GetShape()[1];
	    std::cout << ", " << weightTensor.GetShape()[2] << ", " << weightTensor.GetShape()[3] << std::endl;
            if( numInputs == 3 )
                std::cout << "       bias(" << biasTensor.GetNumDimensions() << ") = " << biasTensor.GetShape()[0] << std::endl;
            std::cout << "           strides      = " << strides[0]      << ", " << strides[1]      << std::endl;
            std::cout << "           kernel_shape = " << kernel_shape[0] << ", " << kernel_shape[1] << std::endl;
            std::cout << "           pads         = " << pads[0]         << ", " << pads[1];
            if( pads.size() >= 4 )
                std::cout << ",  = " << pads[2]         << ", " << pads[3] << std::endl ;
            else
                std::cout << std::endl;
#endif

            DimsHW begin_pad{pads[0], pads[1]}, end_pad{(pads.size()<=2)? pads[0] : pads[2],
                                                        (pads.size()<=2)? pads[1] : pads[3]};
            if( (begin_pad.h() != end_pad.h()) || (begin_pad.w() != end_pad.w()) )
            {
                auto layer = Network()->addPadding(*input0, begin_pad, end_pad );
                input0 = layer->getOutput(0);
            }

            Weights weight{weightTensor.GetDataType(), weightTensorData.data(), weightTensorData.size()};
            Weights bias{biasType, biasTensorData.data(), biasTensorData.size()};
            int nbOutputMaps = weightTensor.GetShape()[0]; 

            IConvolutionLayer* conv;
            {
                conv = Network()->addConvolution(*input0, 
                                                  nbOutputMaps, DimsHW{kernel_shape[0], kernel_shape[1]}, weight, bias);
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

            ITensor* input0 = GetTensor(inputs,0);

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
            ITensor* input0 = GetTensor(inputs,0);

            float alpha = attribute_float(node, "alpha");
            float beta  = attribute_float(node, "beta");
            float k     = attribute_float(node, "bias");
            int window  = attribute_int(node, "depth_radius");

            window = window * 2 + 1;

            ILRNLayer* lrn;
            {
                lrn = Network()->addLRN(*input0, window, alpha, beta, k);
                assert(lrn);
                SetLayer(lrn, node);
            }
            return std::make_unique<SingleLayerOperation>(this, node, lrn);
        }

        OperationPtr Parser::ParseSum(const menoh_impl::node& node) {
            std::string name = NodeName(node);
	    
            std::vector<OutputOfOperation> inputs = InputCheck(node, 2);
            ITensor* input0 = GetTensor(inputs,0);
            ITensor* input1 = GetTensor(inputs,1);

            IElementWiseLayer* add1;
            {
                add1 = Network()->addElementWise(*input0, *input1, ElementWiseOperation::kSUM);
                assert(add1);
                SetLayer(add1, node);
            }
            return std::make_unique<SingleLayerOperation>(this, node, add1);
        }  

        OperationPtr Parser::ParseMul(const menoh_impl::node& node) {
            std::string name = NodeName(node);
	    
            std::vector<OutputOfOperation> inputs = InputCheck(node, 2);
            ITensor* input0 = GetTensor(inputs,0);
            ITensor* input1 = GetTensor(inputs,1);

            IElementWiseLayer* mul;
            { 
                mul = Network()->addElementWise(*input0, *input1, ElementWiseOperation::kPROD);
                assert(mul);
                SetLayer(mul, node);
            }
            return std::make_unique<SingleLayerOperation>(this, node, mul);
        }  

        OperationPtr Parser::ParseAdd(const menoh_impl::node& node) {
            std::string name = NodeName(node);
	    
            std::vector<OutputOfOperation> inputs = InputCheck(node, 2);
            ITensor* input0 = GetTensor(inputs,0);
            ITensor* input1 = GetTensor(inputs,1);

            IElementWiseLayer* add;
            {            
                add = Network()->addElementWise(*input0, *input1, ElementWiseOperation::kSUM);
                assert(add);
                SetLayer(add, node);
            }
            return std::make_unique<SingleLayerOperation>(this, node, add);
        }  

        OperationPtr Parser::ParsePlaceholder(const menoh_impl::node& node) {
            std::string name = NodeName(node);
	    
            std::vector<OutputOfOperation> inputs = InputCheck(node, 0);

            auto it = m_InputShapes.find(name);
            if (it == m_InputShapes.end())
            {
                throw ParseException("Missing input shape for Placeholder '" + name + "'");
            }

            auto dims = (std::vector<uint32_t> const&)attribute_ints(node, "dims");
            const TensorInfo tensorInfo(static_cast<unsigned int>(dims.size()), (const unsigned int*)dims.data(), DataType::kFLOAT);

#ifdef TENSORRT_DEBUG
	    std::cout << "   output = " << tensorInfo.GetNumDimensions() << ", " << tensorInfo.GetNumElements() << std::endl;
	    std::cout << "   outputShape = ";
	    for( unsigned int i=0 ; i<tensorInfo.GetNumDimensions() ; i++ )
	      std::cout << tensorInfo.GetShape()[i] << " ";
	    std::cout << std::endl;
#endif
            ITensor* placeholder;
            {
                Dims inputDims;
                int offset = dims.size() - 3 ;            
                inputDims.nbDims = dims.size() - offset ;
                for( unsigned int i=offset ; i<dims.size() ; i++ )
                    inputDims.d[i-offset] = dims[i];  

                placeholder = Network()->addInput(PrefixNodeName(node).c_str(), nvinfer1::DataType::kFLOAT, inputDims);
                assert(placeholder);
#ifdef TENSORRT_DEBUG
                std::cout << "           inputDims.nbDims = " << inputDims.nbDims;
                for( int i=0 ; i<inputDims.nbDims ; i++ )
                    std::cout << std::endl << "           inputDims.d[" << i << "] = " << inputDims.d[i];
                std::cout << std::endl; 
#endif
            }

            IScaleLayer* scale_l;
            {
                const Weights power{DataType::kFLOAT, nullptr, 0};
                const Weights shift{DataType::kFLOAT, nullptr, 0};
                const Weights scale{DataType::kFLOAT, nullptr, 0};

                scale_l = Network()->addScale(*placeholder, ScaleMode::kUNIFORM, shift, scale, power);
                assert(scale_l);
                SetLayer(scale_l, node);
            }

            return std::make_unique<SingleLayerOperation>(this, node, scale_l);
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
            ITensor* input0 = GetTensor(inputs,0);

            IActivationLayer* activate;
            {
                activate = Network()->addActivation(*input0, activationType);
                assert(activate);
                SetLayer(activate, node);
            }
            return std::make_unique<SingleLayerOperation>(this, node, activate);
        }

        OperationPtr Parser::ParseSoftmax(const menoh_impl::node& node) {
            std::string name = NodeName(node);

            std::vector<OutputOfOperation> inputs = InputCheck(node, 1);
            ITensor* input0 = GetTensor(inputs,0);

            ISoftMaxLayer* softmax;
            {
                softmax = Network()->addSoftMax(*input0);
                assert(softmax);
                SetLayer(softmax, node);
            }

#ifdef TENSORRT_DEBUG
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
            ITensor* input0 = GetTensor(inputs,0);

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
            std::cout << "           input0.name  = " << input0->getName() << std::endl;
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
            ITensor* input0 = GetTensor(inputs,0);

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
            std::cout << "           input0.name  = " << input0->getName() << std::endl;
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
                    // None
                }
                else if( end_pad.d[i] == (begin_pad.d[i] + 1) )
                {
                    begin_pad.d[i] += strides[i];
                    pre_crop.d[i] = 1;
                }
                else
                {
                    std::cerr << "Illeagl Pad " << std::endl;
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

        OperationPtr Parser::ParseGlobalMaxPool(const menoh_impl::node& node) {
            std::string name = NodeName(node);
	    
            std::vector<OutputOfOperation> inputs = InputCheck(node, 1);
            ITensor* input0 = GetTensor(inputs,0);

            IPoolingLayer* pool;
            {
                Dims dims = input0->getDimensions();
                if( dims.nbDims != 3 )
                    throw ParseException("GlobalAvgMaxPool layser's input dimensions must be 3 (three).");
                DimsHW kernel_shape({dims.d[1], dims.d[2]});
                pool = Network()->addPooling(*input0, PoolingType::kMAX, kernel_shape);
                assert(pool);
                SetLayer(pool, node);
            } 
            return std::make_unique<SingleLayerOperation>(this, node, pool);
        }
        
        OperationPtr Parser::ParseGlobalAvgPool(const menoh_impl::node& node) {
            std::string name = NodeName(node);
	    
            std::vector<OutputOfOperation> inputs = InputCheck(node, 1);
            ITensor* input0 = GetTensor(inputs,0);
            
            IPoolingLayer* pool;
            {
                Dims dims = input0->getDimensions();
                if( dims.nbDims != 3 )
                    throw ParseException("GlobalAvgPool layser's input dimensions must be 3 (three).");
                DimsHW kernel_shape({dims.d[1], dims.d[2]});
                pool = Network()->addPooling(*input0, PoolingType::kAVERAGE, kernel_shape);
                assert(pool);
                SetLayer(pool, node);
            } 
            return std::make_unique<SingleLayerOperation>(this, node, pool);
        }

        OperationPtr Parser::ParseGemm(const menoh_impl::node& node) {
            std::string name = NodeName(node);
            
            std::vector<OutputOfOperation> inputs = InputCheck(node, 3);
            ITensor* input0 = GetTensor(inputs,0);

            ConstOperation<float>* weightNode;
            ConstOperation<float>* biasNode;

            weightNode = boost::polymorphic_downcast<ConstOperation<float>*>(inputs[1].m_Value);
            biasNode   = boost::polymorphic_downcast<ConstOperation<float>*>(inputs[2].m_Value);

            Vector<float> weightTensorData;
            ConstTensor weightTensor = weightNode->GetConstTensor(weightTensorData);
            Weights weight{weightTensor.GetDataType(), weightTensorData.data(), weightTensorData.size()};

            Vector<float> biasTensorData;
            ConstTensor biasTensor  = biasNode->GetConstTensor(biasTensorData);
            Weights bias{biasTensor.GetDataType(), biasTensorData.data(), biasTensorData.size()};

            if (weightTensor.GetShape()[0] != biasTensor.GetShape()[0])
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
                full = Network()->addFullyConnected(*input0, bias.count, weight, bias);
                assert(full);
                SetLayer(full, node);
            }

            return std::make_unique<SingleLayerOperation>(this, node, full);
        }

        OperationPtr Parser::ParseUnsqueeze(const menoh_impl::node& node) {
            std::string name = NodeName(node);
            
            std::vector<OutputOfOperation> inputs = InputCheck(node, 1);
            ITensor* input0 = GetTensor(inputs,0);

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
                    throw ParseException(boost::str(boost::format("Name %1% used by more than one node") % name));
                }

                m_Operations[name] = std::move(operation);
            }
            else
            {
                throw ParseException(boost::str(
                    boost::format("Unsupported operation %1% in Menoh::graph") % operation));
            }
        }

        void Parser::LoadGraph(const menoh_impl::graph& graph,
                                    std::unordered_map<std::string, array> const& parameter_table) {

            for( unsigned int i=0; i<graph.node_list().size() ; ++i)
            {
                const menoh_impl::node& my_node = graph.node_list().at(i);
                m_Nodes[NodeName(my_node)] = &my_node;
            }

            for( auto param : parameter_table )
            {
                auto arr = param.second;
                array param_arr(arr.dtype(), std::move(arr.dims()), std::move(arr.data()));
                m_Params[param.first] = param_arr;
            }

            std::vector<const menoh_impl::node*> targetNodes;
            for (const std::string& name : m_Outputs)
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
                        targetNodes.push_back(&node);
                        found = true;
                        break;
                    }
                }
                if( !found )
                    throw ParseException("Couldn't find requested output node '" + name + "' in graph");
            }

            for( auto node : targetNodes )
                m_Outputs.push_back(NodeName(*node));
	    
            std::vector<const menoh_impl::node*> sortedNodes;
            if (!Util::GraphTopologicalSort<const menoh_impl::node*>(
                targetNodes,
                [this](const menoh_impl::node* node)
                {
                    auto outputs = InputNodes(*node);
                    std::vector<const menoh_impl::node*> nodesOnly;
                    for (const auto & o : outputs) {
                        nodesOnly.push_back(o.m_Value);
                    }
                    return nodesOnly;
                },
                sortedNodes))
            {
                throw ParseException("Cycle detected in graph");
            }

            for (const auto& it : sortedNodes)
            {
                LoadNode(*it);
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
                                         const std::map<std::string, TensorShape>& inputShapes,
                                         const std::vector<std::string>& outputs){
            Cleanup();

            if (outputs.size() == 0)
            {
                throw ParseException("requestedOutputs must have at least one entry");
            }
            m_InputShapes = inputShapes;
            m_Outputs     = outputs;

            m_Network = builder->createNetwork();
            assert(m_Network);

            try
            {
                LoadGraph(graph, parameter_table);
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
            m_InputShapes.clear();
            m_Outputs.clear();
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
