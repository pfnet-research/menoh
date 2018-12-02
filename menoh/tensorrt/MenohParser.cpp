
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
#include <menoh/tensorrt/MenohParser.hpp>

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

        using LayerBindingId = int;
        
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

        void PrintDims(Dims dims) {
            std::cout << "nbDims = " << dims.nbDims << std::endl;
            for( int i=0 ; i<dims.nbDims ; i++ ){
  	        std::cout << "d[" << i << "] = " << dims.d[i] << std::endl;
  	        std::cout << "type[" << i << "] = " << (int)dims.type[i] << std::endl;
            }
        }
      
        std::string GetNodeName( const menoh_impl::node& node ) {
            std::string name;
            for(auto it = node.input_name_list.begin(); it != node.input_name_list.end(); ++it) {
                name += *it;
            }          
            for(auto it = node.output_name_list.begin(); it != node.output_name_list.end(); ++it) {
                name += *it;
            }          
            return name;
        }

        std::string GetPrefixNodeName( const menoh_impl::node& node ) {
          std::string name(node.op_type + ":" + GetNodeName(node));
          return name;
        }

        float ReadMandatoryNodeFloatAttribute(const menoh_impl::node& node, const std::string& name){
            return attribute_float( node, name );  
        }

        uint32_t ReadMandatoryNodeUint32Attribute(const menoh_impl::node& node, const std::string& name){
            return attribute_int( node, name );  
        }

        std::vector<uint32_t> ReadMandatoryNodeUint32ListAttribute(const menoh_impl::node& node, const std::string& name){
            return (std::vector<uint32_t> const&)attribute_ints( node, name );
        }

        std::vector<uint32_t> ReadOptionalNodeUint32ListAttribute(const menoh_impl::node& node, const std::string& name)
        {
            return (std::vector<uint32_t> const&)attribute_ints( node, name );
        }

        class SingleLayerParsedMenohOperation : public ParsedMenohOperation{
        public:
            SingleLayerParsedMenohOperation(MenohParser* parser, const menoh_impl::node& node, ILayer* layer)
            : ParsedMenohOperation(parser, node)
            , m_Layer(layer)
            {
            }

            ITensor* ResolveOutputSlot(unsigned int MenohOutputIndex) override
            {
                BOOST_ASSERT(m_Layer);
                if ((int)MenohOutputIndex >= m_Layer->getNbOutputs())
                {
                    throw ParseException(
                        boost::str(boost::format("The requested output slot #%1% "
                            "for %2% does not exisMenohOutputIndext") % MenohOutputIndex % m_Layer->getName()));
                }
                return m_Layer->getOutput(MenohOutputIndex);
            }

        protected:
            ILayer* m_Layer;
        };

        class DeferredSingleLayerParsedMenohOperation : public SingleLayerParsedMenohOperation {
        public:
            DeferredSingleLayerParsedMenohOperation(MenohParser* parser, const menoh_impl::node& node)
            : SingleLayerParsedMenohOperation(parser, node, nullptr)
            {
            }

            ITensor* ResolveOutputSlot(unsigned int MenohOutputIndex) override
            {
                if (!m_Layer)
                {
                    CreateLayerDeferred();
                }
                return SingleLayerParsedMenohOperation::ResolveOutputSlot(MenohOutputIndex);
            }

        private:
            virtual void CreateLayerDeferred() = 0;
        };
         

        MenohParser::MenohParser()
            : m_Network(){
        }

        void MenohParser::SetLayer(ILayer* layer){
            m_LayerMap[layer->getName()] = layer;
            m_Layer = layer;
        }

        const node* MenohParser::ResolveIdentityNode(const node* node){
            if (node->op_type != "Identity")
            {
                return node;
            }

            if (node->input_name_list.size() != 1)
            {
                throw ParseException("Identity node does not have correct amount of inputs!");
            }

            auto it = m_NodesByName.find(node->input_name_list.at(0));
            if (it != m_NodesByName.end())
            {
 	        const menoh_impl::node* inputNode = it->second;
                return ResolveIdentityNode(inputNode);
            }
            else
            {
                throw ParseException("Cannot find what the Identity node is linked to!");
            }
        }

        std::vector<OutputOfConstNodeDef>
        MenohParser::GetMenohInputNodes(const menoh_impl::node& node) const {
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
                for( auto const& n : m_NodesByName )
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

                for( auto const& n : m_ParamByName )
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

        std::vector<OutputOfParsedMenohOperation>
        MenohParser::GetInputParsedMenohOperationsChecked(const menoh_impl::node& node, std::size_t expectedNumInputs){
            std::string name = GetNodeName(node);

            std::vector<OutputOfConstNodeDef> nodes = GetMenohInputNodes(node);
            const std::size_t numInputs = node.input_name_list.size();
            if (numInputs != expectedNumInputs)
            {
                throw ParseException(boost::str(boost::format("Unexpected number of inputs for node %1%. "
							      "Expected %2%, found %3%") % name % expectedNumInputs % numInputs));
            }
            std::vector<OutputOfParsedMenohOperation> result;
            for (auto&& node : nodes)
            {
                auto it = m_ParsedMenohOperations.find(GetNodeName(*(node.m_IndexedValue)));
                if (it == m_ParsedMenohOperations.end())
                {
                    throw ParseException("Node with name '" + GetNodeName(*(node.m_IndexedValue)) + "' has not been parsed");
                }
                ParsedMenohOperation* parsedOp = it->second.get();
                parsedOp = parsedOp->ResolveIdentityOperations();
                result.push_back(OutputOfParsedMenohOperation(parsedOp,node.m_Index));
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

        ParsedMenohOperationPtr MenohParser::ParseFC(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            boost::ignore_unused(graph);
	    
            ILayer* layer = AddFullyConnectedLayer(node, GetNodeName(node).c_str());
            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, layer);
        }

        class ParsedIdentityMenohOperation : public ParsedMenohOperation {
        public:
            ParsedIdentityMenohOperation(MenohParser* parser, const menoh_impl::node& node, ParsedMenohOperation* representative)
                : ParsedMenohOperation(parser, node)
                , m_Representative(representative)
            {
            }

            virtual ITensor* ResolveOutputSlot(unsigned int MenohOutputIndex) override
            {
                BOOST_ASSERT(m_Representative);
                return m_Representative->ResolveOutputSlot(MenohOutputIndex);
            }

            virtual ParsedMenohOperation* ResolveIdentityOperations() override
            {
                return m_Representative->ResolveIdentityOperations();
            }

        private:
            ParsedMenohOperation* m_Representative;
        };

        ParsedMenohOperationPtr MenohParser::ParseIdentity(const menoh_impl::node& node, const menoh_impl::graph& graph) {
	    boost::ignore_unused(graph);

	    std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 1);
            return std::make_unique<ParsedIdentityMenohOperation>(this, node, inputs[0].m_IndexedValue);
        }

        template <typename T>
        class MenohVector {

        public :

          MenohVector( const T* data_ = nullptr, const unsigned int size_ = 0 )
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
        class ParsedConstMenohOperation : public DeferredSingleLayerParsedMenohOperation {
        public:
            ParsedConstMenohOperation(MenohParser* parser, const menoh_impl::node& node,
                                      const T* tensorData, const TensorInfo& tensorInfo)
                : DeferredSingleLayerParsedMenohOperation(parser, node)
                , m_Storage(tensorData, tensorInfo.GetNumElements())
                , m_TensorInfo(tensorInfo)
                , name(GetNodeName(node))
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
                const1 = m_Parser->m_Network->addConstant(dimentions, weights);
                const1->setName(name.c_str());
                std::string pname("tensor_"+name);
                const1->getOutput(0)->setName(pname.c_str());
                m_Layer = const1;
            }

            ConstTensor GetConstTensor(MenohVector<T>& outputTensorData)
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
            MenohVector<T> m_Storage;
            TensorInfo m_TensorInfo;
            Weights weights;
            Dims dimentions;
            std::string name;
        };      

        DataType ConvertMenohTensorDataType(const dtype_t MenohDataType) {
            switch (MenohDataType)
            {
            case dtype_t::float_:
                return DataType::kFLOAT;
                break;
            /*
            case dtype_t::DT_INT32:
                return DataType::kINT32;
                break;
            */
            default:
                throw ParseException(boost::str(
                    boost::format("Unknown DataType %1% for node")
                    % dtype_to_string(MenohDataType)));
            }
        }

        struct ParseMenohTensorValueList {
            template<typename DataType>
            static void Parse(
                const TensorProto& MenohTensor,
                unsigned int dstElements,
                std::vector<int8_t>& outputData);

            template <typename DataType>
            static void ReadData(const void* srcData, unsigned int numSrcElements,
                std::vector<int8_t>& dstData, unsigned int numDstElements)
            {
                if (numSrcElements == 0)
                {
                    return;
                }

                if (numDstElements == 0)
                {
                    numDstElements = numSrcElements;
                }

                dstData.resize(std::max(numSrcElements, numDstElements) * sizeof(DataType));

                const DataType* srcTensor = reinterpret_cast<const DataType*>(srcData);
                DataType* dstTensor = reinterpret_cast<DataType*>(dstData.data());

                std::copy(srcTensor, srcTensor + numSrcElements, dstTensor);

                if (numDstElements > numSrcElements)
                {
                    std::fill(dstTensor + numSrcElements, dstTensor + numDstElements, srcTensor[numSrcElements - 1]);
                }
            }

        };

        template <>
        void ParseMenohTensorValueList::Parse<float>(const TensorProto& MenohTensor,
            unsigned int dstElements, std::vector<int8_t>& outputData) {
            ReadData<float>(MenohTensor.float_val(), static_cast<unsigned int>(MenohTensor.size()),
                outputData, dstElements);
        }

        template <>
        void ParseMenohTensorValueList::Parse<int32_t>(const TensorProto& MenohTensor,
            unsigned int dstElements, std::vector<int8_t>& outputData) {
            ReadData<int32_t>(MenohTensor.int_val(), static_cast<unsigned int>(MenohTensor.size()),
                outputData, dstElements);
        }

        template <template<typename> class OperatorType, typename T = int8_t>
        struct MakeMenohOperation {
            template<typename DataType, class... Args>
            inline static std::unique_ptr<OperatorType<DataType>> Parse(MenohParser* parser, const menoh_impl::node& node,
                Args&&... args)
            {
                return std::make_unique<OperatorType<DataType>>(parser, node, std::forward<Args>(args)...);
            }
        };
      
        template <>
        struct MakeMenohOperation<ParsedConstMenohOperation> {
            template<typename DataType, class... Args>
            inline static std::unique_ptr<ParsedConstMenohOperation<DataType>> Parse(MenohParser* parser,
						      const menoh_impl::node& node,
						      const MenohVector<int8_t>& tensorData,
						      const TensorInfo& tensorInfo)
            {
                return std::make_unique<ParsedConstMenohOperation<DataType>>(parser, node,
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

        ParsedMenohOperationPtr MenohParser::ParseConst(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            boost::ignore_unused(graph);
            std::string name = GetNodeName(node);

            BOOST_ASSERT(node.op_type == "Const");

#ifdef TENSORRT_DEBUG
            std::cout << std::endl << " [node] : Const, " << name << std::endl;
#endif
            auto it = m_ParamByName.find(name);
            if (it == m_ParamByName.end() )
            {
                throw ParseException(boost::str(boost::format("ParseConst : not found %1%") % name));
            }

            const dtype_t MenohDataType = dtype_t::float_;
            const DataType dataType = ConvertMenohTensorDataType(MenohDataType);
            unsigned int numElements = 0U;
            auto arr  = m_ParamByName[name];
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

            MenohVector<int8_t> tensorData((const int8_t *)arr.data(), numElements*GetDataTypeSize(dataType));
            if ((unsigned int)tensorData.size() > tensorInfo.GetNumBytes())
            {
                throw ParseException(boost::str(
                    boost::format("Number of elements (%1%) should be less than or equal \
                    to the number of elements implied by the shape argument (%2%) for Const node - %3%")
                    % (tensorData.size() / GetDataTypeSize(dataType))
                    % tensorInfo.GetNumElements()
                    % name));
            }

            return InvokeParseFunction<MakeMenohOperation<ParsedConstMenohOperation>>::Result<ParsedMenohOperationPtr>(
                                       dataType, this, node, tensorData, tensorInfo);
        }

        template<typename Type>
        bool MenohParser::HasParsedConstTensor(const std::string & nodeName) const {
            auto it = m_ParsedMenohOperations.find(nodeName);
            if (it == m_ParsedMenohOperations.end() ||
                dynamic_cast<ParsedConstMenohOperation<Type>*>(it->second.get()) == nullptr)
            {
                return false;
            }
            else
            {
                return true;
            }
        }

        ParsedMenohOperationPtr MenohParser::ParseBatchNormalization(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            boost::ignore_unused(graph);
            std::string name = GetNodeName(node);

            std::vector<OutputOfConstNodeDef> nodes = GetMenohInputNodes(node);
            unsigned int numInputs = static_cast<unsigned int>(nodes.size());
            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, numInputs);

            if (!HasParsedConstTensor<float>(GetNodeName(inputs[1].m_IndexedValue->GetNode()))
             || !HasParsedConstTensor<float>(GetNodeName(inputs[2].m_IndexedValue->GetNode()))
             || !HasParsedConstTensor<float>(GetNodeName(inputs[3].m_IndexedValue->GetNode()))
             || !HasParsedConstTensor<float>(GetNodeName(inputs[4].m_IndexedValue->GetNode())))
            {
                throw ParseException("only supports BatchNormalization layers with constant weights");
            }

            ITensor* input0 = inputs[0].m_IndexedValue->ResolveOutputSlot(inputs[0].m_Index);

            ParsedConstMenohOperation<float>* scaleNode =
                boost::polymorphic_downcast<ParsedConstMenohOperation<float>*>(inputs[1].m_IndexedValue);
            MenohVector<float> scaleTensorData;
            ConstTensor scaleTensor = scaleNode->GetConstTensor(scaleTensorData);

            ParsedConstMenohOperation<float>* biasNode =
                boost::polymorphic_downcast<ParsedConstMenohOperation<float>*>(inputs[2].m_IndexedValue);
            MenohVector<float> biasTensorData;
            ConstTensor biasTensor = biasNode->GetConstTensor(biasTensorData);

            ParsedConstMenohOperation<float>* meanNode =
                boost::polymorphic_downcast<ParsedConstMenohOperation<float>*>(inputs[3].m_IndexedValue);
            MenohVector<float> meanTensorData;
            ConstTensor meanTensor = meanNode->GetConstTensor(meanTensorData);

            ParsedConstMenohOperation<float>* varianceNode =
                boost::polymorphic_downcast<ParsedConstMenohOperation<float>*>(inputs[4].m_IndexedValue);
            MenohVector<float> varianceTensorData;
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
                scale = m_Network->addScale(*input0, ScaleMode::kCHANNEL, combined_bias_weights, combined_scale_weights, {});
                assert(scale);
                scale->setName(GetPrefixNodeName(node).c_str());
                std::string pname("tensor_"+name);
                scale->getOutput(0)->setName(pname.c_str());
                SetLayer(scale);
            }
            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, scale);
        }

        ParsedMenohOperationPtr MenohParser::ParseConv2D(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            boost::ignore_unused(graph);
            std::string name = GetNodeName(node);

            std::vector<OutputOfConstNodeDef> nodes = GetMenohInputNodes(node);
            unsigned int numInputs = static_cast<unsigned int>(nodes.size());
            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, numInputs);

            if (!HasParsedConstTensor<float>(GetNodeName(inputs[1].m_IndexedValue->GetNode()))
                || (numInputs == 3 && !HasParsedConstTensor<float>(GetNodeName(inputs[2].m_IndexedValue->GetNode()))))
            {
                throw ParseException("only supports Convolution layers with constant weights and biases");
            }

            ParsedConstMenohOperation<float>* weightNode =
                boost::polymorphic_downcast<ParsedConstMenohOperation<float>*>(inputs[1].m_IndexedValue);
            MenohVector<float> weightTensorData;
            ConstTensor weightTensor = weightNode->GetConstTensor(weightTensorData);

            ParsedConstMenohOperation<float>* biasNode   = 
                (numInputs == 3) ? boost::polymorphic_downcast<ParsedConstMenohOperation<float>*>(inputs[2].m_IndexedValue)
                                 : nullptr;
            MenohVector<float> biasTensorData;
            ConstTensor biasTensor = (numInputs == 3) ? biasNode->GetConstTensor(biasTensorData) : ConstTensor();

            DataType biasType = (numInputs == 3) ? biasNode->GetConstTensor(biasTensorData).GetDataType() : DataType::kFLOAT;

            std::vector<int> strides, kernel_shape, pads;
            std::tie(strides, kernel_shape, pads) = attributes_for_2d_data_processing(node);

            ITensor* input0 = inputs[0].m_IndexedValue->ResolveOutputSlot(inputs[0].m_Index);

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
            std::cout << "           input0.name  = " << input0->getName() << std::endl;
#endif

            DimsHW begin_pad{pads[0], pads[1]}, end_pad{(pads.size()<=2)? pads[0] : pads[2],
                                                        (pads.size()<=2)? pads[1] : pads[3]};
            if( (begin_pad.h() != end_pad.h()) || (begin_pad.w() != end_pad.w()) )
            {
                auto layer = m_Network->addPadding(*input0, begin_pad, end_pad );
                input0 = layer->getOutput(0);
            }

            Weights weight{weightTensor.GetDataType(), weightTensorData.data(), weightTensorData.size()};
            Weights bias{biasType, biasTensorData.data(), biasTensorData.size()};
            int nbOutputMaps = weightTensor.GetShape()[0]; 

            IConvolutionLayer* conv;
            {
                conv = m_Network->addConvolution(*input0, 
                                                  nbOutputMaps, DimsHW{kernel_shape[0], kernel_shape[1]}, weight, bias);
                assert(conv);
                conv->setName(GetPrefixNodeName(node).c_str());
                conv->setStride(DimsHW{strides[0], strides[1]});
                if( (begin_pad.h() == end_pad.h()) || (begin_pad.w() == end_pad.w()) )
                {
                    conv->setPadding(begin_pad);
                }    
                std::string pname("tensor_"+name);
                conv->getOutput(0)->setName(pname.c_str());
                SetLayer(conv);   
            }
            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, conv);
        }  
 
        ParsedMenohOperationPtr MenohParser::ParseConcat(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            boost::ignore_unused(graph);
            std::string name = GetNodeName(node);
	    
            std::vector<OutputOfConstNodeDef> nodes = GetMenohInputNodes(node);
            unsigned int numInputs = static_cast<unsigned int>(nodes.size());
            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, numInputs);

            ITensor* input0 = inputs[0].m_IndexedValue->ResolveOutputSlot(inputs[0].m_Index);

            auto axis = get<int>(node.attribute_table.at("axis"));
            axis += (axis<0) ? input0->getDimensions().nbDims : (-1);

            if( axis == 0 )
            {
                std::vector<ITensor*> itensors;
                for(unsigned int i=0 ; i<numInputs ; i++ )
                {
                    itensors.push_back(inputs[i].m_IndexedValue->ResolveOutputSlot(inputs[i].m_Index));
                }

                IConcatenationLayer* concat;
                {
                    concat = m_Network->addConcatenation(itensors.data(), itensors.size());
                    assert(concat);
                    concat->setName(GetPrefixNodeName(node).c_str());
                    concat->getOutput(0)->setName(name.c_str());
                    SetLayer(concat);
                }
                return std::make_unique<SingleLayerParsedMenohOperation>(this, node, concat);
            }
            else
            {
                throw ParseException("only supports Concat layers with legal axis");
            }
        }

        ParsedMenohOperationPtr MenohParser::ParseLrn(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            boost::ignore_unused(graph);
            std::string name = GetNodeName(node);

            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 1);
            ITensor* input0 = inputs[0].m_IndexedValue->ResolveOutputSlot(inputs[0].m_Index);

            float alpha = ReadMandatoryNodeFloatAttribute(node, "alpha");
            float beta  = ReadMandatoryNodeFloatAttribute(node, "beta");
            float k     = ReadMandatoryNodeFloatAttribute(node, "bias");
            int window  = ReadMandatoryNodeUint32Attribute(node, "depth_radius");
            window = window * 2 + 1;

            ILRNLayer* lrn;
            {
                lrn = m_Network->addLRN(*input0, window, alpha, beta, k);
                assert(lrn);
                lrn->setName(GetPrefixNodeName(node).c_str());
                lrn->getOutput(0)->setName(name.c_str());
                SetLayer(lrn);
            }
            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, lrn);
        }

        class ParsedMatMulMenohOperation : public DeferredSingleLayerParsedMenohOperation {
        public:
            ParsedMatMulMenohOperation(MenohParser* parser, const menoh_impl::node& node)
                : DeferredSingleLayerParsedMenohOperation(parser, node) {
            }

            void CreateLayerDeferred() override {
                BOOST_ASSERT(m_Layer == nullptr);
                m_Layer = m_Parser->AddFullyConnectedLayer(m_Node, nullptr, GetNodeName(m_Node).c_str());
            }
        };

        ParsedMenohOperationPtr MenohParser::ParseMatMul(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            boost::ignore_unused(graph);

            return std::make_unique<ParsedMatMulMenohOperation>(this, node);
        }

        ParsedMenohOperationPtr MenohParser::ParseSum(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            boost::ignore_unused(graph);
            std::string name = GetNodeName(node);
	    
            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 2);
            ITensor* input0 = inputs[0].m_IndexedValue->ResolveOutputSlot(inputs[0].m_Index);
            ITensor* input1 = inputs[1].m_IndexedValue->ResolveOutputSlot(inputs[1].m_Index);
#ifdef TENSORRT_DEBUG
            std::cout << "           input0.name = " << input0->getName() << std::endl;
            std::cout << "           input1.name = " << input1->getName() << std::endl;
#endif            
            IElementWiseLayer* add1;
            add1 = m_Network->addElementWise(*input0, *input1, ElementWiseOperation::kSUM);
            assert(add1);
            add1->setName(GetPrefixNodeName(node).c_str());
            add1->getOutput(0)->setName(name.c_str());
            SetLayer(add1);

            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, add1);
        }  

        ParsedMenohOperationPtr MenohParser::ParseMul(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            boost::ignore_unused(graph);
            std::string name = GetNodeName(node);
	    
            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 2);
            ITensor* input0 = inputs[0].m_IndexedValue->ResolveOutputSlot(inputs[0].m_Index);
            ITensor* input1 = inputs[1].m_IndexedValue->ResolveOutputSlot(inputs[1].m_Index);
#ifdef TENSORRT_DEBUG
            std::cout << "           input0.name = " << input0->getName() << std::endl;
            std::cout << "           input1.name = " << input1->getName() << std::endl;
#endif            
            IElementWiseLayer* mul;
            { 
                mul = m_Network->addElementWise(*input0, *input1, ElementWiseOperation::kPROD);
                assert(mul);
                mul->setName(GetPrefixNodeName(node).c_str());
                mul->getOutput(0)->setName(name.c_str());
                SetLayer(mul);
            }
            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, mul);
        }  

        ParsedMenohOperationPtr MenohParser::ParseAdd(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            boost::ignore_unused(graph);
            std::string name = GetNodeName(node);
	    
            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 2);
            ITensor* input0 = inputs[0].m_IndexedValue->ResolveOutputSlot(inputs[0].m_Index);
            ITensor* input1 = inputs[1].m_IndexedValue->ResolveOutputSlot(inputs[1].m_Index);
#ifdef TENSORRT_DEBUG
            std::cout << "           input0.name = " << input0->getName() << std::endl;
            std::cout << "           input1.name = " << input1->getName() << std::endl;
#endif            
            IElementWiseLayer* add;
            {            
                add = m_Network->addElementWise(*input0, *input1, ElementWiseOperation::kSUM);
                assert(add);
                add->setName(GetPrefixNodeName(node).c_str());
                add->getOutput(0)->setName(name.c_str());
                SetLayer(add);
            }
            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, add);
        }  

        ParsedMenohOperationPtr MenohParser::ParsePlaceholder(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            boost::ignore_unused(graph);
            std::string name = GetNodeName(node);
	    
            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 0);

            auto it = m_InputShapes.find(name);
            if (it == m_InputShapes.end())
            {
                throw ParseException("Missing input shape for Placeholder '" + name + "'");
            }

            auto dims = ReadOptionalNodeUint32ListAttribute(node, "dims");
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

                placeholder = m_Network->addInput(GetPrefixNodeName(node).c_str(), nvinfer1::DataType::kFLOAT, inputDims);
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

                scale_l = m_Network->addScale(*placeholder, ScaleMode::kUNIFORM, shift, scale, power);
                assert(scale_l);
                scale_l->setName(GetPrefixNodeName(node).c_str());
                std::string pname("tensor_"+name);
                scale_l->getOutput(0)->setName(pname.c_str());
                SetLayer(scale_l);
            }

            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, scale_l);
        }

        ParsedMenohOperationPtr MenohParser::ParseRelu(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            boost::ignore_unused(graph);

            return AddActivationLayer(node, ActivationType::kRELU);
        }

        ParsedMenohOperationPtr MenohParser::ParseSigmoid(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            boost::ignore_unused(graph);

            return AddActivationLayer(node, ActivationType::kSIGMOID);
        }

        ParsedMenohOperationPtr MenohParser::ParseTanh(const menoh_impl::node& node, const menoh_impl::graph& graph ) {
            boost::ignore_unused(graph);

            return AddActivationLayer(node, ActivationType::kTANH);
        }

        ParsedMenohOperationPtr MenohParser::AddActivationLayer(const menoh_impl::node& node, ActivationType activationType) {
            std::string name = GetNodeName(node);

            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 1);

            ITensor* input0 = inputs[0].m_IndexedValue->ResolveOutputSlot(inputs[0].m_Index);
#ifdef TENSORRT_DEBUG
            std::cout << "           input0.name = " << input0->getName() << std::endl;
#endif            
            IActivationLayer* activate = m_Network->addActivation(*input0, activationType);
            assert(activate);
            activate->setName(GetPrefixNodeName(node).c_str());
                                
            std::string pname("tensor_"+name);
            activate->getOutput(0)->setName(pname.c_str());
            SetLayer(activate);
            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, activate);
        }

        ParsedMenohOperationPtr MenohParser::ParseSoftmax(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            boost::ignore_unused(graph);
            std::string name = GetNodeName(node);

            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 1);
            ITensor* input0 = inputs[0].m_IndexedValue->ResolveOutputSlot(inputs[0].m_Index);
#ifdef TENSORRT_DEBUG
            std::cout << "           input0.name = " << input0->getName() << std::endl;
#endif            
            ISoftMaxLayer* softmax = m_Network->addSoftMax(*input0);
            assert(softmax);
            softmax->setName(GetPrefixNodeName(node).c_str());           
            std::string pname("tensor_"+name);
            softmax->getOutput(0)->setName(pname.c_str());
            SetLayer(softmax);
#ifdef TENSORRT_DEBUG
            std::cout << "           softmax.getAxes() = " << softmax->getAxes() << std::endl;
            std::cout << "           output.name = " << softmax->getOutput(0)->getName() << std::endl;
#endif            
            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, softmax);
        }

        ParsedMenohOperationPtr MenohParser::ParseMaxPool(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            boost::ignore_unused(graph);
            std::string name = GetNodeName(node);

            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 1);
            if (inputs.size() != 1)
            {
                throw ParseException("MaxPooling expects one input!");
            }
            ITensor* input0 = inputs[0].m_IndexedValue->ResolveOutputSlot(inputs[0].m_Index);

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
                auto layer = m_Network->addPadding(*input0, begin_pad, end_pad );
                input0 = layer->getOutput(0);
            }
            
            IPoolingLayer* pool;
            {
                pool = m_Network->addPooling(*input0, PoolingType::kMAX, DimsHW{kernel_shape[0], kernel_shape[1]});
                assert(pool);
                pool->setName(GetPrefixNodeName(node).c_str());
                pool->setStride(DimsHW{strides[0], strides[1]});
                if( (begin_pad.h() == end_pad.h()) && (begin_pad.w() == end_pad.w()) )
                {
                    pool->setPadding(begin_pad);
                }    
                std::string pname("tensor_"+name);
                pool->getOutput(0)->setName(pname.c_str());
                SetLayer(pool);
            } 
            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, pool);
        }          

        ParsedMenohOperationPtr MenohParser::ParseAvgPool(const menoh_impl::node& node, const menoh_impl::graph& graph) {
 
            boost::ignore_unused(graph);
            std::string name = GetNodeName(node);

            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 1);
            if (inputs.size() != 1)
            {
                throw ParseException("AveragePooling expects one input!");
            }
            ITensor* input0 = inputs[0].m_IndexedValue->ResolveOutputSlot(inputs[0].m_Index);

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
                pool = m_Network->addPooling(*input0, PoolingType::kAVERAGE, DimsHW{kernel_shape[0], kernel_shape[1]});
                assert(pool);
                pool->setName(GetPrefixNodeName(node).c_str());
                pool->setStride(DimsHW{strides[0],strides[1]});
                std::string pname("tensor_"+name);
                pool->getOutput(0)->setName(pname.c_str());
                SetLayer(pool);
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
                auto layer = m_Network->addPadding(*input0, DimsHW{ -pre_crop.d[0], -pre_crop.d[1]}, 
                                                            DimsHW{-post_crop.d[0],-post_crop.d[1]});
                assert(layer);
                layer->setName(GetPrefixNodeName(node).c_str());
                std::string pname("tensor_"+name);
                layer->getOutput(0)->setName(pname.c_str());
                SetLayer(layer);
                return std::make_unique<SingleLayerParsedMenohOperation>(this, node, layer);
            }
            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, pool);
        }

        ParsedMenohOperationPtr MenohParser::ParseGlobalMaxPool(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            boost::ignore_unused(graph);
            std::string name = GetNodeName(node);
	    
            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 1);
            if (inputs.size() != 1)
            {
              throw ParseException("GlobalMaxPooling expects one input!");
            }
            ITensor* input0 = inputs[0].m_IndexedValue->ResolveOutputSlot(inputs[0].m_Index);

#ifdef TENSORRT_DEBUG
            std::cout << "           input0.name  = " << input0->getName() << std::endl;
#endif
            IPoolingLayer* pool;
            {
                Dims dims = input0->getDimensions();
                if( dims.nbDims != 3 )
                    assert(0);
                DimsHW kernel_shape({dims.d[1], dims.d[2]});
                pool = m_Network->addPooling(*input0, PoolingType::kMAX, kernel_shape);
                assert(pool);
                pool->setName(GetPrefixNodeName(node).c_str());
                std::string pname("tensor_"+name);
                pool->getOutput(0)->setName(pname.c_str());
                SetLayer(pool);
            } 
            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, pool);
        }
        
        ParsedMenohOperationPtr MenohParser::ParseGlobalAvgPool(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            boost::ignore_unused(graph);
            std::string name = GetNodeName(node);
	    
            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 1);
            if (inputs.size() != 1)
            {
                throw ParseException("GlobalAveragePooling expects one input!");
            }
            ITensor* input0 = inputs[0].m_IndexedValue->ResolveOutputSlot(inputs[0].m_Index);

#ifdef TENSORRT_DEBUG
            std::cout << "           input0.name  = " << input0->getName() << std::endl;
#endif
            IPoolingLayer* pool;
            {
                Dims dims = input0->getDimensions();
                if( dims.nbDims != 3 )
                    throw ParseException("GlobalAvgPooling layser's input dimensions must be 3 (three).");
                DimsHW kernel_shape({dims.d[1], dims.d[2]});
                pool = m_Network->addPooling(*input0, PoolingType::kAVERAGE, kernel_shape);
                assert(pool);
                pool->setName(GetPrefixNodeName(node).c_str());
                std::string pname("tensor_"+name);
                pool->getOutput(0)->setName(pname.c_str());
                SetLayer(pool);
            } 
            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, pool);
        }

        ILayer* MenohParser::AddFullyConnectedLayer(const menoh_impl::node& matMulNodeDef, 
                                                    const menoh_impl::node* addNodeDef, const char* name){
            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(matMulNodeDef, 1);
            ITensor* input0 = inputs[0].m_IndexedValue->ResolveOutputSlot(inputs[0].m_Index);
#ifdef TENSORRT_DEBUG
            ITensor* input1 = inputs[1].m_IndexedValue->ResolveOutputSlot(inputs[1].m_Index);
            ITensor* input2 = inputs[2].m_IndexedValue->ResolveOutputSlot(inputs[2].m_Index);

            std::cout << "           input0.name = " << input0->getName() << std::endl;
            std::cout << "           input1.name = " << input1->getName() << std::endl;
            std::cout << "           input2.name = " << input2->getName() << std::endl;
#endif            
            ParsedConstMenohOperation<float>* biasNode = nullptr;
            if (addNodeDef != nullptr)
            {
                std::vector<OutputOfParsedMenohOperation> addInputs = GetInputParsedMenohOperationsChecked(*addNodeDef, 2);
                if (HasParsedConstTensor<float>(GetNodeName(addInputs[0].m_IndexedValue->GetNode())))
                {
                    biasNode = boost::polymorphic_downcast<ParsedConstMenohOperation<float>*>(addInputs[0].m_IndexedValue);
                }
                else if (HasParsedConstTensor<float>(GetNodeName(addInputs[1].m_IndexedValue->GetNode())))
                {
                    biasNode = boost::polymorphic_downcast<ParsedConstMenohOperation<float>*>(addInputs[1].m_IndexedValue);
                }
                else
                {
                    throw ParseException("only supports fully connected layers with constant bias");
                }
            }

            ParsedConstMenohOperation<float>* weightNode = nullptr;
            std::vector<OutputOfParsedMenohOperation> mulInputs = GetInputParsedMenohOperationsChecked(matMulNodeDef, 2);
            if (HasParsedConstTensor<float>(GetNodeName(mulInputs[0].m_IndexedValue->GetNode())))
            {
                weightNode = boost::polymorphic_downcast<ParsedConstMenohOperation<float>*>(mulInputs[0].m_IndexedValue);
            }
            else if (HasParsedConstTensor<float>(GetNodeName(mulInputs[1].m_IndexedValue->GetNode())))
            {
                weightNode = boost::polymorphic_downcast<ParsedConstMenohOperation<float>*>(mulInputs[1].m_IndexedValue);
            }
            else
            {
                throw ParseException("only supports fully connected layers with constant weights");
            }

            MenohVector<float> weightTensorData;
            ConstTensor weightTensor = weightNode->GetConstTensor(weightTensorData);
            Weights weight{weightTensor.GetDataType(), weightTensorData.data(), weightTensorData.size()};

            MenohVector<float> biasTensorData;
            ConstTensor biasTensor  = biasNode->GetConstTensor(biasTensorData);
            Weights bias{biasTensor.GetDataType(), biasTensorData.data(), biasTensorData.size()};

            if (weightTensor.GetShape()[1] != biasTensor.GetShape()[0])
            {
                throw ParseException("shape of matmul and bias do not match");
            }

            IFullyConnectedLayer* full;
            full = m_Network->addFullyConnected(*input0, biasTensor.GetShape()[0], weight, bias);
            assert(full);
            std::string fullname(std::string("FC:")+name);
            full->setName(fullname.c_str());
            std::string pname("tensor_"+std::string(name));
            full->getOutput(0)->setName(pname.c_str());
            SetLayer(full);

            return full;
        }

        ILayer* MenohParser::AddFullyConnectedLayer(const menoh_impl::node& node, const char* name){
            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 3);
            ITensor* input0 = inputs[0].m_IndexedValue->ResolveOutputSlot(inputs[0].m_Index);
#ifdef TENSORRT_DEBUG
            ITensor* input1 = inputs[1].m_IndexedValue->ResolveOutputSlot(inputs[1].m_Index);
            ITensor* input2 = inputs[2].m_IndexedValue->ResolveOutputSlot(inputs[2].m_Index);

            std::cout << "           input0.name = " << input0->getName() << std::endl;
            std::cout << "           input1.name = " << input1->getName() << std::endl;
            std::cout << "           input2.name = " << input2->getName() << std::endl;
#endif            
            ParsedConstMenohOperation<float>* weightNode = nullptr;
            ParsedConstMenohOperation<float>* biasNode   = nullptr;

            weightNode = boost::polymorphic_downcast<ParsedConstMenohOperation<float>*>(inputs[1].m_IndexedValue);
            biasNode   = boost::polymorphic_downcast<ParsedConstMenohOperation<float>*>(inputs[2].m_IndexedValue);

            MenohVector<float> weightTensorData;
            ConstTensor weightTensor = weightNode->GetConstTensor(weightTensorData);
            Weights weight{weightTensor.GetDataType(), weightTensorData.data(), weightTensorData.size()};
            MenohVector<float> biasTensorData;
            ConstTensor biasTensor  = biasNode->GetConstTensor(biasTensorData);
            Weights bias{biasTensor.GetDataType(), biasTensorData.data(), biasTensorData.size()};
            if (weightTensor.GetShape()[0] != biasTensor.GetShape()[0])
            {
                throw ParseException("shape of weight and bias do not match");
            }

            IFullyConnectedLayer* full;
            full = m_Network->addFullyConnected(*input0, bias.count, weight, bias);
            assert(full);
            std::string fullname(std::string("FC:")+name);
            full->setName(fullname.c_str());
            std::string pname("tensor_"+std::string(name));
            full->getOutput(0)->setName(pname.c_str());
            SetLayer(full);

            return full;
        }

        ParsedMenohOperationPtr MenohParser::ParseGemm(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            boost::ignore_unused(graph);
            std::string name = GetNodeName(node);
            
            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 3);
            ITensor* input0 = inputs[0].m_IndexedValue->ResolveOutputSlot(inputs[0].m_Index);
#ifdef TENSORRT_DEBUG
            ITensor* input1 = inputs[1].m_IndexedValue->ResolveOutputSlot(inputs[1].m_Index);
            ITensor* input2 = inputs[2].m_IndexedValue->ResolveOutputSlot(inputs[2].m_Index);

            std::cout << "           input0.name = " << input0->getName() << std::endl;
            std::cout << "           input1.name = " << input1->getName() << std::endl;
            std::cout << "           input2.name = " << input2->getName() << std::endl;
#endif            
            ParsedConstMenohOperation<float>* weightNode = nullptr;
            ParsedConstMenohOperation<float>* biasNode   = nullptr;

            weightNode = boost::polymorphic_downcast<ParsedConstMenohOperation<float>*>(inputs[1].m_IndexedValue);
            biasNode   = boost::polymorphic_downcast<ParsedConstMenohOperation<float>*>(inputs[2].m_IndexedValue);

            MenohVector<float> weightTensorData;
            ConstTensor weightTensor = weightNode->GetConstTensor(weightTensorData);
            Weights weight{weightTensor.GetDataType(), weightTensorData.data(), weightTensorData.size()};
            MenohVector<float> biasTensorData;
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
            full = m_Network->addFullyConnected(*input0, bias.count, weight, bias);
            assert(full);
            std::string fullname(std::string("Gemm:")+name);
            full->setName(fullname.c_str());
            std::string pname("tensor_"+std::string(name));
            full->getOutput(0)->setName(pname.c_str());
            SetLayer(full);

            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, full);
        }

        ParsedMenohOperationPtr MenohParser::ParseUnsqueeze(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            boost::ignore_unused(graph);
            std::string name = GetNodeName(node);
            
            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 1);
            ITensor* input0 = inputs[0].m_IndexedValue->ResolveOutputSlot(inputs[0].m_Index);
#ifdef TENSORRT_DEBUG
            std::cout << "           input0.name = " << input0->getName() << std::endl;
#endif            
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
            shuffle = m_Network->addShuffle(*input0);
            assert(shuffle);
            shuffle->setReshapeDimensions(new_shape);
            std::string fullname(std::string("Unsqueeze:")+name);
            shuffle->setName(fullname.c_str());
            std::string pname("tensor_"+std::string(name));
            shuffle->getOutput(0)->setName(pname.c_str());
            SetLayer(shuffle);

            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, shuffle);
        }

        void MenohParser::LoadNode(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            std::string name = GetNodeName(node);
	    
#ifdef TENSORRT_DEBUG
            std::cout << std::endl << "    LoadNode(" << node.op_type << ") : " << name << std::endl;
#endif            
            const std::string& operation = node.op_type;
            auto it = ms_OperationNameToParsingFunctions.find(operation);
            if (it != ms_OperationNameToParsingFunctions.end())
            {
                auto func = it->second;
                ParsedMenohOperationPtr parsedMenohOperation = (this->*func)(node, graph);
                auto it = m_ParsedMenohOperations.find(name);
                if (it != m_ParsedMenohOperations.end())
                {
                    throw ParseException(boost::str(boost::format("Name %1% used by more than one node") % name));
                }

                m_ParsedMenohOperations[name] = std::move(parsedMenohOperation);
                std::string output_name = node.output_name_list[0];

                if (std::find(m_RequestedOutputs.begin(), m_RequestedOutputs.end(), output_name) !=
                    m_RequestedOutputs.end())
                {
                }
            }
            else
            {
                throw ParseException(boost::str(
                    boost::format("Unsupported operation %1% in Menoh::graph") % operation));
            }
        }

        void MenohParser::LoadGraph(const menoh_impl::graph& graph,
                                    std::unordered_map<std::string, array> const& parameter_table) {

            m_NodesByName.clear();
            m_ParamByName.clear();
            m_NetworkInputsBindingInfo.clear();
            m_NetworkOutputsBindingInfo.clear();

            for( unsigned int i=0; i<graph.node_list().size() ; ++i)
            {
                const menoh_impl::node& my_node = graph.node_list().at(i);
                m_NodesByName[GetNodeName(my_node)] = &my_node;
            }

            for( auto param : parameter_table )
            {
                auto arr = param.second;
                array param_arr(arr.dtype(), std::move(arr.dims()), std::move(arr.data()));
                m_ParamByName[param.first] = param_arr;
            }

            std::vector<const menoh_impl::node*> targetNodes;
            for (const std::string& requestedOutputName : m_RequestedOutputs)
            {
                bool found = false;

#ifdef TENSORRT_DEBUG
                std::cout << "requestedOutputName = " << requestedOutputName << std::endl;
#endif
                for( unsigned int i=0; i<graph.node_list().size(); ++i)
                {
                    const menoh_impl::node& node = graph.node_list().at(i);
                     
                    auto nodeIt = std::find(node.output_name_list.begin(), node.output_name_list.end(), requestedOutputName);
                    if (nodeIt != node.output_name_list.end())
                    {
                        targetNodes.push_back(&node);
                        found = true;
                        break;
                    }
                }
                if( !found )
                    throw ParseException("Couldn't find requested output node '" + requestedOutputName + "' in graph");
            }

            for( auto node : targetNodes )
                m_RequestedOutputs.push_back(GetNodeName(*node));
	    
            std::vector<const menoh_impl::node*> sortedNodes;
            if (!Util::GraphTopologicalSort<const menoh_impl::node*>(
                targetNodes,
                [this](const menoh_impl::node* node)
                {
                    auto outputs = GetMenohInputNodes(*node);
                    std::vector<const menoh_impl::node*> nodesOnly;
                    for (const auto & o : outputs) {
                        nodesOnly.push_back(o.m_IndexedValue);
                    }
                    return nodesOnly;
                },
                sortedNodes))
            {
                throw ParseException("Cycle detected in graph");
            }

            for (const auto& it : sortedNodes)
            {
                LoadNode(*it, graph);
            }

#ifdef TENSORRT_DEBUG
            std::cout << "markOutput.node   = " << m_Layer->getName() << std::endl;
            std::cout << "markOutput.output = " << m_Layer->getOutput(0)->getName() << std::endl;
#endif
            m_Network->markOutput(*m_Layer->getOutput(0));
        }

        INetworkDefinition* MenohParser::CreateNetworkFromGraph(
                                         IBuilder* builder,
                                         const menoh_impl::graph& graph,
                                         std::unordered_map<std::string, array> const& parameter_table,
                                         const std::map<std::string, TensorShape>& inputShapes,
                                         const std::vector<std::string>& requestedOutputs){
            m_Network = builder->createNetwork();
            assert(m_Network);

            m_InputShapes = inputShapes;
            if (requestedOutputs.size() == 0)
            {
                throw ParseException("requestedOutputs must have at least one entry");
            }
            m_RequestedOutputs = requestedOutputs;
            try
            {
                LoadGraph(graph, parameter_table);
            }
            catch (const ParseException& e)
            {
                Cleanup();
                throw e;
            }

            return m_Network;
        }

        void MenohParser::Cleanup(){
            m_InputShapes.clear();
            m_RequestedOutputs.clear();
            m_NodesByName.clear();
            m_ParamByName.clear();
            m_ParsedMenohOperations.clear();
        }  

        const std::map<std::string, MenohParser::OperationParsingFunction> MenohParser::ms_OperationNameToParsingFunctions = {
          { "Const",                 &MenohParser::ParseConst },
          { "FC",                    &MenohParser::ParseFC },
          { "Gemm",                  &MenohParser::ParseGemm },
          { "Unsqueeze",             &MenohParser::ParseUnsqueeze },
          { "Identity",              &MenohParser::ParseIdentity },
          { "Sum",                   &MenohParser::ParseSum },
          { "BatchNormalization",    &MenohParser::ParseBatchNormalization },
          { "Conv",                  &MenohParser::ParseConv2D },
          { "Concat",                &MenohParser::ParseConcat },
          { "LRN",                   &MenohParser::ParseLrn },
          { "MatMul",                &MenohParser::ParseMatMul },
          { "Mul",                   &MenohParser::ParseMul },
          { "Add",                   &MenohParser::ParseAdd },
          { "Placeholder",           &MenohParser::ParsePlaceholder },
          { "Relu",                  &MenohParser::ParseRelu },
          { "Sigmoid",               &MenohParser::ParseSigmoid },
          { "Softmax",               &MenohParser::ParseSoftmax },
          { "Tanh",                  &MenohParser::ParseTanh },
          { "MaxPool",               &MenohParser::ParseMaxPool },
          { "AveragePool",           &MenohParser::ParseAvgPool },
          { "GlobalMaxPool",         &MenohParser::ParseGlobalMaxPool },
          { "GlobalAveragePool",     &MenohParser::ParseGlobalAvgPool },
        };

    } // namespace tensorrt_backend
} // namespace menoh_impl
