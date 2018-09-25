
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
#include <menoh/tensorrt/MenohParser.hpp>
#include <menoh/tensorrt/TensorRTUtil.hpp>

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

            ITensor* ResolveTensorRTOutputSlot(unsigned int MenohOutputIndex) override
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

        /// A SingleLayerParsedMenohOperation for deferred layer creation
        class DeferredSingleLayerParsedMenohOperation : public SingleLayerParsedMenohOperation {
        public:
            DeferredSingleLayerParsedMenohOperation(MenohParser* parser, const menoh_impl::node& node)
            : SingleLayerParsedMenohOperation(parser, node, nullptr)
            {
            }

            ITensor* ResolveTensorRTOutputSlot(unsigned int MenohOutputIndex) override
            {
                if (!m_Layer)
                {
                    CreateLayerDeferred();
                }
                return SingleLayerParsedMenohOperation::ResolveTensorRTOutputSlot(MenohOutputIndex);
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
                // For some reason const node can have "Control Inputs". We ignore them for now.
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
            // Fetch the tensorflow nodes connected as inputs and validate the size.
            std::vector<OutputOfConstNodeDef> nodes = GetMenohInputNodes(node);
            const std::size_t numInputs = node.input_name_list.size();
            if (numInputs != expectedNumInputs)
            {
                throw ParseException(boost::str(boost::format("Unexpected number of inputs for node %1%. "
							      "Expected %2%, found %3%") % name % expectedNumInputs % numInputs));
            }
            // Fetch the corresponding ParsedMenohOperation operations
            std::vector<OutputOfParsedMenohOperation> result;
            for (auto&& node : nodes)
            {
                auto it = m_ParsedMenohOperations.find(GetNodeName(*(node.m_IndexedValue)));
                if (it == m_ParsedMenohOperations.end())
                {
                    throw ParseException("Node with name '" + GetNodeName(*(node.m_IndexedValue)) + "' has not been parsed");
                }
                ParsedMenohOperation* parsedOp = it->second.get();
                // Transparently 'skip' any Identity operations. This simplifies the logic inside the ParseXXX() functions.
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

        /// An ParsedMenohOperation which forwards to another (used for Identity nodes).
        class ParsedIdentityMenohOperation : public ParsedMenohOperation {
        public:
            ParsedIdentityMenohOperation(MenohParser* parser, const menoh_impl::node& node, ParsedMenohOperation* representative)
                : ParsedMenohOperation(parser, node)
                , m_Representative(representative)
            {
            }

            virtual ITensor* ResolveTensorRTOutputSlot(unsigned int MenohOutputIndex) override
            {
                BOOST_ASSERT(m_Representative);
                return m_Representative->ResolveTensorRTOutputSlot(MenohOutputIndex);
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
            // Any requests for the output slots of this node should be forwarded to the node connected as input.
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

        /// An ParsedMenohOperation for a Const node.
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

                // Update the result to point to the user provided storage
                ConstTensor constTensor(outInfo, outputTensorData);
                return constTensor;
            }

            ConstTensor GetConstTensor(bool swizzleForConvolutionWeights, std::vector<T>& outputTensorData)
            {
                const TensorInfo outInfo = m_TensorInfo;

                outputTensorData.resize(m_TensorInfo.GetNumElements());
                memcpy(outputTensorData.data(), m_Storage.data(), m_TensorInfo.GetNumBytes());

                // Update the result to point to the user provided storage
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
                // If there are no entries in the list, perform no action
                if (numSrcElements == 0)
                {
                    return;
                }

                // If no size was provided, use the length of the value list
                if (numDstElements == 0)
                {
                    numDstElements = numSrcElements;
                }

                // Allocate memory
                dstData.resize(std::max(numSrcElements, numDstElements) * sizeof(DataType));

                const DataType* srcTensor = reinterpret_cast<const DataType*>(srcData);
                DataType* dstTensor = reinterpret_cast<DataType*>(dstData.data());

                // Copy the value list entries into the destination
                std::copy(srcTensor, srcTensor + numSrcElements, dstTensor);

                if (numDstElements > numSrcElements)
                {
                    // Use the last element in the list to fill the remaining entries
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

        ParsedMenohOperationPtr MenohParser::ParseConv2D(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            boost::ignore_unused(graph);
            std::string name = GetNodeName(node);
            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 3);

            if (!HasParsedConstTensor<float>(GetNodeName(inputs[1].m_IndexedValue->GetNode()))
             || !HasParsedConstTensor<float>(GetNodeName(inputs[2].m_IndexedValue->GetNode())))
            {
                throw ParseException("TensorRT only supports Convolution layers with constant weights and biases");
            }

            ParsedConstMenohOperation<float>* weightNode =
                boost::polymorphic_downcast<ParsedConstMenohOperation<float>*>(inputs[1].m_IndexedValue);
            ParsedConstMenohOperation<float>* biasNode   = 
                boost::polymorphic_downcast<ParsedConstMenohOperation<float>*>(inputs[2].m_IndexedValue);
            
            MenohVector<float> weightTensorData;
            ConstTensor weightTensor = weightNode->GetConstTensor(weightTensorData);
            MenohVector<float> biasTensorData;
            ConstTensor biasTensor   = biasNode->GetConstTensor(biasTensorData);

            std::vector<int> strides, kernel_shape, pads;
            std::tie(strides, kernel_shape, pads) = attributes_for_2d_data_processing(node);


#ifdef TENSORRT_DEBUG
	    std::cout << "     weight(" << weightTensor.GetNumDimensions() << ") = "
		      << weightTensor.GetShape()[0] << ", " << weightTensor.GetShape()[1];
	    std::cout << ", " << weightTensor.GetShape()[2] << ", " << weightTensor.GetShape()[3] << std::endl;
	    std::cout << "       bias(" << biasTensor.GetNumDimensions() << ") = " << biasTensor.GetShape()[0] << std::endl;
#endif
            
            ITensor* input0 = inputs[0].m_IndexedValue->ResolveTensorRTOutputSlot(inputs[0].m_Index);
            Weights weight{weightTensor.GetDataType(), weightTensorData.data(), weightTensorData.size()};
            Weights bias{biasTensor.GetDataType(), biasTensorData.data(), biasTensorData.size()};

            IConvolutionLayer* conv1;
            conv1 = m_Network->addConvolution(*input0, 
                                              bias.count, DimsHW{kernel_shape[0], kernel_shape[1]}, weight, bias);
            assert(conv1);
            conv1->setName(name.c_str());
            conv1->setStride(DimsHW{strides[0], strides[1]});
            conv1->setPadding(DimsHW{pads[0], pads[1]});
            std::string pname("tensor_"+name);
            conv1->getOutput(0)->setName(pname.c_str());
            SetLayer(conv1);   
            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, conv1);
        }  
 
        ParsedMenohOperationPtr MenohParser::ParseConcat(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            boost::ignore_unused(graph);
            std::string name = GetNodeName(node);
	    
            std::vector<OutputOfConstNodeDef> nodes = GetMenohInputNodes(node);
            // In tensorflow, we have the last input of the Concat layer as the axis for concatenation
            unsigned int numInputs = static_cast<unsigned int>(nodes.size());
            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, numInputs);

            std::vector<ITensor*> itensors(numInputs);
            for(unsigned int i=0 ; i<numInputs ; i++ )
            {
                itensors.push_back(inputs[i].m_IndexedValue->ResolveTensorRTOutputSlot(inputs[i].m_Index));
            }
            ITensor** tensors = itensors.data();
            IConcatenationLayer* concat = m_Network->addConcatenation(tensors, numInputs);
            assert(concat);
            concat->setName(name.c_str());
            concat->getOutput(0)->setName(name.c_str());
            SetLayer(concat);

            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, concat);
        }

        ParsedMenohOperationPtr MenohParser::ParseLrn(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            boost::ignore_unused(graph);
            std::string name = GetNodeName(node);

            float alpha = ReadMandatoryNodeFloatAttribute(node, "alpha");
            float beta  = ReadMandatoryNodeFloatAttribute(node, "beta");
            float k     = ReadMandatoryNodeFloatAttribute(node, "bias");

            int window = ReadMandatoryNodeUint32Attribute(node, "depth_radius");
            // The window size must be an odd value. For a window size of (2 * n + 1), TensorFlow defines depth_radius = n.
            window = window * 2 + 1;

            ILRNLayer* lrn = m_Network->addLRN(*m_Layer->getOutput(0), window, alpha, beta, k);
            assert(lrn);
            lrn->setName(name.c_str());
            lrn->getOutput(0)->setName(name.c_str());
            SetLayer(lrn);
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

        ParsedMenohOperationPtr MenohParser::ParseMul(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            boost::ignore_unused(graph);
            std::string name = GetNodeName(node);
	    
            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 2);
            ITensor* input0 = inputs[0].m_IndexedValue->ResolveTensorRTOutputSlot(inputs[0].m_Index);
            ITensor* input1 = inputs[1].m_IndexedValue->ResolveTensorRTOutputSlot(inputs[1].m_Index);
#ifdef TENSORRT_DEBUG
            std::cout << "           input0.name = " << input0->getName() << std::endl;
            std::cout << "           input1.name = " << input1->getName() << std::endl;
#endif            
            IMatrixMultiplyLayer* matrix1;
            matrix1 = m_Network->addMatrixMultiply(*input0, false, *input1, false);
            assert(matrix1);
            matrix1->setName(name.c_str());
            matrix1->getOutput(0)->setName(name.c_str());
            SetLayer(matrix1);
            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, matrix1);
        }  

        void PrintDims(Dims dims) {
            std::cout << "nbDims = " << dims.nbDims << std::endl;
            for( int i=0 ; i<dims.nbDims ; i++ ){
  	        std::cout << "d[" << i << "] = " << dims.d[i] << std::endl;
  	        std::cout << "type[" << i << "] = " << (int)dims.type[i] << std::endl;
            }
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
                //int offset = 0;
                inputDims.nbDims = dims.size() - offset ;
                for( unsigned int i=offset ; i<dims.size() ; i++ )
                    inputDims.d[i-offset] = dims[i];  

                placeholder = m_Network->addInput(name.c_str(), nvinfer1::DataType::kFLOAT, inputDims);
                assert(placeholder);
#ifdef TENSORRT_DEBUG
                std::cout << "           inputDims.nbDims = " << inputDims.nbDims;
                for( int i=0 ; i<inputDims.nbDims ; i++ )
                    std::cout << std::endl << "           inputDims.d[" << i << "] = " << inputDims.d[i];
                std::cout << std::endl; 
#endif
            }

            IScaleLayer* scale1;
            {
                const float powerParam = 1;
                const float scaleParam = 1;
                const Weights power{DataType::kFLOAT, nullptr, 0};
                const Weights shift{DataType::kFLOAT, nullptr, 0};
                const Weights scale{DataType::kFLOAT, nullptr, 0};

                scale1 = m_Network->addScale(*placeholder, ScaleMode::kUNIFORM, shift, scale, power);
                assert(scale1);
                scale1->setName(name.c_str());           

                std::string pname("tensor_"+name);
                scale1->getOutput(0)->setName(pname.c_str());
                SetLayer(scale1);
            }

            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, scale1);
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

            ITensor* input0 = inputs[0].m_IndexedValue->ResolveTensorRTOutputSlot(inputs[0].m_Index);
#ifdef TENSORRT_DEBUG
            std::cout << "           input0.name = " << input0->getName() << std::endl;
#endif            
            IActivationLayer* activate = m_Network->addActivation(*input0, activationType);
            assert(activate);
            activate->setName(name.c_str());

            std::string pname("tensor_"+name);
            activate->getOutput(0)->setName(pname.c_str());
            SetLayer(activate);
            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, activate);
        }

        ParsedMenohOperationPtr MenohParser::ParseSoftmax(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            boost::ignore_unused(graph);
            std::string name = GetNodeName(node);

            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 1);
            ITensor* input0 = inputs[0].m_IndexedValue->ResolveTensorRTOutputSlot(inputs[0].m_Index);
#ifdef TENSORRT_DEBUG
            std::cout << "           input0.name = " << input0->getName() << std::endl;
#endif            
            ISoftMaxLayer* softmax = m_Network->addSoftMax(*input0);
            assert(softmax);
            softmax->setName(name.c_str());           

            std::string pname("tensor_"+name);
            softmax->getOutput(0)->setName(pname.c_str());
#ifdef TENSORRT_DEBUG
            std::cout << "           softmax.getAxes() = " << softmax->getAxes() << std::endl;
            std::cout << "           output.name = " << softmax->getOutput(0)->getName() << std::endl;
#endif            
            SetLayer(softmax);
            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, softmax);
        }

        ParsedMenohOperationPtr MenohParser::ParseMaxPool(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            return ParsePooling2d(node, graph, PoolingType::kMAX);
        }

        ParsedMenohOperationPtr MenohParser::ParseAvgPool(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            return ParsePooling2d(node, graph, PoolingType::kAVERAGE);
        }          

        ParsedMenohOperationPtr MenohParser::ParsePooling2d(const menoh_impl::node& node, const menoh_impl::graph& graph, 
                                                            PoolingType pooltype){
            boost::ignore_unused(graph);
            std::string name = GetNodeName(node);
	    
            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 1);
            if (inputs.size() != 1)
            {
                throw ParseException("2D Pooling expects one input!");
            }

            std::vector<int> strides, kernel_shape, pads;
            std::tie(strides, kernel_shape, pads) = attributes_for_2d_data_processing(node);
            
#ifdef TENSORRT_DEBUG
            std::cout << "           strides      = " << strides[0]      << ", " << strides[1]      << std::endl;
            std::cout << "           kernel_shape = " << kernel_shape[0] << ", " << kernel_shape[1] << std::endl;
            std::cout << "           pads         = " << pads[0]         << ", " << pads[1]         << std::endl;
#endif            

            ITensor* input0 = inputs[0].m_IndexedValue->ResolveTensorRTOutputSlot(inputs[0].m_Index);
#ifdef TENSORRT_DEBUG
            std::cout << "           input0.name  = " << input0->getName() << std::endl;
#endif

            IPoolingLayer* pool;
            pool = m_Network->addPooling(*input0, pooltype, DimsHW{kernel_shape[0], kernel_shape[1]});
            assert(pool);
            pool->setName(name.c_str());
            pool->setStride( DimsHW{strides[0], strides[1]});
            pool->setPadding(DimsHW{pads[0],    pads[1]});

            std::string pname("tensor_"+name);
            pool->getOutput(0)->setName(pname.c_str());
            SetLayer(pool);
            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, pool);
        }

        ILayer* MenohParser::AddFullyConnectedLayer(const menoh_impl::node& matMulNodeDef, 
                                                    const menoh_impl::node* addNodeDef, const char* name){
            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(matMulNodeDef, 1);
            ITensor* input0 = inputs[0].m_IndexedValue->ResolveTensorRTOutputSlot(inputs[0].m_Index);
#ifdef TENSORRT_DEBUG
            ITensor* input1 = inputs[1].m_IndexedValue->ResolveTensorRTOutputSlot(inputs[1].m_Index);
            ITensor* input2 = inputs[2].m_IndexedValue->ResolveTensorRTOutputSlot(inputs[2].m_Index);

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
                    throw ParseException("TensorRT only supports fully connected layers with constant bias");
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
                throw ParseException("TensorRT only supports fully connected layers with constant weights");
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
            full->setName(name);

            std::string pname("tensor_"+std::string(name));
            full->getOutput(0)->setName(pname.c_str());
            SetLayer(full);
            return full;
        }

        ILayer* MenohParser::AddFullyConnectedLayer(const menoh_impl::node& node, const char* name){
            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 3);
            ITensor* input0 = inputs[0].m_IndexedValue->ResolveTensorRTOutputSlot(inputs[0].m_Index);
#ifdef TENSORRT_DEBUG
            ITensor* input1 = inputs[1].m_IndexedValue->ResolveTensorRTOutputSlot(inputs[1].m_Index);
            ITensor* input2 = inputs[2].m_IndexedValue->ResolveTensorRTOutputSlot(inputs[2].m_Index);

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
            full->setName(name);
            std::string pname("tensor_"+std::string(name));
            full->getOutput(0)->setName(pname.c_str());
            SetLayer(full);
            return full;
        }

        void MenohParser::LoadNode(const menoh_impl::node& node, const menoh_impl::graph& graph) {
            std::string name = GetNodeName(node);
	    
            // get the type of the node (assume float)

            const std::string& operation = node.op_type;
            auto it = ms_OperationNameToParsingFunctions.find(operation);
            if (it != ms_OperationNameToParsingFunctions.end())
            {
                auto func = it->second;
                ParsedMenohOperationPtr parsedMenohOperation = (this->*func)(node, graph);
                // Store the parsed operation so that dependent layers can connect to it
                auto it = m_ParsedMenohOperations.find(name);
                if (it != m_ParsedMenohOperations.end())
                {
                    throw ParseException(boost::str(boost::format("Name %1% used by more than one node") % name));
                }

                m_ParsedMenohOperations[name] = std::move(parsedMenohOperation);
                std::string output_name = node.output_name_list[0];

                // If this node was requested as an output from the network then trackOutputBinding
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

            // add all nodes to our map
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

            // Find the output nodes the user requested
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
	    
            // Sort them into a linear ordering such that all inputs of a node are before the node itself
            std::vector<const menoh_impl::node*> sortedNodes;
            if (!tensorRTUtil::GraphTopologicalSort<const menoh_impl::node*>(
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
            // cleanup, in case we reuse this parser
            m_InputShapes.clear();
            m_RequestedOutputs.clear();
            m_NodesByName.clear();
            m_ParamByName.clear();
            m_ParsedMenohOperations.clear();
        }  

        const std::map<std::string, MenohParser::OperationParsingFunction> MenohParser::ms_OperationNameToParsingFunctions = {
          { "Const",                 &MenohParser::ParseConst },
          { "FC",                    &MenohParser::ParseFC },
          { "Identity",              &MenohParser::ParseIdentity },
          { "Conv",                  &MenohParser::ParseConv2D },
          { "ConcatV2",              &MenohParser::ParseConcat },
          { "LRN",                   &MenohParser::ParseLrn },
          { "MatMul",                &MenohParser::ParseMatMul },
          { "Mul",                   &MenohParser::ParseMul },
          { "Placeholder",           &MenohParser::ParsePlaceholder },
          { "Relu",                  &MenohParser::ParseRelu },
          { "Sigmoid",               &MenohParser::ParseSigmoid },
          { "Softmax",               &MenohParser::ParseSoftmax },
          { "Tanh",                  &MenohParser::ParseTanh },
          { "MaxPool",               &MenohParser::ParseMaxPool },
          { "AvgPool",               &MenohParser::ParseAvgPool },
        };

    } // namespace tensorrt_backend
} // namespace menoh_impl
