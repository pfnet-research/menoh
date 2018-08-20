
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

#include <armnn/ArmNN.hpp>

#include <armnnUtils/Permute.hpp>
#include <armnnUtils/GraphTopologicalSort.hpp>

#include <menoh/arm/MenohParser.hpp>

using namespace armnn;

namespace menoh_impl {
    namespace armnn_backend {

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
	
        const armnn::PermutationVector NHWCToArmNN = { 0, 2, 3, 1 };
        const armnn::PermutationVector ArmNNToNHWC = { 0, 3, 1, 2 };

        IConnectableLayer* AddSwizzleLayer(INetwork& network, IOutputSlot& input, const PermutationVector& mapping,
            const std::string& name){
            // Add swizzle layer
            IConnectableLayer* const layer = network.AddPermuteLayer(mapping, name.c_str());

            // Connect intput to swizzle layer
            input.Connect(layer->GetInputSlot(0));

            // Setup swizzled output
            const TensorInfo outInfo = armnnUtils::Permuted(input.GetTensorInfo(), mapping);
            layer->GetOutputSlot(0).SetTensorInfo(outInfo);

            return layer;
        }

        IConnectableLayer* SwizzleInDeswizzleOut(INetwork& network, IOutputSlot& input, IConnectableLayer& layer,
            const std::string& name){
            // Add swizzle layer
            IConnectableLayer* const swizzleLayer = AddSwizzleLayer(network, input, NHWCToArmNN, "swizzle_for-" + name);

            // Connect swizzledInput to layer
            swizzleLayer->GetOutputSlot(0).Connect(layer.GetInputSlot(0));

            // Add deswizzle layer
            IConnectableLayer* const deswizzleLayer = AddSwizzleLayer(network, layer.GetOutputSlot(0), ArmNNToNHWC,
                "deswizzle_for-" + name);

            return deswizzleLayer;
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

        TensorInfo PrepareReshape(const TensorInfo& input, const std::vector<int32_t>& targetDims) {
            std::vector<unsigned int> outDims(targetDims.begin(), targetDims.end());
            const auto stretchDim = std::find(targetDims.begin(), targetDims.end(), -1);

            if (stretchDim != targetDims.end())
            {
                if (std::find(std::next(stretchDim), targetDims.end(), -1) != targetDims.end())
                {
                    throw ParseException("At most one component of shape can be -1");
                }

                auto targetNumElements = boost::numeric_cast<unsigned int>(
					     std::accumulate(targetDims.begin(), targetDims.end(),
                                             -1, std::multiplies<int32_t>()));
                auto stretchIndex = static_cast<size_t>(std::distance(targetDims.begin(), stretchDim));
                outDims[stretchIndex] = input.GetNumElements() / targetNumElements;
            }

            TensorInfo reshapeInfo = input;
            reshapeInfo.SetShape(TensorShape{ static_cast<unsigned int>(outDims.size()), outDims.data() });

            return reshapeInfo;
        }

        // We need the input0Slot to guide the reshape for input1Slot
        IOutputSlot* BroadcasMenohorAddandMul(IOutputSlot* input0Slot, IOutputSlot* input1Slot,
                                              bool isNHWC, INetwork& m_Network,
                                              const menoh_impl::node& node){
            const TensorInfo& input1Info = input1Slot->GetTensorInfo();
            const TensorInfo inputTensorInfo = input0Slot->GetTensorInfo();
            const unsigned int matchDim = inputTensorInfo.GetNumDimensions() - (isNHWC ? 1 : 3);
            std::array<unsigned int, MaxNumOfTensorDimensions> reshapedDimensions;
            std::fill_n(reshapedDimensions.begin(), inputTensorInfo.GetNumDimensions(), 1);
            reshapedDimensions[matchDim] = input1Info.GetShape()[0];

            armnn::TensorInfo reshapedInfo = input1Info;
            reshapedInfo.SetShape(TensorShape{ inputTensorInfo.GetNumDimensions(), reshapedDimensions.data() });

            const std::string reshapeLayerName = "reshape_for-" + GetNodeName(node);
            ReshapeDescriptor reshapeDesc;
            reshapeDesc.m_TargetShape = reshapedInfo.GetShape();
            IConnectableLayer* const reshapeLayer = m_Network.AddReshapeLayer(reshapeDesc, reshapeLayerName.c_str());

            input1Slot->Connect(reshapeLayer->GetInputSlot(0));
            reshapeLayer->GetOutputSlot(0).SetTensorInfo(reshapedInfo);

            input1Slot = &reshapeLayer->GetOutputSlot(0);

            return input1Slot;
        }

        inline void CalculateSamePadding(uint32_t inputSize, uint32_t stride,
                                        uint32_t filterSize, bool samePadding,
                                        uint32_t* paddingFront, uint32_t* paddingBack) {
            *paddingFront = 0;
            *paddingBack = 0;

            if (samePadding) {
                uint32_t outputSize = (inputSize + stride - 1) / stride;
                uint32_t temp = (outputSize - 1) * stride + filterSize;
                if (temp > inputSize) {
                    *paddingFront = (temp - inputSize) / 2;
                    *paddingBack = (temp - inputSize) - *paddingFront;
                }
            }
        }

        void CalcPadding(uint32_t input, uint32_t kernel, uint32_t stride, uint32_t& outPadHead, uint32_t& outPadTail,
                        bool samePadding){
            CalculateSamePadding(input, stride, kernel, samePadding, &outPadHead, &outPadTail);
        }

        /// An ParsedMenohOperation where the Armnn equivalent is a single layer,
        /// with output slots that correspond directly to the Menoh node outputs.
        class SingleLayerParsedMenohOperation : public ParsedMenohOperation{
        public:
            SingleLayerParsedMenohOperation(MenohParser* parser, const menoh_impl::node& node, IConnectableLayer* layer)
            : ParsedMenohOperation(parser, node)
            , m_Layer(layer)
            {
            }

            IOutputSlot& ResolveArmnnOutputSlot(unsigned int MenohOutputIndex) override
            {
                BOOST_ASSERT(m_Layer);
                // Assume one-to-one mapping between Menoh and armnn output slots.
                unsigned int armnnOutputSlotIdx = MenohOutputIndex;
                if (armnnOutputSlotIdx >= m_Layer->GetNumOutputSlots())
                {
                    throw ParseException(
                        boost::str(boost::format("The requested output slot #%1% "
                            "for %2% does not exist") % armnnOutputSlotIdx % m_Layer->GetName()));
                }
                return m_Layer->GetOutputSlot(armnnOutputSlotIdx);
            }

        protected:
            IConnectableLayer* m_Layer;
        };

        /// A SingleLayerParsedMenohOperation for deferred layer creation
        class DeferredSingleLayerParsedMenohOperation : public SingleLayerParsedMenohOperation {
        public:
            DeferredSingleLayerParsedMenohOperation(MenohParser* parser, const menoh_impl::node& node)
            : SingleLayerParsedMenohOperation(parser, node, nullptr)
            {
            }

            IOutputSlot& ResolveArmnnOutputSlot(unsigned int MenohOutputIndex) override
            {
                if (!m_Layer)
                {
                    CreateLayerDeferred();
                }
                return SingleLayerParsedMenohOperation::ResolveArmnnOutputSlot(MenohOutputIndex);
            }

        private:
            virtual void CreateLayerDeferred() = 0;
        };
         

        MenohParser::MenohParser()
            : m_Network(nullptr, nullptr){
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

#ifdef ARM_DEBUG
	    std::cout << std::endl << " [node] : " << node.op_type << " , " << name << std::endl;
	    for( unsigned int j=0; j<node.input_name_list.size(); ++j )
		std::cout << "    input : " << node.input_name_list.at(j) << std::endl;

            for( unsigned int j=0; j<node.output_name_list.size(); ++j )
		std::cout << "   output : " << node.output_name_list.at(j) << std::endl;
#endif
            return result;
        }  

        ParsedMenohOperationPtr MenohParser::ParseAdd(const menoh_impl::node& node, const menoh_impl::graph& graph) {
	    boost::ignore_unused(graph);
	    std::string name = GetNodeName(node);
	    
	    std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 2);

            // If one of the inputs is a MatMul and the other is a const, then we handle both nodes together as FullyConnected
            if (inputs[0].m_IndexedValue->GetNode().op_type == "MatMul" &&
                HasParsedConstTensor<float>(GetNodeName(inputs[1].m_IndexedValue->GetNode())))
            {
                IConnectableLayer* layer =
                    AddFullyConnectedLayer(inputs[0].m_IndexedValue->GetNode(), &node, name.c_str());
                return std::make_unique<SingleLayerParsedMenohOperation>(this, node, layer);
            }
            else if (HasParsedConstTensor<float>(GetNodeName(inputs[0].m_IndexedValue->GetNode())) &&
                                                 inputs[1].m_IndexedValue->GetNode().op_type == "MatMul")
            {
                IConnectableLayer* layer =
                    AddFullyConnectedLayer(inputs[1].m_IndexedValue->GetNode(), &node, name.c_str());
                return std::make_unique<SingleLayerParsedMenohOperation>(this, node, layer);
            }
            else
            {
                // Otherwise it's just a regular addition
                return AddAdditionLayer(node);
            }
        }

        ParsedMenohOperationPtr MenohParser::ParseBiasAdd(const menoh_impl::node& node, const menoh_impl::graph& graph) {
	    boost::ignore_unused(graph);

	    return AddAdditionLayer(node, true);
        }


        ParsedMenohOperationPtr MenohParser::ParseFC(const menoh_impl::node& node, const menoh_impl::graph& graph) {
	    boost::ignore_unused(graph);
	    
            IConnectableLayer* layer = AddFullyConnectedLayer(node, GetNodeName(node).c_str());
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

            virtual IOutputSlot& ResolveArmnnOutputSlot(unsigned int MenohOutputIndex) override
            {
                BOOST_ASSERT(m_Representative);
                return m_Representative->ResolveArmnnOutputSlot(MenohOutputIndex);
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

	  MenohVector( const T* data_ = nullptr, const int size_ = 0 )
	    : my_data(data_)
	    , my_size(size_) {}
	    
          const T* data() const noexcept { return my_data; }
          const int size() const { return my_size; }
          void set_data(const T* data_)  { my_data = data_; }
          void set_size(const int size_) { my_size = size_; }
	  
        private:

          const T*  my_data;
	  int my_size;
        };

        /// An ParsedMenohOperation for a Const node.
        /// Creation of the armnn ConstLayer is deferred until it is actually needed, because Const nodes are mostly used
        /// for weight inputs to MatMul/Conv2D nodes and in these cases armnn doesn't need a ConstLayer.
        template <typename T>
        class ParsedConstMenohOperation : public DeferredSingleLayerParsedMenohOperation {
        public:
            ParsedConstMenohOperation(MenohParser* parser, const menoh_impl::node& node,
                                      const T* tensorData, const TensorInfo& tensorInfo)
                : DeferredSingleLayerParsedMenohOperation(parser, node),
                m_Storage(tensorData, tensorInfo.GetNumElements()),
		m_TensorInfo(tensorInfo)
            {
                BOOST_ASSERT(tensorInfo.GetDataType() == GetDataType<T>());
            }

            void CreateLayerDeferred() override
            {
                BOOST_ASSERT(m_Layer == nullptr);
                m_Layer = m_Parser->m_Network->AddConstantLayer(ConstTensor(m_TensorInfo, m_Storage), GetNodeName(m_Node).c_str());
                m_Layer->GetOutputSlot(0).SetTensorInfo(m_TensorInfo);
            }

            ConstTensor GetConstTensor(MenohVector<T>& outputTensorData) const
            {
                const TensorInfo outInfo = m_TensorInfo;

                outputTensorData.set_data(m_Storage.data());
                outputTensorData.set_size(m_Storage.size());

		// Update the result to point to the user provided storage
                ConstTensor constTensor(outInfo, outputTensorData);
                return constTensor;
            }

            ConstTensor GetConstTensor(bool swizzleForConvolutionWeights, std::vector<T>& outputTensorData) const
            {
                // Mappings from TensorFlow filter tensors to the ArmNN filter tensors.
                // Tensorflow weights are [H, W, In, Out]
                // ArmNN weights are [Out, In, H, W]
                static const PermutationVector HWIOToOIHW = {2, 3, 1, 0};

                const TensorInfo outInfo = swizzleForConvolutionWeights
                                        ? armnnUtils::Permuted(m_TensorInfo, HWIOToOIHW)
                                        : m_TensorInfo;

                outputTensorData.resize(m_TensorInfo.GetNumElements());

                // Copy or swizzle from the permanent storage into the storage the caller provided.
                if (swizzleForConvolutionWeights)
                {
                    armnnUtils::Permute(outInfo.GetShape(), HWIOToOIHW, m_Storage.data(), outputTensorData.data());
                }
                else
                {
                    memcpy(outputTensorData.data(), m_Storage.data(), m_TensorInfo.GetNumBytes());
                }
                // Update the result to point to the user provided storage
                ConstTensor constTensor(outInfo, outputTensorData);
                return constTensor;
            }

        private:
            ///< Manages the lifetime of the tensor data.
            MenohVector<T> m_Storage;
            ///< Describes the layout of the tensor and points to the data in m_Storage.
            TensorInfo m_TensorInfo;
        };      

        DataType ConvertMenohTensorDataType(const dtype_t MenohDataType) {
            switch (MenohDataType)
            {
            case dtype_t::float_:
                return DataType::Float32;
                break;
		/*
            case dtype_t::DT_INT32:
                return DataType::Signed32;
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
                if (dataType == DataType::Float32)
                {
                    return FuncType::template Parse<float>(std::forward<Args>(args)...);
                }
                else if (dataType == DataType::Signed32)
                {
                    return FuncType::template Parse<int32_t>(std::forward<Args>(args)...);
                }

                return ResType();
            }

            template<class... Args>
            inline static void Result(DataType dataType, Args&&... args)
            {
                if (dataType == DataType::Float32)
                {
                    FuncType::template Parse<float>(std::forward<Args>(args)...);
                }
                else if (dataType == DataType::Signed32)
                {
                    FuncType::template Parse<int32_t>(std::forward<Args>(args)...);
                }
            }
        };  


      ParsedMenohOperationPtr MenohParser::ParseConst(const menoh_impl::node& node, const menoh_impl::graph& graph) {
	    boost::ignore_unused(graph);
	    std::string name = GetNodeName(node);

	    BOOST_ASSERT(node.op_type == "Const");

#ifdef ARM_DEBUG
	    std::cout << std::endl << " [node] : Const, " << name << std::endl;
#endif

            auto it = m_ParamByName.find(name);
            if (it == m_ParamByName.end() )
            {
                throw ParseException(boost::str(boost::format("ParseConst : not found %1%") % name));
            }

	    auto arr  = m_ParamByName[name];

            const dtype_t MenohDataType = dtype_t::float_;
            const DataType dataType = ConvertMenohTensorDataType(MenohDataType);
            unsigned int numElements = 0U;

	    std::vector<unsigned int> dimensionSizes(arr.dims().data(), arr.dims().data()+arr.dims().size());
            if (!dimensionSizes.empty())
            {
                numElements = std::accumulate(dimensionSizes.begin(), dimensionSizes.end(),
                                            1U, std::multiplies<unsigned int>());
            }
            TensorProto menohTensor;
            if (dataType == DataType::Float32)
            {
	        menohTensor.set_float( (float *)arr.data(), numElements );
            }
            else if (dataType == DataType::Signed32)
            {
	        menohTensor.set_int(   (int32_t *)arr.data(), numElements );
            }

            const int8_t *data = menohTensor.int8_val();
            MenohVector<int8_t> tensorData(data, menohTensor.int8_size());

            const TensorInfo tensorInfo(static_cast<unsigned int>(dimensionSizes.size()), dimensionSizes.data(), dataType);
#ifdef ARM_DEBUG
	    std::cout << "   output = " << tensorInfo.GetNumDimensions() << ", " << tensorInfo.GetNumElements() << std::endl;
	    std::cout << "   outputShape = ";
	    for( unsigned int i=0 ; i<tensorInfo.GetNumDimensions() ; i++ )
	      std::cout << tensorInfo.GetShape()[i] << " ";
	    std::cout << std::endl;
#endif
            if (tensorData.size() > tensorInfo.GetNumBytes())
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

	    IOutputSlot& inputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
	    TensorInfo inputTensorInfo = inputSlot.GetTensorInfo();
            if (!HasParsedConstTensor<float>(GetNodeName(inputs[1].m_IndexedValue->GetNode()))
	     || !HasParsedConstTensor<float>(GetNodeName(inputs[2].m_IndexedValue->GetNode())))
            {
                throw ParseException("ArmNN only supports Convolution layers with constant weights and biases");
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

            // read the dilations, if present - only [1,1,1,1] (the default) is supported
            std::vector<uint32_t> dilations = ReadOptionalNodeUint32ListAttribute(node, "dilations");
            if (!dilations.empty())
            {
                for (auto dilation : dilations)
                {
                    if (dilation != 1u)
                    {
                        throw ParseException("ArmNN only supports Convolution layers with dilations [1,1,1,1]");
                    }
                }
            }

            Convolution2dDescriptor desc;
            desc.m_BiasEnabled = true;

	    std::string dataFormat = "NCHW";

	    if (dataFormat == "WHCH")
            {
	        desc.m_StrideX = strides[0];
	        desc.m_StrideY = strides[1];
            }
            else
	    if (dataFormat == "NCHW")
            {
	        desc.m_StrideX = strides[0];
	        desc.m_StrideY = strides[1];
            }
            else
            {
	        throw ParseException("Unsupported data format passed for Conv2D. Only NHWC and NCHW supported");
            }

            uint32_t inputHeight = inputTensorInfo.GetShape()[2];
            uint32_t inputWidth  = inputTensorInfo.GetShape()[3];

            uint32_t weightHeight = weightTensor.GetShape()[2];
            uint32_t weightWidth  = weightTensor.GetShape()[3];

#ifdef ARM_DEBUG
	    std::cout << "      input(" << inputTensorInfo.GetNumDimensions() << ") = "
		      << inputTensorInfo.GetShape()[0] << ", " << inputTensorInfo.GetShape()[1];
	    std::cout << ", " << inputTensorInfo.GetShape()[2] << ", " << inputTensorInfo.GetShape()[3] << std::endl;
	    std::cout << "     weight(" << weightTensor.GetNumDimensions() << ") = "
		      << weightTensor.GetShape()[0] << ", " << weightTensor.GetShape()[1];
	    std::cout << ", " << weightTensor.GetShape()[2] << ", " << weightTensor.GetShape()[3] << std::endl;
	    std::cout << "       bias(" << biasTensor.GetNumDimensions() << ") = " << biasTensor.GetShape()[0] << std::endl;
#endif
	    
            bool padding = false;
            TensorInfo outputInfo;
            if (pads[0] && pads[1])
            {
                padding = true;
                outputInfo = TensorInfo({ inputTensorInfo.GetShape()[0],
                                          weightTensor.GetShape()[0],
                                          static_cast<uint32_t>(ceil(
                                              static_cast<float>(inputHeight) /
                                              static_cast<float>(desc.m_StrideY))),
                                          static_cast<uint32_t>(ceil(
                                              static_cast<float>(inputWidth) /
                                              static_cast<float>(desc.m_StrideX)))
                                        }, DataType::Float32);
            }
            else
            {
                padding = false;
                outputInfo = TensorInfo({ inputTensorInfo.GetShape()[0],
                                          weightTensor.GetShape()[0],
                                          static_cast<uint32_t>(ceil(
                                              static_cast<float>(inputHeight - weightHeight + 1) /
                                              static_cast<float>(desc.m_StrideY))),
                                          static_cast<uint32_t>(ceil(
                                              static_cast<float>(inputWidth - weightWidth + 1) /
                                              static_cast<float>(desc.m_StrideX)))
                                        }, DataType::Float32);
            }
#ifdef ARM_DEBUG
	    std::cout << "   output = " << outputInfo.GetNumDimensions() << ", " << outputInfo.GetNumElements() << std::endl;
	    std::cout << "   outputShape = ";
	    for( unsigned int i=0 ; i<outputInfo.GetNumDimensions() ; i++ )
	      std::cout << outputInfo.GetShape()[i] << " ";
	    std::cout << std::endl;
#endif
            CalcPadding(inputHeight, weightHeight, desc.m_StrideY, desc.m_PadTop,  desc.m_PadBottom, padding);
            CalcPadding(inputWidth,  weightWidth,  desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight,  padding);
	    
	    IConnectableLayer* layer = m_Network->AddConvolution2dLayer(desc, weightTensor, biasTensor, name.c_str());
            layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

            if (dataFormat == "NHWC")
            {
	        layer = SwizzleInDeswizzleOut(*m_Network, inputSlot, *layer, name);
            }
            else
            {
                inputSlot.Connect(layer->GetInputSlot(0));
            }
            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, layer);
        }  

        ParsedMenohOperationPtr MenohParser::ParseDepthwiseConv2D(const menoh_impl::node& node, const menoh_impl::graph& graph) {
	    boost::ignore_unused(graph);
	    std::string name = GetNodeName(node);

	    std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 2);
            IOutputSlot& inputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
            TensorInfo inputTensorInfo = inputSlot.GetTensorInfo();

            if (!HasParsedConstTensor<float>(GetNodeName(inputs[1].m_IndexedValue->GetNode()))
             || !HasParsedConstTensor<float>(GetNodeName(inputs[2].m_IndexedValue->GetNode())))
            {
                throw ParseException("ArmNN only supports Depthwise Convolution layers with constant weights");
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

            DepthwiseConvolution2dDescriptor desc;
            desc.m_BiasEnabled = true;

            const std::string dataFormat = "NCHW";

            if (dataFormat == "NHWC")
            {
                desc.m_StrideX = strides[2];
                desc.m_StrideY = strides[1];
                // Swizzle input to supported memory layout
                inputTensorInfo = armnnUtils::Permuted(inputSlot.GetTensorInfo(), NHWCToArmNN);
            }
            else if (dataFormat == "NCHW")
            {
                desc.m_StrideX = strides[3];
                desc.m_StrideY = strides[2];
            }
            else
            {
                throw ParseException("Unsupported data format passed for DepthwiseConv2dNative. Only NHWC and NCHW supported");
            }

            uint32_t inputHeight = inputTensorInfo.GetShape()[2];
            uint32_t inputWidth  = inputTensorInfo.GetShape()[3];

            uint32_t weightHeight = weightTensor.GetShape()[2];
            uint32_t weightWidth  = weightTensor.GetShape()[3];

            bool padding = false;
            TensorInfo outputInfo;
            if( pads[0] || pads[1] )
            {
                padding = true;
                outputInfo = TensorInfo({ inputTensorInfo.GetShape()[0],
                                        weightTensor.GetShape()[0] * weightTensor.GetShape()[1],
                                        static_cast<uint32_t>(ceil(
                                            static_cast<float>(inputHeight) /
                                            static_cast<float>(desc.m_StrideY))),
                                        static_cast<uint32_t>(ceil(
                                            static_cast<float>(inputWidth) /
                                            static_cast<float>(desc.m_StrideX)))
                                        }, DataType::Float32);
            }
            else
            {
                padding = false;
                outputInfo = TensorInfo({ inputTensorInfo.GetShape()[0],
                                        weightTensor.GetShape()[0] * weightTensor.GetShape()[1],
                                        static_cast<uint32_t>(ceil(
                                            static_cast<float>(inputHeight - weightHeight + 1) /
                                            static_cast<float>(desc.m_StrideY))),
                                        static_cast<uint32_t>(ceil(
                                            static_cast<float>(inputWidth - weightWidth + 1) /
                                            static_cast<float>(desc.m_StrideX)))
                                        }, DataType::Float32);
            }

            CalcPadding(inputHeight, weightHeight, desc.m_StrideY, desc.m_PadTop,  desc.m_PadBottom, padding);
            CalcPadding(inputWidth,  weightWidth,  desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight,  padding);

            IConnectableLayer* layer = m_Network->AddDepthwiseConvolution2dLayer(desc, weightTensor, name.c_str());
            layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

            if (dataFormat == "NHWC")
            {
 	        layer = SwizzleInDeswizzleOut(*m_Network, inputSlot, *layer, name);
            }
            else
            {
                inputSlot.Connect(layer->GetInputSlot(0));
            }

            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, layer);
        }       

        ParsedMenohOperationPtr MenohParser::ParseFusedBatchNorm(const menoh_impl::node& node, const menoh_impl::graph& graph) {
	    boost::ignore_unused(graph);
	    std::string name = GetNodeName(node);

	    std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 5);

            if (!HasParsedConstTensor<float>(GetNodeName(inputs[1].m_IndexedValue->GetNode())))
            {
                throw ParseException("ArmNN only supports FusedBatchNormalization layers with constant scale");
            }
            ParsedConstMenohOperation<float>* scaleNode =
                boost::polymorphic_downcast<ParsedConstMenohOperation<float> *>(inputs[1].m_IndexedValue);

            if (!HasParsedConstTensor<float>(GetNodeName(inputs[2].m_IndexedValue->GetNode())))
            {
                throw ParseException("ArmNN only supports FusedBatchNormalization layers with constant offset");
            }
            ParsedConstMenohOperation<float>* offsetNode =
                boost::polymorphic_downcast<ParsedConstMenohOperation<float> *>(inputs[2].m_IndexedValue);

            if (!HasParsedConstTensor<float>(GetNodeName(inputs[3].m_IndexedValue->GetNode())))
            {
                throw ParseException("ArmNN only supports FusedBatchNormalization layers with constant mean");
            }
            ParsedConstMenohOperation<float>* meanNode =
                boost::polymorphic_downcast<ParsedConstMenohOperation<float> *>(inputs[3].m_IndexedValue);

            if (!HasParsedConstTensor<float>(GetNodeName(inputs[4].m_IndexedValue->GetNode())))
            {
                throw ParseException("ArmNN only supports FusedBatchNormalization layers with constant variance");
            }
            ParsedConstMenohOperation<float>* varianceNode =
                boost::polymorphic_downcast<ParsedConstMenohOperation<float> *>(inputs[4].m_IndexedValue);

            // The descriptor only has the epsilon attribute
            BatchNormalizationDescriptor desc;
            desc.m_Eps = ReadMandatoryNodeFloatAttribute(node, "epsilon");

            // data for the parsed tensor args (scale, offset, mean, variance) must be stored locally until the layer is added
            std::vector<float> scaleTensorData;
            ConstTensor scaleTensor = scaleNode->GetConstTensor(false, scaleTensorData);

            std::vector<float> offsetTensorData;
            ConstTensor offsetTensor = offsetNode->GetConstTensor(false, offsetTensorData);

            std::vector<float> meanTensorData;
            ConstTensor meanTensor = meanNode->GetConstTensor(false, meanTensorData);

            std::vector<float> varianceTensorData;
            ConstTensor varianceTensor = varianceNode->GetConstTensor(false, varianceTensorData);

            IConnectableLayer* layer = m_Network->AddBatchNormalizationLayer(desc,
                                                                            meanTensor,
                                                                            varianceTensor,
                                                                            offsetTensor,
                                                                            scaleTensor,
									    name.c_str());

            IOutputSlot& inputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);

//            const std::string dataFormat = ReadMandatoryNodeStringAttribute(node, "data_format");
            const std::string dataFormat = "NCHW";

            if (dataFormat == "NHWC")
            {
                const TensorInfo outputTensorInfo = armnnUtils::Permuted(inputSlot.GetTensorInfo(), NHWCToArmNN);
                layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);
                layer = SwizzleInDeswizzleOut(*m_Network, inputSlot, *layer, name);
            }
            else
            {
                layer->GetOutputSlot(0).SetTensorInfo(inputSlot.GetTensorInfo());
                inputSlot.Connect(layer->GetInputSlot(0));
            }

            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, layer);
        }

        ParsedMenohOperationPtr MenohParser::ParseConcat(const menoh_impl::node& node, const menoh_impl::graph& graph) {
	    boost::ignore_unused(graph);
	    std::string name = GetNodeName(node);
	    
	    std::vector<OutputOfConstNodeDef> nodes = GetMenohInputNodes(node);
            // In tensorflow, we have the last input of the Concat layer as the axis for concatenation
            unsigned int numInputs = static_cast<unsigned int>(nodes.size());
            unsigned int numConcatView = numInputs - 1;

            OriginsDescriptor concatDescriptor(static_cast<uint32_t>(numConcatView), MaxNumOfTensorDimensions);
            std::vector<unsigned int>mergeDimSizes(MaxNumOfTensorDimensions, 0u);

            unsigned int mergeDim = 0;
            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, numInputs);

            // The last input is the axis for concatenation
            if (!HasParsedConstTensor<int32_t>(GetNodeName(inputs[numInputs - 1].m_IndexedValue->GetNode())))
            {
                throw ParseException("ArmNN only supports Concat with constant axis");
            }
            ParsedConstMenohOperation<int32_t>* shapeNode =
                    boost::polymorphic_downcast<ParsedConstMenohOperation<int32_t>*>(inputs[numInputs - 1].m_IndexedValue);

            std::vector<int32_t> axisTensorData;
            ConstTensor axisTensor = shapeNode->GetConstTensor(false, axisTensorData);

            // This concatDim indicates the data format: 3 is the NHWC, 1 is the NCHW
            const unsigned int concatDimInput = static_cast<unsigned int>(axisTensorData[0]);

            // Armnn supports concatenation along the channel dimension for data format NHWC and NCHW
            if (concatDimInput == 0 || concatDimInput == 2)
            {
                throw ParseException("The dimension for concatenation is not supported by Armnn");
            }

            // This is the only concatDim we support in Armnn
            const unsigned int concatDim = 1;
            for (unsigned int viewIndex = 0; viewIndex < numConcatView; ++viewIndex)
            {
                // need to double check whether it should be
                IOutputSlot& inputSlot =
                    inputs[viewIndex].m_IndexedValue->ResolveArmnnOutputSlot(inputs[viewIndex].m_Index);
                TensorInfo inputTensorInfo = inputSlot.GetTensorInfo();

                if (inputTensorInfo.GetNumDimensions() != MaxNumOfTensorDimensions)
                {
                    throw ParseException("The number of dimensions for input tensors of the concatenation op should be 4");
                }

                if (concatDimInput == 3)
                {
                    inputTensorInfo = armnnUtils::Permuted(inputTensorInfo, NHWCToArmNN);
                }

                for (unsigned int dim = 0; dim < MaxNumOfTensorDimensions; ++dim)
                {
                    mergeDimSizes[dim] = inputTensorInfo.GetShape()[dim];
                }

                for (unsigned int j = 0; j < concatDim; ++j)
                {
                    concatDescriptor.SetViewOriginCoord(viewIndex, j, 0);
                }

                concatDescriptor.SetViewOriginCoord(viewIndex, concatDim, mergeDim);
                mergeDim += mergeDimSizes[concatDim];

                for (unsigned int j = concatDim+1; j < MaxNumOfTensorDimensions; ++j)
                {
                    concatDescriptor.SetViewOriginCoord(viewIndex, j, 0);
                }
            }

            mergeDimSizes[concatDim] = mergeDim;
            armnn::IConnectableLayer *layer = m_Network->AddMergerLayer(concatDescriptor, name.c_str());

            layer->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo(MaxNumOfTensorDimensions, mergeDimSizes.data(),
                                                                    DataType::Float32));

            for (unsigned int v = 0; v < numConcatView; ++v)
            {
                IOutputSlot& inputSlot = inputs[v].m_IndexedValue->ResolveArmnnOutputSlot(inputs[v].m_Index);
                if (concatDimInput == 3)
                {
                    IConnectableLayer* const swizzleLayer = AddSwizzleLayer(*m_Network, inputSlot, NHWCToArmNN,
                                                                            "swizzle_for-" + name);
                    swizzleLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(v));
                }
                else
                {
                    inputSlot.Connect(layer->GetInputSlot(v));
                }
            }

            if (concatDimInput == 3)
            {
                IConnectableLayer* const deswizzleLayer = AddSwizzleLayer(*m_Network, layer->GetOutputSlot(0), ArmNNToNHWC,
									  "deswizzle_for-" + name);
                layer = deswizzleLayer;
            }

            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, layer);
        }

        ParsedMenohOperationPtr MenohParser::ParseShape(const menoh_impl::node& node, const menoh_impl::graph& graph) {
	    boost::ignore_unused(graph);

	    // Note: The Shape layer is handled in a special way, because:
            //        1. ARMNN doesn't support int32 tensors which it outputs
            //        2. ARMNN works with statically shaped tensors which are known at parse time
            //        3. because of 1. and 2. we treat the output of Shape as a temporary const int32
            //           tensor which may be used as an input to other ops, most likely a Reshape

#if 0
            const tensorflow::DataType MenohDataType = ReadMandatoryNodeTypeAttribute(node, "out_type");
            if (MenohDataType != tensorflow::DT_INT32)
            {
                throw ParseException("Armnn only supports DT_INT32 as out_type");
            }
#endif
            const std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 1);
            IOutputSlot& prevLayerOutputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
            const TensorInfo& prevLayerTensorInfo = prevLayerOutputSlot.GetTensorInfo();
            unsigned int prevLayerDimensions = prevLayerTensorInfo.GetNumDimensions();

            std::vector<int32_t> shapeTensorData;
            shapeTensorData.reserve(prevLayerDimensions);

            for (unsigned int i=0; i<prevLayerDimensions; ++i)
            {
                shapeTensorData.push_back(static_cast<int32_t>(prevLayerTensorInfo.GetShape()[i]));
            }

            TensorInfo shapeTensorInfo(1, &prevLayerDimensions, DataType::Signed32);

            return std::make_unique<ParsedConstMenohOperation<int32_t>>(this,
                                                                    node,
                                                                    &shapeTensorData[0],
                                                                    shapeTensorInfo);
        }

        ParsedMenohOperationPtr MenohParser::ParseReshape(const menoh_impl::node& node, const menoh_impl::graph& graph) {
	    boost::ignore_unused(graph);
	    std::string name = GetNodeName(node);
	    
            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 2);
            ParsedMenohOperation* inputNode = inputs[0].m_IndexedValue;

            if (!HasParsedConstTensor<int32_t>(GetNodeName(inputs[1].m_IndexedValue->GetNode())))
            {
                throw ParseException("ArmNN only supports Reshape layers with constant shapes");
            }
            ParsedConstMenohOperation<int32_t>* shapeNode =
                boost::polymorphic_downcast<ParsedConstMenohOperation<int32_t>*>(inputs[1].m_IndexedValue);

            armnn::IOutputSlot& prevLayerOutputSlot = inputNode->ResolveArmnnOutputSlot(inputs[0].m_Index);
            TensorInfo inputTensorInfo = prevLayerOutputSlot.GetTensorInfo();

            std::vector<int32_t> shapeTensorData;
            ConstTensor shapeTensor = shapeNode->GetConstTensor(false, shapeTensorData);
            const TensorInfo outputTensorInfo = PrepareReshape(inputTensorInfo, shapeTensorData);

            TensorShape targetShape = outputTensorInfo.GetShape();
            ReshapeDescriptor reshapeDesc;
            reshapeDesc.m_TargetShape = targetShape;

            IConnectableLayer* layer = m_Network->AddReshapeLayer(reshapeDesc, name.c_str());
            prevLayerOutputSlot.Connect(layer->GetInputSlot(0));
            layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, layer);
        }   

        ParsedMenohOperationPtr MenohParser::ParseLrn(const menoh_impl::node& node, const menoh_impl::graph& graph) {
	    boost::ignore_unused(graph);
	    std::string name = GetNodeName(node);
	    
	    std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 1);

            NormalizationDescriptor normalizationDescriptor;
            normalizationDescriptor.m_NormMethodType = NormalizationAlgorithmMethod::LocalBrightness;
            normalizationDescriptor.m_NormChannelType = NormalizationAlgorithmChannel::Across;
            normalizationDescriptor.m_Alpha = ReadMandatoryNodeFloatAttribute(node, "alpha");
            normalizationDescriptor.m_Beta = ReadMandatoryNodeFloatAttribute(node, "beta");
            normalizationDescriptor.m_K = ReadMandatoryNodeFloatAttribute(node, "bias");
            normalizationDescriptor.m_NormSize = ReadMandatoryNodeUint32Attribute(node, "depth_radius");

            // The window size must be an odd value. For a window size of (2 * n + 1), TensorFlow defines depth_radius = n.
            normalizationDescriptor.m_NormSize = normalizationDescriptor.m_NormSize * 2 + 1;

            IOutputSlot& prevLayerOutputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);

            IConnectableLayer* layer = m_Network->AddNormalizationLayer(normalizationDescriptor,
									name.c_str());

            const TensorInfo permutedInfo = armnnUtils::Permuted(prevLayerOutputSlot.GetTensorInfo(), NHWCToArmNN);
            layer->GetOutputSlot(0).SetTensorInfo(permutedInfo);

            layer = SwizzleInDeswizzleOut(*m_Network, prevLayerOutputSlot, *layer, name);

            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, layer);
        }
 
        /// An ParsedMenohOperation for a MatMul node.
        /// Creation of the armnn FullyConnected layer is deferred until it is actually needed, because MatMul nodes are
        /// often used for the first part of a biased FullyConnected (MatMul followed by Add) and in these cases armnn doesn't
        /// need a separate layer for the MatMul.
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

	    // Defer the creation of the layer (see ParsedMatMulMenohOperation).
            return std::make_unique<ParsedMatMulMenohOperation>(this, node);
        }

        ParsedMenohOperationPtr MenohParser::ParseMul(const menoh_impl::node& node, const menoh_impl::graph& graph) {
	    boost::ignore_unused(graph);
	    std::string name = GetNodeName(node);
	    
            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 2);

            IConnectableLayer* const layer = m_Network->AddMultiplicationLayer(name.c_str());
            IOutputSlot* input0Slot = &inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
            IOutputSlot* input1Slot = &inputs[1].m_IndexedValue->ResolveArmnnOutputSlot(inputs[1].m_Index);

            auto const input0NumDims = input0Slot->GetTensorInfo().GetNumDimensions();
            auto const input1NumDims = input1Slot->GetTensorInfo().GetNumDimensions();

            if (input0NumDims < input1NumDims)
            {
                const bool isNHWC = true;
                input0Slot = BroadcasMenohorAddandMul(input1Slot, input0Slot, isNHWC, *m_Network, node);
            }
            if (input1NumDims < input0NumDims)
            {
                const bool isNHWC = true;
                input1Slot = BroadcasMenohorAddandMul(input0Slot, input1Slot, isNHWC, *m_Network, node);
            }

            input0Slot->Connect(layer->GetInputSlot(0));
            input1Slot->Connect(layer->GetInputSlot(1));

            if (input0NumDims < input1NumDims)
            {
                layer->GetOutputSlot(0).SetTensorInfo(input1Slot->GetTensorInfo());
            }
            else
            {
                layer->GetOutputSlot(0).SetTensorInfo(input0Slot->GetTensorInfo());
            }
            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, layer);
        }  

        ParsedMenohOperationPtr MenohParser::ParsePlaceholder(const menoh_impl::node& node, const menoh_impl::graph& graph) {
	    boost::ignore_unused(graph);
	    std::string name = GetNodeName(node);
	    
            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 0);

            const LayerBindingId layerId = boost::numeric_cast<LayerBindingId>(m_NetworkInputsBindingInfo.size());

            auto it = m_InputShapes.find(name);
            if (it == m_InputShapes.end())
            {
	        throw ParseException("Missing input shape for Placeholder '" + name + "'");
            }

            auto dims = ReadOptionalNodeUint32ListAttribute(node, "dims");
            const TensorInfo tensorInfo(static_cast<unsigned int>(dims.size()), (const unsigned int*)dims.data(), DataType::Float32);
	    
            IConnectableLayer* const layer = m_Network->AddInputLayer(layerId, name.c_str());

#ifdef ARM_DEBUG
	    std::cout << "   output = " << tensorInfo.GetNumDimensions() << ", " << tensorInfo.GetNumElements() << std::endl;
	    std::cout << "   outputShape = ";
	    for( unsigned int i=0 ; i<tensorInfo.GetNumDimensions() ; i++ )
	      std::cout << tensorInfo.GetShape()[i] << " ";
	    std::cout << std::endl;
#endif
            layer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

            TrackInputBinding(layer, layerId, tensorInfo);

            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, layer);
        }

        ParsedMenohOperationPtr MenohParser::ParseRelu(const menoh_impl::node& node, const menoh_impl::graph& graph) {
	    boost::ignore_unused(graph);
	    
            ActivationDescriptor activationDesc;
            activationDesc.m_Function = ActivationFunction::ReLu;
            return AddActivationLayer(node, activationDesc);
        }

        ParsedMenohOperationPtr MenohParser::ParseRelu6(const menoh_impl::node& node, const menoh_impl::graph& graph ) {
	    boost::ignore_unused(graph);

            ActivationDescriptor activationDesc;
            activationDesc.m_Function = ActivationFunction::BoundedReLu;
            activationDesc.m_A = 6.0f;
            activationDesc.m_B = 0.0f;

            return AddActivationLayer(node, activationDesc);
        }

        ParsedMenohOperationPtr MenohParser::ParseSigmoid(const menoh_impl::node& node, const menoh_impl::graph& graph) {
	    boost::ignore_unused(graph);
	    
            ActivationDescriptor activationDesc;
            activationDesc.m_Function = ActivationFunction::Sigmoid;

            return AddActivationLayer(node, activationDesc);
        }

        ParsedMenohOperationPtr MenohParser::ParseSoftmax(const menoh_impl::node& node, const menoh_impl::graph& graph) {
	    boost::ignore_unused(graph);
	    std::string name = GetNodeName(node);
	    
            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 1);

            SoftmaxDescriptor softmaxDescriptor;
            IConnectableLayer* const layer = m_Network->AddSoftmaxLayer(softmaxDescriptor, name.c_str());

            IOutputSlot& prevLayerSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
            prevLayerSlot.Connect(layer->GetInputSlot(0));
            layer->GetOutputSlot(0).SetTensorInfo(prevLayerSlot.GetTensorInfo());

            TensorInfo outputInfo = prevLayerSlot.GetTensorInfo();
#ifdef ARM_DEBUG
	    std::cout << "   output = " << outputInfo.GetNumDimensions() << ", " << outputInfo.GetNumElements() << std::endl;
	    std::cout << "   outputShape = ";
	    for( unsigned int i=0 ; i<outputInfo.GetNumDimensions() ; i++ )
	      std::cout << outputInfo.GetShape()[i] << " ";
	    std::cout << std::endl;
#endif
            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, layer);
        }

        ParsedMenohOperationPtr MenohParser::ParseSoftplus(const menoh_impl::node& node, const menoh_impl::graph& graph) {
	    boost::ignore_unused(graph);
	    
            ActivationDescriptor activationDesc;
            activationDesc.m_Function = ActivationFunction::SoftReLu;

            return AddActivationLayer(node, activationDesc);
        }    

        ParsedMenohOperationPtr MenohParser::ParseTanh(const menoh_impl::node& node, const menoh_impl::graph& graph ) {
	    boost::ignore_unused(graph);
	    
            ActivationDescriptor activationDesc;
            activationDesc.m_Function = ActivationFunction::TanH;
            activationDesc.m_A = 1.0f;
            activationDesc.m_B = 1.0f;

            return AddActivationLayer(node, activationDesc);
        }

        ParsedMenohOperationPtr MenohParser::AddActivationLayer(const menoh_impl::node& node, ActivationDescriptor& activationDesc) {
	    std::string name = GetNodeName(node);

	    std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 1);

            IConnectableLayer* const layer = m_Network->AddActivationLayer(activationDesc, name.c_str());

            IOutputSlot& prevLayerOutputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
            prevLayerOutputSlot.Connect(layer->GetInputSlot(0));
            layer->GetOutputSlot(0).SetTensorInfo(prevLayerOutputSlot.GetTensorInfo());

            TensorInfo tensorInfo = prevLayerOutputSlot.GetTensorInfo();
#ifdef ARM_DEBUG
	    std::cout << "   output = " << tensorInfo.GetNumDimensions() << ", " << tensorInfo.GetNumElements() << std::endl;
	    std::cout << "   outputShape = ";
	    for( unsigned int i=0 ; i<tensorInfo.GetNumDimensions() ; i++ )
	      std::cout << tensorInfo.GetShape()[i] << " ";
	    std::cout << std::endl;
#endif
            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, layer);
        }

        ParsedMenohOperationPtr MenohParser::ParseMaxPool(const menoh_impl::node& node, const menoh_impl::graph& graph) {
	    return ParsePooling2d(node, graph, PoolingAlgorithm::Max);
        }

        ParsedMenohOperationPtr MenohParser::ParseAvgPool(const menoh_impl::node& node, const menoh_impl::graph& graph) {
	    return ParsePooling2d(node, graph, PoolingAlgorithm::Average);
        }          

        ParsedMenohOperationPtr MenohParser::ParsePooling2d(const menoh_impl::node& node, const menoh_impl::graph& graph, 
                                                            PoolingAlgorithm pooltype){
	    boost::ignore_unused(graph);
	    std::string name = GetNodeName(node);
	    
	    std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 1);
            IOutputSlot& inputSlot = inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
            TensorInfo inputTensorInfo = inputSlot.GetTensorInfo();

            if (inputs.size() != 1)
            {
                throw ParseException("2D Pooling expects one input!");
            }

            std::vector<int> strides, kernel_shape, pads;
            std::tie(strides, kernel_shape, pads) = attributes_for_2d_data_processing(node);

            Pooling2dDescriptor pooling2dDescriptor;
            pooling2dDescriptor.m_PoolType = pooltype;
            pooling2dDescriptor.m_PaddingMethod = PaddingMethod::Exclude;
            pooling2dDescriptor.m_OutputShapeRounding = OutputShapeRounding::Floor;

            pooling2dDescriptor.m_StrideX    = strides[0];
            pooling2dDescriptor.m_StrideY    = strides[1];
            pooling2dDescriptor.m_PoolWidth  = kernel_shape[0];
            pooling2dDescriptor.m_PoolHeight = kernel_shape[1];

            uint32_t inputHeight = inputTensorInfo.GetShape()[2];
            uint32_t inputWidth  = inputTensorInfo.GetShape()[3];
#ifdef ARM_DEBUG
	    std::cout << "   input = " << inputTensorInfo.GetNumDimensions() << ", " << inputTensorInfo.GetNumElements() << std::endl;
	    std::cout << "   inputShape = ";
	    for( unsigned int i=0 ; i<inputTensorInfo.GetNumDimensions() ; i++ )
	      std::cout << inputTensorInfo.GetShape()[i] << " ";
	    std::cout << std::endl;
#endif
            bool padding = false;
            TensorInfo outputInfo;
            if( pads[0] && pads[1] )
	    {  
                padding = true;
                outputInfo = TensorInfo({ inputTensorInfo.GetShape()[0],
                                          inputTensorInfo.GetShape()[1],
                                          static_cast<uint32_t>(ceil(
                                              static_cast<float>(inputHeight) /
                                              static_cast<float>(pooling2dDescriptor.m_StrideY))),
                                          static_cast<uint32_t>(ceil(
                                              static_cast<float>(inputWidth) /
                                              static_cast<float>(pooling2dDescriptor.m_StrideX)))
                                        }, DataType::Float32);
            }
            else
            {
                padding = false;
                outputInfo = TensorInfo({ inputTensorInfo.GetShape()[0],
                                          inputTensorInfo.GetShape()[1],
                                          static_cast<uint32_t>(ceil(
                                              static_cast<float>(inputHeight - pooling2dDescriptor.m_PoolHeight + 1) /
                                              static_cast<float>(pooling2dDescriptor.m_StrideY))),
                                          static_cast<uint32_t>(ceil(
                                              static_cast<float>(inputWidth - pooling2dDescriptor.m_PoolWidth + 1) /
                                              static_cast<float>(pooling2dDescriptor.m_StrideX)))
                                        }, DataType::Float32);
            }

            CalcPadding(inputWidth, pooling2dDescriptor.m_PoolWidth, pooling2dDescriptor.m_StrideX,
                            pooling2dDescriptor.m_PadLeft, pooling2dDescriptor.m_PadRight, padding);
            CalcPadding(inputHeight, pooling2dDescriptor.m_PoolHeight, pooling2dDescriptor.m_StrideY,
                            pooling2dDescriptor.m_PadTop, pooling2dDescriptor.m_PadBottom, padding);
#ifdef ARM_DEBUG
	    std::cout << "   output = " << outputInfo.GetNumDimensions() << ", " << outputInfo.GetNumElements() << std::endl;
	    std::cout << "   outputShape = ";
	    for( unsigned int i=0 ; i<outputInfo.GetNumDimensions() ; i++ )
	      std::cout << outputInfo.GetShape()[i] << " ";
	    std::cout << std::endl;
#endif
            IConnectableLayer* layer = m_Network->AddPooling2dLayer(pooling2dDescriptor, name.c_str());
            if (layer == nullptr)
            {
                throw ParseException("Failed to add pooling2d layer");
            }

            layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

            inputSlot.Connect(layer->GetInputSlot(0));

            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, layer);
        }

        ParsedMenohOperationPtr MenohParser::AddAdditionLayer(const menoh_impl::node& node, bool isBiasAdd){
	    std::string name = GetNodeName(node);

	    std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 2);

            IOutputSlot* input0Slot = &inputs[0].m_IndexedValue->ResolveArmnnOutputSlot(inputs[0].m_Index);
            IOutputSlot* input1Slot = &inputs[1].m_IndexedValue->ResolveArmnnOutputSlot(inputs[1].m_Index);

            const TensorInfo& input0Info = input0Slot->GetTensorInfo();
            const TensorInfo& input1Info = input1Slot->GetTensorInfo();

            if (isBiasAdd)
            {
                // BiasAdd takes bias as a 1D tensor. We need to add a reshape layer to create a 4D tensor
                // with the same data in the correct dimension for broadcast in addition.
                if(input1Info.GetNumDimensions() != 1)
                {
                    throw ParseException("Unsupported bias for BiasAdd. It should be a 1D vector.");
                }

                const std::string dataFormat = "NCHW";
                const bool isNHWC = (dataFormat == "NHWC");
                const bool isNCHW = (dataFormat == "NCHW");

                if (!isNHWC && ! isNCHW)
                {
                    throw ParseException("Only NHWC or NCHW supported for BiasAdd");
                }

                input1Slot = BroadcasMenohorAddandMul(input0Slot, input1Slot, isNHWC, *m_Network, node);
            }
            else
            {
                if (input0Info.GetNumDimensions() == 1)
                {
                    const bool isNHWC = true;
                    input0Slot = BroadcasMenohorAddandMul(input1Slot, input0Slot, isNHWC, *m_Network, node);
                }

                if (input1Info.GetNumDimensions() == 1)
                {
                    const bool isNHWC = true;
                    input1Slot = BroadcasMenohorAddandMul(input0Slot, input1Slot, isNHWC, *m_Network, node);
                }
            }

            IConnectableLayer* const layer = m_Network->AddAdditionLayer(name.c_str());

            input0Slot->Connect(layer->GetInputSlot(0));
            input1Slot->Connect(layer->GetInputSlot(1));

            if (input0Info.GetNumDimensions() == 1 && isBiasAdd == false)
            {
                layer->GetOutputSlot(0).SetTensorInfo(input1Slot->GetTensorInfo());
            }
            else
            {
                layer->GetOutputSlot(0).SetTensorInfo(input0Slot->GetTensorInfo());
            }

            return std::make_unique<SingleLayerParsedMenohOperation>(this, node, layer);
        }

        IConnectableLayer* MenohParser::AddFullyConnectedLayer(const menoh_impl::node& matMulNodeDef, 
                                                               const menoh_impl::node* addNodeDef, const char* armnnLayerName){
            // find bias const (if applicable)
            ParsedConstMenohOperation<float>* biasNode = nullptr;
            if (addNodeDef != nullptr)
            {
                std::vector<OutputOfParsedMenohOperation> addInputs = GetInputParsedMenohOperationsChecked(*addNodeDef, 2);
                // find our inputs
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
                    throw ParseException("ArmNN only supports fully connected layers with constant bias");
                }
            }

            // find matmul inputs
            ParsedConstMenohOperation<float>* weightNode = nullptr;
            ParsedMenohOperation* inputNode  = nullptr;
            unsigned int inputIdx = 0;
            std::vector<OutputOfParsedMenohOperation> mulInputs = GetInputParsedMenohOperationsChecked(matMulNodeDef, 2);
            if (HasParsedConstTensor<float>(GetNodeName(mulInputs[0].m_IndexedValue->GetNode())))
            {
                weightNode = boost::polymorphic_downcast<ParsedConstMenohOperation<float>*>(mulInputs[0].m_IndexedValue);
                inputNode = mulInputs[1].m_IndexedValue;
                inputIdx = mulInputs[1].m_Index;
            }
            else if (HasParsedConstTensor<float>(GetNodeName(mulInputs[1].m_IndexedValue->GetNode())))
            {
                weightNode = boost::polymorphic_downcast<ParsedConstMenohOperation<float>*>(mulInputs[1].m_IndexedValue);
                inputNode = mulInputs[0].m_IndexedValue;
                inputIdx = mulInputs[0].m_Index;
            }
            else
            {
                throw ParseException("ArmNN only supports fully connected layers with constant weights");
            }

            MenohVector<float> weightTensorData;
            // handle weight
            ConstTensor weights = weightNode->GetConstTensor(weightTensorData);

            FullyConnectedDescriptor desc;
            desc.m_BiasEnabled = addNodeDef != nullptr;

            IConnectableLayer* layer = nullptr;
            // make the layer
            if (addNodeDef != nullptr)
            {
                MenohVector<float> biasTensorData;
                ConstTensor biases = biasNode->GetConstTensor(biasTensorData);

                if (weights.GetShape()[1] != biases.GetShape()[0])
                {
                    throw ParseException("shape of matmul and bias do not match");
                }

                layer = m_Network->AddFullyConnectedLayer(desc, weights, biases, armnnLayerName);
            }
            else
            {
                layer = m_Network->AddFullyConnectedLayer(desc, weights, armnnLayerName);
            }

            BOOST_ASSERT(layer != nullptr);

            inputNode->ResolveArmnnOutputSlot(inputIdx).Connect(layer->GetInputSlot(0));
            unsigned int batches = inputNode->ResolveArmnnOutputSlot(inputIdx).GetTensorInfo().GetShape()[0];

            // handle output
            TensorInfo outputInfo({ batches, weights.GetShape()[1] }, DataType::Float32);
            layer->GetOutputSlot(0).SetTensorInfo(outputInfo);
            return layer;
        }

        IConnectableLayer* MenohParser::AddFullyConnectedLayer(const menoh_impl::node& node, const char* armnnLayerName){
            std::vector<OutputOfParsedMenohOperation> inputs = GetInputParsedMenohOperationsChecked(node, 3);

            ParsedMenohOperation*             inputNode  = nullptr;
            ParsedConstMenohOperation<float>* weightNode = nullptr;
            ParsedConstMenohOperation<float>* biasNode   = nullptr;

            unsigned int inputIdx = 0;
            inputNode  = inputs[0].m_IndexedValue;
            inputIdx   = inputs[0].m_Index;
            weightNode = boost::polymorphic_downcast<ParsedConstMenohOperation<float>*>(inputs[1].m_IndexedValue);
            biasNode   = boost::polymorphic_downcast<ParsedConstMenohOperation<float>*>(inputs[2].m_IndexedValue);

            MenohVector<float> weightTensorData;
            ConstTensor weights = weightNode->GetConstTensor(weightTensorData);
            MenohVector<float> biasTensorData;
            ConstTensor biases  = biasNode->GetConstTensor(biasTensorData);

            if (weights.GetShape()[0] != biases.GetShape()[0])
            {
                throw ParseException("shape of weight and bias do not match");
            }

            FullyConnectedDescriptor desc;
            desc.m_BiasEnabled           = true;
            desc.m_TransposeWeightMatrix = true;

            IConnectableLayer* layer = nullptr;
            layer = m_Network->AddFullyConnectedLayer(desc, weights, biases, armnnLayerName);

            BOOST_ASSERT(layer != nullptr);
            inputNode->ResolveArmnnOutputSlot(inputIdx).Connect(layer->GetInputSlot(0));
            unsigned int batches = inputNode->ResolveArmnnOutputSlot(inputIdx).GetTensorInfo().GetShape()[0];
            // handle output
            TensorInfo outputInfo({ batches, weights.GetShape()[0] }, DataType::Float32);
            layer->GetOutputSlot(0).SetTensorInfo(outputInfo);
#ifdef ARM_DEBUG
	    std::cout << "   output = " << outputInfo.GetNumDimensions() << ", " << outputInfo.GetNumElements() << std::endl;
	    std::cout << "   outputShape = ";
	    for( unsigned int i=0 ; i<outputInfo.GetNumDimensions() ; i++ )
	      std::cout << outputInfo.GetShape()[i] << " ";
	    std::cout << std::endl;
#endif
	    return layer;
        }

        void MenohParser::LoadNode(const menoh_impl::node& node, const menoh_impl::graph& graph) {
	    std::string name = GetNodeName(node);
	    
	    // get the type of the node (assume float)
            dtype_t type = dtype_t::float_;

            const std::string& operation = node.op_type;
            auto it = ms_OperationNameToParsingFunctions.find(operation);
            if (it != ms_OperationNameToParsingFunctions.end())
            {
                auto func = it->second;
                ParsedMenohOperationPtr parsedMenohOperation = (this->*func)(node, graph);
                ParsedMenohOperation* parsedMenohOperationRaw = parsedMenohOperation.get();
                // Store the parsed operation so that dependent layers can connect to it
                auto it = m_ParsedMenohOperations.find(name);
                if (it != m_ParsedMenohOperations.end())
                {
		  throw ParseException(boost::str(boost::format("Name %1% used by more than one node") % name));
                }

                m_ParsedMenohOperations[name] = std::move(parsedMenohOperation);
                // If this node was requested as an output from the network then add an ArmNN output layer
                if (std::find(m_RequestedOutputs.begin(), m_RequestedOutputs.end(), name) !=
                    m_RequestedOutputs.end())
                {
                    const LayerBindingId layerId = boost::numeric_cast<LayerBindingId>(m_NetworkOutputsBindingInfo.size());
                    IOutputSlot& prevSlot = parsedMenohOperationRaw->ResolveArmnnOutputSlot(0);

                    TensorInfo tensorInfo = prevSlot.GetTensorInfo();
                    IConnectableLayer* outputLayer = m_Network->AddOutputLayer(layerId, node.output_name_list.at(0).c_str());

                    prevSlot.Connect(outputLayer->GetInputSlot(0));

		    TrackOutputBinding(outputLayer, layerId, tensorInfo);
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

#ifdef ARM_DEBUG
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
            if (!armnnUtils::GraphTopologicalSort<const menoh_impl::node*>(
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
        }
 
        armnn::INetworkPtr MenohParser::CreateNetworkFromGraph(const menoh_impl::graph& graph,
						               std::unordered_map<std::string, array> const& parameter_table,
	                                                       const std::map<std::string, TensorShape>& inputShapes,
                                                               const std::vector<std::string>& requestedOutputs){
            m_Network = INetwork::Create();

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

	    Cleanup();

            return std::move(m_Network);
        }

        BindingPointInfo MenohParser::GetNetworkInputBindingInfo(const std::string& name) const
        {
            return GetBindingInfo(name, "input", m_NetworkInputsBindingInfo);
        }

        BindingPointInfo MenohParser::GetNetworkOutputBindingInfo(const std::string& name) const
        {
            return GetBindingInfo(name, "output", m_NetworkOutputsBindingInfo);
        }

        std::pair<LayerBindingId, TensorInfo> MenohParser::GetBindingInfo(const std::string& layerName,
            const char* bindingPointDesc,
            const std::unordered_map<std::string, BindingPointInfo>& nameToBindingInfo)
        {
            auto it = nameToBindingInfo.find(layerName);
            if (it == nameToBindingInfo.end())
            {
                throw InvalidArgumentException(boost::str(boost::format("Unknown %1% '%2%'") % bindingPointDesc % layerName));
            }
            return it->second;
        }

        void MenohParser::TrackInputBinding(IConnectableLayer* layer, LayerBindingId id, const TensorInfo& tensorInfo)
        {
            return TrackBindingPoint(layer, id, tensorInfo, "input", m_NetworkInputsBindingInfo);
        }

        void MenohParser::TrackOutputBinding(IConnectableLayer* layer, LayerBindingId id, const TensorInfo& tensorInfo)
        {
            return TrackBindingPoint(layer, id, tensorInfo, "output", m_NetworkOutputsBindingInfo);
        }

        void MenohParser::TrackBindingPoint(IConnectableLayer* layer,
            LayerBindingId id,
            const TensorInfo& tensorInfo,
            const char* bindingPointDesc,
            std::unordered_map<std::string, BindingPointInfo>& nameToBindingInfo)
        {
            const std::string layerName = layer->GetName();
            auto it = nameToBindingInfo.find(layerName);
            if (it == nameToBindingInfo.end())
            {
                nameToBindingInfo[layerName] = std::make_pair(id, tensorInfo);
            }
            else
            {
                throw ParseException(boost::str(
                boost::format("Id %1% used by more than one %2% layer") % id % bindingPointDesc));
            }
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
          { "Add",                   &MenohParser::ParseAdd },
          { "BiasAdd",               &MenohParser::ParseBiasAdd },
          { "FC",                    &MenohParser::ParseFC },
          { "Identity",              &MenohParser::ParseIdentity },
          { "Conv",                  &MenohParser::ParseConv2D },
          { "DepthwiseConv2dNative", &MenohParser::ParseDepthwiseConv2D },
          { "FusedBatchNorm",        &MenohParser::ParseFusedBatchNorm },
          { "ConcatV2",              &MenohParser::ParseConcat },
          { "LRN",                   &MenohParser::ParseLrn },
          { "MatMul",                &MenohParser::ParseMatMul },
          { "Mul",                   &MenohParser::ParseMul },
          { "Placeholder",           &MenohParser::ParsePlaceholder },
          { "Relu",                  &MenohParser::ParseRelu },
          { "Relu6",                 &MenohParser::ParseRelu6 },
          { "Reshape",               &MenohParser::ParseReshape },
          { "Shape",                 &MenohParser::ParseShape },
          { "Sigmoid",               &MenohParser::ParseSigmoid },
          { "Softmax",               &MenohParser::ParseSoftmax },
          { "Softplus",              &MenohParser::ParseSoftplus },
          { "Tanh",                  &MenohParser::ParseTanh },
          { "MaxPool",               &MenohParser::ParseMaxPool },
          { "AvgPool",               &MenohParser::ParseAvgPool },
        };

    } // namespace armnn_backend
} // namespace menoh_impl
