
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

#include <boost/format.hpp>
#include <boost/core/ignore_unused.hpp>
#include <boost/log/trivial.hpp>

#include <armnn/ArmNN.hpp>

#include <armnnUtils/Permute.hpp>

#include <menoh/armnn/Parser.hpp>

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
	
        void dumpTensorInfo(TensorInfo& info) {
            std::cout << "   output = " << info.GetNumDimensions() << ", " << info.GetNumElements() << std::endl;
            std::cout << "   outputShape = ";
            for( unsigned int i=0 ; i<info.GetNumDimensions() ; i++ )
                std::cout << info.GetShape()[i] << " ";
            std::cout << std::endl;
        }

        const armnn::PermutationVector NHWCToArmNN = { 0, 2, 3, 1 };
        const armnn::PermutationVector ArmNNToNHWC = { 0, 3, 1, 2 };

        IConnectableLayer* AddSwizzleLayer(INetwork& network, IOutputSlot& input, const PermutationVector& mapping,
            const std::string& name){
            IConnectableLayer* const layer = network.AddPermuteLayer(mapping, name.c_str());

            input.Connect(layer->GetInputSlot(0));

            const TensorInfo outInfo = armnnUtils::Permuted(input.GetTensorInfo(), mapping);
            layer->GetOutputSlot(0).SetTensorInfo(outInfo);

            return layer;
        }

        IConnectableLayer* SwizzleInDeswizzleOut(INetwork& network, IOutputSlot& input, IConnectableLayer& layer,
            const std::string& name){
            IConnectableLayer* const swizzleLayer = AddSwizzleLayer(network, input, NHWCToArmNN, "swizzle_for-" + name);

            swizzleLayer->GetOutputSlot(0).Connect(layer.GetInputSlot(0));

            IConnectableLayer* const deswizzleLayer = AddSwizzleLayer(network, layer.GetOutputSlot(0), ArmNNToNHWC,
                "deswizzle_for-" + name);

            return deswizzleLayer;
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

            const std::string reshapeLayerName = "reshape_for-" + NodeName(node);
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

        class SingleLayerOperation : public Operation{
        public:
            SingleLayerOperation(Parser* parser, const menoh_impl::node& node, IConnectableLayer* layer)
            : Operation(parser, node)
            , m_Layer(layer)
            {
            }

            IOutputSlot& Output(unsigned int index) override
            {
                assert(m_Layer);

                if ((int)index >= m_Layer->GetNumOutputSlots())
                {
                    std::string msg("The requested output slot ");
                    msg += index;
                    msg += "for ";
                    msg += m_Layer->GetName();
                    msg += " does not exist Indext";
                    throw ParseException(msg);
                }
                return m_Layer->GetOutputSlot(index);
            }

        protected:
            IConnectableLayer* m_Layer;
        };

        class DeferredSingleLayerOperation : public SingleLayerOperation {
        public:
            DeferredSingleLayerOperation(Parser* parser, const menoh_impl::node& node)
            : SingleLayerOperation(parser, node, nullptr)
            {
            }

            IOutputSlot& Output(unsigned int index) override
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
            : m_Network(nullptr, nullptr){
        }

        const node* Parser::IdentityNode(const node* node){
            if (node->op_type != "Identity")
            {
                return node;
            }

            if (node->input_name_list.size() != 1)
            {
                throw ParseException("Identity node does not have correct amount of inputs!");
            }

            auto it = m_Nodes.find(node->input_name_list.at(0));
            if (it != m_Nodes.end())
            {
 	        const menoh_impl::node* inputNode = it->second;
                return IdentityNode(inputNode);
            }
            else
            {
                throw ParseException("Cannot find what the Identity node is linked to!");
            }
        }

        IOutputSlot* Parser::GetSlot(OutputOfOperation& input) {
            return &input.m_Value->Output(input.m_Index);
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
            std::vector<OutputOfConstNodeDef> nodes = InputNodes(node);

            const std::size_t numInputs = node.input_name_list.size();
            if (numInputs != expectedNumInputs)
            {
                std::string msg("Unexpected number of inputs for node ");
                msg += name;
                msg += ". Expected ";
                msg += expectedNumInputs;
                msg += ", found ";
                msg += numInputs;
                throw ParseException(msg);
            }

            std::vector<OutputOfOperation> result;
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

#ifdef ARM_DEBUG
            std::cout << std::endl << " [node] : " << node.op_type << " , " << name << std::endl;
            for( unsigned int j=0; j<node.input_name_list.size(); ++j )
                std::cout << "    input : " << node.input_name_list.at(j) << std::endl;

            for( unsigned int j=0; j<node.output_name_list.size(); ++j )
                std::cout << "   output : " << node.output_name_list.at(j) << std::endl;
#endif
            return result;
        }  

        OperationPtr Parser::ParseAdd(const menoh_impl::node& node) {
            std::string name = NodeName(node);
	    
            std::vector<OutputOfOperation> inputs = InputCheck(node, 2);

            if (HasParsedConst(inputs[1]) && inputs[0].m_Value->GetNode().op_type == "MatMul")
            {
                IConnectableLayer* layer =
                    AddFullyConnectedLayer(inputs[0].m_Value->GetNode(), &node, name.c_str());
                return std::make_unique<SingleLayerOperation>(this, node, layer);
            }
            else if (HasParsedConst(inputs[0]) && inputs[1].m_Value->GetNode().op_type == "MatMul")
            {
                IConnectableLayer* layer =
                    AddFullyConnectedLayer(inputs[1].m_Value->GetNode(), &node, name.c_str());
                return std::make_unique<SingleLayerOperation>(this, node, layer);
            }
            else
            {
                return AddAdditionLayer(node);
            }
        }

        OperationPtr Parser::ParseBiasAdd(const menoh_impl::node& node) {
            return AddAdditionLayer(node, true);
        }

        OperationPtr Parser::ParseFC(const menoh_impl::node& node) {
	    
            IConnectableLayer* layer = AddFullyConnectedLayer(node, NodeName(node).c_str());

            return std::make_unique<SingleLayerOperation>(this, node, layer);
        }

        OperationPtr Parser::ParseGemm(const menoh_impl::node& node) {
            std::string name = NodeName(node);
            
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
      
            IConnectableLayer* layer = AddFullyConnectedLayer(node, NodeName(node).c_str());

            return std::make_unique<SingleLayerOperation>(this, node, layer);
        }

        class ParsedIdentityOperation : public Operation {
        public:
            ParsedIdentityOperation(Parser* parser, const menoh_impl::node& node, Operation* op)
                : Operation(parser, node)
                , m_Op(op)
            {
            }

            virtual IOutputSlot& Output(unsigned int index) override
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

        template <typename T>
        class Vector {

        public:

	        Vector( const T* data_ = nullptr, const int size_ = 0 )
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

        template <typename T>
        class ConstOperation : public DeferredSingleLayerOperation {
        public:
            ConstOperation(Parser* parser, const menoh_impl::node& node,
                                      const T* tensorData, const TensorInfo& tensorInfo)
                : DeferredSingleLayerOperation(parser, node),
                m_Storage(tensorData, tensorInfo.GetNumElements()),
                m_TensorInfo(tensorInfo)
            {
                assert(tensorInfo.GetDataType() == GetDataType<T>());
            }

            void CreateLayerDeferred() override
            {
                assert(m_Layer == nullptr);
                m_Layer = m_Parser->m_Network->AddConstantLayer(ConstTensor(m_TensorInfo, m_Storage), NodeName(m_Node).c_str());
                m_Layer->GetOutputSlot(0).SetTensorInfo(m_TensorInfo);
            }

            ConstTensor GetConstTensor(Vector<T>& outputTensorData) const
            {
                const TensorInfo outInfo = m_TensorInfo;

                outputTensorData.set_data(m_Storage.data());
                outputTensorData.set_size(m_Storage.size());

                ConstTensor constTensor(outInfo, outputTensorData);
                return constTensor;
            }

            ConstTensor GetConstTensor(bool swizzleForConvolutionWeights, std::vector<T>& outputTensorData) const
            {
                static const PermutationVector HWIOToOIHW = {2, 3, 1, 0};

                const TensorInfo outInfo = swizzleForConvolutionWeights
                                        ? armnnUtils::Permuted(m_TensorInfo, HWIOToOIHW)
                                        : m_TensorInfo;

                outputTensorData.resize(m_TensorInfo.GetNumElements());

                if (swizzleForConvolutionWeights)
                {
                    armnnUtils::Permute(outInfo.GetShape(), HWIOToOIHW, m_Storage.data(), outputTensorData.data(), sizeof(T));
                }
                else
                {
                    memcpy(outputTensorData.data(), m_Storage.data(), m_TensorInfo.GetNumBytes());
                }
                ConstTensor constTensor(outInfo, outputTensorData);
                return constTensor;
            }

        private:
            Vector<T> m_Storage;
            TensorInfo m_TensorInfo;
        };      

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
            inline static std::unique_ptr<OperatorType<DataType>> Parse(Parser* parser, const menoh_impl::node& node,
                Args&&... args)
            {
                return std::make_unique<OperatorType<DataType>>(parser, node, std::forward<Args>(args)...);
            }
        };
      
        template <>
        struct MakeMenohOperation<ConstOperation> {
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

        OperationPtr Parser::ParseConst(const menoh_impl::node& node) {
            std::string name = NodeName(node);

            assert(node.op_type == "Const");

            auto it = m_Params.find(name);
            if (it == m_Params.end() )
            {
                throw ParseException("ParseConst : not found " + name);
            }

            auto arr  = m_Params[name];
            std::vector<unsigned int> sizes(arr.dims().data(), arr.dims().data()+arr.dims().size());

            unsigned int numElements = 1U;
            {
                auto size = sizes.size();
                for(int i=0 ; i<size ; i++ )
                {
                    numElements *= sizes[i];
                }
            }  

            const DataType dataType = DataType::Float32; // dtype_t::float_:

            const TensorInfo tensorInfo(static_cast<unsigned int>(sizes.size()), sizes.data(), dataType);
#ifdef ARM_DEBUG
            dumpTensorInfo(tensorInfo);
#endif
            Vector<int8_t> tensorData((const int8_t *)arr.data(), numElements*GetDataTypeSize(dataType));
            if (tensorData.size() > tensorInfo.GetNumBytes())
            {
              std::string msg("Number of elements (" + (tensorData.size() / GetDataTypeSize(dataType)));
              msg += ") should be less than or equal to the number of elements implied by the shape argument (";
              msg += tensorInfo.GetNumElements() + ") for Const node - " + name + ")"; 
            }

            return InvokeParseFunction<MakeMenohOperation<ConstOperation>>::Result<OperationPtr>(
                                       dataType, this, node, tensorData, tensorInfo);
        }

        template<typename Type>
        bool Parser::HasParsedConst(const std::string & nodeName) const {
            auto it = m_Operations.find(nodeName);
            return (it != m_Operations.end() &&
                    dynamic_cast<ConstOperation<Type>*>(it->second.get()) != nullptr);
        }

        bool Parser::HasParsedConst(OutputOfOperation& input) {
            return HasParsedConst<float>(NodeName(input.m_Value->GetNode()));
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

            ConstOperation<float>* weightNode = static_cast<ConstOperation<float>*>(inputs[1].m_Value);
            Vector<float> weightTensorData;
            ConstTensor weightTensor = weightNode->GetConstTensor(weightTensorData);

            ConstOperation<float>* biasNode;
            Vector<float> biasTensorData;
            ConstTensor biasTensor;
            if( numInputs == 3 ) {
                biasNode   = static_cast<ConstOperation<float>*>(inputs[2].m_Value);
                biasTensor = biasNode->GetConstTensor(biasTensorData);
            }

            std::vector<int> strides, kernel_shape, pads;
            std::tie(strides, kernel_shape, pads) = attributes_for_2d_data_processing(node);

            std::vector<uint32_t> dilations = (std::vector<uint32_t> const&)attribute_ints( node, "dilations");
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
            desc.m_BiasEnabled = (numInputs == 3) ? true : false;

            std::string dataFormat = "NCHW";

            if (dataFormat == "WHCH")
            {
	        desc.m_StrideX = strides[0];
	        desc.m_StrideY = strides[1];
            }
            else if (dataFormat == "NCHW")
            {
	        desc.m_StrideX = strides[0];
	        desc.m_StrideY = strides[1];
            }
            else
            {
	        throw ParseException("Unsupported data format passed for Conv2D. Only NHWC and NCHW supported");
            }

            IOutputSlot* slot = GetSlot(inputs[0]);
            TensorInfo inputTensorInfo = slot->GetTensorInfo();

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
            if(numInputs == 3){            
                std::cout << ", " << weightTensor.GetShape()[2] << ", " << weightTensor.GetShape()[3] << std::endl;
                std::cout << "       bias(" << biasTensor.GetNumDimensions() << ") = " << biasTensor.GetShape()[0] << std::endl;
            }
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
	    
            IConnectableLayer* layer;
            if( numInputs == 3 )
                layer = m_Network->AddConvolution2dLayer(desc, weightTensor, biasTensor, name.c_str());
            else
                layer = m_Network->AddConvolution2dLayer(desc, weightTensor, name.c_str());
                
            layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

            if (dataFormat == "NHWC")
            {
	        layer = SwizzleInDeswizzleOut(*m_Network, *slot, *layer, name);
            }
            else
            {
                slot->Connect(layer->GetInputSlot(0));
            }
            return std::make_unique<SingleLayerOperation>(this, node, layer);
        }  

        OperationPtr Parser::ParseDepthwiseConv2D(const menoh_impl::node& node) {
            std::string name = NodeName(node);

            std::vector<OutputOfOperation> inputs = InputCheck(node, 2);
            IOutputSlot& inputSlot = inputs[0].m_Value->Output(inputs[0].m_Index);
            TensorInfo inputTensorInfo = inputSlot.GetTensorInfo();

            if (!HasParsedConst(inputs[1]) || !HasParsedConst(inputs[2]))
            {
                throw ParseException("ArmNN only supports Depthwise Convolution layers with constant weights");
            }

            ConstOperation<float>* weightNode = static_cast<ConstOperation<float>*>(inputs[1].m_Value);
            Vector<float> weightTensorData;
            ConstTensor weightTensor = weightNode->GetConstTensor(weightTensorData);

            ConstOperation<float>* biasNode   = static_cast<ConstOperation<float>*>(inputs[2].m_Value);
            Vector<float> biasTensorData;
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

            return std::make_unique<SingleLayerOperation>(this, node, layer);
        }       

        OperationPtr Parser::ParseBatchNormalization(const menoh_impl::node& node) {
            std::string name = NodeName(node);

            std::vector<OutputOfOperation> inputs = InputCheck(node, 5);

            if (!HasParsedConst(inputs[1]))
            {
                throw ParseException("ArmNN only supports BatchNormalization layers with constant scale");
            }
            ConstOperation<float>* scaleNode = static_cast<ConstOperation<float>*>(inputs[1].m_Value);
            std::vector<float> scaleTensorData;
            ConstTensor scaleTensor = scaleNode->GetConstTensor(false, scaleTensorData);

            if (!HasParsedConst(inputs[2]))
            {
                throw ParseException("ArmNN only supports BatchNormalization layers with constant offset");
            }
            ConstOperation<float>* offsetNode = static_cast<ConstOperation<float>*>(inputs[2].m_Value);
            std::vector<float> offsetTensorData;
            ConstTensor offsetTensor = offsetNode->GetConstTensor(false, offsetTensorData);

            if (!HasParsedConst(inputs[3]))
            {
                throw ParseException("ArmNN only supports BatchNormalization layers with constant mean");
            }
            ConstOperation<float>* meanNode = static_cast<ConstOperation<float>*>(inputs[3].m_Value);
            std::vector<float> meanTensorData;
            ConstTensor meanTensor = meanNode->GetConstTensor(false, meanTensorData);

            if (!HasParsedConst(inputs[4]))
            {
                throw ParseException("ArmNN only supports BatchNormalization layers with constant variance");
            }
            ConstOperation<float>* varianceNode = static_cast<ConstOperation<float>*>(inputs[4].m_Value);
            std::vector<float> varianceTensorData;
            ConstTensor varianceTensor = varianceNode->GetConstTensor(false, varianceTensorData);

            BatchNormalizationDescriptor desc;
            desc.m_Eps = attribute_float(node, "epsilon");

            IConnectableLayer* layer = m_Network->AddBatchNormalizationLayer(desc,
                                                                             meanTensor,
                                                                             varianceTensor,
                                                                             offsetTensor,
                                                                             scaleTensor,
                                                                             name.c_str());

            IOutputSlot* output = GetSlot(inputs[0]);

            const std::string dataFormat = "NCHW";

            if (dataFormat == "NHWC")
            {
                const TensorInfo outputTensorInfo = armnnUtils::Permuted(output->GetTensorInfo(), NHWCToArmNN);
                layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);
                layer = SwizzleInDeswizzleOut(*m_Network, *output, *layer, name);
            }
            else
            {
                layer->GetOutputSlot(0).SetTensorInfo(output->GetTensorInfo());
                output->Connect(layer->GetInputSlot(0));
            }

            return std::make_unique<SingleLayerOperation>(this, node, layer);
        }

        OperationPtr Parser::ParseConcatV2(const menoh_impl::node& node) {
            std::string name = NodeName(node);
	    
            std::vector<OutputOfConstNodeDef> nodes = InputNodes(node);

            unsigned int numInputs = static_cast<unsigned int>(nodes.size());
            unsigned int numConcatView = numInputs - 1;

            OriginsDescriptor concatDescriptor(static_cast<uint32_t>(numConcatView), MaxNumOfTensorDimensions);
            std::vector<unsigned int>mergeDimSizes(MaxNumOfTensorDimensions, 0u);

            unsigned int mergeDim = 0;
            std::vector<OutputOfOperation> inputs = InputCheck(node, numInputs);

            if (!HasParsedConst<int32_t>(NodeName(inputs[numInputs - 1].m_Value->GetNode())))
            {
                throw ParseException("ArmNN only supports Concat with constant axis");
            }

            ConstOperation<int32_t>* shapeNode = static_cast<ConstOperation<int32_t>*>(inputs[numInputs - 1].m_Value);

            std::vector<int32_t> axisTensorData;
            ConstTensor axisTensor = shapeNode->GetConstTensor(false, axisTensorData);

            const unsigned int concatDimInput = static_cast<unsigned int>(axisTensorData[0]);

            if (concatDimInput == 0 || concatDimInput == 2)
            {
                throw ParseException("The dimension for concatenation is not supported by Armnn");
            }

            const unsigned int concatDim = 1;
            for (unsigned int viewIndex = 0; viewIndex < numConcatView; ++viewIndex)
            {
                IOutputSlot& inputSlot =
                    inputs[viewIndex].m_Value->Output(inputs[viewIndex].m_Index);
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
                IOutputSlot* slot = GetSlot(inputs[v]);
                if (concatDimInput == 3)
                {
                    IConnectableLayer* const swizzleLayer = AddSwizzleLayer(*m_Network, *slot, NHWCToArmNN,
                                                                            "swizzle_for-" + name);
                    swizzleLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(v));
                }
                else
                {
                    slot->Connect(layer->GetInputSlot(v));
                }
            }

            if (concatDimInput == 3)
            {
                IConnectableLayer* const deswizzleLayer = AddSwizzleLayer(*m_Network, layer->GetOutputSlot(0), ArmNNToNHWC,
									  "deswizzle_for-" + name);
                layer = deswizzleLayer;
            }
            return std::make_unique<SingleLayerOperation>(this, node, layer);
        }
        
        OperationPtr Parser::ParseShape(const menoh_impl::node& node) {

            std::vector<OutputOfOperation> inputs = InputCheck(node, 1);

            IOutputSlot* slot = GetSlot(inputs[0]);
            const TensorInfo& tensorInfo = slot->GetTensorInfo();
            unsigned int dims = tensorInfo.GetNumDimensions();

            std::vector<int32_t> shapeData;
            shapeData.reserve(dims);

            for (unsigned int i=0; i<dims; ++i)
            {
                shapeData.push_back(static_cast<int32_t>(tensorInfo.GetShape()[i]));
            }

            TensorInfo shapeInfo(1, &dims, DataType::Signed32);

            return std::make_unique<ConstOperation<int32_t>>(this, node, &shapeData[0], shapeInfo);
        }

        OperationPtr Parser::ParseReshape(const menoh_impl::node& node) {
	    std::string name = NodeName(node);
	    
            std::vector<OutputOfOperation> inputs = InputCheck(node, 2);
            Operation* inputNode = inputs[0].m_Value;

            if (!HasParsedConst(inputs[1]))
            {
                throw ParseException("ArmNN only supports Reshape layers with constant shapes");
            }
            ConstOperation<int32_t>* shapeNode = static_cast<ConstOperation<int32_t>*>(inputs[1].m_Value);

            armnn::IOutputSlot& prevLayerOutputSlot = inputNode->Output(inputs[0].m_Index);
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

            return std::make_unique<SingleLayerOperation>(this, node, layer);
        }   

        OperationPtr Parser::ParseLrn(const menoh_impl::node& node) {
	    std::string name = NodeName(node);
	    
	    std::vector<OutputOfOperation> inputs = InputCheck(node, 1);

            NormalizationDescriptor desc;
            desc.m_NormMethodType  = NormalizationAlgorithmMethod::LocalBrightness;
            desc.m_NormChannelType = NormalizationAlgorithmChannel::Across;

            desc.m_Alpha    = attribute_float(node, "alpha");
            desc.m_Beta     = attribute_float(node, "beta");
            desc.m_K        = attribute_float(node, "bias");
            desc.m_NormSize = attribute_int(node,   "depth_radius");
            desc.m_NormSize = desc.m_NormSize * 2 + 1;

            IOutputSlot* slot = GetSlot(inputs[0]);

            IConnectableLayer* layer = m_Network->AddNormalizationLayer(desc, name.c_str());

            const TensorInfo permutedInfo = armnnUtils::Permuted(slot->GetTensorInfo(), NHWCToArmNN);
            layer->GetOutputSlot(0).SetTensorInfo(permutedInfo);

            layer = SwizzleInDeswizzleOut(*m_Network, *slot, *layer, name);

            return std::make_unique<SingleLayerOperation>(this, node, layer);
        }
 
        class ParsedMatMulMenohOperation : public DeferredSingleLayerOperation {
        public:
            ParsedMatMulMenohOperation(Parser* parser, const menoh_impl::node& node)
                : DeferredSingleLayerOperation(parser, node) {
            }

            void CreateLayerDeferred() override {
                assert(m_Layer == nullptr);
                m_Layer = m_Parser->AddFullyConnectedLayer(m_Node, nullptr, NodeName(m_Node).c_str());
            }
        };

        OperationPtr Parser::ParseMatMul(const menoh_impl::node& node) {

            return std::make_unique<ParsedMatMulMenohOperation>(this, node);
        }

        OperationPtr Parser::ParseMul(const menoh_impl::node& node) {
	    std::string name = NodeName(node);
	    
            std::vector<OutputOfOperation> inputs = InputCheck(node, 2);

            IConnectableLayer* const layer = m_Network->AddMultiplicationLayer(name.c_str());

            IOutputSlot* input0 = GetSlot(inputs[0]);
            IOutputSlot* input1 = GetSlot(inputs[1]);

            auto const input0NumDims = input0->GetTensorInfo().GetNumDimensions();
            auto const input1NumDims = input1->GetTensorInfo().GetNumDimensions();

            if (input0NumDims < input1NumDims)
            {
                const bool isNHWC = true;
                input0 = BroadcasMenohorAddandMul(input1, input0, isNHWC, *m_Network, node);
            }
            if (input1NumDims < input0NumDims)
            {
                const bool isNHWC = true;
                input1 = BroadcasMenohorAddandMul(input0, input1, isNHWC, *m_Network, node);
            }

            input0->Connect(layer->GetInputSlot(0));
            input1->Connect(layer->GetInputSlot(1));

            if (input0NumDims < input1NumDims)
            {
                layer->GetOutputSlot(0).SetTensorInfo(input1->GetTensorInfo());
            }
            else
            {
                layer->GetOutputSlot(0).SetTensorInfo(input0->GetTensorInfo());
            }
            return std::make_unique<SingleLayerOperation>(this, node, layer);
        }  

        OperationPtr Parser::ParsePlaceholder(const menoh_impl::node& node) {
            std::string name = NodeName(node);
	    
            std::vector<OutputOfOperation> inputs = InputCheck(node, 0);

            const LayerBindingId layerId = static_cast<LayerBindingId>(m_NetworkInputsBindingInfo.size());

            auto dims = (std::vector<uint32_t> const&)attribute_ints(node, "dims");
            const TensorInfo tensorInfo(static_cast<unsigned int>(dims.size()), (const unsigned int*)dims.data(), DataType::Float32);
	    
            IConnectableLayer* const layer = m_Network->AddInputLayer(layerId, name.c_str());

            layer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

            TrackInputBinding(layer, layerId, tensorInfo);

            return std::make_unique<SingleLayerOperation>(this, node, layer);
        }

        OperationPtr Parser::ParseRelu(const menoh_impl::node& node) {
	    
            ActivationDescriptor desc;
            desc.m_Function = ActivationFunction::ReLu;
            return AddActivationLayer(node, desc);
        }

        OperationPtr Parser::ParseRelu6(const menoh_impl::node& node) {

            ActivationDescriptor desc;
            desc.m_Function = ActivationFunction::BoundedReLu;
            desc.m_A = 6.0f;
            desc.m_B = 0.0f;

            return AddActivationLayer(node, desc);
        }

        OperationPtr Parser::ParseSigmoid(const menoh_impl::node& node) {
	    
            ActivationDescriptor desc;
            desc.m_Function = ActivationFunction::Sigmoid;

            return AddActivationLayer(node, desc);
        }

        OperationPtr Parser::ParseSoftplus(const menoh_impl::node& node) {
	    
            ActivationDescriptor desc;
            desc.m_Function = ActivationFunction::SoftReLu;

            return AddActivationLayer(node, desc);
        }    

        OperationPtr Parser::ParseTanh(const menoh_impl::node& node) {
	    
            ActivationDescriptor desc;
            desc.m_Function = ActivationFunction::TanH;
            desc.m_A = 1.0f;
            desc.m_B = 1.0f;

            return AddActivationLayer(node, desc);
        }

        OperationPtr Parser::AddActivationLayer(const menoh_impl::node& node, ActivationDescriptor& desc) {
            std::string name = NodeName(node);

            std::vector<OutputOfOperation> inputs = InputCheck(node, 1);

            IConnectableLayer* const layer = m_Network->AddActivationLayer(desc, name.c_str());

            IOutputSlot* slot = GetSlot(inputs[0]);
            slot->Connect(layer->GetInputSlot(0));
            layer->GetOutputSlot(0).SetTensorInfo(slot->GetTensorInfo());

            return std::make_unique<SingleLayerOperation>(this, node, layer);
        }

        OperationPtr Parser::ParseSoftmax(const menoh_impl::node& node) {
            std::string name = NodeName(node);
	    
            std::vector<OutputOfOperation> inputs = InputCheck(node, 1);

            SoftmaxDescriptor desc;
            IConnectableLayer* const layer = m_Network->AddSoftmaxLayer(desc, name.c_str());

            IOutputSlot* prev = GetSlot(inputs[0]);
            prev->Connect(layer->GetInputSlot(0));
            layer->GetOutputSlot(0).SetTensorInfo(prev->GetTensorInfo());

            return std::make_unique<SingleLayerOperation>(this, node, layer);
        }

        OperationPtr Parser::ParseMaxPool(const menoh_impl::node& node) {
            return ParsePooling2d(node, PoolingAlgorithm::Max);
        }

        OperationPtr Parser::ParseAvgPool(const menoh_impl::node& node) {
            return ParsePooling2d(node, PoolingAlgorithm::Average);
        }          

        OperationPtr Parser::ParsePooling2d(const menoh_impl::node& node, PoolingAlgorithm pooltype){
            std::string name = NodeName(node);
	    
            std::vector<OutputOfOperation> inputs = InputCheck(node, 1);

            std::vector<int> strides, kernel_shape, pads;
            std::tie(strides, kernel_shape, pads) = attributes_for_2d_data_processing(node);
#ifdef ARM_DEBUG
            std::cout << "           strides      = " << strides[0]      << ", " << strides[1]      << std::endl;
            std::cout << "           kernel_shape = " << kernel_shape[0] << ", " << kernel_shape[1] << std::endl;
            std::cout << "           pads         = " << pads[0]         << ", " << pads[1]         << std::endl;
#endif            

            Pooling2dDescriptor desc;
            desc.m_PoolType            = pooltype;
            desc.m_PaddingMethod       = PaddingMethod::Exclude;
            desc.m_OutputShapeRounding = OutputShapeRounding::Floor;

            desc.m_StrideX    = strides[0];
            desc.m_StrideY    = strides[1];
            desc.m_PoolWidth  = kernel_shape[0];
            desc.m_PoolHeight = kernel_shape[1];

            IOutputSlot* slot = GetSlot(inputs[0]);
            TensorInfo inputTensorInfo = slot->GetTensorInfo();

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
                                          inputTensorInfo.GetShape()[1],
                                          static_cast<uint32_t>(ceil(
                                                                static_cast<float>(inputHeight - desc.m_PoolHeight + 1) /
                                                                static_cast<float>(desc.m_StrideY))),
                                          static_cast<uint32_t>(ceil(
                                                                static_cast<float>(inputWidth - desc.m_PoolWidth + 1) /
                                                                static_cast<float>(desc.m_StrideX)))
                                        }, DataType::Float32);
            }

            CalcPadding(inputWidth,  desc.m_PoolWidth, desc.m_StrideX,  desc.m_PadLeft, desc.m_PadRight, padding);
            CalcPadding(inputHeight, desc.m_PoolHeight, desc.m_StrideY, desc.m_PadTop, desc.m_PadBottom, padding);

#ifdef ARM_DEBUG
            std::cout << "   output = " << outputInfo.GetNumDimensions() << ", " << outputInfo.GetNumElements() << std::endl;
            std::cout << "   outputShape = ";
            for( unsigned int i=0 ; i<outputInfo.GetNumDimensions() ; i++ )
                std::cout << outputInfo.GetShape()[i] << " ";
            std::cout << std::endl;
#endif
            IConnectableLayer* layer = m_Network->AddPooling2dLayer(desc, name.c_str());
            if (layer == nullptr)
            {
                throw ParseException("Failed to add pooling2d layer");
            }

            layer->GetOutputSlot(0).SetTensorInfo(outputInfo);

            slot->Connect(layer->GetInputSlot(0));

            return std::make_unique<SingleLayerOperation>(this, node, layer);
        }

        OperationPtr Parser::ParseGlobalMaxPool(const menoh_impl::node& node) {
            return ParseGlobalPooling2d(node, PoolingAlgorithm::Max);
        }

        OperationPtr Parser::ParseGlobalAvgPool(const menoh_impl::node& node) {
            return ParseGlobalPooling2d(node, PoolingAlgorithm::Average);
        }          

        OperationPtr Parser::ParseGlobalPooling2d(const menoh_impl::node& node, PoolingAlgorithm pooltype){
            std::string name = NodeName(node);

            std::vector<OutputOfOperation> inputs = InputCheck(node, 1);

            IOutputSlot* slot = GetSlot(inputs[0]);
            TensorInfo inputTensorInfo = slot->GetTensorInfo();

            uint32_t inputHeight = inputTensorInfo.GetShape()[2];
            uint32_t inputWidth  = inputTensorInfo.GetShape()[3];

            Pooling2dDescriptor desc;
            desc.m_PoolType   = pooltype;
            desc.m_PoolWidth  = inputWidth;
            desc.m_PoolHeight = inputHeight;

            TensorInfo outputTensorInfo;
            outputTensorInfo = TensorInfo({inputTensorInfo.GetShape()[0], inputTensorInfo.GetShape()[1], 1, 1}, DataType::Float32);

#ifdef ARM_DEBUG
            dumpTensorInfo(inputTensorInfo);
            dumpTensorInfo(outputTensorInfo);
#endif

            IConnectableLayer* layer = m_Network->AddPooling2dLayer(desc, name.c_str());
            if (layer == nullptr)
            {
                throw ParseException("Failed to add pooling2d layer");
            }

            layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

            slot->Connect(layer->GetInputSlot(0));

            return std::make_unique<SingleLayerOperation>(this, node, layer);
        }

        OperationPtr Parser::AddAdditionLayer(const menoh_impl::node& node, bool isBiasAdd){
            std::string name = NodeName(node);

            std::vector<OutputOfOperation> inputs = InputCheck(node, 2);

            IOutputSlot* input0 = GetSlot(inputs[0]);
            IOutputSlot* input1 = GetSlot(inputs[1]);

            const TensorInfo& input0Info = input0->GetTensorInfo();
            const TensorInfo& input1Info = input1->GetTensorInfo();

            if (isBiasAdd)
            {
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

                input1 = BroadcasMenohorAddandMul(input0, input1, isNHWC, *m_Network, node);
            }
            else
            {
                if (input0Info.GetNumDimensions() == 1)
                {
                    const bool isNHWC = true;
                    input0 = BroadcasMenohorAddandMul(input1, input0, isNHWC, *m_Network, node);
                }

                if (input1Info.GetNumDimensions() == 1)
                {
                    const bool isNHWC = true;
                    input1 = BroadcasMenohorAddandMul(input0, input1, isNHWC, *m_Network, node);
                }
            }

            IConnectableLayer* const layer = m_Network->AddAdditionLayer(name.c_str());

            input0->Connect(layer->GetInputSlot(0));
            input1->Connect(layer->GetInputSlot(1));

            if (input0Info.GetNumDimensions() == 1 && isBiasAdd == false)
            {
                layer->GetOutputSlot(0).SetTensorInfo(input1->GetTensorInfo());
            }
            else
            {
                layer->GetOutputSlot(0).SetTensorInfo(input0->GetTensorInfo());
            }

            return std::make_unique<SingleLayerOperation>(this, node, layer);
        }

        IConnectableLayer* Parser::AddFullyConnectedLayer(const menoh_impl::node& matMulNodeDef, 
                                                          const menoh_impl::node* addNodeDef, const char* armnnLayerName){
            ConstOperation<float>* biasNode = nullptr;
            if (addNodeDef != nullptr)
            {
                std::vector<OutputOfOperation> addInputs = InputCheck(*addNodeDef, 2);

                if (HasParsedConst(addInputs[0]))
                {
                    biasNode = static_cast<ConstOperation<float>*>(addInputs[0].m_Value);
                }
                else if (HasParsedConst(addInputs[1]))
                {
                    biasNode = static_cast<ConstOperation<float>*>(addInputs[1].m_Value);
                }
                else
                {
                    throw ParseException("ArmNN only supports fully connected layers with constant bias");
                }
            }

            ConstOperation<float>* weightNode = nullptr;
            Operation* inputNode  = nullptr;
            unsigned int inputIdx = 0;

            std::vector<OutputOfOperation> mulInputs = InputCheck(matMulNodeDef, 2);

            if (HasParsedConst(mulInputs[0]))
            {
                weightNode = static_cast<ConstOperation<float>*>(mulInputs[0].m_Value);
                inputNode  = mulInputs[1].m_Value;
                inputIdx   = mulInputs[1].m_Index;
            }
            else if (HasParsedConst(mulInputs[1]))
            {
                weightNode = static_cast<ConstOperation<float>*>(mulInputs[1].m_Value);
                inputNode  = mulInputs[0].m_Value;
                inputIdx   = mulInputs[0].m_Index;
            }
            else
            {
                throw ParseException("ArmNN only supports fully connected layers with constant weights");
            }

            Vector<float> weightTensorData;

            ConstTensor weights = weightNode->GetConstTensor(weightTensorData);

            FullyConnectedDescriptor desc;
            desc.m_BiasEnabled = addNodeDef != nullptr;

            IConnectableLayer* layer = nullptr;

            if (addNodeDef != nullptr)
            {
                Vector<float> biasTensorData;
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

            inputNode->Output(inputIdx).Connect(layer->GetInputSlot(0));
            unsigned int batches = inputNode->Output(inputIdx).GetTensorInfo().GetShape()[0];

            TensorInfo outputInfo({ batches, weights.GetShape()[1] }, DataType::Float32);
            layer->GetOutputSlot(0).SetTensorInfo(outputInfo);
            return layer;
        }

        IConnectableLayer* Parser::AddFullyConnectedLayer(const menoh_impl::node& node, const char* armnnLayerName){
            std::vector<OutputOfOperation> inputs = InputCheck(node, 3);

            unsigned int           inputIdx   = inputs[0].m_Index;
            Operation*             inputNode  = inputs[0].m_Value;

            ConstOperation<float>* weightNode = static_cast<ConstOperation<float>*>(inputs[1].m_Value);
            Vector<float> weightTensorData;
            ConstTensor weights = weightNode->GetConstTensor(weightTensorData);

            ConstOperation<float>* biasNode   = static_cast<ConstOperation<float>*>(inputs[2].m_Value);
            Vector<float> biasTensorData;
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
            inputNode->Output(inputIdx).Connect(layer->GetInputSlot(0));
            unsigned int batches = inputNode->Output(inputIdx).GetTensorInfo().GetShape()[0];

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

        void Parser::LoadNode(const menoh_impl::node& node) {
	    std::string name = NodeName(node);
	    
            dtype_t type = dtype_t::float_;

            const std::string& operation = node.op_type;
            auto it = m_Functions.find(operation);
            if (it != m_Functions.end())
            {
                auto func = it->second;
                OperationPtr parsedMenohOperation = (this->*func)(node);
                Operation* parsedMenohOperationRaw = parsedMenohOperation.get();

                auto it = m_Operations.find(name);
                if (it != m_Operations.end())
                {
		  throw ParseException(boost::str(boost::format("Name %1% used by more than one node") % name));
                }

                m_Operations[name] = std::move(parsedMenohOperation);

                if (std::find(m_Outputs.begin(), m_Outputs.end(), name) !=
                    m_Outputs.end())
                {
                    const LayerBindingId layerId = static_cast<LayerBindingId>(m_NetworkOutputsBindingInfo.size());
                    IOutputSlot& prevSlot = parsedMenohOperationRaw->Output(0);

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

        void Parser::CheckOutput(const menoh_impl::graph& graph, const std::vector<std::string>& outputs) {

            std::vector<const menoh_impl::node*> outputNodes;
            for (const std::string& name : outputs)
            {
                bool found = false;

#ifdef ARM_DEBUG
                std::cout << "OutputName = " << name << std::endl;
#endif
                for( unsigned int i=0; i<graph.node_list().size(); ++i)
                {
                    const menoh_impl::node& node = graph.node_list().at(i);
                     
                    auto nodeIt = std::find(node.output_name_list.begin(), node.output_name_list.end(), name);
                    if (nodeIt != node.output_name_list.end())
                    {
		        outputNodes.push_back(&node);
                        found = true;
                        break;
                    }
                }

                if( !found )
                    throw ParseException("Couldn't find requested output node '" + name + "' in graph");
            }

	    for( auto node : outputNodes )
	      m_Outputs.push_back(NodeName(*node));
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
        }
 
        armnn::INetworkPtr Parser::CreateNetworkFromGraph(const menoh_impl::graph& graph,
						               std::unordered_map<std::string, array> const& parameter_table,
                                                               const std::vector<std::string>& outputs){
            Cleanup();

            m_Network = INetwork::Create();
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

            return std::move(m_Network);
        }

        void Parser::Cleanup(){
            m_Nodes.clear();
            m_Params.clear();
            m_Operations.clear();
        }  

        BindingPointInfo Parser::GetNetworkInputBindingInfo(const std::string& name) const
        {
            return GetBindingInfo(name, "input", m_NetworkInputsBindingInfo);
        }

        BindingPointInfo Parser::GetNetworkOutputBindingInfo(const std::string& name) const
        {
            return GetBindingInfo(name, "output", m_NetworkOutputsBindingInfo);
        }

        std::pair<LayerBindingId, TensorInfo> Parser::GetBindingInfo(const std::string& layerName,
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

        void Parser::TrackInputBinding(IConnectableLayer* layer, LayerBindingId id, const TensorInfo& tensorInfo)
        {
            TrackBindingPoint(layer, id, tensorInfo, "input", m_NetworkInputsBindingInfo);
        }

        void Parser::TrackOutputBinding(IConnectableLayer* layer, LayerBindingId id, const TensorInfo& tensorInfo)
        {
            TrackBindingPoint(layer, id, tensorInfo, "output", m_NetworkOutputsBindingInfo);
        }

        void Parser::TrackBindingPoint(IConnectableLayer* layer,
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

        const std::map<std::string, Parser::ParseFunction> Parser::m_Functions = {
	  { "Const",                 &Parser::ParseConst },
          { "Add",                   &Parser::ParseAdd },
          { "Sum",                   &Parser::ParseAdd },
          { "BiasAdd",               &Parser::ParseBiasAdd },
          { "FC",                    &Parser::ParseFC },
          { "Gemm",                  &Parser::ParseGemm },
          { "Identity",              &Parser::ParseIdentity },
          { "Conv",                  &Parser::ParseConv2D },
          { "DepthwiseConv2dNative", &Parser::ParseDepthwiseConv2D },
          { "BatchNormalization",    &Parser::ParseBatchNormalization },
          { "ConcatV2",              &Parser::ParseConcatV2 },
          { "LRN",                   &Parser::ParseLrn },
          { "MatMul",                &Parser::ParseMatMul },
          { "Mul",                   &Parser::ParseMul },
          { "Placeholder",           &Parser::ParsePlaceholder },
          { "Relu",                  &Parser::ParseRelu },
          { "Relu6",                 &Parser::ParseRelu6 },
          { "Reshape",               &Parser::ParseReshape },
          { "Shape",                 &Parser::ParseShape },
          { "Sigmoid",               &Parser::ParseSigmoid },
          { "Softmax",               &Parser::ParseSoftmax },
          { "Softplus",              &Parser::ParseSoftplus },
          { "Tanh",                  &Parser::ParseTanh },
          { "MaxPool",               &Parser::ParseMaxPool },
          { "AveragePool",           &Parser::ParseAvgPool },
          { "GlobalMaxPool",         &Parser::ParseGlobalMaxPool },
          { "GlobalAveragePool",     &Parser::ParseGlobalAvgPool },
        };

    } // namespace armnn_backend
} // namespace menoh_impl
