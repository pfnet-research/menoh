#pragma once

#include <functional>
#include <map>
#include <vector>

namespace Util
{

namespace
{

enum class NodeState
{
    Visiting,
    Visited,
};

template<typename TNodeId>
bool Visit(
    TNodeId current,
    std::function<std::vector<TNodeId>(TNodeId)> getIncomingEdges,
    std::vector<TNodeId>& outSorted,
    std::map<TNodeId, NodeState>& nodeStates)
{
    auto currentStateIt = nodeStates.find(current);
    if (currentStateIt != nodeStates.end())
    {
        if (currentStateIt->second == NodeState::Visited)
        {
            return true;
        }
        if (currentStateIt->second == NodeState::Visiting)
        {
            return false;
        }
        else
        {
            assert(0);
        }
    }

    nodeStates[current] = NodeState::Visiting;

    for (TNodeId inputNode : getIncomingEdges(current))
    {
        Visit(inputNode, getIncomingEdges, outSorted, nodeStates);
    }

    nodeStates[current] = NodeState::Visited;

    outSorted.push_back(current);
    return true;
}

}

template<typename TNodeId, typename TTargetNodes>
bool GraphTopologicalSort(
    const TTargetNodes& targetNodes,
    std::function<std::vector<TNodeId>(TNodeId)> getIncomingEdges,
    std::vector<TNodeId>& outSorted)
{
    outSorted.clear();
    std::map<TNodeId, NodeState> nodeStates;

    for (TNodeId targetNode : targetNodes)
    {
        if (!Visit(targetNode, getIncomingEdges, outSorted, nodeStates))
        {
            return false;
        }
    }

    return true;
}

}
