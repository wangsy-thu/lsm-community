use std::collections::HashMap;
use crate::comm_table::CommID;

/// Define the community neighbors, represents the neighbors of a given community.
pub(crate) type CommNeighbors = HashMap<CommID, Vec<(u32, u32)>>;

#[allow(dead_code)]
/// Define the community Graph.
pub(crate) struct CommGraph {
    adj_map: HashMap<CommID, Vec<CommID>>, // Adj map of this community graph.
    bridge_map: HashMap<(CommID, CommID), (u32, u32)>, // Record bridges of each edge.
    community_count: u32, // Community count.
    edge_count: u32, // Edge count.
    bridge_count: u32  // Bridge count.
}