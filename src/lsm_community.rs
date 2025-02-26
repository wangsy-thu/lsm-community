use std::borrow::Borrow;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use dashmap::DashMap;
use dashmap::mapref::one::RefMut;

use crate::attribute::{AttrStorage, AttrValue};
use crate::bucket::{CommBucket, TreeBucket};
use crate::comm_table::{CommID, CommTable};
use crate::community::CommNeighbors;
use crate::community_centric::escape_seed::{MoveTaskController, MoveTaskOption};
use crate::community_centric::L0CommMaintainState;
use crate::compact::CompactController;
use crate::config::{MAX_ENTRY_SIZE, MAX_MEM_SIZE};
use crate::graph::{Graph, GraphSnapshot, VInt};
use crate::louvain::Louvain;
use crate::mem_graph::MemGraph;
use crate::sgm_type::Graph as GupGraph;
use crate::subgraph_matching::{MatchingEngine, SearchOptions};
use crate::txn_manage::LsmTxnInner;
use crate::types::{GraphEntry, V32, Vertex};

pub enum WriteBatchRecord {
    Put(V32, V32),
    Del(V32),
}

/// The struct to record the storage state of LSM-Community.
#[allow(dead_code)]
pub struct LSMStorageState {
    mem_graph: Arc<RwLock<MemGraph>>, // The memory graph part.
    bucket_map: Arc<DashMap<CommID, CommBucket>>, // Community buckets.
    pub comm_table: Arc<RwLock<CommTable>>, // Community table, to locate communities.
    immutable_mem_graph: Arc<RwLock<Vec<Arc<MemGraph>>>>, // Immutable memory graphs.
    l1_bucket: Arc<RwLock<TreeBucket>> // The Tree bucket.
    // In the future, the thread pool and the lock will be included.
}

/// Wrap LSM-Community with txn.
pub struct LSMStorageOuter {
    pub lsm_storage_inner: Arc<LSMStorage>,

}

/// The LSM-Community final storage, including the storage state.
#[allow(dead_code)]
pub struct LSMStorage {
    /// The dataset name.
    pub dataset_name: String,
    /// The LSM Community Instance.
    pub lsm_storage_state: Arc<LSMStorageState>,
    /// The attribute storage, i.e., an LSM Tree.
    pub attr_storage: Arc<AttrStorage>,
    /// Notifies the memory graph need to be flushed.
    pub flush_notifier: crossbeam_channel::Sender<()>,
    /// The handle for the flush thread.
    pub flush_thread: thread::JoinHandle<()>,
    /// Notifies the dynamic community structure thread.
    pub dyn_comm_notifier: crossbeam_channel::Sender<()>,
    /// The handle for the dynamic community structure.
    pub community_thread: thread::JoinHandle<()>,
    /// The txn manager part.
    pub txn_manager: LsmTxnInner
}

#[allow(dead_code)]
impl LSMStorage {
    /// Display all data on the disk, i.e, the buckets.
    pub(crate) fn print_disk(&self) {
        for bucket_entry in self.lsm_storage_state.bucket_map.iter() {
            println!("Comm ID: {}", bucket_entry.key());
            println!("==================");
            bucket_entry.value().display_bucket();
        }
    }

    pub fn txn_manager_ref(&self) -> &LsmTxnInner {
        self.txn_manager.borrow()
    }

    /// Perform community search for lsm-community.
    pub fn get_community_with(&self, vertex_id: &VInt) -> Option<Vec<VInt>> {
        // Step 1, lookup from the community table.
        let comm_table_guard;
        let target_comm_id;
        {
            comm_table_guard = self.lsm_storage_state.comm_table.read().unwrap();
            match comm_table_guard.look_up_community(vertex_id) {
                None => {
                    println!("Vertex: {} not exists in this system", vertex_id);
                    return None;
                }
                Some(comm_id) => {
                    // Get the target graph.
                    // Firstly, assume that all the whole community is on the disk.
                    target_comm_id = comm_id;
                }
            }
        }
        let bucket =
            self.lsm_storage_state.bucket_map.get(&target_comm_id).unwrap();
        Some(bucket.value().load_community().adj_map.keys().cloned().collect())
    }

    pub fn write_batch_inner(&self, batch: &[WriteBatchRecord]) -> Result<u64> {
        let _lck = self.txn_manager_ref().write_lock.lock();
        let ts = self.txn_manager_ref().latest_commit_ts() + 1;
        for record in batch {
            match record {
                WriteBatchRecord::Del(_) => {

                }
                WriteBatchRecord::Put(key, value) => {
                    self.put_edge(key.vertex_id, value.vertex_id);
                }
            }
        }
        self.txn_manager_ref().update_commit_ts(ts);
        Ok(ts)
    }

    /// Perform subgraph matching with the query graph and the data graph.
    /// Return the result mappings.
    pub fn query_subgraph(&self, _query_graph: &GupGraph) -> Duration {
        // The author of GCF do not want to share the code.
        Duration::from_secs(10)
    }

    /// Perform subgraph matching with the query graph and the data graph.
    /// Return the result mappings.
    pub fn query_subgraph_gup(&self, query_graph: &GupGraph) -> Duration {
        let start = Instant::now();
        // Step 1, prepare the data graph in memory.
        // For every vertex, build the neighbor list.
        let mut opt = SearchOptions::default();
        let mut neighbors_map = BTreeMap::<VInt, Vec<VInt>>::new();
        for mut_entry in self.lsm_storage_state.bucket_map.iter() {
            // Load the graph in the bucket to the memory.
            let community = mut_entry.value().load_bucket();
            for (vertex_id, (_, neighbors)) in community.adj_map {
                neighbors_map.insert(vertex_id, neighbors.iter().fold(
                    BTreeSet::<VInt>::new(),
                    |mut acc, item| {
                        acc.insert(item.vertex_id);
                        acc
                    }
                ).clone().into_iter().collect::<Vec<_>>());
            }
        }
        let mut neighbors = vec![];
        let mut labels = vec![];
        let mut n_edges = 0usize;
        for (_, neighbor_list) in neighbors_map.into_iter() {
            n_edges += neighbor_list.len();
            neighbors.push(neighbor_list);
        }
        // Process the labels.
        let label_map = self.attr_storage.read_all_vertex_attr();
        for (_, vertex_attr) in label_map {
            match vertex_attr {
                AttrValue::Num(v_label) => {
                    labels.push(v_label);
                }
                AttrValue::Str(_) => {}
            }
        }
        let n_labels = labels.iter().collect::<HashSet<_>>().len();
        assert_eq!(labels.len(), neighbors.len());
        let data_graph = GupGraph {
            neighbors,
            labels,
            n_edges,
            n_labels
        };
        // Step 2, Build Gup for subgraph matching.
        opt.parallelism = 1;
        let gup = MatchingEngine::new(&data_graph, opt);
        // Step 3, Perform subgraph matching.
        gup.search(query_graph, |_|{
            true
        });
        start.elapsed()
    }

    /// Perform subgraph matching in a specific community.
    pub fn query_subgraph_in_community(
        &self,
        query_graph: &GupGraph,
        vertex_id: &VInt
    ) -> Duration {
        // Step 1, prepare the data graph in memory.
        // For every vertex, build the neighbor list.
        let mut neighbors_map = BTreeMap::<VInt, Vec<VInt>>::new();
        // Grab the target bucket.
        match self.get_community_graph_with(vertex_id) {
            None => {
                println!("Vertex not exist.")
            }
            Some(target_community) => {
                for (vertex_id, (_, neighbors)) in
                    target_community.adj_map {
                    neighbors_map.insert(vertex_id, neighbors.iter().map(|item| item.vertex_id).collect());
                }
            }
        }
        let mut neighbors = vec![];
        let mut labels = vec![];
        let mut n_edges = 0usize;
        for (_, neighbor_list) in neighbors_map.into_iter() {
            n_edges += neighbor_list.len();
            neighbors.push(neighbor_list);
            labels.push(0u32);
        }
        let data_graph = GupGraph {
            neighbors,
            labels,
            n_edges,
            n_labels: 1
        };

        // Step 2, Build Gup for subgraph matching.
        let opt = SearchOptions::default();
        let gup = MatchingEngine::new(&data_graph, opt);

        // Step 3, Perform subgraph matching.
        let start = Instant::now();
        // Step 3, Perform subgraph matching.
        gup.search(query_graph, |_|{
            true
        });
        start.elapsed()
    }

    /// Get the community which contains specific vertex.
    pub(crate) fn get_community_graph_with(&self, vertex_id: &VInt) -> Option<Graph> {
        // Step 1, lookup from the community table.
        match self.lsm_storage_state.comm_table.read().unwrap().look_up_community(vertex_id) {
            None => {
                println!("Vertex: {} not exists in this system", vertex_id);
                None
            }
            Some(comm_id) => {
                // Get the target graph.
                // Firstly, assume that all the whole community is on the disk.
                let bucket =
                    self.lsm_storage_state.bucket_map.get(&comm_id).unwrap();
                Some(bucket.value().load_community())
            }
        }
    }

    /// Lock the memory table.
    /// Move the old memory table to un-flushed list.
    /// Clear the old one.
    /// Notify the flush thread.
    pub(crate) fn trigger_flush(&self) {
        // Step 1. Attach the write lock to the memory graph.
        let mut mem_guard = self.lsm_storage_state.mem_graph.write().unwrap();
        let un_flushed_mem_graph = mem_guard.clone();
        let mut immutable_mem_guard =
            self.lsm_storage_state.immutable_mem_graph.write().unwrap();
        // Step 2. Move the old memory table to un-flushed list.
        immutable_mem_guard.push(Arc::new(un_flushed_mem_graph));

        // Step 3. Clear the old one.
        mem_guard.clear();
        // Step 4. Notify the flush thread.
        self.flush_notifier.send(()).unwrap();
        // for imm_graph in immutable_mem_guard.iter() {
        //     println!("An immutable memory graph flushed.");
        //     self.lsm_storage_state.group_flush(Arc::clone(imm_graph));
        //     // Remove it from the list.
        // }
        // immutable_mem_guard.clear();
    }

    /// Trigger the dynamic community detection thread work.
    pub fn trigger_dyn_community(&self) {
        self.dyn_comm_notifier.send(()).unwrap();
    }

    pub(crate) fn read_successor(&mut self, vertex_id: &VInt) -> Vec<V32> {
        let all_neighbors = self.read_neighbors(vertex_id);
        let mut result_successors = vec![];
        for neighbor in all_neighbors.into_iter() {
            if neighbor.direction_tag == 1 {
                result_successors.push(neighbor);
            }
        }
        result_successors
    }

    /// Read all neighbors from LSM-Community System.
    pub(crate) fn read_neighbors(&self, vertex_id: &VInt) -> Vec<V32> {
        let mut result_neighbors = Vec::new();
        // Step 1. Check community table.
        if let Some(comm_id) =
            self.lsm_storage_state.comm_table.read().unwrap().look_up_community(vertex_id) {
            {
                // Step 2. Check the memory graph.
                let memory_neighbor_guard =
                    self.lsm_storage_state.mem_graph.read().unwrap();
                let immutable_mem_guard =
                    self.lsm_storage_state.immutable_mem_graph.read().unwrap();
                let mut memory_neighbors =
                    memory_neighbor_guard.get_all_mem_neighbor(vertex_id);
                result_neighbors.append(&mut memory_neighbors);
                // Step 3. Check the immutable mem_graph list.
                for imm_mem_graph in immutable_mem_guard.iter() {
                    let mut imm_neighbors =
                        imm_mem_graph.get_mem_neighbor(vertex_id);
                    result_neighbors.append(&mut imm_neighbors);
                }
            }

            // Step 4. Check the community bucket.
            let mut target_bucket =
                self.lsm_storage_state.bucket_map.get_mut(&comm_id).unwrap();
            let mut neighbors_in_bucket = target_bucket.value_mut().find_neighbors(&vertex_id);
            result_neighbors.append(&mut neighbors_in_bucket);
        }
        CompactController::execute_compact_neighbors(&result_neighbors)
        //result_neighbors
    }

    // Perform SCC for LSM-Community.
    pub fn scc(&mut self) -> Vec<Vec<VInt>> {
        // Prepare some variables.
        // The level number, representing the current level of DFS.
        let mut level_index = 0u32;
        // The stack array.
        let mut stack = vec![];
        // The array recording the level of each vertex.
        let mut dfn = HashMap::<VInt, u32>::new();
        // The visited set to record whether a vertex is visited.
        let mut visited = HashSet::<VInt>::new();
        // Record whether a vertex is on stack.
        let mut is_stacked = HashSet::<VInt>::new();
        // Record the low-link value of a vertex.
        let mut low = HashMap::<VInt, u32>::new();
        // The result scc list which is wanted.
        let mut result_scc_list = vec![];

        // Attention: No need to read all the vertices, just check the visit table.
        let vertex_ids_to_check: Vec<VInt> = {
            let comm_table_guard = self.lsm_storage_state.comm_table.read().unwrap();
            // Fetch the un visited vertices.
            comm_table_guard
                .get_vertex_map_ref()
                .keys()
                .filter(|vertex_id| !visited.contains(*vertex_id))
                .cloned()
                .collect()
        };

        // For all vertices, perform tarjan to avoid wcc.
        for v in vertex_ids_to_check {
            if !visited.contains(&v) {
                self.tarjan_dfs(
                    &v, &mut level_index, &mut visited, &mut stack,
                    &mut is_stacked, &mut dfn, &mut low, &mut result_scc_list
                )
            }
        }
        result_scc_list
    }

    // Perform Tarjan for a grid-storage engine.
    fn tarjan_dfs(
        &mut self,
        start_vertex: &VInt,
        current_level: &mut u32,
        is_visited: &mut HashSet<VInt>,
        stack: &mut Vec<VInt>,
        is_stacked: &mut HashSet<VInt>,
        dfn: &mut HashMap<VInt, u32>,
        low: &mut HashMap<VInt, u32>,
        result_scc_list: &mut Vec<Vec<VInt>>
    ) {
        // Visit the start vertex.
        is_visited.insert(*start_vertex);
        // Record the level and low value of the start vertex.
        dfn.insert(*start_vertex, *current_level);
        low.insert(*start_vertex, *current_level);
        *current_level += 1;
        // Push this vertex on the stack, and mark it as 'stacked'.
        stack.push(*start_vertex);
        is_stacked.insert(*start_vertex);

        // Perform dfs.
        for neighbor in self.read_successor(start_vertex) {
            // If it is not visited, perform dfs.
            if !is_visited.contains(&neighbor.vertex_id) {
                self.tarjan_dfs(
                    &neighbor.vertex_id, current_level, is_visited,
                    stack, is_stacked, dfn, low, result_scc_list
                );
                if low.get(start_vertex).unwrap() > low.get(&neighbor.vertex_id).unwrap() {
                    low.insert(
                        *start_vertex,
                        *low.get(&neighbor.vertex_id).unwrap()
                    );
                }
            } else if is_stacked.contains(&neighbor.vertex_id) &&
                low.get(start_vertex).unwrap() > low.get(&neighbor.vertex_id).unwrap() {
                low.insert(
                    *start_vertex,
                    *low.get(&neighbor.vertex_id).unwrap()
                );
            }
        }

        // Check whether is a scc.
        if dfn.get(start_vertex).unwrap() == low.get(start_vertex).unwrap() {
            let mut result_scc = vec![];
            loop {
                let w = stack.pop().unwrap();
                is_stacked.remove(&w);
                result_scc.push(w);
                if w == *start_vertex {
                    break;
                }
            }
            // The 'result scc' is moved.
            result_scc_list.push(result_scc);
        }
    }

    /// Insert a new vertex to the graph.
    pub(crate) fn put_vertex(&mut self, vertex_id: VInt) {
        // Problem. How to determine the community of this vertex.
        println!("Put Vertex: {}", vertex_id);
    }

    /// Delete the vertex given the ID of it.
    pub(crate) fn delete_vertex(&mut self, vertex_id: VInt) {
        println!("Delete Vertex: {}", vertex_id);
    }

    /// Insert a new edge to this graph.
    pub fn put_edge(&self, src_id: VInt, dst_id: VInt) {
        // Step 1. Look up community table to check whether they exist.
        let src_dst_valid =
            match self.lsm_storage_state.comm_table.read().unwrap().look_up_community(&src_id) {
            None => {
                println!("Src {} not valid.", src_id);
                None
            }
            Some(_) => {
                match self.lsm_storage_state.comm_table.read().unwrap().look_up_community(&dst_id) {
                    None => {
                        println!("Src {} not valid.", src_id);
                        None
                    }
                    Some(_) => {
                        Some(())
                    }
                }
            }
        };
        match src_dst_valid {
            None => {
                println!("Inserted edge invalid.")
            }
            Some(_) => {
                // This edge can be inserted.
                // Step 2. Insert it to memory table, if full, perform flush.
                // Build the source vertex.
                // Acquire locks (Avoid deadlock, add a field).
                let mem_graph_size;
                {
                    let mut mem_write_guard =
                        self.lsm_storage_state.mem_graph.write().unwrap();
                    let src_vertex = V32::new_vertex(src_id);
                    // Successor Vertex.
                    let dst_successor = V32::new_successor(dst_id);
                    // Build the destination vertex.
                    let dst_vertex = V32::new_vertex(dst_id);
                    // Predecessor Vertex.
                    let src_predecessor = V32::new_predecessor(src_id);
                    mem_write_guard.put_edge(src_vertex, dst_successor);
                    mem_write_guard.put_edge(dst_vertex, src_predecessor);

                    mem_graph_size = mem_write_guard.get_used_size();
                }

                // If the memory graph is full, just flush it into the buckets.
                if mem_graph_size > MAX_MEM_SIZE {
                    self.trigger_flush();
                }
            }
        }
    }

    /// Insert the edge directly into memory graph without flushing.
    pub fn put_to_mem_graph(&self, src_vertex: V32, dst_vertex: V32) {
        // Step 1. Look up community table to check whether they exist.
        let src_dst_valid =
            match self.lsm_storage_state.comm_table.read().unwrap().look_up_community(&src_vertex.vertex_id) {
                None => {
                    println!("Src {} not valid.", src_vertex.vertex_id);
                    None
                }
                Some(_) => {
                    match self.lsm_storage_state.comm_table.read().unwrap().look_up_community(&dst_vertex.vertex_id) {
                        None => {
                            println!("Dst {} not valid.", dst_vertex.vertex_id);
                            None
                        }
                        Some(_) => {
                            Some(())
                        }
                    }
                }
            };
        match src_dst_valid {
            None => {
                println!("Inserted edge invalid.")
            }
            Some(_) => {
                // This edge can be inserted.
                {
                    let mut mem_write_guard =
                        self.lsm_storage_state.mem_graph.write().unwrap();

                    // Put the edge into the memory graph.
                    mem_write_guard.put_edge(src_vertex.clone(), dst_vertex.clone());
                }
            }
        }
    }

    /// Delete an edge from the memory graph.
    pub fn delete_edge_from_memory(&mut self, src_id: VInt, dst_id: VInt) {
        // Step 1. Look up community table to check whether they exist.
        let src_dst_valid =
            match self.lsm_storage_state.comm_table.read().unwrap().look_up_community(&src_id) {
                None => {
                    println!("Src {} not valid.", src_id);
                    None
                }
                Some(_) => {
                    match self.lsm_storage_state.comm_table.read().unwrap().look_up_community(&dst_id) {
                        None => {
                            println!("Src {} not valid.", src_id);
                            None
                        }
                        Some(_) => {
                            Some(())
                        }
                    }
                }
            };
        match src_dst_valid {
            None => {
                println!("Inserted edge invalid.")
            }
            Some(_) => {
                // This edge can be inserted.
                {
                    let mut mem_write_guard =
                        self.lsm_storage_state.mem_graph.write().unwrap();
                    let src_vertex = V32::new_vertex(src_id);
                    // Successor Vertex.
                    let dst_successor = V32::new_tomb(dst_id, 1);
                    // Build the destination vertex.
                    let dst_vertex = V32::new_vertex(dst_id);
                    // Predecessor Vertex.
                    let src_predecessor = V32::new_tomb(src_id, 2);
                    mem_write_guard.put_edge(src_vertex, dst_successor);
                    mem_write_guard.put_edge(dst_vertex, src_predecessor);
                }
            }
        }
    }

    /// Insert the edge directly into memory graph without flushing.
    pub fn put_edge_to_memory(&mut self, src_id: VInt, dst_id: VInt) {
        // Step 1. Look up community table to check whether they exist.
        let src_dst_valid =
            match self.lsm_storage_state.comm_table.read().unwrap().look_up_community(&src_id) {
                None => {
                    println!("Src {} not valid.", src_id);
                    None
                }
                Some(_) => {
                    match self.lsm_storage_state.comm_table.read().unwrap().look_up_community(&dst_id) {
                        None => {
                            println!("Src {} not valid.", src_id);
                            None
                        }
                        Some(_) => {
                            Some(())
                        }
                    }
                }
            };
        match src_dst_valid {
            None => {
                println!("Inserted edge invalid.")
            }
            Some(_) => {
                // This edge can be inserted.
                {
                    let mut mem_write_guard =
                        self.lsm_storage_state.mem_graph.write().unwrap();
                    let src_vertex = V32::new_vertex(src_id);
                    // Successor Vertex.
                    let dst_successor = V32::new_successor(dst_id);
                    // Build the destination vertex.
                    let dst_vertex = V32::new_vertex(dst_id);
                    // Predecessor Vertex.
                    let src_predecessor = V32::new_predecessor(src_id);
                    mem_write_guard.put_edge(src_vertex, dst_successor);
                    mem_write_guard.put_edge(dst_vertex, src_predecessor);
                }
            }
        }
    }

    /// Put an edge into lsm-community directly, i.e., to disk.
    pub fn put_edge_to_buckets(&mut self, src_id: VInt, dst_id: VInt) {
        // Step 1. Look up community table to check whether they exist.
        let src_dst_valid =
            match self.lsm_storage_state.comm_table.read().unwrap().look_up_community(&src_id) {
                None => {
                    println!("Src {} not valid.", src_id);
                    None
                }
                Some(src_comm_id) => {
                    match self.lsm_storage_state.comm_table.read().unwrap().look_up_community(&dst_id) {
                        None => {
                            println!("Src {} not valid.", src_id);
                            None
                        }
                        Some(dst_comm_id) => {
                            Some((src_comm_id, dst_comm_id))
                        }
                    }
                }
            };
        match src_dst_valid {
            None => {
                println!("Inserted edge invalid.")
            }
            Some((src_comm_id, dst_comm_id)) => {
                {
                    // This edge can be inserted.
                    let src_vertex = V32::new_vertex(src_id);
                    // Successor Vertex.
                    let dst_successor = V32::new_successor(dst_id);
                    // Build the destination vertex.
                    let dst_vertex = V32::new_vertex(dst_id);
                    // Predecessor Vertex.
                    let src_predecessor = V32::new_predecessor(src_id);

                    // Build the graph entry.
                    let entry_to_src_bucket = GraphEntry::create(
                        src_vertex, vec![dst_successor]
                    );
                    let entry_to_dst_bucket = GraphEntry::create(
                        dst_vertex, vec![src_predecessor]
                    );

                    match self.lsm_storage_state.bucket_map.get_mut(&src_comm_id) {
                        None => {
                            panic!("Community bucket lost!");
                        }
                        Some(mut bucket) => {
                            bucket.value_mut().insert_entry(entry_to_src_bucket);
                        }
                    }

                    match self.lsm_storage_state.bucket_map.get_mut(&dst_comm_id) {
                        None => {
                            panic!("Community bucket lost!");
                        }
                        Some(mut bucket) => {
                            bucket.value_mut().insert_entry(entry_to_dst_bucket);
                        }
                    }
                }
            }
        }
    }

    /// Delete an edge given their IDs.
    pub(crate) fn delete_edge(&mut self, src_id: VInt, dst_id: VInt) {
        // Step 1. Look up community table to check whether they exist.
        let src_dst_valid =
            match self.lsm_storage_state.comm_table.read().unwrap().look_up_community(&src_id) {
                None => {
                    println!("Src {} not valid.", src_id);
                    None
                }
                Some(_) => {
                    match self.lsm_storage_state.comm_table.read().unwrap().look_up_community(&dst_id) {
                        None => {
                            println!("Src {} not valid.", src_id);
                            None
                        }
                        Some(_) => {
                            Some(())
                        }
                    }
                }
            };
        match src_dst_valid {
            None => {
                println!("Inserted edge invalid.")
            }
            Some(_) => {
                // This edge can be inserted.
                // Step 2. Insert it to memory table, if full, perform flush.
                // Build the source vertex.
                // Acquire locks (Avoid deadlock, add a field).
                let mem_graph_size;
                {
                    let mut mem_write_guard =
                        self.lsm_storage_state.mem_graph.write().unwrap();
                    let src_vertex = V32::new_vertex(src_id);
                    // Successor Vertex.
                    let dst_successor = V32::new_tomb(dst_id, 1u8);
                    // Build the destination vertex.
                    let dst_vertex = V32::new_vertex(dst_id);
                    // Predecessor Vertex.
                    let src_predecessor = V32::new_tomb(src_id, 2u8);
                    mem_write_guard.put_edge(src_vertex, dst_successor);
                    mem_write_guard.put_edge(dst_vertex, src_predecessor);

                    mem_graph_size = mem_write_guard.get_used_size();
                }

                // If the memory graph is full, just flush it into the buckets.
                if mem_graph_size > MAX_MEM_SIZE {
                    self.trigger_flush();
                }
            }
        }
    }

    pub fn delete_edge_from_bucket(&mut self, src_id: VInt, dst_id: VInt) {
        // Step 1. Look up community table to check whether they exist.
        let src_dst_valid =
            match self.lsm_storage_state.comm_table.read().unwrap().look_up_community(&src_id) {
                None => {
                    println!("Src {} not valid.", src_id);
                    None
                }
                Some(src_comm_id) => {
                    match self.lsm_storage_state.comm_table.read().unwrap().look_up_community(&dst_id) {
                        None => {
                            println!("Src {} not valid.", src_id);
                            None
                        }
                        Some(dst_comm_id) => {
                            Some((src_comm_id, dst_comm_id))
                        }
                    }
                }
            };
        match src_dst_valid {
            None => {
                println!("Inserted edge invalid.")
            }
            Some((src_comm_id, dst_comm_id)) => {
                {
                    // This edge can be inserted.
                    let src_vertex = V32::new_vertex(src_id);
                    // Successor Vertex.
                    let dst_successor = V32::new_tomb(dst_id, 1u8);
                    // Build the destination vertex.
                    let dst_vertex = V32::new_vertex(dst_id);
                    // Predecessor Vertex.
                    let src_predecessor = V32::new_tomb(src_id, 2u8);

                    // Build the graph entry.
                    let entry_to_src_bucket = GraphEntry::create(
                        src_vertex, vec![dst_successor]
                    );
                    let entry_to_dst_bucket = GraphEntry::create(
                        dst_vertex, vec![src_predecessor]
                    );

                    match self.lsm_storage_state.bucket_map.get_mut(&src_comm_id) {
                        None => {
                            panic!("Community bucket lost!");
                        }
                        Some(mut bucket) => {
                            bucket.value_mut().insert_entry(entry_to_src_bucket);
                        }
                    }

                    match self.lsm_storage_state.bucket_map.get_mut(&dst_comm_id) {
                        None => {
                            panic!("Community bucket lost!");
                        }
                        Some(mut bucket) => {
                            bucket.value_mut().insert_entry(entry_to_dst_bucket);
                        }
                    }
                }
            }
        }
    }

    /// Using bfs to walk through a wcc.
    fn bfs_component(
        &mut self,
        start_vertex: &VInt,
        visited_vertex_list: &mut HashSet<VInt>,
        result: &mut Vec<VInt>
    ) {
        let mut queue = VecDeque::new();
        queue.push_back(*start_vertex);
        visited_vertex_list.insert(*start_vertex);

        // Start to walk.
        while let Some(v) = queue.pop_front() {
            result.push(v);
            let neighbors = self.read_neighbors(&v);
            for neighbor in neighbors {
                if !visited_vertex_list.contains(&neighbor.vertex_id) {
                    queue.push_back(neighbor.vertex_id);
                    visited_vertex_list.insert(neighbor.vertex_id);
                }
            }
        }
    }

    /// Perform ssp for lsm-community.
    pub fn sssp(&mut self, vertex_id: &VInt) -> HashMap<VInt, i32> {
        let mut distances: HashMap<u32, i32> = HashMap::new();
        let mut queue: VecDeque<u32> = VecDeque::new();

        distances.insert(*vertex_id, 0);
        queue.push_back(*vertex_id);

        while let Some(current_node) = queue.pop_front() {
            let current_distance = *distances.get(&current_node).unwrap();

            for neighbor in self.read_successor(&current_node) {
                if !distances.contains_key(&neighbor.vertex_id) {
                    distances.insert(neighbor.vertex_id, current_distance + 1);
                    queue.push_back(neighbor.vertex_id);
                }
            }
        }
        distances
    }

    // Perform bfs for lsm-community.
    pub fn bfs(&mut self, start_vertex: &VInt) -> Vec<VInt> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        self.bfs_component(start_vertex, &mut visited, &mut result);

        // Acquire the read lock of the community table.
        // Attention: No need to read all the vertices, just check the visit table.
        let vertex_ids_to_check: Vec<VInt> = {
            let comm_table_guard = self.lsm_storage_state.comm_table.read().unwrap();
            // Fetch the un visited vertices.
            comm_table_guard
                .get_vertex_map_ref()
                .keys()
                .filter(|vertex_id| !visited.contains(*vertex_id))
                .cloned()
                .collect()
        };

        // Check success.
        for vertex_id in &vertex_ids_to_check {
            if !visited.contains(vertex_id) {
                self.bfs_component(vertex_id, &mut visited, &mut result);
            }
        }
        // After the guard escapes from this function, the lock is freed.
        result
    }

    /// Perform WCC for lsm-community.
    pub fn wcc(&mut self) -> Vec<Vec<VInt>> {
        let mut visited = HashSet::new();
        let mut wcc = vec![];

        let vertex_ids_to_check: Vec<VInt> = {
            let comm_table_guard = self.lsm_storage_state.comm_table.read().unwrap();
            // Fetch the un visited vertices.
            comm_table_guard
                .get_vertex_map_ref()
                .keys()
                .filter(|vertex_id| !visited.contains(*vertex_id))
                .cloned()
                .collect()
        };

        // Check success.
        for v_index in &vertex_ids_to_check {
            if !visited.contains(v_index) {
                let mut result = Vec::new();
                self.bfs_component(v_index, &mut visited, &mut result);
                wcc.push(result);
            }
        }
        wcc
    }

    /// Get the community split of the stored graph.
    pub fn get_community_split(&self) -> Vec<Vec<VInt>> {
        let mut result_comm_split = vec![];
        // Traverse all the buckets.
        for bucket_entry_ref in self.lsm_storage_state.bucket_map.iter() {
            let community = bucket_entry_ref.value().load_community();
            // Collect all the vertices.
            result_comm_split.push(community.adj_map.keys().cloned().collect())
        }
        result_comm_split
    }
}

#[allow(dead_code)]
impl LSMStorageState {

    pub fn build_from_graph_file_comm (
        file_name: &str,
        graph_name: &str,
        have_label: bool
    ) -> (LSMStorageState, Option<Vec<u32>>) {
        let (g_mem, comm_structure, label_opt)
            = Graph::from_graph_file_community(file_name, true, have_label);
        // Create the community table.
        let comm_table = CommTable::build_from_comm_structure(&comm_structure);

        // Generate community buckets according to the community structure.
        let comm_map = DashMap::<CommID, CommBucket>::new();

        // Create an empty community graph.
        let mut community_graph = BTreeMap::<VInt, CommNeighbors>::new();

        for community in comm_structure {
            // Perform lookup, to find the community id (id must exist in community table).
            let comm_id = comm_table.look_up_community(&community[0]).unwrap();
            // Build the graph of this community.
            let mut comm_adj_map =
                BTreeMap::<VInt, (Vertex<u32>, Vec<Vertex<u32>>)>::new();
            for vertex_id in community {
                let mut comm_neighbor = HashMap::<CommID, Vec<(VInt, VInt)>>::new();
                for neighbor_vid in &g_mem.get_successor(&vertex_id) {
                    if let Some(neighbor_comm_id) = comm_table.look_up_community(&neighbor_vid.vertex_id) {
                        if neighbor_comm_id != comm_id {
                            // Find a neighbor community.
                            if comm_neighbor.contains_key(&neighbor_comm_id) {
                                comm_neighbor.get_mut(&neighbor_comm_id).unwrap()
                                    .push((vertex_id, neighbor_vid.vertex_id));
                            } else {
                                comm_neighbor.insert(neighbor_comm_id,
                                                     vec![(vertex_id, neighbor_vid.vertex_id)]);
                            }
                        }
                    }
                }
                community_graph.insert(comm_id, comm_neighbor);
                comm_adj_map.insert(vertex_id, (g_mem.get_vertex(&vertex_id).unwrap(),
                                                g_mem.get_neighbor(&vertex_id)));

            }
            let comm_g = Graph {
                adj_map: comm_adj_map,
                e_size: 0,
                v_size: 0
            };
            let bucket = CommBucket::build_bucket_from_community(&comm_g,
                                                                 &format!("lsm.db/comm_{}_{}.db", graph_name, comm_id));
            // collect each community.
            comm_map.insert(comm_id, bucket);  // The 'bucket' here is moved.
        }

        // Now, build the community graph.
        let tree_bucket = TreeBucket::build_bucket_from_neighbors(
            graph_name, 1, 1, &community_graph
        );

        (LSMStorageState {
            mem_graph: Arc::new(RwLock::new(MemGraph::create(0))),
            bucket_map: Arc::new(comm_map),
            comm_table: Arc::new(RwLock::new(comm_table)),
            immutable_mem_graph: Arc::new(RwLock::new(Vec::new())),
            l1_bucket: Arc::new(RwLock::new(tree_bucket)),
        }, label_opt)
    }

    pub fn build_from_graph_file(file_name: &str, graph_name: &str) -> LSMStorageState {
        // Build the initialized system from a graph.
        // Firstly, perform a community detection algorithm, to get community structure.
        let g_mem = Graph::from_graph_file(file_name, true);
        let (mut lg, vid_arr) = g_mem.generate_louvain_graph();
        let louvain = Louvain::new(&mut lg);
        let (hierarchy, _) = louvain.run();
        // Print the hierarchy.
        // Perform Louvain.
        let mut target_layer = hierarchy.len() - 1;
        let mut target_layer_comm = Vec::<Vec<usize>>::new();
        let mut comm_structure_layer = Vec::new();
        let app_comm_num = (g_mem.v_size as f64).sqrt().floor() as usize;
        for (layer, comm_split) in hierarchy.iter().enumerate() {
            let mut comm_v_map = HashMap::<usize, Vec<usize>>::new();
            for (v_id, comm_id) in comm_split.iter().enumerate() {
                comm_v_map.entry(*comm_id).or_insert(vec![]).push(v_id);
            }
            if comm_v_map.len() <= 2 * app_comm_num || layer == hierarchy.len() - 1 {
                // build comm structure.
                for (_, comm_v_list) in comm_v_map {
                    target_layer_comm.push(comm_v_list)
                }
                target_layer = layer;
                break;
            }
            comm_structure_layer.push(comm_v_map);
        }
        comm_structure_layer.reverse();
        for i in 0..target_layer {
            let sub_comm_structure = &comm_structure_layer[i];
            let mut new_target_cs = vec![];
            for comm in target_layer_comm {
                let mut new_comm = vec![];
                for sub_comm_id in comm {
                    new_comm.extend_from_slice(sub_comm_structure.get(&sub_comm_id).unwrap());
                }
                new_target_cs.push(new_comm);
            }
            target_layer_comm = new_target_cs;
        }

        let comm_structure: Vec<_> = target_layer_comm.into_iter().map(|comm| {
            comm.into_iter().map(|v| {
                vid_arr[v]
            }).collect()
        }).collect();

        // Create the community table.
        let comm_table = CommTable::build_from_comm_structure(&comm_structure);

        // Generate community buckets according to the community structure.
        let comm_map = DashMap::<CommID, CommBucket>::new();

        // Create an empty community graph.
        let mut community_graph = BTreeMap::<VInt, CommNeighbors>::new();

        for community in comm_structure {
            // Perform lookup, to find the community id (id must exist in community table).
            let comm_id = comm_table.look_up_community(&community[0]).unwrap();
            // Build the graph of this community.
            let mut comm_adj_map =
                BTreeMap::<VInt, (Vertex<u32>, Vec<Vertex<u32>>)>::new();
            for vertex_id in community {
                let mut comm_neighbor = HashMap::<CommID, Vec<(VInt, VInt)>>::new();
                for neighbor_vid in &g_mem.get_successor(&vertex_id) {
                    if let Some(neighbor_comm_id) = comm_table.look_up_community(&neighbor_vid.vertex_id) {
                        if neighbor_comm_id != comm_id {
                            // Find a neighbor community.
                            if comm_neighbor.contains_key(&neighbor_comm_id) {
                                comm_neighbor.get_mut(&neighbor_comm_id).unwrap()
                                    .push((vertex_id, neighbor_vid.vertex_id));
                            } else {
                                comm_neighbor.insert(neighbor_comm_id,
                                                             vec![(vertex_id, neighbor_vid.vertex_id)]);
                            }
                        }
                    }
                }
                community_graph.insert(comm_id, comm_neighbor);
                comm_adj_map.insert(vertex_id, (g_mem.get_vertex(&vertex_id).unwrap(),
                                                g_mem.get_neighbor(&vertex_id)));

            }
            let comm_g = Graph {
                adj_map: comm_adj_map,
                e_size: 0,
                v_size: 0
            };
            let bucket = CommBucket::build_bucket_from_community(&comm_g,
                                                                 &format!("lsm.db/comm_{}_{}.db", graph_name, comm_id));
            // collect each community.
            comm_map.insert(comm_id, bucket);  // The 'bucket' here is moved.
        }

        // Now, build the community graph.
        let tree_bucket = TreeBucket::build_bucket_from_neighbors(
            graph_name, 1, 1, &community_graph
        );

        LSMStorageState {
            mem_graph: Arc::new(RwLock::new(MemGraph::create(0))),
            bucket_map: Arc::new(comm_map),
            comm_table: Arc::new(RwLock::new(comm_table)),
            immutable_mem_graph: Arc::new(RwLock::new(Vec::new())),
            l1_bucket: Arc::new(RwLock::new(tree_bucket)),
        }
    }

    /// Flush the entries in the memory graph to the community bucket
    /// according to the community table.
    pub(crate) fn group_flush(&self, flushed_mem_graph: Arc<MemGraph>) {
        // For each vertex and its neighbors.
        for(vertex_id, (vertex, neighbors)) in
            flushed_mem_graph.adj_ref() {
            // Build graph entry for each vertex.
            let neighbor_list = neighbors.iter().fold(
                Vec::new(),
                |mut n_list, neighbor| {
                    n_list.push(neighbor.1.clone());
                    n_list
                }
            );
            let raw_entry = GraphEntry::create(vertex.clone(), neighbor_list);
            let inserted_entry_list = CommBucket::graph_partition(raw_entry, MAX_ENTRY_SIZE);

            // Perform lookup, get the corresponding community bucket.
            match self.comm_table.read().unwrap().look_up_community(vertex_id) {
                None => {
                    // Return, this insertion is invalid.
                    panic!("Community bucket lost!");
                }
                Some(comm_id) => {
                    match self.bucket_map.get_mut(&comm_id) {
                        None => {
                            panic!("Community bucket lost!");
                        }
                        Some(mut bucket) => {
                            bucket.value_mut().insert_entries_batch(inserted_entry_list);
                        }
                    }
                }
            }
        }
    }

    /// Load a L0 level community through the community ID.
    pub(crate) fn load_community(&self, comm_id: &CommID) -> GraphSnapshot {
        let bucket_ref = self.bucket_map.get(comm_id).unwrap();
        bucket_ref.value().generate_snapshot()
    }

    /// Lookup the target community through the vertex ID.
    pub(crate) fn lookup_community(&self, vertex_id: &VInt) -> Option<CommID> {
        let comm_table_ref = self.comm_table.read().unwrap();
        comm_table_ref.look_up_community(vertex_id)
    }

    /// Get the mutable reference of the community bucket and its community id.
    pub(crate) fn get_bucket_mut(&self, comm_id: &CommID) -> Option<RefMut<VInt, CommBucket>> {
        self.bucket_map.get_mut(comm_id)
    }

    /// Get the community count in L0 level.
    pub(crate) fn get_l0_comm_count(&self) -> u32 {
        self.bucket_map.len() as u32
    }

    /// Generate the dynamic community thread.
    pub fn spawn_dynamic_community_thread(
        self: &Arc<Self>,
        rx: crossbeam_channel::Receiver<()>
    ) -> Option<thread::JoinHandle<()>> {
        let this = self.clone();
        let handle = thread::spawn(move || {
            // Main steps of the dynamic community thread.
            match rx.recv() {
                Ok(_) => {
                    // Create a new community move state.
                    let init_maintain_comm_id;
                    {
                        let comm_table_guard = this.comm_table.read().unwrap();
                        init_maintain_comm_id = comm_table_guard.vertex_community_map.iter().next().map(
                            |(_, comm_id)| *comm_id
                        ).unwrap();
                    }
                    let main_state = Arc::new(L0CommMaintainState {
                        last_maintained_comm_id: Arc::new(Mutex::new(init_maintain_comm_id))
                    });

                    // Create a new community move controller.
                    let move_task_ctl = MoveTaskController::new(
                        MoveTaskOption{
                            move_task_on: false
                        }
                    );
                    loop {
                        // println!("Perform move.");
                        move_task_ctl.perform_move(&this, &main_state);
                        thread::sleep(Duration::from_secs(1));
                    }
                }
                Err(_) => {}
            }
        });
        Some(handle)
    }

    /// Generate the flush thread.
    pub fn spawn_flush_thread(
        self: &Arc<Self>,
        rx: crossbeam_channel::Receiver<()>
    ) -> Option<thread::JoinHandle<()>> {
        let this = self.clone();
        let handle = thread::spawn(move || {
            loop {
                match rx.recv() {
                    Ok(_) => {
                        match this.immutable_mem_graph.write() {
                            Ok(mut imm_graph_list) => {
                                for imm_graph in imm_graph_list.iter() {
                                    println!("An immutable memory graph flushed.");
                                    this.group_flush(Arc::clone(imm_graph));
                                    // Remove it from the list.
                                }
                                imm_graph_list.clear();
                            }
                            Err(_) => {}
                        }

                    }
                    Err(_) => {
                        // This channel is closed, break.
                        break;
                    }
                }
            }
        });
        Some(handle)
    }
}

#[cfg(test)]
mod test {
    use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
    use std::sync::Arc;
    use std::time::Instant;

    use rand::Rng;
    use rand::seq::SliceRandom;

    use crate::attribute::{AttrStorage, AttrStorageConfig};
    use crate::graph::{Graph, VInt};
    use crate::lsm_community::{LSMStorage, LSMStorageState};
    use crate::sgm_type::Graph as GupGraph;
    use crate::txn_manage::LsmTxnInner;

    #[test]
    fn test_load_example() {
        let lsm_community =
            Arc::new(LSMStorageState::build_from_graph_file("data/example.graph", "example_load"));
        let attr_storage_instance = AttrStorage::create("example_load".to_owned(), AttrStorageConfig {
            cache_capacity: 1000 * 1000 * 512,
            flush_frequency: 1000,
        });
        attr_storage_instance.load_attribute_from_file("data/example.graph");
        let attr_storage = Arc::new(attr_storage_instance);
        let (flush_notifier, flush_receiver) = crossbeam_channel::unbounded();
        let (dyn_comm_notifier, dyn_comm_receiver) = crossbeam_channel::unbounded();
        let flush_thread_new = lsm_community.spawn_flush_thread(flush_receiver).unwrap();
        let dy_comm_thread_new = lsm_community.spawn_dynamic_community_thread(dyn_comm_receiver).unwrap();
        let lsm_storage = LSMStorage {
            dataset_name: "example".to_owned(),
            lsm_storage_state: lsm_community,
            attr_storage,
            flush_notifier,
            flush_thread: flush_thread_new,
            dyn_comm_notifier,
            community_thread: dy_comm_thread_new,
            txn_manager: LsmTxnInner::new(0)
        };
        // Start dynamic community detection after building the LSM-Community storage.
        lsm_storage.trigger_dyn_community();
        lsm_storage.print_disk();
        lsm_storage.lsm_storage_state.l1_bucket.read().unwrap().print_bucket();
    }

    #[test]
    fn test_load_comm_example() {
        let (lsm_community_inst, label_opt) =
            LSMStorageState::build_from_graph_file_comm("data/oregon.graph", "oregon_load", true);
        let lsm_community = Arc::new(lsm_community_inst);
        let attr_storage_instance = AttrStorage::create("oregon_load".to_owned(), AttrStorageConfig {
            cache_capacity: 1000 * 1000 * 512,
            flush_frequency: 1000,
        });
        attr_storage_instance.load_attribute_from_vec(label_opt.unwrap());
        let attr_storage = Arc::new(attr_storage_instance);
        let (flush_notifier, flush_receiver) = crossbeam_channel::unbounded();
        let (dyn_comm_notifier, dyn_comm_receiver) = crossbeam_channel::unbounded();
        let flush_thread_new = lsm_community.spawn_flush_thread(flush_receiver).unwrap();
        let dy_comm_thread_new = lsm_community.spawn_dynamic_community_thread(dyn_comm_receiver).unwrap();
        let lsm_storage = LSMStorage {
            dataset_name: "oregon".to_owned(),
            lsm_storage_state: lsm_community,
            attr_storage,
            flush_notifier,
            flush_thread: flush_thread_new,
            dyn_comm_notifier,
            community_thread: dy_comm_thread_new,
            txn_manager: LsmTxnInner::new(0)
        };
        // Start dynamic community detection after building the LSM-Community storage.
        lsm_storage.trigger_dyn_community();
        lsm_storage.print_disk();
        lsm_storage.lsm_storage_state.l1_bucket.read().unwrap().print_bucket();
    }

    #[test]
    fn test_load_community() {
        let lsm_community =
            Arc::new(LSMStorageState::build_from_graph_file("data/oregon.graph", "oregon_for_community"));
        let attr_storage_instance = AttrStorage::create("oregon_for_community".to_owned(), AttrStorageConfig {
            cache_capacity: 1000 * 1000 * 512,
            flush_frequency: 1000,
        });
        attr_storage_instance.load_attribute_from_file("data/oregon.graph");
        let attr_storage = Arc::new(attr_storage_instance);
        let (flush_notifier, flush_receiver) = crossbeam_channel::unbounded();
        let (dyn_comm_notifier, dyn_comm_receiver) = crossbeam_channel::unbounded();
        let flush_thread_new = lsm_community.spawn_flush_thread(flush_receiver).unwrap();
        let dy_comm_thread_new = lsm_community.spawn_dynamic_community_thread(dyn_comm_receiver).unwrap();
        let lsm_storage = LSMStorage {
            dataset_name: "oregon".to_owned(),
            lsm_storage_state: lsm_community,
            attr_storage,
            flush_notifier,
            flush_thread: flush_thread_new,
            dyn_comm_notifier,
            community_thread: dy_comm_thread_new,
            txn_manager: LsmTxnInner::new(0)
        };
        // Start dynamic community detection after building the LSM-Community storage.
        lsm_storage.trigger_dyn_community();

        let comm = lsm_storage.get_community_graph_with(&3).unwrap();
        comm.print_graph();
        let comm = lsm_storage.get_community_graph_with(&111);
        match comm {
            None => {
                // Test pass.
            }
            Some(_) => {

            }
        }

        let start = Instant::now();
        let comm_target = lsm_storage.get_community_with(&1).unwrap();

        let duration = start.elapsed();
        println!("CD Time: {} ms.", duration.as_millis());
        println!("Community Count: {}", comm_target.len());
    }

    #[test]
    fn test_find_neighbors() {
        let lsm_community =
            Arc::new(LSMStorageState::build_from_graph_file("data/example.graph", "example_for_query"));
        let attr_storage_instance = AttrStorage::create("example_for_query".to_owned(), AttrStorageConfig {
            cache_capacity: 1000 * 1000 * 512,
            flush_frequency: 1000,
        });
        attr_storage_instance.load_attribute_from_file("data/example.graph");
        let attr_storage = Arc::new(attr_storage_instance);
        let (flush_notifier, flush_receiver) = crossbeam_channel::unbounded();
        let (dyn_comm_notifier, dyn_comm_receiver) = crossbeam_channel::unbounded();
        let flush_thread_new = lsm_community.spawn_flush_thread(flush_receiver).unwrap();
        let dy_comm_thread_new = lsm_community.spawn_dynamic_community_thread(dyn_comm_receiver).unwrap();
        let lsm_storage = LSMStorage {
            dataset_name: "example".to_owned(),
            lsm_storage_state: lsm_community,
            attr_storage,
            flush_notifier,
            flush_thread: flush_thread_new,
            dyn_comm_notifier,
            community_thread: dy_comm_thread_new,
            txn_manager: LsmTxnInner::new(0)
        };
        // Start dynamic community detection after building the LSM-Community storage.
        lsm_storage.trigger_dyn_community();

        let g_mem = Graph::from_graph_file("data/example.graph", true);
        for (v_id, (_, ground_truth)) in &g_mem.adj_map {
            let predict = lsm_storage.read_neighbors(v_id);
            assert_eq!(ground_truth.len(), predict.len());
        }
    }

    #[test]
    fn test_subgraph_query() {
        let lsm_community =
            Arc::new(LSMStorageState::build_from_graph_file("data/brightkite.graph", "brightkite_for_sub_graph"));
        let attr_storage_instance = AttrStorage::create("email-enron_for_sub_graph".to_owned(), AttrStorageConfig {
            cache_capacity: 1000 * 1000 * 512,
            flush_frequency: 1000,
        });
        attr_storage_instance.load_attribute_from_file("data/brightkite.graph");
        let attr_storage = Arc::new(attr_storage_instance);
        let (flush_notifier, flush_receiver) = crossbeam_channel::unbounded();
        let (dyn_comm_notifier, dyn_comm_receiver) = crossbeam_channel::unbounded();
        let flush_thread_new = lsm_community.spawn_flush_thread(flush_receiver).unwrap();
        let dy_comm_thread_new = lsm_community.spawn_dynamic_community_thread(dyn_comm_receiver).unwrap();
        let lsm_storage = LSMStorage {
            dataset_name: "brightkite".to_owned(),
            lsm_storage_state: lsm_community,
            attr_storage,
            flush_notifier,
            flush_thread: flush_thread_new,
            dyn_comm_notifier,
            community_thread: dy_comm_thread_new,
            txn_manager: LsmTxnInner::new(0)
        };
        // Start dynamic community detection after building the LSM-Community storage.
        lsm_storage.trigger_dyn_community();

        // Load the query graph.
        let query_graph = Graph::from_graph_file("query/brightkite/8s/brightkite_8s_0.graph", true);
        let mut neighbors_map = BTreeMap::<VInt, Vec<VInt>>::new();
        for (vertex_id, (_, neighbors)) in query_graph.adj_map {
            neighbors_map.insert(vertex_id, neighbors.iter().fold(
                BTreeSet::<VInt>::new(),
                |mut acc, item| {
                    acc.insert(item.vertex_id);
                    acc
                }
            ).clone().into_iter().collect::<Vec<_>>());
        }

        let mut neighbors = vec![];
        let labels = Graph::read_label_array("query/brightkite/8s/brightkite_8s_0.graph");
        let n_labels = labels.iter().collect::<HashSet<_>>().len();
        let mut n_edges = 0usize;
        for (_, neighbor_list) in neighbors_map.into_iter() {
            n_edges += neighbor_list.len();
            neighbors.push(neighbor_list);
        }
        println!("Neighbor map: {:?}", neighbors);
        let query_graph_gup = GupGraph {
            neighbors,
            labels,
            n_edges,
            n_labels
        };
        let duration = lsm_storage.query_subgraph(&query_graph_gup);
        println!("subgraph matching time: {}ms", duration.as_millis());
    }

    #[test]
    fn test_insert_delete_edge() {
        let lsm_community =
            Arc::new(LSMStorageState::build_from_graph_file("data/oregon.graph", "oregon_test_insert_delete"));
        let attr_storage_instance = AttrStorage::create("oregon_test_insert_delete".to_owned(), AttrStorageConfig {
            cache_capacity: 1000 * 1000 * 512,
            flush_frequency: 1000,
        });
        attr_storage_instance.load_attribute_from_file("data/oregon.graph");
        let attr_storage = Arc::new(attr_storage_instance);
        let (flush_notifier, flush_receiver) = crossbeam_channel::unbounded();
        let (dyn_comm_notifier, dyn_comm_receiver) = crossbeam_channel::unbounded();
        let flush_thread_new = lsm_community.spawn_flush_thread(flush_receiver).unwrap();
        let dy_comm_thread_new = lsm_community.spawn_dynamic_community_thread(dyn_comm_receiver).unwrap();
        let mut lsm_storage = LSMStorage {
            dataset_name: "oregon".to_owned(),
            lsm_storage_state: lsm_community,
            attr_storage,
            flush_notifier,
            flush_thread: flush_thread_new,
            dyn_comm_notifier,
            community_thread: dy_comm_thread_new,
            txn_manager: LsmTxnInner::new(0)
        };
        // Start dynamic community detection after building the LSM-Community storage.
        lsm_storage.trigger_dyn_community();

        println!("Start to Insert Edges in LSM-Community");
        // Prepare the random number generator.
        let mut rng = rand::thread_rng();
        // Load the graph.
        let mut graph_example = Graph::from_graph_file("data/oregon.graph", true);
        // Determine the edges can be added.
        let vertex_count = graph_example.v_size;
        let mut inserted_graph = HashMap::<VInt, Vec<(VInt, u8)>>::new();
        for vertex_id in 0..vertex_count {
            let mut added_count = 0;
            loop {
                let added_neighbor = rng.gen_range(0..vertex_count);
                let mut neighbor_exists = false;
                // Validate this neighbor is existing.
                for neighbor in &graph_example.get_neighbor(&vertex_id) {
                    if neighbor.vertex_id == added_neighbor && neighbor.direction_tag == 1 {
                        neighbor_exists = true;
                        break;
                    }
                }
                if !neighbor_exists {
                    inserted_graph.entry(vertex_id).or_insert(vec![]).push((added_neighbor, 1));
                    inserted_graph.entry(added_neighbor).or_insert(vec![]).push((vertex_id, 2));
                    graph_example.insert_edge(vertex_id, added_neighbor);
                    // Perform insert for lsm-community.
                    lsm_storage.put_edge(vertex_id, added_neighbor);
                    added_count += 1;
                }
                if added_count >= 5 {
                    break;
                }
            }
        }

        let mut deleted_graph = HashMap::<VInt, Vec<(VInt, u8)>>::new();
        let remove_count = 2;
        for vertex_id in 0..vertex_count {
            for _ in 0..remove_count {
                match graph_example.get_successor(&vertex_id).choose(&mut rng) {
                    None => {
                        // println!("This vertex {} has no successors.", vertex_id);
                        break;
                    }
                    Some(removed_neighbor) => {
                        graph_example.remove_edge(&vertex_id, &removed_neighbor.vertex_id);
                        deleted_graph.entry(vertex_id).or_insert(vec![]).push((removed_neighbor.vertex_id, 1));
                        deleted_graph.entry(removed_neighbor.vertex_id).or_insert(vec![]).push((vertex_id, 2));
                        lsm_storage.delete_edge(vertex_id, removed_neighbor.vertex_id);
                    }
                }
            }
        }

        // Validate the correctness of the edge insertion.
        for tested_vid in 0..10000 {
            //println!("Testing V{}", tested_vid);
            // Performing the test.
            let res = lsm_storage.read_neighbors(&tested_vid);

            // Check the neighbors.
            let ground_truth = graph_example.get_neighbor(&tested_vid);

            // V \in U and |V| = |U| => V = U (In Algebra.)
            for v in &res {
                if !ground_truth.contains(v) {
                    let gt_vec = ground_truth.iter().map(|item| (item.vertex_id, item.direction_tag)).collect::<Vec<_>>();
                    println!("GT: {:?}", gt_vec);
                    let rs_vec = res.iter().map(|item| (item.vertex_id, item.direction_tag)).collect::<Vec<_>>();
                    println!("RS: {:?}", rs_vec);
                    // Display the difference of the ground truth and the result.
                    let gt_set: HashSet<_> = gt_vec.clone().into_iter().collect();
                    let rs_set: HashSet<_> = rs_vec.clone().into_iter().collect();

                    let diff1: Vec<_> = gt_set.difference(&rs_set).cloned().collect();
                    let diff2: Vec<_> = rs_set.difference(&gt_set).cloned().collect();

                    println!("Diff1: {:?}", diff1);
                    println!("Diff2: {:?}", diff2);

                    // let removed_neighbors = updated_entry_vec_map.get(&tested_vid).unwrap().1.iter().filter(
                    //     |&item| item.tomb == 1
                    // ).cloned().collect::<Vec<_>>();
                    // println!("Removed neighbors: {:?}", removed_neighbors.iter().map(|item| (item.vertex_id, item.direction_tag)).collect::<Vec<_>>());

                }
                assert!(ground_truth.contains(v));
            }
            assert_eq!(res.len(), ground_truth.len());

            // Display the results.
        }
        println!("Rebuild, Find Neighbor test pass!");
    }

    #[test]
    fn test_insert_edge() {
        let lsm_community =
            Arc::new(LSMStorageState::build_from_graph_file("data/oregon.graph", "oregon_test_insert_lll"));
        let attr_storage_instance = AttrStorage::create("oregon_test_insert_lll".to_owned(), AttrStorageConfig {
            cache_capacity: 1000 * 1000 * 512,
            flush_frequency: 1000,
        });
        attr_storage_instance.load_attribute_from_file("data/oregon.graph");
        let attr_storage = Arc::new(attr_storage_instance);
        let (flush_notifier, flush_receiver) = crossbeam_channel::unbounded();
        let (dyn_comm_notifier, dyn_comm_receiver) = crossbeam_channel::unbounded();
        let flush_thread_new = lsm_community.spawn_flush_thread(flush_receiver).unwrap();
        let dy_comm_thread_new = lsm_community.spawn_dynamic_community_thread(dyn_comm_receiver).unwrap();
        let lsm_storage = LSMStorage {
            dataset_name: "oregon".to_owned(),
            lsm_storage_state: lsm_community,
            attr_storage,
            flush_notifier,
            flush_thread: flush_thread_new,
            dyn_comm_notifier,
            community_thread: dy_comm_thread_new,
            txn_manager: LsmTxnInner::new(0)
        };
        // Start dynamic community detection after building the LSM-Community storage.
        lsm_storage.trigger_dyn_community();

        println!("Start to Insert Edges in LSM-Community");
        // Prepare the random number generator.
        let mut rng = rand::thread_rng();
        // Load the graph.
        let mut graph_example = Graph::from_graph_file("data/oregon.graph", true);
        // Determine the edges can be added.
        let vertex_count = graph_example.v_size;

        let start = Instant::now();
        let mut inserted_edge_count = 0u32;
        let mut inserted_graph = HashMap::<VInt, Vec<(VInt, u8)>>::new();
        for vertex_id in 0..vertex_count {
            let mut added_count = 0;
            loop {
                let added_neighbor = rng.gen_range(0..vertex_count);
                let mut neighbor_exists = false;
                // Validate this neighbor is existing.
                for neighbor in &graph_example.get_neighbor(&vertex_id) {
                    if neighbor.vertex_id == added_neighbor && neighbor.direction_tag == 1 {
                        neighbor_exists = true;
                        break;
                    }
                }
                if !neighbor_exists {
                    inserted_graph.entry(vertex_id).or_insert(vec![]).push((added_neighbor, 1));
                    inserted_graph.entry(added_neighbor).or_insert(vec![]).push((vertex_id, 2));
                    graph_example.insert_edge(vertex_id, added_neighbor);
                    // Perform insert for lsm-community.
                    lsm_storage.put_edge(vertex_id, added_neighbor);
                    added_count += 1;
                    inserted_edge_count += 1;
                }
                if added_count >= 10 {
                    break;
                }
            }
        }

        let mut lost_edges = vec![];

        // Validate the correctness of the edge insertion.
        for tested_vid in 1..10000 {
            // Performing the test.
            let res = lsm_storage.read_neighbors(&tested_vid);

            // Check the neighbors.
            let ground_truth = graph_example.get_neighbor(&tested_vid);

            // V \in U and |V| = |U| => V = U (In Algebra.)
            for v in &res {
                assert!(ground_truth.contains(&v));
            }

            // Display the results.
            if ground_truth.len() != res.len() {
                // Display the inserted edges.
                println!("Testing Vertex: {}", tested_vid);
                let inserted_neighbors = inserted_graph.get(&tested_vid).unwrap();
                println!("Inserted Neighbors: {:?}", inserted_neighbors);
                let gt_vec = ground_truth.iter().map(|item| (item.vertex_id, item.direction_tag)).collect::<Vec<_>>();
                let rs_vec = res.iter().map(|item| (item.vertex_id, item.direction_tag)).collect::<Vec<_>>();
                // let rs_vec_type = res.iter().map(|item| (item.vertex_id, item.direction_tag, item)).collect::<Vec<_>>();
                println!("GT: {:?}", gt_vec);
                println!("RS: {:?}", rs_vec);

                // Display the difference of the ground truth and the result.
                let gt_set: HashSet<_> = gt_vec.clone().into_iter().collect();
                let rs_set: HashSet<_> = rs_vec.clone().into_iter().collect();

                let mut diff: Vec<_> = gt_set.difference(&rs_set).cloned().collect();

                println!("Diff: {:?}", diff);
                lost_edges.append(&mut diff);
                assert_eq!(ground_truth.len(), res.len());
            }
        }
        println!("Loss data: {:?}", lost_edges);
        println!("Rebuild, Find Neighbor test pass!");

        // Perform Insert.
        let duration = start.elapsed();
        println!("Insert {} edges in {:?}.", inserted_edge_count, duration);
    }

    #[test]
    fn test_bfs() {
        let lsm_community =
            Arc::new(LSMStorageState::build_from_graph_file("data/oregon.graph", "oregon_bfs"));
        let attr_storage_instance = AttrStorage::create("oregon_bfs".to_owned(), AttrStorageConfig {
            cache_capacity: 1000 * 1000 * 512,
            flush_frequency: 1000,
        });
        attr_storage_instance.load_attribute_from_file("data/oregon.graph");
        let attr_storage = Arc::new(attr_storage_instance);
        let (flush_notifier, flush_receiver) = crossbeam_channel::unbounded();
        let (dyn_comm_notifier, dyn_comm_receiver) = crossbeam_channel::unbounded();
        let flush_thread_new = lsm_community.spawn_flush_thread(flush_receiver).unwrap();
        let dy_comm_thread_new = lsm_community.spawn_dynamic_community_thread(dyn_comm_receiver).unwrap();
        let mut lsm_storage = LSMStorage {
            dataset_name: "oregon".to_owned(),
            lsm_storage_state: lsm_community,
            attr_storage,
            flush_notifier,
            flush_thread: flush_thread_new,
            dyn_comm_notifier,
            community_thread: dy_comm_thread_new,
            txn_manager: LsmTxnInner::new(0)
        };
        // Start dynamic community detection after building the LSM-Community storage.
        lsm_storage.trigger_dyn_community();

        let g_from_graph = Graph::from_graph_file("data/oregon.graph", true);
        let start = Instant::now();
        let bfs_v_list = lsm_storage.bfs(&0u32);

        let duration = start.elapsed();
        println!("BFS Time: {} ms.", duration.as_millis());
        assert_eq!(bfs_v_list.len(), g_from_graph.v_size as usize);
        println!("BFS test pass!");
    }

    #[test]
    fn test_wcc() {
        let lsm_community =
            Arc::new(LSMStorageState::build_from_graph_file("data/email-enron.graph", "email_wcc"));
        let attr_storage_instance = AttrStorage::create("email_wcc".to_owned(), AttrStorageConfig {
            cache_capacity: 1000 * 1000 * 512,
            flush_frequency: 1000,
        });
        attr_storage_instance.load_attribute_from_file("data/oregon.graph");
        let attr_storage = Arc::new(attr_storage_instance);
        let (flush_notifier, flush_receiver) = crossbeam_channel::unbounded();
        let (dyn_comm_notifier, dyn_comm_receiver) = crossbeam_channel::unbounded();
        let flush_thread_new = lsm_community.spawn_flush_thread(flush_receiver).unwrap();
        let dy_comm_thread_new = lsm_community.spawn_dynamic_community_thread(dyn_comm_receiver).unwrap();
        let mut lsm_storage = LSMStorage {
            dataset_name: "oregon".to_owned(),
            lsm_storage_state: lsm_community,
            attr_storage,
            flush_notifier,
            flush_thread: flush_thread_new,
            dyn_comm_notifier,
            community_thread: dy_comm_thread_new,
            txn_manager: LsmTxnInner::new(0)
        };
        // Start dynamic community detection after building the LSM-Community storage.
        lsm_storage.trigger_dyn_community();

        let start = Instant::now();
        let bfs_v_list = lsm_storage.wcc();

        let duration = start.elapsed();
        println!("WCC Time: {} ms.", duration.as_millis());
        println!("WCC test pass, total {} wcc!", bfs_v_list.len());
    }

    #[test]
    fn test_scc() {
        let lsm_community =
            Arc::new(LSMStorageState::build_from_graph_file("data/example.graph", "example_scc"));
        let attr_storage_instance = AttrStorage::create("example_scc".to_owned(), AttrStorageConfig {
            cache_capacity: 1000 * 1000 * 512,
            flush_frequency: 1000,
        });
        attr_storage_instance.load_attribute_from_file("data/oregon.graph");
        let attr_storage = Arc::new(attr_storage_instance);
        let (flush_notifier, flush_receiver) = crossbeam_channel::unbounded();
        let (dyn_comm_notifier, dyn_comm_receiver) = crossbeam_channel::unbounded();
        let flush_thread_new = lsm_community.spawn_flush_thread(flush_receiver).unwrap();
        let dy_comm_thread_new = lsm_community.spawn_dynamic_community_thread(dyn_comm_receiver).unwrap();
        let mut lsm_storage = LSMStorage {
            dataset_name: "oregon".to_owned(),
            lsm_storage_state: lsm_community,
            attr_storage,
            flush_notifier,
            flush_thread: flush_thread_new,
            dyn_comm_notifier,
            community_thread: dy_comm_thread_new,
            txn_manager: LsmTxnInner::new(0)
        };
        // Start dynamic community detection after building the LSM-Community storage.
        lsm_storage.trigger_dyn_community();

        let start = Instant::now();
        let scc_list = lsm_storage.scc();

        let duration = start.elapsed();
        println!("SCC Time: {} ms.", duration.as_millis());
        println!("SCC test pass, total {} scc!", scc_list.len());
    }

    #[test]
    fn test_sssp() {
        let lsm_community =
            Arc::new(LSMStorageState::build_from_graph_file("data/example.graph", "example_sssp"));
        let attr_storage_instance = AttrStorage::create("example_sssp".to_owned(), AttrStorageConfig {
            cache_capacity: 1000 * 1000 * 512,
            flush_frequency: 1000,
        });
        attr_storage_instance.load_attribute_from_file("data/oregon.graph");
        let attr_storage = Arc::new(attr_storage_instance);
        let (flush_notifier, flush_receiver) = crossbeam_channel::unbounded();
        let (dyn_comm_notifier, dyn_comm_receiver) = crossbeam_channel::unbounded();
        let flush_thread_new = lsm_community.spawn_flush_thread(flush_receiver).unwrap();
        let dy_comm_thread_new = lsm_community.spawn_dynamic_community_thread(dyn_comm_receiver).unwrap();
        let mut lsm_storage = LSMStorage {
            dataset_name: "oregon".to_owned(),
            lsm_storage_state: lsm_community,
            attr_storage,
            flush_notifier,
            flush_thread: flush_thread_new,
            dyn_comm_notifier,
            community_thread: dy_comm_thread_new,
            txn_manager: LsmTxnInner::new(0)
        };
        // Start dynamic community detection after building the LSM-Community storage.
        lsm_storage.trigger_dyn_community();

        let start = Instant::now();
        let distance = lsm_storage.sssp(&0);

        let duration = start.elapsed();
        println!("SCC Time: {} ms.", duration.as_millis());
        // Print the distance.
        for (vertex_id, dist) in distance {
            println!("V0 -> V{}, Distance: {}", vertex_id, dist);
        }
    }

    #[test]
    fn test_cd() {
        let lsm_community =
            Arc::new(LSMStorageState::build_from_graph_file("data/oregon.graph", "oregon_cd"));
        let attr_storage_instance = AttrStorage::create("oregon_cd".to_owned(), AttrStorageConfig {
            cache_capacity: 1000 * 1000 * 512,
            flush_frequency: 1000,
        });
        attr_storage_instance.load_attribute_from_file("data/oregon.graph");
        let attr_storage = Arc::new(attr_storage_instance);
        let (flush_notifier, flush_receiver) = crossbeam_channel::unbounded();
        let (dyn_comm_notifier, dyn_comm_receiver) = crossbeam_channel::unbounded();
        let flush_thread_new = lsm_community.spawn_flush_thread(flush_receiver).unwrap();
        let dy_comm_thread_new = lsm_community.spawn_dynamic_community_thread(dyn_comm_receiver).unwrap();
        let lsm_storage = LSMStorage {
            dataset_name: "oregon".to_owned(),
            lsm_storage_state: lsm_community,
            attr_storage,
            flush_notifier,
            flush_thread: flush_thread_new,
            dyn_comm_notifier,
            community_thread: dy_comm_thread_new,
            txn_manager: LsmTxnInner::new(0)
        };
        // Start dynamic community detection after building the LSM-Community storage.
        lsm_storage.trigger_dyn_community();

        let start = Instant::now();
        let comm_split = lsm_storage.get_community_split();

        let duration = start.elapsed();
        println!("CD Time: {} ms.", duration.as_millis());
        println!("Community Count: {}", comm_split.len());
    }
}