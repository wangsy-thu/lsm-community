use std::borrow::Borrow;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use bytes::Bytes;

use crate::graph::VInt;
use crate::types::{Encode, V32, V32_SIZE, Vertex};
use crate::wal::Wal;

/// Give the edge direction another name.
type VDir = u8;

/// The reason for selection this sort of data structure for the basic of the memory graph.
type AdjMap = BTreeMap<VInt, (Vertex<u32>, BTreeMap<(VInt, VDir), Vertex<u32>>)>;

// MemGraph to cache the recent
// ly updated graph data, including some entries
// organized by their keys.
#[allow(dead_code)]
#[derive(Clone)]
pub struct MemGraph {
    pub(crate) graph: AdjMap,
    pub(crate) wal: Option<Wal>, // The Wal for crash recovery.
    id: usize, // The id of the MemGraph.
    approximate_size: Arc<AtomicUsize>, // The number of bytes in this MemGraph.
}

/// Check whether the directed edge in the graph.
pub fn has_edge(adj_map: &AdjMap, u: &V32, v: &V32) -> bool {
    // Check whether this graph has directed edge (u -> v).
    if adj_map.contains_key(&u.vertex_id) {
        // Contains u for sure, use unwrap.
        if adj_map.get(&u.vertex_id).unwrap().1.contains_key(&(v.vertex_id, v.direction_tag)) {
            adj_map.get(&u.vertex_id).unwrap().1.get(&(v.vertex_id, v.direction_tag)).unwrap().tomb == 0
        } else {
            false
        }
    } else {
        false
    }
}

/// Remove a directed edge from the graph.
pub fn remove_edge(adj_map: &mut AdjMap, u: &V32, v: &V32) {
    // Remove the wanted edge from the memory graph.
    if has_edge(adj_map, u, v) {
        adj_map.get_mut(&u.vertex_id).unwrap().1.remove(&(v.vertex_id, v.direction_tag));
    }
}

#[allow(dead_code)]
impl MemGraph {
    pub fn create(id: usize) -> Self {
        // Function to create a MemGraph.
        return MemGraph {
            graph: AdjMap::new(),
            wal: None,
            id,
            approximate_size: Arc::new(Default::default()),
        }
    }

    pub fn create_with_wal(id: usize, _: impl AsRef<Path>) -> Result<Self> {
        // Create a new MemGraph from the wal.
        Ok(Self {
            graph: AdjMap::new(),
            wal: None,
            id,
            approximate_size: Arc::new(Default::default()),
        })
    }

    pub fn get_mem_neighbor(&self, u: &VInt) -> Vec<Vertex<u32>> {
        // Get neighbor of u in mem_graph.
        let mut neighbors = Vec::new();
        if self.graph.contains_key(u) {
            for (_, neighbor) in &self.graph.get(u).unwrap().1 {
                // println!("A candidate neighbor: {}", neighbor);
                if neighbor.tomb == 0 {
                    neighbors.push(neighbor.clone());
                }
            }
        }
        neighbors
    }

    /// Acquire all the neighbors, no matter it is a tomb.
    pub fn get_all_mem_neighbor(&self, u: &VInt) -> Vec<Vertex<u32>> {
        // Get neighbor of u in mem_graph.
        let mut neighbors = Vec::new();
        if self.graph.contains_key(u) {
            for (_, neighbor) in &self.graph.get(u).unwrap().1 {
                // println!("A candidate neighbor: {}", neighbor);
                neighbors.push(neighbor.clone());
            }
        }
        neighbors
    }

    pub fn put_vertex(&mut self, u: Vertex<u32>) {
        // Insert a vertex to mem_graph.
        // Case 1: del(u), check whether key u exists, if not, perform put,
        // elif single, perform remove, else, do nothing.
        if self.graph.contains_key(&u.vertex_id) {
            if u.tomb == 1 {
                if self.graph.get(&u.vertex_id).unwrap().1.len() == 0 {
                    self.graph.remove(&u.vertex_id);
                    self.graph.insert(u.vertex_id, (u, BTreeMap::default()));
                } else if self.graph.get(&u.vertex_id).unwrap().1.iter().filter(|(_, neighbor)| {
                    neighbor.tomb == 0
                }).count() == 0 {
                    self.graph.get_mut(&u.vertex_id).unwrap().0.tomb = 1;
                }
            }
            // Size not change.
        } else {
            self.graph.insert(u.vertex_id, (u, BTreeMap::default()));
            // Increase the size.
            self.approximate_size.fetch_add(V32_SIZE, Ordering::Relaxed);
        }
    }

    pub fn put_edge(&mut self, u: Vertex<u32>, v: Vertex<u32>) {
        // Put an edge in this mem_graph in paradigm of C-RMW(Check-Read-Modify-Write).
        // Firstly push the edge into the map.
        // Case 1: (u, del(v)), check whether key u exists, if not, perform put, else, perform remove.
        if v.tomb == 1 {
            // This neighbor should be removed.
            if self.graph.contains_key(&u.vertex_id) {
                if has_edge(&self.graph, &u, &v) {
                    self.approximate_size.fetch_sub(V32_SIZE, Ordering::Relaxed);
                    remove_edge(&mut self.graph, &u, &v);
                }
                self.graph.get_mut(&u.vertex_id).unwrap().1.insert((v.vertex_id, v.direction_tag), v);
                self.approximate_size.fetch_add(V32_SIZE, Ordering::Relaxed);
            } else {
                let mut new_b_tree = BTreeMap::<(VInt, VDir), Vertex<u32>>::new();
                new_b_tree.insert((v.vertex_id, v.direction_tag), v);
                self.graph.insert(u.vertex_id, (u, new_b_tree));
                self.approximate_size.fetch_add(2 * V32_SIZE, Ordering::Relaxed);
            }
        } else {
            // Case 2: (u, v), check whether key u exists, if not, perform put, else, perform modify.
            if self.graph.contains_key(&u.vertex_id) {
                if !has_edge(&self.graph, &u, &v) {
                    self.graph.get_mut(&u.vertex_id).unwrap().1.insert((v.vertex_id, v.direction_tag), v);
                    self.approximate_size.fetch_add(V32_SIZE, Ordering::Relaxed);
                }
            } else {
                let mut new_b_tree = BTreeMap::<(VInt, VDir), Vertex<u32>>::new();
                new_b_tree.insert((v.vertex_id, v.direction_tag), v);
                self.graph.insert(u.vertex_id, (u, new_b_tree));
                self.approximate_size.fetch_add(2 * V32_SIZE, Ordering::Relaxed);
            }
        }

        // Next put it into the wal if this put is valid.
        if let Some(ref wal) = self.wal {
            wal.put(&Bytes::from(u.encode().to_vec()), &Bytes::from(v.encode().to_vec()))
                .expect("Push to wal failed.");
        }
    }

    pub(crate) fn clear(&mut self) {
        // Flush this mem_graph into disk.
        self.graph.clear();
        self.approximate_size.store(0, Ordering::SeqCst);
    }

    pub(crate) fn print_graph(&self) {
        println!("MemGraph");
        for (u, neighbors) in &self.graph {
            print!("{}->", u);
            for v in &neighbors.1 {
                print!("{}->", v.0.0)
            }
            println!("END");
        }
    }

    /// Return the immutable reference of the adj map.
    pub(crate) fn adj_ref(&self) -> &BTreeMap<VInt, (Vertex<u32>, BTreeMap<(VInt, VDir), Vertex<u32>>)> {
        self.graph.borrow()
    }

    /// Return the used size of this memory graph.
    pub(crate) fn get_used_size(&self) -> usize {
        self.approximate_size.load(Ordering::Relaxed)
    }
}


#[cfg(test)]
mod test_mem_graph {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    use bytes::Bytes;
    use crossbeam_skiplist::SkipMap;
    use rand::prelude::SliceRandom;
    use rand::Rng;

    use crate::graph::{Graph, VInt};
    use crate::mem_graph::MemGraph;
    use crate::types::{V32_SIZE, Vertex};

    fn generate_example() -> MemGraph {
        let mut mem_g = MemGraph::create(0);
        mem_g.put_vertex(Vertex::new_vertex(1u32));
        mem_g.put_vertex(Vertex::new_vertex(2u32));
        mem_g.put_vertex(Vertex::new_vertex(3u32));
        mem_g.put_edge(Vertex::new_vertex(1u32), Vertex::new_successor(2u32));
        mem_g.put_edge(Vertex::new_vertex(1u32), Vertex::new_successor(3u32));
        mem_g.put_edge(Vertex::new_vertex(2u32), Vertex::new_predecessor(3u32));
        mem_g
    }

    #[test]
    fn test_skip_list() {
        let map = SkipMap::<Bytes, Bytes>::new();
        map.insert(Bytes::from(1u32.to_le_bytes().to_vec()), Bytes::from(2u32.to_le_bytes().to_vec()));
        map.insert(Bytes::from(1u32.to_le_bytes().to_vec()), Bytes::from(3u32.to_le_bytes().to_vec()));
        map.insert(Bytes::from(2u32.to_le_bytes().to_vec()), Bytes::from(3u32.to_le_bytes().to_vec()));

        let value = map.get(&Bytes::from(1u32.to_le_bytes().to_vec())).map(
            |e| e.value().clone());
        let mut byte_array = [0u8; 4];
        byte_array.copy_from_slice(&value.unwrap()[0..4]);
        assert_eq!(3u32, u32::from_le_bytes(byte_array));
        println!("3 = {}", u32::from_le_bytes(byte_array));
        // From this test, we can see that, in this kind of SkipMap, multiple of the same key is not allowed.
        // The solution is, when put a new entry, perform RMW (read-modify-write).
        // Choice 1: Avoid using skip list
        // Choice 2: Use skip list + Vec
    }

    #[test]
    fn test_put_edge() {
        let mem_g = generate_example();
        assert_eq!(mem_g.graph.iter().len(), 3);
        let mut edge_count = 0u32;
        for (_, neighbors) in mem_g.graph {
            edge_count += neighbors.1.iter().count() as u32;
        }
        assert_eq!(edge_count, 3);
    }

    #[test]
    fn test_put_edge_tomb() {
        let mut mem_g = generate_example();

        mem_g.put_edge(Vertex::new_vertex(1u32), Vertex::new_tomb(3u32, 1u8));
        assert_eq!(mem_g.graph.iter().len(), 3);
        let mut edge_count = 0u32;
        for (_, neighbors) in &mem_g.graph {
            edge_count += neighbors.1.iter().count() as u32;
        }
        assert_eq!(edge_count, 3);
        let key = 1u32;
        for (_, v) in &mem_g.graph.get(&key).unwrap().1 {
            if v.vertex_id == 3u32 {
                assert_eq!(v.tomb, 1);
                break;
            }
        }
        mem_g.print_graph();
    }

    #[test]
    fn test_put_vertex() {
        let mut mem_g = generate_example();
        mem_g.put_vertex(Vertex::new_vertex(4u32));
        mem_g.put_edge(Vertex::new_vertex(3u32), Vertex::new_successor(4u32));
        assert_eq!(mem_g.graph.len(), 4);
        let mut edge_count = 0u32;
        for (_, neighbors) in &mem_g.graph {
            edge_count += neighbors.1.iter().count() as u32;
        }
        assert_eq!(edge_count, 4);
    }

    #[test]
    fn test_put_vertex_tomb() {
        let mut mem_g = generate_example();
        mem_g.put_vertex(Vertex::new_vertex(5u32));
        mem_g.put_vertex(Vertex::new_tomb(1u32, 0));
        mem_g.put_vertex(Vertex::new_tomb(5u32, 0));
        assert_eq!(mem_g.graph.len(), 4);
        let mut edge_count = 0u32;
        for (_, neighbors) in &mem_g.graph {
            edge_count += neighbors.1.iter().count() as u32;
        }
        assert_eq!(edge_count, 3);
        for (u_id, u) in mem_g.graph {
            if u_id == 5u32 {
                assert_eq!(u.0.tomb, 1);
            }
            if u_id == 1u32 {
                assert_eq!(u.0.tomb, 0);
            }
        }
    }

    #[test]
    fn test_get_mem_neighbor() {
        let mut mem_g = generate_example();
        println!("Memory Graph before put Tomb:");
        mem_g.print_graph();
        mem_g.put_edge(Vertex::new_vertex(1u32), Vertex::new_tomb(3u32, 1));
        let res = mem_g.get_mem_neighbor(&1u32);
        mem_g.print_graph();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].vertex_id, 2u32);
        mem_g.print_graph();
    }

    #[test]
    fn test_spawn() {
        // Test the multi-thread ops.
        let counter = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];
        for _ in 0..10 {
            let counter = Arc::clone(&counter);
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    counter.fetch_add(1, Ordering::SeqCst);
                }
            });
            handles.push(handle);
        }
        for handle in handles {
            handle.join().unwrap();
        }
        println!("Final value: {}", counter.load(Ordering::SeqCst));
    }

    #[test]
    fn test_approximate_size() {
        let mem_g = generate_example();
        let mut ground_truth = 0usize;
        for (_, neighbors) in &mem_g.graph {
            ground_truth = ground_truth + V32_SIZE;
            for _ in &neighbors.1 {
                ground_truth += V32_SIZE;
            }
        }
        assert_eq!(mem_g.approximate_size.load(Ordering::Relaxed), ground_truth);
        println!("Graph Size: {}, Ground Truth: {}", mem_g.approximate_size.load(Ordering::Relaxed), ground_truth)
    }

    /// Here, I need to test the edge insertion of this memory graph.
    #[test]
    fn test_insert_edge() {
        // Test the memory graph through loading the whole graph to memory.
        println!("Start to Insert Edges in LSM-Community");
        let mut mem_graph = MemGraph::create(0);
        // Prepare the random number generator.
        let mut rng = rand::thread_rng();
        // Load the graph.
        let mut graph_example = Graph::from_graph_file("data/oregon.graph", true);
        // Firstly, build the memory graph from the graph.
        for (_, (vertex, neighbors)) in &graph_example.adj_map {
            mem_graph.put_vertex(vertex.clone());
            for neighbor in neighbors {
                mem_graph.put_edge(vertex.clone(), neighbor.clone());
            }
        }

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
                    mem_graph.put_edge(Vertex::new_vertex(vertex_id), Vertex::new_successor(added_neighbor));
                    mem_graph.put_edge(Vertex::new_vertex(added_neighbor), Vertex::new_predecessor(vertex_id));
                    added_count += 1;
                }
                if added_count >= 10 {
                    break;
                }
            }
        }

        // Validate the correctness of the edge insertion.
        for tested_vid in 0..10000 {
            // Performing the test.
            println!("Testing Vertex: {}", tested_vid);
            let res = mem_graph.get_mem_neighbor(&tested_vid);

            // Check the neighbors.
            let ground_truth = graph_example.get_neighbor(&tested_vid);

            // V \in U and |V| = |U| => V = U (In Algebra.)
            for v in &res {
                assert!(ground_truth.contains(v));
            }
            if ground_truth.len() != res.len() {
                println!("GT: {:?}", ground_truth.iter().map(|item| (item.vertex_id, item.direction_tag)).collect::<Vec<_>>());
                println!("RS: {:?}", res.iter().map(|item| item.vertex_id).collect::<Vec<_>>());
                assert_eq!(ground_truth.len(), res.len());
            }
        }
        println!("Rebuild, Find Neighbor test pass!");
    }

    /// Test the edge insertion and deletion of the memory graph.
    #[test]
    fn test_insert_delete_edge () {
        let mut mem_graph = MemGraph::create(0);
        println!("Start to Insert Edges in LSM-Community");
        // Prepare the random number generator.
        let mut rng = rand::thread_rng();
        // Load the graph.
        let mut graph_example = Graph::from_graph_file("data/oregon.graph", true);
        // Firstly, build the memory graph from the graph.
        for (_, (vertex, neighbors)) in &graph_example.adj_map {
            mem_graph.put_vertex(vertex.clone());
            for neighbor in neighbors {
                mem_graph.put_edge(vertex.clone(), neighbor.clone());
            }
        }

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
                    mem_graph.put_edge(Vertex::new_vertex(vertex_id), Vertex::new_successor(added_neighbor));
                    mem_graph.put_edge(Vertex::new_vertex(added_neighbor), Vertex::new_predecessor(vertex_id));
                    added_count += 1;
                }
                if added_count >= 10 {
                    break;
                }
            }
        }

        let mut deleted_graph = HashMap::<VInt, Vec<(VInt, u8)>>::new();
        let remove_count = 1;
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
                        mem_graph.put_edge(Vertex::new_vertex(vertex_id), Vertex::new_tomb(removed_neighbor.vertex_id, 1));
                        mem_graph.put_edge(Vertex::new_vertex(removed_neighbor.vertex_id), Vertex::new_tomb(vertex_id, 2));
                    }
                }
            }
        }
        // Validate the correctness of the edge insertion.
        for tested_vid in 0..vertex_count {
            println!("Testing V{}", tested_vid);
            // Performing the test.
            let res = mem_graph.get_mem_neighbor(&tested_vid);

            // Check the neighbors.
            let ground_truth = graph_example.get_neighbor(&tested_vid);

            // V \in U and |V| = |U| => V = U (In Algebra.)
            // assert_eq!(ground_truth.len(), res.len());
            if ground_truth.len() != res.len() {
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
                assert_eq!(ground_truth.len(), res.len());
            }
            // Display the results.
            for v in &res {
                assert!(ground_truth.contains(v));
            }
        }
        println!("Rebuild, Find Neighbor test pass!");
    }
}