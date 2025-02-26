use std::{fs::File, io::{BufRead, BufReader}};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use rand::prelude::SliceRandom;

use crate::config::READ_BUFFER_SIZE;
use crate::louvain_graph::LouvainGraph;
use crate::types::{GraphEntry, V32, Vertex};
use crate::util::get_current_timestamp;

#[allow(dead_code)]
pub type VInt = u32;


#[allow(dead_code)]
#[derive(Default)]
pub struct Graph {
    pub adj_map: BTreeMap<VInt, (Vertex<u32>, Vec<Vertex<u32>>)>,
    pub v_size: u32,
    pub e_size: u32,
}

/// The subgraph we want to query.
#[allow(dead_code)]
pub(crate) struct QueryGraph {
    pub(crate) adj_map: BTreeMap<VInt, Vec<VInt>>,
}

#[allow(dead_code)]
impl QueryGraph {
    pub fn from_edge_list(edge_list: &Vec<(VInt, VInt)>) -> Self {
        let mut adj_map = BTreeMap::<VInt, Vec<VInt>>::new();
        for (src, dst) in edge_list {
            adj_map.entry(*src).or_insert(vec![*dst]).push(*dst);
        }
        Self {
            adj_map
        }
    }
}

// Graph Snapshot without timestamp, mainly used in community detection.
#[allow(dead_code)]
pub struct GraphSnapshot {
    pub(crate) adj_map: BTreeMap<VInt, Vec<VInt>>,
    pub(crate) v_size: u32,
    pub(crate) e_size: u32
}

#[allow(dead_code)]
impl GraphSnapshot {
    pub fn new() -> GraphSnapshot {
        // Create a new empty Graph Snapshot.
        GraphSnapshot {
            adj_map: BTreeMap::new(),
            v_size: 0u32,
            e_size: 0u32
        }
    }

    pub fn get_neighbor(&self, vertex_id: &VInt) -> Vec<VInt> {
        if self.adj_map.contains_key(vertex_id) {
            self.adj_map.get(vertex_id).unwrap().clone()
        } else {
            vec![]
        }
    }

    /// Load a graph from file, with community and optional labels.
    pub fn from_graph_file_community (
        file_path: &str,
        is_directed: bool
    ) -> (Self, Vec<Vec<u32>>) {
        // Load a Graph from a .graph file, like Lou's experiments.
        let graph_file = File::open(file_path).unwrap();
        let graph_reader = BufReader::with_capacity(READ_BUFFER_SIZE, graph_file);
        let mut adj_map =
            BTreeMap::<VInt, Vec<VInt>>::new();
        let mut community_info = Vec::<u32>::new();
        let mut line_count = 0u32;
        for line in graph_reader.lines() {
            line_count += 1;
            if line_count == 1 {
                // The first line, just skip it.
                continue;
            }

            if let Ok(line) = line {
                let tokens :Vec<&str> = line.split_whitespace().collect();
                if tokens[0] == "v" {
                    let parsed_vid = tokens[1].parse().ok().expect("File format error.");
                    let parsed_comm = tokens[3].parse().ok().expect("File format error.");
                    // Process Vertices.
                    adj_map.insert(parsed_vid, vec![]);
                    community_info.push(parsed_comm);
                }
                if tokens[0] == "e" && tokens.len() == 3 {
                    // Process Edges.
                    let mut edge_vec = Vec::new();
                    let parsed_src_vid = tokens[1].parse().ok().expect("File format error.");
                    let parsed_dst_vid = tokens[2].parse().ok().expect("File format error.");
                    edge_vec.push(parsed_src_vid);
                    edge_vec.push(parsed_dst_vid);

                    adj_map.get_mut(&parsed_src_vid).unwrap().push(parsed_dst_vid);

                    if !is_directed {
                        adj_map.get_mut(&parsed_dst_vid).unwrap().push(parsed_src_vid);
                    }
                }
            }
        }

        let v_size = adj_map.iter().count() as u32;
        let mut e_size = 0u32;
        for vertex_entry in adj_map.iter() {
            e_size += vertex_entry.1.len() as u32;
        }

        // Process the community info.
        let community_struct = community_info.into_iter().enumerate().fold(
            BTreeMap::<VInt, Vec<u32>>::new(),
            |mut acc, (vertex_id, comm_id)| {
                if acc.contains_key(&comm_id) {
                    acc.get_mut(&comm_id).unwrap().push(vertex_id as u32);
                } else {
                    acc.insert(comm_id, vec![vertex_id as u32]);
                }
                acc
            }
        ).values().cloned().collect::<Vec<_>>();

        (GraphSnapshot {
            adj_map,
            v_size,
            e_size,
        }, community_struct)
    }


    /// Load a graph from file.
    pub fn from_graph_file (
        file_path: &str,
        is_directed: bool
    ) -> Self {
        // Load a Graph from a .graph file, like Lou's experiments.
        let graph_file = File::open(file_path).unwrap();
        let graph_reader = BufReader::with_capacity(READ_BUFFER_SIZE, graph_file);
        let mut adj_map =
            BTreeMap::<VInt, Vec<VInt>>::new();
        let mut line_count = 0u32;
        for line in graph_reader.lines() {
            line_count += 1;
            if line_count == 1 {
                // The first line, just skip it.
                continue;
            }

            if let Ok(line) = line {
                let tokens :Vec<&str> = line.split_whitespace().collect();
                if tokens[0] == "v" {
                    let parsed_vid = tokens[1].parse().ok().expect("File format error.");
                    // Process Vertices.
                    adj_map.insert(parsed_vid, vec![]);
                }
                if tokens[0] == "e" && tokens.len() == 3 {
                    // Process Edges.
                    let mut edge_vec = Vec::new();
                    let parsed_src_vid = tokens[1].parse().ok().expect("File format error.");
                    let parsed_dst_vid = tokens[2].parse().ok().expect("File format error.");
                    edge_vec.push(parsed_src_vid);
                    edge_vec.push(parsed_dst_vid);

                    adj_map.get_mut(&parsed_src_vid).unwrap().push(parsed_dst_vid);

                    if !is_directed {
                        adj_map.get_mut(&parsed_dst_vid).unwrap().push(parsed_src_vid);
                    }
                }
            }
        }

        let v_size = adj_map.iter().count() as u32;
        let mut e_size = 0u32;
        for vertex_entry in adj_map.iter() {
            e_size += vertex_entry.1.len() as u32;
        }


        GraphSnapshot {
            adj_map,
            v_size,
            e_size,
        }
    }

    /// Perform sssp algorithm for Grid Storage.
    pub fn sssp(&self, vertex_id: &VInt) -> HashMap<VInt, i32> {
        let mut distances: HashMap<u32, i32> = HashMap::new();
        let mut queue: VecDeque<u32> = VecDeque::new();

        distances.insert(*vertex_id, 0);
        queue.push_back(*vertex_id);

        while let Some(current_node) = queue.pop_front() {
            let current_distance = *distances.get(&current_node).unwrap();

            for neighbor in self.get_neighbor(&current_node) {
                if !distances.contains_key(&neighbor) {
                    distances.insert(neighbor, current_distance + 1);
                    queue.push_back(neighbor);
                }
            }
        }
        distances
    }

    /// Perform LPA community detection.
    pub fn get_community_split(&self, max_iterations: usize) -> Vec<Vec<VInt>> {
        let mut rng = rand::thread_rng();

        // Generate the vertex ID list.
        let all_vertices: Vec<VInt> = (0..self.v_size).collect();

        // Initialize the labels.
        let mut labels: HashMap<VInt, VInt> = all_vertices
            .iter()
            .map(|&v| (v, v))
            .collect();

        // Update the labels.
        for _ in 0..max_iterations {
            let mut changes = false;

            // Randomize the order of vertex accessing.
            let mut vertices = all_vertices.clone();
            vertices.shuffle(&mut rng);

            // Update the label.
            for &vertex in &vertices {
                let neighbors = self.get_neighbor(&vertex);
                if neighbors.is_empty() {
                    continue;
                }

                // Calculate the neighbors.
                let mut label_counts: HashMap<VInt, usize> = HashMap::new();
                for &neighbor in &neighbors {
                    if let Some(&label) = labels.get(&neighbor) {
                        *label_counts.entry(label).or_insert(0) += 1;
                    }
                }

                // Find the common labels.
                if let Some((&new_label, _)) = label_counts
                    .iter()
                    .max_by_key(|&(_, count)| count)
                    .or_else(|| Some((&vertex, &1)))
                {
                    // Record the change.
                    if labels.get(&vertex) != Some(&new_label) {
                        labels.insert(vertex, new_label);
                        changes = true;
                    }
                }
            }

            // Stop it.
            if !changes {
                break;
            }
        }

        // Generate the result.
        let mut communities: HashMap<VInt, Vec<VInt>> = HashMap::new();
        for (&vertex, &label) in &labels {
            communities.entry(label)
                .or_insert_with(Vec::new)
                .push(vertex);
        }

        communities.into_values().collect()
    }

    pub fn get_community_with(&self, start_vertex: &VInt) -> Option<Vec<VInt>> {
        let mut community = HashSet::new();
        let mut vertex_degree = HashMap::new();

        let mut vertex_scores = HashMap::new();
        vertex_scores.insert(*start_vertex, 1.0f64);

        let mut local_vertices = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(*start_vertex);
        local_vertices.insert(*start_vertex);

        while let Some(vertex_id) = queue.pop_front() {
            let neighbors = self.get_neighbor(&vertex_id);
            let vertex_score = *vertex_scores.get(&vertex_id).unwrap_or(&0.0);

            let mut layer_edges = 0;
            let mut layer_vertices = HashSet::new();

            for neighbor in &neighbors {
                layer_vertices.insert(neighbor);
                if local_vertices.contains(&neighbor) {
                    layer_edges += 1;
                }
            }

            let layer_density = if !layer_vertices.is_empty() {
                layer_edges as f64 / layer_vertices.len() as f64
            } else {
                0.0
            };

            if layer_density >= 0.05 && vertex_score >= 0.1 {  // 原来是 0.1 和 0.2
                for neighbor in neighbors {
                    if !local_vertices.contains(&neighbor) {
                        local_vertices.insert(neighbor);
                        let neighbor_score = vertex_score * (layer_density + 0.1);  // 加上一个基础值，减缓衰减
                        vertex_scores.insert(neighbor, neighbor_score);
                        queue.push_back(neighbor);
                    }

                    *vertex_degree.entry(vertex_id).or_insert(0) += 1;
                    *vertex_degree.entry(neighbor).or_insert(0) += 1;
                }
            }
        }

        community.insert(*start_vertex);
        let mut candidates: Vec<_> = self.get_neighbor(start_vertex)
            .into_iter()
            .filter(|v| local_vertices.contains(&v))
            .collect();
        let mut visited = HashSet::new();
        visited.insert(*start_vertex);

        while !candidates.is_empty() {
            let vertex = candidates.remove(0);
            if visited.contains(&vertex) {
                continue;
            }
            visited.insert(vertex);

            let local_neighbors: HashSet<_> = self.get_neighbor(&vertex)
                .into_iter()
                .map(|v| v)
                .filter(|vid| local_vertices.contains(vid))
                .collect();

            let internal_edges = local_neighbors.intersection(&community).count();
            let vertex_deg = *vertex_degree.get(&vertex).unwrap_or(&0);

            let local_total_deg: i32 = vertex_degree.iter()
                .filter(|(v, _)| local_vertices.contains(v))
                .map(|(_, &deg)| deg)
                .sum();

            let community_deg: i32 = community.iter()
                .map(|v| vertex_degree.get(v).unwrap_or(&0))
                .sum();

            let vertex_score = vertex_scores.get(&vertex).unwrap_or(&0.0);
            let modularity_gain = ((internal_edges as f64) -
                ((vertex_deg as f64 * community_deg as f64) / (local_total_deg as f64))) *
                vertex_score;

            if modularity_gain > 0.001 {  // 原来是 0.01
                community.insert(vertex);

                for neighbor in self.get_neighbor(&vertex) {
                    if !visited.contains(&neighbor) &&
                        local_vertices.contains(&neighbor) &&
                        vertex_scores.get(&neighbor).unwrap_or(&0.0) > &0.05 {
                        candidates.push(neighbor);
                    }
                }
            }
        }

        Some(community.into_iter().collect())
    }

    pub fn get_community_with_k_core(
        &self,
        query_vertex: &VInt,
        min_degree: usize,
        max_size: Option<usize>
    ) -> Option<Vec<VInt>> {
        let mut community: HashSet<VInt> = HashSet::new();
        let mut queue: VecDeque<VInt> = VecDeque::new();

        // Acquire the neighbors.
        let query_neighbors = self.get_neighbor(query_vertex);

        if query_neighbors.len() <= min_degree {

            community.insert(*query_vertex);
            community.extend(query_neighbors.iter().cloned());

            for &neighbor in &query_neighbors {
                let neighbor_neighbors = self.get_neighbor(&neighbor);
                for &nn in &neighbor_neighbors {
                    if community.len() >= max_size.unwrap_or(100) {
                        break;
                    }
                    community.insert(nn);
                }
            }
        } else {
            queue.push_back(*query_vertex);
            community.insert(*query_vertex);

            while let Some(current) = queue.pop_front() {
                let neighbors = self.get_neighbor(&current);

                if let Some(size) = max_size {
                    if community.len() >= size {
                        break;
                    }
                }

                for &neighbor in &neighbors {
                    if !community.contains(&neighbor) {
                        let neighbor_neighbors = self.get_neighbor(&neighbor);
                        let connections = neighbor_neighbors
                            .iter()
                            .filter(|&&v| community.contains(&v))
                            .count();

                        if connections >= min_degree.min(neighbor_neighbors.len()) {
                            community.insert(neighbor);
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }

        Some(community.into_iter().collect())
    }

    /// Find scc in the given graph stored in Grid format.
    pub fn scc(&self) -> Vec<Vec<VInt>> {
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

        // `scc_mapping` stores the SCC index for each vertex.
        let mut scc_mapping = HashMap::<VInt, usize>::new();

        // For all vertices, perform tarjan to avoid wcc.
        for v in 0..self.v_size {
            if !visited.contains(&v) {
                self.tarjan_dfs(
                    &v, &mut level_index, &mut visited, &mut stack,
                    &mut is_stacked, &mut dfn, &mut low, &mut result_scc_list
                )
            }
        }

        // Now build the SCC mapping from vertex to SCC index.
        for (i, scc) in result_scc_list.iter().enumerate() {
            for vertex in scc {
                scc_mapping.insert(*vertex, i);
            }
        }
        result_scc_list
    }

    // Perform Tarjan for a grid-storage engine.
    fn tarjan_dfs(
        &self,
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
        for neighbor in self.get_neighbor(start_vertex) {
            // If it is not visited, perform dfs.
            if !is_visited.contains(&neighbor) {
                self.tarjan_dfs(
                    &neighbor, current_level, is_visited,
                    stack, is_stacked, dfn, low, result_scc_list
                );
                if low.get(start_vertex).unwrap() > low.get(&neighbor).unwrap() {
                    low.insert(
                        *start_vertex,
                        *low.get(&neighbor).unwrap()
                    );
                }
            } else if is_stacked.contains(&neighbor) &&
                low.get(start_vertex).unwrap() > low.get(&neighbor).unwrap() {
                low.insert(
                    *start_vertex,
                    *low.get(&neighbor).unwrap()
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

    // Using bfs to walk through a wcc.
    fn bfs_component(
        &self,
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
            let neighbors = self.get_neighbor(&v);
            for neighbor in neighbors {
                if !visited_vertex_list.contains(&neighbor) {
                    queue.push_back(neighbor);
                    visited_vertex_list.insert(neighbor);
                }
            }
        }
    }

    // Perform bfs for psw storage engine.
    pub fn bfs(&self, start_vertex: &VInt) -> Vec<VInt> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        self.bfs_component(start_vertex, &mut visited, &mut result);

        // Check success.
        for v_index in 0..self.v_size {
            if !visited.contains(&v_index) {
                self.bfs_component(&v_index, &mut visited, &mut result);
            }
        }

        result
    }

    /// Compute the wcc of the stored graph.
    pub fn wcc(&self) -> Vec<Vec<VInt>> {
        let mut visited = HashSet::new();
        let mut wcc = vec![];

        // Check success.
        for v_index in 0..self.v_size {
            if !visited.contains(&v_index) {
                let mut result = Vec::new();
                self.bfs_component(&v_index, &mut visited, &mut result);
                wcc.push(result);
            }
        }
        wcc
    }

    pub fn from_entry_list(entries_iter: impl Iterator<Item = GraphEntry<V32>>) -> GraphSnapshot {
        // Load a graph from a graph entry list.
        let mut adj_map = BTreeMap::<VInt, Vec<VInt>>::new();
        for entry in entries_iter {
            if adj_map.contains_key(&entry.key.vertex_id) {
                for neighbor in entry.neighbors {
                    adj_map.get_mut(&entry.key.vertex_id).unwrap().push(neighbor.vertex_id);
                }
            } else {
                let neighbors = entry.neighbors.iter().filter_map(
                    |item| {
                        if item.tomb == 0 && item.direction_tag == 1 {
                            Some(item.vertex_id)
                        } else {
                            None
                        }
                    }
                ).collect();
                adj_map.insert(entry.key.vertex_id, neighbors);
            }
        }
        let v_size = adj_map.len() as u32;
        let e_size = adj_map.iter().fold(
            0u32,
            |mut count, entry| {
                count += entry.1.len() as u32;
                count
            }
        );
        GraphSnapshot {
            adj_map,
            v_size,
            e_size
        }
    }

    pub(crate) fn print_graph(&self) {
        // Print the graph snapshot.
        println!("Graph Snapshot:");
        for vertex_entry in &self.adj_map {
            print!("{}->", vertex_entry.0);
            for v in vertex_entry.1 {
                print!("{}->", v)
            }
            println!("END");
        }
    }
}
#[allow(dead_code)]
impl Graph {
    pub fn new() -> Graph {
        Graph {
            adj_map: BTreeMap::new(),
            v_size: 0,
            e_size: 0,
        }
    }

    /// Get the target vertex.
    pub fn get_vertex(&self, vertex_id: &VInt) -> Option<V32> {
        match self.adj_map.get(vertex_id) {
            None => {None}
            Some((vertex, _)) => {Some(vertex.clone())}
        }
    }

    /// Generate graph entry according to vertex id.
    pub fn generate_entry(&self, vertex_id: VInt) -> Option<GraphEntry<V32>> {
        // Find the neighbors of the given vertex id and return it.
        match self.adj_map.get(&vertex_id) {
            None => {None}
            Some((vertex, neighbors)) => {
                Some(GraphEntry::create(vertex.clone(), neighbors.clone()))
            }
        }
    }

    pub fn from_edges(edges_iter: impl Iterator<Item = (VInt, VInt)>) -> Graph {
        let mut adj_map =
            BTreeMap::<VInt, (Vertex<u32>, Vec<Vertex<u32>>)>::new();
        for (u, v) in edges_iter {
            // RMW: Step 1, Read.
            if adj_map.contains_key(&u) {
                adj_map.get_mut(&u).unwrap().1.push(Vertex::new_successor(v));
            } else {
                adj_map.insert(
                    u,
                    (Vertex::new_vertex(v), vec![Vertex::new_successor(v)])
                );
            }
            // The other direction.
            if adj_map.contains_key(&v) {
                adj_map.get_mut(&v).unwrap().1.push(Vertex::new_predecessor(u));
            } else {
                adj_map.insert(
                    v,
                    (Vertex::new_vertex(v), vec![Vertex::new_predecessor(u)])
                );
            }
        }
        let v_size = adj_map.iter().count() as u32;
        let mut e_size = 0u32;
        for (_, neighbors) in adj_map.iter() {
            for neighbor in &neighbors.1 {
                if neighbor.direction_tag == 1 {
                    e_size += 1;
                }
            }
        }
        Graph {
            adj_map,
            v_size,
            e_size,
        }
    }

    /// Load a graph from file, with community and optional labels.
    pub fn from_graph_file_community (
        file_path: &str,
        is_directed: bool,
        have_label: bool
    ) -> (Graph, Vec<Vec<u32>>, Option<Vec<u32>>) {
        // Load a Graph from a .graph file, like Lou's experiments.
        let graph_file = File::open(file_path).unwrap();
        let graph_reader = BufReader::with_capacity(READ_BUFFER_SIZE, graph_file);
        let mut adj_map =
            BTreeMap::<VInt, (Vertex<u32>, Vec<Vertex<u32>>)>::new();
        let mut community_info = Vec::<u32>::new();
        let mut label_info = Vec::<u32>::new();
        let mut line_count = 0u32;
        for line in graph_reader.lines() {
            line_count += 1;
            if line_count == 1 {
                // The first line, just skip it.
                continue;
            }

            if let Ok(line) = line {
                let tokens :Vec<&str> = line.split_whitespace().collect();
                if tokens[0] == "v" {
                    let parsed_vid = tokens[1].parse().ok().expect("File format error.");
                    let parsed_label = tokens[2].parse().ok().expect("File format error.");
                    let parsed_comm = tokens[3].parse().ok().expect("File format error.");
                    // Process Vertices.
                    adj_map.insert(parsed_vid,
                                   (Vertex::new_vertex(parsed_vid), vec![]));
                    // Process Labels.
                    if have_label {
                        label_info.push(parsed_label);
                    }
                    community_info.push(parsed_comm);
                }
                if tokens[0] == "e" && tokens.len() == 3 {
                    // Process Edges.
                    let mut edge_vec = Vec::new();
                    let parsed_src_vid = tokens[1].parse().ok().expect("File format error.");
                    let parsed_dst_vid = tokens[2].parse().ok().expect("File format error.");
                    edge_vec.push(V32::new_vertex(parsed_src_vid));
                    edge_vec.push(V32::new_vertex(parsed_dst_vid));

                    if edge_vec.len() == 2 {
                        // Whether the edge already exists, the successor direction.
                        let mut successor = edge_vec[1];
                        successor.direction_tag = 1;
                        let mut predecessor = edge_vec[0];
                        predecessor.direction_tag = 2;

                        adj_map.get_mut(&parsed_src_vid).unwrap().1.push(successor.clone());
                        adj_map.get_mut(&parsed_dst_vid).unwrap().1.push(predecessor.clone());

                        if !is_directed {
                            successor.direction_tag = 2;
                            adj_map.get_mut(&parsed_src_vid).unwrap().1.push(successor);
                            predecessor.direction_tag = 1;
                            adj_map.get_mut(&parsed_dst_vid).unwrap().1.push(predecessor);
                        }
                    }
                }
            }
        }

        let v_size = adj_map.iter().count() as u32;
        let mut e_size = 0u32;
        for (_, neighbors) in adj_map.iter() {
            for neighbor in &neighbors.1 {
                if neighbor.direction_tag == 1 {
                    e_size += 1;
                }
            }
        }
        let label_info_opt = if have_label {
            Some(label_info)
        } else {
            None
        };

        // Process the community info.
        let community_struct = community_info.into_iter().enumerate().fold(
            BTreeMap::<VInt, Vec<u32>>::new(),
            |mut acc, (vertex_id, comm_id)| {
                if acc.contains_key(&comm_id) {
                    acc.get_mut(&comm_id).unwrap().push(vertex_id as u32);
                } else {
                    acc.insert(comm_id, vec![vertex_id as u32]);
                }
                acc
            }
        ).values().cloned().collect::<Vec<_>>();

        (Graph {
            adj_map,
            v_size,
            e_size,
        }, community_struct, label_info_opt)
    }

    /// Load a graph from file, with community and community map.
    pub fn from_graph_file_community_map (
        file_path: &str,
        is_directed: bool
    ) -> (Graph, Vec<Vec<u32>>, Vec<u32>) {
        // Load a Graph from a .graph file, like Lou's experiments.
        let graph_file = File::open(file_path).unwrap();
        let graph_reader = BufReader::with_capacity(READ_BUFFER_SIZE, graph_file);
        let mut adj_map =
            BTreeMap::<VInt, (Vertex<u32>, Vec<Vertex<u32>>)>::new();
        let mut community_info = Vec::<u32>::new();
        let mut line_count = 0u32;
        for line in graph_reader.lines() {
            line_count += 1;
            if line_count == 1 {
                // The first line, just skip it.
                continue;
            }

            if let Ok(line) = line {
                let tokens :Vec<&str> = line.split_whitespace().collect();
                if tokens[0] == "v" {
                    let parsed_vid = tokens[1].parse().ok().expect("File format error.");
                    let parsed_comm = tokens[3].parse().ok().expect("File format error.");
                    // Process Vertices.
                    adj_map.insert(parsed_vid,
                                   (Vertex::new_vertex(parsed_vid), vec![]));
                    community_info.push(parsed_comm);
                }
                if tokens[0] == "e" && tokens.len() == 3 {
                    // Process Edges.
                    let mut edge_vec = Vec::new();
                    let parsed_src_vid = tokens[1].parse().ok().expect("File format error.");
                    let parsed_dst_vid = tokens[2].parse().ok().expect("File format error.");
                    edge_vec.push(V32::new_vertex(parsed_src_vid));
                    edge_vec.push(V32::new_vertex(parsed_dst_vid));

                    if edge_vec.len() == 2 {
                        // Whether the edge already exists, the successor direction.
                        let mut successor = edge_vec[1];
                        successor.direction_tag = 1;
                        let mut predecessor = edge_vec[0];
                        predecessor.direction_tag = 2;

                        adj_map.get_mut(&parsed_src_vid).unwrap().1.push(successor.clone());
                        adj_map.get_mut(&parsed_dst_vid).unwrap().1.push(predecessor.clone());

                        if !is_directed {
                            successor.direction_tag = 2;
                            adj_map.get_mut(&parsed_src_vid).unwrap().1.push(successor);
                            predecessor.direction_tag = 1;
                            adj_map.get_mut(&parsed_dst_vid).unwrap().1.push(predecessor);
                        }
                    }
                }
            }
        }

        let v_size = adj_map.iter().count() as u32;
        let mut e_size = 0u32;
        for (_, neighbors) in adj_map.iter() {
            for neighbor in &neighbors.1 {
                if neighbor.direction_tag == 1 {
                    e_size += 1;
                }
            }
        }

        // Process the community info.
        let community_struct = community_info.iter().enumerate().fold(
            BTreeMap::<VInt, Vec<u32>>::new(),
            |mut acc, (vertex_id, comm_id)| {
                if acc.contains_key(comm_id) {
                    acc.get_mut(comm_id).unwrap().push(vertex_id as u32);
                } else {
                    acc.insert(*comm_id, vec![vertex_id as u32]);
                }
                acc
            }
        ).values().cloned().collect::<Vec<_>>();

        (Graph {
            adj_map,
            v_size,
            e_size,
        }, community_struct, community_info)
    }

    /// Load a graph from file without community and labels.
    pub fn from_graph_file(file_path: &str, is_directed: bool) -> Graph {
        // Load a Graph from a .graph file, like Lou's experiments.
        let graph_file = File::open(file_path).unwrap();
        let graph_reader = BufReader::new(graph_file);
        let mut adj_map = 
            BTreeMap::<VInt, (Vertex<u32>, Vec<Vertex<u32>>)>::new();
        let mut line_count = 0u32;
        for line in graph_reader.lines() {
            line_count += 1;
            if line_count == 1 {
                // The first line, just skip it.
                continue;
            }

            if let Ok(line) = line {
                let tokens :Vec<&str> = line.split_whitespace().collect();
                if tokens[0] == "v" {
                    let parsed_vid = tokens[1].parse().ok().expect("File format error.");
                    // Process Vertices.
                    adj_map.insert(parsed_vid,
                                   (Vertex::new_vertex(parsed_vid), vec![]));
                }
                if tokens[0] == "e" && tokens.len() == 3 {
                    // Process Edges.
                    let mut edge_vec = Vec::new();
                    let parsed_src_vid = tokens[1].parse().ok().expect("File format error.");
                    let parsed_dst_vid = tokens[2].parse().ok().expect("File format error.");
                    edge_vec.push(V32::new_vertex(parsed_src_vid));
                    edge_vec.push(V32::new_vertex(parsed_dst_vid));

                    if edge_vec.len() == 2 {
                        // Whether the edge already exists, the successor direction.
                        let mut successor = edge_vec[1];
                        successor.direction_tag = 1;
                        let mut predecessor = edge_vec[0];
                        predecessor.direction_tag = 2;

                        adj_map.get_mut(&parsed_src_vid).unwrap().1.push(successor.clone());
                        adj_map.get_mut(&parsed_dst_vid).unwrap().1.push(predecessor.clone());

                        if !is_directed {
                            successor.direction_tag = 2;
                            adj_map.get_mut(&parsed_src_vid).unwrap().1.push(successor);
                            predecessor.direction_tag = 1;
                            adj_map.get_mut(&parsed_dst_vid).unwrap().1.push(predecessor);
                        }
                    }
                }
            }
        }

        let v_size = adj_map.iter().count() as u32;
        let mut e_size = 0u32;
        for (_, neighbors) in adj_map.iter() {
            for neighbor in &neighbors.1 {
                if neighbor.direction_tag == 1 {
                    e_size += 1;
                }
            }
        }
        Graph {
            adj_map,
            v_size,
            e_size,
        }
    }


    /// Load the label array from the graph file.
    pub fn read_label_array(file_path: &str) -> Vec<u32> {
        // Load a Graph from a .graph file, like Lou's experiments.
        let graph_file = File::open(file_path).unwrap();
        let graph_reader = BufReader::new(graph_file);
        let mut line_count = 0u32;
        let mut label_vec = vec![];

        for line in graph_reader.lines() {
            line_count += 1;
            if line_count == 1 {
                // The first line, just skip it.
                continue;
            }

            if let Ok(line) = line {
                let tokens :Vec<&str> = line.split_whitespace().collect();
                if tokens[0] == "v" {
                    let parsed_label = tokens[2].parse().ok().expect("File format error.");
                    label_vec.push(parsed_label);
                }
            }
        }
        label_vec
    }

    /// Generate the louvain graph for community detection.
    pub fn generate_louvain_graph(&self) -> (LouvainGraph, Vec<u32>) {
        // Process the louvain_vertex_to_vertex_id vec.
        let mut lo_vid_arr = vec![0; self.v_size as usize];
        let mut louvain_graph = LouvainGraph::new(self.v_size as usize);
        let mut vid_lo_map = HashMap::<VInt, usize>::new();
        for (v_idx, (vertex_id, _)) in self.adj_map.iter().enumerate() {
            vid_lo_map.insert(*vertex_id, v_idx);
            lo_vid_arr[v_idx] = *vertex_id;
        }
        for (vertex_id, (_, neighbors)) in &self.adj_map {
            for neighbor in neighbors {
                if neighbor.direction_tag == 1 {
                    // This neighbor is a successor.
                    let source = *vertex_id as usize;
                    let target = neighbor.vertex_id as usize;
                    louvain_graph.insert_edge(source, target, 1.0);
                }
            }
        }
        (louvain_graph, lo_vid_arr)
    }

    pub fn from_txt_file(file_path: &str) -> Graph {
        let graph_file = File::open(file_path).unwrap();
        let graph_reader = BufReader::new(graph_file);
        let mut adj_map = BTreeMap::<VInt, (Vertex<u32>, Vec<Vertex<u32>>)>::new();
        // Read the edge list from the file.
        for line in graph_reader.lines() {
            if let Ok(line) = line {
                let edge_vec: Vec<Vertex<u32>> = line
                .split_whitespace()
                .filter_map(|sv| Some(Vertex {
                    vertex_id: sv.parse().ok().expect("File format error."),
                    timestamp: get_current_timestamp(),
                    direction_tag: 0,
                    tomb: 0u8
                }))
                .collect();
                
                if edge_vec.len() == 2 {
                    // Whether u or v already exists.
                    if !adj_map.contains_key(&edge_vec[0].vertex_id) {
                        adj_map.insert(edge_vec[0].vertex_id, (edge_vec[0], Vec::new()));
                    }

                    if !adj_map.contains_key(&edge_vec[1].vertex_id) {
                        adj_map.insert(edge_vec[1].vertex_id, (edge_vec[1], Vec::new()));
                    }

                    // Whether the edge already exists, the successor direction.
                    let mut successor = edge_vec[1].clone();
                    successor.direction_tag = 1;
                    let mut predecessor = edge_vec[0].clone();
                    predecessor.direction_tag = 2;

                    // The successor direction.
                    if adj_map.get(&edge_vec[0].vertex_id).unwrap().1.contains(&successor) {
                        continue;
                    } else {
                        adj_map.get_mut(&edge_vec[0].vertex_id).unwrap().1.push(successor);
                    }

                    // The predecessor direction.
                    if adj_map.get(&edge_vec[1].vertex_id).unwrap().1.contains(&predecessor) {
                        continue;
                    } else {
                        adj_map.get_mut(&edge_vec[1].vertex_id).unwrap().1.push(predecessor);
                    }
                }
            }
        }
        let v_size = adj_map.iter().count() as u32;
        let mut e_size = 0u32;
        for (_, neighbors) in adj_map.iter() {
            for neighbor in &neighbors.1 {
                if neighbor.direction_tag == 1 {
                    e_size += 1;
                }
            }
        }
        Graph {
            adj_map,
            v_size,
            e_size,
        }
    }

    pub fn get_vertex_count(&self) -> u32 {
        self.v_size
    }

    pub fn get_edge_count(&self) -> u32 {
        self.e_size
    }

    pub fn insert_edge(&mut self, u: VInt, v: VInt) {
        // Weather u or v already exists.
        if !self.adj_map.contains_key(&u) {
            self.adj_map.insert(u, (Vertex::new_vertex(u), Vec::new()));
            self.v_size += 1;
        }

        if !self.adj_map.contains_key(&v) {
            self.adj_map.insert(v, (Vertex::new_vertex(v), Vec::new()));
            self.v_size += 1;
        }

        // The successor direction.
        if self.adj_map.get(&u).unwrap().1.iter().any(|item| item.vertex_id == v && item.direction_tag == 1) {
            return;
        } else {
            self.adj_map.get_mut(&u).unwrap().1.push(Vertex::new_successor(v));
            self.e_size += 1;
        }

        // The predecessor direction.
        if self.adj_map.get(&v).unwrap().1.iter().any(|item| item.vertex_id == u && item.direction_tag == 2) {
            return;
        } else {
            self.adj_map.get_mut(&v).unwrap().1.push(Vertex::new_predecessor(u));
        }
    }

    /// Remove an existing edge from the graph.
    pub fn remove_edge(&mut self, u: &VInt, v: &VInt) {
        // If edge (u, v) exists, perform remove.
        if self.has_edge(u, v) {
            // Perform edge deletion.
            self.adj_map.get_mut(u).unwrap().1.retain(
                |item| !(item.vertex_id == *v && item.direction_tag == 1)
            );
            // Another direction.
            self.adj_map.get_mut(v).unwrap().1.retain(
                |item| !(item.vertex_id == *u && item.direction_tag == 2)
            );
            // Modify the edge count.
            self.e_size -= 1;
        }
        // Else, do nothing.
    }

    pub fn insert_vertex(&mut self, u: VInt) {
        if !self.adj_map.contains_key(&u) {
            self.adj_map.insert(u, (Vertex::new_vertex(u), Vec::new()));
            self.v_size += 1;
        }
    }

    pub(crate) fn get_neighbor(&self, u: &VInt) -> Vec<V32> {
        if !self.adj_map.contains_key(u) {
            vec![]
        } else {
            self.adj_map.get(u).unwrap().1.clone()
        }
    }

    pub(crate) fn get_successor(&self, u: &VInt) -> Vec<V32> {
        if !self.adj_map.contains_key(u) {
            vec![]
        } else {
            self.adj_map.get(u).unwrap().1
                .iter()
                .filter(|&item| item.direction_tag == 1)
                .cloned()
                .collect()
        }
    }

    /// If an edge exists in this graph.
    pub fn has_edge(&self, src_id: &VInt, dst_id: &VInt) -> bool {
        if self.adj_map.contains_key(src_id) {
            for neighbor in &self.adj_map.get(src_id).unwrap().1 {
                // It is a successor, and it exists.
                if neighbor.vertex_id == *dst_id && neighbor.direction_tag == 1 {
                    return true;
                }
            }
        }
        return false;
    }


    pub fn print_graph(&self) {
        for e in self.adj_map.iter() {
            print!("{} -> ", e.0);
            for v in e.1.1.iter() {
                print!("{} ", v);
            }
            println!();
        }
    }

    pub fn generate_undirected_snapshot(&self) -> GraphSnapshot {
        let mut new_adj_map = BTreeMap::<VInt, Vec<VInt>>::new();
        for (vid, (_, neighbors)) in &self.adj_map {
            if !new_adj_map.contains_key(vid) {
                new_adj_map.insert(*vid, vec![]);
            }
            for neighbor in neighbors {
                if neighbor.tomb == 0 && neighbor.direction_tag == 1 && neighbor.vertex_id > *vid {
                    new_adj_map.entry(*vid)
                        .or_insert(vec![])
                        .push(neighbor.vertex_id);
                    // The another direction.
                    new_adj_map.entry(neighbor.vertex_id)
                        .or_insert(vec![])
                        .push(*vid);
                }
            }
        }
        let v_size = new_adj_map.len() as u32;
        let mut e_size = 0u32;
        for vertex_entry in &new_adj_map {
            e_size += vertex_entry.1.len() as u32;
        }
        GraphSnapshot {
            adj_map: new_adj_map,
            v_size,
            e_size,
        }
    }
}

#[cfg(test)]
mod test_graph {
    use crate::graph::Graph;

    #[test]
    fn perform_test() {
        let edge_iter = vec![(1, 2), (2, 3), (3, 1)].into_iter();
        let mut g = Graph::from_edges(edge_iter);
        println!("Vertex count: {}", g.get_vertex_count());
        println!("Edge count: {}", g.get_edge_count());
        g.print_graph();

        g.insert_edge(3, 4);
        println!("After Insert Edge (3, 4)");
        g.print_graph();

        println!("After Insert Vertex 5");
        g.print_graph();
    }

    #[test]
    fn test_load_big_graph() {
        println!("start");
        let g_from_graph = Graph::from_graph_file("data/oregon.graph", false);
        println!("Vertex Count: {}, Edge Count: {}", g_from_graph.v_size, g_from_graph.e_size);
        g_from_graph.print_graph();
    }

    #[test]
    fn test_get_neighbor() {
        println!("start");
        let g_from_graph = Graph::from_graph_file("data/oregon.graph", false);
        let res = g_from_graph.get_neighbor(&0u32);
        print!("Neighbors of V0: ");
        for v in res {
            print!("{} -> ", v);
        }
        println!("END");
    }
}