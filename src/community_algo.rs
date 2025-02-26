use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ops::AddAssign;

use rand::seq::IteratorRandom;
use rand::thread_rng;
use slotmap::DenseSlotMap;

use crate::graph::{GraphSnapshot, VInt};

type CommunityId = slotmap::DefaultKey;

// Define the edge structure used in modularity.
#[derive(Debug, PartialEq)]
#[allow(dead_code)]
pub struct ModEdge {
    source: u32,
    target: u32,
    weight: f32,
}

// Define the community structure.
#[allow(dead_code)]
#[derive(Default, Debug, PartialEq)]
pub struct Community {
    id: CommunityId, // ID of community.
    weight_sum: f64, // Total weight in this community.
    vertex_set: Vec<VInt>, // Logic vertex id in this community.
    connections_weight: RefCell<BTreeMap<CommunityId, f32>>,
    connections_count: RefCell<BTreeMap<CommunityId, i32>>,
}

#[allow(dead_code)]
impl Community {
    fn new(id: CommunityId) -> Community {
        // Create a new empty community.
        Community {
            id,
            .. Default::default()
        }
    }

    pub fn get_vertex_set(&self) -> Vec<VInt> {
        self.vertex_set.clone()
    }

    fn size(&self) -> u32 {
        // Get the vertex count in this community.
        self.vertex_set.len() as u32
    }

    fn seed(&mut self, vertex: VInt, cs: &CommunityStructure) {
        // Push a new vertex in this community.
        self.vertex_set.push(vertex);
        self.weight_sum += cs.weight_list[vertex as usize];
    }

    fn add(&mut self, vertex: u32, cs: &CommunityStructure){
        // Add a vertex to this community.
        self.vertex_set.push(vertex);
        self.weight_sum += cs.weight_list[vertex as usize];
    }

    fn remove(&mut self, vertex: u32, cs: &mut CommunityStructure) {
        // Remove the vertex from this community.
        self.vertex_set.retain(|&n| n != vertex);
        self.weight_sum -= cs.weight_list[vertex as usize];
        // If this community remain no vertex, remove this community from cs.
        if self.vertex_set.is_empty() {
            cs.community_list.retain(|&c| c != self.id);
        }
    }
}

#[derive(Default, Debug)]
#[allow(dead_code)]
pub struct CommunityCatalog {
    pub map: DenseSlotMap<CommunityId, Community>
}

#[allow(dead_code)]
impl CommunityCatalog {
    pub fn create_new(&mut self) -> CommunityId {
        self.map.insert_with_key(|id| Community::new(id))
    }

    #[inline]
    pub fn get(&self, key: &CommunityId) -> Option<&Community> {
        // Get the reference of the wanted community.
        self.map.get(*key)
    }

    #[inline]
    pub fn get_mut(&mut self, key: &CommunityId) -> Option<&mut Community> {
        // Get the mutable reference of the wanted community.
        self.map.get_mut(*key)
    }

    #[inline]
    pub fn remove(&mut self, key: &CommunityId) {
        // Remove the community structure.
        self.map.remove(*key);
    }
}

#[derive(Default, Debug, PartialEq)]
pub struct CommunityStructure {
    community_count: u32, // Number of communities.
    community_list: Vec<CommunityId>,  // Direct communities.
    vertex_connections_weight: Vec<BTreeMap<CommunityId, f32>>,
    vertex_connections_count: Vec<BTreeMap<CommunityId, i32>>,
    vertex_community_map: Vec<CommunityId>, // Locate the community of a vertex.
    weight_list: Vec<f64>, // Vertex weight, one per vertex.
    topology: Vec<Vec<ModEdge>>,  // Topology structure, one per vertex.
    inv_map: BTreeMap<VInt, CommunityId>,
    graph_weight_sum: f64,
}

#[allow(dead_code)]
impl CommunityStructure {
    pub fn new(graph: &GraphSnapshot, modularity: &mut Modularity) -> CommunityStructure {
        let vertex_count = graph.v_size;
        let cc = &mut modularity.cc;
        let mut cs = CommunityStructure {
            community_count: vertex_count,
            community_list: Vec::new(),
            vertex_connections_weight: Vec::with_capacity(vertex_count as usize),
            vertex_connections_count: Vec::with_capacity(vertex_count as usize),
            vertex_community_map: Vec::with_capacity(vertex_count as usize),
            weight_list: Vec::with_capacity(vertex_count as usize),
            topology: Vec::with_capacity(vertex_count as usize),
            inv_map: BTreeMap::default(),
            graph_weight_sum: 0.0
        };
        let mut index = 0u32;
        // Create one community and one inverse community per node.
        // All weights to 0.0.
        for _ in 0..graph.adj_map.len() {
            cs.vertex_community_map.push(cc.create_new());
            cs.vertex_connections_weight.push(BTreeMap::default());
            cs.vertex_connections_count.push(BTreeMap::default());
            cs.weight_list.push(0.0);
            cc.get_mut(&cs.vertex_community_map[index as usize]).unwrap().seed(index, &cs);

            // New hidden community.
            let hidden = cc.create_new();
            cc.get_mut(&hidden).unwrap().vertex_set.push(index);
            cs.inv_map.insert(index, hidden);

            cs.community_list.push(cs.vertex_community_map[index as usize]);
            index += 1;
        }

        for (vertex, neighbors) in graph.adj_map.clone() {
            cs.topology.push(Vec::new());
            for neighbor in neighbors {
                let weight: f32 = 1.0;
                // Finally add a single edge with the summed weight of all parallel edges.
                cs.weight_list[vertex as usize] += weight as f64;
                let modularity_edge = ModEdge {
                    source: vertex,
                    target: neighbor,
                    weight
                };
                cs.topology[vertex as usize].push(modularity_edge);
                let adj_com = cc.get(&cs.vertex_community_map[neighbor as usize]).unwrap();

                cs.vertex_connections_weight[vertex as usize].insert(adj_com.id, weight);
                cs.vertex_connections_count[vertex as usize].insert(adj_com.id, 1);

                let vertex_com = cc.get(&cs.vertex_community_map[vertex as usize]).unwrap();
                vertex_com.connections_weight.borrow_mut().insert(adj_com.id, weight);
                vertex_com.connections_count.borrow_mut().insert(adj_com.id, 1);

                cs.vertex_connections_weight[neighbor as usize].insert(vertex_com.id, weight);
                cs.vertex_connections_count[neighbor as usize].insert(vertex_com.id, 1);

                adj_com.connections_weight.borrow_mut().insert(vertex_com.id, weight);
                adj_com.connections_count.borrow_mut().insert(vertex_com.id, 1);

                cs.graph_weight_sum += weight as f64;
            }
        }
        cs.graph_weight_sum /= 2.0;
        cs
    }

    fn add_vertex_to(&mut self, vertex: u32, com_id: CommunityId, cc: &mut CommunityCatalog) {
        // Add a vertex to a specific community.
        // First of all, define an inline function usually used in next stages.
        #[inline]
        fn add_vertex<V: AddAssign + Copy + From<u8>>(map: &mut BTreeMap<CommunityId,V>, key: CommunityId, weight: V) {
            let w = map.entry(key).or_insert(V::from(0));
            *w += weight;
        }

        self.vertex_community_map[vertex as usize] = com_id;

        {
            let com = cc.get_mut(&com_id).unwrap();
            com.add(vertex, self);
        }

        for e in &self.topology[vertex as usize] {
            let neighbor = &e.target;

            add_vertex(&mut self.vertex_connections_weight[*neighbor as usize], com_id, e.weight);
            add_vertex(&mut self.vertex_connections_count[*neighbor as usize], com_id, 1);

            let adj_com = cc.get(&self.vertex_community_map[*neighbor as usize]).unwrap();

            add_vertex(&mut adj_com.connections_weight.borrow_mut(), com_id, e.weight);
            add_vertex(&mut adj_com.connections_count.borrow_mut(), com_id, 1);

            add_vertex(&mut self.vertex_connections_weight[vertex as usize], adj_com.id, e.weight);
            add_vertex(&mut self.vertex_connections_count[vertex as usize], adj_com.id, 1);

            if com_id != adj_com.id {
                let com = cc.get(&com_id).unwrap();
                add_vertex(&mut com.connections_weight.borrow_mut(), adj_com.id, e.weight);
                add_vertex(&mut com.connections_count.borrow_mut(), adj_com.id, 1);
            }
        }
    }

    pub fn remove_vertex_from_community(&mut self, vertex: VInt, cc: &mut CommunityCatalog) {
        // Remove the target vertex from the community.
        // Similar to the previous function, define a new usually used inline function.
        #[inline]
        fn remove_vertex(weights_map: &mut BTreeMap<CommunityId, f32>,
                         count_map: &mut BTreeMap<CommunityId, i32>,
                         key: CommunityId, weight: f32) {
            let count = count_map.get(&key).unwrap().clone();
            if count - 1 == 0 {
                weights_map.remove(&key);
                count_map.remove(&key);
            } else {
                let count = count_map.get_mut(&key).unwrap();
                *count -= 1;
                let w = weights_map.get_mut(&key).unwrap();
                *w -= weight;
            }
        }

        {
            let community = cc.get(&self.vertex_community_map[vertex as usize]).unwrap();
            for e in &self.topology[vertex as usize] {
                let neighbor = &e.target;
                // Remove Vertex Connection to this community.
                remove_vertex(&mut self.vertex_connections_weight[*neighbor as usize],
                              &mut self.vertex_connections_count[*neighbor as usize],
                              community.id.clone(), e.weight);

                // Remove Adjacency Community's connection to this community.
                let adj_com = cc.get(&self.vertex_community_map[*neighbor as usize]).unwrap();
                remove_vertex(&mut adj_com.connections_weight.borrow_mut(),
                              &mut adj_com.connections_count.borrow_mut(),
                              community.id.clone(), e.weight);

                if vertex == *neighbor {
                    continue;
                }

                if adj_com.id != community.id {
                    remove_vertex(&mut community.connections_weight.borrow_mut(),
                                  &mut community.connections_count.borrow_mut(),
                                  adj_com.id, e.weight);
                }

                remove_vertex(&mut self.vertex_connections_weight[vertex as usize],
                              &mut self.vertex_connections_count[vertex as usize],
                              adj_com.id, e.weight);
            }
        }

        {
            let community = cc.get_mut(&self.vertex_community_map[vertex as usize]).unwrap();
            community.remove(vertex, self);
        }
    }

    fn move_vertex_to(&mut self, vertex: VInt, to: CommunityId, cc: &mut CommunityCatalog) {
        // Move a vertex from one community to another.
        // println!("Move Vertex {} from Community {:?} to Community {:?}",
        //          vertex, self.vertex_community_map[vertex as usize], to);
        self.remove_vertex_from_community(vertex, cc);
        self.add_vertex_to(vertex, to, cc);
    }

    fn zoom_out(&mut self, cc: &mut CommunityCatalog) {
        let community_count = self.community_list.len();
        self.community_list.sort();
        let mut new_topology: Vec<Vec<ModEdge>> = Vec::with_capacity(community_count);
        let mut index = 0u32;
        let mut vertex_2_community: Vec<CommunityId> = Vec::with_capacity(community_count);
        let mut vertex_connections_weight: Vec<BTreeMap<CommunityId, f32>> = Vec::with_capacity(community_count);
        let mut vertex_connections_count: Vec<BTreeMap<CommunityId, i32>> = Vec::with_capacity(community_count);
        let mut new_inv_map: BTreeMap<VInt, CommunityId> = BTreeMap::default();

        for com_id in &self.community_list {
            vertex_connections_weight.push(BTreeMap::default());
            vertex_connections_count.push(BTreeMap::default());

            new_topology.push(Vec::new());
            vertex_2_community.push(cc.create_new());

            let mut weight_sum = 0.0f32;
            let hidden_id = cc.create_new();

            let vertices = cc.get(com_id).unwrap().vertex_set.clone();
            for vertex_id in vertices {
                let old_hidden_vertices = {
                    let old_hidden = cc.get(&self.inv_map.get(&vertex_id).unwrap()).unwrap();
                    old_hidden.vertex_set.clone()
                };
                let hidden = cc.get_mut(&hidden_id).unwrap();
                hidden.vertex_set.extend(old_hidden_vertices);
            }

            new_inv_map.insert(index, hidden_id);
            {
                let com = cc.get(com_id).unwrap();
                for adj_com_id in com.connections_weight.borrow().keys() {
                    let target = self.community_list.binary_search(adj_com_id).unwrap();
                    let weight = com.connections_weight.borrow().get(adj_com_id).unwrap().clone();
                    weight_sum += if target as u32 == index {
                        2.0 * weight
                    } else {
                        weight
                    };

                    let e = ModEdge {
                        source: index,
                        target: target as u32,
                        weight
                    };
                    new_topology[index as usize].push(e);
                }
            }

            self.weight_list[index as usize] = weight_sum as f64;
            let com = cc.get_mut(&vertex_2_community[index as usize]).unwrap();
            com.seed(index, &self);

            index += 1;
        }

        for com_id in &self.community_list {
            cc.remove(com_id);
        }
        self.community_list.clear();

        for i in 0..community_count {
            let com = cc.get(&vertex_2_community[i]).unwrap();
            self.community_list.push(com.id);
            for e in &new_topology[i] {
                vertex_connections_weight[i].insert(vertex_2_community[e.target as usize], e.weight);
                vertex_connections_count[i].insert(vertex_2_community[e.target as usize], 1);
                com.connections_weight.borrow_mut().insert(vertex_2_community[e.target as usize], e.weight);
                com.connections_count.borrow_mut().insert(vertex_2_community[e.target as usize], 1);
            }
        }

        self.community_count = community_count as u32;
        self.topology = new_topology;
        self.vertex_community_map = vertex_2_community;
        self.vertex_connections_weight = vertex_connections_weight;
        self.vertex_connections_count = vertex_connections_count;
    }
}

// Define the core of community detection, i.e., Modularity.
#[allow(dead_code)]
pub struct Modularity {
    modularity: f64,
    modularity_resolution: f64,
    is_randomized: bool,
    use_weight: bool,
    resolution: f64,
    noise: u32,
    pub cc : CommunityCatalog,
    pub community_by_node: Vec<i32>,
}

#[allow(dead_code)]
impl Modularity {
    pub fn new(resolution: f64, noise: u32) -> Modularity {
        // Create a new modularity struct.
        Modularity {
            modularity: 0.0,
            modularity_resolution: 0.0,
            is_randomized: false,
            use_weight: true,
            resolution,
            noise,
            cc: Default::default(),
            community_by_node: Default::default()
        }
    }

    pub fn execute(&mut self, graph: &GraphSnapshot) -> (f64, f64) {
        // Perform the community detection algorithm.
        let mut structure = CommunityStructure::new(&graph, self);
        let mut communities: Vec<i32> = vec![0; graph.v_size as usize];

        if graph.v_size > 0 {
            let (modularity, modularity_resolution) = self.compute_modularity(
                graph, &mut structure, &mut communities);
            self.modularity = modularity;
            self.modularity_resolution = modularity_resolution;
        } else {
            self.modularity = 0.0;
            self.modularity_resolution = 0.0;
        }
        // Save the result.
        self.community_by_node = communities;

        // Return result.
        (self.modularity, self.modularity_resolution)

    }

    fn compute_modularity(&mut self,
                          graph: &GraphSnapshot,
                          cs: &mut CommunityStructure,
                          communities: &mut Vec<i32>) -> (f64, f64) {
        let node_degrees = cs.weight_list.clone();
        let total_weight = cs.graph_weight_sum;

        let mut some_change = true;
        while some_change {
            some_change = false;
            let mut local_change = true;
            while local_change {
                local_change = false;
                let mut start = 0usize;
                if self.is_randomized {
                    let mut rng = thread_rng();
                    start = (1..cs.community_count as usize).choose(&mut rng).unwrap();
                }
                for step in 0..cs.community_count {
                    let i = (step + start as u32) % cs.community_count;
                    if let Some(best_community) = self.update_best_community(cs, i, self.resolution) {
                        if cs.vertex_community_map[i as usize] != best_community {
                            cs.move_vertex_to(i, best_community, &mut self.cc);
                            local_change = true;
                        }
                    }
                }
                some_change = local_change || some_change;
            }
            if some_change {
                cs.zoom_out(&mut self.cc);
                println!("Zooming Out: {} communities left", cs.community_count);
            }
        }
        let mut com_structure: Vec<usize> = vec![0; graph.v_size as usize];
        let noise_map = self.fill_com_structure(cs, &mut com_structure);
        let degree_count = self.fill_degree_count(graph, cs, &mut com_structure, &node_degrees);

        let computed_modularity = self.final_q(&mut com_structure, &degree_count, graph, total_weight, 1.0);
        let computed_modularity_resolution = self.final_q(&mut com_structure,
                                                          &degree_count, graph, total_weight, self.resolution);
        for i in 0..communities.len() {
            communities[i] = if noise_map[i] { -1 } else { com_structure[i] as i32 }
        }
        (computed_modularity, computed_modularity_resolution)
    }

    fn fill_com_structure(&self, cs: &CommunityStructure, com_structure: &mut Vec<usize>) -> Vec<bool> {
        let mut noise_map: Vec<bool> = vec![false; com_structure.len()];
        for (count, com_id) in cs.community_list.iter().enumerate() {
            let com = self.cc.get(com_id).unwrap();
            for vertex in &com.vertex_set {
                let hidden_id = cs.inv_map.get(vertex).unwrap();
                let hidden = self.cc.get(hidden_id).unwrap();
                let is_noise = hidden.vertex_set.len() as u32 <= self.noise;
                for vertex_int in &hidden.vertex_set {
                    com_structure[*vertex_int as usize] = count;
                    noise_map[*vertex_int as usize] = is_noise;
                }
            }
        }
        noise_map
    }

    fn fill_degree_count(&self, graph: &GraphSnapshot, cs: &CommunityStructure,
                         com_structure: &Vec<usize>, vertex_degrees: &Vec<f64>) -> Vec<f64> {
        let mut degree_count: Vec<f64> = vec![0.0; cs.community_list.len()];
        for vertex_entry in graph.adj_map.iter() {
            let vertex = vertex_entry.0;
            if self.use_weight {
                degree_count[com_structure[*vertex as usize]] += vertex_degrees[*vertex as usize];
            } else {
                degree_count[com_structure[*vertex as usize]] +=
                    graph.adj_map.get(vertex).unwrap().len() as f64;
            }
        }
        degree_count
    }

    fn update_best_community(&mut self, cs: &CommunityStructure,
                             vertex_id: u32, current_resolution: f64) -> Option<CommunityId> {
        let mut best = 0.0f64;
        let mut best_community: Option<CommunityId> = None;

        for com_id in cs.vertex_connections_weight[vertex_id as usize].keys() {
            let q_value = self.q(vertex_id, com_id, cs, current_resolution);
            if q_value > best {
                best = q_value;
                best_community = Some(*com_id);
            }
        }
        best_community
    }

    fn q(&self, vertex: u32, com_id: &CommunityId,
         cs: &CommunityStructure, current_resolution: f64) -> f64 {
        let mut edges_to = 0.0f64;
        if let Some(edge_to_float) = cs.vertex_connections_weight[vertex as usize].get(com_id) {
            edges_to = *edge_to_float as f64;
        }
        let weight_sum = self.cc.get(com_id).unwrap().weight_sum;
        let vertex_weight = cs.weight_list[vertex as usize];
        let mut q_value = current_resolution * edges_to -
            (vertex_weight * weight_sum) / (2.0 * cs.graph_weight_sum);

        let vertex_communities_len = self.cc.get(
            &cs.vertex_community_map[vertex as usize]).unwrap().size();
        if (cs.vertex_community_map[vertex as usize] == *com_id) && (vertex_communities_len > 1) {
            q_value = current_resolution * edges_to -
                (vertex_weight * (weight_sum - vertex_weight)) / (2.0 * cs.graph_weight_sum);
        }
        if (cs.vertex_community_map[vertex as usize] == *com_id) && (vertex_communities_len == 1) {
            q_value = 0.0;
        }
        q_value
    }

    fn final_q(&self, com_structure: &Vec<usize>, degrees: &Vec<f64>, graph: &GraphSnapshot,
               total_weight: f64, used_resolution: f64) -> f64 {
        let mut res = 0.0f64;
        let mut internal: Vec<f64> = vec![0.0; degrees.len()];

        for vertex_entry in &graph.adj_map {
            let vertex = vertex_entry.0;
            let neighbors = vertex_entry.1;
            for neighbor in neighbors {
                if com_structure[*neighbor as usize] == com_structure[*vertex as usize]{
                    if self.use_weight {
                        internal[com_structure[*neighbor as usize]] += 1.0;
                    } else {
                        internal[com_structure[*neighbor as usize]] += 1.0;
                    }
                }
            }
        }
        for i in 0..degrees.len() {
            internal[i] /= 2.0;
            res += used_resolution * (internal[i] / total_weight) -
                (degrees[i] / (2.0 * total_weight)).powi(2);
        }
        res
    }
}

