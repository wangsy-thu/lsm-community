use std::borrow::{Borrow, BorrowMut};
// A community table which includes a vertex-to-community map and a community manager.
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::graph::{GraphSnapshot, VInt};

pub(crate) type CommID = VInt;
pub(crate) type CommStructure = Vec<Vec<VInt>>;

// Define the metadata of a community.
#[allow(dead_code)]
pub(crate) struct CommMeta {
    comm_id: CommID,  // The ID of the community.
    contain_vertices: BTreeSet<VInt> // Vertex contained in this community.
}

#[allow(dead_code)]
impl CommMeta {

    pub(crate) fn create(c_id: CommID, vertex_list: &Vec<VInt>) -> CommMeta {
        // Create a new community in memory.
        CommMeta {
            comm_id: c_id,
            contain_vertices: vertex_list.iter()
                .fold(BTreeSet::new(), |mut v_list, vertex| {
                    v_list.insert(vertex.clone());
                    v_list
            })
        }
    }

    pub(crate) fn seed(&mut self, vertex_id: VInt) {
        // Insert a vertex to this community.
        self.contain_vertices.insert(vertex_id);
    }

    pub(crate) fn remove(&mut self, vertex_id: &VInt) {
        // Remove a vertex from this community.
        if self.contain_vertices.contains(vertex_id) {
            self.contain_vertices.remove(vertex_id);
        }
    }
}

/// Community ID manager, which is used for management the IDs of the communities.
pub(crate) struct IDManager {
    comm_id_set: BTreeSet<CommID> // IDs contained in this manager.
}

#[allow(dead_code)]
impl IDManager {
    pub(crate) fn create() -> IDManager {
        // Create a new empty manager.
        IDManager {
            comm_id_set: Default::default()
        }
    }

    pub(crate) fn allocate_comm(&mut self) -> CommID {
        // Allocate a new community ID.
        if self.comm_id_set.is_empty() {
            self.comm_id_set.insert(0u32);
            return 0u32;
        }
        let pre_new_id = self.comm_id_set.iter().next_back().unwrap().clone();
        self.comm_id_set.insert(pre_new_id + 1);
        pre_new_id + 1
    }

    pub(crate) fn remove_comm(&mut self, comm_id: CommID) {
        // Remove a community.
        if self.comm_id_set.contains(&comm_id) {
            self.comm_id_set.remove(&comm_id);
        }
    }

    pub(crate) fn contains(&self, comm_id: CommID) -> bool {
        // Check whether this community id exists in the manager.
        self.comm_id_set.contains(&comm_id)
    }
}

#[allow(dead_code)]
pub(crate) struct CommManager {
    comm_id_manager: IDManager,
    community_map: BTreeMap<CommID, CommMeta>
}

#[allow(dead_code)]
impl CommManager {
    pub(crate) fn create() -> CommManager {
        CommManager {
            comm_id_manager: IDManager::create(),
            community_map: Default::default()
        }
    }

    pub(crate) fn build_from_comm_structure(community_structure: &CommStructure) -> CommManager {
        // Build a community manager from a community structure.
        let mut comm_map = BTreeMap::<CommID, CommMeta>::new();
        // Build community-to-vertex map.
        let mut id_manager = IDManager::create();
        for community in community_structure {
            let new_comm_id = id_manager.allocate_comm();
            let comm_meta = CommMeta::create(new_comm_id, community);
            comm_map.insert(new_comm_id, comm_meta);
        }
        CommManager {
            comm_id_manager: id_manager,
            community_map: comm_map
        }
    }

    pub(crate) fn get_comm_map_ref(&self) -> &BTreeMap<CommID, CommMeta> {
        // Borrow the community map.
        self.community_map.borrow()
    }

    pub(crate) fn get_comm_map_mut(&mut self) -> &mut BTreeMap<CommID, CommMeta> {
        // Borrow the mutable reference of community map.
        self.community_map.borrow_mut()
    }

    pub(crate) fn get_comm_mut(&mut self, comm_id: &CommID) -> Option<&mut CommMeta> {
        // Get the mutable reference of the specific.
        if self.community_map.contains_key(comm_id) {
            Some(self.community_map.get_mut(comm_id).unwrap())
        } else {
            None
        }
    }
}

/// Community Table, used for lookup community information for vertex,
/// and locate the corresponding community bucket of a vertex.
/// The vertex may be inserted or queried.
pub struct CommTable {
    pub vertex_community_map: HashMap<VInt, CommID>, // Map from a vertex to a community.
    community_manager: CommManager // A community manager field.
}

#[allow(dead_code)]
impl CommTable {

    pub(crate) fn get_vertex_map_ref(&self) -> &HashMap<VInt, CommID> {
        self.vertex_community_map.borrow()
    }

    /// Change the community table after performing the vertex escaping.
    pub fn vertex_escape(
        &mut self,
        source_community_id: &CommID,
        vertex_community_map: &HashMap<VInt, CommID>
    ) {
        // Main steps of the vertex escape.
        {
            // Step 1. Locate the source community.
            let source_comm = self.community_manager.get_comm_mut(source_community_id).unwrap();
            for (vertex, _) in vertex_community_map {
                // Step 2. Remove the corresponding vertices.
                source_comm.remove(vertex);
            }
        }

        for (vertex, dst_comm_id) in vertex_community_map {
            // Step 3. Add the removed vertices to corresponding communities.
            let dst_comm = self.community_manager.get_comm_mut(dst_comm_id).unwrap();
            dst_comm.seed(*vertex);
            // Step 4. Modify the community table (just overwrite directly).
            self.vertex_community_map.insert(*vertex, *dst_comm_id);
        }
    }

    pub(crate) fn read_community_graph(file_name: &str) -> (GraphSnapshot, CommStructure) {
        // Read graphs and their community structures from files.
        // Load a Graph from a .graph file, like Lou's experiments.
        let graph_file = File::open(file_name).unwrap();
        let graph_reader = BufReader::new(graph_file);
        let mut adj_map = BTreeMap::<VInt, Vec<VInt>>::new();
        let mut comm_structure_map = HashMap::<VInt, Vec<VInt>>::new();
        let mut line_count = 0u32;
        for line in graph_reader.lines() {
            line_count += 1;
            if line_count == 1 {
                // The first line, just skip it.
                continue;
            }

            if let Ok(line) = line {
                let tokens :Vec<&str> = line.split_whitespace().collect();
                if tokens[0] == "v" && tokens.len() == 4 {
                    let parsed_vid = tokens[1].parse::<VInt>().ok().expect("File format error.");
                    let comm_id = tokens[3].parse::<CommID>().ok().expect("File format error.");
                    // Process Vertices, and Record Community.
                    adj_map.insert(parsed_vid, vec![]);
                    comm_structure_map.entry(comm_id).or_insert_with(Vec::new).push(parsed_vid);
                }
                if tokens[0] == "e" && tokens.len() == 3 {
                    // Process Edges.
                    let mut edge_vec = Vec::new();
                    let parsed_src_vid = tokens[1].parse().ok().expect("File format error.");
                    let parsed_dst_vid = tokens[2].parse().ok().expect("File format error.");
                    edge_vec.push(parsed_src_vid);
                    edge_vec.push(parsed_dst_vid);

                    if edge_vec.len() == 2 {
                        // Whether the edge already exists, the successor direction.
                        let successor = edge_vec[1];
                        let predecessor = edge_vec[0];

                        adj_map.get_mut(&parsed_src_vid).unwrap().push(successor);
                        adj_map.get_mut(&parsed_dst_vid).unwrap().push(predecessor);
                    }
                }
            }
        }

        let v_size = adj_map.iter().count() as u32;
        let mut e_size = 0u32;
        for vertex_entry in adj_map.iter() {
            e_size += vertex_entry.1.len() as u32;
        }
        let g_snapshot = GraphSnapshot {
            adj_map,
            v_size,
            e_size,
        };
        let comm_structure = comm_structure_map.into_iter().fold(Vec::new(), |mut comm_structure, vertices| {
                comm_structure.push(vertices.1);
                comm_structure
            });
        (g_snapshot, comm_structure)
    }

    pub(crate) fn build_from_comm_structure(community_structure: &CommStructure) -> CommTable {
        // Build the community manager.
        let comm_manager = CommManager::build_from_comm_structure(community_structure);
        // Build the vc map.
        let mut vc_map = HashMap::<VInt, CommID>::new();
        for (comm_id, comm_meta) in comm_manager.get_comm_map_ref() {
            for vertex in &comm_meta.contain_vertices {
                vc_map.insert(vertex.clone(), comm_id.clone());
            }
        }
        CommTable {
            vertex_community_map: vc_map,
            community_manager: comm_manager
        }
    }

    pub(crate) fn look_up_community(&self, vertex_id: &VInt) -> Option<CommID> {
        // Find the community of a specific vertex.
        if self.vertex_community_map.contains_key(vertex_id) {
            Some(self.vertex_community_map.get(vertex_id).unwrap().clone())
        } else {
            None
        }
    }

    pub(crate) fn insert_vertex(&mut self, vertex_id: VInt, comm_id: CommID) {
        // Insert a new vertex into LSM-Community.
        // Firstly, check whether the vertex exists.
        if !self.vertex_community_map.contains_key(&vertex_id) {
            self.vertex_community_map.insert(vertex_id, comm_id);
            match self.community_manager.get_comm_mut(&comm_id) {
                None => {
                    // Perform the logic of generating a new community.
                    // But this part is optional, may be deleted in the future.
                }
                Some(comm_meta) => {comm_meta.seed(vertex_id)}
            }
        }
    }

    pub(crate) fn print_table(&self) {
        // Print this table.
        for (comm_id, comm_meta) in &self.community_manager.community_map {
            println!("Community: {}, Vertices: {:?}", comm_id, comm_meta.contain_vertices);
        }
    }
}

#[allow(unused_imports)]
mod tests {
    use crate::comm_table::CommTable;

    #[test]
    fn test_read_comm_structure() {
        // Generate a new community structure for test.
        // Read a community structure from file.
        let (graph_snapshot, comm_structure) =
            CommTable::read_community_graph("data/example.graph");
        graph_snapshot.print_graph();
        for community in comm_structure {
            println!("Community: {:?}", community);
        }
    }

    #[test]
    fn test_build_comm_structure() {
        // Build the community table.
        let (_, comm_structure) =
            CommTable::read_community_graph("data/example.graph");
        let comm_table = CommTable::build_from_comm_structure(&comm_structure);
        comm_table.print_table();
    }

    #[test]
    fn test_insert_comm_structure() {
        // Test the insertion of vertex in the community structure.
        // Build the community table.
        let (_, comm_structure) =
            CommTable::read_community_graph("data/example.graph");
        let mut comm_table = CommTable::build_from_comm_structure(&comm_structure);
        comm_table.insert_vertex(13, 2);
        comm_table.print_table();
    }

    #[test]
    fn test_lookup() {
        let (_, comm_structure) =
            CommTable::read_community_graph("data/example.graph");
        let comm_table = CommTable::build_from_comm_structure(&comm_structure);
        match comm_table.look_up_community(&1) {
            None => {}
            Some(comm_id) => {
                println!("V1 in Community: {}", comm_id);
            }
        }
    }
}

