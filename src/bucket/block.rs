use std::collections::BTreeMap;
use std::mem::size_of;
use std::u32;

use byteorder::{ByteOrder, LittleEndian};
use dashmap::DashMap;
use num::traits::ToBytes;

use crate::bucket::page::Page;
use crate::comm_table::CommID;
use crate::community::CommNeighbors;
use crate::config::{BLOCK_SIZE, PAGE_DATA_CAPACITY};
use crate::graph::VInt;
use crate::types::{GraphEntry, PartialDecode, V32, V32_SIZE};
use crate::types::{Decode, Encode};

#[allow(dead_code)]
pub(crate) const CSR_META_SIZE :usize = 3 * size_of::<u32>();

#[allow(dead_code)]
pub(crate) const TREE_META_SIZE :usize = 3 * size_of::<u32>();

#[allow(dead_code)]
pub(crate) const KV_META_SIZE :usize = 3 * size_of::<u32>();

// Define the DATA_SIZE.
#[allow(dead_code)]
pub(crate) const CSR_DATA_SIZE :usize = BLOCK_SIZE - CSR_META_SIZE;

// Define the KV_DATA_SIZE
#[allow(dead_code)]
pub(crate) const KV_DATA_SIZE :usize = BLOCK_SIZE - KV_META_SIZE;

/// Define a new trait, i.e., Query, to let Comm Block support get_neighbor.
#[allow(dead_code)]
pub(crate) trait Query {
    /// Get neighbors function.
    fn get_neighbors(&self, vertex_id: &VInt) -> Option<Vec<V32>>;

    /// Generate entries for caching.
    fn generate_entries(&self) -> Vec<GraphEntry<V32>>;
}


/// Define a new trait, i.e., Serialize, to let Comm Block support load from several pages.
#[allow(dead_code)]
pub(crate) trait Serialize {
    /// Encode something to number of pages.
    fn encode_to_pages(&self) -> Vec<Page>;
}

/// Define a new trait, i.e., Deserialize, to let Comm Block support load from several pages.
#[allow(dead_code)]
pub(crate) trait Deserialize: Sized {
    /// Decode something from number of pages.
    fn decode_from_pages(page_list: &Vec<Page>) -> Option<Self>;
}

// Community Block Enum, represent the sorted blocks and KV blocks.
pub(crate) enum CommBlock {
    CSR(CSRCommBlock),  // Sorted part.
    KV(KVCommBlock)  // KV part.
}

/// Tree Bucket is a storage unit which stores each community in each layer.
/// Each tree bucket represents a community (collection of IDs) in each layer.
/// The basic format is like the CSR, consists of arrays and offsets.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct TreeBlock {
    pub(crate) block_type: u32, // Type of blocks, default 2, represents tree.
    pub(crate) max_comm_id: u32, // Maximum community id in this tree block.
    pub(crate) min_comm_id: u32, // Minimum community id in this tree block.
    pub(crate) comm_count: u32, // Number of community stored in this block.
    pub(crate) neighbor_count: u32, // Number of neighbor communities in this block.
    pub(crate) comm_list: Vec<(u32, u32)>, // ID of community in each layer, with offsets.
    pub(crate) neighbor_comm_list: Vec<(u32, u32)>, // ID of neighbors, with offset.
    pub(crate) bridge_list: Vec<(u32, u32)>, // Edges in the next layer work as bridges.
}

impl Encode for TreeBlock {
    fn encode(&self) -> Vec<u8> {
        // Encode a community block to byte slice.
        let mut encode_bytes = Vec::<u8>::new();

        // Encode the meta data part.
        encode_bytes.extend_from_slice(&self.block_type.to_le_bytes());
        encode_bytes.extend_from_slice(&self.max_comm_id.to_le_bytes());
        encode_bytes.extend_from_slice(&self.min_comm_id.to_le_bytes());

        // Encode the community count and neighbor count.
        encode_bytes.extend_from_slice(&self.comm_count.to_le_bytes());
        encode_bytes.extend_from_slice(&self.neighbor_count.to_le_bytes());

        // Encode the community list and their offsets.
        for (comm_id, offset) in &self.comm_list {
            // The vertex information.
            encode_bytes.extend_from_slice(&comm_id.to_le_bytes());
            // The offset part.
            encode_bytes.extend_from_slice(&offset.to_le_bytes());
        }

        // Encode the neighbor community list and their offset part.
        for (neighbor_comm, offset) in &self.neighbor_comm_list {
            // Encode the neighbors.
            encode_bytes.extend_from_slice(&neighbor_comm.to_le_bytes());
            // With their offsets.
            encode_bytes.extend_from_slice(&offset.to_le_bytes());
        }

        // The last one, the bridge list.
        for (src_id, dst_id) in &self.bridge_list {
            // Encode the neighbors.
            encode_bytes.extend_from_slice(&src_id.to_le_bytes());
            // With their offsets.
            encode_bytes.extend_from_slice(&dst_id.to_le_bytes());
        }

        // Return value.
        encode_bytes
    }
}

impl Decode for TreeBlock {
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        // Check whether the bytes are valid.
        if bytes.len() < CSR_META_SIZE {
            None
        } else {
            // Parse the meta data part.
            let mut parse_index = 0usize;
            let block_type = LittleEndian::read_u32(&bytes[parse_index..parse_index + size_of::<u32>()]);
            parse_index += size_of::<u32>();
            let max_comm_id = LittleEndian::read_u32(&bytes[parse_index..parse_index + size_of::<u32>()]);
            parse_index += size_of::<u32>();
            let min_comm_id = LittleEndian::read_u32(&bytes[parse_index..parse_index + size_of::<u32>()]);
            parse_index += size_of::<u32>();
            let comm_count = LittleEndian::read_u32(&&bytes[parse_index..parse_index + size_of::<u32>()]);
            parse_index += size_of::<u32>();
            let neighbor_count = LittleEndian::read_u32(&&bytes[parse_index..parse_index + size_of::<u32>()]);
            parse_index += size_of::<u32>();

            // Parse the vertex list.
            let mut comm_list = vec![];
            for _ in 0..comm_count {
                // Parse the comm_id and the offset.
                let comm_id = LittleEndian::read_u32(&bytes[parse_index..parse_index + size_of::<u32>()]);
                parse_index += size_of::<u32>();
                let offset = LittleEndian::read_u32(&bytes[parse_index..parse_index + size_of::<u32>()]);
                parse_index += size_of::<u32>();
                // Collect them, 'comm_id' and 'offset' are moved.
                comm_list.push((comm_id, offset));
            }

            // Parse the neighbor community list.
            let mut neighbor_comm_list = vec![];
            loop {
                if parse_index >= TREE_META_SIZE + (2 + (comm_count + neighbor_count) * 2) as usize
                    * size_of::<u32>() {
                    break;
                }
                let neighbor_comm_id = LittleEndian::read_u32(&bytes[parse_index..parse_index + size_of::<u32>()]);
                parse_index += size_of::<u32>();
                let offset = LittleEndian::read_u32(&&bytes[parse_index..parse_index + size_of::<u32>()]);
                parse_index += size_of::<u32>();
                neighbor_comm_list.push(
                    (neighbor_comm_id, offset)
                );
            }

            // Parse the bridge list.
            let mut bridge_list = vec![];
            loop {
                if parse_index >= bytes.len() {
                    break;
                }
                let src = LittleEndian::read_u32(&bytes[parse_index..parse_index + size_of::<u32>()]);
                parse_index += size_of::<u32>();
                let dst = LittleEndian::read_u32(&&bytes[parse_index..parse_index + size_of::<u32>()]);
                parse_index += size_of::<u32>();
                bridge_list.push((src, dst));
            }

            // Return the value.
            Some(TreeBlock {
                block_type,
                max_comm_id,
                min_comm_id,
                comm_count,
                neighbor_count,
                comm_list,
                neighbor_comm_list,
                bridge_list,
            })
        }
    }
}

#[allow(dead_code)]
impl TreeBlock {

    /// Create a new tree block with a list of community neighbor.
    pub(crate) fn create(comm_neighbor_list: &Vec<(CommID, CommNeighbors)>) -> TreeBlock {
        // Compute two levels of offsets.
        let mut neighbor_offset = 0u32;
        let mut bridge_offset = 0u32;
        let mut comm_list = vec![];
        let mut neighbor_comm_list = vec![];
        let mut bridge_list = vec![];

        // Walk through all the neighbors.
        for (comm_id, comm_neighbor_map) in comm_neighbor_list {
            comm_list.push((*comm_id, neighbor_offset));
            for (neighbor_comm_id, bridges) in comm_neighbor_map {
                // Process IDs of those neighbors.
                neighbor_comm_list.push((*neighbor_comm_id, bridge_offset));
                neighbor_offset += 1;
                // Process bridges of those neighbors.
                bridge_list.extend_from_slice(bridges);
                bridge_offset += bridges.len() as u32;
            }
        }

        TreeBlock {
            block_type: 2,
            max_comm_id: comm_list.iter().max_by_key(|&(comm_id, _)| comm_id).unwrap().clone().0,
            min_comm_id: comm_list.iter().min_by_key(|&(comm_id, _)| comm_id).unwrap().clone().0,
            comm_count: comm_list.len() as u32,
            neighbor_count: neighbor_comm_list.len() as u32,
            comm_list,
            neighbor_comm_list,
            bridge_list,
        }
    }
}

impl Serialize for TreeBlock {
    fn encode_to_pages(&self) -> Vec<Page> {
        let mut res_pages = vec![];
        let tree_block_bytes = self.encode();
        let page_count = tree_block_bytes.len().div_ceil(PAGE_DATA_CAPACITY);
        let mut place_index = 0usize;
        for page_index in 0..page_count {
            let mut used_size = PAGE_DATA_CAPACITY;
            let mut has_next = 1u8;
            let mut next_pointer = (page_index + 1) * PAGE_DATA_CAPACITY;
            if place_index == page_count - 1 {
                used_size = tree_block_bytes.len() - place_index * PAGE_DATA_CAPACITY;
                has_next = 0;
                next_pointer = tree_block_bytes.len();
            }
            let page = Page {
                used_size,
                block_type: 0,
                block_offset: page_index,
                has_next,
                next_page_num: 0,
                data: tree_block_bytes[page_index * PAGE_DATA_CAPACITY..next_pointer].to_vec(),
            };
            res_pages.push(page);
            place_index += 1;
        }
        res_pages
    }
}

impl Deserialize for TreeBlock {
    fn decode_from_pages(page_list: &Vec<Page>) -> Option<Self> {
        let mut collected_bytes = vec![];
        for page in page_list {
            collected_bytes.extend_from_slice(&page.data);
        }
        TreeBlock::from_bytes(&collected_bytes)
    }
}


#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct KVCommBlock {
    pub(crate) block_type: u32,  // Type of blocks, 0 represents CSR, 1 represents KV.
    pub(crate) max_vertex_id: u32,  // Maximum vertex id in this community block.
    pub(crate) min_vertex_id: u32,  // Minimum vertex id in this community block.
    pub(crate) unsorted_entry_list: Vec<GraphEntry<V32>>  // Graph entry list.
}

/// A tiny CSR block for exp.
#[derive(Debug, Clone, Default)]
pub struct TinyCSRBlock {
    pub vertex_count: u32,  // Number of vertices stored here.
    pub vertex_list: Vec<(VInt,u32)>,
    pub edge_list: Vec<VInt>
}

impl Encode for TinyCSRBlock {
    /// Encode a Tiny CSR block into byte stream.
    fn encode(&self) -> Vec<u8> {
        // Encode a community block to byte slice.
        let mut encode_bytes = Vec::<u8>::new();

        // Encode the vertex count.
        encode_bytes.extend_from_slice(&self.vertex_count.to_le_bytes());

        // Encode the vertex list and their offsets.
        for (vertex, offset) in &self.vertex_list {
            // The vertex information.
            encode_bytes.extend_from_slice(&vertex.to_le_bytes());
            // The offset part.
            encode_bytes.extend_from_slice(&offset.to_le_bytes());
        }

        // Encode the edge_list part.
        for neighbor in &self.edge_list {
            // Encode the neighbors.
            encode_bytes.extend_from_slice(&neighbor.to_le_bytes());
        }

        // Return value.
        encode_bytes
    }
}

impl Decode for TinyCSRBlock {
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        // Check whether the bytes are valid.
        if bytes.len() < 4 {
            None
        } else {
            // Parse the meta data part.
            let mut parse_index = 0usize;

            let vertex_count = LittleEndian::read_u32(&bytes[parse_index..parse_index + size_of::<u32>()]);
            parse_index += size_of::<u32>();

            // Parse the vertex list.
            let mut vertex_list = vec![];
            for _ in 0..vertex_count {
                // Parse the vertex and the offset.
                let vertex = LittleEndian::read_u32(&bytes[parse_index..parse_index + size_of::<u32>()]);
                parse_index += size_of::<u32>();
                let offset = LittleEndian::read_u32(&bytes[parse_index..parse_index + size_of::<u32>()]);
                parse_index += size_of::<u32>();
                // Collect them, 'vertex' and 'offset' are moved.
                vertex_list.push((vertex, offset));
            }

            // Parse the edge list.
            let mut edge_list = vec![];
            loop {
                if parse_index >= bytes.len() {
                    break;
                }
                edge_list.push(
                    LittleEndian::read_u32(&bytes[parse_index..parse_index + size_of::<u32>()])
                );
                parse_index += size_of::<u32>();
            }

            // Return the value.
            Some(TinyCSRBlock {
                vertex_count,
                vertex_list,
                edge_list
            })
        }
    }
}

impl Serialize for TinyCSRBlock {
    fn encode_to_pages(&self) -> Vec<Page> {
        let mut res_pages = vec![];
        let csr_block_bytes = self.encode();
        let page_count = csr_block_bytes.len().div_ceil(PAGE_DATA_CAPACITY);
        let mut place_index = 0usize;
        for page_index in 0..page_count {
            let mut used_size = PAGE_DATA_CAPACITY;
            let mut has_next = 1u8;
            let mut next_pointer = (page_index + 1) * PAGE_DATA_CAPACITY;
            if place_index == page_count - 1 {
                used_size = csr_block_bytes.len() - place_index * PAGE_DATA_CAPACITY;
                has_next = 0;
                next_pointer = csr_block_bytes.len();
            }
            let page = Page {
                used_size,
                block_type: 0,
                block_offset: page_index,
                has_next,
                next_page_num: 0,
                data: csr_block_bytes[page_index * PAGE_DATA_CAPACITY..next_pointer].to_vec(),
            };
            res_pages.push(page);
            place_index += 1;
        }
        res_pages
    }
}

impl Deserialize for TinyCSRBlock {
    fn decode_from_pages(page_list: &Vec<Page>) -> Option<Self> {
        let mut collected_bytes = vec![];
        for page in page_list {
            collected_bytes.extend_from_slice(&page.data);
        }
        TinyCSRBlock::from_bytes(&collected_bytes)
    }
}

impl TinyCSRBlock {
    pub fn get_neighbors(&self, vertex_id: &VInt) -> Option<Vec<VInt>> {
        for (v_index, (vertex, offset)) in self.vertex_list.iter().enumerate() {
            if *vertex == *vertex_id {
                let mut neighbors = vec![];
                if v_index == (self.vertex_count - 1) as usize {
                    // The last one, fetch from offset to the end.
                    for e_id in *offset as usize..self.edge_list.len() {
                        neighbors.push(self.edge_list[e_id]);
                    }
                } else {
                    // Not the last one, fetch from offset to next offset.
                    for e_id in *offset..self.vertex_list[v_index + 1].1 {
                        neighbors.push(self.edge_list[e_id as usize]);
                    }
                }
                return Some(neighbors);
            }
        }
        None
    }

    pub fn build_from_map(neighbors_map: &BTreeMap<VInt, Vec<VInt>>) -> Self {
        // Generate the vertex list and edge list.
        let mut vertex_list = vec![];
        let mut edge_list = vec![];
        let mut offset = 0u32;
        for (vertex, neighbors) in neighbors_map {
            vertex_list.push(
                (*vertex, offset)
            );
            offset += neighbors.len() as u32;
            edge_list.extend_from_slice(neighbors);
        }

        TinyCSRBlock {
            vertex_count: vertex_list.len() as u32,
            vertex_list,
            edge_list,
        }
    }

    pub fn build_from_dash_map(neighbors_map: &DashMap<VInt, Vec<VInt>>) -> Self {
        // Generate the vertex list and edge list.
        let mut vertex_list = vec![];
        let mut edge_list = vec![];
        let mut offset = 0u32;
        for vertex_ent in neighbors_map.iter() {
            let vertex = vertex_ent.key();
            let neighbors = vertex_ent.value();
            vertex_list.push(
                (*vertex, offset)
            );
            offset += neighbors.len() as u32;
            edge_list.extend_from_slice(neighbors);
        }

        TinyCSRBlock {
            vertex_count: vertex_list.len() as u32,
            vertex_list,
            edge_list,
        }
    }

    pub fn display(&self) {
        println!("=====CSR======");
        println!("Vertex: {:?}",
                 self.vertex_list);
        println!("Edge: {:?}", self.edge_list);
    }

    pub fn generate_map(&self) -> BTreeMap<VInt, Vec<VInt>> {
        let mut res = BTreeMap::<VInt, Vec<VInt>>::new();
        for (id, (vertex_id, offset)) in self.vertex_list.iter().enumerate() {
            let offset_end = if id == self.vertex_list.len() - 1 {
                self.edge_list.len()
            } else {
                self.vertex_list[id + 1].1 as usize
            };
            res.insert(*vertex_id, self.edge_list[*offset as usize..offset_end].to_vec());
        }
        res
    }

    pub fn to_map(self) -> BTreeMap<VInt, Vec<VInt>> {
        let mut res = BTreeMap::<VInt, Vec<VInt>>::new();
        for (id, (vertex_id, offset)) in self.vertex_list.iter().enumerate() {
            let offset_end = if id == self.vertex_list.len() - 1 {
                self.edge_list.len()
            } else {
                self.vertex_list[id + 1].1 as usize
            };
            res.insert(*vertex_id, self.edge_list[*offset as usize..offset_end].to_vec());
        }
        res
    }

    pub fn get_exact_neighbors(&self, vertex_id: &VInt) -> Option<Vec<VInt>> {
        let offset_end = if *vertex_id as usize == self.vertex_list.len() - 1 {
            self.edge_list.len()
        } else {
            self.vertex_list[(*vertex_id + 1) as usize].1 as usize
        };
        Some(self.edge_list[self.vertex_list[*vertex_id as usize].1 as usize..offset_end].to_vec())
    }
}



/// The struct CSR Block, to store the sorted part of some vertices.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct CSRCommBlock {
    pub(crate) block_type: u32,  // Type of blocks, 0 represents CSR, 1 represents KV.
    pub(crate) max_vertex_id: u32,  // Maximum vertex id in this community block.
    pub(crate) min_vertex_id: u32,  // Minimum vertex id in this community block.
    pub(crate) vertex_count: u32,
    pub(crate) vertex_list: Vec<(V32, u32)>, // Store the vertex information and their offsets.
    pub(crate) edge_list: Vec<V32> // Store the edge list, indexed by the offsets of the vertex list.
}

impl Serialize for CommBlock {
    fn encode_to_pages(&self) -> Vec<Page> {
        match self {
            CommBlock::CSR(csr_block) => {
                csr_block.encode_to_pages()
            }
            CommBlock::KV(kv_block) => {
                kv_block.encode_to_pages()
            }
        }
    }
}

impl Deserialize for CommBlock {
    fn decode_from_pages(page_list: &Vec<Page>) -> Option<Self> {
        let mut collected_bytes = vec![];
        for page in page_list {
            collected_bytes.extend_from_slice(&page.data);
        }
        CommBlock::from_bytes(&collected_bytes)
    }
}

impl Query for CommBlock {
    fn get_neighbors(&self, vertex_id: &VInt) -> Option<Vec<V32>> {
        match self {
            CommBlock::CSR(csr_block) => {
                csr_block.get_neighbors(vertex_id)
            }
            CommBlock::KV(unsorted_block) => {
                unsorted_block.get_neighbors(vertex_id)
            }
        }
    }

    fn generate_entries(&self) -> Vec<GraphEntry<V32>> {
        match self {
            CommBlock::CSR(csr_block) => {
                csr_block.generate_entries()
            }
            CommBlock::KV(kv_block) => {
                kv_block.generate_entries()
            }
        }
    }
}

impl Query for CSRCommBlock {
    fn get_neighbors(&self, vertex_id: &VInt) -> Option<Vec<V32>> {
        for (v_index, (vertex, offset)) in self.vertex_list.iter().enumerate() {
            if vertex.vertex_id == *vertex_id {
                let mut neighbors = vec![];
                if v_index == (self.vertex_count - 1) as usize {
                    // The last one, fetch from offset to the end.
                    for e_id in *offset as usize..self.edge_list.len() {
                        neighbors.push(self.edge_list[e_id]);
                    }
                } else {
                    // Not the last one, fetch from offset to next offset.
                    for e_id in *offset..self.vertex_list[v_index + 1].1 {
                        neighbors.push(self.edge_list[e_id as usize]);
                    }
                }
                return Some(neighbors);
            }
        }
        None
    }

    fn generate_entries(&self) -> Vec<GraphEntry<V32>> {
        let mut res_entries = vec![];
        for (v_index, (vertex, offset)) in self.vertex_list.iter().enumerate() {
            let mut neighbors = vec![];
            if v_index == (self.vertex_count - 1) as usize {
                // The last one, fetch from offset to the end.
                for e_id in *offset as usize..self.edge_list.len() {
                    neighbors.push(self.edge_list[e_id]);
                }
            } else {
                // Not the last one, fetch from offset to next offset.
                for e_id in *offset..self.vertex_list[v_index + 1].1 {
                    neighbors.push(self.edge_list[e_id as usize]);
                }
            }
            // Generate graph entries.
            res_entries.push(
                GraphEntry::create(vertex.clone(), neighbors)
            );
        }

        res_entries
    }
}

impl Query for KVCommBlock {
    fn get_neighbors(&self, vertex_id: &VInt) -> Option<Vec<V32>> {
        let mut v_exist = false;
        let mut neighbors = vec![];
        for g_entry in self.unsorted_entry_list.iter() {
            if g_entry.key.vertex_id == *vertex_id {
                // Find it.
                v_exist = true;
                neighbors.extend_from_slice(&g_entry.neighbors);
            }
        }
        if v_exist {
            Some(neighbors)
        } else {
            None
        }
    }

    fn generate_entries(&self) -> Vec<GraphEntry<V32>> {
        let mut res = vec![];
        for entry in &self.unsorted_entry_list {
            res.push(entry.clone())
        }
        res
    }
}

/// The encode trait of the CSR block.
impl Encode for CSRCommBlock {

    /// Encode a CSR block into byte stream.
    fn encode(&self) -> Vec<u8> {
        // Encode a community block to byte slice.
        let mut encode_bytes = Vec::<u8>::new();

        // Encode the meta data part.
        encode_bytes.extend_from_slice(&self.block_type.to_le_bytes());
        encode_bytes.extend_from_slice(&self.max_vertex_id.to_le_bytes());
        encode_bytes.extend_from_slice(&self.min_vertex_id.to_le_bytes());

        // Encode the vertex count.
        encode_bytes.extend_from_slice(&self.vertex_count.to_le_bytes());

        // Encode the vertex list and their offsets.
        for (vertex, offset) in &self.vertex_list {
            // The vertex information.
            encode_bytes.extend_from_slice(&vertex.encode());
            // The offset part.
            encode_bytes.extend_from_slice(&offset.to_le_bytes());
        }

        // Encode the edge_list part.
        for neighbor in &self.edge_list {
            // Encode the neighbors.
            encode_bytes.extend_from_slice(&neighbor.encode());
        }

        // Return value.
        encode_bytes
    }
}

/// The Decode trait of the CSR block.
impl Decode for CSRCommBlock {
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        // Check whether the bytes are valid.
        if bytes.len() < CSR_META_SIZE {
            None
        } else {
            // Parse the meta data part.
            let mut parse_index = 0usize;
            let block_type = LittleEndian::read_u32(&bytes[parse_index..parse_index + size_of::<u32>()]);
            parse_index += size_of::<u32>();
            let max_vertex_id = LittleEndian::read_u32(&bytes[parse_index..parse_index + size_of::<u32>()]);
            parse_index += size_of::<u32>();
            let min_vertex_id = LittleEndian::read_u32(&bytes[parse_index..parse_index + size_of::<u32>()]);
            parse_index += size_of::<u32>();
            let vertex_count = LittleEndian::read_u32(&&bytes[parse_index..parse_index + size_of::<u32>()]);
            parse_index += size_of::<u32>();

            // Parse the vertex list.
            let mut vertex_list = vec![];
            for _ in 0..vertex_count {
                // Parse the vertex and the offset.
                let vertex = V32::from_bytes(&bytes[parse_index..parse_index + V32_SIZE]).unwrap();
                parse_index += V32_SIZE;
                let offset = LittleEndian::read_u32(&bytes[parse_index..parse_index + size_of::<u32>()]);
                parse_index += size_of::<u32>();
                // Collect them, 'vertex' and 'offset' are moved.
                vertex_list.push((vertex, offset));
            }

            // Parse the edge list.
            let mut edge_list = vec![];
            loop {
                if parse_index >= bytes.len() {
                    break;
                }
                edge_list.push(
                    V32::from_bytes(&bytes[parse_index..parse_index + V32_SIZE]).unwrap()
                );
                parse_index += V32_SIZE;
            }

            // Return the value.
            Some(CSRCommBlock {
                block_type,
                max_vertex_id,
                min_vertex_id,
                vertex_count,
                vertex_list,
                edge_list
            })
        }
    }
}

#[allow(dead_code)]
impl Serialize for CSRCommBlock {
    fn encode_to_pages(&self) -> Vec<Page> {
        let mut res_pages = vec![];
        let csr_block_bytes = self.encode();
        let page_count = csr_block_bytes.len().div_ceil(PAGE_DATA_CAPACITY);
        let mut place_index = 0usize;
        for page_index in 0..page_count {
            let mut used_size = PAGE_DATA_CAPACITY;
            let mut has_next = 1u8;
            let mut next_pointer = (page_index + 1) * PAGE_DATA_CAPACITY;
            if place_index == page_count - 1 {
                used_size = csr_block_bytes.len() - place_index * PAGE_DATA_CAPACITY;
                has_next = 0;
                next_pointer = csr_block_bytes.len();
            }
            let page = Page {
                used_size,
                block_type: 0,
                block_offset: page_index,
                has_next,
                next_page_num: 0,
                data: csr_block_bytes[page_index * PAGE_DATA_CAPACITY..next_pointer].to_vec(),
            };
            res_pages.push(page);
            place_index += 1;
        }
        res_pages
    }
}

impl Deserialize for KVCommBlock {
    fn decode_from_pages(page_list: &Vec<Page>) -> Option<Self> {
        let mut collected_bytes = vec![];
        for page in page_list {
            collected_bytes.extend_from_slice(&page.data);
        }
        KVCommBlock::from_bytes(&collected_bytes)
    }
}

impl Deserialize for CSRCommBlock {
    fn decode_from_pages(page_list: &Vec<Page>) -> Option<Self> {
        let mut collected_bytes = vec![];
        for page in page_list {
            collected_bytes.extend_from_slice(&page.data);
        }
        CSRCommBlock::from_bytes(&collected_bytes)
    }
}

#[allow(dead_code)]
impl CSRCommBlock {
    pub(crate) fn from_entries(graph_entries: &Vec<GraphEntry<V32>>) -> Self {
        // Create a temporal graph.
        let mut temp_adj_map = BTreeMap::<VInt, (V32, Vec<V32>)>::new();

        for entry in graph_entries {
            temp_adj_map.entry(entry.key.vertex_id)
                .or_insert((entry.key.clone(), vec![]))
                .1.extend_from_slice(&entry.neighbors);
        }

        // Generate the vertex list and edge list.
        let mut vertex_list = vec![];
        let mut edge_list = vec![];
        let mut offset = 0u32;
        for (_, (vertex, neighbors)) in &temp_adj_map {
            vertex_list.push(
                (vertex.clone(), offset)
            );
            offset += neighbors.len() as u32;
            edge_list.extend_from_slice(neighbors);
        }
        let max_vertex_id = if let Some(last) = temp_adj_map.last_key_value() {
            *last.0
        } else {
            0u32
        };
        let min_vertex_id = if let Some(first) = temp_adj_map.first_key_value() {
            *first.0
        } else {
            0u32
        };

        CSRCommBlock {
            block_type: 0,
            max_vertex_id,
            min_vertex_id,
            vertex_count: vertex_list.len() as u32,
            vertex_list,
            edge_list,
        }
    }
}

impl Decode for CommBlock {
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let block_type = LittleEndian::read_u32(&bytes[0..0 + size_of::<u32>()]);
        if block_type == 1 {
            // Parse bytes into kv block.
            let parsed_kv_block = KVCommBlock::from_bytes(bytes).unwrap();
            Some(CommBlock::KV(parsed_kv_block))
        } else {
            // Parse bytes into sorted block.
            let parsed_csr_block = CSRCommBlock::from_bytes(bytes).unwrap();
            Some(CommBlock::CSR(parsed_csr_block))
        }
    }
}

impl Encode for CommBlock {
    fn encode(&self) -> Vec<u8> {
        // Encode a community block to bytes.
        match self {
            CommBlock::CSR(block) => {
                // Parse the CSR block.
                block.encode()
            }
            CommBlock::KV(block) => {
                block.encode()
            }
        }
    }
}

impl Encode for KVCommBlock {
    fn encode(&self) -> Vec<u8> {
        // Encode a community block to byte slice.
        let mut encode_bytes = Vec::<u8>::new();

        // Encode the meta data part.
        encode_bytes.extend_from_slice(&self.block_type.to_le_bytes());
        encode_bytes.extend_from_slice(&self.max_vertex_id.to_le_bytes());
        encode_bytes.extend_from_slice(&self.min_vertex_id.to_le_bytes());

        // Encode the entry list part.
        for g_entry in &self.unsorted_entry_list {
            encode_bytes.extend_from_slice(&g_entry.encode());
        }

        // Return value.
        encode_bytes
    }
}

impl Decode for KVCommBlock {
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        // Check whether the bytes are valid.
        if bytes.len() < KV_META_SIZE {
            None
        } else {
            // Parse the meta data part.
            let mut parse_index = 0usize;
            let block_type = LittleEndian::read_u32(&bytes[parse_index..parse_index + size_of::<u32>()]);
            parse_index += size_of::<u32>();
            let max_vertex_id = LittleEndian::read_u32(&bytes[parse_index..parse_index + size_of::<u32>()]);
            parse_index += size_of::<u32>();
            let min_vertex_id = LittleEndian::read_u32(&bytes[parse_index..parse_index + size_of::<u32>()]);
            parse_index += size_of::<u32>();
            let used_size = bytes.len();

            let mut entry_list = Vec::<GraphEntry<V32>>::new();

            loop {
                if parse_index >= used_size {
                    break;
                }
                // Parse graph entry one by one.
                let entry_size = GraphEntry::determine_size(&bytes[parse_index..]);
                let graph_entry = GraphEntry::from_bytes(&bytes[parse_index..parse_index + entry_size]).unwrap();
                entry_list.push(graph_entry);
                parse_index += entry_size;
            }

            Some(KVCommBlock {
                block_type,
                max_vertex_id,
                min_vertex_id,
                unsorted_entry_list: entry_list
            })
        }
    }
}

impl Serialize for KVCommBlock {
    fn encode_to_pages(&self) -> Vec<Page> {
        let mut res_pages = vec![];
        let kv_block_bytes = self.encode();
        // println!("Block Bytes Length {}", kv_block_bytes.len());
        let page_count = kv_block_bytes.len().div_ceil(PAGE_DATA_CAPACITY);
        let mut place_index = 0usize;
        for page_index in 0..page_count {
            let mut used_size = PAGE_DATA_CAPACITY;
            let mut has_next = 1u8;
            let mut next_pointer = (page_index + 1) * PAGE_DATA_CAPACITY;
            // The last page.
            if place_index == page_count - 1 {
                used_size = kv_block_bytes.len() - place_index * PAGE_DATA_CAPACITY;
                has_next = 0;
                next_pointer = kv_block_bytes.len();
            }
            let page = Page {
                used_size,
                block_type: 1,
                block_offset: page_index,
                has_next,
                next_page_num: 0,
                data: kv_block_bytes[page_index * PAGE_DATA_CAPACITY..next_pointer].to_vec(),
            };
            res_pages.push(page);
            place_index += 1;
        }
        res_pages
    }
}


#[cfg(test)]
pub mod test_comm_block {
    use std::collections::HashMap;

    use crate::bucket::block::{CommBlock, CSRCommBlock, Deserialize, KVCommBlock, Query, Serialize, TreeBlock};
    use crate::bucket::CommBucket;
    use crate::comm_table::CommID;
    use crate::community::CommNeighbors;
    use crate::graph::Graph;
    use crate::types::{Decode, Encode, GraphEntry, V32, Vertex};

    #[test]
    fn test_csr_block_encode_decode() {
        // Test the Encoding and Decoding of the Sorted Community Block.
        // Prepare the Graph Entries.
        let u1 = Vertex::new(23u32, 0u8);
        let v1 = Vertex::new(24u32, 1u8);
        let v2 = Vertex::new(25u32, 1u8);
        let v3 = Vertex::new(26u32, 1u8);
        let entry1 = GraphEntry::<V32>::create(u1, vec![v1, v2, v3]);

        let u2 = Vertex::new(26u32, 0u8);
        let v4 = Vertex::new(24u32, 1u8);
        let v5 = Vertex::new(25u32, 1u8);
        let v6 = Vertex::new(26u32, 1u8);
        let entry2 = GraphEntry::<V32>::create(u2, vec![v4, v5, v6]);

        let mut entry_list = Vec::<GraphEntry<V32>>::new();
        entry_list.push(entry1);
        entry_list.push(entry2);

        let csr_block = CSRCommBlock::from_entries(&entry_list);

        let comm_block_bytes = csr_block.encode();

        let comm_block_load = CommBlock::from_bytes(&comm_block_bytes).unwrap();
        match comm_block_load {
            CommBlock::CSR(csr_comm_block) => {
                assert_eq!(csr_block.block_type, csr_comm_block.block_type);
                assert_eq!(csr_block.min_vertex_id, csr_comm_block.min_vertex_id);
                assert_eq!(csr_block.max_vertex_id, csr_comm_block.max_vertex_id);
                assert_eq!(csr_block.vertex_count, csr_comm_block.vertex_count);
            }
            CommBlock::KV(_) => {
                println!("An error happens.");
            }
        }
    }

    /// Test the encoding of KV blocks to pages.
    #[test]
    fn test_kv_block_enc_dec_pages() {
        // Test the Encoding and Decoding of the KV Community Block.
        // Prepare the Graph Entries.
        let u1 = Vertex::new(23u32, 0u8);
        let v1 = Vertex::new(24u32, 1u8);
        let v2 = Vertex::new(25u32, 1u8);
        let v3 = Vertex::new(26u32, 1u8);
        let entry1 = GraphEntry::<V32>::create(u1, vec![v1, v2, v3]);

        let u2 = Vertex::new(26u32, 0u8);
        let v4 = Vertex::new(24u32, 1u8);
        let v5 = Vertex::new(25u32, 1u8);
        let v6 = Vertex::new(26u32, 1u8);
        let entry2 = GraphEntry::<V32>::create(u2, vec![v4, v5, v6]);
        println!("Entry Size: {}", entry2.get_entry_size());

        let mut entry_list = Vec::<GraphEntry<V32>>::new();
        entry_list.push(entry1.clone());
        entry_list.push(entry2.clone());

        let comm_block = KVCommBlock {
            max_vertex_id: 0,
            min_vertex_id: 0,
            block_type: 1,
            unsorted_entry_list: entry_list
        };

        let u3 = Vertex::new(29u32, 0u8);
        let v7 = Vertex::new(24u32, 1u8);
        let v8 = Vertex::new(25u32, 1u8);
        let v9 = Vertex::new(26u32, 1u8);
        let entry3 = GraphEntry::<V32>::create(u3, vec![v7, v8, v9]);

        let kv_block_pages = comm_block.encode_to_pages();
        let kv_page = kv_block_pages[0].clone();
        println!("Used size: {}", kv_page.used_size);

        let mut comm_block_decode = KVCommBlock::decode_from_pages(&kv_block_pages).unwrap();
        comm_block_decode.unsorted_entry_list.push(entry3);

        // Decode it.
        let kv_block_pages_new = comm_block_decode.encode_to_pages();
        assert_eq!(kv_block_pages_new.len(), 1);
        let kv_page = kv_block_pages_new[0].clone();
        println!("Used size: {}", kv_page.used_size);
    }

    #[test]
    fn test_kv_block_encode_decode() {
        // Test the Encoding and Decoding of the KV Community Block.
        // Prepare the Graph Entries.
        let u1 = Vertex::new(23u32, 0u8);
        let v1 = Vertex::new(24u32, 1u8);
        let v2 = Vertex::new(25u32, 1u8);
        let v3 = Vertex::new(26u32, 1u8);
        let entry1 = GraphEntry::<V32>::create(u1, vec![v1, v2, v3]);

        let u2 = Vertex::new(26u32, 0u8);
        let v4 = Vertex::new(24u32, 1u8);
        let v5 = Vertex::new(25u32, 1u8);
        let v6 = Vertex::new(26u32, 1u8);
        let entry2 = GraphEntry::<V32>::create(u2, vec![v4, v5, v6]);

        let mut entry_list = Vec::<GraphEntry<V32>>::new();
        entry_list.push(entry1);
        entry_list.push(entry2);

        let comm_block = KVCommBlock {
            max_vertex_id: 0,
            min_vertex_id: 0,
            block_type: 1,
            unsorted_entry_list: entry_list
        };

        let comm_block_bytes = comm_block.encode();

        let comm_block_load = CommBlock::from_bytes(&comm_block_bytes).unwrap();
        match comm_block_load {
            CommBlock::CSR(_) => {
                println!("An error happen.");
            }
            CommBlock::KV(kv) => {
                println!("Parsing KV Block.");
                for entry_load in &kv.unsorted_entry_list {
                    println!("{}", entry_load);
                }
            }
        }
    }

    #[test]
    fn test_csr_query() {
        // Test the Querying of the CSR Community Block.
        // Prepare the Graph Entries.
        let u1 = Vertex::new(23u32, 0u8);
        let v1 = Vertex::new(24u32, 1u8);
        let v2 = Vertex::new(25u32, 1u8);
        let v3 = Vertex::new(26u32, 1u8);
        let entry1 = GraphEntry::<V32>::create(u1, vec![v1, v2, v3]);

        let u2 = Vertex::new(26u32, 0u8);
        let v4 = Vertex::new(24u32, 1u8);
        let v5 = Vertex::new(25u32, 1u8);
        let v6 = Vertex::new(26u32, 1u8);
        let entry2 = GraphEntry::<V32>::create(u2, vec![v4, v5, v6]);

        let mut entry_list = Vec::<GraphEntry<V32>>::new();
        entry_list.push(entry1);
        entry_list.push(entry2);

        let csr_block = CSRCommBlock::from_entries(&entry_list);

        match csr_block.get_neighbors(&26u32) {
            None => {}
            Some(result_neighbor) => {
                println!("result neighbors: {:?}", result_neighbor);
            }
        }
    }

    #[test]
    fn test_kv_query() {
        // Test the Querying of the CSR Community Block.
        // Prepare the Graph Entries.
        let u1 = Vertex::new(23u32, 0u8);
        let v1 = Vertex::new(24u32, 1u8);
        let v2 = Vertex::new(25u32, 1u8);
        let v3 = Vertex::new(26u32, 1u8);
        let entry1 = GraphEntry::<V32>::create(u1, vec![v1, v2, v3]);

        let u2 = Vertex::new(26u32, 0u8);
        let v4 = Vertex::new(24u32, 1u8);
        let v5 = Vertex::new(25u32, 1u8);
        let v6 = Vertex::new(27u32, 1u8);
        let entry2 = GraphEntry::<V32>::create(u2, vec![v4, v5, v6]);

        let mut entry_list = Vec::<GraphEntry<V32>>::new();
        entry_list.push(entry1);
        entry_list.push(entry2);

        let comm_block = KVCommBlock {
            max_vertex_id: 0,
            min_vertex_id: 0,
            block_type: 1,
            unsorted_entry_list: entry_list
        };

        match comm_block.get_neighbors(&26u32) {
            None => {}
            Some(result_neighbor) => {
                println!("result neighbors: {:?}", result_neighbor);
            }
        }
    }

    #[test]
    fn test_load() {
        for tested_data_set in vec!["oregon", "yeast", "oregon"] {
            let dataset_path = "data/".to_owned() + tested_data_set + ".graph";
            let obj_path = "lsm.db/".to_owned() + tested_data_set + "_comm_test.txt";
            let g_from_graph = Graph::from_graph_file(&dataset_path, true);
            let bucket = CommBucket::build_bucket_from_community(&g_from_graph,
                                                                 &obj_path);
            let g_load = bucket.load_community();

            // Check whether these two graph equal.
            assert_eq!(g_from_graph.e_size, g_load.e_size / 2);
        }
        println!("Find Neighbor test pass!");
    }

    #[test]
    fn test_tree_encode_decode() {
        let tb = TreeBlock::create(&prepare_example_comm_graph());
        println!("Ground Truth: {:?}", tb);
        let tree_bytes = tb.encode();

        let tb_load = TreeBlock::from_bytes(&tree_bytes).unwrap();
        println!("Load: {:?}", tb_load);
        assert_eq!(tb_load.comm_list.len(), tb.comm_list.len());
        assert_eq!(tb_load.max_comm_id, tb.max_comm_id);
        assert_eq!(tb_load.min_comm_id, tb.min_comm_id);
        assert_eq!(tb_load.comm_count, tb.comm_count);
        assert_eq!(tb_load.neighbor_count, tb.neighbor_count);
        assert_eq!(tb_load.block_type, tb.block_type);
        assert_eq!(tb_load.neighbor_comm_list.len(), tb.neighbor_comm_list.len());
        assert_eq!(tb_load.bridge_list.len(), tb_load.bridge_list.len());

        let pages = tb.encode_to_pages();
        let tb_load = TreeBlock::decode_from_pages(&pages).unwrap();

        assert_eq!(tb_load.comm_list.len(), tb.comm_list.len());
        assert_eq!(tb_load.max_comm_id, tb.max_comm_id);
        assert_eq!(tb_load.min_comm_id, tb.min_comm_id);
        assert_eq!(tb_load.comm_count, tb.comm_count);
        assert_eq!(tb_load.neighbor_count, tb.neighbor_count);
        assert_eq!(tb_load.block_type, tb.block_type);
        assert_eq!(tb_load.neighbor_comm_list.len(), tb.neighbor_comm_list.len());
        assert_eq!(tb_load.bridge_list.len(), tb_load.bridge_list.len());
    }

    /// Generate an example community graph.
    pub fn prepare_example_comm_graph() -> Vec<(CommID, CommNeighbors)> {
        let mut res_example = vec![];
        // Community 0.
        let mut comm_neighbor_0 = HashMap::<CommID, Vec<(u32, u32)>>::new();
        // c_0 -> [c_1((4, 5)), c_2((4, 12)), c_3((4, 8))]
        comm_neighbor_0.insert(1u32, vec![(4, 12)]);
        comm_neighbor_0.insert(2u32, vec![(4, 5)]);
        comm_neighbor_0.insert(3u32, vec![(4, 8)]);

        // Community 1.
        let mut comm_neighbor_1 = HashMap::<CommID, Vec<(u32, u32)>>::new();
        // c_1 -> [c_0((12, 4))]
        comm_neighbor_1.insert(0u32, vec![(12, 4)]);

        // Community 2.
        let mut comm_neighbor_2 = HashMap::<CommID, Vec<(u32, u32)>>::new();
        // c_2 -> [c_0((5, 4)), c_3(5, 8)]
        comm_neighbor_2.insert(0u32, vec![(5, 4)]);
        comm_neighbor_2.insert(3u32, vec![(5, 8)]);

        // Community 3.
        let mut comm_neighbor_3 = HashMap::<CommID, Vec<(u32, u32)>>::new();
        // c_3 -> [c_0((8, 4)), c_2(8, 5)]
        comm_neighbor_3.insert(0u32, vec![(8, 4)]);
        comm_neighbor_3.insert(2u32, vec![(8, 5)]);

        // Collect those community neighbors.
        res_example.push((0u32, comm_neighbor_0));
        res_example.push((1u32, comm_neighbor_1));
        res_example.push((2u32, comm_neighbor_2));
        res_example.push((3u32, comm_neighbor_3));

        res_example
    }
}