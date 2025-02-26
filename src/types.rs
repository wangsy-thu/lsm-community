use std::cmp::Ordering;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::mem::size_of;

use byteorder::{ByteOrder, LittleEndian};

use crate::util::get_current_timestamp;

pub(crate) type V32 = Vertex<u32>;
pub(crate) static V32_SIZE :usize = 14;

// Define a trait to encode something to Bytes.
pub trait Encode {
    // the method encode itself to Bytes.
    fn encode(&self) -> Vec<u8>;
}

pub trait Decode: Sized {
    fn from_bytes(bytes: &[u8]) -> Option<Self>;
}

pub trait PartialDecode: Sized {
    fn determine_size(bytes: &[u8]) -> usize;  // Determine the space cost of it.
}

// Define the Vertex struct (10 Bytes each, but 12 Bytes when applying size_of).
#[derive(Debug, Eq, Ord, Clone, Copy)]
pub struct Vertex<T>
where Vertex<T>: PartialOrd {
    pub vertex_id: T, // Vertex ID, unique in this system, usually use u32.
    pub(crate) timestamp: u64, // Timestamp, initialized when created, default current value.
    pub direction_tag: u8,  // Direction tag, 0 represents start vertex,
    // 1 represents successor neighbors, 2 represents predecessor neighbors.
    pub(crate) tomb: u8 // Deletion mark, 0 means exist, 1 means deleted, 2 means forced deleted.
    // Note that 2 is usually used in dynamically community detection.
}

#[allow(dead_code)]
impl Vertex<u32> {
    pub fn new(v_id: u32, direction: u8) -> Self {
        Vertex {
            vertex_id: v_id,
            timestamp: get_current_timestamp(),
            direction_tag: direction,
            tomb: 0u8
        }
    }

    pub fn new_successor(v_id: u32) -> Self {
        Vertex {
            vertex_id: v_id,
            timestamp: get_current_timestamp(),
            direction_tag: 1,
            tomb: 0u8
        }
    }

    pub fn new_predecessor(v_id: u32) -> Self {
        Vertex {
            vertex_id: v_id,
            timestamp: get_current_timestamp(),
            direction_tag: 2,
            tomb: 0u8
        }
    }

    pub fn new_vertex(v_id: u32) -> Self {
        Vertex {
            vertex_id: v_id,
            timestamp: get_current_timestamp(),
            direction_tag: 0,
            tomb: 0u8
        }
    }


    #[allow(dead_code)]
    pub fn new_tomb(v_id: u32, direction: u8) -> Self {
        Vertex {
            vertex_id: v_id,
            timestamp: get_current_timestamp(),
            direction_tag: direction,
            tomb: 1u8
        }
    }

    // Create a vertex with an escaping mark.
    pub fn new_escape(v_id: u32) -> Self {
        Vertex {
            vertex_id: v_id,
            timestamp: get_current_timestamp(),
            direction_tag: 0u8,
            tomb: 2u8
        }
    }
}

impl Hash for Vertex<u32> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vertex_id.hash(state);
        self.direction_tag.hash(state);
    }
}

impl Display for Vertex<u32> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let dir_display_tag = match self.direction_tag {
            0 => ">",
            1 => "+",
            2 => "-",
            _ => "error"
        };
        let delete_display_tag = match self.tomb {
            0u8 => "",
            1u8 => "'",
            2u8 => "''",
            _ => {panic!("Unresolved deletion tag.")}
        };
        Ok(write!(f, "({})V{}{}", dir_display_tag, self.vertex_id, delete_display_tag).expect("Display error"))
    }
}

impl PartialEq for Vertex<u32> {
    fn eq(&self, other: &Self) -> bool {
         self.vertex_id == other.vertex_id && self.direction_tag == other.direction_tag
    }
}

impl PartialOrd for Vertex<u32> {
    fn partial_cmp(&self, other: &Vertex<u32>) -> Option<Ordering> {
        if self.vertex_id == other.vertex_id {
            if self.direction_tag == other.direction_tag {
                Some(self.timestamp.cmp(&other.timestamp))
            } else {
                Some(self.direction_tag.cmp(&other.direction_tag))
            }
        } else {
            Some(self.vertex_id.cmp(&other.vertex_id))
        }
    }
}

impl Encode for Vertex<u32> {
    fn encode(&self) -> Vec<u8> {
        let mut le_bytes = self.vertex_id.to_le_bytes().to_vec();
        let timestamp_bytes = self.timestamp.to_le_bytes().to_vec();
        le_bytes.extend_from_slice(&timestamp_bytes);
        le_bytes.push(self.direction_tag);
        le_bytes.push(self.tomb);
        le_bytes
    }
}

impl Decode for Vertex<u32> {
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() == V32_SIZE {
            let mut byte_array = [0u8; 4];
            let mut time_array = [0u8; 8];
            byte_array.copy_from_slice(&bytes[0..4]);
            time_array.copy_from_slice(&bytes[4..12]);
            let v_id = u32::from_le_bytes(byte_array); // Decode the vertex id.
            let timestamp = u64::from_le_bytes(time_array);
            let direction_flag = bytes[12];
            let tomb_flag = match bytes[13] {
                0 => {0u8},
                1 => {1u8},
                2 => {2u8},
                _ => return None,
            }; // Decode the timestamp.
            Some(Vertex{
                vertex_id: v_id,
                timestamp,
                direction_tag: direction_flag,
                tomb: tomb_flag
            })
        } else {
            None
        }
    }
}

// The graph entry is used to store a vertex and its neighbors.
#[derive(Eq, Debug, Clone, Hash)]
pub struct GraphEntry<V: Encode + Decode>
where GraphEntry<V>: PartialOrd {
    pub(crate) key: V, // Vertex Type
    pub(crate) partition_id: u16,  // Partition ID, used for processing huge degree vertices.
    pub(crate) neighbor_size: u16,  // Neighbor Count.
    pub(crate) neighbors: Vec<V>  // Neighbors.
}

impl<V: Encode + Decode> Encode for GraphEntry<V>
where GraphEntry<V>: PartialOrd {
    fn encode(&self) -> Vec<u8> {
        let mut le_bytes = self.key.encode().to_vec();
        let part_bytes_vec = self.partition_id.to_le_bytes();
        let ns_bytes_vec = self.neighbor_size.to_le_bytes();
        le_bytes.extend_from_slice(&part_bytes_vec);
        le_bytes.extend_from_slice(&ns_bytes_vec);
        for neighbor in &self.neighbors {
            le_bytes.extend_from_slice(&neighbor.encode())
        }
        le_bytes
    }
}

impl Display for GraphEntry<V32> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} (size: {}) -> ", self.key, self.neighbor_size).expect("Display error");
        for v in &self.neighbors {
            write!(f, "{} -> ", v).expect("Display error");
        }
        Ok(write!(f, " End").expect("Display error"))
    }
}

impl<V: Encode + Decode> Decode for GraphEntry<V>
where GraphEntry<V>: PartialOrd {
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() >= V32_SIZE + 4 {
            // Step 1. Decode vertex.
            let mut decode_idx: usize = 0;
            let key = V::from_bytes(&bytes[0..V32_SIZE]).unwrap();

            let mut part_byte_array = [0u8; 2];
            part_byte_array.copy_from_slice(&bytes[V32_SIZE..V32_SIZE + 2]);
            let partition_id = u16::from_le_bytes(part_byte_array);

            let mut neighbor_size_byte_array = [0u8; 2];
            neighbor_size_byte_array.copy_from_slice(&bytes[V32_SIZE + 2..V32_SIZE + 4]);
            let neighbor_size = u16::from_le_bytes(neighbor_size_byte_array);
            decode_idx += V32_SIZE + 4;
            let mut neighbors = Vec::new();
            loop {
                if decode_idx >= bytes.len() {
                    break;
                }
                let neighbor =
                    V::from_bytes(&bytes[decode_idx..decode_idx + V32_SIZE]);
                neighbors.push(neighbor.unwrap());
                decode_idx += V32_SIZE;
            }
            Some(GraphEntry {
                key,
                partition_id,
                neighbor_size,
                neighbors
            })
        } else {
            None
        }
    }
}

impl<V: Encode + Decode> PartialDecode for GraphEntry<V>
where GraphEntry<V>: PartialOrd {
    fn determine_size(bytes: &[u8]) -> usize {
        let entry_count = LittleEndian::read_u16(&bytes[V32_SIZE + 2..V32_SIZE + 4]) as usize;
        (entry_count + 1) * V32_SIZE + 4
    }
}

impl Ord for GraphEntry<V32> {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.key.vertex_id == other.key.vertex_id {
            if self.key.timestamp == other.key.timestamp {
                self.partition_id.cmp(&other.partition_id)
            } else {
                self.key.timestamp.cmp(&other.key.timestamp)
            }
        } else {
            self.key.vertex_id.cmp(&other.key.vertex_id)
        }
    }
}

impl PartialEq<Self> for GraphEntry<V32> {
    fn eq(&self, other: &Self) -> bool {
        self.key.vertex_id == other.key.vertex_id &&
            self.key.timestamp == other.key.timestamp &&
            self.key.tomb == other.key.tomb &&
            self.partition_id == other.partition_id
    }
}

impl PartialOrd for GraphEntry<V32> {
    fn partial_cmp(&self, other: &GraphEntry<V32>) -> Option<Ordering> {
        if self.key.vertex_id == other.key.vertex_id {
            if self.key.timestamp == other.key.timestamp {
                Some(self.partition_id.cmp(&other.partition_id))
            } else {
                Some(self.key.timestamp.cmp(&other.key.timestamp))
            }
        } else {
            Some(self.key.vertex_id.cmp(&other.key.vertex_id))
        }
    }
}

#[allow(dead_code)]
impl GraphEntry<V32> {

    /// Compute the number of bytes it needs in a graph entry format.
    pub(crate) fn get_entry_size(&self) -> usize {
        (1 + self.neighbor_size) as usize * V32_SIZE + 4
    }

    /// Compute the number of bytes it needs in a csr format.
    pub(crate) fn get_csr_size(&self) -> usize {
        (self.neighbor_size + 1) as usize * V32_SIZE + size_of::<u32>()
    }

    pub(crate) fn create(vertex: V32, neighbors: Vec<V32>) -> GraphEntry<V32> {
        // Create a new graph entry, attention that the neighbors are moved.
        GraphEntry {
            key: vertex,
            partition_id: 0,
            neighbor_size: neighbors.len() as u16,
            neighbors,
        }
    }

    pub(crate) fn create_with_partition(vertex: V32, neighbors: Vec<V32>, partition_id: u16) -> GraphEntry<V32> {
        // Create a new graph entry, attention that the neighbors are moved.
        GraphEntry {
            key: vertex,
            partition_id,
            neighbor_size: neighbors.len() as u16,
            neighbors,
        }
    }
}

#[cfg(test)]
mod test_transform {
    use crate::types::{Decode, Encode, GraphEntry, V32, Vertex};

    #[test]
    fn test_vertex_encode_decode() {
        let u = Vertex::new(23u32, 0u8);
        let u_bytes = u.encode();
        println!("V32 size: {}", u_bytes.len());
        let v = match Vertex::from_bytes(&u_bytes) {
            None => {return;}
            Some(v) => {v}
        };
        assert!(v.vertex_id == u.vertex_id && v.tomb == u.tomb);
    }

    #[test]
    fn test_entry_decode_encode() {
        let u = Vertex::new(23u32, 0u8);
        let v1 = Vertex::new(24u32, 0u8);
        let v2 = Vertex::new(25u32, 0u8);
        let v3 = Vertex::new(26u32, 0u8);
        let entry = GraphEntry::<V32>::create(u, vec![v1, v2, v3]);
        let entry_bytes = entry.encode();
        let _after_decode: GraphEntry<Vertex<u32>> = GraphEntry::from_bytes(&entry_bytes).unwrap();
        assert!(
            entry.key.vertex_id == _after_decode.key.vertex_id
            && entry.neighbor_size == _after_decode.neighbor_size
        );
        for i in 0..3 {
            assert_eq!(entry.neighbors[i].vertex_id, _after_decode.neighbors[i].vertex_id);
        }
    }
}