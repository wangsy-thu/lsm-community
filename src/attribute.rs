use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use sled::{Db, Tree};

use crate::attribute::AttrKey::{EdgeKey, VertexKey};
use crate::attribute::AttrValue::{Num, Str};
use crate::sgm_type::VInt;
use crate::types::{Decode, Encode};

/// The key of each vertex attribute.
#[derive(Clone)]
pub struct VertexAttrKey {
    pub key_type: u8,  // Type of this key, '0' represents vertex, '1' represents edge.
    pub vertex_id: VInt,  // The vertex id of the stored attribute.
    pub key_name: String,  // The name of the vertex attribute.
}

impl Encode for VertexAttrKey {
    fn encode(&self) -> Vec<u8> {
        // Encode a vertex attribute to byte stream.
        let mut encode_bytes = Vec::<u8>::new();

        // Encode the meta data part.
        encode_bytes.push(self.key_type);
        encode_bytes.extend_from_slice(&self.vertex_id.to_le_bytes());
        encode_bytes.append(&mut self.key_name.clone().into_bytes());

        encode_bytes
    }
}

impl Decode for VertexAttrKey {
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() >= 5 {
            let key_type = bytes[0].clone();
            let mut vid_byte_array = [0u8; 4];
            vid_byte_array.copy_from_slice(&bytes[1..5]);
            let vertex_id = u32::from_le_bytes(vid_byte_array); // Decode the vertex id.
            let str_byte_array = &bytes[5..];
            let key_name_res = String::from_utf8(str_byte_array.to_vec());
            match key_name_res {
                Ok(key_name) => {
                    Some(VertexAttrKey{
                        key_type,
                        vertex_id,
                        key_name,
                    })
                }
                Err(_) => {
                    None
                }
            }
        } else {
            None
        }
    }
}

/// The key of each edge attribute.
#[derive(Clone)]
pub struct EdgeAttrKey {
    pub key_type: u8,  // Type of this key, '0' represents vertex, '1' represents edge.
    pub src_vertex_id: VInt,  // The source vertex id of the stored attribute.
    pub dst_vertex_id: VInt,  // The destination vertex id of the stored attribute.
    pub key_name: String,  // The name of the vertex attribute.
}

impl Decode for EdgeAttrKey {
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() >= 9 {
            let key_type = bytes[0].clone();
            let mut src_vid_byte_array = [0u8; 4];
            src_vid_byte_array.copy_from_slice(&bytes[1..5]);
            let src_vertex_id = u32::from_le_bytes(src_vid_byte_array); // Decode the vertex id.

            let mut dst_vid_byte_array = [0u8; 4];
            dst_vid_byte_array.copy_from_slice(&bytes[5..9]);
            let dst_vertex_id = u32::from_le_bytes(dst_vid_byte_array); // Decode the vertex id.

            let str_byte_array = &bytes[9..];
            let key_name_res = String::from_utf8(str_byte_array.to_vec());

            match key_name_res {
                Ok(key_name) => {
                    Some(EdgeAttrKey{
                        key_type,
                        src_vertex_id,
                        dst_vertex_id,
                        key_name,
                    })
                }
                Err(_) => {
                    None
                }
            }
        } else {
            None
        }
    }
}

impl Encode for EdgeAttrKey {
    fn encode(&self) -> Vec<u8> {
        // Encode an edge attribute to byte stream.
        let mut encode_bytes = Vec::<u8>::new();

        // Encode the meta data part.
        encode_bytes.push(self.key_type);
        encode_bytes.extend_from_slice(&self.src_vertex_id.to_le_bytes());
        encode_bytes.extend_from_slice(&self.dst_vertex_id.to_le_bytes());
        encode_bytes.append(&mut self.key_name.clone().into_bytes());

        encode_bytes
    }
}

/// The combination of the vertex key and the edge key.
#[derive(Clone)]
pub enum AttrKey {
    VertexKey(VertexAttrKey),
    EdgeKey(EdgeAttrKey)
}

impl Encode for AttrKey {
    fn encode(&self) -> Vec<u8> {
        match self {
            VertexKey(vertex_attr_key) => {
                vertex_attr_key.encode()
            }
            EdgeKey(edge_attr_key) => {
                edge_attr_key.encode()
            }
        }
    }
}

impl Decode for AttrKey {
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        // Check the type of the decoded bytes.
        match bytes[0] {
            0u8 => {
                // Vertex attributes.
                let vertex_attr_key_opt = VertexAttrKey::from_bytes(bytes);
                match vertex_attr_key_opt {
                    None => {None}
                    Some(vertex_attr_key) => {
                        Some(VertexKey(vertex_attr_key))
                    }
                }
            },
            1u8 => {
                let edge_attr_key_opt = EdgeAttrKey::from_bytes(bytes);
                match edge_attr_key_opt {
                    None => {None}
                    Some(edge_attr_key) => {
                        Some(EdgeKey(edge_attr_key))
                    }
                }
            },
            _ => {
                // Other types.
                None
            }
        }
    }
}

/// The stored value of the attributes.
pub enum AttrValue {
    Num(u32),
    Str(String)
}

impl Encode for AttrValue {
    fn encode(&self) -> Vec<u8> {
        match self {
            Num(value) => {
                // Encode the type label, '0' represents the number.
                let mut encode_bytes = value.to_le_bytes().to_vec();
                encode_bytes.insert(0, 0u8);
                encode_bytes
            }
            Str(value) => {
                // Encode the type label, '1' represents the string.
                let mut encode_bytes = value.clone().into_bytes().to_vec();
                encode_bytes.insert(0, 1u8);
                encode_bytes
            }
        }
    }
}


impl Decode for AttrValue {
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        match bytes[0] {
            0u8 => {
                // Number type.
                let mut byte_array = [0u8; 4];
                byte_array.copy_from_slice(&bytes[1..5]);
                Some(Num(u32::from_le_bytes(byte_array)))
            },
            1u8 => {
                // String type.
                let str_byte_array = &bytes[1..];
                let value_res =
                    String::from_utf8(str_byte_array.to_vec());
                match value_res {
                    Ok(value_str) => { Some(Str(value_str)) }
                    Err(_) => { None }
                }
            },
            _ => {
                // Other types.
                None
            }
        }
    }
}

/// The configuration of the attribute storage engine.
pub struct AttrStorageConfig {
    pub cache_capacity: u64,
    pub flush_frequency: u64,
}

/// The Attribute storage engine of LSM-Community.
#[allow(dead_code)]
pub struct AttrStorage {
    dataset_name: String, // The name of the stored dataset.
    attr_storage_config: AttrStorageConfig,  // Configuration of the attribute storage.
    tree_storage: Tree,  // The LSM-Tree used for storing the attributes.
    db: Db, // The database instance of the LSM-Tree.
}

#[allow(dead_code)]
impl AttrStorage {
    /// Construction function of AttrStorage.
    pub fn create(
        dataset_name: String,
        attr_storage_config: AttrStorageConfig
    ) -> Self {
        let config = sled::Config::new()
            .path(format!("lsm.db/attr_{}.db", dataset_name))
            .cache_capacity(attr_storage_config.cache_capacity)
            .flush_every_ms(Some(attr_storage_config.flush_frequency));

        let db = config.open().unwrap();
        let tree_storage = db.open_tree(dataset_name.clone()).unwrap();

        Self {
            dataset_name,
            attr_storage_config,
            tree_storage,
            db,
        }
    }

    /// Insert a vertex attribute.
    pub fn put_vertex_entry(
        &self,
        vertex_id: VInt,
        attr_name: String,
        attr_value: AttrValue
    ) {
        // Step 1. Build the attribute key.
        let vertex_key = VertexKey(VertexAttrKey {
            key_type: 0,
            vertex_id,
            key_name: attr_name,
        });
        // Step 2. Put it into the lsm-tree storage.
        self.tree_storage.insert(vertex_key.encode(), attr_value.encode()).unwrap();
    }


    /// Read a vertex attribute value.
    pub fn read_vertex_entry(
        &self,
        vertex_id: VInt,
        attr_name: String,
    ) -> Option<AttrValue> {
        // Step 1. Build attribute key.
        let vertex_key = VertexKey(VertexAttrKey {
            key_type: 0,
            vertex_id,
            key_name: attr_name,
        });

        // Step 2. Read the result.
        let attr_value_opt = self.tree_storage.get(vertex_key.encode()).unwrap();

        // Step 3. Parse the result.
        match attr_value_opt {
            None => {None}
            Some(attr_value) => {
                AttrValue::from_bytes(&attr_value)
            }
        }
    }

    /// Remove the vertex entry.
    pub fn remove_vertex_entry(
        &self,
        vertex_id: VInt,
        attr_name: String,
    ) {
        // Step 1. Build the vertex attr key.
        let vertex_key = VertexKey(VertexAttrKey {
            key_type: 0,
            vertex_id,
            key_name: attr_name,
        });
        // Step 2. Perform remove.
        self.tree_storage.remove(vertex_key.encode()).unwrap();
    }

    /// Insert an edge attribute.
    pub fn put_edge_entry(
        &self,
        src_vertex_id: VInt,
        dst_vertex_id: VInt,
        attr_name: String,
        attr_value: AttrValue
    ) {
        // Step 1. Build the attribute key.
        let edge_key = EdgeKey(EdgeAttrKey {
            key_type: 1,
            src_vertex_id,
            dst_vertex_id,
            key_name: attr_name,
        });
        // Step 2. Put it into the lsm-tree storage.
        self.tree_storage.insert(edge_key.encode(), attr_value.encode()).unwrap();
    }

    /// Read an edge attribute.
    pub fn read_edge_entry(
        &self,
        src_vertex_id: VInt,
        dst_vertex_id: VInt,
        attr_name: String
    ) -> Option<AttrValue> {
        // Step 1. Build attribute key.
        let edge_key = EdgeKey(EdgeAttrKey {
            key_type: 1,
            src_vertex_id,
            dst_vertex_id,
            key_name: attr_name,
        });

        // Step 2. Read the result.
        let attr_value_opt = self.tree_storage.get(edge_key.encode()).unwrap();

        // Step 3. Parse the result.
        match attr_value_opt {
            None => {None}
            Some(attr_value) => {
                AttrValue::from_bytes(&attr_value)
            }
        }
    }

    /// Remove an edge attribute.
    pub fn remove_edge_entry(
        &self,
        src_vertex_id: VInt,
        dst_vertex_id: VInt,
        attr_name: String
    ) {
        // Step 1. Build attribute key.
        let edge_key = EdgeKey(EdgeAttrKey {
            key_type: 1,
            src_vertex_id,
            dst_vertex_id,
            key_name: attr_name,
        });
        // Step 2. Perform remove.
        self.tree_storage.remove(edge_key.encode()).unwrap();
    }

    /// Load the attribute from a vector.
    pub fn load_attribute_from_vec(
        &self,
        label_info: Vec<u32>
    ) {
        // Insert the labels to attribute storage engine.
        for (vertex_id, vertex_label) in label_info.into_iter().enumerate() {
            self.put_vertex_entry(vertex_id as u32, "label".to_owned(), Num(vertex_label));
        }
    }

    /// Load the attributes into the attribute storage.
    pub fn load_attribute_from_file(
        &self,
        file_name: &str
    ) {
        // Read the label of each vertex.
        let graph_file = File::open(file_name).unwrap();
        let graph_reader = BufReader::new(graph_file);
        let mut label_map = BTreeMap::<VInt, u32>::new();
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
                    assert_eq!(tokens.len(), 4);
                    let parsed_vid = tokens[1].parse().ok().expect("File format error.");
                    let label = tokens[2].parse().ok().expect("File format error.");
                    // Process Vertices.
                    label_map.insert(parsed_vid, label);
                }
            }
        }

        // Insert the labels to attribute storage engine.
        for (vertex_id, vertex_label) in label_map.into_iter() {
            self.put_vertex_entry(vertex_id, "label".to_owned(), Num(vertex_label));
        }
    }

    /// Read all the attributes of each vertex.
    pub fn read_all_vertex_attr(&self) -> BTreeMap<VInt, AttrValue> {
        let attr_lsm_iter = self.tree_storage.iter();
        let mut attr_map = BTreeMap::<VInt, AttrValue>::new();
        for entry in attr_lsm_iter {
            let (attr_key_bytes, attr_value_bytes) = entry.unwrap();
            // Parse the vertex key and value.
            let attr_key = AttrKey::from_bytes(&attr_key_bytes).unwrap();
            match attr_key {
                VertexKey(vertex_attr_key) => {
                    // Parse the attr value.
                    let attr_value
                        = AttrValue::from_bytes(&attr_value_bytes).unwrap();
                    attr_map.insert(vertex_attr_key.vertex_id, attr_value);
                }
                EdgeKey(_) => {}
            }
        }
        attr_map
    }
}

#[cfg(test)]
mod test_attribute {
    use crate::attribute::{AttrKey, AttrStorage, AttrStorageConfig, AttrValue, EdgeAttrKey, VertexAttrKey};
    use crate::types::{Decode, Encode};

    /// Test the encoding and decoding of the vertex and edge keys.
    #[test]
    fn test_attr_key_encode_decode() {
        // Firstly, vertex keys.
        let vertex_key = VertexAttrKey {
            key_type: 0,
            vertex_id: 0,
            key_name: "hello".to_string(),
        };

        // Wrap and encode it.
        let attr_key_1 = AttrKey::VertexKey(vertex_key.clone());
        let attr_key_bytes_1 = attr_key_1.encode();

        // Secondly, edge keys.
        let edge_key = EdgeAttrKey {
            key_type: 1,
            src_vertex_id: 0,
            dst_vertex_id: 1,
            key_name: "world".to_string(),
        };

        // Wrap and encode it.
        let attr_key_2 = AttrKey::EdgeKey(edge_key.clone());
        let attr_key_bytes_2 = attr_key_2.encode();

        // Decode them.
        for attr_key_bytes in vec![attr_key_bytes_1, attr_key_bytes_2] {
            let attr_key = AttrKey::from_bytes(&attr_key_bytes).unwrap();
            match attr_key {
                AttrKey::VertexKey(vertex_attr_key) => {
                    assert_eq!(vertex_attr_key.key_name, vertex_key.key_name);
                    assert_eq!(vertex_attr_key.key_type, vertex_key.key_type);
                    assert_eq!(vertex_attr_key.vertex_id, vertex_key.vertex_id);
                }
                AttrKey::EdgeKey(edge_attr_key) => {
                    assert_eq!(edge_attr_key.key_name, edge_key.key_name);
                    assert_eq!(edge_attr_key.key_type, edge_key.key_type);
                    assert_eq!(edge_attr_key.src_vertex_id, edge_key.src_vertex_id);
                    assert_eq!(edge_attr_key.dst_vertex_id, edge_key.dst_vertex_id);
                }
            }
        }
    }

    #[test]
    fn test_vertex_attr_read_write_remove() {
        // Step 1. Build an LSM-Tree.
        let attr_storage = AttrStorage::create(
            "test_vertex".to_owned(),
            AttrStorageConfig {
                cache_capacity: 1000 * 1000 * 512,
                flush_frequency: 1000,
            }
        );

        // Step 2. Insert a vertex attribute key value.
        attr_storage.put_vertex_entry(
            1u32,
            "label".to_owned(),
            AttrValue::Num(34u32)
        );

        // Step 3. Read the value.
        let attr_value = attr_storage.read_vertex_entry(
            1u32,
            "label".to_owned()
        ).unwrap();

        // Check.
        match attr_value {
            AttrValue::Num(attr_value_num) => {
                assert_eq!(attr_value_num, 34u32);
            }
            AttrValue::Str(_) => {
                panic!("Wrong Value Type.")
            }
        }

        // Check read all.
        for (vertex_id, attr_v) in attr_storage.read_all_vertex_attr() {
            match attr_v {
                AttrValue::Num(value) => {
                    println!("Vertex({}) -> Attr({})", vertex_id, value);
                }
                AttrValue::Str(value) => {
                    println!("Vertex({}) -> Attr({})", vertex_id, value);
                }
            }
        }

        // Step 4. Remove it.
        attr_storage.remove_vertex_entry(
            1u32,
            "label".to_owned()
        );

        // Read.
        let attr_value_opt = attr_storage.read_vertex_entry(
            1u32,
            "label".to_owned()
        );

        // Check.
        match attr_value_opt {
            None => {
                // Test Pass.
            }
            Some(_) => {
                panic!("This entry should be removed.");
            }
        }
    }

    #[test]
    fn test_edge_attr_read_write_remove() {
        // Step 1. Build an LSM-Tree.
        let attr_storage = AttrStorage::create(
            "test_edge".to_owned(),
            AttrStorageConfig {
                cache_capacity: 1000 * 1000 * 512,
                flush_frequency: 1000,
            }
        );

        // Step 2. Insert a vertex attribute key value.
        attr_storage.put_edge_entry(
            1u32,
            2u32,
            "label".to_owned(),
            AttrValue::Num(34u32)
        );

        // Step 3. Read the value.
        let attr_value = attr_storage.read_edge_entry(
            1u32,
            2u32,
            "label".to_owned()
        ).unwrap();

        // Check.
        match attr_value {
            AttrValue::Num(attr_value_num) => {
                assert_eq!(attr_value_num, 34u32);
            }
            AttrValue::Str(_) => {
                panic!("Wrong Value Type.")
            }
        }

        // Step 4. Remove it.
        attr_storage.remove_edge_entry(
            1u32,
            2u32,
            "label".to_owned()
        );

        // Read.
        let attr_value_opt = attr_storage.read_edge_entry(
            1u32,
            2u32,
            "label".to_owned()
        );

        // Check.
        match attr_value_opt {
            None => {
                // Test Pass.
            }
            Some(_) => {
                panic!("This entry should be removed.");
            }
        }
    }
}

