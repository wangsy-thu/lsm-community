use std::collections::BTreeMap;
use std::borrow::Borrow;

use anyhow::Result;

use crate::graph::VInt;
use crate::types::{GraphEntry, V32};

/// A community cache store the recently updated entries in memory and store the entries loaded
/// from the disk.
/// When the cache is full, the entries will be flushed into the disk.
/// Attention that when the updated key is found in bucket cache, the corresponding graph entry is
/// removed from the bucket cache.
pub(crate) struct CommCache {
    pub(crate) write_buffer: WriteBuffer, // The write buffer.
    pub(crate) bucket_cache: BucketCache  // The bucket cache.
}

#[allow(dead_code)]
impl CommCache {

    pub(crate) fn get_buffer_size(&self) -> usize {
        // Return the size of the buffer.
        self.write_buffer.get_used_size()
    }

    pub(crate) fn get_cache_size(&self) -> usize {
        // Return the size of the cache.
        self.bucket_cache.get_used_size()
    }

    pub(crate) fn clear_buffer(&mut self) {
        // Clear all entries in the write buffer.
        self.write_buffer.clear().unwrap();
    }

    pub(crate) fn clear_cache(&mut self) {
        // Clear all entries in the bucket cache.
        self.bucket_cache.cached_entries.clear();
    }
    pub(crate) fn get_from_cache(&self, vertex_id: &VInt) -> Option<Vec<GraphEntry<V32>>> {
        match self.bucket_cache.cached_entries.get(&vertex_id) {
            None => {
                None
            }
            Some(entry_list) => {
                let mut res = vec![];
                for entry in entry_list {
                    res.push(entry.clone());
                }
                Some(res)
            }
        }
    }

    pub(crate) fn get_cache_ref(&self) -> &BTreeMap<VInt, Vec<GraphEntry<V32>>> {
        // Get the reference of the cache.
        self.bucket_cache.cached_entries.borrow()
    }

    pub(crate) fn get_buffer_ref(&self) -> &BTreeMap<VInt, Vec<GraphEntry<V32>>> {
        // Get the reference of the buffer.
        self.write_buffer.buffered_entries.borrow()
    }

    pub(crate) fn insert_to_cache(&mut self, entry_list: Vec<GraphEntry<V32>>) {
        // Insert entries to the bucket cache.
        // Double check: the cached entry never exists in buffer.
        let valid_entry_list = entry_list.iter()
            .filter(|&item| {
                !self.write_buffer.buffered_entries.contains_key(&item.key.vertex_id)
        }).cloned().collect();
        self.bucket_cache.load(valid_entry_list);
    }

    pub(crate) fn insert_to_buffer(&mut self, entry_list: Vec<GraphEntry<V32>>) {
        // Firstly, perform disable for bucket cache.
        self.bucket_cache.disable(entry_list.iter().map(|item| {
            item.key.vertex_id
        }).collect());
        for entry in entry_list {
            self.write_buffer.insert_entry(entry).unwrap();
        }
    }

    pub(crate) fn get_from_buffer(&self, vertex_id: &VInt) -> Option<Vec<GraphEntry<V32>>> {
        self.write_buffer.get_entry_list(vertex_id)
    }

    pub(crate) fn create() -> CommCache {
        CommCache {
            write_buffer: WriteBuffer::create(),
            bucket_cache: BucketCache::create()
        }
    }
}

/// The Bucket Cache is used to cache the recently read graph entries from the blocks.
/// During the fetch process, the fetched graph entries are pushed into the bucket cache.
pub(crate) struct BucketCache {
    pub(crate) used_bytes: usize,  // Number of bytes used in this community cache.
    pub(crate) cached_entries: BTreeMap<VInt, Vec<GraphEntry<V32>>>  // Store the recently updated entries.
}

impl BucketCache {
    pub(crate) fn create() -> BucketCache {
        // Create a new empty bucket cache.
        BucketCache {
            used_bytes: 0usize,
            cached_entries: Default::default()
        }
    }

    pub(crate) fn disable(&mut self, vertex_ids: Vec<VInt>) {
        // Disable a vertex id when the corresponding graph entry is updated.
        for v_id in vertex_ids {
            if self.cached_entries.contains_key(&v_id) {
                self.used_bytes -= self.cached_entries.get(&v_id).unwrap().iter()
                    .fold(0,
                          |decrease_size, item| decrease_size + item.get_entry_size());
                self.cached_entries.remove(&v_id);
            }
        }
    }

    pub (crate) fn load(&mut self, graph_entry_list: Vec<GraphEntry<V32>>) {
        // Load the recently read graph entries into bucket cache.
        // Load function take the ownership of graph entry list (A move happens).
        for g_entry in graph_entry_list {
            self.used_bytes += g_entry.get_entry_size();
            if !self.cached_entries.contains_key(&g_entry.key.vertex_id) {
                self.cached_entries.insert(g_entry.key.vertex_id, vec![g_entry]);
            } else {
                self.cached_entries.get_mut(&g_entry.key.vertex_id).unwrap().push(g_entry);
            }
        }
        // Before the loading process, the updated entries are removed from the bucket cache.
    }

    pub(crate) fn get_used_size(&self) -> usize {
        self.used_bytes
    }
}
#[allow(dead_code)]
pub(crate) struct WriteBuffer {
    pub(crate) used_bytes: usize,  // Number of bytes used in this community cache.
    pub(crate) buffered_entries: BTreeMap<VInt, Vec<GraphEntry<V32>>>  // Store the recently updated entries.
}

#[allow(dead_code)]
impl WriteBuffer {

    pub(crate) fn create() -> WriteBuffer {
        WriteBuffer {
            used_bytes: 0,
            buffered_entries: Default::default(),
        }
    }

    pub(crate) fn get_used_size(&self) -> usize {
        self.used_bytes
    }

    pub(crate) fn insert_entry(&mut self, entry: GraphEntry<V32>) -> Result<()> {
        // Insert an entry to cache.
        self.used_bytes += entry.get_entry_size();
        if self.buffered_entries.contains_key(&entry.key.vertex_id) {
            self.buffered_entries.get_mut(&entry.key.vertex_id).unwrap().push(entry);
        } else {
            self.buffered_entries.insert(entry.key.vertex_id, vec![entry]);
        }
        Ok(())
    }

    pub(crate) fn clear(&mut self) -> Result<()> {
        self.buffered_entries.clear();
        self.used_bytes = 0;
        Ok(())
    }

    pub(crate) fn get_entry_list(&self, vertex_id: &VInt) -> Option<Vec<GraphEntry<V32>>> {
        match self.buffered_entries.get(&vertex_id) {
            None => {None}
            Some(entry_list) => {
                let mut res = vec![];
                for entry in entry_list {
                    res.push(entry.clone());
                }
                Some(res)
            }
        }
    }
}

#[cfg(test)]
mod test_comm_cache {
    use crate::bucket::block::CommBlock;
    use crate::bucket::cache::{CommCache, WriteBuffer};
    use crate::bucket::test_comm_bucket::generate_example;

    #[test]
    fn test_insert_cache() {
        // Test cache insert operation.
        let example_block = match generate_example(false) {
            CommBlock::CSR(_) => {
                panic!("An error happen.")
            }
            CommBlock::KV(unsorted_block) => {
                unsorted_block
            }
        };

        let mut cache = WriteBuffer {
            used_bytes: 0,
            buffered_entries: Default::default(),
        };
        for entry in &example_block.unsorted_entry_list {
            match cache.insert_entry(entry.clone()) {
                Ok(_) => {println!("Entry insert success!")}
                Err(_) => {println!("Entry insert fail!")}
            }
        }
        for (gt_entry, cached_entry)
                in cache.buffered_entries.iter().zip(example_block.unsorted_entry_list.iter()) {
            println!("{:?},{}", gt_entry.1, cached_entry)
        }
    }

    #[test]
    fn test_cache_clear() {
        // Test cache clear operations.
        let example_block = match generate_example(false) {
            CommBlock::CSR(_) => {
                panic!("An error happen.")
            }
            CommBlock::KV(kv_block) => {
                kv_block
            }
        };

        let mut cache = WriteBuffer::create();
        for entry in &example_block.unsorted_entry_list {
            match cache.insert_entry(entry.clone()) {
                Ok(_) => {println!("Entry insert success!")}
                Err(_) => {println!("Entry insert fail!")}
            }
        }

        match cache.clear() {
            Ok(_) => {println!("Clear success.")}
            Err(_) => {println!("Clear fail.")}
        }

        assert!(cache.buffered_entries.is_empty());
        assert_eq!(cache.used_bytes, 0);
    }

    #[test]
    fn test_comm_cache_read_buffer() {
        let mut test_cache = CommCache::create();
        let example_block = match generate_example(false) {
            CommBlock::CSR(_) => {
                panic!("An error happen.")
            }
            CommBlock::KV(unsorted_block) => {
                unsorted_block
            }
        };
        let entry_list = example_block.unsorted_entry_list.clone().into_iter()
            .fold(vec![], |mut collector, item| {
                collector.push(item);
                collector
            });
        test_cache.insert_to_buffer(entry_list);
        for ground_truth in example_block.unsorted_entry_list {
            match test_cache.get_from_buffer(&ground_truth.key.vertex_id) {
                None => {
                    panic!("Test Failed.");
                }
                Some(entry_list) => {
                    assert_eq!(ground_truth, entry_list[0]);
                }
            }
        }
    }

    #[test]
    fn test_comm_cache() {
        let mut test_cache = CommCache::create();
        let example_block = match generate_example(false) {
            CommBlock::CSR(_) => {
                panic!("An error happen.")
            }
            CommBlock::KV(unsorted_block) => {
                unsorted_block
            }
        };
        let entry_list = example_block.unsorted_entry_list.clone().into_iter()
            .fold(vec![], |mut collector, item| {
                collector.push(item);
                collector
            });
        test_cache.insert_to_cache(entry_list);
        for ground_truth in example_block.unsorted_entry_list {
            match test_cache.get_from_cache(&ground_truth.key.vertex_id) {
                None => {
                    for cached_entry in &test_cache.bucket_cache.cached_entries {
                        println!("Key: {}, Value: {}", cached_entry.0, cached_entry.1[0]);
                    }
                    panic!("Cache Miss, Test Failed.");
                }
                Some(entry_list) => {
                    println!("Cache Hit, Check value.");
                    assert_eq!(ground_truth, entry_list[0]);
                }
            }
        }
    }

    #[test]
    fn test_conflict() {
        let mut test_cache = CommCache::create();
        let example_block = match generate_example(false) {
            CommBlock::CSR(_) => {
                panic!("An error happen.")
            }
            CommBlock::KV(kv_block) => {
                kv_block
            }
        };
        let entry_list = example_block.unsorted_entry_list.clone().into_iter()
            .fold(vec![], |mut collector, item| {
                collector.push(item);
                collector
            });

        test_cache.insert_to_cache(vec![entry_list[0].clone()]);
        println!("Entry Count in Cache (Before Insert to Buffer): {}", test_cache.bucket_cache.cached_entries.len());
        test_cache.insert_to_buffer(entry_list);
        assert_eq!(test_cache.bucket_cache.used_bytes, 0);

    }
}