use std::borrow::Borrow;
use std::collections::BTreeSet;


use crate::graph::VInt;

/// The page number.
pub type PageNum = usize;

/// The block number, represents a set of page number.
pub type BlockNums = Vec<PageNum>;

/// Page number manager, record and allocate the unused page number.
#[allow(dead_code)]
pub(crate) struct PageNumManager {
    pub(crate) page_number_set: BTreeSet<PageNum>
}

#[allow(dead_code)]
impl PageNumManager {
    pub(crate) fn allocate_page_num(&mut self) -> PageNum {

        if self.page_number_set.is_empty() {
            // If there is no blocks.
            self.page_number_set.insert(0);
            0
        } else {
            // Step 1, find the maximal block number.
            let maximal_page_num = *self.page_number_set.last().unwrap();
            self.page_number_set.insert(maximal_page_num + 1);
            maximal_page_num + 1
        }
    }

    pub(crate) fn add_page_num(&mut self, page_num: PageNum) {
        // Insert a new block number to this manager.
        match self.check_valid(page_num) {
            true => {self.page_number_set.insert(page_num);}
            false => {println!("Block number {} invalid!", page_num)}
        }
    }

    pub(crate) fn check_valid(&self, page_num: PageNum) -> bool {
        // Check whether this block number exists.
        !self.page_number_set.contains(&page_num)
    }
}

#[allow(dead_code)]
pub(crate) trait CommIndex: Sized {

    /// Insert block for rebuild the index.
    fn insert_page_rebuild(&mut self, min_vertex_id: VInt, block_num: usize, block_type: u32);

    /// Insert a new block number to this community index.
    fn insert_page(&mut self, min_vertex_id: VInt, page_num: usize);

    // Get the block number of the sorted part.
    fn get_csr_block(&self, vertex_id: &VInt) -> BlockNums;

    // Get the key-value block number according to the community index.
    fn get_kv_block(&self, vertex_id: &VInt) -> PageNum;

    /// Get the block number according to the vertex ID.
    fn get_block(&self, vertex_id: &VInt) -> BlockNums;

    /// Build the community index from the pair, used for rebuilding from the blocks
    /// in the disk.
    fn build_from_pairs(pair_list: &Vec<(usize, VInt)>) -> Self;

    /// Get the immutable reference of the block numbers.
    fn get_page_nums_ref(&self) -> &BTreeSet<PageNum>;
}

/// Map the Key of graph entry into the offset, used for locating the pages quickly.
/// First element is the index and the second is the max page number.
/// Through block number, we can grab the corresponding block, like the pointer.
/// From now, it needs to be revised, from a range to some numbers.
pub(crate) struct SkipListCommIndex (pub(crate) crossbeam_skiplist::SkipMap<VInt, (PageNum, Vec<PageNum>)>,
                              pub(crate) PageNumManager);

/// Map the community id into the offset, similar to the community index,
pub struct SkipListTreeIndex (pub(crate) crossbeam_skiplist::SkipMap<VInt, Vec<PageNum>>,
                              pub(crate) PageNumManager);

#[allow(dead_code)]
impl SkipListTreeIndex {

    /// Create a new skip list tree index.
    pub(crate) fn new() -> SkipListTreeIndex {
        // Create a new SkipList Tree Index.
        SkipListTreeIndex {
            0: Default::default(),
            1: PageNumManager {
                page_number_set: Default::default(),
            },
        }
    }

    // Insert an interval into this tree index.
    pub(crate) fn insert_interval(&mut self, min_vertex_id: VInt) {
        // Insert a new interval into tree index.
        if !self.0.contains_key(&min_vertex_id) {
            let new_kv_block = self.1.allocate_page_num();
            self.0.insert(min_vertex_id, vec![new_kv_block]);
        }
    }

    /// Append a new page after an exist interval.
    pub(crate) fn append(&mut self, min_vertex_id: VInt) {
        // Append a csr block after the interval.
        // Locate
        let target_entry = self.0.range(0..min_vertex_id + 1).next_back();
        let key_min_vid = match target_entry {
            None => {
                self.0.iter().next_back().unwrap().key().clone()
            }
            Some(target_entry) => {
                target_entry.key().clone()
            }
        };
        if self.0.contains_key(&key_min_vid) {
            // Perform read-modify-write operation.
            let new_sorted_block = self.1.allocate_page_num();
            let mut block_info = self.0.get(&key_min_vid).unwrap().value().clone();
            // Remove the old one.
            self.0.remove(&key_min_vid);
            // Update block info.
            block_info.push(new_sorted_block);
            self.0.insert(key_min_vid, block_info);
        }
    }

    /// Insert a new page number when rebuilding the index.
    fn insert_page_rebuild(&mut self, min_vertex_id: VInt, page_num: usize) {
        // Insert block function used for rebuilding.
        // Register the block number.
        self.1.add_page_num(page_num);
        // Check whether key exists.
        if self.0.contains_key(&min_vertex_id) {
            let mut block_info = self.0.get(&min_vertex_id).unwrap().value().clone();
            // Remove the old one.
            self.0.remove(&min_vertex_id);
            // Update block info.
            block_info.push(page_num);
            self.0.insert(min_vertex_id, block_info);
        } else {
            let mut block_info = vec![];
            block_info.push(page_num);
            self.0.insert(min_vertex_id, block_info);
        }
    }

    /// Insert a new page number to this tree index.
    fn insert_page(&mut self, min_vertex_id: VInt, page_num: usize) {
        // Insert a new block number to this skip list.
        // Check whether the block number valid.
        if !self.1.check_valid(page_num) {
            return;
        }
        self.1.add_page_num(page_num);
        // If this min vertex id exists, insert it into vec.
        if self.0.contains_key(&min_vertex_id) {
            let mut blocks = self.0.get(&min_vertex_id).unwrap().value().clone();
            self.0.remove(&min_vertex_id);
            blocks.push(page_num);
            self.0.insert(min_vertex_id, blocks);
        } else {
            // Allocate a new block number for the sorted part.
            let un_block_num = self.1.allocate_page_num();
            // Else insert this entry into skip list.
            self.0.insert(min_vertex_id, vec![un_block_num]);
        }
    }

    
    /// Get block number according to the vertex ID.
    pub fn get_block(&self, vertex_id: &VInt) -> Vec<PageNum> {
        // Using range query to get the target block.
        // Query: the maximal key not larger than the target key.
        let target_entry = self.0.range(0..vertex_id + 1).next_back();
        let result_blocks = match target_entry {
            None => {vec![]}
            Some(entry) => {
                entry.value().clone()
            }
        };
        result_blocks
    }
}

#[allow(dead_code)]
impl CommIndex for SkipListCommIndex {
    fn insert_page_rebuild(&mut self, min_vertex_id: VInt, block_num: usize, block_type: u32) {
        // Insert block function used for rebuilding.
        // Register the block number.
        self.1.add_page_num(block_num);
        // Check whether key exists.
        if self.0.contains_key(&min_vertex_id) {
            let mut block_info = self.0.get(&min_vertex_id).unwrap().value().clone();
            // Remove the old one.
            self.0.remove(&min_vertex_id);
            // Update block info.
            if block_type == 0 {
                // Sorted block, insert it into list.
                block_info.1.push(block_num);
            } else {
                block_info.0 = block_num;
            }
            self.0.insert(min_vertex_id, block_info);
        } else {
            let mut block_info = (0, vec![]);
            if block_type == 0 {
                // Sorted block, insert it into list.
                block_info.1.push(block_num);
            } else {
                block_info.0 = block_num;
            }
            self.0.insert(min_vertex_id, block_info);
        }
    }

    fn insert_page(&mut self, min_vertex_id: VInt, block_num: usize) {
        // Insert a new block number to this skip list.
        // Check whether the block number valid.
        if !self.1.check_valid(block_num) {
            return;
        }
        self.1.add_page_num(block_num);
        // If this min vertex id exists, insert it into vec.
        if self.0.contains_key(&min_vertex_id) {
            let mut blocks = self.0.get(&min_vertex_id).unwrap().value().clone();
            self.0.remove(&min_vertex_id);
            blocks.1.push(block_num);
            self.0.insert(min_vertex_id, blocks);
        } else {
            // Allocate a new block number for the sorted part.
            let un_block_num = self.1.allocate_page_num();
            // Else insert this entry into skip list.
            self.0.insert(min_vertex_id, (block_num, vec![un_block_num]));
        }
    }

    fn get_csr_block(&self, vertex_id: &VInt) -> Vec<PageNum> {
        // Locate the max vertex id in the index which not larger than the target vid.
        let target_entry = self.0.range(0..vertex_id + 1).next_back();
        let result_blocks = match target_entry {
            None => {vec![]}
            Some(entry) => {
                entry.value().1.clone()
            }
        };
        result_blocks
    }

    fn get_kv_block(&self, vertex_id: &VInt) -> PageNum {
        // Find the wanted and sorted blocks through community index.
        // Similar to find_sorted_blocks().
        // Locate the max vertex id in the index which not larger than the target vid.
        let target_entry = self.0.range(0..vertex_id + 1).next_back();
        let result_block = match target_entry {
            None => {
                // If target not found, return the all block.
                // fetch_all() will is coming soon.
                0usize
            }
            Some(entry) => {
                entry.value().0.clone()
            }
        };
        result_block
    }

    fn get_block(&self, vertex_id: &VInt) -> Vec<PageNum> {
        // Using range query to get the target block.
        // Query: the maximal key not larger than the target key.
        let target_entry = self.0.range(0..vertex_id + 1).next_back();
        let result_blocks = match target_entry {
            None => {vec![]}
            Some(entry) => {
                let mut res = entry.value().1.clone();
                res.insert(0, entry.value().0.clone());
                res
            }
        };
        result_blocks
    }

    fn build_from_pairs(pair_list: &Vec<(usize, VInt)>) -> Self {
        // Build a skip map from a set of pairs.
        let mut comm_idx = SkipListCommIndex::new();
        for (offset, _) in pair_list {
            comm_idx.1.add_page_num(*offset);  // Take this block number.
        }
        for (offset, min_vertex_id) in pair_list {
            // Allocate a new number for the sorted part.
            comm_idx.0.insert(*min_vertex_id, (*offset, vec![comm_idx.1.allocate_page_num()]));
        }
        comm_idx
    }

    /// Return the immutable ref of the block number manager.
    fn get_page_nums_ref(&self) -> &BTreeSet<PageNum> {
        // Immutable reference.
        self.1.page_number_set.borrow()
    }
}

#[allow(dead_code)]
impl SkipListCommIndex {
    pub(crate) fn new() -> SkipListCommIndex {
        // Create a new SkipList community index.
        SkipListCommIndex {
            0: Default::default(),
            1: PageNumManager {
                page_number_set: Default::default(),
            },
        }
    }

    pub(crate) fn init(&mut self) {
        // Initialize.
        self.insert_interval(0);
    }

    pub(crate) fn insert_interval(&mut self, min_vertex_id: VInt) {
        // Insert a new interval into community index.
        if !self.0.contains_key(&min_vertex_id) {
            let new_kv_block = self.1.allocate_page_num();
            self.0.insert(min_vertex_id, (new_kv_block, vec![]));
        }
    }

    /// Append a new page to a specific range.
    pub(crate) fn append(&mut self, min_vertex_id: VInt) {
        // Append a csr block after the interval.
        // Locate
        let target_entry = self.0.range(0..min_vertex_id + 1).next_back();
        let key_min_vid = match target_entry {
            None => {
                self.0.iter().next_back().unwrap().key().clone()
            }
            Some(target_entry) => {
                target_entry.key().clone()
            }
        };
        if self.0.contains_key(&key_min_vid) {
            let new_sorted_block = self.1.allocate_page_num();
            let mut block_info = self.0.get(&key_min_vid).unwrap().value().clone();
            // Remove the old one.
            self.0.remove(&key_min_vid);
            // Update block info.
            block_info.1.push(new_sorted_block);
            self.0.insert(key_min_vid, block_info);
        }
    }

    /// Remove a page from a specific range.
    pub(crate) fn shrink(&mut self, min_vertex_id: VInt) {
        // Firstly, locate the entry.
        let target_entry = self.0.range(0..min_vertex_id + 1).next_back();
        let key_min_vid = match target_entry {
            None => {
                self.0.iter().next_back().unwrap().key().clone()
            }
            Some(target_entry) => {
                target_entry.key().clone()
            }
        };
        if self.0.contains_key(&key_min_vid) {
            let mut block_info = self.0.get(&key_min_vid).unwrap().value().clone();
            // Remove the old one.
            self.0.remove(&key_min_vid);
            // Pop the last page num of the vec.
            if block_info.1.len() > 1 {
                block_info.1.pop();
            }
            self.0.insert(key_min_vid, block_info);
        }
    }
}


#[cfg(test)]
mod test_comm_index {
    use crate::bucket::bucket_index::{CommIndex, SkipListCommIndex};

    #[test]
    fn test_skip_insert() {
        let test_pairs = vec![(0, 0), (1, 10), (2, 20), (3, 30)];
        let comm_idx = SkipListCommIndex::build_from_pairs(&test_pairs);
        for entry in &comm_idx.0 {
            println!("Key: {}, Value: {:?}", entry.key(), entry.value());
        }
    }

    #[test]
    fn test_skip_find_key() {
        let test_pairs = vec![(0, 0), (1, 10), (2, 20), (3, 30)];
        let comm_idx = SkipListCommIndex::build_from_pairs(&test_pairs);
        let blocks = comm_idx.get_block(&5);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0], 0);
        assert_eq!(blocks[1], 4);
        let blocks = comm_idx.get_block(&15);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0], 1);
        assert_eq!(blocks[1], 5);
    }
}