use std::collections::{BTreeMap, HashMap};
use std::fs::{File, metadata};
use std::os::unix::fs::FileExt;
use std::path::Path;

use anyhow::Result;

use block::TreeBlock;
use bucket_index::SkipListTreeIndex;

use crate::bucket::block::{CSRCommBlock, Deserialize, KVCommBlock, Query, Serialize};
use crate::bucket::block::CommBlock::{CSR, KV};
use crate::bucket::bucket_index::{CommIndex, PageNum, SkipListCommIndex};
use crate::bucket::cache::CommCache;
use crate::bucket::page::Page;
use crate::comm_table::CommID;
use crate::community::CommNeighbors;
use crate::compact::CompactController;
use crate::config::{APP_BLOCK_SIZE, BLOCK_SIZE, MAX_CACHE_SIZE, PAGE_SIZE};
use crate::graph::{Graph, GraphSnapshot, VInt};
use crate::types::{Decode, Encode, GraphEntry, V32, V32_SIZE, Vertex};

pub mod block;
pub mod bucket_index;
mod cache;
pub mod page;

/// A file object.
pub struct FileObject(Option<File>, u64);

#[allow(dead_code)]
impl FileObject {
    pub fn read(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        use std::os::unix::fs::FileExt;
        let mut data = vec![0; len as usize];
        self.0
            .as_ref()
            .unwrap()
            .read_at(&mut data[..], offset)?;
        Ok(data)
    }

    pub fn size(&self) -> u64 {
        self.1
    }

    pub fn create_empty_file(path: &Path) -> Result<Self> {
        std::fs::write(path, &"")?;
        File::open(path)?.sync_all()?;
        Ok(FileObject(
            Some(File::options().read(true).write(true).open(path)?),
            0u64,
        ))
    }

    /// Create a new file object (day 2) and write the file to the disk (day 4).
    pub fn create(path: &Path, data: Vec<u8>) -> Result<Self> {
        std::fs::write(path, &data)?;
        File::open(path)?.sync_all()?;
        Ok(FileObject(
            Some(File::options().read(true).write(true).open(path)?),
            data.len() as u64,
        ))
    }

    pub fn open(path: &Path) -> Result<Self> {
        let file = File::options().read(true).write(false).open(path)?;
        let size = file.metadata()?.len();
        Ok(FileObject(Some(file), size))
    }

    pub fn flush(&self, data: Vec<u8>, offset: u64) -> Result<()> {
        // Make the data size same as the BLOCK_SIZE.
        self.0
            .as_ref()
            .unwrap()
            .write_at(&data[..], offset)?;
        Ok(())
    }
}

/// The community bucket to maintain a community in the stored graph.
#[allow(dead_code)]
pub(crate) struct CommBucket {
    pub(crate) comm_index: SkipListCommIndex,  // Community index, to quickly locate the block.
    pub(crate) comm_file: FileObject,  // File to store the community.
    pub(crate) comm_cache: CommCache  // A community cache to store recently inserted entry.
}

/// Tree Bucket, to store the non-leaf nodes of the community tree.
/// Each layer has a community graph.
#[allow(dead_code)]
pub struct TreeBucket {
    layer: u32, // The layer of this community tree.
    tb_id: u32, // The id of this community tree bucket in this layer.
    tree_index: SkipListTreeIndex,  // Tree index, to quickly locate the block.
    tree_file: FileObject  // File to store the community tree.
}

#[allow(dead_code)]
impl TreeBucket {

    /// Display the tree_bucket information.
    pub(crate) fn print_bucket(&self) {
        // Print the tree bucket.
        for index_entry in self.tree_index.0.iter() {
            let page_nums = index_entry.value().clone();
            let interval_start = index_entry.key().clone();
            let tree_block = self.fetch_tree_block_through_nums(&page_nums);
            println!("Interval Start: {}", interval_start);
            println!("Tree Block, Min ID: {}, Max ID: {}", tree_block.min_comm_id, tree_block.max_comm_id);
        }
    }
    
    /// Fetch a page from the file object according to the page number.
    pub(crate) fn fetch_page(&self, page_num: PageNum) -> Page {
        // Step 1. Read the bytes.
        let page_bytes =
            self.tree_file.read((page_num * PAGE_SIZE) as u64, PAGE_SIZE as u64).unwrap();
        // Step 2. Decode it and return the value.
        Page::from_bytes(&page_bytes).unwrap()
    }

    /// Fetch a tree block according to the block numbers, similar to fetching csr in community block.
    pub(crate) fn fetch_tree_block_through_nums(&self, page_nums: &Vec<PageNum>) -> TreeBlock {
        // Fetch a tree block according to the block number, similar to those in community block.
        // Step 1, load the page in bytes.
        let mut tree_pages = vec![];
        for page_num in page_nums {
            tree_pages.push(self.fetch_page(*page_num));
        }
        // Step 2, decode the block from bytes and return value.
        TreeBlock::decode_from_pages(&tree_pages).unwrap()
    }

    /// Build the tree bucket from a list of community neighbors.
    pub(crate) fn build_bucket_from_neighbors(dataset_name: &str, layer: u32, tb_id: u32, comm_graph: &BTreeMap<CommID, CommNeighbors>) -> TreeBucket {
        // Step 1. Group those neighbors according to the size of them.
        let mut group_size = 0usize;  // current size in each group.
        let mut community_group = vec![];
        let mut current_community_group = vec![];
        for (vertex_id, comm_neighbors) in comm_graph {
            // Compute the size of this community neighbors.
            let comm_neighbors_size = comm_neighbors.iter()
                .fold(0usize, |acc, item| {
                    acc + *item.0 as usize + item.1.len() * 2 * size_of::<u32>()
                });

            if comm_neighbors_size > APP_BLOCK_SIZE - 2 * size_of::<u32>() {
                // For sure, need a new block for this vertex.
                // Collect the old block.
                if group_size > 0 {
                    community_group.push(current_community_group);
                    current_community_group = vec![];
                    group_size = 0;
                }
                // Create a new group and continue.
                community_group.push(vec![*vertex_id]);
                continue;
            } else if group_size + comm_neighbors_size > APP_BLOCK_SIZE {
                // For sure, need a new block.
                // Collect the old block.
                community_group.push(current_community_group);
                // Create a new group.
                current_community_group = vec![];
                group_size = 0;
            }
            group_size += comm_neighbors_size;
            current_community_group.push(*vertex_id);
        }
        community_group.push(current_community_group);

        // Step 2. Build the tree index based on the community neighbors.
        let mut tree_index = SkipListTreeIndex::new();
        // Build page info.
        let mut page_num_map = HashMap::<PageNum, Page>::new();
        for c_group in community_group {
            // Insert this interval into community index.
            tree_index.insert_interval(c_group[0].clone());
            // A vertex maps to several blocks.
            // Step 1. Generate a CSR block.
            let new_tree_block = if c_group.len() == 1 {
                TreeBlock::create(
                    &vec![(c_group[0], comm_graph.get(&c_group[0]).unwrap().clone())]
                )
            } else {
                let mut c_neighbor_list = vec![];
                for comm_id in &c_group {
                    c_neighbor_list.push((*comm_id, comm_graph.get(comm_id).unwrap().clone()));
                }
                TreeBlock::create(&c_neighbor_list)
            };

            // Step 2. Serialize this tree block into pages.
            let page_list = new_tree_block.encode_to_pages();

            // Step 3. build the tree index.
            for _ in 0..page_list.len() {
                tree_index.append(c_group[0].clone());
            }

            // Step 4. Complete the metadata of these pages and bind with page number.
            let allocated_page_nums = tree_index.get_block(
                &c_group[0]
            );
            for (page_index, mut new_page) in page_list.into_iter().enumerate() {
                if page_index < allocated_page_nums.len() - 1 {
                    // Not the last page.
                    new_page.has_next = 1u8;
                    new_page.next_page_num = allocated_page_nums[page_index + 1];
                    new_page.block_offset = page_index;
                } else {
                    new_page.has_next = 0u8;
                    new_page.block_offset = page_index;
                }
                page_num_map.insert(allocated_page_nums[page_index], new_page);
            }
        }

        // Step 5. Build the tree file and flush the data.

        // Flush them all.
        // Step 1. Prepare Bucket file.
        let tree_bucket_file_name = format!("lsm.db/{}_tb_{}_{}.txt", dataset_name, layer, tb_id);
        let path = Path::new(&tree_bucket_file_name);

        // Ensure the file and directory exist (cleanup if necessary)
        if path.exists() {
            std::fs::remove_file(path).expect("File remove failed!");
        }
        std::fs::create_dir_all(path.parent().unwrap()).expect("Dir make failed!");

        // Step 2. Create the file object.
        let tree_file = FileObject::create_empty_file(&path).unwrap();

        // Flush the sorted part.
        for (offset, page) in page_num_map.into_iter() {
            match tree_file.flush(page.encode(), (offset * PAGE_SIZE) as u64) {
                Ok(_) => {}
                Err(_) => {}
            }
        }

        // Return value.
        TreeBucket {
            layer,
            tb_id,
            tree_index,
            tree_file
        }
    }

}

#[allow(dead_code)]
impl CommBucket {

    /// To partition the large entries into smaller ones.
    pub(crate) fn graph_partition(large_graph_entry: GraphEntry<V32>, maximal_entry_size: usize) -> Vec<GraphEntry<V32>> {
        // Graph partition, split a large graph entry into several small entries.
        let mut res_small_entries = vec![];
        let mut current_size = 0usize;
        let mut current_neighbor_list = vec![];
        let mut current_partition_id = 0u16;
        for neighbor in large_graph_entry.neighbors {
            if current_size + 2 * V32_SIZE + 4 > maximal_entry_size {
                // Need a new graph entry.
                res_small_entries.push(
                    GraphEntry::create_with_partition(
                        large_graph_entry.key.clone(), current_neighbor_list, current_partition_id)
                );
                // Create a new list avoid move.
                current_neighbor_list = vec![];
                current_size = 0;
            }
            current_neighbor_list.push(neighbor);
            current_size += V32_SIZE;
            current_partition_id += 1;
        }
        // Tackle the last neighbors.
        res_small_entries.push(
            GraphEntry::create_with_partition(
                large_graph_entry.key.clone(), current_neighbor_list, current_partition_id)
        );
        res_small_entries
    }

    fn fetch_page(&self, page_num: PageNum) -> Page {
        // Fetch a page from the file object according to the page number.
        // Step 1. Read the bytes.
        let page_bytes =
            self.comm_file.read((page_num * PAGE_SIZE) as u64, PAGE_SIZE as u64).unwrap();
        // Step 2. Decode it and return the value.
        Page::from_bytes(&page_bytes).unwrap()
    }

    pub(crate) fn fetch_kv_block(&self, page_num: PageNum) -> KVCommBlock {
        // Fetch a kv community block according to the block number, same as fetch_block().
        // Fetch a community block according to its block number.
        // Step 1, load the page in bytes.
        let block_bytes =
            self.fetch_page(page_num).data;
        // Step 2, decode the block from bytes.
        let decoded_comm_block =
            KVCommBlock::from_bytes(&block_bytes).unwrap();
        decoded_comm_block
    }

    pub(crate) fn fetch_csr_block_through_link(&self, start_page_num: PageNum) -> CSRCommBlock {
        // Fetch a community block according to its start block number.
        let mut csr_bytes = vec![];
        // Step 1, load the first page and collect the data.
        let mut current_page = self.fetch_page(start_page_num);
        csr_bytes.extend_from_slice(&current_page.data);
        // Step 2, Check whether it is the last page.
        while current_page.has_next == 1 {
            current_page = self.fetch_page(current_page.next_page_num);
            csr_bytes.extend_from_slice(&current_page.data);
        }

        // Step 3, decode the block from bytes and return value.
        CSRCommBlock::from_bytes(&csr_bytes).unwrap()
    }

    pub(crate) fn fetch_csr_block_through_nums(&self, page_nums: &Vec<PageNum>) -> CSRCommBlock {
        let mut csr_pages = vec![];
        for page_num in page_nums {
            csr_pages.push(self.fetch_page(*page_num));
        }
        CSRCommBlock::decode_from_pages(&csr_pages).unwrap()
    }

    /// Build graph entries from the inserted graph.
    pub(crate) fn build_graph_entries_group(inserted_graph: &Graph)
                                            -> BTreeMap<VInt, GraphEntry<V32>> {
        // Generate the graph entries according to the inserted graph.
        let mut res_entries =
            BTreeMap::<VInt, GraphEntry<V32>>::new();
        for (vertex_id, (v, neighbors)) in &inserted_graph.adj_map {
            let entry = GraphEntry::create(v.clone(), neighbors.clone());
            res_entries.insert(*vertex_id, entry);
        }
        res_entries
    }

    /// Build the community bucket from a community (i.e., a graph instance).
    pub(crate) fn build_bucket_from_community(community: &Graph, bucket_file_name: &str) -> CommBucket {
        // Build a bucket from a graph, is very important to work as an example.
        // Step 1, Generate the un-flushed entries.
        let un_flushed_entries = Self::build_graph_entries_group(
            community);

        // Step 2, Build community index based on graph entry.
        // Step 2-1, Walk the entries and group them.
        // Build the vertex group.
        let mut group_size = 0usize;  // current size in each group.
        let mut vertex_group = vec![];
        let mut current_vertex_group = vec![];
        for (vertex_id, large_entry) in &un_flushed_entries {
            // Compute the size of this vid.
            let large_entry_size = large_entry.get_csr_size();

            if large_entry_size > APP_BLOCK_SIZE - size_of::<u32>() {
                // For sure, need a new block for this vertex.
                // Collect the old block.
                if group_size > 0 {
                    vertex_group.push(current_vertex_group);
                    current_vertex_group = vec![];
                    group_size = 0;
                }
                // Create a new group and continue.
                vertex_group.push(vec![*vertex_id]);
                continue;
            } else if group_size + large_entry_size > APP_BLOCK_SIZE {
                // For sure, need a new block.
                // Collect the old block.
                vertex_group.push(current_vertex_group);
                // Create a new group.
                current_vertex_group = vec![];
                group_size = 0;
            }
            group_size += large_entry_size;
            current_vertex_group.push(*vertex_id);
        }
        vertex_group.push(current_vertex_group);

        // Process each vertex group.
        // Create a new Community index.
        let mut comm_index = SkipListCommIndex::new();

        // Build page info.
        let mut page_num_map = HashMap::<PageNum, Page>::new();
        for v_group in vertex_group {
            // Insert this interval into community index.
            comm_index.insert_interval(v_group[0].clone());
            // A vertex maps to several blocks.
            // Step 1. Generate a CSR block.
            let new_csr_block = if v_group.len() == 1 {
                CSRCommBlock::from_entries(
                    &vec![un_flushed_entries.get(&v_group[0]).unwrap().clone()]
                )
            } else {
                let mut entry_list = vec![];
                for v_in in &v_group {
                    entry_list.push(un_flushed_entries.get(v_in).unwrap().clone());
                }
                CSRCommBlock::from_entries(&entry_list)
            };

            // Step 2. Serialize this csr block into pages.
            let page_list = new_csr_block.encode_to_pages();

            // Step 3. build the community index.
            for _ in 0..page_list.len() {
                comm_index.append(v_group[0].clone());
            }

            // Step 4. Complete the metadata of these pages and bind with page number.
            let allocated_page_nums = comm_index.get_csr_block(
                &v_group[0]
            );
            for (page_index, mut new_page) in page_list.into_iter().enumerate() {
                if page_index < allocated_page_nums.len() - 1 {
                    // Not the last page.
                    new_page.has_next = 1u8;
                    new_page.next_page_num = allocated_page_nums[page_index + 1];
                    new_page.block_offset = page_index;
                } else {
                    new_page.has_next = 0u8;
                    new_page.block_offset = page_index;
                }
                page_num_map.insert(allocated_page_nums[page_index], new_page);
            }
            // Step 5. Finally, do not forget the kv part.
            let new_kv_block = KVCommBlock {
                block_type: 1,
                max_vertex_id: new_csr_block.max_vertex_id,
                min_vertex_id: new_csr_block.min_vertex_id,
                unsorted_entry_list: Default::default(),
            };
            let kv_pages = new_kv_block.encode_to_pages();
            let kv_page_num = comm_index.get_kv_block(&v_group[0]);
            page_num_map.insert(kv_page_num, kv_pages[0].clone());
        }

        // Flush them all.
        // Step 1. Prepare Bucket file.
        let path = Path::new(bucket_file_name);

        // Ensure the file and directory exist (cleanup if necessary)
        if path.exists() {
            std::fs::remove_file(path).expect("File remove failed!");
        }
        std::fs::create_dir_all(path.parent().unwrap()).expect("Dir make failed!");

        // Step 2. Create the file object.
        let file_obj = FileObject::create_empty_file(&path).unwrap();

        // Flush the sorted part.
        for (offset, page) in page_num_map.into_iter() {
            match file_obj.flush(page.encode(), (offset * PAGE_SIZE) as u64) {
                Ok(_) => {}
                Err(_) => {}
            }
        }

        // Return value.
        CommBucket {
            comm_index,
            comm_file: file_obj,
            comm_cache: CommCache::create(),
        }
    }

    /// Generate the graph snapshot from the file.
    pub(crate) fn generate_snapshot(&self) -> GraphSnapshot {
        // Similar to recovering from the object file.
        // Collect all the graph entries.
        let mut graph_entries = Vec::new();
        // 1. From block cache.
        for (_, entry_group) in self.comm_cache.get_buffer_ref().clone() {
            graph_entries.extend(entry_group);
        }
        // 2. From all the blocks.
        for index_item in &self.comm_index.0 {
            let (kv_page_num, csr_page_nums) = index_item.value().clone();
            let csr_block = self.fetch_csr_block_through_nums(&csr_page_nums);
            let kv_block = self.fetch_kv_block(kv_page_num);
            graph_entries.extend(csr_block.generate_entries());
            graph_entries.extend(kv_block.unsorted_entry_list);
        }
        GraphSnapshot::from_entry_list(graph_entries.into_iter())
    }

    /// Find the valid vertex of given vertex in this community bucket.
    pub fn find_vertex(&self, vertex_id: &VInt) -> Option<V32> {
        // Find the bucket cache firstly.
        return match self.comm_cache.get_from_cache(vertex_id) {
            None => {
                // Cache miss.
                // println!("Cache miss for vertex: {}", vertex_id);
                // Step 0, Find the neighbors from the buffer.
                match self.comm_cache.get_from_buffer(vertex_id) {
                    None => {
                        // Nothing in the buffer.
                    }
                    Some(buffered_entry_list) => {
                        for g_entry in &buffered_entry_list {
                            return Some(g_entry.key.clone());
                        }
                    }
                }

                // Step 1, Find the community block through the community index.
                let target_kv_block_num = self.comm_index.get_kv_block(vertex_id);
                let target_csr_block_nums = self.comm_index.get_csr_block(vertex_id);

                // Step 2, Load the community block and search the neighbors through the binary search.
                let mut target_blocks = vec![];
                // println!("Unsorted page number: {}", target_kv_block_num);
                let kv_block = self.fetch_kv_block(target_kv_block_num);

                let csr_block = self.fetch_csr_block_through_nums(&target_csr_block_nums);

                target_blocks.push(CSR(csr_block));
                target_blocks.push(KV(kv_block));

                // Step 3, Filter the invalid neighbors, i.e., the neighbors marked as deleted.
                for block in &target_blocks {
                    match block {
                        CSR(csr_block) => {
                            for g_entry in csr_block.generate_entries() {
                                if g_entry.key.vertex_id == *vertex_id {
                                    return Some(g_entry.key.clone());
                                }
                            }
                        }
                        KV(kv_block) => {
                            for g_entry in &kv_block.unsorted_entry_list {
                                if g_entry.key.vertex_id == *vertex_id {
                                    return Some(g_entry.key.clone());
                                }
                            }
                        }
                    }
                }
                None
            }
            Some(entry_list) => {
                // Cache hit.
                // println!("Cache hit for vertex: {}", vertex_id);
                for g_entry in &entry_list {
                    if g_entry.key.vertex_id == *vertex_id {
                        return Some(g_entry.key.clone());
                    }
                }
                None
            }
        }
    }

    /// Find the neighbors of given vertex in this community bucket.
    pub(crate) fn find_neighbors(&mut self, vertex_id: &VInt) -> Vec<V32> {
        // Find the neighbors of a specific vertex.
        let mut result_neighbors = Vec::<V32>::new();
        // Find the bucket cache firstly.
        match self.comm_cache.get_from_cache(vertex_id) {
            None => {
                // Cache miss.
                // println!("Cache miss for vertex: {}", vertex_id);
                // Step 0, Find the neighbors from the buffer.
                match self.comm_cache.get_from_buffer(vertex_id) {
                    None => {
                        // Nothing in the buffer.
                    }
                    Some(buffered_entry_list) => {
                        for g_entry in &buffered_entry_list {
                            for neighbor in &g_entry.neighbors {
                                result_neighbors.push(neighbor.clone())
                            }
                        }
                    }
                }

                // Step 1, Find the community block through the community index.
                let target_kv_block_num = self.comm_index.get_kv_block(vertex_id);
                let target_csr_block_nums = self.comm_index.get_csr_block(vertex_id);

                // Step 2, Load the community block and search the neighbors through the binary search.
                let mut target_blocks = vec![];
                // println!("Unsorted page number: {}", target_kv_block_num);
                let kv_block = self.fetch_kv_block(target_kv_block_num);

                let csr_block = self.fetch_csr_block_through_nums(&target_csr_block_nums);

                target_blocks.push(CSR(csr_block));
                target_blocks.push(KV(kv_block));

                // Step 3, Filter the invalid neighbors, i.e., the neighbors marked as deleted.
                for block in &target_blocks {
                    match block {
                        CSR(csr_block) => {
                            for g_entry in csr_block.generate_entries() {
                                // Push this entry into the bucket cache.
                                self.comm_cache.insert_to_cache(vec![g_entry.clone()]);
                                if g_entry.key.vertex_id == *vertex_id {
                                    for neighbor in &g_entry.neighbors {
                                        result_neighbors.push(neighbor.clone())
                                    }
                                }
                            }
                        }
                        KV(kv_block) => {
                            for g_entry in &kv_block.unsorted_entry_list{
                                // Push this entry into the bucket cache.
                                self.comm_cache.insert_to_cache(vec![g_entry.clone()]);
                                if g_entry.key.vertex_id == *vertex_id {
                                    for neighbor in &g_entry.neighbors {
                                        result_neighbors.push(neighbor.clone())
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Some(entry_list) => {
                // Cache hit.
                // println!("Cache hit for vertex: {}", vertex_id);
                for g_entry in &entry_list{
                    if g_entry.key.vertex_id == *vertex_id {
                        for neighbor in &g_entry.neighbors {
                            result_neighbors.push(neighbor.clone())
                        }
                    }
                }
            }
        }
        CompactController::execute_compact_neighbors(&result_neighbors)
    }

    /// Rebuild the bucket from the obj file.
    pub(crate) fn build_bucket_from_obj_file(bucket_file_name: &str) -> CommBucket {
        // The problem is to rebuild the community index.
        // Step 1, check the count of community blocks.
        let file_metadata = metadata(bucket_file_name).unwrap();
        let page_count = file_metadata.len() as usize / PAGE_SIZE;

        // Step 2, Open the object file.
        let path = Path::new(bucket_file_name);

        // Create the file object.
        let file_obj = FileObject::open(&path).unwrap();
        let mut comm_index = SkipListCommIndex::new();

        // Fetch all the pages and rebuild all the blocks.
        for block_num in 0..page_count {
            let page = Page::from_bytes(&file_obj.read(
                (block_num * PAGE_SIZE) as u64, PAGE_SIZE as u64).unwrap()).unwrap();
            if page.block_type == 1 {
                // Unsorted block.
                let kv_block = KVCommBlock::decode_from_pages(&vec![page]).unwrap();
                comm_index.insert_page_rebuild(
                    kv_block.min_vertex_id,
                    block_num,
                    1
                )
            } else if page.block_type == 0 && page.block_offset == 0 {
                // Walk through the linked list to get all the pages.
                let mut csr_pages = vec![];
                let mut current_page = page;
                csr_pages.push(current_page.clone());
                let mut csr_page_numbers = vec![];
                csr_page_numbers.push(block_num);

                loop {
                    if current_page.has_next == 0 {
                        break;
                    }
                    let next_page_number = current_page.next_page_num;
                    current_page = Page::from_bytes(&file_obj.read(
                        (next_page_number * PAGE_SIZE) as u64, PAGE_SIZE as u64).unwrap()).unwrap();
                    csr_pages.push(current_page.clone());
                    csr_page_numbers.push(next_page_number);
                }

                let csr_block = CSRCommBlock::decode_from_pages(&csr_pages).unwrap();
                for csr_pn in csr_page_numbers {
                    comm_index.insert_page_rebuild(
                        csr_block.min_vertex_id,
                        csr_pn,
                        0
                    )
                }
            }
        }

        // Return value.
        CommBucket {
            comm_index,
            comm_file: file_obj,
            comm_cache: CommCache::create(),
        }
    }

    /// Flush the data in cache to the community blocks.
    pub(crate) fn flush(&mut self) -> Result<()> {
        // Main logic:
        // Step 1: Group the data in cache with their key according to the range.
        // Apply the stream programming to group the entries.
        let index_interval = self.comm_cache.get_buffer_ref().clone().into_iter().fold(
            BTreeMap::<VInt, Vec<GraphEntry<V32>>>::new(),
            |mut acc, item| {
                // If the keys of the items are located in the same space, put them together.
                // Locate the min vertex.
                let target_entry = self.comm_index.0.range(0..item.0 + 1).next_back().unwrap();
                let target_key = target_entry.key();
                if acc.contains_key(target_key) {
                    acc.get_mut(target_key).unwrap().extend_from_slice(&item.1);
                } else {
                    acc.insert(*target_key, item.1.clone());
                }
                acc
            }
        );
        for (min_vertex_id, group) in index_interval {
            // Compute the size of this group.
            let mut group_size = 0usize;
            for entry in &group {
                group_size += entry.get_entry_size();
            }
            // Load the unsorted block.
            let kv_page_num = self.comm_index.get_kv_block(&min_vertex_id);
            let kv_block_page = self.fetch_page(kv_page_num);
            let mut kv_block =
                KVCommBlock::decode_from_pages(&vec![kv_block_page.clone()]).unwrap();
            // Check whether the unsorted block is enough.
            if kv_block_page.used_size + group_size < BLOCK_SIZE {

                // If enough, just insert it into KV block and flush back.
                for entry in group {
                    kv_block.unsorted_entry_list.push(entry);
                }
                // Flush back.
                let kv_pages = kv_block.encode_to_pages();
                let flushed_page = kv_pages[0].clone();
                // println!("new page size: {}, flushed into page num: {}", flushed_page.used_size, kv_page_num);
                match self.comm_file.flush(flushed_page.encode().to_vec(), (kv_page_num * PAGE_SIZE) as u64) {
                    Ok(_) => {}
                    Err(_) => {}
                }
            } else {
                // If not enough, merge unsorted block and data in cache to sorted part.
                // Load all the sorted blocks.
                let csr_block_nums = self.comm_index.get_csr_block(&min_vertex_id);
                let csr_block = self.fetch_csr_block_through_nums(&csr_block_nums);
                // Collect all entries.
                let mut new_arranged_entries = Vec::<GraphEntry<V32>>::new();

                // 1, Collect the entries in unsorted block.
                for entry in kv_block.unsorted_entry_list {
                    new_arranged_entries.push(entry);
                }
                // 2, Collect the entries in sorted_block.
                new_arranged_entries.append(&mut csr_block.generate_entries());

                // 3, Collect the entries from the buffer.
                new_arranged_entries.extend_from_slice(&group);

                // 4, Perform compaction.
                let new_arranged_entries_comp = CompactController::execute_compaction(&new_arranged_entries);

                // Generate a new csr block.
                let new_csr_block = CSRCommBlock::from_entries(&new_arranged_entries_comp);
                let new_pages = new_csr_block.encode_to_pages();

                // Arrange or shrink block if needed.
                let origin_block_count = self.comm_index.get_csr_block(&min_vertex_id);
                if origin_block_count.len() < new_pages.len() {
                    // Arrange new blocks.
                    for _ in origin_block_count.len()..new_pages.len() {
                        self.comm_index.append(min_vertex_id);
                    }
                }
                if origin_block_count.len() > new_pages.len() {
                    // Perform shrink.
                    for _ in new_pages.len()..origin_block_count.len() {
                        self.comm_index.shrink(min_vertex_id);
                    }
                }
                // Complete the metadata of these pages and bind with page number.
                let mut page_num_map = HashMap::<PageNum, Page>::new();
                let allocated_page_nums = self.comm_index.get_csr_block(
                    &min_vertex_id
                );
                for (page_index, mut new_page) in new_pages.into_iter().enumerate() {
                    if page_index < allocated_page_nums.len() - 1 {
                        // Not the last page.
                        new_page.has_next = 1u8;
                        new_page.next_page_num = allocated_page_nums[page_index + 1];
                        new_page.block_offset = page_index;
                    } else {
                        new_page.has_next = 0u8;
                        new_page.block_offset = page_index;
                    }
                    page_num_map.insert(allocated_page_nums[page_index], new_page);
                }

                // Create new blocks and flush.
                for (page_num, page) in page_num_map {
                    match self.comm_file.flush(page.encode().to_vec(), (page_num * PAGE_SIZE) as u64) {
                        Ok(_) => {}
                        Err(_) => {println!("An error happen");}
                    }
                }

                // Flush the unsorted block.
                let comm_block = KVCommBlock {
                    block_type: 1,
                    max_vertex_id: kv_block.max_vertex_id,
                    min_vertex_id: kv_block.min_vertex_id,
                    unsorted_entry_list: Default::default(),
                };

                match self.comm_file.flush(comm_block.encode_to_pages()[0].encode().to_vec(), (kv_page_num * PAGE_SIZE) as u64) {
                    Ok(_) => {}
                    Err(_) => {}
                }
            }
        }
        // Finally, clear the cache.
        self.comm_cache.clear_buffer();
        Ok(())
    }

    /// Display the content in cache.
    pub(crate) fn display_cache(&self) {
        for (v_id, entries) in &self.comm_cache.bucket_cache.cached_entries {
            print!("Vertex ID: {}, Cached Entries: ", v_id);
            for entry in entries {
                print!("{} ", entry);
            }
            println!();
        }
    }

    pub(crate) fn insert_entries_batch(&mut self, inserted_entries: Vec<GraphEntry<V32>>) {
        // A complex logic: Batch insert entries to this bucket.
        // Step 1, Check whether the community cache is full.
        let mut total_bytes = 0usize;
        for entry in &inserted_entries {
            total_bytes += entry.get_entry_size();
        }
        if self.comm_cache.get_buffer_size() + total_bytes < MAX_CACHE_SIZE {
            // If the cache is not full, insert them to cache.
            self.comm_cache.insert_to_buffer(inserted_entries);
        } else {
            // Firstly insert the entries into cache.
            // Will be flushed together with the old data.
            self.comm_cache.insert_to_buffer(inserted_entries);
            // If the cache is full, or not enough for the entry, perform flush().
            match self.flush() {
                Ok(_) => {}
                Err(_) => {}
            }
        }
    }

    // Load the graph from this community.
    pub fn load_community(&self) -> Graph {
        // Step 1. Load the entries from the write buffer (Perform copy).
        let mut all_entries = Vec::new();
        for (_, entries) in self.comm_cache.get_buffer_ref() {
            all_entries.append(&mut entries.clone());
        }
        // Step 2. Load all the entries from the disk.
        for index_entry in self.comm_index.0.iter() {
            let (kv_page_num, csr_page_nums) = index_entry.value().clone();
            let csr_block = self.fetch_csr_block_through_nums(&csr_page_nums);
            let kv_block = self.fetch_kv_block(kv_page_num);
            for entry in csr_block.generate_entries() {
                all_entries.push(entry);
            }
            for entry in kv_block.unsorted_entry_list {
                all_entries.push(entry);
            }
        }
        // Step 3. Gather them and build the graph.
        // Step 3.1 - Gather all the vertices.
        let mut adj_map = BTreeMap::<VInt, (Vertex<u32>, Vec<Vertex<u32>>)>::new();
        let mut v_size = 0u32;
        let mut e_size = 0u32;
        for entry in &all_entries {
            // Processing vertex.
            if !adj_map.contains_key(&entry.key.vertex_id) && entry.key.tomb == 0 {
                adj_map.insert(entry.key.vertex_id, (entry.key.clone(), vec![]));
                v_size += 1;
            }
        }

        // Step 3.2 - Gather all the edges.
        for entry in &all_entries {
            let vertex_id  = entry.key.vertex_id;
            for neighbor in &entry.neighbors {
                // Check the target exists in this graph.
                if adj_map.contains_key(&neighbor.vertex_id) && neighbor.tomb == 0 {
                    adj_map.get_mut(&vertex_id).unwrap().1.push(neighbor.clone());
                    e_size += 1;
                }
            }
        }
        // Step 3.3 - Return the graph.
        Graph {
            adj_map,
            v_size,
            e_size
        }
    }

    /// Load the whole information in this bucket to a graph.
    pub fn load_bucket(&self) -> Graph {
        // Step 1. Load the entries from the write buffer (Perform copy).
        let mut all_entries = Vec::new();
        for (_, entries) in self.comm_cache.get_buffer_ref() {
            all_entries.append(&mut entries.clone());
        }
        // Step 2. Load all the entries from the disk.
        for index_entry in self.comm_index.0.iter() {
            let (kv_page_num, csr_page_nums) = index_entry.value().clone();
            let csr_block = self.fetch_csr_block_through_nums(&csr_page_nums);
            let kv_block = self.fetch_kv_block(kv_page_num);
            for entry in csr_block.generate_entries() {
                all_entries.push(entry);
            }
            for entry in kv_block.unsorted_entry_list {
                all_entries.push(entry);
            }
        }
        // Step 3. Gather them and build the graph.
        // Step 3.1 - Gather all the vertices.
        let mut adj_map = BTreeMap::<VInt, (Vertex<u32>, Vec<Vertex<u32>>)>::new();
        let mut v_size = 0u32;
        let mut e_size = 0u32;
        for entry in &all_entries {
            // Processing vertex.
            if !adj_map.contains_key(&entry.key.vertex_id) && entry.key.tomb == 0 {
                adj_map.insert(entry.key.vertex_id, (entry.key.clone(), vec![]));
                v_size += 1;
            }
        }

        // Step 3.2 - Gather all the edges.
        for entry in &all_entries {
            let vertex_id  = entry.key.vertex_id;
            for neighbor in &entry.neighbors {
                // No more check, just insert it.
                adj_map.get_mut(&vertex_id).unwrap().1.push(neighbor.clone());
                e_size += 1;
            }
        }

        // Step 3.3 - Return the graph.
        Graph {
            adj_map,
            v_size,
            e_size
        }
    }

    pub(crate) fn insert_entry(&mut self, g_entry: GraphEntry<V32>) {
        // A specific case of batch insertion, just make the inserted
        // entry a vector.
        self.insert_entries_batch(vec![g_entry])
    }

    /// Display the bucket.
    pub(crate) fn display_bucket(&self) {
        // Print the Unsorted block.
        for index_entry in self.comm_index.0.iter() {
            let (unsorted_page_num, csr_page_nums) = index_entry.value().clone();
            let interval_start = index_entry.key().clone();
            let csr_block = self.fetch_csr_block_through_nums(&csr_page_nums);
            let kv_block = self.fetch_kv_block(unsorted_page_num);
            println!("Interval Start: {}", interval_start);
            println!("Unsorted Block, Min ID: {}, Max ID: {}", kv_block.min_vertex_id, kv_block.max_vertex_id);
            println!("CSR Block, Min ID: {}, Max ID: {}", csr_block.min_vertex_id, csr_block.max_vertex_id);
        }
    }
}

#[cfg(test)]
mod test_comm_bucket {
    use std::collections::{BTreeMap, HashMap, HashSet};
    use std::fs;
    use std::iter::FromIterator;
    use std::path::Path;
    use std::time::Instant;

    use rand::Rng;
    use rand::seq::SliceRandom;

    use crate::bucket::{CommBucket, FileObject};
    use crate::bucket::block::{CommBlock, CSRCommBlock, KVCommBlock};
    use crate::bucket::block::CommBlock::{CSR, KV};
    use crate::bucket::page::Page;
    use crate::config::PAGE_SIZE;
    use crate::graph::{Graph, VInt};
    use crate::louvain::Louvain;
    use crate::types::{Decode, Encode, GraphEntry, V32, Vertex};

    use super::block::test_comm_block::prepare_example_comm_graph;
    use super::TreeBucket;

    /// Generate community blocks as example.
    pub(crate) fn generate_example(is_csr: bool) -> CommBlock {

        let u1 = Vertex::new(23u32, 0u8);
        let v1 = Vertex::new_successor(24u32);
        let v2 = Vertex::new_successor(25u32);
        let v3 = Vertex::new_successor(26u32);
        let entry1 = GraphEntry::<V32>::create(u1, vec![v1, v2, v3]);

        let u2 = Vertex::new(26u32, 0u8);
        let v4 = Vertex::new_successor(24u32);
        let v5 = Vertex::new_successor(25u32);
        let v6 = Vertex::new_successor(26u32);
        let entry2 = GraphEntry::<V32>::create(u2, vec![v4, v5, v6]);

        let mut entry_list = Vec::<GraphEntry<V32>>::new();
        entry_list.push(entry1);
        entry_list.push(entry2);

        if !is_csr {
            KV(KVCommBlock {
                block_type: 1,
                min_vertex_id: 23u32,
                max_vertex_id: 26u32,
                unsorted_entry_list: entry_list
            })
        } else {
            CSR(CSRCommBlock::from_entries(
                &entry_list.into_iter().collect()
            ))
        }
    }

    fn prepare_csr_in_file(bucket_path: &str) -> CSRCommBlock {
        let path = Path::new(bucket_path);

        // Ensure the file and directory exist (cleanup if necessary)
        if path.exists() {
            fs::remove_file(path).expect("File remove failed!");
        }
        fs::create_dir_all(path.parent().unwrap()).expect("Dir make failed!");


        let comm_block = match generate_example(true) {
            KV(_) => {
                panic!("An error happen.")
            }
            CSR(block) => {
                block
            }
        };
        let block_bytes = comm_block.encode().to_vec();
        let page = Page {
            used_size: block_bytes.len(),
            block_type: 0,
            block_offset: 0,
            has_next: 0,
            next_page_num: 0,
            data: block_bytes,
        };
        let initial_data = b"Initial data".to_vec();
        let path = Path::new(bucket_path);
        let file_obj = FileObject::create(&path, initial_data.clone()).unwrap();

        let page_bytes = page.encode();
        // assert_eq!(page_bytes.len(), PAGE_SIZE);
        file_obj.flush(page_bytes, PAGE_SIZE as u64).unwrap();
        comm_block
    }

    #[test]
    fn test_load_comm_blocks() {
        let comm_block = prepare_csr_in_file("lsm.db/test_file.txt");
        let path = Path::new("lsm.db/test_file.txt");
        let file_obj = FileObject::open(&path).unwrap();
        let comm_block_bytes = file_obj.read(PAGE_SIZE as u64, PAGE_SIZE as u64).unwrap();
        // println!("{:?}", comm_block_bytes);
        let page_load = Page::from_bytes(&comm_block_bytes).unwrap();
        let comm_block_load = CSRCommBlock::from_bytes(
            &page_load.data
        ).unwrap();

        // Check the meta data.
        assert_eq!(comm_block_load.block_type, comm_block.block_type);
        assert_eq!(comm_block_load.max_vertex_id, comm_block.max_vertex_id);
        assert_eq!(comm_block_load.min_vertex_id, comm_block.min_vertex_id);
        assert_eq!(comm_block_load.vertex_count, comm_block.vertex_count);

        // Check the content.
        // 1, Vertex list.
        assert_eq!(comm_block_load.vertex_list.len(), comm_block.vertex_list.len());
        for ((vertex, offset), (vertex_gt, offset_gt)) in
            comm_block_load.vertex_list.iter().zip(comm_block.vertex_list.iter()) {
            assert_eq!(vertex, vertex_gt);
            assert_eq!(offset_gt, offset);
        }

        // 2, Edge list.
        assert_eq!(comm_block_load.edge_list.len(), comm_block.edge_list.len());
        for (neighbor, neighbor_gt) in comm_block_load.edge_list.iter().zip(comm_block.edge_list.iter()) {
            assert_eq!(neighbor_gt, neighbor);
        }
    }

    #[test]
    fn test_comm_detection() {
        let graph_hprd = Graph::from_graph_file("data/email-enron.graph", true);
        let (mut lg, vid_arr) = graph_hprd.generate_louvain_graph();
        let louvain = Louvain::new(&mut lg);
        let (hierarchy, modularities) = louvain.run();
        print!("Layer Count: {}", hierarchy.len());
        // Print the hierarchy.
        // Perform Louvain.
        let mut target_layer = hierarchy.len() - 1;
        let mut target_layer_comm = Vec::<Vec<usize>>::new();
        let mut comm_structure_layer = Vec::new();
        let app_comm_num = (graph_hprd.v_size as f64).sqrt().floor() as usize;
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
                println!("Choose Layer: {}", layer);
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

        println!("VCount: {}, Total: {}", graph_hprd.v_size, comm_structure.iter().fold(0usize, |mut item, arr: &Vec<u32>| {item += arr.len(); item}));
        print!("Comm Count: {}, Comm Structure: {:?}", comm_structure.len(), comm_structure);
        println!("Modularities: {:?}", modularities);
    }

    #[test]
    fn test_find_query_graph() {
        for tested_data_set in vec!["hprd", "yeast"] {
            let dataset_path = "data/".to_owned() + tested_data_set + ".graph";
            let obj_path = "lsm.db/".to_owned() + tested_data_set + "_comm_for_bucket_test.txt";
            let g_from_graph = Graph::from_graph_file(&dataset_path, true);
            let mut bucket = CommBucket::build_bucket_from_community(&g_from_graph,
                                                                     &obj_path);

            for tested_vid in 0..100 {
                // Performing the test.
                let res = bucket.find_neighbors(&tested_vid);

                // Check the neighbors.
                let ground_truth = g_from_graph.get_neighbor(&tested_vid);

                // V \in U and |V| = |U| => V = U (In Algebra.)
                for v in &res {
                    assert!(ground_truth.contains(v));
                }
                assert_eq!(ground_truth.len(), res.len());
            }
        }
        println!("Find Neighbor test pass!");
        test_load_from_obj_file();
    }

    #[test]
    fn test_generate_snapshot() {
        let graph_hprd = Graph::from_graph_file("data/oregon.graph", true);
        let bucket = CommBucket::build_bucket_from_community(&graph_hprd,
                                                             "lsm.db/oregon_comm_2.txt");
        let hprd_snapshot = bucket.generate_snapshot();
        hprd_snapshot.print_graph();
    }

    fn test_load_from_obj_file() {
        let mut bucket = CommBucket::build_bucket_from_obj_file("lsm.db/hprd_comm_for_bucket_test.txt");
        let g_from_graph = Graph::from_graph_file("data/hprd.graph", true);

        for tested_vid in 0..10000 {
            // Performing the test.
            let res = bucket.find_neighbors(&tested_vid);

            // Check the neighbors.
            let ground_truth = g_from_graph.get_neighbor(&tested_vid);

            // V \in U and |V| = |U| => V = U (In Algebra.)
            for v in &res {
                assert!(ground_truth.contains(v));
            }
            assert_eq!(ground_truth.len(), res.len());
        }
        println!("Rebuild, Find Neighbor test pass!");
    }

    /// Test the insertion of the bucket through a ``one-by-one'' manner.
    #[test]
    fn test_insert_one_by_one() {
        // Prepare the random number generator.
        let mut rng = rand::thread_rng();
        // Load the graph.
        let mut graph_hprd = Graph::from_graph_file("data/oregon.graph", true);
        let mut bucket = CommBucket::build_bucket_from_community(&graph_hprd,
                                                                 "lsm.db/oregon_comm_one_by_one.txt");
        // Determine the edges can be added.
        let vertex_count = graph_hprd.v_size;
        let mut updated_entry_vec_map = BTreeMap::<VInt, (V32, Vec<V32>)>::new();
        let mut updated_entries = vec![];
        for vertex_id in 0..vertex_count {
            updated_entry_vec_map.insert(vertex_id, (V32::new_vertex(vertex_id), vec![]));
        }

        let mut inserted_edge_count = 0u32;
        let mut deleted_edge_count = 0u32;
        for vertex_id in 0..vertex_count {
            let mut added_count = 0;
            let remove_count = 1;
            loop {
                let added_neighbor = rng.gen_range(0..vertex_count);
                let mut neighbor_exists = false;
                // Validate this neighbor is existing.
                for neighbor in &graph_hprd.get_neighbor(&vertex_id) {
                    if neighbor.vertex_id == added_neighbor && neighbor.direction_tag == 1 {
                        neighbor_exists = true;
                        break;
                    }
                }
                if !neighbor_exists {
                    updated_entry_vec_map.get_mut(&vertex_id).unwrap().1.push(V32::new_successor(added_neighbor));
                    updated_entry_vec_map.get_mut(&added_neighbor).unwrap().1.push(V32::new_predecessor(vertex_id));
                    added_count += 1;
                    inserted_edge_count += 1;
                }
                if added_count > 10 {
                    break;
                }
            }

            for _ in 0..remove_count {
                match graph_hprd.get_successor(&vertex_id).choose(&mut rng) {
                    None => {
                        // println!("This vertex has no successors.");
                        break;
                    }
                    Some(removed_neighbor) => {
                        updated_entry_vec_map.get_mut(&vertex_id).unwrap().1.push(V32::new_tomb(removed_neighbor.vertex_id, 1u8));
                        updated_entry_vec_map.get_mut(&removed_neighbor.vertex_id).unwrap().1.push(V32::new_tomb(vertex_id, 2u8));
                        deleted_edge_count += 1;
                    }
                }
            }
        }
        // Collect the inserted entries.
        let start = Instant::now();
        for (_, (v, neighbors)) in &updated_entry_vec_map {
            // Update the ground truth.
            for neighbor in neighbors {
                let inserted_g_entry = GraphEntry::create(v.clone(), vec![neighbor.clone()]);
                // Perform insert.
                bucket.insert_entry(inserted_g_entry);
                // thread::sleep(Duration::from_millis(1));
                if neighbor.tomb == 0 && neighbor.direction_tag == 1 {
                    // Insert edge.
                    graph_hprd.insert_edge(v.vertex_id, neighbor.vertex_id);
                } else if neighbor.tomb == 1 && neighbor.direction_tag == 1 {
                    // Delete edge.
                    graph_hprd.remove_edge(&v.vertex_id, &neighbor.vertex_id);
                }
            }
            // Note that values here are moved.
            updated_entries.push(GraphEntry::create(v.clone(), neighbors.clone()));
        }

        let duration = start.elapsed();
        println!("Insert {} edges in {:?}.", inserted_edge_count, duration);
        println!("Delete {} edges in {:?}.", deleted_edge_count, duration);

        // Valid the neighbors.
        // Validate the correctness of the edge insertion.
        let mut lost_edges = vec![];
        for tested_vid in 0..vertex_count + 10 {
            // Performing the test.
            let res = bucket.find_neighbors(&tested_vid);

            // Check the neighbors.
            let ground_truth = graph_hprd.get_neighbor(&tested_vid);

            // V \in U and |V| = |U| => V = U (In Algebra.)
            for v in &res {
                assert!(ground_truth.contains(&v));
            }

            // Display the results.
            if ground_truth.len() != res.len() {
                // Display the inserted edges.
                println!("Testing Vertex: {}", tested_vid);
                let inserted_edges_with_vertex = updated_entry_vec_map.get(&tested_vid).unwrap().clone();
                let inserted_neighbors = inserted_edges_with_vertex.1.iter().map(|item| (item.vertex_id, item.direction_tag)).collect::<Vec<_>>();
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
    }

    #[test]
    fn test_insert_batch() {
        // Prepare the random number generator.
        let mut rng = rand::thread_rng();
        // Load the graph.
        let mut graph_hprd = Graph::from_graph_file("data/oregon.graph", true);
        let mut bucket = CommBucket::build_bucket_from_community(&graph_hprd,
                                                                 "lsm.db/oregon_comm_1.txt");
        // Determine the edges can be added.
        let vertex_count = graph_hprd.v_size;
        let mut updated_entry_vec_map = BTreeMap::<VInt, (V32, Vec<V32>)>::new();
        let mut updated_entries = vec![];
        for vertex_id in 0..vertex_count {
            updated_entry_vec_map.insert(vertex_id, (V32::new_vertex(vertex_id), vec![]));
        }

        let mut inserted_edge_count = 0u32;
        let mut deleted_edge_count = 0u32;
        for vertex_id in 0..vertex_count {
            let mut added_count = 0;
            let remove_count = 20;
            loop {
                let added_neighbor = rng.gen_range(0..vertex_count);
                let mut neighbor_exists = false;
                // Validate this neighbor is existing.
                for neighbor in &graph_hprd.get_neighbor(&vertex_id) {
                    if neighbor.vertex_id == added_neighbor && neighbor.direction_tag == 1 {
                        neighbor_exists = true;
                        break;
                    }
                }
                if !neighbor_exists {
                    updated_entry_vec_map.get_mut(&vertex_id).unwrap().1.push(V32::new_successor(added_neighbor));
                    updated_entry_vec_map.get_mut(&added_neighbor).unwrap().1.push(V32::new_predecessor(vertex_id));
                    added_count += 1;
                    inserted_edge_count += 1;
                }
                if added_count > 10 {
                    break;
                }
            }

            for _ in 0..remove_count {
                match graph_hprd.get_successor(&vertex_id).choose(&mut rng) {
                    None => {
                        // println!("This vertex has no successors.");
                        break;
                    }
                    Some(removed_neighbor) => {
                        updated_entry_vec_map.get_mut(&vertex_id).unwrap().1.push(V32::new_tomb(removed_neighbor.vertex_id, 1u8));
                        updated_entry_vec_map.get_mut(&removed_neighbor.vertex_id).unwrap().1.push(V32::new_tomb(vertex_id, 2u8));
                        deleted_edge_count += 1;
                    }
                }
            }
        }
        // Collect the inserted entries.
        for (_, (v, neighbors)) in &updated_entry_vec_map {
            // Update the ground truth.
            for neighbor in neighbors {
                // thread::sleep(Duration::from_millis(1));
                if neighbor.tomb == 0 && neighbor.direction_tag == 1 {
                    // Insert edge.
                    graph_hprd.insert_edge(v.vertex_id, neighbor.vertex_id);
                } else if neighbor.tomb == 1 && neighbor.direction_tag == 1 {
                    // Delete edge.
                    graph_hprd.remove_edge(&v.vertex_id, &neighbor.vertex_id);
                }
            }
            // Note that values here are moved.
            updated_entries.push(GraphEntry::create(v.clone(), neighbors.clone()));
        }
        // Perform Insert.
        let start = Instant::now();
        // Insert one by one.
        for e in updated_entries {
            bucket.insert_entry(e);
        }
        // bucket.insert_entries_batch(updated_entries);
        let duration = start.elapsed();
        println!("Insert {} edges and delete {} edges in {:?}.",
                 inserted_edge_count, deleted_edge_count, duration);

        // Valid the neighbors.
        for tested_vid in 0..10000 {
            // Performing the test.
            println!("test vertex: {}", tested_vid);
            let res = bucket.find_neighbors(&tested_vid);

            // Check the neighbors.
            let ground_truth = graph_hprd.get_neighbor(&tested_vid);

            // V \in U and |V| = |U| => V = U (In Algebra.)
            for v in &res {
                assert!(ground_truth.contains(v));
            }
            assert_eq!(ground_truth.len(), res.len());
        }
        println!("Rebuild, Find Neighbor test pass!");
    }

    #[test]
    fn test_tree_bucket_create() {
        let comm_neighbor_vec = prepare_example_comm_graph();
        let comm_neighbor_treemap = BTreeMap::from_iter(comm_neighbor_vec);
        let example_tree_bucket = TreeBucket::build_bucket_from_neighbors(
            "example", 0, 0, &comm_neighbor_treemap);
        example_tree_bucket.print_bucket();
    }
}