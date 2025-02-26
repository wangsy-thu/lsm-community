use std::collections::BTreeMap;

use crate::bucket::CommBucket;
use crate::config::MAX_ENTRY_SIZE;
use crate::graph::VInt;
use crate::types::{GraphEntry, V32};

/// Compact the graph entries, like which in an LSM-Tree.
#[allow(dead_code)]
pub struct CompactController {

}

#[allow(dead_code)]
impl CompactController {

    /// Create a new compaction controller.
    pub fn new() -> Self {
        Self {}
    }

    /// Execute the compaction logic.
    pub fn execute_compaction(
        graph_entry_list: &Vec<GraphEntry<V32>>
    ) -> Vec<GraphEntry<V32>> {
        // The tasks in compaction is:
        // 1. Remove the redundant graph entry of the same key and value.
        // 2. Remove the deleted entries.

        // Main Steps.
        // S1. Group the entries by vertex id of the key.
        let entry_groups_vid :Vec<(V32, Vec<GraphEntry<V32>>)> = graph_entry_list.iter().fold (
            BTreeMap::<VInt, (V32, Vec<GraphEntry<V32>>)>::new(),
            |mut acc, item| {
                if acc.contains_key(&item.key.vertex_id) {
                    acc.get_mut(&item.key.vertex_id).unwrap().1.push(item.clone());
                } else {
                    acc.insert(item.key.vertex_id, (item.key.clone(), vec![item.clone()]));
                }
                acc
            }
        ).values().cloned().collect();

        // S2. For each group of entries, collect all the neighbors of the same vertex id.
        let mut entries_after_compact = vec![];
        for (compacted_vertex, entry_group_vid) in entry_groups_vid.into_iter() {
            // Gather all the vertices and neighbors.
            let mut vertex_list_of_vid = vec![];
            let mut neighbors_of_vid = vec![];
            for mut entry in entry_group_vid.into_iter() {
                vertex_list_of_vid.push(entry.key);
                neighbors_of_vid.append(&mut entry.neighbors);
            }
            // Find the key with maximal timestamp.
            let valid_key = vertex_list_of_vid.into_iter().max_by_key(|item| item.timestamp).unwrap();
            if valid_key.tomb != 2 {
                // S3. Group the neighbors by vertex id and direction tag.
                let neighbor_groups_nid :Vec<Vec<V32>> = neighbors_of_vid.into_iter().fold(
                    BTreeMap::<(VInt, u8), Vec<V32>>::new(),
                    |mut acc, item| {
                        if acc.contains_key(&(item.vertex_id, item.direction_tag)) {
                            acc.get_mut(&(item.vertex_id, item.direction_tag)).unwrap().push(item.clone());
                        } else {
                            acc.insert((item.vertex_id, item.direction_tag), vec![item.clone()]);
                        }
                        acc
                    }
                ).values().cloned().collect();

                // S4. For each group of each neighbor, find the neighbor with the maximal timestamp.
                let mut neighbors_after_compaction = vec![];
                for neighbor_group in neighbor_groups_nid.into_iter() {
                    let valid_neighbor = neighbor_group.into_iter().max_by_key(
                        |item| item.timestamp
                    ).unwrap();
                    // If the neighbor is deleted, not include it to the final result.
                    if valid_neighbor.tomb == 0 {
                        neighbors_after_compaction.push(valid_neighbor);
                    }
                }

                // S5. Construct the new graph entry.
                let compacted_entry = GraphEntry::create(
                    compacted_vertex, neighbors_after_compaction
                );
                let mut partitioned_entry_list =
                    CommBucket::graph_partition(compacted_entry, MAX_ENTRY_SIZE);
                entries_after_compact.append(&mut partitioned_entry_list);
            }
        }

        // S6. Return the new graph entry list.
        entries_after_compact
    }

    /// Compact all the neighbors of a given vertex.
    pub fn execute_compact_neighbors(neighbors: &Vec<V32>) -> Vec<V32> {
        // The main steps of this function is similar to step 3 and step 4 which in compaction.
        // S1. Group the neighbors by vertex id and direction tag.
        let neighbor_groups_nid :Vec<Vec<V32>> = neighbors.iter().fold(
            BTreeMap::<(VInt, u8), Vec<V32>>::new(),
            |mut acc, item| {
                if acc.contains_key(&(item.vertex_id, item.direction_tag)) {
                    acc.get_mut(&(item.vertex_id, item.direction_tag)).unwrap().push(item.clone());
                } else {
                    acc.insert((item.vertex_id, item.direction_tag), vec![item.clone()]);
                }
                acc
            }
        ).values().cloned().collect();

        // S4. For each group of each neighbor, find the neighbor with the maximal timestamp.
        let mut neighbors_after_compaction = vec![];
        for neighbor_group in neighbor_groups_nid.into_iter() {
            let valid_neighbor = neighbor_group.into_iter().max_by_key(
                |item| item.timestamp
            ).unwrap();
            // If the neighbor is deleted, not include it to the final result.
            if valid_neighbor.tomb == 0 {
                neighbors_after_compaction.push(valid_neighbor);
            }
        }
        neighbors_after_compaction
    }
}

/// Test whether the compaction works.
#[allow(dead_code)]
#[cfg(test)]
mod test_compact {
    use std::thread;
    use std::time::Duration;
    use crate::compact::CompactController;
    use crate::types::{GraphEntry, V32, Vertex};

    /// Directly test whether it works.
    #[test]
    fn test_compact() {
        // Main steps of testing compaction.

        // Step 1. Build original graph entries.
        let u1 = Vertex::new(23u32, 0u8);
        let v1 = Vertex::new(24u32, 1u8);
        let v2 = Vertex::new(25u32, 1u8);
        let v3 = Vertex::new(26u32, 1u8);
        let entry1 = GraphEntry::<V32>::create(u1, vec![v1, v2, v3]);
        thread::sleep(Duration::from_millis(1));

        let u2 = Vertex::new(23u32, 0u8);
        let v4 = Vertex::new_tomb(24u32, 1u8);
        let v5 = Vertex::new(25u32, 1u8);
        let v6 = Vertex::new(26u32, 2u8);
        let entry2 = GraphEntry::<V32>::create(u2, vec![v4, v5, v6]);
        thread::sleep(Duration::from_millis(1));

        let u3 = Vertex::new(29u32, 0u8);
        let v7 = Vertex::new(24u32, 1u8);
        let v8 = Vertex::new(25u32, 1u8);
        let v9 = Vertex::new(26u32, 1u8);
        let entry3 = GraphEntry::<V32>::create(u3, vec![v7, v8, v9]);

        // Step 2. Perform compact.
        let entries_after_compaction = CompactController::execute_compaction(&vec![entry1, entry2, entry3]);

        // Step 3. Valid whether success.
        // Firstly, there are only two entries in the list.
        assert_eq!(entries_after_compaction.len(), 2);
        // Check the inside elements.
        for ge_after_compaction in entries_after_compaction {
            if ge_after_compaction.key.vertex_id == 23u32 {
                // It should contain v25(+), v26(+), and v26(-)
                assert!(ge_after_compaction.neighbors.iter().any(
                    |item| item.vertex_id == 25u32 && item.direction_tag == 1)
                );
                assert!(ge_after_compaction.neighbors.iter().any(
                    |item| item.vertex_id == 26u32 && item.direction_tag == 1)
                );
                assert!(ge_after_compaction.neighbors.iter().any(
                    |item| item.vertex_id == 26u32 && item.direction_tag == 2)
                );
                // It should not contain v24.
                assert!(!ge_after_compaction.neighbors.iter().any(
                    |item| item.vertex_id == 24u32)
                );
                // It should only contain three items.
                assert_eq!(ge_after_compaction.neighbors.len(), 3);
                assert_eq!(ge_after_compaction.neighbor_size, 3);
            }
        }
    }
}