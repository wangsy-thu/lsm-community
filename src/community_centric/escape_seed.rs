use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use crate::bucket::CommBucket;
use crate::comm_table::CommID;
use crate::community_centric::L0CommMaintainState;
use crate::graph::VInt;
use crate::lsm_community::LSMStorageState;
use crate::types::{GraphEntry, V32};

/// Here is the move operation, which is the combination of
/// 'escape' and 'seed'.
#[allow(dead_code)]
#[derive(Debug)]
#[derive(Serialize, Deserialize)]
pub(crate) struct MoveTask {
    src_comm_id: CommID,  // The community which vertices escape from.
    move_vc_map: HashMap<VInt, CommID>  // Record the vertex moved.
}

/// Record some options of the 'move' task.
#[allow(dead_code)]
pub(crate) struct MoveTaskOption {
    pub(crate) move_task_on: bool
}

/// The 'move' task controller, used to find which community bucket need to
/// be performed by move.
/// And apply the result.
#[allow(dead_code)]
pub struct MoveTaskController {
    move_task_option: MoveTaskOption // Record some options of the 'move' task.
}

#[allow(dead_code)]
impl MoveTaskController {

    /// Create a new move task option.
    pub fn new(options: MoveTaskOption) -> Self {
        Self {
            move_task_option: options
        }
    }

    /// Generate a new 'move' task according to the current maintaining task.
    pub fn generate_maintain_task(
        &self,
        maintain_state: &Arc<L0CommMaintainState>,
        origin_state: &Arc<LSMStorageState>) -> Option<MoveTask> {
        // Step 1. Determine the community ready for performed 'move' according to the state.
        if let Some(src_comm_id) = maintain_state
            .get_community_for_move(Arc::clone(&origin_state)) {
            // Step 2. Find the neighbor dst community.
            // Step 2-1. Load the src community.
            let src_community = origin_state.load_community(&src_comm_id);
            // Step 2-2. Determine the vertices need to be removed (Compute the in-out degree).
            let mut removed_vertex_map = HashMap::new();
            for vertex_entry in &src_community.adj_map {
                // Compute each neighbor's community according to the community map.
                let mut out_neighbor_comm_map = HashMap::<CommID, u32>::new();
                let mut in_neighbor_count = 0u32;
                for neighbor in vertex_entry.1 {
                    if src_community.adj_map.contains_key(neighbor) {
                        // It is an in-neighbor.
                        in_neighbor_count += 1;
                    } else {
                        if let Some(neighbor_comm_id) = origin_state.lookup_community(neighbor) {
                            // It is an out-neighbor.
                            if out_neighbor_comm_map.contains_key(&neighbor_comm_id) {
                                // Plus one.
                                *out_neighbor_comm_map.get_mut(&neighbor_comm_id).unwrap() += 1;
                            } else {
                                // Put one.
                                out_neighbor_comm_map.insert(neighbor_comm_id, 1u32);
                            }
                        } else {
                            panic!("Vertex not register in the community table.")
                        }

                    }
                    // Find the largest one.
                    if !out_neighbor_comm_map.is_empty() {
                        let (max_comm_id, max_nei_count) = out_neighbor_comm_map.iter()
                            .max_by_key(|&(_, neighbor_count)| *neighbor_count).unwrap();
                        if in_neighbor_count < *max_nei_count {
                            // It is allowed to be moved.
                            removed_vertex_map.insert(*vertex_entry.0, *max_comm_id);
                        }
                    }
                }
            }
            if removed_vertex_map.is_empty() {
                None
            } else {
                Some(MoveTask {
                    src_comm_id,
                    move_vc_map: removed_vertex_map,
                })
            }
        } else {
            None
        }
    }

    /// Execute according to the corresponding task.
    /// The state should be changed.
    /// Some files may be removed and added.
    /// But in this task, no file should be moved.
    pub fn execute_maintain(
        &self,
        origin_state: &Arc<LSMStorageState>,
        _maintain_state: &Arc<L0CommMaintainState>,
        task: &MoveTask) -> Result<()> {
        // The problem is: given the src community, the dst community list, and the removed vertex,
        // implement vertex escaping and seeding.
        let mut src_bucket_ref = origin_state.get_bucket_mut(&task.src_comm_id).unwrap();
        let mut comm_table_ref = origin_state.comm_table.write().unwrap();
        {
            // Ensure Atomic, the following code may be moved to LSMCommState.
            let g_entry_list = src_bucket_ref.value_mut().vertex_escape(&task.move_vc_map.keys().map(|item| *item).collect());
            // Update the community table.
            // Generate a new cv map.
            let ce_map = g_entry_list.iter().fold(HashMap::new(), |mut acc, item| {
                let target_comm_id = task.move_vc_map.get(&item.key.vertex_id).unwrap();
                acc.entry(*target_comm_id).or_insert(vec![item.clone()]).push(item.clone());
                acc
            });
            for (comm_id, moved_entry_list) in ce_map.into_iter() {
                let mut dst_bucket = origin_state.get_bucket_mut(&comm_id).unwrap();
                // Generate the graph entry.
                dst_bucket.insert_entries_batch(moved_entry_list);
            }
            // Finally, modify the community table.
            comm_table_ref.vertex_escape(&task.src_comm_id, &task.move_vc_map);
        }
        Ok(())
    }

    /// Perform the ``move'' operation for dynamic community detection.
    pub fn perform_move(
        &self,
        origin_state: &Arc<LSMStorageState>,
        maintain_state: &Arc<L0CommMaintainState>,
    ) {
        // Main steps of the move operation.
        if self.move_task_option.move_task_on {
            // Step 1. Generate a new move task.
            let option_move_task = self.generate_maintain_task(
                maintain_state, origin_state
            );

            // Step 2. Execute the move task.
            match option_move_task {
                None => {
                    println!("No community need to performed move.")
                }
                Some(move_task) => {
                    self.execute_maintain(
                        origin_state,
                        maintain_state,
                        &move_task
                    ).unwrap();
                    // println!("Perform Move, Move Task: {:?}", &move_task);
                }
            }
        }
    }
}

#[allow(dead_code)]
impl CommBucket {
    /// Remove the escaped vertices force.
    /// Return the graph entry need to be removed.
    pub(crate) fn vertex_escape(
        &mut self,
        escape_vertex_list: &Vec<VInt>
    ) -> Vec<GraphEntry<V32>> {
        // Main steps of vertex escaping.
        // Collect -> Remove.
        let mut escape_entry_list = vec![];
        // Step 1. Collect all the neighbors of the escape vertex list.
        for escape_vertex_id in escape_vertex_list {
            // Find neighbors.
            let escape_neighbors = self.find_neighbors(escape_vertex_id);
            // Build entry.
            let escape_vertex = self.find_vertex(escape_vertex_id).unwrap();
            let escape_entry = GraphEntry::create(
                escape_vertex.clone(), escape_neighbors.clone()
            );
            escape_entry_list.push(escape_entry);

            // Step 2. Remove all.
            self.insert_entry(GraphEntry::create(V32::new_escape(*escape_vertex_id), vec![]));

            // Step 3. Remove the corresponding neighbors.
            // The following steps is a big mistake.
            // Remember, graph entries maintains the neighbors of the given vertices.
            // Bucket is only an organizer, the neighborhood relationships never change.
            // In general, the following process should be deleted.
            // for e_neighbor in &escape_neighbors {
            //     // Check whether it is in the current bucket.
            //     let current_comm_id = comm_table_ref.look_up_community(
            //         escape_vertex_id).unwrap();
            //     let neighbor_comm_id = comm_table_ref.look_up_community(
            //         &e_neighbor.vertex_id).unwrap();
            //     if current_comm_id == neighbor_comm_id {
            //         // Insert a new tomb graph entry.
            //         if !escape_vertex_list.contains(&e_neighbor.vertex_id) {
            //             let mut e_neigh_vertex = e_neighbor.clone();
            //             e_neigh_vertex.tomb = 0;
            //             e_neigh_vertex.direction_tag = 0;
            //             let vertex_predecessor = V32::new_tomb(
            //                 escape_vertex.vertex_id, 2
            //             );
            //             self.insert_entry(
            //                 GraphEntry::create(e_neigh_vertex, vec![vertex_predecessor])
            //             )
            //         }
            //     }
            // }
        }
        escape_entry_list
    }
}

