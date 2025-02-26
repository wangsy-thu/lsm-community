use std::sync::{Arc, Mutex};

use crate::comm_table::CommID;
use crate::lsm_community::LSMStorageState;

pub mod escape_seed;

/// This is one of the main contribution, i.e., the community centric computation model.
/// Here write some notes to make the logic clear (will be deleted later).
/// This struct record the state of the maintaining process of the L0 level community.
#[allow(dead_code)]
pub(crate) struct L0CommMaintainState {
    pub(crate) last_maintained_comm_id: Arc<Mutex<CommID>>,
}

#[allow(dead_code)]
impl L0CommMaintainState {

    /// Select a community for 'move' operation.
    pub(crate) fn get_community_for_move(&self, storage_state: Arc<LSMStorageState>) -> Option<CommID> {
        // Perform round strategy.
        let comm_count = storage_state.get_l0_comm_count();
        // Update the community maintain state.
        let res = Some((*self.last_maintained_comm_id.lock().unwrap() + 1) % comm_count);
        let mut last_modify_comm_guard = self.last_maintained_comm_id.lock().unwrap();
        *last_modify_comm_guard += 1;
        res
    }

    pub(crate) fn set_community_for_move(&mut self, comm_id: CommID) {
        match self.last_maintained_comm_id.lock() {
            Ok(mut last_id) => {
                *last_id = comm_id;
            }
            Err(_) => {}
        }
    }
}

