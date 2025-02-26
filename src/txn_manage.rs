pub mod txn;
pub mod watermark;

use std::{
    collections::{BTreeMap, HashSet},
    sync::{atomic::AtomicBool, Arc},
};

use crossbeam_skiplist::SkipMap;
use parking_lot::Mutex;
use crate::lsm_community::LSMStorage;
use self::{txn::Transaction, watermark::Watermark};

pub struct CommittedTxnData {
    pub(crate) key_hashes: HashSet<u32>,
    #[allow(dead_code)]
    pub(crate) read_ts: u64,
    #[allow(dead_code)]
    pub(crate) commit_ts: u64,
}

pub struct LsmTxnInner {
    pub write_lock: Mutex<()>,
    pub commit_lock: Mutex<()>,
    pub ts: Arc<Mutex<(u64, Watermark)>>,
    pub committed_txn_list: Arc<Mutex<BTreeMap<u64, CommittedTxnData>>>,
}

impl LsmTxnInner {
    pub fn new(initial_ts: u64) -> Self {
        Self {
            write_lock: Mutex::new(()),
            commit_lock: Mutex::new(()),
            ts: Arc::new(Mutex::new((initial_ts, Watermark::new()))),
            committed_txn_list: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    pub fn latest_commit_ts(&self) -> u64 {
        self.ts.lock().0
    }

    pub fn update_commit_ts(&self, ts: u64) {
        self.ts.lock().0 = ts;
    }

    /// All ts (strictly) below this ts can be garbage collected.
    pub fn watermark(&self) -> u64 {
        let ts = self.ts.lock();
        ts.1.watermark().unwrap_or(ts.0)
    }

    pub fn new_txn(&self, inner: Arc<LSMStorage>, serializable: bool) -> Arc<Transaction> {
        let mut ts = self.ts.lock();
        let read_ts = ts.0;
        ts.1.add_reader(read_ts);
        Arc::new(Transaction {
            inner,
            read_ts,
            local_storage: Arc::new(SkipMap::new()),
            committed: Arc::new(AtomicBool::new(false)),
            key_hashes: if serializable {
                Some(Mutex::new((HashSet::new(), HashSet::new())))
            } else {
                None
            },
        })
    }
}
