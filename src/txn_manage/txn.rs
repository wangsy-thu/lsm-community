use std::{
    collections::HashSet
    ,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::{bail, Result};
use bytes::Bytes;
use crossbeam_skiplist::SkipMap;
use parking_lot::Mutex;

use crate::lsm_community::{LSMStorage, WriteBatchRecord};
use crate::txn_manage::CommittedTxnData;
use crate::types::{Decode, Encode, V32};

pub struct Transaction {
    pub(crate) read_ts: u64,
    pub(crate) inner: Arc<LSMStorage>,
    pub(crate) local_storage: Arc<SkipMap<Bytes, Bytes>>,
    pub(crate) committed: Arc<AtomicBool>,
    /// Write set and read set
    pub(crate) key_hashes: Option<Mutex<(HashSet<u32>, HashSet<u32>)>>,
}

impl Transaction {
    pub fn get(&mut self, key: &[u8]) -> Result<Option<Bytes>> {
        if self.committed.load(Ordering::SeqCst) {
            panic!("cannot operate on committed txn!");
        }
        if let Some(guard) = &self.key_hashes {
            let mut guard = guard.lock();
            let (_, read_set) = &mut *guard;
            read_set.insert(farmhash::hash32(key));
        }
        if let Some(entry) = self.local_storage.get(key) {
            return if entry.value().is_empty() {
                Ok(None)
            } else {
                Ok(Some(entry.value().clone()))
            }
        }
        let vertex_id = V32::from_bytes(key).unwrap().vertex_id;
        let res_neighbors = self.inner.read_neighbors(&vertex_id);
        let res_bytes = res_neighbors.into_iter().fold(Vec::<u8>::new(), |mut acc, item| {
            acc.extend_from_slice(&item.encode());
            acc
        });
        Ok(Some(Bytes::from(res_bytes)))
    }

    pub fn put(&self, key: &[u8], value: &[u8]) {
        if self.committed.load(Ordering::SeqCst) {
            panic!("cannot operate on committed txn!");
        }
        self.local_storage
            .insert(Bytes::copy_from_slice(key), Bytes::copy_from_slice(value));
        if let Some(key_hashes) = &self.key_hashes {
            let mut key_hashes = key_hashes.lock();
            let (write_hashes, _) = &mut *key_hashes;
            write_hashes.insert(farmhash::hash32(key));
        }
    }

    pub fn commit(&mut self) -> Result<()> {
        self.committed
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .expect("cannot operate on committed txn!");
        let _commit_lock = self.inner.txn_manager_ref().commit_lock.lock();
        let serialize_check;
        if let Some(guard) = &self.key_hashes {
            let guard = guard.lock();
            let (write_set, read_set) = &*guard;
            println!(
                "commit txn: write_set: {:?}, read_set: {:?}",
                write_set, read_set
            );
            if !write_set.is_empty() {
                let committed_txn_lst = self.inner.txn_manager_ref().committed_txn_list.lock();
                for (_, txn_data) in committed_txn_lst.range((self.read_ts + 1)..) {
                    for key_hash in read_set {
                        if txn_data.key_hashes.contains(key_hash) {
                            bail!("serializable check failed");
                        }
                    }
                }
            }
            serialize_check = true;
        } else {
            serialize_check = false;
        }
        let batch = self
            .local_storage
            .iter()
            .map(|entry| {
                if entry.value().is_empty() {
                    WriteBatchRecord::Del(V32::from_bytes(entry.key()).unwrap())
                } else {
                    WriteBatchRecord::Put(V32::from_bytes(entry.key()).unwrap(),
                                          V32::from_bytes(entry.value()).unwrap())
                }
            })
            .collect::<Vec<_>>();
        let ts = self.inner.write_batch_inner(&batch)?;
        if serialize_check {
            let mut committed_txn_lst = self.inner.txn_manager_ref().committed_txn_list.lock();
            let mut key_hashes = self.key_hashes.as_ref().unwrap().lock();
            let (write_set, _) = &mut *key_hashes;

            let old_data = committed_txn_lst.insert(
                ts,
                CommittedTxnData {
                    key_hashes: std::mem::take(write_set),
                    read_ts: self.read_ts,
                    commit_ts: ts,
                },
            );
            assert!(old_data.is_none());

            // remove unneeded txn data
            let watermark = self.inner.txn_manager_ref().watermark();
            while let Some(entry) = committed_txn_lst.first_entry() {
                if *entry.key() < watermark {
                    entry.remove();
                } else {
                    break;
                }
            }
        }
        Ok(())
    }
}

impl Drop for Transaction {
    fn drop(&mut self) {
        self.inner.txn_manager_ref().ts.lock().1.remove_reader(self.read_ts)
    }
}
