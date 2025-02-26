#[allow(dead_code)]
pub(crate) const BLOCK_SIZE :usize = PAGE_SIZE - PAGE_META_SIZE;

#[allow(dead_code)]
pub(crate) const PAGE_META_SIZE :usize = 3 * size_of::<usize>() + 2 * size_of::<u8>();

#[allow(dead_code)]
pub(crate) const PAGE_DATA_CAPACITY :usize = PAGE_SIZE - PAGE_META_SIZE;

#[allow(dead_code)]
pub(crate) const PAGE_SIZE :usize = 4 * 1024;

#[allow(dead_code)]
pub(crate) const MAX_BLOCK_SIZE :usize = 3 * 1024 + 512;

#[allow(dead_code)]
pub(crate) const MAX_CACHE_SIZE :usize = 1024;

#[allow(dead_code)]
pub(crate) const APP_BLOCK_SIZE :usize = 2 * 1024;

#[allow(dead_code)]
pub(crate) const MAX_ENTRY_SIZE :usize = 512;

#[allow(dead_code)]
pub(crate) const MAX_MEM_SIZE :usize = 256 * 1024;

#[allow(dead_code)]
pub(crate) const READ_BUFFER_SIZE :usize = 128 * 1024 * 1024;