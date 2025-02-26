use byteorder::{ByteOrder, LittleEndian};

use crate::config::{PAGE_META_SIZE, PAGE_SIZE};
use crate::types::{Decode, Encode};


/// Define the basic storage unit, i.e., the Page.
#[derive(Clone)]
pub(crate) struct Page {
    pub(crate) used_size: usize, // Used byte count.
    pub(crate) block_type: u8, // The block type of its payload.
    // 0 represents csr, 1 represents unsorted.
    pub(crate) block_offset: usize,  // The page number in its block.
    pub(crate) has_next: u8,  // Mark whether it has next page, 0 represents the last page.
    pub(crate) next_page_num: usize, // The next page number.
    pub(crate) data: Vec<u8> // The data field.
}

impl Encode for Page {
    fn encode(&self) -> Vec<u8> {
        // Encode a page to byte stream.
        let mut encode_bytes = Vec::<u8>::new();

        // Encode the meta data part.
        encode_bytes.extend_from_slice(&self.used_size.to_le_bytes());
        encode_bytes.extend_from_slice(&self.block_type.to_le_bytes());
        encode_bytes.extend_from_slice(&self.block_offset.to_le_bytes());
        encode_bytes.extend_from_slice(&self.has_next.to_le_bytes());
        encode_bytes.extend_from_slice(&self.next_page_num.to_le_bytes());
        encode_bytes.extend_from_slice(&self.data);

        // Align to the PAGE_SIZE.
        if self.data.len() + 26 <= PAGE_SIZE {
            encode_bytes.append(&mut vec![0; PAGE_SIZE - self.data.len() - 26])
        }

        // Return the value.
        encode_bytes
    }
}

impl Decode for Page {
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < PAGE_META_SIZE {
            None
        } else {
            // Parse the meta data part.
            let mut parse_index = 0usize;
            let used_size = LittleEndian::read_u64(&bytes[parse_index..parse_index + size_of::<usize>()]) as usize;
            parse_index += size_of::<usize>();
            let block_type = bytes[parse_index];
            parse_index += size_of::<u8>();
            let block_offset = LittleEndian::read_u64(&bytes[parse_index..parse_index + size_of::<usize>()]) as usize;
            parse_index += size_of::<usize>();
            let has_next = bytes[parse_index];
            parse_index += size_of::<u8>();
            let next_page_num = LittleEndian::read_u64(&bytes[parse_index..parse_index + size_of::<usize>()]) as usize;
            parse_index += size_of::<usize>();

            // Parse the data field.
            let data = bytes[parse_index..parse_index + used_size].to_vec();

            Some(
                Page {
                    used_size,
                    block_type,
                    block_offset,
                    has_next,
                    next_page_num,
                    data,
                }
            )
        }
    }
}

#[cfg(test)]
mod test_page {
    use crate::bucket::page::Page;
    use crate::config::PAGE_SIZE;
    use crate::types::{Decode, Encode};

    #[test]
    fn test_page_encode_decode() {
        let page_ground_truth = Page {
            used_size: 12,
            block_type: 0,
            block_offset: 0,
            has_next: 0,
            next_page_num: 0,
            data: vec![0u8; 12],
        };
        let page_bytes = page_ground_truth.encode();

        // Check whether a page is 4KB.
        assert_eq!(page_bytes.len(), PAGE_SIZE);

        let page_load = Page::from_bytes(&page_bytes).unwrap();

        // Valid the result.
        assert_eq!(page_ground_truth.used_size, page_load.used_size);
        assert_eq!(page_ground_truth.block_offset, page_load.block_offset);
        assert_eq!(page_ground_truth.has_next, page_load.has_next);
        assert_eq!(page_ground_truth.next_page_num, page_load.next_page_num);
        assert_eq!(page_ground_truth.data.len(), page_load.data.len());
    }
}