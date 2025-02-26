use std::time::{SystemTime, UNIX_EPOCH};

pub fn get_current_timestamp() -> u64 {
    let now = SystemTime::now();
    let duration_since_epoch = now.duration_since(UNIX_EPOCH).expect("Time went backwards");
    let timestamp = duration_since_epoch.as_micros() as u64;
    timestamp
}

#[allow(unused)]
pub fn parse_timestamp(timestamp: u64) -> SystemTime {
    let duration = std::time::Duration::from_micros(timestamp);
    UNIX_EPOCH + duration
}


#[cfg(test)]
mod util_test {
    use crate::util::{get_current_timestamp, parse_timestamp};

    #[test]
    pub fn test_timestamp() {
        let cur_timestamp = get_current_timestamp();
        println!("Timestamp: {}", cur_timestamp);
        let cur_timestamp1 = get_current_timestamp();
        println!("Timestamp: {}", cur_timestamp1);
        println!("Diff: {}", cur_timestamp1 - cur_timestamp);

        let datetime = parse_timestamp(cur_timestamp);
        println!("Parse Timestamp: {}, result: {:?}", cur_timestamp, datetime);
    }
}