use std::io::Write;

use env_logger::{Builder, Env};

pub fn init_logger() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new dir.
    std::fs::create_dir_all("logs")?;

    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("logs/default.log")?;

    Builder::from_env(Env::default().default_filter_or("info"))
        .format(|buf, record| {
            writeln!(
                buf,
                "{} [{:<5}] {} - {}",
                chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f"),
                record.level(),
                record.target(),
                record.args()
            )
        })
        // Put it
        .target(env_logger::Target::Pipe(Box::new(log_file)))
        .init();

    Ok(())
}