LSM-Community
===

Welcome to the repository of LSM-Community.
The corresponding paper titles: "LSM-Community: A Graph Storage System Exploiting Community Structure in Social Networks".

## 1 - Experiment Environment.
Please install Rust 1.82.0.
And please set the maximum opened file count.
```shell
ulimit -u 4096
```

## 2 - Project Build.

Just run:
```shell
cargo build --release
```

To test whether the building process is success.
You can run:

```shell
./target/release/lsm_community
```

## 3 - Run the Experiments

Just run those three shell scripts.
```shell
# Experiment 1.
sh run_exp1.sh

# Experiment 2.
sh run_exp2.sh

# Experiment 3.
sh run_exp3.sh
```
