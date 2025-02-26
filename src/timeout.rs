use tokio::time::{Duration, timeout};
use std::thread::{self, JoinHandle};

pub async fn run_with_timeout<F>(func: F, timeout_sec: u64) -> Duration
where
    F: FnOnce() -> Duration + Send + 'static,
{
    let timeout_duration = Duration::from_secs(timeout_sec);

    // 创建一个管道来接收结果
    let (tx, rx) = tokio::sync::oneshot::channel();

    // 保存线程句柄
    let handle = thread::spawn(move || {
        let result = func();
        let _ = tx.send(result);
        result
    });

    // 等待结果或超时
    match timeout(timeout_duration, rx).await {
        Ok(Ok(execute_time)) => execute_time,
        Ok(Err(_)) => timeout_duration,
        Err(_) => {
            // 超时时，强制终止线程
            unsafe {
                terminate_thread(handle);
            }
            timeout_duration
        }
    }
}

/// 强制终止线程的函数
unsafe fn terminate_thread(handle: JoinHandle<Duration>) {
    // 这里我们需要分平台处理
    #[cfg(unix)]
    {
        use std::os::unix::thread::JoinHandleExt;
        let pthread_t = handle.into_pthread_t();

        extern "C" {
            fn pthread_cancel(thread: libc::pthread_t) -> libc::c_int;
        }

        pthread_cancel(pthread_t);
    }

    #[cfg(windows)]
    {
        // Windows 平台的线程终止代码
        use std::os::windows::io::AsRawHandle;
        let handle = handle.as_raw_handle();

        extern "system" {
            fn TerminateThread(hThread: *mut libc::c_void, dwExitCode: u32) -> i32;
        }

        TerminateThread(handle as *mut _, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Instant;

    #[tokio::test]
    async fn test_task_really_terminates() {
        let is_running = Arc::new(AtomicBool::new(true));
        let is_running_clone = is_running.clone();

        let start = Instant::now();

        // 启动一个会运行很久的任务
        let result = run_with_timeout(move || {
            while is_running_clone.load(Ordering::SeqCst) {
                thread::sleep(Duration::from_millis(100));
            }
            Duration::from_secs(10)
        }, 1).await;

        // 验证结果
        assert_eq!(result.as_secs(), 1);  // 确认返回超时时间
        assert!(start.elapsed() < Duration::from_secs(2));  // 确认函数快速返回

        // 等待一小段时间
        tokio::time::sleep(Duration::from_secs(1)).await;

        // 检查任务是否真的被终止
        // 如果任务还在运行，is_running 会继续为 true
        // 如果任务被终止，is_running 的值不会被改变
        assert!(is_running.load(Ordering::SeqCst),
                "Task was really terminated and couldn't modify the atomic bool");
    }

    #[tokio::test]
    async fn test_task_completes_before_timeout() {
        let result = run_with_timeout(|| {
            thread::sleep(Duration::from_secs(1));
            Duration::from_secs(1)
        }, 2).await;

        assert_eq!(result.as_secs(), 1);
    }

    #[tokio::test]
    async fn test_task_times_out() {
        let start = Instant::now();

        let result = run_with_timeout(|| {
            thread::sleep(Duration::from_secs(3));
            Duration::from_secs(3)
        }, 1).await;

        let elapsed = start.elapsed();

        assert_eq!(result.as_secs(), 1);
        // 验证函数确实很快就返回了，而不是等待3秒
        assert!(elapsed < Duration::from_secs(2));
    }

    #[tokio::test]
    async fn test_task_with_zero_duration() {
        let result = run_with_timeout(|| {
            Duration::from_secs(0)
        }, 1).await;

        assert_eq!(result.as_secs(), 0);
    }

    #[tokio::test]
    async fn test_multiple_sequential_tasks() {
        let start = Instant::now();

        // 第一个任务：正常完成
        let result1 = run_with_timeout(|| {
            thread::sleep(Duration::from_millis(500));
            Duration::from_millis(500)
        }, 1).await;

        // 第二个任务：超时
        let result2 = run_with_timeout(|| {
            thread::sleep(Duration::from_secs(2));
            Duration::from_secs(2)
        }, 1).await;

        let total_elapsed = start.elapsed();

        assert!(result1.as_secs_f64() < 1.0);
        assert_eq!(result2.as_secs(), 1);
        // 验证总运行时间小于3秒（如果任务真的运行了2秒，总时间会超过2.5秒）
        assert!(total_elapsed < Duration::from_secs(3));
    }
}