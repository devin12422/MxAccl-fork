#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <vector>
#include <thread>
#include <tuple>
#include <atomic>
#include <mutex>
#include <condition_variable>

#include <memx/accl/utils/sync_queue.hpp>

class Task {
    public:
        virtual bool execute() = 0; // if execute returns true we put it back on the task queue
        virtual ~Task() {}
};

template <typename F, typename... Args>
class CallbackTask: public Task {
    public:
        template <typename FnT, typename... Ts>
        CallbackTask(FnT&& function, Ts&&... args): func(std::forward<FnT>(function)), args(std::forward<Ts>(args)...) {}
        bool execute() override { 
            return std::apply(func, args); 
        }
    private:
        F func;
        std::tuple<Args...> args;
};

class thread_pool {
    public:
        thread_pool(const std::string& label, size_t workers, size_t continious,size_t max_jobs=0);
        template <typename F, typename... Args>
        void submitTask(F&& function, Args&&... args);
        void wait();
        void stop();
        bool stopped();

    private:
        void workerTarget();
        std::string m_label;
        size_t m_task_count{0};
        size_t m_done_count{0};
        std::chrono::milliseconds m_timeout{50ms};
        std::atomic<bool> m_stop{false};
        std::mutex m_mutex;
        std::condition_variable m_done_condition;
        bool m_continious;
        sync_queue<Task*> m_task_queue;
        std::vector<std::thread> m_workers;
};

using namespace std::chrono_literals;

inline thread_pool::thread_pool(const std::string& label,
        size_t workers, size_t continious, size_t max_jobs):
    m_label(label),
    m_continious(continious),
    m_task_queue(sync_queue<Task*>(max_jobs)){
    for (size_t i = 0; i < workers; ++i) {
        m_workers.push_back(std::thread(&thread_pool::workerTarget, this));
        // TODO try setting thread scheduling priority
    }
}

inline void thread_pool::workerTarget() {
    while (!m_stop.load()) {
        std::optional<Task*> opt = m_task_queue.pop(m_timeout);
        if (!opt.has_value()) {
            continue;
        }
        Task* ptask = opt.value();
        if(m_continious){
            if (ptask->execute()) {
                while(!m_task_queue.push(ptask,m_timeout)) {
                    if (m_stop.load()) {
                        return;
                    }
                }
                continue;
            }
            delete ptask;
            {
                std::lock_guard lock(m_mutex);
                m_done_count++;
                if (m_done_count == m_task_count) {
                    m_done_condition.notify_one();
                }
            }
        }
        else{
            ptask->execute();
            delete ptask;
        }
    }
}

inline void thread_pool::stop() {
    if(m_stop.load()){
        return;
    }
    m_stop.store(true);
    for (auto& worker: m_workers) {
        worker.join();
    }
    if(m_continious){
        while(m_done_count < m_task_count){
            std::optional<Task*> opt = m_task_queue.pop();
            if (!opt.has_value()) {
                break;
            }
            Task* ptask = opt.value();
            delete ptask;
            m_done_count++;
        }
    }
}

inline bool thread_pool::stopped() {
    return std::none_of(m_workers.begin(), m_workers.end(), [](const std::thread& th){ return th.joinable(); });
}

inline void thread_pool::wait() {
    if (m_task_count == 0) {
        return;
    }
    std::unique_lock lock(m_mutex);
    m_done_condition.wait(lock, [this]() { return m_done_count == m_task_count; });
    lock.unlock();
}

template <typename F, typename... Args>
inline void thread_pool::submitTask(F&& function, Args&&... args) {
    Task* pt = new CallbackTask<F, Args...>(std::forward<F>(function), std::forward<Args>(args)...);
    m_task_queue.push(pt,m_timeout);
    {
        std::lock_guard lock(m_mutex);
        m_task_count++;
    }
}

#endif
