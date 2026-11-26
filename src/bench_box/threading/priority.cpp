#include <bench_box/threading/priority.h>
#include <algorithm>

#if defined(_WIN32)
	#define WIN32_LEAN_AND_MEAN
	#define NOMINMAX
	#include <windows.h>
#elif defined(__linux__)
	#include <pthread.h>
	#include <sched.h>
	#include <sys/resource.h>
	#include <cerrno>
#elif defined(__APPLE__)
	#include <pthread.h>
	#include <pthread/qos.h>
	#include <sched.h>
	#include <sys/resource.h>
#endif

namespace bench_box {
	[[nodiscard]] const char* thread_priority_name(thread_priority prio) noexcept {
		switch (prio) {
			case thread_priority::background: return "background";
			case thread_priority::low: return "low";
			case thread_priority::normal: return "normal";
			case thread_priority::high: return "high";
			case thread_priority::realtime: return "realtime";
		}
		return "unknown";
	}

	[[nodiscard]] std::optional<thread_priority> parse_thread_priority(std::string_view str) noexcept {
		if (str == "background") {
			return thread_priority::background;
		}
		if (str == "low") {
			return thread_priority::low;
		}
		if (str == "normal") {
			return thread_priority::normal;
		}
		if (str == "high") {
			return thread_priority::high;
		}
		if (str == "realtime") {
			return thread_priority::realtime;
		}
		return std::nullopt;
	}

	#if defined(_WIN32)
		// because of windows 7 compat flags, the SDK won't expose the win8+ API we need for disabling power throttling
		struct power_throttling_state {
			uint32_t version;
			uint32_t control_mask;
			uint32_t state_mask;
		};

		static constexpr uint32_t power_throttling_current_version = 1u;
		static constexpr uint32_t power_throttling_execution_speed = 0x1u;
		static constexpr int process_information_class_power_throttling = 4;

		using set_process_information_t = BOOL (WINAPI*)(HANDLE, int, void*, uint32_t);

		static bool disable_win_process_power_throttling() noexcept {
			const auto kernel32 = GetModuleHandleW(L"kernel32.dll");
			if (kernel32 == nullptr) {
				return true; // pre win8: no throttling to disable
			}
			const auto set_process_information = reinterpret_cast<set_process_information_t>(GetProcAddress(kernel32, "SetProcessInformation"));
			if (set_process_information == nullptr) {
				return true; // pre win8: no throttling to disable
			}
			power_throttling_state state = {
				power_throttling_current_version,
				power_throttling_execution_speed,
				0u // state mask 0 = opt out of the controlled policy
			};
			return set_process_information(GetCurrentProcess(), process_information_class_power_throttling, &state, sizeof(state)) != 0;
		}

		static bool enable_win_privilege(const wchar_t* name) noexcept {
			HANDLE token = nullptr;
			if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token) == 0) {
				return false;
			}

			LUID luid = {};
			bool enabled = false;
			if (LookupPrivilegeValueW(nullptr, name, &luid) != 0) {
				TOKEN_PRIVILEGES privileges = {};
				privileges.PrivilegeCount = 1;
				privileges.Privileges[0].Luid = luid;
				privileges.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
				// adjusttokenprivileges succeeds even when it doesn't assign anything, so the last error has to be checked as well
				SetLastError(ERROR_SUCCESS);
				enabled = AdjustTokenPrivileges(token, FALSE, &privileges, 0, nullptr, nullptr) != 0 && GetLastError() == ERROR_SUCCESS;
			}

			CloseHandle(token);
			return enabled;
		}

		[[nodiscard]] static int to_win_thread_priority(thread_priority prio) noexcept {
			switch (prio) {
				case thread_priority::background: return THREAD_PRIORITY_IDLE;
				case thread_priority::low: return THREAD_PRIORITY_BELOW_NORMAL;
				case thread_priority::normal: return THREAD_PRIORITY_NORMAL;
				case thread_priority::high: return THREAD_PRIORITY_HIGHEST;
				case thread_priority::realtime: return THREAD_PRIORITY_TIME_CRITICAL;
			}
			return THREAD_PRIORITY_NORMAL;
		}

		[[nodiscard]] static uint32_t to_win_priority_class(thread_priority prio) noexcept {
			switch (prio) {
				case thread_priority::background: return IDLE_PRIORITY_CLASS;
				case thread_priority::low: return BELOW_NORMAL_PRIORITY_CLASS;
				case thread_priority::normal: return NORMAL_PRIORITY_CLASS;
				case thread_priority::high: return ABOVE_NORMAL_PRIORITY_CLASS;
				case thread_priority::realtime: return REALTIME_PRIORITY_CLASS; // needs SeIncreaseBasePriorityPrivilege
			}
			return NORMAL_PRIORITY_CLASS;
		}
	#elif defined(__linux__)
		[[nodiscard]] static int to_linux_nice(thread_priority prio) noexcept {
			switch (prio) {
				case thread_priority::background: return 10;
				case thread_priority::low: return 5;
				case thread_priority::normal: return 0;
				case thread_priority::high: return -10;
				case thread_priority::realtime: return 0; // unused, realtime switches the scheduling policy instead
			}
			return 0;
		}
	#elif defined(__APPLE__)
		[[nodiscard]] static qos_class_t to_apple_qos(thread_priority prio) noexcept {
			switch (prio) {
				case thread_priority::background: return QOS_CLASS_BACKGROUND;
				case thread_priority::low: return QOS_CLASS_UTILITY;
				case thread_priority::normal: return QOS_CLASS_DEFAULT;
				case thread_priority::high: return QOS_CLASS_USER_INITIATED;
				case thread_priority::realtime: return QOS_CLASS_USER_INTERACTIVE; // strongest unprivileged level
			}
			return QOS_CLASS_DEFAULT;
		}
	#endif

	[[nodiscard]] bool set_process_priority([[maybe_unused]] thread_priority prio) noexcept {
		#if defined(_WIN32)
			const auto target = to_win_priority_class(prio);
			if (prio == thread_priority::realtime) {
				// realtime requires additional privilege, and its disabled by default even for admin
				const bool already_realtime = GetPriorityClass(GetCurrentProcess()) == REALTIME_PRIORITY_CLASS;
				if (!enable_win_privilege(L"SeIncreaseBasePriorityPrivilege") && !already_realtime) {
					return false;
				}
			}
			if (SetPriorityClass(GetCurrentProcess(), target) == 0) {
				return false;
			}
			return GetPriorityClass(GetCurrentProcess()) == target;
		#elif defined(__linux__) || defined(__APPLE__)
			// nice / qos are per thread, so this only affects the calling thread, threads created afterwards inherit the setting
			return set_thread_priority(prio);
		#else
			return false;
		#endif
	}

	[[nodiscard]] bool disable_process_power_throttling() noexcept {
		#if defined(_WIN32)
			return disable_win_process_power_throttling();
		#else
			return true;
		#endif
	}

	[[nodiscard]] bool set_thread_priority([[maybe_unused]] thread_priority prio) noexcept {
		#if defined(_WIN32)
			if (prio == thread_priority::realtime && GetPriorityClass(GetCurrentProcess()) != REALTIME_PRIORITY_CLASS) {
				return false; // priorities above 15 only exist inside the realtime class
			}
			const auto thread = GetCurrentThread();
			const auto target = to_win_thread_priority(prio);
			if (SetThreadPriority(thread, target) == 0) {
				return false;
			}
			return GetThreadPriority(thread) == target;
		#elif defined(__linux__)
			if (prio == thread_priority::realtime) {
				sched_param param = {};
				param.sched_priority = 50;
				if (sched_setscheduler(0, SCHED_FIFO, &param) != 0) {
					return false;
				}
				return sched_getscheduler(0) == SCHED_FIFO;
			}

			// leave a realtime policy behind if a previous call set one
			if (sched_getscheduler(0) != SCHED_OTHER) {
				sched_param param = {};
				if (sched_setscheduler(0, SCHED_OTHER, &param) != 0) {
					return false;
				}
			}

			const auto target = to_linux_nice(prio);
			// nice is per thread on linux and pid 0 means the calling thread
			if (setpriority(PRIO_PROCESS, 0, target) != 0) {
				return false;
			}
			errno = 0;
			const auto applied = getpriority(PRIO_PROCESS, 0);
			return errno == 0 && applied == target;
		#elif defined(__APPLE__)
			const auto legacy_background = (prio == thread_priority::background) ? PRIO_DARWIN_BG : 0;
			if (setpriority(PRIO_DARWIN_THREAD, 0, legacy_background) != 0) {
				return false;
			}

			int policy = 0;
			sched_param current = {};
			if (pthread_getschedparam(pthread_self(), &policy, &current) != 0) {
				return false;
			}

			// leave a realtime policy behind if a previous call set one
			if (prio != thread_priority::realtime && policy != SCHED_OTHER) {
				sched_param other = {};
				if (pthread_setschedparam(pthread_self(), SCHED_OTHER, &other) != 0) {
					return false;
				}
			}

			const bool qos_applied = pthread_set_qos_class_self_np(to_apple_qos(prio), 0) == 0;
			if (prio != thread_priority::realtime) {
				return qos_applied;
			}

			// realtime is a hint plus a real scheduling policy, the latter needs root
			const auto max_priority = sched_get_priority_max(SCHED_FIFO);
			if (max_priority < 1) {
				return false;
			}
			sched_param realtime = {};
			realtime.sched_priority = std::min(50, max_priority);
			if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &realtime) != 0) {
				return false;
			}

			policy = 0;
			sched_param applied = {};
			if (pthread_getschedparam(pthread_self(), &policy, &applied) != 0) {
				return false;
			}
			return qos_applied && policy == SCHED_FIFO;
		#else
			return false;
		#endif
	}
} // namespace bench_box
