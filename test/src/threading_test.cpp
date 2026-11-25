#include <gtest/gtest.h>
#include <bench_box/threading/threading.h>
#include <array>

namespace bench_box {
	TEST(Threading, ParsePriority) {
		EXPECT_EQ(parse_thread_priority("background"), thread_priority::background);
		EXPECT_EQ(parse_thread_priority("low"), thread_priority::low);
		EXPECT_EQ(parse_thread_priority("normal"), thread_priority::normal);
		EXPECT_EQ(parse_thread_priority("high"), thread_priority::high);
		EXPECT_EQ(parse_thread_priority("realtime"), thread_priority::realtime);

		EXPECT_EQ(parse_thread_priority("HIGH"), std::nullopt);
		EXPECT_EQ(parse_thread_priority("urgent"), std::nullopt);
		EXPECT_EQ(parse_thread_priority(""), std::nullopt);
	}

	TEST(Threading, PriorityNameRoundTrip) {
		constexpr auto priorities = std::to_array<thread_priority>({
			thread_priority::background,
			thread_priority::low,
			thread_priority::normal,
			thread_priority::high,
			thread_priority::realtime
		});

		for (const auto prio : priorities) {
			auto parsed = parse_thread_priority(thread_priority_name(prio));
			ASSERT_TRUE(parsed.has_value());
			EXPECT_EQ(parsed.value(), prio);
		}
	}
} // namespace bench_box
