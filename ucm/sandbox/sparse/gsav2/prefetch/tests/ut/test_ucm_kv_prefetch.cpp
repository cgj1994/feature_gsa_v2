/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 *
 * UT for the UCMKVPrefetch class declared in include/kvcache_prefetch.h.
 * Since the load-to-host-buffer migration the class is a thin synchronous
 * wrapper around UC::StoreV1: StoreInit injects the store pointer plus the
 * model layer count, UCMLoadSSD submits one Load shard per (blockId, layerID)
 * pair and returns a PrefetchStatus code, and UCMLookup returns the per-block
 * existence vector (an empty vector on any failure). The old background
 * waiter-thread model (UCMWait / AsyncWaitLoop / mTaskHandle / mWaitQueue_)
 * and its ucm_wait pybind binding have been removed.
 */
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>
#include "../common/mock_store_v1.h"
#include "kvcache_prefetch.h"
#include "kvcache_trans_ascend.h"

using ucmprefetch::UCMKVPrefetch;
using ucmprefetch_test::MockStoreV1;

namespace {
UC::Detail::BlockId MakeBlockId(uint8_t v)
{
    UC::Detail::BlockId id{};
    for (auto& b : id) { b = std::byte{v}; }
    return id;
}

/* UCMLoadSSD status codes (PrefetchStatus is private inside the class):
 * 0 = GSA_PREFETCH_DONE, 1 = GSA_STORE_EMPTY, 2 = GSA_BLOCKID_EMPTY,
 * 3 = GSA_HANDLE_ERROR, 4 = GSA_LOOKUP_ERROR. */
constexpr int32_t kPrefetchDone = 0;
constexpr int32_t kStoreEmpty = 1;
constexpr int32_t kBlockIdEmpty = 2;
constexpr int32_t kHandleError = 3;
}  // namespace

TEST(UCMKVPrefetch, StoreInitRejectsNullPointer)
{
    UCMKVPrefetch p;
    EXPECT_THROW(p.StoreInit(reinterpret_cast<uintptr_t>(nullptr), 2), std::runtime_error);
}

TEST(UCMKVPrefetch, StoreInitAcceptsValidPointer)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), 2);
    /* No exception. */
    SUCCEED();
}

TEST(UCMKVPrefetch, UCMLoadSSDReturnsStoreEmptyOnNullStore)
{
    UCMKVPrefetch p;
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    EXPECT_EQ(p.UCMLoadSSD(ids), kStoreEmpty);
}

TEST(UCMKVPrefetch, UCMLoadSSDReturnsBlockIdEmptyOnEmptyInput)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), 2);
    std::vector<UC::Detail::BlockId> empty;
    EXPECT_EQ(p.UCMLoadSSD(empty), kBlockIdEmpty);
}

TEST(UCMKVPrefetch, UCMLoadSSDSubmitsShardPerBlockAndLayer)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), 3);
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1), MakeBlockId(2)};
    EXPECT_EQ(p.UCMLoadSSD(ids), kPrefetchDone);
    ASSERT_EQ(store.calls.size(), 1u);
    const auto& call = store.calls[0];
    EXPECT_EQ(call.name, "Load");
    /* one shard per (blockId, layerID) pair: 2 blocks x 3 layers */
    ASSERT_EQ(call.task.size(), 6u);
    for (size_t i = 0; i < call.task.size(); ++i) {
        EXPECT_EQ(call.task[i].owner, ids[i / 3]);
        EXPECT_EQ(call.task[i].index, i % 3);
    }
}

TEST(UCMKVPrefetch, UCMLoadSSDReturnsHandleErrorOnLoadFailure)
{
    MockStoreV1 store;
    store.loadFails = true;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), 2);
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    EXPECT_EQ(p.UCMLoadSSD(ids), kHandleError);
}

TEST(UCMKVPrefetch, UCMLookupReturnsEmptyOnNullStore)
{
    UCMKVPrefetch p;
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    EXPECT_TRUE(p.UCMLookup(ids).empty());
}

TEST(UCMKVPrefetch, UCMLookupReturnsEmptyOnEmptyBlockIds)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), 2);
    std::vector<UC::Detail::BlockId> empty;
    EXPECT_TRUE(p.UCMLookup(empty).empty());
}

TEST(UCMKVPrefetch, UCMLookupReturnsExistenceVector)
{
    MockStoreV1 store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), 2);
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1), MakeBlockId(2), MakeBlockId(3)};
    auto result = p.UCMLookup(ids);
    EXPECT_EQ(result.size(), ids.size());
    for (auto v : result) { EXPECT_EQ(v, 1u); }
}

TEST(UCMKVPrefetch, UCMLookupReturnsEmptyOnStoreFailure)
{
    MockStoreV1 store;
    store.lookupFails = true;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), 2);
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    EXPECT_TRUE(p.UCMLookup(ids).empty());
}

TEST(UCMKVPrefetch, UCMLookupReturnsEmptyOnSizeMismatch)
{
    /* Force a mismatch by overriding Lookup to return a vector of different
     * size. MockStoreV1 returns exactly num entries, so trigger via a custom
     * subclass that returns a long vector. */
    struct LongStore : MockStoreV1 {
        UC::Expected<std::vector<uint8_t>> Lookup(const UC::Detail::BlockId*, size_t num) override
        { return std::vector<uint8_t>(num + 1, 1u); }
    } store;
    UCMKVPrefetch p;
    p.StoreInit(reinterpret_cast<uintptr_t>(&store), 2);
    std::vector<UC::Detail::BlockId> ids = {MakeBlockId(1)};
    EXPECT_TRUE(p.UCMLookup(ids).empty());
}
