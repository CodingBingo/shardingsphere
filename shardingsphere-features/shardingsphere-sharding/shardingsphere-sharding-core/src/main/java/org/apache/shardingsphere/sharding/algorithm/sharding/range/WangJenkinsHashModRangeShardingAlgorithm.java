package org.apache.shardingsphere.sharding.algorithm.sharding.range;

import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.collect.Range;
import com.google.common.primitives.Longs;
import lombok.Getter;
import lombok.Setter;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.shardingsphere.sharding.api.sharding.ShardingAutoTableAlgorithm;
import org.apache.shardingsphere.sharding.api.sharding.standard.PreciseShardingValue;
import org.apache.shardingsphere.sharding.api.sharding.standard.RangeShardingValue;
import org.apache.shardingsphere.sharding.api.sharding.standard.StandardShardingAlgorithm;
import org.apache.shardingsphere.sharding.support.WangJenkinsHashAlgorithm;

import java.util.*;
import java.util.stream.Collectors;

@Getter
@Setter
public class WangJenkinsHashModRangeShardingAlgorithm implements StandardShardingAlgorithm<Comparable<?>>, ShardingAutoTableAlgorithm {
    private static final String SHARDING_COUNT_KEY = "sharding-count";
    private static final String SHARDING_RANGES_KEY = "sharding-ranges";

    private Properties props = new Properties();

    private volatile Map<Long, Range<Long>> partitionRange;

    private int shardingCount;

    private boolean watch = false;

    @Override
    public void init() {
        shardingCount = getShardingCount();
        if ((shardingCount & shardingCount - 1) == 0) {
            watch = true;
        }

        partitionRange = calculatePartitionRange();
    }

    private int getShardingCount() {
        Preconditions.checkArgument(props.containsKey(SHARDING_COUNT_KEY), "Sharding count cannot be null.");
        return Integer.parseInt(props.getProperty(SHARDING_COUNT_KEY));
    }

    protected Map<Long, Range<Long>> calculatePartitionRange() {
        Preconditions.checkState(props.containsKey(SHARDING_RANGES_KEY), "Sharding ranges cannot be null.");
        List<String> partitionRanges = Splitter.on(";").trimResults().splitToList(props.getProperty(SHARDING_RANGES_KEY));
        Map<Long, Range<Long>> result = new HashMap<>(partitionRanges.size() + 1, 1);
        for (String partitionRangeStr : partitionRanges) {
            List<Long> partitionRange = Splitter.on(",").trimResults().splitToList(partitionRangeStr)
                    .stream().map(Longs::tryParse).filter(Objects::nonNull).sorted().collect(Collectors.toList());
            Preconditions.checkArgument(CollectionUtils.isNotEmpty(partitionRange) && partitionRange.size() == 3, "Sharding ranges is not valid.");
            Long index = partitionRange.get(0);
            Long low = partitionRange.get(1);
            Long high = partitionRange.get(2);
            result.put(index, Range.closedOpen(low, high));
        }

        return result;
    }

    private long hashShardingValue(final Comparable<?> shardingValue) {
        return Math.abs(WangJenkinsHashAlgorithm.wjHash(shardingValue.hashCode()));
    }

    @Override
    public final String doSharding(final Collection<String> availableTargetNames, final PreciseShardingValue<Comparable<?>> shardingValue) {
        for (String each : availableTargetNames) {
            long mod;
            if (watch) {
                mod = hashShardingValue(shardingValue.getValue()) & (shardingCount - 1);
            } else {
                mod = hashShardingValue(shardingValue.getValue()) % shardingCount;
            }

            long partition = getPartition(mod);
            if (each.endsWith(String.valueOf(partition))) {
                return each;
            }
        }
        return null;
    }

    @Override
    public final Collection<String> doSharding(final Collection<String> availableTargetNames, final RangeShardingValue<Comparable<?>> shardingValue) {
        throw new UnsupportedOperationException("Wang/Jenkins hash mod range sharding algorithm can not tackle with range query.");
    }

    private Long getPartition(final Long value) {
        for (Map.Entry<Long, Range<Long>> entry : partitionRange.entrySet()) {
            if (entry.getValue().contains(value)) {
                return entry.getKey();
            }
        }
        throw new UnsupportedOperationException("");
    }

    @Override
    public final int getAutoTablesAmount() {
        return partitionRange.size();
    }

    @Override
    public Collection<String> getAllPropertyKeys() {
        return Arrays.asList(SHARDING_RANGES_KEY, SHARDING_COUNT_KEY);
    }

    @Override
    public String getType() {
        return "WJ_HASH_MOD_RANGE";
    }
}
