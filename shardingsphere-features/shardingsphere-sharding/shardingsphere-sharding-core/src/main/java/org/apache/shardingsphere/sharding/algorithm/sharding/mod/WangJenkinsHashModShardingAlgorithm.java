package org.apache.shardingsphere.sharding.algorithm.sharding.mod;

import com.google.common.base.Preconditions;
import lombok.Getter;
import lombok.Setter;
import org.apache.shardingsphere.sharding.api.sharding.ShardingAutoTableAlgorithm;
import org.apache.shardingsphere.sharding.api.sharding.standard.PreciseShardingValue;
import org.apache.shardingsphere.sharding.api.sharding.standard.RangeShardingValue;
import org.apache.shardingsphere.sharding.api.sharding.standard.StandardShardingAlgorithm;
import org.apache.shardingsphere.sharding.support.WangJenkinsHashAlgorithm;

import java.util.Collection;
import java.util.Collections;
import java.util.Properties;

@Getter
@Setter
public final class WangJenkinsHashModShardingAlgorithm implements StandardShardingAlgorithm<Comparable<?>>, ShardingAutoTableAlgorithm {
    private static final String SHARDING_COUNT_KEY = "sharding-count";

    private Properties props = new Properties();

    private int shardingCount;

    private boolean watch = false;

    @Override
    public void init() {
        shardingCount = getShardingCount();
        if ((shardingCount & shardingCount - 1) == 0) {
            watch = true;
        }
    }

    private int getShardingCount() {
        Preconditions.checkArgument(props.containsKey(SHARDING_COUNT_KEY), "Sharding count cannot be null.");
        return Integer.parseInt(props.getProperty(SHARDING_COUNT_KEY));
    }

    @Override
    public String doSharding(final Collection<String> availableTargetNames, final PreciseShardingValue<Comparable<?>> shardingValue) {
        for (String each : availableTargetNames) {
            long mod;
            if (watch) {
                mod = hashShardingValue(shardingValue.getValue()) & (shardingCount - 1);
            } else {
                mod = hashShardingValue(shardingValue.getValue()) % shardingCount;
            }
            if (each.endsWith(String.valueOf(mod))) {
                return each;
            }
        }
        return null;
    }

    @Override
    public Collection<String> doSharding(final Collection<String> availableTargetNames, final RangeShardingValue<Comparable<?>> shardingValue) {
        return availableTargetNames;
    }

    private long hashShardingValue(final Comparable<?> shardingValue) {
        return Math.abs(WangJenkinsHashAlgorithm.wjHash(shardingValue.hashCode()));
    }

    @Override
    public int getAutoTablesAmount() {
        return shardingCount;
    }

    @Override
    public String getType() {
        return "WJ_HASH_MOD";
    }

    @Override
    public Collection<String> getAllPropertyKeys() {
        return Collections.singletonList(SHARDING_COUNT_KEY);
    }
}
