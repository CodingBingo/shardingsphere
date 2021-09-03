package org.apache.shardingsphere.sharding.algorithm.sharding.mod;

import com.google.common.collect.Range;
import org.apache.shardingsphere.sharding.api.sharding.standard.PreciseShardingValue;
import org.apache.shardingsphere.sharding.api.sharding.standard.RangeShardingValue;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class WangJenkinsHashModShardingAlgorithmTest {
    private WangJenkinsHashModShardingAlgorithm wangJenkinsHashModShardingAlgorithm;

    @Before
    public void setup() {
        wangJenkinsHashModShardingAlgorithm = new WangJenkinsHashModShardingAlgorithm();
        wangJenkinsHashModShardingAlgorithm.getProps().setProperty("sharding-count", "4");
        wangJenkinsHashModShardingAlgorithm.init();
    }

    @Test
    public void assertPreciseDoSharding() {
        List<String> availableTargetNames = Arrays.asList("t_order_0", "t_order_1", "t_order_2", "t_order_3");
        assertThat(wangJenkinsHashModShardingAlgorithm.doSharding(availableTargetNames, new PreciseShardingValue<>("t_order", "order_type", "a")), is("t_order_1"));
    }

    @Test
    public void assertRangeDoSharding() {
        List<String> availableTargetNames = Arrays.asList("t_order_0", "t_order_1", "t_order_2", "t_order_3");
        Collection<String> actual = wangJenkinsHashModShardingAlgorithm.doSharding(availableTargetNames, new RangeShardingValue<>("t_order", "create_time", Range.closed("a", "f")));
        assertThat(actual.size(), is(4));
    }
}
