"""`webui.gpus` 的纯函数测试。

只用标准库 ``unittest``：仓库不依赖 pytest，而这一层是「交付给别人也要能跑」的部分，
不该为了测试再拉一个依赖进来（装了 pytest 也能直接收集这些用例）。

跑法::

    .venv/bin/python -m unittest discover webui/tests -v

**全部用例都不需要真的有显卡**：`GpuDevice` 是纯数据，`parse_*` 是纯函数，探测层
只在 ``TestLiveProbe`` 里做一次「不崩就行」的宽松检查。
"""

from __future__ import annotations

import unittest

from webui import gpus
from webui.gpus import DeviceChoice, GpuDevice, Inventory


def device(
    index: int,
    *,
    uuid: str | None = None,
    name: str = "Test GPU",
    free: int | None = 24_000,
    total: int | None = 24_576,
    used: int | None = None,
    util: int | None = 0,
) -> GpuDevice:
    """造一张假的卡。uuid 默认按 index 生成，省得每处都写一遍。"""
    return GpuDevice(
        index=index,
        uuid=uuid if uuid is not None else f"GPU-fake-{index}",
        name=name,
        memory_total_mb=total,
        memory_free_mb=free,
        memory_used_mb=used,
        utilization_pct=util,
    )


class TestParseVisibleDevices(unittest.TestCase):
    """`CUDA_VISIBLE_DEVICES` 的三种语义必须分得开，否则会静默绑错卡。"""

    def test_unset_means_no_restriction(self) -> None:
        self.assertIsNone(gpus.parse_visible_devices(None))

    def test_empty_string_disables_cuda(self) -> None:
        self.assertEqual(gpus.parse_visible_devices(""), set())

    def test_disable_tokens(self) -> None:
        for raw in ("-1", "none", "None", "void", "nodevfiles", " -1 "):
            with self.subTest(raw=raw):
                self.assertEqual(gpus.parse_visible_devices(raw), set())

    def test_index_list(self) -> None:
        self.assertEqual(gpus.parse_visible_devices("2,3"), {"2", "3"})

    def test_whitespace_is_tolerated(self) -> None:
        self.assertEqual(gpus.parse_visible_devices(" 2, 3 "), {"2", "3"})

    def test_uuid_mixed_with_index(self) -> None:
        parsed = gpus.parse_visible_devices("GPU-abc,1")
        self.assertEqual(parsed, {"GPU-abc", "1"})

    def test_trailing_comma_does_not_produce_empty_token(self) -> None:
        self.assertEqual(gpus.parse_visible_devices("2,"), {"2"})


class TestParseSmiOutput(unittest.TestCase):
    """喂手写的 nvidia-smi 输出，核对字段与脏行处理。"""

    def test_normal_row(self) -> None:
        text = "0, GPU-abcd-1234, NVIDIA GeForce RTX 4090 D, 24564, 1600, 23000, 3"
        [parsed] = gpus.parse_smi_output(text)
        self.assertEqual(parsed.index, 0)
        self.assertEqual(parsed.uuid, "GPU-abcd-1234")
        self.assertEqual(parsed.name, "NVIDIA GeForce RTX 4090 D")
        self.assertEqual(parsed.memory_total_mb, 24564)
        self.assertEqual(parsed.memory_used_mb, 1600)
        self.assertEqual(parsed.memory_free_mb, 23000)
        self.assertEqual(parsed.utilization_pct, 3)
        self.assertEqual(parsed.source, "nvidia-smi")
        self.assertEqual(parsed.used_mb, 1600)

    def test_used_falls_back_when_smi_omits_it(self) -> None:
        """短行里没有已用量时，用 total - free 兜底（含驱动预留，不精确但聊胜于无）。"""
        short = GpuDevice(index=0, uuid="GPU-a", name="X", memory_total_mb=100, memory_free_mb=40)
        self.assertEqual(short.used_mb, 60)

    def test_name_with_comma_keeps_memory_readable(self) -> None:
        """卡名自带逗号时不能把显存读成名字的一部分。"""
        text = "1, GPU-xyz, GeForce GTX 1080, rev. 2, 8192, 92, 8100, 10"
        [parsed] = gpus.parse_smi_output(text)
        self.assertEqual(parsed.name, "GeForce GTX 1080, rev. 2")
        self.assertEqual(parsed.memory_total_mb, 8192)
        self.assertEqual(parsed.memory_free_mb, 8100)
        self.assertEqual(parsed.utilization_pct, 10)

    def test_na_utilization_becomes_none(self) -> None:
        text = "0, GPU-a, Some GPU, 8192, 4096, 4096, N/A"
        [parsed] = gpus.parse_smi_output(text)
        self.assertIsNone(parsed.utilization_pct)
        self.assertFalse(parsed.is_busy)

    def test_mig_short_row_is_kept_without_memory(self) -> None:
        """字段不足（MIG 之类的短行）不该让整次探测崩掉。"""
        text = "0, GPU-a, MIG 3g.20gb"
        [parsed] = gpus.parse_smi_output(text)
        self.assertEqual(parsed.name, "MIG 3g.20gb")
        self.assertIsNone(parsed.memory_total_mb)
        self.assertIsNone(parsed.memory_free_mb)

    def test_dirty_rows_are_skipped(self) -> None:
        text = "\n".join([
            "",
            "not-an-index, GPU-a, GPU, 1, 2, 3, 4",
            "GPU-above-missing-name",
            "2, GPU-b, Good GPU, 8192, 92, 8000, 0",
        ])
        parsed = gpus.parse_smi_output(text)
        self.assertEqual([item.index for item in parsed], [2])

    def test_bare_uuid_gets_gpu_prefix(self) -> None:
        """不带前缀的 uuid 会让 torch 报 No CUDA GPUs are available，必须补上。"""
        text = "0, abcd-1234, GPU, 1, 1, 1, 0"
        [parsed] = gpus.parse_smi_output(text)
        self.assertEqual(parsed.uuid, "GPU-abcd-1234")

    def test_empty_name_falls_back(self) -> None:
        text = "3, , , 1, 1, 1, 0"
        [parsed] = gpus.parse_smi_output(text)
        self.assertTrue(parsed.name)
        self.assertNotEqual(parsed.name.strip(), "")


class TestParseSmiProcesses(unittest.TestCase):
    def test_normal_row(self) -> None:
        text = "GPU-abcd, 12345, python, 4096"
        [parsed] = gpus.parse_smi_processes(text)
        self.assertEqual(parsed.gpu_uuid, "GPU-abcd")
        self.assertEqual(parsed.pid, 12345)
        self.assertEqual(parsed.name, "python")
        self.assertEqual(parsed.used_memory_mb, 4096)

    def test_process_name_with_comma(self) -> None:
        text = "GPU-abcd, 12345, /usr/bin/python, -m, train, 4096"
        [parsed] = gpus.parse_smi_processes(text)
        self.assertEqual(parsed.name, "/usr/bin/python, -m, train")
        self.assertEqual(parsed.pid, 12345)
        self.assertEqual(parsed.used_memory_mb, 4096)

    def test_blank_output_means_no_processes(self) -> None:
        self.assertEqual(gpus.parse_smi_processes(""), [])
        self.assertEqual(gpus.parse_smi_processes("\n\n"), [])


class TestDeviceDerived(unittest.TestCase):
    def test_key_prefers_uuid(self) -> None:
        self.assertEqual(device(2).key, "GPU-fake-2")

    def test_key_falls_back_to_index(self) -> None:
        self.assertEqual(device(2, uuid="").key, "index:2")

    def test_cuda_visible_requires_uuid(self) -> None:
        self.assertEqual(device(0).cuda_visible, "GPU-fake-0")

    def test_cuda_visible_is_none_without_uuid(self) -> None:
        self.assertIsNone(device(0, uuid="").cuda_visible)

    def test_short_label_drops_vendor_prefix(self) -> None:
        labelled = device(1, name="NVIDIA GeForce RTX 4090 D")
        self.assertEqual(labelled.short_label, "GPU 1: GeForce RTX 4090 D")

    def test_label_mentions_free_memory(self) -> None:
        self.assertIn("GB 空闲", device(0, free=24_064).label)


class TestBusy(unittest.TestCase):
    def test_high_utilization_is_busy(self) -> None:
        self.assertTrue(device(0, util=gpus.BUSY_UTILIZATION_PCT).is_busy)

    def test_low_free_memory_is_busy(self) -> None:
        self.assertTrue(device(0, free=gpus.BUSY_FREE_MEMORY_MB - 1).is_busy)

    def test_idle_card(self) -> None:
        self.assertFalse(device(0, free=24_000, util=0).is_busy)

    def test_unknown_readings_are_not_busy(self) -> None:
        """信息不足时不做判断：读不到数据不该把所有卡都判死。"""
        self.assertFalse(device(0, free=None, util=None).is_busy)


class TestSelectDevices(unittest.TestCase):
    def test_sorted_by_free_memory_descending(self) -> None:
        pool, _ = gpus.select_devices([device(0, free=8_000), device(1, free=40_000),
                                       device(2, free=24_000)])
        self.assertEqual([item.index for item in pool], [1, 2, 0])

    def test_busy_cards_are_skipped(self) -> None:
        pool, notes = gpus.select_devices([device(0, free=40_000),
                                           device(1, free=512, util=99)])
        self.assertEqual([item.index for item in pool], [0])
        self.assertTrue(any("跳过" in note for note in notes))

    def test_all_busy_falls_back_to_full_pool(self) -> None:
        """全忙时宁可挤一张忙卡，也不能让分配结果为空。"""
        pool, notes = gpus.select_devices([device(0, free=512), device(1, free=256)])
        self.assertEqual(len(pool), 2)
        self.assertTrue(any("都在忙" in note for note in notes))

    def test_excluded_cards_are_dropped(self) -> None:
        cards = [device(0), device(1), device(2)]
        pool, notes = gpus.select_devices(cards, excluded={"GPU-fake-1"})
        self.assertEqual([item.index for item in pool], [0, 2])
        self.assertTrue(any("排除" in note for note in notes))

    def test_unknown_memory_sorts_last_but_stays(self) -> None:
        pool, _ = gpus.select_devices([device(0, free=None, total=None), device(1, free=8_000)])
        self.assertEqual([item.index for item in pool], [1, 0])

    def test_empty_input(self) -> None:
        pool, notes = gpus.select_devices([])
        self.assertEqual(pool, [])
        self.assertEqual(notes, [])


class TestPlanAssignments(unittest.TestCase):
    def test_zero_count(self) -> None:
        self.assertEqual(gpus.plan_assignments([device(0)], 0), [])

    def test_no_devices(self) -> None:
        self.assertEqual(gpus.plan_assignments([], 5), [])

    def test_single_device_takes_everything(self) -> None:
        assigned = gpus.plan_assignments([device(0)], 3)
        self.assertEqual(len(assigned), 3)

    def test_round_robin_over_five_cards(self) -> None:
        cards = [device(index) for index in range(5)]
        assigned = gpus.plan_assignments(cards, 11)
        self.assertEqual(
            [item.index for item in assigned],
            [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0],
        )


class TestAutoAssignments(unittest.TestCase):
    def test_busy_cards_are_not_used(self) -> None:
        cards = [device(0, free=40_000), device(1, free=128)]
        assigned, _ = gpus.auto_assignments(cards, 3)
        self.assertEqual({item.index for item in assigned}, {0})

    def test_empty_pool_returns_empty(self) -> None:
        assigned, _ = gpus.auto_assignments([], 3)
        self.assertEqual(assigned, [])


class TestDeviceChoice(unittest.TestCase):
    def test_from_device_binds_uuid_and_local_index(self) -> None:
        choice = DeviceChoice.from_device(device(3))
        self.assertEqual(choice.key, "GPU-fake-3")
        self.assertEqual(choice.cuda_visible, "GPU-fake-3")
        self.assertEqual(choice.device_arg, "cuda:0")
        self.assertTrue(choice.is_gpu)

    def test_cpu_choice_is_not_gpu(self) -> None:
        self.assertFalse(gpus.CPU_CHOICE.is_gpu)
        self.assertIsNone(gpus.CPU_CHOICE.cuda_visible)

    def test_short_label_is_used_for_registry(self) -> None:
        choice = DeviceChoice.from_device(device(1, name="NVIDIA GeForce RTX 4090 D"))
        self.assertEqual(gpus.short_label(choice), "GPU 1: GeForce RTX 4090 D")


class TestBuildChoices(unittest.TestCase):
    def test_auto_expands_to_one_choice_per_task(self) -> None:
        inventory = Inventory(devices=[device(0), device(1)], physical_count=2)
        choices, _ = gpus.build_choices(
            next(option for option in gpus.device_options(inventory) if option.key == gpus.AUTO_KEY),
            5,
            inventory,
        )
        self.assertEqual(len(choices), 5)
        self.assertEqual([choice.key for choice in choices],
                         ["GPU-fake-0", "GPU-fake-1"] * 2 + ["GPU-fake-0"])

    def test_auto_without_gpus_falls_back_to_cpu(self) -> None:
        inventory = Inventory(devices=[], physical_count=0)
        choices, _ = gpus.build_choices(
            DeviceChoice(key=gpus.AUTO_KEY, label="auto", device_arg="auto"), 4, inventory
        )
        self.assertEqual([choice.key for choice in choices], ["cpu"] * 4)

    def test_explicit_choice_applies_to_all_tasks(self) -> None:
        inventory = Inventory(devices=[device(0)], physical_count=1)
        one = DeviceChoice.from_device(device(0))
        choices, _ = gpus.build_choices(one, 3, inventory)
        self.assertEqual(choices, [one] * 3)

    def test_device_options_offer_auto_only_with_multiple_cards(self) -> None:
        single = gpus.device_options(Inventory(devices=[device(0)], physical_count=1))
        self.assertNotIn(gpus.AUTO_KEY, [option.key for option in single])
        self.assertEqual(single[-1].key, "cpu")

        multi = gpus.device_options(Inventory(devices=[device(0), device(1)], physical_count=2))
        self.assertEqual(multi[0].key, gpus.AUTO_KEY)
        self.assertIn("2 张显卡", multi[0].label)


class TestIsVisible(unittest.TestCase):
    def test_index_token(self) -> None:
        self.assertTrue(gpus._is_visible(device(3), {"3"}))

    def test_uuid_token_is_case_insensitive(self) -> None:
        self.assertTrue(gpus._is_visible(device(0, uuid="GPU-AbCd"), {"gpu-abcd"}))

    def test_not_listed(self) -> None:
        self.assertFalse(gpus._is_visible(device(3), {"4"}))


class TestLiveProbe(unittest.TestCase):
    """本机上的实况探测：只断言「不崩、自洽」，不假定机器一定有几张卡。"""

    def test_detect_devices_is_self_consistent(self) -> None:
        inventory = gpus.detect_devices(force=True, with_processes=True)
        self.assertLessEqual(len(inventory.devices), inventory.physical_count)
        for item in inventory.devices:
            self.assertTrue(item.cuda_visible, "不可绑定的卡不该留在候选池里")
            self.assertTrue(item.uuid)
        if inventory.source == "nvidia-smi":
            self.assertGreater(inventory.physical_count, 0)

    def test_cache_returns_same_object(self) -> None:
        gpus.clear_cache()
        first = gpus.detect_devices()
        self.assertIs(first, gpus.detect_devices())


if __name__ == "__main__":
    unittest.main()
