package com.qlh.inference.ui

import com.qlh.inference.status.ModelRuntimeStatus
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test

class ModelRuntimeStatusLabelTest {
    @Test
    fun `a native crop keeps its native name and explicitly identifies the source interval`() {
        val model = ModelRuntimeStatus(
            loaded = true, name = "qwen35 2B Q4_K - Medium", layers = "24",
            loadedLayerRange = listOf(8, 32), selectedName = "unrelated-selection.gguf",
        )
        assertEquals("已加载层段 [8,32)：qwen35 2B Q4_K - Medium", modelRuntimeStatusLabel(model))
        assertEquals("层段 [8,32)", loadedModelScopeLabel(model))
        assertFalse(modelRuntimeStatusLabel(model).contains("整模"))
    }

    @Test
    fun `a loaded whole model is named as a whole model`() {
        assertEquals("已加载整模：local model", modelRuntimeStatusLabel(
            ModelRuntimeStatus(loaded = true, name = "local model"),
        ))
    }

    @Test
    fun `unloaded snapshot ignores leftover display metadata and local selection`() {
        assertEquals("尚未加载模型", modelRuntimeStatusLabel(ModelRuntimeStatus(
            loaded = false, name = "old native name", selectedName = "selected.gguf",
            loadedLayerRange = listOf(8, 32),
        )))
    }

    @Test
    fun `missing native name never borrows the selected file identity`() {
        assertEquals("已加载层段 [8,32)：当前模型", modelRuntimeStatusLabel(ModelRuntimeStatus(
            loaded = true, selectedName = "selected.gguf", loadedLayerRange = listOf(8, 32),
        )))
    }

    @Test
    fun `malformed nonempty range never claims whole model loading`() {
        for (range in listOf(listOf(8), listOf(-1, 32), listOf(32, 8))) {
            assertEquals("已加载层段：crop", modelRuntimeStatusLabel(ModelRuntimeStatus(
                loaded = true, name = "crop", loadedLayerRange = range,
            )))
        }
    }
}
