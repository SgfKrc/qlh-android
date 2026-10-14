package com.qlh.inference.ui

import com.qlh.inference.network.ServerModelSummary
import org.junit.Assert.*
import org.junit.Test

class ModelIntentCatalogFilterTest {
    private val models = listOf(
        ServerModelSummary(modelId = "qwen2-5-7b", name = "Qwen2.5-7B", isAvailable = false),
        ServerModelSummary(modelId = "qwen3-5-9b", name = "Qwen3.5-9B", isAvailable = false),
        ServerModelSummary(modelId = "glm-9b", name = "GLM 9B", isAvailable = true),
        ServerModelSummary(modelId = "tiny", name = "Tiny model", isAvailable = true),
    )

    @Test fun `9B search includes uncached models in catalog order`() {
        assertEquals(models.subList(1, 3), filterModelIntentCatalog(models, " 9B "))
        assertFalse(filterModelIntentCatalog(models, "9B").first().isAvailable)
    }

    @Test fun `search matches display name or exact registry ID ignoring case`() {
        assertEquals(listOf(models[1]), filterModelIntentCatalog(models, "QWEN3.5"))
        assertEquals(listOf(models[1]), filterModelIntentCatalog(models, "QWEN3-5-9B"))
        assertEquals(listOf(models[3]), filterModelIntentCatalog(models, "Tiny MODEL"))
    }

    @Test fun `clearing search restores the whole catalog including unavailable sources`() {
        assertEquals(models, filterModelIntentCatalog(models, ""))
        assertEquals(models, filterModelIntentCatalog(models, "  "))
    }

    @Test fun `unmatched and empty catalogs produce no candidates without altering catalog`() {
        assertTrue(filterModelIntentCatalog(models, "70B").isEmpty())
        assertTrue(filterModelIntentCatalog(emptyList(), "9B").isEmpty())
        assertEquals(4, models.size)
        assertEquals("qwen3-5-9b", models[1].modelId)
    }
}
