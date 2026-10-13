package com.qlh.inference.worker

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.qlh.inference.service.ModelSupplyManifest
import com.qlh.inference.service.canonicalSupplyJson
import com.qlh.inference.service.supplySha256
import org.junit.Assert.*
import org.junit.Test

class WorkerPrepareManifestTest {
    private fun manifest(): ModelSupplyManifest {
        val json = JsonParser.parseString("""{
            "manifest_kind":"qlh.segment_supply.v1","operation_id":"supply_11111111111111111111111111111111","node_id":"android_test",
            "source_model_id":"source_gguf","source_model_sha256":"${"b".repeat(64)}","tokenizer_sha256":"${"d".repeat(64)}",
            "total_layers":4,"layer_range":[2,4],"segment_mode":"tail","layer_weight_bytes":[0,0,8,8],
            "hidden_size":1,"artifact_sha256":"${"c".repeat(64)}","format":"gguf","conversion_proof":null,
            "files":[{"path":"segment.gguf","size_bytes":16,"sha256":"${"c".repeat(64)}","kind":"weights"},
                {"path":"segment.manifest.json","size_bytes":10,"sha256":"${"e".repeat(64)}","kind":"manifest"}]
        }""").asJsonObject
        return resign(json)
    }
    private fun resign(json: JsonObject): ModelSupplyManifest {
        json.remove("manifest_sha256")
        json.addProperty("manifest_sha256", supplySha256(canonicalSupplyJson(json).toByteArray(Charsets.UTF_8)))
        return ModelSupplyManifest.parse(json)
    }
    private fun identity(manifest: ModelSupplyManifest): JsonObject = WorkerPrepareTransportTest.wire()["identity"].asJsonObject.deepCopy().apply {
        addProperty("manifest_sha256", manifest.manifestSha256)
        getAsJsonObject("stage_model_identity").addProperty("model_id", "segment.gguf")
    }
    private fun rejected(body: () -> Unit) {
        try { body(); fail("different supplied identity accepted") }
        catch (error: WorkerPrepareException) { assertTrue(error.code in setOf("prepare_manifest_mismatch", "prepare_artifact_identity_mismatch")) }
    }

    @Test fun `verified manifest binds artifact source without conflating master cross-format identity`() {
        val manifest = manifest()
        val identity = WorkerPrepareIdentity.parse(identity(manifest))
        identity.verifyManifest(manifest)
        assertNotEquals(identity.text("model_id"), manifest.sourceModelId)
        assertNotEquals(identity.text("model_sha256"), manifest.sourceSha256)
        assertEquals(identity.text("source_model_sha256"), manifest.sourceSha256)
    }

    @Test fun `same artifact bytes from another operation or node cannot become loaded-ready`() {
        val first = manifest()
        for ((key, value) in listOf("operation_id" to "supply_22222222222222222222222222222222", "node_id" to "android_other")) {
            val second = resign(first.json.deepCopy().apply { addProperty(key, value) })
            assertEquals(first.artifactSha256, second.artifactSha256)
            val proposed = identity(first).apply { addProperty("manifest_sha256", second.manifestSha256) }
            rejected { WorkerPrepareIdentity.parse(proposed).verifyManifest(second) }
        }
    }

    @Test fun `wrong manifest digest source tokenizer shape or stage model id is rejected`() {
        val manifest = manifest()
        val changes = listOf<(JsonObject) -> Unit>(
            { it.addProperty("manifest_sha256", "f".repeat(64)) },
            { it.addProperty("source_model_sha256", "f".repeat(64)) },
            { it.addProperty("source_model_id", "other_source") },
            { it.addProperty("tokenizer_sha256", "f".repeat(64)) },
            { it.addProperty("hidden_size", 2) },
            { it.add("layer_range", JsonParser.parseString("[1,4]")) },
            { it.addProperty("total_layers", 5); it.addProperty("segment_mode", "middle") },
            { it.getAsJsonObject("stage_model_identity").addProperty("model_id", "other.gguf") },
        )
        for (change in changes) rejected { WorkerPrepareIdentity.parse(identity(manifest).apply(change)).verifyManifest(manifest) }
    }
}
