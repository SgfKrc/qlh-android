package com.qlh.inference.service

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.qlh.inference.network.canonicalHost
import java.io.File
import java.io.FileOutputStream
import java.nio.file.Files
import java.nio.file.StandardCopyOption

internal fun supplyBindingSha256(cluster: String, host: String, port: Int, epoch: Int): String =
    supplySha256(canonicalSupplyJson(JsonObject().apply {
        addProperty("cluster", cluster); addProperty("host", canonicalHost(host))
        addProperty("port", port); addProperty("epoch", epoch)
    }).toByteArray(Charsets.UTF_8))

/** Only an explicitly registered operation is current; published directories are reuse storage. */
internal class ModelSupplyCurrent(private val root: File) {
    private val pointer get() = File(root, POINTER)
    internal data class Registration(val manifest: ModelSupplyManifest, val previous: ByteArray?)

    fun read(bindingSha256: String): ModelSupplyManifest? = synchronized(lock) {
        val bytes = readBytes() ?: return@synchronized null
        val record = JsonParser.parseString(bytes.toString(Charsets.UTF_8)).asJsonObject
        checkSupply(record.keySet() == setOf("binding_sha256", "manifest"), "SUPPLY_CURRENT_INVALID", "current fields")
        if (ModelSupplyManifest.text(record, "binding_sha256") != bindingSha256) return@synchronized null
        ModelSupplyManifest.parse(record.getAsJsonObject("manifest"))
    }

    fun select(manifest: ModelSupplyManifest, bindingSha256: String): Registration = synchronized(lock) {
        val directory = directory(manifest)
        val marker = File(directory, ModelSupplyManifest.MARKER)
        checkSupply(!Files.isSymbolicLink(marker.toPath()) && marker.isFile && marker.length() <= LIMIT &&
            ModelSupplyManifest.parse(marker.readText(Charsets.UTF_8)).json == manifest.json,
            "SUPPLY_MANIFEST_DIGEST_MISMATCH", "current must refer to the verified published operation")
        val previous = readBytes()
        val record = JsonObject().apply {
            addProperty("binding_sha256", bindingSha256); add("manifest", manifest.json.deepCopy())
        }
        write(canonicalSupplyJson(record).toByteArray(Charsets.UTF_8))
        Registration(manifest, previous)
    }

    /** A failed or retired registration cannot be recovered as current after restart. */
    fun rollback(registration: Registration, bindingSha256: String) = synchronized(lock) {
        if (read(bindingSha256)?.manifestSha256 == registration.manifest.manifestSha256) {
            if (registration.previous == null) Files.deleteIfExists(pointer.toPath())
            else write(registration.previous)
        }
    }

    fun directory(manifest: ModelSupplyManifest): File {
        val directory = File(root, "segment-${manifest.manifestSha256}")
        checkSupply(!Files.isSymbolicLink(root.toPath()) && !Files.isSymbolicLink(directory.toPath()) &&
            directory.canonicalFile.parentFile == root.canonicalFile,
            "SUPPLY_UNSAFE_PATH", "current directory escaped")
        return directory
    }

    private fun readBytes(): ByteArray? {
        checkSupply(!Files.isSymbolicLink(root.toPath()) && !Files.isSymbolicLink(pointer.toPath()),
            "SUPPLY_UNSAFE_PATH", "current pointer escaped")
        if (!pointer.exists()) return null
        checkSupply(pointer.isFile && pointer.length() <= LIMIT, "SUPPLY_CURRENT_INVALID", "current pointer invalid")
        return pointer.readBytes()
    }

    private fun write(bytes: ByteArray) {
        checkSupply(bytes.size <= LIMIT, "SUPPLY_CURRENT_INVALID", "current pointer exceeds limit")
        val temporary = File(root, ".current.part")
        checkSupply(!Files.isSymbolicLink(temporary.toPath()) && !Files.isSymbolicLink(pointer.toPath()),
            "SUPPLY_UNSAFE_PATH", "current pointer escaped")
        FileOutputStream(temporary).use { it.write(bytes); it.fd.sync() }
        Files.move(temporary.toPath(), pointer.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
    }

    companion object {
        const val POINTER = "current.supply.json"
        private const val LIMIT = 4 * 1024 * 1024L
        private val lock = Any()
    }
}

/** Reserve the current range even if its verification fails; never fall through to another SHA. */
internal fun preferCurrentSupplyCandidates(
    candidates: List<LayerArtifactCandidate>,
    current: ModelSupplyManifest?,
    reference: String,
): List<LayerArtifactCandidate> {
    if (current == null) return candidates
    return candidates.filter { candidate ->
        val d = candidate.descriptor
        if (d.startLayer != current.start || d.endLayerExclusive != current.end) true
        else candidate.reference == reference && d.artifactName == current.weight.path &&
            d.artifactSha256 == current.artifactSha256 && d.sourceModelSha256 == current.sourceSha256 &&
            d.sourceModelId == current.sourceModelId && d.mode == current.json["segment_mode"].asString &&
            d.hiddenSize == current.json["hidden_size"].asInt && d.tokenizerSha256 == current.json["tokenizer_sha256"].asString
    }
}
