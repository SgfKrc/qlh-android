package com.qlh.inference.service

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import java.io.File
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.attribute.BasicFileAttributes
import java.security.MessageDigest

/** Bounded collection of reproducible segments and partials covered by a completed receiver manifest. */
internal class ModelSupplyGc(
    private val root: File,
    private val limits: Limits = Limits(),
    private val nanoTime: () -> Long = System::nanoTime,
) {
    data class Limits(
        val scanEntries: Int = 64,
        // Shared directory deletion budget, including retired receive directories.
        val deleteSegments: Int = 8,
        val verifyBytes: Long = 16L * 1024 * 1024 * 1024,
        val elapsedNanos: Long = 30_000_000_000L,
    ) {
        init { require(scanEntries > 0 && deleteSegments > 0 && verifyBytes >= 0 && elapsedNanos > 0) }
    }
    data class Result(
        val scanned: Int, val deleted: Int, val verifiedBytes: Long,
        val deletedPartials: Int = 0, val reclaimedPartialBytes: Long = 0,
    ) {
        fun logMessage(): String = "gc_summary scanned=$scanned deleted=$deleted verified_bytes=$verifiedBytes" +
            " deleted_partials=$deletedPartials reclaimed_partial_bytes=$reclaimedPartialBytes"
    }
    private data class Snapshot(val file: File, val attributes: BasicFileAttributes)
    private data class Proof(
        val directory: File, val attributes: BasicFileAttributes, val snapshots: List<Snapshot>,
    )
    private class BudgetExhausted : Exception()

    suspend fun collect(
        nodeId: String,
        stillCurrent: suspend () -> Boolean = { true },
        commit: suspend (() -> Unit) -> Unit = { it() },
        // Only supplied after receive.validateFiles, publication, registration and the terminal reply succeed.
        completedManifest: ModelSupplyManifest? = null,
    ): Result {
        var scanned = 0
        var deleted = 0
        var deletedPartials = 0
        var reclaimedPartialBytes = 0L
        var verifiedBytes = 0L
        val started = nanoTime()
        val context = currentCoroutineContext()
        suspend fun live() {
            context.ensureActive()
            if (nanoTime() - started >= limits.elapsedNanos || !stillCurrent()) throw BudgetExhausted()
        }
        fun requireBytes(size: Long) {
            if (size > limits.verifyBytes - verifiedBytes) throw BudgetExhausted()
        }
        suspend fun inventory(directory: File, allowed: Set<String>, exact: Boolean): Proof {
            ModelSupplyPaths.check(directory)
            val before = attributes(directory)
            checkSupply(before.isDirectory, "SUPPLY_GC_PROOF_CHANGED", "directory unavailable")
            val snapshots = mutableListOf<Snapshot>()
            Files.newDirectoryStream(directory.toPath()).use { children ->
                for (child in children) {
                    live()
                    checkSupply(snapshots.size < allowed.size && child.fileName.toString() in allowed,
                        "SUPPLY_GC_PROOF_CHANGED", "unproved directory entry")
                    val file = ModelSupplyPaths.child(directory, child.fileName.toString())
                    val attr = attributes(file)
                    checkSupply(attr.isRegularFile, "SUPPLY_GC_PROOF_CHANGED", "unproved file type")
                    snapshots += Snapshot(file, attr)
                }
            }
            checkSupply(!exact || snapshots.size == allowed.size, "SUPPLY_GC_PROOF_CHANGED", "incomplete inventory")
            return Proof(directory, before, snapshots)
        }
        suspend fun hash(file: File, length: Long, exact: Boolean = true): String {
            requireBytes(length)
            val before = attributes(ModelSupplyPaths.child(requireNotNull(file.parentFile), file.name))
            checkSupply(before.isRegularFile && (if (exact) before.size() == length else before.size() >= length),
                "SUPPLY_GC_PROOF_CHANGED", "file size differs")
            val digest = MessageDigest.getInstance("SHA-256")
            Files.newInputStream(file.toPath(), java.nio.file.StandardOpenOption.READ, LinkOption.NOFOLLOW_LINKS).use { input ->
                val buffer = ByteArray(1024 * 1024)
                var remaining = length
                while (remaining > 0) {
                    live()
                    val n = input.read(buffer, 0, minOf(buffer.size.toLong(), remaining).toInt())
                    checkSupply(n > 0, "SUPPLY_GC_PROOF_CHANGED", "file shortened")
                    digest.update(buffer, 0, n); remaining -= n; verifiedBytes += n
                }
                checkSupply(!exact || input.read() == -1, "SUPPLY_GC_PROOF_CHANGED", "file grew")
            }
            checkSupply(unchanged(before, attributes(file)), "SUPPLY_GC_PROOF_CHANGED", "file mutated")
            return digest.digest().joinToString("") { "%02x".format(it.toInt() and 255) }
        }
        suspend fun verifyCompleted(directory: File, manifest: ModelSupplyManifest): Proof {
            val proof = inventory(directory, manifest.files.map { it.path }.toSet() + ModelSupplyManifest.MARKER, true)
            checkSupply(readManifest(directory) == manifest, "SUPPLY_GC_PROOF_CHANGED", "marker differs")
            requireBytes(manifest.files.fold(0L) { n, f -> Math.addExact(n, f.size) })
            for (file in manifest.files) checkSupply(
                hash(ModelSupplyPaths.child(directory, file.path), file.size) == file.sha256,
                "SUPPLY_GC_PROOF_CHANGED", "completed file digest differs")
            return proof
        }
        var covering: Proof? = null
        var coveringLease: ModelSupplyStorage.Lease? = null
        try {
            ModelSupplyPaths.check(root)
            if (!root.isDirectory) return Result(0, 0, 0)
            if (completedManifest?.nodeId == nodeId) {
                coveringLease = runCatching {
                    ModelSupplyStorage.transfer(root, ModelSupplyPaths.child(root, "segment-${completedManifest.manifestSha256}"))
                }.getOrNull()
            }
            Files.newDirectoryStream(root.toPath()).use { entries ->
                for (entry in entries) {
                    if (scanned >= limits.scanEntries || deleted >= limits.deleteSegments) break
                    live(); scanned++
                    val directory = entry.toFile()
                    val partial = Regex("\\.receive-[0-9a-f]{64}").matches(directory.name)
                    if (!partial && !Regex("segment-[0-9a-f]{64}").matches(directory.name)) continue
                    try {
                        ModelSupplyPaths.check(directory)
                        if (!directory.isDirectory || synchronized(ModelSupplyStorage.lock) { protected(directory) }) continue
                        val proof: Proof
                        var resume: ModelSupplyReceiver.ResumeData? = null
                        var manifest: ModelSupplyManifest? = null
                        if (partial) {
                            val completed = completedManifest ?: continue
                            if (coveringLease == null) continue
                            resume = ModelSupplyReceiver.readResume(directory) ?: continue
                            val old = resume.manifest
                            if (old.nodeId != nodeId || old.operationId == completed.operationId ||
                                !old.sameContentIdentity(completed) || old.files.toSet() != completed.files.toSet()) continue
                            // A successful fresh registration and its complete marker are both required.
                            if (synchronized(ModelSupplyStorage.lock) {
                                    ModelSupplyCurrent(root).retainedManifestSha256() != completed.manifestSha256
                                }) continue
                            val completeDirectory = ModelSupplyPaths.child(root, "segment-${completed.manifestSha256}")
                            if (covering == null) covering = verifyCompleted(completeDirectory, completed)
                            val allowed = old.files.flatMap { listOf(it.path, it.path + ".part") }.toSet() + ModelSupplyReceiver.RESUME_MARKER
                            proof = inventory(directory, allowed, false)
                            checkSupply(proof.snapshots.any { it.file.name == ModelSupplyReceiver.RESUME_MARKER },
                                "SUPPLY_GC_PROOF_CHANGED", "missing journal")
                            for (snapshot in proof.snapshots) {
                                val name = snapshot.file.name
                                if (name == ModelSupplyReceiver.RESUME_MARKER) continue
                                val completeFile = old.files.find { it.path == name }
                                if (completeFile != null) {
                                    checkSupply(hash(snapshot.file, completeFile.size) == completeFile.sha256,
                                        "SUPPLY_GC_PROOF_CHANGED", "staged file digest differs")
                                } else {
                                    val file = old.files.single { it.path + ".part" == name }
                                    val record = resume.records[file.path] ?: throw ModelSupplyException(
                                        "SUPPLY_GC_PROOF_CHANGED", "unproved partial")
                                    // Uncheckpointed suffixes remain unknown data and are deliberately retained.
                                    requireBytes(Math.multiplyExact(record.length, 2))
                                    checkSupply(hash(snapshot.file, record.length) == record.sha256 &&
                                        hash(ModelSupplyPaths.child(completeDirectory, file.path), record.length, false) == record.sha256,
                                        "SUPPLY_GC_PROOF_CHANGED", "prefix not covered")
                                }
                            }
                        } else {
                            manifest = readManifest(directory)
                            if (manifest.nodeId != nodeId || directory.name != "segment-${manifest.manifestSha256}") continue
                            // Preserve the existing skip behavior for a segment exceeding the remaining byte budget.
                            if (manifest.files.fold(0L) { n, f -> Math.addExact(n, f.size) } > limits.verifyBytes - verifiedBytes) continue
                            proof = verifyCompleted(directory, manifest)
                        }
                        live()
                        commit { synchronized(ModelSupplyStorage.lock) {
                            context.ensureActive()
                            if (nanoTime() - started >= limits.elapsedNanos) throw BudgetExhausted()
                            // This fence is shared with receiver/donor lease acquisition; hashes never hold it.
                            if (!protected(directory) && intact(proof) &&
                                (if (partial) {
                                    ModelSupplyReceiver.readResume(directory) == resume && covering != null && intact(covering!!) &&
                                        ModelSupplyCurrent(root).retainedManifestSha256() == completedManifest!!.manifestSha256 &&
                                        readManifest(covering!!.directory) == completedManifest
                                } else readManifest(directory) == manifest)) {
                                context.ensureActive()
                                val marker = if (partial) ModelSupplyReceiver.RESUME_MARKER else ModelSupplyManifest.MARKER
                                for (snapshot in proof.snapshots.sortedBy { it.file.name == marker }) {
                                    context.ensureActive()
                                    if (nanoTime() - started >= limits.elapsedNanos) throw BudgetExhausted()
                                    Files.delete(ModelSupplyPaths.child(directory, snapshot.file.name).toPath())
                                }
                                ModelSupplyPaths.check(directory)
                                Files.delete(directory.toPath())
                                deleted++
                                if (partial) {
                                    deletedPartials++
                                    reclaimedPartialBytes += proof.snapshots.sumOf { it.attributes.size() }
                                }
                            }
                        } }
                    } catch (cancelled: CancellationException) { throw cancelled }
                    catch (exhausted: BudgetExhausted) { throw exhausted }
                    catch (_: Exception) { /* Unknown proof, mutation or storage error: retain. */ }
                }
            }
        } catch (cancelled: CancellationException) { throw cancelled }
        catch (_: Exception) { /* Invalid current pointer or exhausted budget cannot authorize deletion. */ }
        finally { coveringLease?.close() }
        return Result(scanned, deleted, verifiedBytes, deletedPartials, reclaimedPartialBytes)
    }

    /** Called only under the storage deletion fence. Also checks directory replacement and exact inventory. */
    private fun intact(proof: Proof): Boolean {
        val directory = ModelSupplyPaths.child(root, proof.directory.name)
        val after = attributes(directory)
        if (!after.isDirectory || after.isOther || after.isSymbolicLink ||
            after.fileKey() != proof.attributes.fileKey() || after.creationTime() != proof.attributes.creationTime()) return false
        val expected = proof.snapshots.map { it.file.name }.toSet()
        val names = mutableSetOf<String>()
        Files.newDirectoryStream(directory.toPath()).use { children ->
            for (child in children) {
                if (names.size >= expected.size || child.fileName.toString() !in expected) return false
                names += child.fileName.toString()
            }
        }
        return names == expected && proof.snapshots.all {
            unchanged(it.attributes, attributes(ModelSupplyPaths.child(directory, it.file.name)))
        }
    }
    private fun readManifest(directory: File): ModelSupplyManifest {
        val marker = ModelSupplyPaths.child(directory, ModelSupplyManifest.MARKER)
        checkSupply(marker.isFile && marker.length() <= MARKER_LIMIT, "SUPPLY_GC_PROOF_CHANGED", "marker unavailable")
        return ModelSupplyManifest.parse(marker.readText(Charsets.UTF_8))
    }
    private fun protected(directory: File): Boolean {
        if (ModelSupplyStorage.retained(root, directory)) return true
        return ModelSupplyCurrent(root).retainedManifestSha256() == directory.name.removePrefix("segment-")
    }
    private fun attributes(file: File): BasicFileAttributes =
        Files.readAttributes(file.toPath(), BasicFileAttributes::class.java, LinkOption.NOFOLLOW_LINKS)
    private fun unchanged(before: BasicFileAttributes, after: BasicFileAttributes): Boolean =
        after.isRegularFile && !after.isOther && !after.isSymbolicLink && before.size() == after.size() &&
            before.lastModifiedTime() == after.lastModifiedTime() && before.fileKey() == after.fileKey() &&
            before.creationTime() == after.creationTime()

    private companion object { const val MARKER_LIMIT = 4 * 1024 * 1024L }
}
