package com.qlh.inference.service

import com.google.gson.JsonArray
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import com.qlh.inference.network.httpBaseUrl
import com.qlh.inference.worker.GgufLayerWeightReader
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import okhttp3.Call
import okhttp3.HttpUrl
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.ResponseBody
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.RandomAccessFile
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.StandardCopyOption
import java.nio.file.attribute.BasicFileAttributes
import java.security.MessageDigest
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

internal data class ModelSupplyProgress(
    val phase: String,
    val downloadedBytes: Long,
    val totalBytes: Long,
    val actualDownloadedBytes: Long,
    val reusedBytes: Long,
    val resumedBytes: Long,
)

/** Numeric, local transfer timings only: no manifest, endpoint, token or filesystem fields. */
internal data class ModelSupplyTransferDiagnostic(
    val elapsedNanos: Long,
    val readNanos: Long,
    val liveCheckNanos: Long,
    val writeHashNanos: Long,
    val checkpointNanos: Long,
    // Includes the controller's existing progress fence and synchronous control send.
    val progressNanos: Long,
    val actualNewBytes: Long,
    val actualTotalBytes: Long,
    val terminal: Boolean,
) {
    fun logMessage(): String = "transfer_perf elapsed_ms=${elapsedNanos / 1_000_000}" +
        " read_ms=${readNanos / 1_000_000} live_check_ms=${liveCheckNanos / 1_000_000}" +
        " write_hash_ms=${writeHashNanos / 1_000_000} checkpoint_ms=${checkpointNanos / 1_000_000}" +
        " progress_ms=${progressNanos / 1_000_000} actual_new_bytes=$actualNewBytes" +
        " actual_total_bytes=$actualTotalBytes terminal=${if (terminal) 1 else 0}"
}

/** Owned by the receiving coroutine; diagnostic callbacks are observational only. */
private class TransferDiagnostics(
    private val nanoTime: () -> Long,
    private val report: (ModelSupplyTransferDiagnostic) -> Unit,
) {
    enum class Stage { READ, LIVE_CHECK, WRITE_HASH, CHECKPOINT, PROGRESS }
    private val elapsed = LongArray(Stage.entries.size)
    private var windowStart = nanoTime()
    private var previousActual = 0L
    private var active = false

    fun start(): Long = nanoTime()

    fun record(stage: Stage, start: Long) {
        elapsed[stage.ordinal] += (nanoTime() - start).coerceAtLeast(0)
        active = true
    }

    fun emit(actual: Long, terminal: Boolean = false) {
        val now = nanoTime()
        val duration = (now - windowStart).coerceAtLeast(0)
        if (!active || (!terminal && duration < 5_000_000_000L)) return
        val snapshot = ModelSupplyTransferDiagnostic(
            duration, elapsed[0], elapsed[1], elapsed[2], elapsed[3], elapsed[4],
            actual - previousActual, actual, terminal,
        )
        windowStart = now
        previousActual = actual
        elapsed.fill(0)
        active = false
        // A logging failure must never change resume durability or the transfer outcome.
        try { report(snapshot) } catch (_: Exception) { }
    }
}

/** One cancellable transfer, confined to a bound master's origin and private storage. */
internal class ModelSupplyReceiver(
    private val root: File,
    private val boundHost: String,
    private val checkpointIntervalBytes: Long = 64L * 1024 * 1024,
    readTimeoutMillis: Long = 60_000,
    private val client: OkHttpClient = OkHttpClient.Builder().followRedirects(false).followSslRedirects(false)
        .connectTimeout(15, TimeUnit.SECONDS).readTimeout(readTimeoutMillis, TimeUnit.MILLISECONDS).build(),
    private val nanoTime: () -> Long = System::nanoTime,
    private val digests: VerifiedPrivateModelDigests = VerifiedPrivateModelDigests.shared,
    private val createLink: (File, File) -> Unit = { source, target -> Files.createLink(target.toPath(), source.toPath()) },
) {
    init { require(checkpointIntervalBytes > 0 && readTimeoutMillis > 0) }
    private val call = AtomicReference<Call?>()
    fun cancel() { call.get()?.cancel() }

    suspend fun receive(
        manifest: ModelSupplyManifest,
        apiPort: Int,
        token: String,
        stillCurrent: suspend () -> Boolean,
        progress: suspend (ModelSupplyProgress) -> Unit,
        publish: suspend (() -> Unit) -> Unit,
        diagnostics: (ModelSupplyTransferDiagnostic) -> Unit = {},
        allowLocalCutFallback: Boolean = false,
        localWeight: (suspend (File, suspend (Long) -> Unit) -> Unit)? = null,
    ): File {
        checkSupply(apiPort in 1..65535 && token.isNotBlank() && token.length <= 4096 && token.none { it.code < 32 || it.code == 127 },
            "SUPPLY_INVALID_REQUEST", "port or token")
        checkSupply(stillCurrent(), "SUPPLY_CANCELLED", "binding changed")
        ensureDirectory(root)
        val transfer = TransferState(
            manifest.files.fold(0L) { n, file -> Math.addExact(n, file.size) },
            progress, TransferDiagnostics(nanoTime, diagnostics),
        )
        val receiving = ModelSupplyStorage.pending(root)
        var stageLease: ModelSupplyStorage.Lease? = null
        try {
            val final = confined("segment-${manifest.manifestSha256}")
            if (final.exists()) {
                val marker = safeFile(final, ModelSupplyManifest.MARKER)
                checkSupply(marker.isFile && marker.length() <= 4 * 1024 * 1024 &&
                    ModelSupplyManifest.parse(marker.readText(Charsets.UTF_8)).json == manifest.json,
                    "SUPPLY_MANIFEST_DIGEST_MISMATCH", "published marker differs")
                transfer.report("verifying", 0)
                validateFiles(final, manifest)
                checkSupply(stillCurrent(), "SUPPLY_CANCELLED", "binding changed")
                transfer.completed = transfer.total; transfer.reused = transfer.total
                transfer.report("registering")
                return final
            }
            val stage = confined(".receive-${manifest.manifestSha256}")
            synchronized(ModelSupplyStorage.lock) {
                ensureDirectory(stage)
                stageLease = ModelSupplyStorage.transfer(root, stage)
            }
            val journal = ResumeJournal(stage, manifest)
            val inventory = receiveInventory(manifest)
            // Staged targets are byte donors only; their validated source marker never publishes them.
            val candidates = (scanSuppliedSegmentRoots(root).filter { (_, existing) ->
                manifest.sameContentIdentity(existing)
            }.flatMap { (directory, existing) ->
                existing.files.map { ReuseCandidate(directory, it) }
            } + inventory.complete).groupBy { fileKey(it.file) }
            for (file in manifest.files) {
                currentCoroutineContext().ensureActive()
                checkSupply(stillCurrent(), "SUPPLY_CANCELLED", "binding changed")
                val target = safeFile(stage, file.path)
                val partial = safeFile(stage, file.path + ".part")
                if (target.exists() && matches(target, file)) {
                    currentCoroutineContext().ensureActive()
                    checkSupply(stillCurrent(), "SUPPLY_CANCELLED", "binding changed")
                    journal.recordSource()
                    transfer.completed += file.size; transfer.reused += file.size
                    Files.deleteIfExists(partial.toPath())
                    transfer.report("reusing")
                    continue
                }
                var reusedFile = false
                for ((directory, entry) in candidates[fileKey(file)].orEmpty()) {
                    val donor = runCatching { ModelSupplyStorage.transfer(root, directory) }.getOrNull() ?: continue
                    try {
                        val candidate = runCatching { safeFile(directory, entry.path) }.getOrNull() ?: continue
                        transfer.report("verifying_reuse")
                        val verified = verifyFile(candidate, file) ?: continue
                        currentCoroutineContext().ensureActive()
                        checkSupply(stillCurrent(), "SUPPLY_CANCELLED", "binding changed")
                        // Never truncate a target that may already be a link to a published file.
                        Files.deleteIfExists(target.toPath())
                        val linked = try {
                            digests.linkVerified(candidate, target, verified, createLink)
                        } catch (_: IOException) {
                            copyReusable(candidate, partial, file.size, stillCurrent)
                            currentCoroutineContext().ensureActive()
                            checkSupply(stillCurrent(), "SUPPLY_CANCELLED", "binding changed")
                            Files.move(partial.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
                            false
                        } catch (_: UnsupportedOperationException) {
                            copyReusable(candidate, partial, file.size, stillCurrent)
                            currentCoroutineContext().ensureActive()
                            checkSupply(stillCurrent(), "SUPPLY_CANCELLED", "binding changed")
                            Files.move(partial.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
                            false
                        }
                        // Recheck live bytes after link/copy, including changes during verification.
                        if (!matches(target, file, fresh = !linked)) {
                            Files.deleteIfExists(target.toPath())
                            continue
                        }
                        currentCoroutineContext().ensureActive()
                        checkSupply(stillCurrent(), "SUPPLY_CANCELLED", "binding changed")
                        journal.recordSource()
                        Files.deleteIfExists(partial.toPath())
                        transfer.completed += file.size; transfer.reused += file.size
                        transfer.report("reusing")
                        reusedFile = true
                        break
                    } finally { donor.close() }
                }
                if (reusedFile) continue
                transfer.report("verifying_resume")
                var prefix = trustedPrefix(journal, file, partial, stillCurrent)
                prefix = importPrefix(inventory.prefixes[fileKey(file)].orEmpty(), file, partial, prefix, journal, stillCurrent)
                val offset = prefix.length
                val completedPrefix = if (file.size > 0 && offset == file.size) verifyFile(partial, file) else null
                if (completedPrefix != null) {
                    currentCoroutineContext().ensureActive()
                    checkSupply(stillCurrent(), "SUPPLY_CANCELLED", "binding changed")
                    journal.recordSource()
                    digests.moveVerified(partial, target, completedPrefix)
                    transfer.completed += file.size; transfer.reused += file.size
                    transfer.report("reusing")
                    continue
                }
                if (file.size > 0 && offset == file.size) {
                    Files.deleteIfExists(partial.toPath())
                    throw ModelSupplyException("SUPPLY_ARTIFACT_DIGEST_MISMATCH", "complete prefix digest mismatch")
                }
                checkSupply(root.usableSpace >= file.size - offset, "SUPPLY_DISK_INSUFFICIENT", "insufficient private storage")
                if (file.kind == "weights" && localWeight != null) {
                    transfer.report("building", transfer.completed)
                    // A local build replaces any trusted download prefix; never resume its bytes.
                    journal.forget(file)
                    Files.deleteIfExists(partial.toPath())
                    try {
                        localWeight(partial) { built ->
                            checkSupply(built in 0..file.size, "SUPPLY_ARTIFACT_DIGEST_MISMATCH", "cut size")
                            transfer.report("building", transfer.completed + built)
                        }
                    } catch (error: ModelSupplyException) {
                        if (!allowLocalCutFallback || error.reasonCode !in LOCAL_CUT_FALLBACK_REASONS) throw error
                        currentCoroutineContext().ensureActive()
                        checkSupply(stillCurrent(), "SUPPLY_CANCELLED", "binding changed before fallback")
                        Files.deleteIfExists(partial.toPath())
                        checkSupply(root.usableSpace >= file.size, "SUPPLY_DISK_INSUFFICIENT", "insufficient fallback storage")
                        transfer.report("transfer_fallback_local_cut_unavailable")
                        downloadFile(manifest, file, partial, Prefix(0, MessageDigest.getInstance("SHA-256")),
                            journal, apiPort, token, stillCurrent, transfer)
                    }
                } else if (file.size == 0L) {
                    FileOutputStream(partial).use { it.fd.sync() }
                } else {
                    if (localWeight == null) transfer.report("transfer_fallback_local_cut_unavailable", transfer.completed + offset)
                    downloadFile(manifest, file, partial, prefix, journal, apiPort, token, stillCurrent, transfer)
                }
                transfer.report("verifying", transfer.completed + file.size)
                val verified = verifyFile(partial, file, fresh = true)
                if (verified == null) {
                    Files.deleteIfExists(partial.toPath())
                    throw ModelSupplyException("SUPPLY_ARTIFACT_DIGEST_MISMATCH", "file digest mismatch")
                }
                currentCoroutineContext().ensureActive()
                checkSupply(stillCurrent(), "SUPPLY_CANCELLED", "binding changed")
                journal.recordSource()
                digests.moveVerified(partial, target, verified)
                transfer.completed += file.size
            }
            validateFiles(stage, manifest)
            transfer.report("registering", transfer.total)
            currentCoroutineContext().ensureActive()
            checkSupply(stillCurrent(), "SUPPLY_CANCELLED", "binding changed")
            publish {
                journal.discard()
                FileOutputStream(safeFile(stage, ModelSupplyManifest.MARKER)).use {
                    it.write(canonicalSupplyJson(manifest.json).toByteArray(Charsets.UTF_8)); it.fd.sync()
                }
                Files.move(stage.toPath(), final.toPath(), StandardCopyOption.ATOMIC_MOVE)
            }
            return final
        } finally {
            stageLease?.close()
            receiving.close()
            transfer.diagnostics.emit(transfer.actualDownloaded, terminal = true)
        }
    }


    private class TransferState(
        val total: Long,
        private val progress: suspend (ModelSupplyProgress) -> Unit,
        val diagnostics: TransferDiagnostics,
    ) {
        var completed = 0L
        var actualDownloaded = 0L
        var reused = 0L
        var resumed = 0L

        suspend fun report(phase: String, bytes: Long = completed) {
            progress(ModelSupplyProgress(phase, bytes, total, actualDownloaded, reused, resumed))
        }
    }

    /** Keep HTTP admission and resource cleanup outside the hot transfer state machine. */
    private suspend fun downloadFile(
        manifest: ModelSupplyManifest,
        file: ModelSupplyManifest.SupplyFile,
        partial: File,
        prefix: Prefix,
        journal: ResumeJournal,
        apiPort: Int,
        token: String,
        stillCurrent: suspend () -> Boolean,
        transfer: TransferState,
    ) {
        val offset = prefix.length
        val request = Request.Builder().url(fileUrl(boundHost, apiPort, manifest.operationId, manifest.nodeId, file.path))
            .header("X-QLH-Supply-Token", token).header("Accept-Encoding", "identity")
            .apply { if (offset > 0) header("Range", "bytes=$offset-") }.build()
        val active = client.newCall(request)
        call.set(active)
        try {
            currentCoroutineContext().ensureActive()
            checkTransferCurrent(stillCurrent, transfer.diagnostics)
            val started = transfer.diagnostics.start()
            val response = try { active.execute() }
                finally { transfer.diagnostics.record(TransferDiagnostics.Stage.READ, started) }
            response.use {
                checkSupply(response.code in setOf(200, 206), if (response.code in 300..399) "SUPPLY_REDIRECT_FORBIDDEN" else "SUPPLY_HTTP_${response.code}", "supply HTTP failed")
                val body = response.body ?: throw ModelSupplyException("SUPPLY_HTTP_BODY_MISSING", "empty response")
                val plan = try {
                    planModelDownloadWrite(offset, response.code, body.contentLength(), response.header("Content-Range"), file.size)
                } catch (_: IllegalArgumentException) {
                    throw ModelSupplyException("SUPPLY_RANGE_INVALID", "invalid partial response")
                }
                if (response.code == 206) {
                    val range = Regex("bytes ([0-9]+)-([0-9]+)/([0-9]+)", RegexOption.IGNORE_CASE)
                        .matchEntire(response.header("Content-Range").orEmpty())
                    checkSupply(range != null && range.groupValues[2].toLongOrNull() == file.size - 1,
                        "SUPPLY_RANGE_INVALID", "partial response must cover the declared suffix")
                }
                transfer.resumed += plan.startBytes
                FileOutputStream(partial, plan.append).use { output ->
                    val digest = if (plan.append) prefix.digest else MessageDigest.getInstance("SHA-256")
                    transferBody(body, output, partial, file, journal, digest,
                        plan.startBytes, active, stillCurrent, transfer)
                }
            }
        } finally { call.compareAndSet(active, null) }
    }

    /** Non-inline so nested use/catch cleanup cannot expand receive's coroutine state machine. */
    private suspend fun transferBody(
        body: ResponseBody,
        output: FileOutputStream,
        partial: File,
        file: ModelSupplyManifest.SupplyFile,
        journal: ResumeJournal,
        digest: MessageDigest,
        startBytes: Long,
        active: Call,
        stillCurrent: suspend () -> Boolean,
        transfer: TransferState,
    ) {
        var downloaded = startBytes
        var checkpointAt = startBytes
        val timing = transfer.diagnostics
        try {
            body.byteStream().use { input ->
                // Bound live participation checks by completed chunks, not OkHttp short reads.
                val buffer = ByteArray(minOf(DOWNLOAD_CHUNK_BYTES.toLong(), checkpointIntervalBytes).toInt())
                while (true) {
                    currentCoroutineContext().ensureActive()
                    checkTransferCurrent(stillCurrent, timing)
                    var count = 0
                    var readFailure: IOException? = null
                    while (count < buffer.size) {
                        currentCoroutineContext().ensureActive()
                        val started = timing.start()
                        val read = try { input.read(buffer, count, buffer.size - count) }
                            catch (error: IOException) { readFailure = error; break }
                            finally { timing.record(TransferDiagnostics.Stage.READ, started) }
                        if (read < 0) break
                        count += read
                        transfer.actualDownloaded = Math.addExact(transfer.actualDownloaded, read.toLong())
                    }
                    if (count == 0) {
                        readFailure?.let { throw it }
                        break
                    }
                    currentCoroutineContext().ensureActive()
                    checkTransferCurrent(stillCurrent, timing)
                    if (active.isCanceled()) throw IOException("supply HTTP cancelled")
                    checkSupply(count.toLong() <= file.size - downloaded, "SUPPLY_SIZE_MISMATCH", "response exceeds declared size")
                    val started = timing.start()
                    try {
                        output.write(buffer, 0, count)
                        digest.update(buffer, 0, count)
                        downloaded += count
                    } finally { timing.record(TransferDiagnostics.Stage.WRITE_HASH, started) }
                    if (downloaded < file.size && downloaded - checkpointAt >= checkpointIntervalBytes) {
                        checkpoint(output, journal, file, downloaded, digest, timing)
                        checkpointAt = downloaded
                    }
                    reportDownload(transfer, transfer.completed + downloaded)
                    // A short prefix read before an IO error is safe only after the live write fence.
                    // The outer catch syncs and journals these written bytes before rethrowing.
                    readFailure?.let {
                        currentCoroutineContext().ensureActive()
                        checkTransferCurrent(stillCurrent, timing)
                        throw it
                    }
                }
            }
            val started = timing.start()
            try { output.fd.sync() }
                finally { timing.record(TransferDiagnostics.Stage.CHECKPOINT, started) }
            checkSupply(downloaded == file.size, "SUPPLY_SIZE_MISMATCH", "incomplete response")
        } catch (error: Exception) {
            // Save the durable prefix before any suspending terminal progress callback.
            if (downloaded > checkpointAt && partial.length() == downloaded) {
                try { checkpoint(output, journal, file, downloaded, digest, timing) }
                catch (checkpointError: Exception) { error.addSuppressed(checkpointError) }
            }
            // Includes bytes read into a discarded chunk, recorded before controller live fences.
            try { reportDownload(transfer, transfer.completed + downloaded) }
            catch (progressError: Exception) {
                if (progressError !== error) error.addSuppressed(progressError)
            }
            throw error
        }
    }

    private suspend fun checkTransferCurrent(stillCurrent: suspend () -> Boolean, timing: TransferDiagnostics) {
        val started = timing.start()
        try { checkSupply(stillCurrent(), "SUPPLY_CANCELLED", "binding changed") }
        finally { timing.record(TransferDiagnostics.Stage.LIVE_CHECK, started) }
    }

    private suspend fun reportDownload(transfer: TransferState, bytes: Long) {
        val started = transfer.diagnostics.start()
        try { transfer.report("downloading", bytes) }
        finally {
            transfer.diagnostics.record(TransferDiagnostics.Stage.PROGRESS, started)
            // Sample at existing chunk boundaries; no timer job or asynchronous control send.
            transfer.diagnostics.emit(transfer.actualDownloaded)
        }
    }

    private fun checkpoint(
        output: FileOutputStream,
        journal: ResumeJournal,
        file: ModelSupplyManifest.SupplyFile,
        downloaded: Long,
        digest: MessageDigest,
        timing: TransferDiagnostics,
    ) {
        val started = timing.start()
        try {
            output.fd.sync()
            journal.checkpoint(file, downloaded, digestSnapshot(digest))
        } finally { timing.record(TransferDiagnostics.Stage.CHECKPOINT, started) }
    }

    private data class Prefix(val length: Long, val digest: MessageDigest)
    internal data class Checkpoint(val path: String, val length: Long, val sha256: String)
    internal data class ResumeData(val manifest: ModelSupplyManifest, val records: Map<String, Checkpoint>)
    private data class PartialCandidate(val stage: File, val file: ModelSupplyManifest.SupplyFile, val record: Checkpoint)
    private data class ReuseCandidate(val directory: File, val file: ModelSupplyManifest.SupplyFile)
    private data class ReceiveInventory(
        val prefixes: Map<Triple<String, Long, String>, List<PartialCandidate>>,
        val complete: List<ReuseCandidate>,
    )

    private fun fileKey(file: ModelSupplyManifest.SupplyFile) = Triple(file.sha256, file.size, file.kind)
    private fun digestHex(digest: MessageDigest) = digest.digest().joinToString("") { "%02x".format(it.toInt() and 255) }
    private fun digestSnapshot(digest: MessageDigest) = digestHex(digest.clone() as MessageDigest)


    private inner class ResumeJournal(val stage: File, val manifest: ModelSupplyManifest) {
        val records = readResume(stage)?.takeIf { it.manifest == manifest }?.records?.toMutableMap() ?: linkedMapOf()

        fun checkpoint(file: ModelSupplyManifest.SupplyFile, length: Long, sha256: String) {
            checkSupply(length in 1..file.size && safeFile(stage, file.path + ".part").length() >= length,
                "SUPPLY_RESUME_INVALID", "checkpoint prefix size")
            save(records + (file.path to Checkpoint(file.path, length, sha256)))
        }

        fun forget(file: ModelSupplyManifest.SupplyFile) {
            if (file.path in records) save(records - file.path)
        }

        // Keep a validated source marker even after .part becomes a complete target.
        fun recordSource() { save(records.toMap()) }

        private fun save(next: Map<String, Checkpoint>) {
            val entries = manifest.files.associateBy { it.path }
            val json = JsonObject().apply {
                addProperty("schema_version", 1); add("manifest", manifest.json.deepCopy())
                add("prefixes", JsonArray().apply {
                    for ((path, record) in next.toSortedMap()) add(JsonObject().apply {
                        addProperty("path", path); addProperty("file_sha256", entries.getValue(path).sha256)
                        addProperty("size_bytes", entries.getValue(path).size)
                        addProperty("prefix_bytes", record.length); addProperty("prefix_sha256", record.sha256)
                    })
                })
                addProperty("journal_sha256", supplySha256(canonicalSupplyJson(this).toByteArray(Charsets.UTF_8)))
            }
            val bytes = canonicalSupplyJson(json).toByteArray(Charsets.UTF_8)
            checkSupply(bytes.size <= MAX_JOURNAL_BYTES, "SUPPLY_RESUME_INVALID", "checkpoint exceeds limit")
            val marker = safeFile(stage, RESUME_MARKER)
            val temporary = File.createTempFile(".supply-checkpoint-", ".tmp", stage)
            try {
                FileOutputStream(temporary).use { it.write(bytes); it.fd.sync() }
                safeFile(stage, temporary.name); safeFile(stage, RESUME_MARKER)
                Files.move(temporary.toPath(), marker.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
                records.clear(); records.putAll(next)
            } finally { Files.deleteIfExists(temporary.toPath()) }
        }

        fun discard() { Files.deleteIfExists(safeFile(stage, RESUME_MARKER).toPath()) }
    }

    private fun receiveInventory(manifest: ModelSupplyManifest): ReceiveInventory {
        val inventory = mutableMapOf<Triple<String, Long, String>, MutableList<PartialCandidate>>()
        val complete = mutableListOf<ReuseCandidate>()
        for (stage in root.listFiles().orEmpty()) {
            if (!Regex("\\.receive-[0-9a-f]{64}").matches(stage.name) || Files.isSymbolicLink(stage.toPath()) || !stage.isDirectory ||
                stage.canonicalFile.parentFile != root.canonicalFile) continue
            val resume = readResume(stage) ?: continue
            // The journal's validated public manifest is its source marker.
            if (!manifest.sameContentIdentity(resume.manifest)) continue
            val files = resume.manifest.files.associateBy { it.path }
            for (file in resume.manifest.files) {
                val target = runCatching { safeFile(stage, file.path) }.getOrNull() ?: continue
                if (target.isFile) complete += ReuseCandidate(stage, file)
            }
            for ((path, record) in resume.records) {
                val file = files.getValue(path)
                inventory.getOrPut(fileKey(file)) { mutableListOf() }.add(PartialCandidate(stage, file, record))
            }
        }
        return ReceiveInventory(inventory.mapValues { (_, candidates) -> candidates.sortedByDescending { it.record.length } }, complete)
    }

    private fun unchanged(before: BasicFileAttributes, after: BasicFileAttributes) = after.isRegularFile &&
        before.size() == after.size() && before.lastModifiedTime() == after.lastModifiedTime() && before.fileKey() == after.fileKey()

    private suspend fun readPrefix(source: File, length: Long, output: FileOutputStream?, stillCurrent: suspend () -> Boolean): MessageDigest? {
        val before = try { Files.readAttributes(source.toPath(), BasicFileAttributes::class.java, LinkOption.NOFOLLOW_LINKS) }
            catch (_: IOException) { return null }
        if (!before.isRegularFile || before.size() < length) return null
        val digest = MessageDigest.getInstance("SHA-256")
        var remaining = length
        Files.newInputStream(source.toPath(), java.nio.file.StandardOpenOption.READ, LinkOption.NOFOLLOW_LINKS).use { input ->
            val buffer = ByteArray(1024 * 1024)
            while (remaining > 0) {
                currentCoroutineContext().ensureActive()
                checkSupply(stillCurrent(), "SUPPLY_CANCELLED", "binding changed")
                val count = input.read(buffer, 0, minOf(buffer.size.toLong(), remaining).toInt())
                if (count < 0) return null
                output?.write(buffer, 0, count)
                digest.update(buffer, 0, count); remaining -= count
            }
        }
        val after = try { Files.readAttributes(source.toPath(), BasicFileAttributes::class.java, LinkOption.NOFOLLOW_LINKS) }
            catch (_: IOException) { return null }
        return digest.takeIf { unchanged(before, after) }
    }

    private suspend fun trustedPrefix(journal: ResumeJournal, file: ModelSupplyManifest.SupplyFile, partial: File, stillCurrent: suspend () -> Boolean): Prefix {
        val record = journal.records[file.path]
        if (record != null && partial.isFile && partial.length() <= file.size) {
            val digest = readPrefix(partial, record.length, null, stillCurrent)
            if (digest != null && digestSnapshot(digest) == record.sha256) {
                // Discard bytes beyond the last durable checkpoint after a crash.
                RandomAccessFile(partial, "rw").use { it.setLength(record.length); it.fd.sync() }
                return Prefix(record.length, digest)
            }
        }
        Files.deleteIfExists(partial.toPath())
        journal.records.remove(file.path)
        return Prefix(0, MessageDigest.getInstance("SHA-256"))
    }

    private suspend fun importPrefix(candidates: List<PartialCandidate>, file: ModelSupplyManifest.SupplyFile, partial: File,
                                     existing: Prefix, journal: ResumeJournal, stillCurrent: suspend () -> Boolean): Prefix {
        for (candidate in candidates) {
            if (candidate.record.length <= existing.length || candidate.stage == journal.stage) continue
            val donor = runCatching { ModelSupplyStorage.transfer(root, candidate.stage) }.getOrNull() ?: continue
            try {
                val source = try { safeFile(candidate.stage, candidate.file.path + ".part") }
                    catch (_: ModelSupplyException) { continue }
                if (!source.isFile || source.length() < candidate.record.length) continue
                checkSupply(root.usableSpace >= candidate.record.length, "SUPPLY_DISK_INSUFFICIENT", "insufficient storage for prefix copy")
                val temporary = File.createTempFile(".supply-prefix-", ".tmp", journal.stage)
                try {
                    val digest = FileOutputStream(temporary).use { output ->
                        val hash = readPrefix(source, candidate.record.length, output, stillCurrent)
                        if (hash != null && digestSnapshot(hash) == candidate.record.sha256) { output.fd.sync(); hash } else null
                    } ?: continue
                    currentCoroutineContext().ensureActive()
                    checkSupply(stillCurrent(), "SUPPLY_CANCELLED", "binding changed")
                    safeFile(journal.stage, partial.name)
                    Files.move(temporary.toPath(), partial.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
                    journal.checkpoint(file, candidate.record.length, candidate.record.sha256)
                    return Prefix(candidate.record.length, digest)
                } finally { Files.deleteIfExists(temporary.toPath()) }
            } finally { donor.close() }
        }
        return existing
    }

    private suspend fun copyReusable(source: File, partial: File, size: Long, stillCurrent: suspend () -> Boolean) {
        checkSupply(root.usableSpace >= size, "SUPPLY_DISK_INSUFFICIENT", "insufficient storage for reuse copy")
        FileOutputStream(partial).use { output ->
            Files.newInputStream(source.toPath(), java.nio.file.StandardOpenOption.READ, LinkOption.NOFOLLOW_LINKS).use { input ->
                val buffer = ByteArray(1024 * 1024)
                var copied = 0L
                while (true) {
                    currentCoroutineContext().ensureActive()
                    checkSupply(stillCurrent(), "SUPPLY_CANCELLED", "binding changed")
                    val n = input.read(buffer)
                    if (n < 0) break
                    checkSupply(n.toLong() <= size - copied, "SUPPLY_ARTIFACT_DIGEST_MISMATCH", "reuse source grew")
                    output.write(buffer, 0, n); copied += n
                }
            }
            output.fd.sync()
        }
    }

    internal suspend fun validateFiles(directory: File, manifest: ModelSupplyManifest) {
        for (file in manifest.files) checkSupply(matches(safeFile(directory, file.path), file), "SUPPLY_ARTIFACT_DIGEST_MISMATCH", "file verification failed")
        val nativeFile = safeFile(directory, manifest.native.path)
        checkSupply(nativeFile.length() <= 4 * 1024 * 1024, "SUPPLY_INVALID_MANIFEST", "native manifest exceeds limit")
        val native = JsonParser.parseString(nativeFile.readText(Charsets.UTF_8)).asJsonObject
        val expected = mapOf("source_layer_range" to manifest.json["layer_range"], "n_layer" to manifest.json["total_layers"],
            "source_model_id" to manifest.json["source_model_id"], "source_model_sha256" to manifest.json["source_model_sha256"],
            "tokenizer_sha256" to manifest.json["tokenizer_sha256"], "artifact_sha256" to manifest.json["artifact_sha256"],
            "hidden_size" to manifest.json["hidden_size"], "conversion_proof" to manifest.json["conversion_proof"],
            "layer_weight_bytes" to manifest.json["layer_weight_bytes"])
        checkSupply(expected.all { (key, value) -> native[key] == value } &&
            native["kept_block_count"]?.asInt == manifest.end - manifest.start,
            "SUPPLY_NATIVE_MANIFEST_MISMATCH", "native identity differs from supply")
        val descriptor = LayerArtifactManifestParser.parse(native.toString(), manifest.native.path).getOrElse {
            throw ModelSupplyException("SUPPLY_NATIVE_MANIFEST_MISMATCH", "native manifest rejected by catalog")
        }
        checkSupply(descriptor.artifactName == manifest.weight.path && descriptor.mode == manifest.json["segment_mode"].asString &&
            descriptor.startLayer == manifest.start && descriptor.endLayerExclusive == manifest.end &&
            descriptor.hiddenSize == manifest.json["hidden_size"].asInt,
            "SUPPLY_NATIVE_MANIFEST_MISMATCH", "native catalog identity mismatch")
        val scan = safeFile(directory, manifest.weight.path).inputStream().use { GgufLayerWeightReader.scan(it, manifest.end - manifest.start) }
        checkSupply(scan != null && scan.layerWeightBytes == manifest.localLayerBytes, "SUPPLY_LAYER_BYTES_MISMATCH", "GGUF layer bytes differ")
        val metadataDifferences = GgufMetadataContract.differences(scan?.ggufMetadata, native["gguf_metadata"], manifest.end - manifest.start)
        checkSupply(metadataDifferences.isEmpty(), "SUPPLY_GGUF_METADATA_MISMATCH", metadataDifferences.joinToString("; "))
        checkSupply(GgufLayerWeightReader.matchesTokenizerSha256(manifest.json["tokenizer_sha256"].asString, scan),
            "SUPPLY_TOKENIZER_DIGEST_MISMATCH", "GGUF tokenizer differs")
    }

    private suspend fun matches(file: File, expected: ModelSupplyManifest.SupplyFile, fresh: Boolean = false): Boolean =
        verifyFile(file, expected, fresh) != null

    private suspend fun verifyFile(
        file: File,
        expected: ModelSupplyManifest.SupplyFile,
        fresh: Boolean = false,
    ): VerifiedPrivateModelDigests.Verification? {
        val before = try { Files.readAttributes(file.toPath(), BasicFileAttributes::class.java, LinkOption.NOFOLLOW_LINKS) }
            catch (_: IOException) { return null }
        if (!before.isRegularFile || before.size() != expected.size) return null
        val verified = try { digests.verify(file, fresh = fresh) } catch (_: IOException) { return null }
        return verified.takeIf { it.digest == expected.sha256 }
    }
    private fun confined(name: String) = safeFile(root, name)
    private fun safeFile(directory: File, name: String): File = ModelSupplyPaths.child(directory, name)
    private fun ensureDirectory(file: File) {
        ModelSupplyPaths.check(file)
        checkSupply(file.isDirectory || file.mkdirs(), "SUPPLY_STORAGE_UNAVAILABLE", "private directory unavailable")
        ModelSupplyPaths.check(file)
    }
    companion object {
        internal fun readResume(stage: File): ResumeData? = try {
            val marker = ModelSupplyPaths.child(stage, RESUME_MARKER)
            if (!marker.isFile || marker.length() > MAX_JOURNAL_BYTES) null else {
                val json = JsonParser.parseString(marker.readText(Charsets.UTF_8)).asJsonObject
                checkSupply(json.keySet() == setOf("schema_version", "manifest", "prefixes", "journal_sha256"),
                    "SUPPLY_RESUME_INVALID", "resume fields")
                val unsigned = json.deepCopy().apply { remove("journal_sha256") }
                checkSupply(ModelSupplyManifest.integer(json["schema_version"]) == 1L &&
                    supplySha256(canonicalSupplyJson(unsigned).toByteArray(Charsets.UTF_8)) == ModelSupplyManifest.text(json, "journal_sha256"),
                    "SUPPLY_RESUME_INVALID", "resume digest")
                val manifest = ModelSupplyManifest.parse(json.getAsJsonObject("manifest"))
                checkSupply(stage.name == ".receive-${manifest.manifestSha256}", "SUPPLY_RESUME_INVALID", "resume stage identity")
                val records = linkedMapOf<String, Checkpoint>()
                val entries = manifest.files.associateBy { it.path }
                val prefixes = json.getAsJsonArray("prefixes")
                checkSupply(prefixes.size() <= entries.size, "SUPPLY_RESUME_INVALID", "resume inventory")
                for (element in prefixes) {
                    val record = element.asJsonObject
                    checkSupply(record.keySet() == setOf("path", "file_sha256", "size_bytes", "prefix_bytes", "prefix_sha256"),
                        "SUPPLY_RESUME_INVALID", "resume file fields")
                    val path = ModelSupplyManifest.text(record, "path")
                    val file = entries[path] ?: throw ModelSupplyException("SUPPLY_RESUME_INVALID", "resume file identity")
                    val length = ModelSupplyManifest.integer(record["prefix_bytes"], 1, file.size)
                    val sha = ModelSupplyManifest.text(record, "prefix_sha256")
                    checkSupply(path !in records && Regex("[0-9a-f]{64}").matches(sha) &&
                        ModelSupplyManifest.text(record, "file_sha256") == file.sha256 &&
                        ModelSupplyManifest.integer(record["size_bytes"]) == file.size,
                        "SUPPLY_RESUME_INVALID", "resume file binding")
                    ModelSupplyPaths.child(stage, path + ".part")
                    records[path] = Checkpoint(path, length, sha)
                }
                ResumeData(manifest, records)
            }
        } catch (_: Exception) { null }
        internal const val RESUME_MARKER = ".supply-resume.json"
        private val LOCAL_CUT_FALLBACK_REASONS = setOf(
            "SUPPLY_LOCAL_SOURCE_UNAVAILABLE", "SUPPLY_LOCAL_CUT_UNSUPPORTED", "SUPPLY_LOCAL_CUT_TIMEOUT",
        )
        private const val DOWNLOAD_CHUNK_BYTES = 256 * 1024
        private const val MAX_JOURNAL_BYTES = 4 * 1024 * 1024
        fun fileUrl(host: String, port: Int, operation: String, node: String, filename: String): HttpUrl {
            checkSupply(ModelSupplyManifest.OPERATION.matches(operation), "SUPPLY_INVALID_OPERATION", "operation id")
            ModelSupplyManifest.safeBasename(filename)
            return httpBaseUrl(host, port).toHttpUrl().newBuilder().addPathSegments("api/models/supply")
                .addPathSegment(operation).addPathSegment("files").addPathSegment(node).addPathSegment(filename).build()
        }
    }
}
