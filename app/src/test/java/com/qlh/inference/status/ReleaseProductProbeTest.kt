package com.qlh.inference.status

import com.google.gson.JsonParser
import com.qlh.inference.worker.WorkerPrepareIdentity
import com.qlh.inference.worker.WorkerPrepareTransportTest
import com.qlh.inference.worker.TaskWorkerSnapshot
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.IOException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import java.util.concurrent.ExecutionException

class ReleaseProductProbeTest {
    private fun identity() = WorkerPrepareIdentity.parse(WorkerPrepareTransportTest.wire()["identity"].asJsonObject)
    private fun manifest() = JsonParser.parseString("""{
        "source_layer_range":[2,4],"n_layer":4,"kept_block_count":2,"nextn_predict_layers":0,
        "artifact":"C:\\synthetic\\tail.gguf","mode":"tail","hidden_size":1,
        "artifact_sha256":"${"c".repeat(64)}","source_model_id":"source_gguf",
        "source_model_sha256":"${"b".repeat(64)}","tokenizer_sha256":"${"d".repeat(64)}"
    }""").asJsonObject

    @Test fun `product and release requests have distinct explicit flags and strict nonce parsing`() {
        val nonce = "a".repeat(32)
        assertFalse(ReleaseRuntimeProbe.request(arrayOf("--qlh-release-probe", nonce))!!.product)
        assertTrue(ReleaseRuntimeProbe.request(arrayOf("--qlh-product-probe", nonce))!!.product)
        assertNull(ReleaseRuntimeProbe.request(arrayOf("--qlh-product-probe", nonce, "more")))
        assertNull(ReleaseRuntimeProbe.request(arrayOf("--qlh-product-probe", "A".repeat(32))))
    }

    @Test fun `receipt source range and total come from verified native artifact metadata`() {
        val raw = manifest().toString()
        val parsed = ReleaseRuntimeProbe.PreparedMetadata.parse(identity(), raw, "tail.gguf")
        assertEquals(listOf(2, 4), parsed.range)
        assertEquals(4, parsed.total)
        assertEquals(2, parsed.decoderLayers)
        assertEquals(ReleaseRuntimeProbe.sha256(raw.toByteArray()), parsed.manifestSha256)
        val changes = listOf<Pair<String, Any>>("n_layer" to 5, "kept_block_count" to 4,
            "artifact" to "another.gguf", "artifact_sha256" to "e".repeat(64), "hidden_size" to 2)
        for ((key, value) in changes) {
            val changed = manifest().apply {
                if (value is Int) addProperty(key, value) else addProperty(key, value.toString())
            }
            try { ReleaseRuntimeProbe.PreparedMetadata.parse(identity(), changed.toString(), "tail.gguf"); fail(key) }
            catch (_: Exception) { }
        }
    }

    @Test fun `source MTP is not subtracted twice from decoder-only head middle and tail receipts`() {
        for ((range, mode) in listOf(listOf(0, 4) to "head", listOf(4, 8) to "middle", listOf(8, 32) to "tail")) {
            val owner = WorkerPrepareIdentity.parse(identity().wire().apply {
                add("layer_range", com.google.gson.Gson().toJsonTree(range))
                addProperty("total_layers", 32)
                addProperty("segment_mode", mode)
            })
            val native = manifest().apply {
                add("source_layer_range", com.google.gson.Gson().toJsonTree(range))
                addProperty("n_layer", 32)
                addProperty("block_count", 33)
                addProperty("kept_block_count", range[1] - range[0])
                addProperty("nextn_predict_layers", 1)
                addProperty("mode", mode)
                add("gguf_metadata", JsonParser.parseString("""{"block_count":${range[1] - range[0]},"nextn_predict_layers":0}"""))
            }
            val receipt = ReleaseRuntimeProbe.PreparedMetadata.parse(owner, native.toString(), "tail.gguf")
            assertEquals(range[1] - range[0], receipt.decoderLayers)
            assertEquals(range, receipt.range)
            assertEquals(32, receipt.total)
        }
    }

    @Test fun `source MTP cannot conceal an extra retained decoder or MTP block`() {
        val native = manifest().apply {
            addProperty("nextn_predict_layers", 1)
            addProperty("kept_block_count", 3)
        }
        try { ReleaseRuntimeProbe.PreparedMetadata.parse(identity(), native.toString(), "tail.gguf"); fail("extra block") }
        catch (_: IllegalArgumentException) { }
    }

    @Test fun `receipt rejects cropped metadata retaining MTP or a mismatching block count`() {
        for (declared in listOf("""{"block_count":2,"nextn_predict_layers":1}""",
            """{"block_count":3,"nextn_predict_layers":0}""")) {
            val native = manifest().apply { add("gguf_metadata", JsonParser.parseString(declared)) }
            try { ReleaseRuntimeProbe.PreparedMetadata.parse(identity(), native.toString(), "tail.gguf"); fail("wrong cut metadata") }
            catch (_: IllegalArgumentException) { }
        }
    }

    @Test fun `full byte hash is streamed with exact size and cooperative cancellation`() {
        val bytes = ByteArray(2 * 1024 * 1024 + 7) { (it % 251).toByte() }
        var checks = 0
        assertEquals(ReleaseRuntimeProbe.sha256(bytes), ReleaseRuntimeProbe.rehash(ByteArrayInputStream(bytes), bytes.size.toLong()) { checks++ })
        assertTrue(checks >= 4)
        for (size in listOf(bytes.size - 1L, bytes.size + 1L)) {
            try { ReleaseRuntimeProbe.rehash(ByteArrayInputStream(bytes), size) { }; fail("wrong size") }
            catch (_: IllegalArgumentException) { }
        }
        try {
            ReleaseRuntimeProbe.rehash(ByteArrayInputStream(bytes), bytes.size.toLong()) { throw CancellationException() }
            fail("cancelled hash")
        } catch (_: CancellationException) { }
    }

    @Test fun `committed generation change during locked capture is never published as a valid receipt`() = runBlocking {
        val original = identity()
        val next = WorkerPrepareIdentity.parse(original.wire().apply { addProperty("generation", original.generation + 1) })
        var reads = 0
        val snapshot = ReleaseRuntimeProbe.capture({ "distributed" }, { TaskWorkerSnapshot() },
            { if (reads++ == 0) original else next }) { ReleaseRuntimeProbe.RuntimeState(ReleaseRuntimeProbe.NativeState()) }
        assertEquals(ReleaseRuntimeProbe.ErrorCode.RUNTIME_IDENTITY_CHANGED, snapshot.errorCode)
    }

    @Test fun `product poll returns immediately caches one nonce and never queues another hash`() {
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        var calls = 0
        ReleaseRuntimeProbe.ProductReader<String>(3000).use { reader ->
            try {
                val started = System.nanoTime()
                assertNull(reader.poll("first") { calls++; entered.countDown(); release.await(); "immutable receipt" })
                assertTrue(TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - started) < 500)
                assertTrue(entered.await(1, TimeUnit.SECONDS))
                assertNull(reader.poll("first") { fail("must reuse"); "bad" })
                try { reader.poll("another") { fail("must not queue"); "bad" }; fail("busy") }
                catch (_: RejectedExecutionException) { }
                release.countDown()
                val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2)
                var result: String? = null
                while (result == null && System.nanoTime() < deadline) { result = reader.poll("first") { "bad" }; Thread.sleep(5) }
                assertEquals("immutable receipt", result)
                assertEquals(result, reader.poll("first") { "must not rehash" })
                assertEquals(1, calls)
            } finally { release.countDown() }
        }
    }

    @Test fun `timed out product nonce stays terminal and cannot restart hashing`() {
        val entered = CountDownLatch(1)
        val release = CountDownLatch(1)
        ReleaseRuntimeProbe.ProductReader<String>(30).use { reader ->
            try {
                assertNull(reader.poll("timed-out") { entered.countDown(); release.await(); "late" })
                assertTrue(entered.await(1, TimeUnit.SECONDS))
                Thread.sleep(50)
                repeat(2) {
                    try { reader.poll("timed-out") { fail("must never restart hash"); "bad" }; fail("timeout") }
                    catch (_: TimeoutException) { }
                    catch (error: ExecutionException) {
                        assertTrue(error.cause is kotlinx.coroutines.TimeoutCancellationException ||
                            error.cause is InterruptedException)
                    }
                }
            } finally { release.countDown() }
        }
    }

    private class Proc : ReleaseProcessTree.Reader {
        val files = linkedMapOf<String, String>()
        val directories = linkedMapOf<String, List<String>>()
        val denied = mutableSetOf<String>()
        val owners = linkedMapOf<Int, Int>()
        var missingChildren = false
        var secondScan: (() -> Unit)? = null
        private var scanCount = 0
        var changedChildren = false
        private var childrenReads = 0
        init { add(10, 1, listOf(10, 11), mapOf(11 to "20")); add(20, 10, listOf(20, 21), mapOf(21 to "30")); add(30, 20, listOf(30)) }
        fun add(pid: Int, parent: Int, tids: List<Int>, children: Map<Int, String> = emptyMap()) {
            owners[pid] = 10000
            val fields = MutableList(20) { "0" }; fields[0] = "S"; fields[1] = parent.toString(); fields[19] = (pid * 100).toString()
            files["/proc/$pid/stat"] = "$pid (name with) parentheses) " + fields.joinToString(" ")
            files["/proc/$pid/cmdline"] = "process-$pid\u0000--fixture\u0000".replace("\\u0000", "\u0000")
            files["/proc/$pid/maps"] = "1000-2000 r--p 00000000 00:01 10 /segments/$pid.gguf\n"
            directories["/proc/$pid/task"] = tids.map { it.toString() }
            tids.forEach { files["/proc/$pid/task/$it/children"] = children[it].orEmpty() }
        }
        override fun text(path: String, limit: Int): String {
            if (path in denied) throw IOException("synthetic denial")
            if (missingChildren && path.endsWith("/children")) throw ReleaseProcessTree.MissingProcFile(path)
            if (path == "/proc/10/task/11/children" && changedChildren && ++childrenReads > 1) return "20 99"
            return files[path] ?: throw ReleaseProcessTree.MissingProcFile(path)
        }
        override fun list(path: String): List<String> {
            if (path in denied) throw IOException("directory denied")
            if (path == "/proc") {
                if (++scanCount == 2) secondScan?.invoke()
                return owners.keys.map { it.toString() } + listOf("self", "net")
            }
            return directories[path] ?: throw IOException("missing task directory")
        }
        override fun uid(pid: Int): Int {
            if ("$pid:uid" in denied) throw IOException("UID denied")
            return owners[pid] ?: throw IOException("PID vanished")
        }
        override fun link(path: String) = "/fixture"
        override fun connections(pid: Int): ReleaseProcessTree.Connections {
            if ("$pid:connections" in denied) throw IOException("connections denied")
            return ReleaseProcessTree.Connections(emptyList(), true)
        }
    }
    @Suppress("UNCHECKED_CAST")
    private fun processes(tree: Map<String, Any?>) = tree["processes"] as List<Map<String, Any?>>

    @Test fun `all tasks children and grandchildren are observed with stable process identities`() {
        val tree = ReleaseProcessTree.capture(10, Proc())
        assertEquals(true, tree["complete"])
        assertEquals(listOf(10, 20, 30), processes(tree).map { it["pid"] })
        assertEquals(listOf(20), processes(tree)[0]["child_pids"])
        assertEquals(listOf(30), processes(tree)[1]["child_pids"])
        assertEquals(mapOf<String, Any?>("pid" to 30, "created_at" to 3000L), processes(tree)[2]["identity_after"])
    }

    @Test fun `denied task maps connection and missing descendant never become complete`() {
        for (denied in listOf("/proc/10/task/11/children", "/proc/20/maps", "/proc/30/stat", "20:connections")) {
            val tree = ReleaseProcessTree.capture(10, Proc().apply { this.denied += denied })
            assertEquals(denied, false, tree["complete"])
            assertTrue(processes(tree).any { it["children_complete"] == false || it["maps_complete"] == false ||
                it["connections_complete"] == false || it["created_at"] == null })
        }
    }

    @Test fun `changed descendants and empty malformed maps cannot claim complete enumeration`() {
        assertEquals(false, ReleaseProcessTree.capture(10, Proc().apply { changedChildren = true })["complete"])
        for (raw in listOf("", "not a maps row")) {
            val tree = ReleaseProcessTree.capture(10, Proc().apply { files["/proc/20/maps"] = raw })
            assertEquals(false, tree["complete"])
            assertEquals(false, processes(tree)[1]["maps_complete"])
        }
    }

    @Test fun `absent kernel children file falls back to complete UID PPID descendant closure`() {
        val tree = ReleaseProcessTree.capture(10, Proc().apply { missingChildren = true; add(40, 1, listOf(40)) })
        assertEquals(true, tree["complete"])
        assertEquals("android_proc_uid_ppid", tree["source"])
        assertEquals(listOf(10, 20, 30), processes(tree).map { it["pid"] })
        assertEquals(listOf(20), processes(tree)[0]["child_pids"])
        assertEquals(listOf(30), processes(tree)[1]["child_pids"])
        assertTrue(processes(tree).all { it["children_complete"] == true })
        val proof = tree["enumeration"] as Map<*, *>
        assertEquals("app_uid_descendants", proof["scope"])
        assertEquals(true, proof["complete"])
        assertEquals(proof["before"], proof["after"])
        assertEquals(4, ((proof["before"] as Map<*, *>)["roster"] as List<*>).size)
    }

    @Test fun `absent children file on descendant also selects the global scan fallback`() {
        val tree = ReleaseProcessTree.capture(10, Proc().apply { files.remove("/proc/20/task/21/children") })
        assertEquals(true, tree["complete"])
        assertEquals("android_proc_uid_ppid", tree["source"])
        assertEquals(3, processes(tree).size)
    }

    @Test fun `UID enumeration retains denied candidates and never treats unknown owner as unrelated`() {
        for (denial in listOf("/proc", "20:uid", "/proc/20/stat", "/proc/30/maps", "/proc/20/task", "20:connections")) {
            val tree = ReleaseProcessTree.capture(10, Proc().apply { missingChildren = true; denied += denial })
            assertEquals(denial, false, tree["complete"])
        }
        val tree = ReleaseProcessTree.capture(10, Proc().apply {
            missingChildren = true; add(40, 1, listOf(40)); owners[40] = 20000; denied += "40:uid"
        })
        assertEquals(false, tree["complete"])
    }

    @Test fun `UID scans reject newly forked reparented vanished and reused processes`() {
        val changes: List<Proc.() -> Unit> = listOf(
            { add(40, 10, listOf(40)) },
            { add(30, 10, listOf(30)) },
            { owners.remove(30) },
            { files["/proc/20/stat"] = files.getValue("/proc/20/stat").removeSuffix("2000") + "2001" },
            { owners[20] = 20000 },
        )
        for (change in changes) {
            val proc = Proc().apply { missingChildren = true }
            proc.secondScan = { proc.change() }
            val tree = ReleaseProcessTree.capture(10, proc)
            assertEquals(false, tree["complete"])
            assertTrue((tree["errors"] as List<*>).contains("uid_roster_changed_or_incomplete"))
        }
    }

    @Test fun `other UID is explicitly excluded and a solitary process needs actual double scan proof`() {
        val proc = Proc().apply { missingChildren = true; add(20, 1, listOf(20)); owners[20] = 20000; owners[30] = 20000 }
        val tree = ReleaseProcessTree.capture(10, proc)
        assertEquals(true, tree["complete"])
        assertEquals(listOf(10), processes(tree).map { it["pid"] })
        assertEquals(emptyList<Int>(), processes(tree).single()["child_pids"])
        val proof = tree["enumeration"] as Map<*, *>
        val before = proof["before"] as Map<*, *>
        assertEquals(3, (before["owners"] as List<*>).size)
    }

    @Test fun `different UID child or unreadable external parent cannot be silently excluded`() {
        val child = ReleaseProcessTree.capture(10, Proc().apply { missingChildren = true; owners[20] = 20000 })
        assertEquals(false, child["complete"])
        assertTrue((child["errors"] as List<*>).contains("cross_uid_descendant"))
        val unknown = ReleaseProcessTree.capture(10, Proc().apply {
            missingChildren = true; add(40, 1, listOf(40)); owners[40] = 20000; denied += "/proc/40/stat"
        })
        assertEquals(false, unknown["complete"])
    }

    @Test fun `TCP INFO requires actual valid state and never promotes an available peer`() {
        val local = listOf<Any>("127.0.0.1", 40000)
        val peer = listOf<Any>("127.0.0.2", 8888)
        assertEquals("ESTABLISHED", ReleaseProcessTree.tcpInfoConnection(1, local, peer, "42")["status"])
        assertEquals("CLOSE_WAIT", ReleaseProcessTree.tcpInfoConnection(8, local, peer, "42")["status"])
        assertEquals("LISTEN", ReleaseProcessTree.tcpInfoConnection(10, local, null, "42")["status"])
        assertEquals("jni_tcp_info", ReleaseProcessTree.tcpInfoConnection(1, local, peer, "42")["state_source"])
        for (state in listOf(-1, 0, 12, 99)) {
            try { ReleaseProcessTree.tcpInfoConnection(state, local, peer, "42"); fail("unknown TCP state") }
            catch (_: IllegalArgumentException) { }
        }
        try { ReleaseProcessTree.tcpInfoConnection(1, local, null, "42"); fail("missing established peer") }
        catch (_: IllegalArgumentException) { }
    }

    @Test fun `TCP tables decode actual IPv4 IPv6 endpoints inode and state`() {
        val rows = ReleaseProcessTree.parseTcpTable("""  sl  local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode
  0: 0100007F:9C40 0200007F:22B8 01 00000000:00000000 00:00000000 00000000 1000 0 42
  1: 00000000000000000000000001000000:22B8 00000000000000000000000000000000:0000 0A 00000000:00000000 00:00000000 00000000 1000 0 43
""")
        assertEquals("ESTABLISHED", rows["42"]!!["status"])
        assertEquals(listOf("127.0.0.1", 40000), rows["42"]!!["local"])
        assertEquals(listOf("127.0.0.2", 8888), rows["42"]!!["remote"])
        assertEquals("LISTEN", rows["43"]!!["status"])
        try { ReleaseProcessTree.parseTcpTable("invalid"); fail("malformed TCP table") } catch (_: IllegalArgumentException) { }
    }
}
