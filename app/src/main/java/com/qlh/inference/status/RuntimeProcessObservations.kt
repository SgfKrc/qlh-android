package com.qlh.inference.status

import android.os.ParcelFileDescriptor
import android.system.ErrnoException
import android.system.Os
import android.system.OsConstants
import java.io.File
import java.io.IOException
import java.io.ByteArrayOutputStream
import java.net.InetSocketAddress
import java.security.MessageDigest

/** OS facts for this process only. It duplicates descriptors to inspect, never to send or receive. */
internal object RuntimeProcessObservations {
    private data class Descriptors(val sockets: Map<Int, String>, val complete: Boolean)

    private fun descriptors(): Descriptors {
        val files = File("/proc/self/fd").listFiles() ?: return Descriptors(emptyMap(), false)
        val sockets = linkedMapOf<Int, String>()
        var complete = true
        for (file in files) {
            val fd = file.name.toIntOrNull() ?: continue
            try {
                val target = Os.readlink(file.path)
                if (target.startsWith("socket:[")) sockets[fd] = target
            } catch (error: ErrnoException) {
                // listFiles closes its own directory fd; a disappeared descriptor is not a live socket.
                if (error.errno != OsConstants.ENOENT) complete = false
            }
        }
        return Descriptors(sockets, complete)
    }

    fun capture(): Map<String, Any?> {
        val before = descriptors()
        val connections = mutableListOf<Map<String, Any?>>()
        val errors = mutableListOf<String>()
        for ((number, inode) in before.sockets) {
            try {
                ParcelFileDescriptor.fromFd(number).use { copy ->
                    // If the original fd was reused while dup ran, this cannot prove the original socket.
                    if (Os.readlink("/proc/self/fd/$number") != inode) {
                        errors += "socket_descriptor_changed"
                    } else {
                        val local = Os.getsockname(copy.fileDescriptor)
                        try {
                            val remote = Os.getpeername(copy.fileDescriptor)
                            if (local is InetSocketAddress && remote is InetSocketAddress) {
                                connections += linkedMapOf(
                                    "fd" to number, "socket_inode" to inode,
                                    "local_address" to local.address.hostAddress, "local_port" to local.port,
                                    "remote_address" to remote.address.hostAddress, "remote_port" to remote.port,
                                )
                            }
                        } catch (error: ErrnoException) {
                            // A listening/unconnected socket has no peer, and is fully observed as such.
                            if (error.errno != OsConstants.ENOTCONN) errors += "socket_peer_unavailable"
                        }
                    }
                }
            } catch (_: IOException) {
                errors += "socket_descriptor_unavailable"
            } catch (_: ErrnoException) {
                errors += "socket_inspection_failed"
            } catch (_: SecurityException) {
                errors += "socket_inspection_denied"
            }
        }
        val after = descriptors()
        val socketsComplete = before.complete && after.complete && before.sockets == after.sockets && errors.isEmpty()
        val maps = try {
            // A bounded read; omit arbitrary paths from non-model/non-native mappings.
            File("/proc/self/maps").inputStream().use { input ->
                val bytes = ByteArrayOutputStream()
                val buffer = ByteArray(8192)
                var oversized = false
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) break
                    if (bytes.size() + count > 4 * 1024 * 1024) {
                        oversized = true
                        break
                    }
                    bytes.write(buffer, 0, count)
                }
                if (oversized) null else bytes.toString("UTF-8")
            }
        } catch (_: IOException) { null } catch (_: SecurityException) { null }
        val children = try {
            File("/proc/self/task/${android.os.Process.myPid()}/children").readText().trim()
                .split(Regex("\\s+")).filter { it.isNotBlank() }.map { it.toInt() }
        } catch (_: IOException) { null } catch (_: SecurityException) { null } catch (_: NumberFormatException) { null }
        return linkedMapOf(
            "pid" to android.os.Process.myPid(),
            "socket_source" to "os_process_descriptors",
            "sockets_complete" to socketsComplete,
            "socket_errors" to errors.distinct(),
            "connections" to connections,
            "maps_complete" to (maps != null),
            "maps_sha256" to maps?.let { raw -> MessageDigest.getInstance("SHA-256").digest(raw.toByteArray(Charsets.UTF_8)).joinToString("") { "%02x".format(it) } },
            "model_mappings" to maps?.lineSequence()?.filter { line ->
                val value = line.lowercase()
                value.contains(".gguf") || value.contains(".safetensors")
            }?.toList().orEmpty(),
            "native_mappings" to maps?.lineSequence()?.filter { line ->
                line.contains("libqlh_llama_jni.so") || line.contains("libllama.so") || line.contains("libggml")
            }?.toList().orEmpty(),
            "children_complete" to (children != null),
            "child_pids" to children,
        )
    }
}
