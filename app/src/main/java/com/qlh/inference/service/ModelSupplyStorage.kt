package com.qlh.inference.service

import java.io.File
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.attribute.BasicFileAttributes

/** Shared by registration, pending artifact resolution, native handles and the GC commit. */
internal object ModelSupplyStorage {
    val lock = Any()
    private val pending = mutableMapOf<String, Int>()
    private val references = mutableMapOf<String, Int>()

    class Lease internal constructor(private val release: () -> Unit) : AutoCloseable {
        private var closed = false
        override fun close() = synchronized(lock) {
            if (!closed) { release(); closed = true }
        }
    }

    fun pending(root: File): Lease = synchronized(lock) { retain(pending, root.canonicalPath) }

    /** The handle owns this pin through pending JNI load, inference and successful JNI unload. */
    fun artifact(file: File): Lease? = synchronized(lock) {
        val directory = file.absoluteFile.parentFile ?: return@synchronized null
        if (!Regex("segment-[0-9a-f]{64}").matches(directory.name)) {
            return@synchronized null
        }
        ModelSupplyPaths.check(directory)
        ModelSupplyPaths.check(file)
        retain(references, directory.canonicalPath)
    }

    /** Receive owners and byte donors acquire this before opening data; GC uses the same deletion fence. */
    fun transfer(root: File, directory: File): Lease = synchronized(lock) {
        ModelSupplyPaths.check(root)
        ModelSupplyPaths.check(directory)
        checkSupply(directory.canonicalFile.parentFile == root.canonicalFile && directory.isDirectory &&
            Regex("(?:segment-|\\.receive-)[0-9a-f]{64}").matches(directory.name),
            "SUPPLY_UNSAFE_PATH", "transfer directory unavailable")
        retain(references, directory.canonicalPath)
    }

    // GC callers must hold lock until deletion has completed.
    fun retained(root: File, directory: File): Boolean =
        pending.getOrDefault(root.canonicalPath, 0) > 0 || references.getOrDefault(directory.canonicalPath, 0) > 0

    private fun retain(counts: MutableMap<String, Int>, key: String): Lease {
        counts[key] = Math.addExact(counts.getOrDefault(key, 0), 1)
        return Lease {
            val count = counts.getValue(key)
            if (count == 1) counts.remove(key) else counts[key] = count - 1
        }
    }
}

/** Reject links, junctions/reparse objects and redirected ancestors before touching private data. */
internal object ModelSupplyPaths {
    fun check(file: File) {
        checkSupply(file.absoluteFile.toPath().normalize().toFile() == file.canonicalFile,
            "SUPPLY_UNSAFE_PATH", "private path redirected")
        // canonicalFile alone does not resolve Windows junction ancestors.
        var ancestor: File? = file.absoluteFile
        while (ancestor != null) {
            val path = ancestor.toPath()
            if (Files.exists(path, LinkOption.NOFOLLOW_LINKS)) {
                val attributes = Files.readAttributes(path, BasicFileAttributes::class.java, LinkOption.NOFOLLOW_LINKS)
                checkSupply(!attributes.isSymbolicLink && !attributes.isOther,
                    "SUPPLY_UNSAFE_PATH", "private path is a reparse object")
            }
            ancestor = ancestor.parentFile
        }
    }

    fun child(directory: File, name: String): File {
        check(directory)
        val file = File(directory, name)
        check(file)
        checkSupply(file.canonicalFile.parentFile == directory.canonicalFile,
            "SUPPLY_UNSAFE_PATH", "private path escaped")
        return file
    }
}
