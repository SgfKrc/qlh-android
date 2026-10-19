package com.qlh.inference

import com.qlh.inference.network.ClientSnapshotCursor
import org.junit.Assert.*
import org.junit.Test

class ClientSnapshotCursorTest {
    @Test fun `stale revisions old boots unknown schema pending and nonfinite observations fail closed`() {
        val cursor = ClientSnapshotCursor()
        fun accept(revision: Long? = 7, boot: String = "boot", observed: Double = 100.0,
                   version: Int = 2, validity: String = "valid") =
            cursor.accept(version, revision, mapOf("boot_id" to boot), observed, validity)
        assertTrue(accept())
        assertFalse(accept(revision = 6))
        assertFalse(accept(revision = -1))
        assertFalse(accept(revision = null))
        assertFalse(accept(boot = "old", observed = 99.0))
        assertFalse(accept(observed = Double.NaN))
        assertFalse(accept(observed = Double.POSITIVE_INFINITY))
        assertFalse(accept(validity = "pending"))
        assertFalse(accept(version = 1))
        assertFalse(accept(boot = ""))
        assertTrue(accept(revision = 1, boot = "new", observed = 101.0))
        assertFalse(accept(boot = "boot", observed = 100.0))
    }
}
