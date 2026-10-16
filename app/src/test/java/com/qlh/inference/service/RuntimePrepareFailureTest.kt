package com.qlh.inference.service

import com.qlh.inference.worker.WorkerPrepareException
import com.qlh.inference.worker.WorkerPrepareRequest
import com.qlh.inference.worker.WorkerPrepareTransportTest
import org.junit.Assert.*
import org.junit.Test

class RuntimePrepareFailureTest {
    @Test fun `diagnostics retain failure location without messages causes or raw paths`() {
        val error = IllegalArgumentException("credential=private-token content://private/document").apply {
            initCause(IllegalStateException("manifest and authorization secret"))
            stackTrace = arrayOf(StackTraceElement("com.qlh.inference.status.ReleaseRuntimeProbe", "parse", "ReleaseRuntimeProbe.kt", 264))
        }
        val diagnostic = RuntimePrepareFailure.diagnostic(RuntimePrepareFailure.Stage.METADATA, error)
        assertTrue(diagnostic.contains("stage=METADATA"))
        assertTrue(diagnostic.contains("IllegalArgumentException"))
        assertTrue(diagnostic.contains("ReleaseRuntimeProbe.parse:264"))
        for (secret in listOf("credential", "private-token", "content://", "authorization", "secret")) {
            assertFalse(diagnostic.contains(secret))
        }
    }

    @Test fun `receipt metadata failures cross prepare ACK as specific safe code`() {
        val cause = IllegalArgumentException("private native manifest")
        val error = RuntimePrepareFailure.exception(RuntimePrepareFailure.Stage.METADATA, cause)
        val outer = RuntimePrepareFailure.exception(RuntimePrepareFailure.Stage.BIND, error)
        assertSame(error, outer)
        assertSame(cause, outer.cause)
        assertEquals("prepare_receipt_metadata_invalid", outer.code)
        val ack = WorkerPrepareRequest.parse(WorkerPrepareTransportTest.wire()).ack("error", false, outer.code)
        assertEquals("prepare_receipt_metadata_invalid", ack["error"].asString)
        assertFalse(ack["loaded"].asBoolean)
        assertFalse(ack.toString().contains("private native manifest"))
    }

    @Test fun `existing artifact verification codes survive diagnostic wrappers`() {
        val error = WorkerPrepareException("prepare_manifest_mismatch", "private detail")
        assertSame(error, RuntimePrepareFailure.exception(RuntimePrepareFailure.Stage.VERIFY, error))
        assertSame(error, RuntimePrepareFailure.exception(RuntimePrepareFailure.Stage.LOAD, error))
    }

    @Test fun `malformed metadata integers are classified as receipt metadata instead of request errors`() {
        val error = WorkerPrepareException("prepare_invalid_request", "integer bounds")
        val wrapped = RuntimePrepareFailure.exception(RuntimePrepareFailure.Stage.METADATA, error)
        assertEquals("prepare_receipt_metadata_invalid", wrapped.code)
        assertSame(error, wrapped.cause)
    }
}
