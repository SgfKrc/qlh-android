package com.qlh.inference.status

import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.async
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import org.junit.Assert.*
import org.junit.Test

class AndroidRuntimeStatusInvalidationTest {
    @Test
    fun `a snapshot started before remote retirement cannot restore the old loaded card`() = runBlocking {
        val changes = AndroidRuntimeStatusInvalidation()
        val old = AndroidRuntimeStatus(model = ModelRuntimeStatus(
            loaded = true, name = "qwen35 2B Q4_K - Medium", loadedLayerRange = listOf(8, 32),
        ))
        val finish = CompletableDeferred<Unit>()
        val staleRead = async(start = CoroutineStart.UNDISPATCHED) {
            changes.readSnapshot { finish.await(); old }
        }
        var actual = old
        changes.afterModelChange { actual = AndroidRuntimeStatus() }
        finish.complete(Unit)
        assertNull(withTimeout(5000) { staleRead.await() })
        assertSame(actual, changes.readSnapshot { actual })
        assertFalse(actual.model.loaded)
    }

    @Test
    fun `failed replacement still refreshes the owner after the previous slot was freed`() = runBlocking {
        val changes = AndroidRuntimeStatusInvalidation()
        var actual = AndroidRuntimeStatus(model = ModelRuntimeStatus(loaded = true))
        val result = changes.afterModelChange {
            actual = AndroidRuntimeStatus()
            Result.failure<Unit>(IllegalStateException("prepare_native_load_failed"))
        }
        assertTrue(result.isFailure)
        assertEquals(1L, changes.revisions.value)
        assertSame(actual, changes.readSnapshot { actual })
        assertFalse(actual.model.loaded)
    }

    @Test
    fun `cancelled lifecycle only requests a fresh snapshot after native work has joined`() = runBlocking {
        val changes = AndroidRuntimeStatusInvalidation()
        val entered = CompletableDeferred<Unit>()
        val finish = CompletableDeferred<Unit>()
        val work = launch(start = CoroutineStart.UNDISPATCHED) {
            changes.afterModelChange {
                withContext(NonCancellable) { entered.complete(Unit); finish.await() }
            }
        }
        entered.await()
        work.cancel()
        assertEquals(0L, changes.revisions.value)
        assertFalse(work.isCompleted)
        finish.complete(Unit)
        withTimeout(5000) { work.join() }
        assertEquals(1L, changes.revisions.value)
    }
}
